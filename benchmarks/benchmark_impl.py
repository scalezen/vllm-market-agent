"""TTFT and end-to-end latency vs. prompt length, one run per YAML config.

    python benchmarks/benchmark_impl.py benchmarks/configs/ttft-prefill-baseline.yaml

Starts benchmarks/serve.sh with the config's server settings, streams each request,
and appends one row per request to benchmarks/results/<config name>.csv. Each row
records the config hash, git commit, model and hardware.
"""
import argparse
import asyncio
import csv
import hashlib
import json
import platform
import subprocess
import time
import uuid
from pathlib import Path

import duckdb
import yaml

from managed_server import ManagedServer
from market_agent.config import aclient
from request_result import RequestResult

# Config paths are relative to the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent

OPTIONAL_INT = (int, type(None))
SCHEMA = {
    "server": {
        "model": str,
        "tool_call_parser": str,
        "reasoning_parser": str,
        "serve_script": str,
        "ready_timeout_s": int,
        "continuous_batching": bool,
        "max_num_seqs": OPTIONAL_INT,
        "prefix_cache": bool,
        "kv_cache_quantization": bool,
        "kv_cache_quantization_bits": OPTIONAL_INT,
    },
    "prompt": {
        "ticker": str,
        "signal_file": str,
        "noise_file": str,
        "noise_levels": list,
        "system": str,
        "user": str,
        "report": str,
    },
    "request": {"prefill_only": bool, "max_tokens": OPTIONAL_INT, "enable_thinking": bool},
    "run": {"concurrencies": list, "reps": int, "results_dir": str},
}


def load_config(path: Path) -> dict:
    """Parse and validate a run config; every field is required, unknown fields are rejected."""
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict) or set(cfg) != set(SCHEMA):
        raise ValueError(f"{path}: needs exactly the sections {sorted(SCHEMA)}")
    for section, fields in SCHEMA.items():
        if set(cfg[section]) != set(fields):
            raise ValueError(f"{path}: [{section}] needs exactly {sorted(fields)}")
        for key, kind in fields.items():
            value = cfg[section][key]
            # bool is an int subclass, and YAML 1.1 reads e.g. ON/NO as booleans.
            if not isinstance(value, kind) or (isinstance(value, bool) and kind is not bool):
                hint = " (quote it)" if kind is str else ""
                raise ValueError(f"{path}: {section}.{key} = {value!r} has the wrong type{hint}")
    levels, concurrencies = cfg["prompt"]["noise_levels"], cfg["run"]["concurrencies"]
    if not levels or not all(type(n) is int and n >= 0 for n in levels):
        raise ValueError(f"{path}: prompt.noise_levels must be a non-empty list of non-negative ints")
    if not concurrencies or not all(type(n) is int and n >= 1 for n in concurrencies):
        raise ValueError(f"{path}: run.concurrencies must be a non-empty list of positive ints")
    server = cfg["server"]
    request = cfg["request"]
    if request["prefill_only"] and not (request["max_tokens"] or 0) >= 1:
        raise ValueError(f"{path}: request.prefill_only needs max_tokens >= 1")
    if not request["prefill_only"] and request["max_tokens"] is not None:
        raise ValueError(f"{path}: request.max_tokens must be null unless prefill_only")
    if server["ready_timeout_s"] < 1:
        raise ValueError(f"{path}: server.ready_timeout_s must be positive")
    if server["max_num_seqs"] is not None and not server["continuous_batching"]:
        raise ValueError(f"{path}: server.max_num_seqs needs continuous_batching")
    if server["kv_cache_quantization_bits"] not in (None, 4, 8):
        raise ValueError(f"{path}: server.kv_cache_quantization_bits must be 4, 8 or null")
    if server["kv_cache_quantization_bits"] is not None and not server["kv_cache_quantization"]:
        raise ValueError(f"{path}: server.kv_cache_quantization_bits needs kv_cache_quantization")
    return cfg


def _run(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()


def provenance(path: Path) -> dict:
    """Columns that tie every result row to exactly what produced it."""
    dirty = _run("git", "status", "--porcelain", "--untracked-files=no")
    chip = _run("sysctl", "-n", "machdep.cpu.brand_string") or platform.processor()
    mem = _run("sysctl", "-n", "hw.memsize")
    return {
        "config": path.stem,
        "config_sha": hashlib.sha256(path.read_bytes()).hexdigest()[:12],
        "git_commit": _run("git", "rev-parse", "--short", "HEAD") + ("-dirty" if dirty else ""),
        "hardware": f"{chip}, {int(mem) // 2**30} GB" if mem.isdigit() else chip,
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _titles(path: Path, limit: int | None = None) -> list[str]:
    """First `limit` headline titles of a parquet file, in file order."""
    sql = "SELECT headline FROM read_parquet(?)"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return [h for (h,) in duckdb.execute(sql, [str(path)]).fetchall()]


def build_messages(prompt: dict, signal: list[str], noise: list[str], bust_prefix_cache: bool) -> list[dict]:
    # A unique id at the very front makes every prompt distinct from its first
    # token, so no request reuses another's cached KV and TTFT is a cold prefill.
    system = f"[request {uuid.uuid4().hex[:8]}] {prompt['system']}" if bust_prefix_cache else prompt["system"]
    tool_call = {
        "id": "call_0",
        "type": "function",
        "function": {"name": "get_stock_news", "arguments": json.dumps({"ticker": prompt["ticker"]})},
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt["user"].format(ticker=prompt["ticker"])},
        {"role": "assistant", "content": None, "tool_calls": [tool_call]},
        {"role": "tool", "tool_call_id": "call_0", "content": "\n".join(noise + signal)},
        {"role": "user", "content": prompt["report"]},
    ]


async def one_request(model: str, messages: list[dict], request: dict) -> RequestResult:
    """Stream one completion to the end, timestamping the first token on the way."""
    # No response_format: the server validates structured output after
    # generation, which would land in end-to-end latency.
    limits = {"max_tokens": request["max_tokens"]} if request["max_tokens"] else {}
    t0 = time.perf_counter()
    ttft = ttat = usage = finish = None
    chunks = reasoning_chars = 0
    try:
        stream = await aclient.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            # vllm-mlx extension; explicit because templates differ (Qwen3 thinks unless
            # told not to, Gemma 4 only when asked) and the server default is an env var.
            extra_body={"enable_thinking": request["enable_thinking"]},
            **limits,
        )
        async for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if not chunk.choices:
                continue
            finish = chunk.choices[0].finish_reason or finish
            delta = chunk.choices[0].delta
            # The qwen3 reasoning parser puts thinking in delta.reasoning(_content), not delta.content.
            think = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None) or ""
            answer = delta.content or ""
            if think or answer:  # the first chunk is often role-only; don't timestamp it
                chunks += 1
                reasoning_chars += len(think)
                print(f"chunk id: {chunks} arrived at time {time.perf_counter() - t0}")
                if ttft is None:
                    ttft = time.perf_counter() - t0
            if answer and ttat is None:
                ttat = time.perf_counter() - t0
    except Exception as exc:  # 503s under load, context overflow, timeouts are results too
        return RequestResult(False, time.perf_counter() - t0, error=str(exc)[:150])
    # A stream can end cleanly with no tokens (seen at ~22k-token prompts); no TTFT means failure.
    return RequestResult(
        ttft is not None,
        time.perf_counter() - t0,
        ttft,
        ttat,
        usage.prompt_tokens if usage else None,
        usage.completion_tokens if usage else None,
        reasoning_chars,
        chunks,
        "" if ttft is not None else f"no tokens (finish_reason={finish})",
    )


async def run_point(cfg: dict, signal: list[str], noise: list[str], concurrency: int) -> list[dict]:
    """Fire `concurrency` requests at once; one row per request."""
    bust = not cfg["server"]["prefix_cache"]
    results = await asyncio.gather(
        *(
            one_request(cfg["server"]["model"], build_messages(cfg["prompt"], signal, noise, bust), cfg["request"])
            for _ in range(concurrency)
        )
    )
    return [
        {
            "prefill_only": cfg["request"]["prefill_only"],
            "max_tokens": cfg["request"]["max_tokens"],
            "enable_thinking": cfg["request"]["enable_thinking"],
            "noise_headlines": len(noise),
            "concurrency": concurrency,
            "request": i,
            "ok": r.ok,
            "prompt_tokens": r.prompt_tokens,
            "ttft_s": r.ttft_s,
            #"ttat_s": r.ttat_s,
            "e2e_s": r.e2e_s,
            "completion_tokens": r.completion_tokens,
            "chunks": r.chunks,
            "reasoning_chars": r.reasoning_chars,
            "error": r.error,
        }
        for i, r in enumerate(results)
    ]


async def run_all(cfg: dict, stamp: dict) -> list[dict]:
    prompt = cfg["prompt"]
    signal = _titles(REPO_ROOT / prompt["signal_file"])
    pool = _titles(REPO_ROOT / prompt["noise_file"], max(prompt["noise_levels"]))
    # Discarded: first-request overheads (weights paged in, kernels compiled).
    warmup = await one_request(
        cfg["server"]["model"],
        build_messages(prompt, signal, pool[: min(prompt["noise_levels"])], True),
        cfg["request"],
    )
    print(f"warm-up: ok={warmup.ok} e2e={warmup.e2e_s:.2f}s {warmup.error}")
    rows = []
    for n_noise in prompt["noise_levels"]:
        for concurrency in cfg["run"]["concurrencies"]:
            for rep in range(cfg["run"]["reps"]):
                for row in await run_point(cfg, signal, pool[:n_noise], concurrency):
                    row = {**stamp, "rep": rep, **row}
                    print(row)
                    rows.append(row)
    return rows


def write_rows(path: Path, rows: list[dict]) -> Path:
    """Append to `path`; if its header differs, write a timestamped sibling instead."""
    columns = list(rows[0].keys())
    has_content = path.exists() and path.stat().st_size > 0
    if has_content:
        with path.open(newline="") as f:
            if next(csv.reader(f), []) != columns:
                path = path.with_name(f"{path.stem}-{time.strftime('%Y%m%d-%H%M%S')}{path.suffix}")
                has_content = False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not has_content:
            writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="run config, e.g. benchmarks/configs/ttft-prefill-baseline.yaml")
    config_path = parser.parse_args().config.resolve()
    cfg = load_config(config_path)
    stamp = {**provenance(config_path), "model": cfg["server"]["model"]}

    script = REPO_ROOT / cfg["server"]["serve_script"]
    with ManagedServer(cfg["server"], script, Path(f"/tmp/bench_serve_{config_path.stem}.log")):
        rows = asyncio.run(run_all(cfg, stamp))
    results = REPO_ROOT / cfg["run"]["results_dir"] / f"{config_path.stem}.csv"
    print(f"wrote {write_rows(results, rows)}")


if __name__ == "__main__":
    main()
