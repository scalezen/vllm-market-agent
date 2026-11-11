"""TTFT and end-to-end latency vs. prompt length, one run per YAML config.

    python benchmarks/benchmark_impl.py benchmarks/configs/vllm-mlx/ttft-prefill-baseline.yaml

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
CONFIGS_DIR = REPO_ROOT / "benchmarks" / "configs"

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
        "stream_interval": OPTIONAL_INT,
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
    # max_tokens: an int, a list of ints (swept, e.g. for TPOT), or null.
    "request": {"prefill_only": bool, "max_tokens": (int, list, type(None)), "enable_thinking": bool},
    # requests: total requests per point, at most `concurrency` in flight; null = concurrency.
    "run": {"concurrencies": list, "reps": int, "requests": OPTIONAL_INT, "results_dir": str},
}
# May be omitted (read as null), so configs written before the field existed keep
# their config_sha and still match their committed results.
OPTIONAL_FIELDS = {("server", "stream_interval"), ("run", "requests")}


def load_config(path: Path) -> dict:
    """Parse and validate a run config; every field except OPTIONAL_FIELDS is required,
    unknown fields are rejected."""
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict) or set(cfg) != set(SCHEMA):
        raise ValueError(f"{path}: needs exactly the sections {sorted(SCHEMA)}")
    for section, fields in SCHEMA.items():
        if not isinstance(cfg[section], dict):
            raise ValueError(f"{path}: [{section}] must be a mapping")
        for sec, key in OPTIONAL_FIELDS:
            if sec == section:
                cfg[section].setdefault(key, None)
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
    if not request["prefill_only"] and request["max_tokens"] is not None:
        raise ValueError(f"{path}: request.max_tokens must be null unless prefill_only")
    # Normalise to a list: the run loop sweeps it; null (full generation) becomes [None].
    tokens = request["max_tokens"]
    tokens = tokens if isinstance(tokens, list) else [tokens]
    if request["prefill_only"] and not (
        tokens and all(type(n) is int and n >= 1 for n in tokens) and len(set(tokens)) == len(tokens)
    ):
        raise ValueError(f"{path}: request.prefill_only needs max_tokens >= 1 (an int or a list of unique ints)")
    request["max_tokens"] = tokens
    if server["ready_timeout_s"] < 1:
        raise ValueError(f"{path}: server.ready_timeout_s must be positive")
    if server["max_num_seqs"] is not None and not server["continuous_batching"]:
        raise ValueError(f"{path}: server.max_num_seqs needs continuous_batching")
    if server["kv_cache_quantization_bits"] not in (None, 4, 8):
        raise ValueError(f"{path}: server.kv_cache_quantization_bits must be 4, 8 or null")
    if server["kv_cache_quantization_bits"] is not None and not server["kv_cache_quantization"]:
        raise ValueError(f"{path}: server.kv_cache_quantization_bits needs kv_cache_quantization")
    if server["stream_interval"] is not None and server["stream_interval"] < 1:
        raise ValueError(f"{path}: server.stream_interval must be >= 1 or null")
    n_requests = cfg["run"]["requests"]
    if n_requests is not None and n_requests < max(concurrencies):
        raise ValueError(f"{path}: run.requests must be >= every concurrency, or null")
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


async def timed_stream_request(model: str, messages: list[dict], request: dict) -> RequestResult:
    """Stream one completion to the end, timestamping the first token on the way.

    TTFT rule:
      full generation: first chunk with visible text (reasoning or answer).
      prefill_only:    the chunk carrying finish_reason, i.e. when token max_tokens
                       has been generated. Visible text can't be used: reasoning
                       parsers hide the first tokens (Qwen3 <think>; Gemma 4
                       <|channel>, thought, ...), and vllm-mlx sends no chunk for a
                       fully hidden token but an empty one for a partly hidden one,
                       so chunk counts differ by parser too.
    """
    # No response_format: the server validates structured output after
    # generation, which would land in end-to-end latency.
    limits = {"max_tokens": request["max_tokens"]} if request["max_tokens"] else {}
    t0 = time.perf_counter()
    ttft = ttat = usage = finish = finish_s = None
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
            if chunk.choices[0].finish_reason and finish_s is None:
                finish, finish_s = chunk.choices[0].finish_reason, time.perf_counter() - t0
            delta = chunk.choices[0].delta
            # The qwen3 reasoning parser puts thinking in delta.reasoning(_content), not delta.content.
            think = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None) or ""
            answer = delta.content or ""
            if think or answer:  # the first chunk is often role-only; don't timestamp it
                chunks += 1
                reasoning_chars += len(think)
                # print(f"chunk id: {chunks} arrived at time {time.perf_counter() - t0}")
                if ttft is None:
                    ttft = time.perf_counter() - t0
            if answer and ttat is None:
                ttat = time.perf_counter() - t0
    except Exception as exc:  # 503s under load, context overflow, timeouts are results too
        return RequestResult(ok=False, e2e_s=time.perf_counter() - t0, error=str(exc)[:150])
    completion = usage.completion_tokens if usage else None
    if request["prefill_only"]:
        # Stamp only if all max_tokens were generated: a stream can end cleanly
        # with no tokens (seen at ~22k-token prompts).
        ttft = finish_s if completion is not None and completion >= request["max_tokens"] else None
    return RequestResult(
        ok=ttft is not None,
        e2e_s=time.perf_counter() - t0,
        ttft_s=ttft,
        ttat_s=ttat,
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=completion,
        reasoning_chars=reasoning_chars,
        chunks=chunks,
        error="" if ttft is not None else f"no TTFT (finish_reason={finish}, completion_tokens={completion})",
    )


async def run_point(cfg: dict, request: dict, signal: list[str], noise: list[str], concurrency: int) -> list[dict]:
    """Send run.requests requests (default: `concurrency`), at most `concurrency` in
    flight; the next starts as soon as one finishes. One row per request.

    `request` is cfg["request"] with max_tokens set to this point's single value.
    start_s is when the request was sent, from the start of the point, so the point's
    wall time is max(start_s + e2e_s). e2e_s and ttft_s run from start_s.
    """
    bust = not cfg["server"]["prefix_cache"]
    in_flight = asyncio.Semaphore(concurrency)
    t_point = time.perf_counter()

    async def one() -> tuple[float, RequestResult]:
        async with in_flight:
            start = time.perf_counter() - t_point
            messages = build_messages(cfg["prompt"], signal, noise, bust)
            return start, await timed_stream_request(cfg["server"]["model"], messages, request)

    results = await asyncio.gather(*(one() for _ in range(cfg["run"]["requests"] or concurrency)))
    return [
        {
            "prefill_only": request["prefill_only"],
            "max_tokens": request["max_tokens"],
            "enable_thinking": request["enable_thinking"],
            "noise_headlines": len(noise),
            "concurrency": concurrency,
            "request": i,
            "start_s": start,
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
        for i, (start, r) in enumerate(results)
    ]


async def run_all(cfg: dict, stamp: dict) -> list[dict]:
    prompt = cfg["prompt"]
    signal = _titles(REPO_ROOT / prompt["signal_file"])
    pool = _titles(REPO_ROOT / prompt["noise_file"], max(prompt["noise_levels"]))
    # Discarded: first-request overheads (weights paged in, kernels compiled).
    warmup = await timed_stream_request(
        cfg["server"]["model"],
        build_messages(prompt, signal, pool[: min(prompt["noise_levels"])], True),
        {**cfg["request"], "max_tokens": cfg["request"]["max_tokens"][0]},
    )
    print(f"warm-up: ok={warmup.ok} e2e={warmup.e2e_s:.2f}s {warmup.error}")
    rows = []
    for n_noise in prompt["noise_levels"]:
        for max_tokens in cfg["request"]["max_tokens"]:
            request = {**cfg["request"], "max_tokens": max_tokens}
            for concurrency in cfg["run"]["concurrencies"]:
                for rep in range(cfg["run"]["reps"]):
                    for row in await run_point(cfg, request, signal, pool[:n_noise], concurrency):
                        row = {**stamp, "rep": rep, **row}
                        print(row)
                        rows.append(row)
    return rows


def results_path(cfg: dict, config_path: Path) -> Path:
    """results_dir/<config's subfolder under benchmarks/configs>/<config stem>.csv.

    Configs are grouped by serving stack (e.g. configs/vllm-mlx/); results mirror
    that, so results_dir in the config stays "benchmarks/results" and moving a
    config between folders doesn't change its bytes or its config_sha."""
    parent = config_path.parent
    sub = parent.relative_to(CONFIGS_DIR) if parent.is_relative_to(CONFIGS_DIR) else Path()
    return REPO_ROOT / cfg["run"]["results_dir"] / sub / f"{config_path.stem}.csv"


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
    parser.add_argument("config", type=Path, help="run config, e.g. benchmarks/configs/vllm-mlx/ttft-prefill-baseline.yaml")
    config_path = parser.parse_args().config.resolve()
    cfg = load_config(config_path)
    stamp = {**provenance(config_path), "model": cfg["server"]["model"]}

    script = REPO_ROOT / cfg["server"]["serve_script"]
    with ManagedServer(cfg["server"], script, Path(f"/tmp/bench_serve_{config_path.stem}.log")):
        rows = asyncio.run(run_all(cfg, stamp))
    results = results_path(cfg, config_path)
    print(f"wrote {write_rows(results, rows)}")


if __name__ == "__main__":
    main()
