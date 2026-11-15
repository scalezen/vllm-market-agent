"""load_config: tests to check if bad configs are rejected."""

import copy

import pytest
import yaml

from benchmark_impl import CONFIGS_DIR, load_config

# Smallest valid config; each bad case below changes one thing in it.
VALID = {
    "server": {
        "model": "mlx-community/Qwen3-4B-4bit",
        "tool_call_parser": "qwen",
        "reasoning_parser": "qwen3",
        "serve_script": "benchmarks/serve.sh",
        "ready_timeout_s": 300,
        "continuous_batching": True,
        "max_num_seqs": 8,
        "prefix_cache": False,
        "kv_cache_quantization": False,
        "kv_cache_quantization_bits": None,
    },
    "prompt": {
        "ticker": "WMT",
        "signal_file": "data/raw/WMT_news.parquet",
        "noise_file": "data/raw/no_WMT_news.parquet",
        "noise_levels": [50, 100],
        "system": "system prompt",
        "user": "{ticker}?",
        "report": "report prompt",
    },
    "request": {"prefill_only": True, "max_tokens": 2, "enable_thinking": True},
    "run": {"concurrencies": [1, 4], "reps": 5, "results_dir": "benchmarks/results"},
}

DELETE = object()  # marker: remove the key instead of setting it


def write(tmp_path, cfg):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def changed(edits):
    """VALID with `edits` applied: {"section.key": value} or {"section": value}."""
    cfg = copy.deepcopy(VALID)
    for dotted, value in edits.items():
        *parents, key = dotted.split(".")
        target = cfg
        for p in parents:
            target = target[p]
        if value is DELETE:
            del target[key]
        else:
            target[key] = value
    return cfg


def test_valid_config_loads_and_is_normalised(tmp_path):
    cfg = load_config(write(tmp_path, VALID))
    assert cfg.request.max_tokens == [2]  # int -> list, the run loop sweeps it
    assert cfg.server.stream_interval is None  # optional, filled in
    assert cfg.run.requests is None


@pytest.mark.parametrize("path", sorted(CONFIGS_DIR.rglob("*.yaml")), ids=lambda p: str(p.relative_to(CONFIGS_DIR)))
def test_committed_configs_load(path):
    load_config(path)


# edits to VALID and the text the error must contain: "<field path>: <reason>"
BAD = {
    # structure
    "missing section": ({"run": DELETE}, "run: Field required"),
    "unknown section": ({"extra": {}}, "extra: Extra inputs are not permitted"),
    "section not a mapping": ({"server": ["a"]}, "server: Input should be a valid dictionary"),
    "missing field": ({"server.model": DELETE}, "server.model: Field required"),
    "unknown field": ({"run.repz": 5}, "run.repz: Extra inputs are not permitted"),
    # types
    "string for int": ({"server.ready_timeout_s": "300"}, "server.ready_timeout_s: Input should be a valid integer"),
    "bool for int": ({"run.reps": True}, "run.reps: Input should be a valid integer"),
    "bool for str": ({"prompt.ticker": False}, "prompt.ticker: Input should be a valid string (quote it)"),
    # values
    "no noise levels": ({"prompt.noise_levels": []}, "prompt.noise_levels: List should have at least 1 item"),
    "negative noise level": (
        {"prompt.noise_levels": [50, -1]},
        "prompt.noise_levels.1: Input should be greater than or equal to 0",
    ),
    "zero concurrency": ({"run.concurrencies": [0]}, "run.concurrencies.0: Input should be greater than 0"),
    "max_tokens without prefill_only": (
        {"request.prefill_only": False},
        "request: Value error, max_tokens must be null unless prefill_only",
    ),
    "max_tokens zero": ({"request.max_tokens": 0}, "request: Value error, prefill_only needs max_tokens >= 1"),
    "max_tokens repeated": ({"request.max_tokens": [2, 2]}, "request: Value error, prefill_only needs max_tokens >= 1"),
    "zero ready timeout": ({"server.ready_timeout_s": 0}, "server.ready_timeout_s: Input should be greater than 0"),
    "max_num_seqs without batching": (
        {"server.continuous_batching": False},
        "server: Value error, max_num_seqs needs continuous_batching",
    ),
    "kv bits not 4 or 8": (
        {"server.kv_cache_quantization": True, "server.kv_cache_quantization_bits": 6},
        "server.kv_cache_quantization_bits: Input should be 4 or 8",
    ),
    "kv bits without kv quant": (
        {"server.kv_cache_quantization_bits": 4},
        "server: Value error, kv_cache_quantization_bits needs kv_cache_quantization",
    ),
    "zero stream_interval": ({"server.stream_interval": 0}, "server.stream_interval: Input should be greater than 0"),
    "requests below concurrency": ({"run.requests": 2}, "run: Value error, requests must be >= every concurrency"),
}


@pytest.mark.parametrize("edits, message", BAD.values(), ids=BAD.keys())
def test_bad_config_is_rejected(tmp_path, edits, message):
    with pytest.raises(ValueError) as exc:
        load_config(write(tmp_path, changed(edits)))
    assert message in str(exc.value)
