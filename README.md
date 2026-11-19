# vllm-market-agent

[![ci](https://github.com/scalezen/vllm-market-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/scalezen/vllm-market-agent/actions/workflows/ci.yml)
[![nightly](https://github.com/scalezen/vllm-market-agent/actions/workflows/nightly.yml/badge.svg)](https://github.com/scalezen/vllm-market-agent/actions/workflows/nightly.yml)

Measurements of LLM inference serving: performance profile of prefill and decode with respect to
number of input tokens and decoded output tokens,, and impact of
continuous batching. A stock-news sentiment agent supplies the prompts.

## Setup

- Server: vllm-mlx, requires Apple silicon
- Models: `mlx-community/Qwen3-4B-4bit` and `mlx-community/Gemma4-E2B-IT-Text-int4`

## What is measured

| Experiment | Measured against |
|---|---|
| Prefill | Time to token 2 (prefill plus one decode step) vs prompt length (1.4k–20k tokens) |
| Decode | Time to output token k vs k (k = 2 to 128), and its slope, the time per output token, vs prompt length |
| Batching | Time to finish a fixed workload, and decode throughput, vs requests in flight (1–64) |

## Reproduce

```bash
conda env create -f environment.yaml && conda activate vllm-market-agent
python scripts/download_headlines_corpus.py WMT AMZN    # per-ticker and noise headlines, into data/raw/
python -m bench_harness.run_benchmarks benchmarks/configs/vllm-mlx/ttft-prefill-baseline.yaml
```

Each benchmark config run starts its own server with the config's settings and stops it. Results
are appended to the CSV of the same name under `benchmarks/results/vllm-mlx/`.

Tests, apart from those that need a live server:

```bash
python -m pytest tests -m "not slow"
```

## Layout

| Path | Status |
|---|---|
| `python/bench_harness/` (`benchmark_impl.py`, `run_benchmarks.py`, `managed_server.py`, `schemas/`) | Benchmark harness |
| `benchmarks/configs/`, `benchmarks/results/` | One YAML config per experiment, and corresponding results |
| `benchmarks/notebooks/` | Analysis of the committed results: fits, tables and plots |
| `tests/` | Unit tests for the harness (config validation, prompt building), the similarity kernel and the agent's tools. Agent tests that need a live server are marked `slow` |
| `python/market_agent/` | Sentiment agent: scaffolding that generates the workload |
| `cpp/numerical_primitives/`, `python/numerical_primitives/` | Top-k cosine-similarity search in C++ (pybind11), with a numpy fallback. It is used by the agent's `get_stock_news_top_k` tool and is not part of the benchmarks |

Developed with AI coding assistance.
