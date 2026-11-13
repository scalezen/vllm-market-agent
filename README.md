# vllm-market-agent

[![ci](https://github.com/scalezen/vllm-market-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/scalezen/vllm-market-agent/actions/workflows/ci.yml)
[![nightly](https://github.com/scalezen/vllm-market-agent/actions/workflows/nightly.yml/badge.svg)](https://github.com/scalezen/vllm-market-agent/actions/workflows/nightly.yml)

Measurements of LLM inference serving: where the time goes in prefill and decode, and what
continuous batching gives. A stock-news sentiment agent supplies the prompts.

## Setup

- Server: vllm-mlx
- Models: Qwen3-4B and Gemma 4 E2B

## What is measured

| Experiment | Measured against |
|---|---|
| Prefill | Time to token 2 (prefill plus one decode step) vs prompt length (1.4k–20k tokens) |
| Decode | Time to output token vs number of output tokens |
| Batching | Time to finish a fixed workload, and decode throughput, vs requests in flight (1–64) |

## Reproduce

```bash
conda env create -f environment.yaml && conda activate vllm-market-agent
python scripts/download_headlines_corpus.py WMT AMZN    # per-ticker and noise headlines, into data/raw/
python benchmarks/run_benchmarks.py benchmarks/configs/vllm-mlx/ttft-prefill-baseline.yaml
```

Each config starts its own server with the config's settings and stops it afterwards. Results
are appended to the CSV of the same name under `benchmarks/results/vllm-mlx/`.

## Layout

| Path | Status |
|---|---|
| `benchmarks/` (`benchmark_impl.py`, `run_benchmarks.py`, `managed_server.py`, `request_result.py`) | Benchmark harness: the engineered part |
| `benchmarks/configs/`, `benchmarks/results/` | One YAML config per experiment, and its committed results |
| `python/market_agent/` | Sentiment agent: scaffolding that generates the workload |
