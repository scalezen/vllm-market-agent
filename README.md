# vllm-market-agent

[![nightly](https://github.com/scalezen/vllm-market-agent/actions/workflows/nightly.yml/badge.svg)](https://github.com/scalezen/vllm-market-agent/actions/workflows/nightly.yml)

* This repo measures how vLLM behaves under different configurations of the run 
- continuous batching, KV-cache, quantization, and the tradeoffs between
them under a synthetic workload. The stock-sentiment agent generates
the workload.

## Quick start

`serve.sh` takes env-var overrides for the model, and for the two serving
configs this repo actually measures:
CONTINUOUS_BATCHING=true MAX_NUM_SEQS=8 ./serve.sh
KV_CACHE_QUANTIZATION=true KV_CACHE_QUANTIZATION_BITS=4 ./serve.sh

Both can be combined. See `serve.sh` itself for every flag.


