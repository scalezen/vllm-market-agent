# vllm-market-agent

[![ci](https://github.com/scalezen/vllm-market-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/scalezen/vllm-market-agent/actions/workflows/ci.yml)
[![nightly](https://github.com/scalezen/vllm-market-agent/actions/workflows/nightly.yml/badge.svg)](https://github.com/scalezen/vllm-market-agent/actions/workflows/nightly.yml)

This repo measures how vLLM behaves under different configurations of the server:  
continuous batching, KV-cache, quantization, and the tradeoffs between
them under a synthetic workload. The stock-sentiment agent generates
the workload.

## Quick start

`bench/serve.sh` takes env-var overrides for the model

See `bench/serve.sh` itself for every flag and environment variable.


