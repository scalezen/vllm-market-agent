#!/usr/bin/env bash
# Start the local vLLM-MLX server with tool calling enabled.
# Port 8010 matches the default VLLM_BASE_URL in sentiment_agent.py (vllm-mlx's own default is 8000).
exec vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit \
  --port 8010 \
  --enable-auto-tool-choice \
  --tool-call-parser llama
