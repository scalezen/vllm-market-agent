#!/usr/bin/env bash
# Start the local vLLM-MLX server with tool calling enabled.
# Port 8010 matches the default VLLM_BASE_URL in sentiment_agent.py (vllm-mlx's own default is 8000).
#
# Default model: Qwen3-4B (4-bit, ~2.3 GB) - more reliable tool calling than Llama 3.2 3B.
# --reasoning-parser qwen3 moves Qwen3's <think>...</think> text out of the
# answer into a separate `reasoning` field, so the printed analysis stays clean.
#
# Override without editing, e.g. the smaller model:
#   MODEL=mlx-community/Qwen3-1.7B-4bit ./serve.sh
# (then run the agent with VLLM_MODEL set to the same model name)
MODEL="${MODEL:-mlx-community/Qwen3-4B-4bit}"
PARSER="${PARSER:-qwen}"
PORT="${PORT:-8010}"
# Embedding model served on /v1/embeddings; the agent uses it to de-duplicate headlines.
# (Set the same EMBED_MODEL when running the agent if you change it.)
EMBED_MODEL="${EMBED_MODEL:-mlx-community/embeddinggemma-300m-6bit}"

exec vllm-mlx serve "$MODEL" \
  --port "$PORT" \
  --enable-auto-tool-choice \
  --tool-call-parser "$PARSER" \
  --reasoning-parser qwen3 \
  --embedding-model "$EMBED_MODEL"
