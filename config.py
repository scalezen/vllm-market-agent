"""Settings (from environment variables) and the shared OpenAI clients."""
import asyncio
import os

from openai import AsyncOpenAI, OpenAI

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8010/v1")
MODEL_NAME = os.environ.get("VLLM_MODEL", "mlx-community/Qwen3-4B-4bit")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "mlx-community/embeddinggemma-300m-6bit")
# Headlines with cosine similarity at or above this are treated as the same story.
DEDUP_THRESHOLD = float(os.environ.get("DEDUP_THRESHOLD", "0.85"))
# Freshness windows for tool data.
NEWS_MAX_AGE_DAYS = 7
ANALYST_CHANGES_MAX_AGE_DAYS = 30  # about 1 month, to match the 0m ratings snapshot
MAX_STEPS = 6
DEFAULT_CONCURRENCY = 4
# vllm-mlx's default engine runs one generation at a time and answers 503 to the
# rest, so LLM calls are serialized by default while tool fetches still overlap.
# Raise this if you start the server with continuous batching.
LLM_CONCURRENCY = int(os.environ.get("LLM_CONCURRENCY", "1"))

client = OpenAI(base_url=BASE_URL, api_key="local-dev")  # sync: used by tools (run in threads)
aclient = AsyncOpenAI(base_url=BASE_URL, api_key="local-dev")  # vLLM ignores the key
llm_slots = asyncio.Semaphore(LLM_CONCURRENCY)
