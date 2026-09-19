"""Token accounting."""
import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache

from loguru import logger

from config import MODEL_NAME


@lru_cache(maxsize=1)
def _tokenizer():
    """The served model's tokenizer (from the local HF cache), or None if unavailable."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    except Exception as exc:
        logger.warning("Tokenizer unavailable ({}); thinking tokens will be estimated", exc)
        return None


def count_tokens(text: str) -> int:
    """Token count of `text`; ~4 chars/token estimate if the tokenizer cannot be loaded."""
    if not text:
        return 0
    tok = _tokenizer()
    return len(tok.encode(text, add_special_tokens=False)) if tok else max(1, len(text) // 4)


@dataclass
class Usage:
    """Token and timing totals for one ticker (updated from the event loop and tool threads)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0  # includes thinking_tokens
    thinking_tokens: int = 0
    embedding_tokens: int = 0
    llm_seconds: float = 0.0  # time inside generation requests, excluding queueing for a slot
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_chat(self, response, seconds: float) -> None:
        u = response.usage
        message = response.choices[0].message
        # vllm-mlx returns thinking as `reasoning_content` and folds it into completion_tokens.
        extra = message.model_extra or {}
        thinking = count_tokens(extra.get("reasoning_content") or extra.get("reasoning") or "")
        with self._lock:
            self.prompt_tokens += u.prompt_tokens if u else 0
            self.completion_tokens += u.completion_tokens if u else 0
            self.thinking_tokens += thinking
            self.llm_seconds += seconds

    def add_embedding(self, response) -> None:
        with self._lock:
            self.embedding_tokens += response.usage.total_tokens if response.usage else 0

    def as_dict(self, wall_seconds: float) -> dict:
        total = self.prompt_tokens + self.completion_tokens + self.embedding_tokens
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "thinking_tokens": self.thinking_tokens,  # subset of completion_tokens
            "embedding_tokens": self.embedding_tokens,
            "total_tokens": total,
            "llm_seconds": round(self.llm_seconds, 2),
            "wall_seconds": round(wall_seconds, 2),
            # generated tokens (thinking included) per second of model time
            "tokens_per_second": round(self.completion_tokens / self.llm_seconds, 1) if self.llm_seconds else None,
        }


# Set per ticker in run_agent; asyncio.to_thread copies the context, so tools see it too.
current_usage: ContextVar[Usage | None] = ContextVar("current_usage", default=None)
