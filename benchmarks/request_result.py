"""Timings and token counts for one streamed benchmark request."""
from pydantic import BaseModel, ConfigDict


class RequestResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    ok: bool
    e2e_s: float
    ttft_s: float | None = None  # first content-bearing chunk, reasoning or answer
    ttat_s: float | None = None  # first answer chunk, after </think>
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_chars: int = 0
    chunks: int = 0
    error: str = ""
