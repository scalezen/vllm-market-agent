"""A benchmark run config: the four sections of a YAML file in benchmarks/configs/."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, field_validator, model_validator


class _Section(BaseModel):
    # strict: no coercion, so "300" is not an int and a YAML boolean (NO, ON) is not a str or an int.
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")


class ServerConfig(_Section):
    model: str
    tool_call_parser: str
    reasoning_parser: str
    serve_script: str
    ready_timeout_s: PositiveInt
    continuous_batching: bool
    max_num_seqs: int | None
    prefix_cache: bool
    kv_cache_quantization: bool
    kv_cache_quantization_bits: Literal[4, 8] | None
    # May be omitted, so configs written before it existed keep their config_sha.
    stream_interval: PositiveInt | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.max_num_seqs is not None and not self.continuous_batching:
            raise ValueError("max_num_seqs needs continuous_batching")
        if self.kv_cache_quantization_bits is not None and not self.kv_cache_quantization:
            raise ValueError("kv_cache_quantization_bits needs kv_cache_quantization")
        return self


class PromptConfig(_Section):
    ticker: str
    signal_file: str
    noise_file: str
    noise_levels: list[NonNegativeInt] = Field(min_length=1)
    system: str
    user: str
    report: str


class RequestConfig(_Section):
    prefill_only: bool
    # In the file: an int, a list of ints (swept, e.g. for TPOT) or null. Stored as a list; null is [None].
    max_tokens: list[int | None]
    enable_thinking: bool

    @field_validator("max_tokens", mode="before")
    @classmethod
    def _as_list(cls, value: object) -> object:
        return value if isinstance(value, list) else [value]

    @model_validator(mode="after")
    def _check(self) -> Self:
        tokens = self.max_tokens
        if not self.prefill_only and tokens != [None]:
            raise ValueError("max_tokens must be null unless prefill_only")
        if self.prefill_only and not (
            tokens and all(n is not None and n >= 1 for n in tokens) and len(set(tokens)) == len(tokens)
        ):
            raise ValueError("prefill_only needs max_tokens >= 1 (an int or a list of unique ints)")
        return self


class RunConfig(_Section):
    concurrencies: list[PositiveInt] = Field(min_length=1)
    reps: int
    # Total requests per point, at most `concurrency` in flight; null = concurrency.
    # May be omitted, like server.stream_interval.
    requests: int | None = None
    results_dir: str

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.requests is not None and self.requests < max(self.concurrencies):
            raise ValueError("requests must be >= every concurrency, or null")
        return self


class BenchmarkConfig(_Section):
    server: ServerConfig
    prompt: PromptConfig
    request: RequestConfig
    run: RunConfig
