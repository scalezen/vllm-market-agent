"""Report for the generic (free-form question) task."""
from pydantic import Field, field_validator

from .base import BaseReport


class GenericReport(BaseReport):
    summary: str = Field(description="Overall response for the stock.")
    # Bounds are not enforced by decoding, so out-of-range values are clamped after parsing.
    confidence: float = Field(description="Confidence in the overall response, 0.0 to 1.0.")

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return min(max(v, 0.0), 1.0)
