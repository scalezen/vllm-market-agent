"""Report for the sentiment task."""
from typing import Literal

from pydantic import Field, field_validator

from .base import BaseReport

Signal = Literal["Bullish", "Bearish", "Neutral", "Unavailable"]


class SentimentReport(BaseReport):
    sentiment: Literal["Bullish", "Bearish", "Neutral"] = Field(description="Overall sentiment for the stock.")
    # Bounds are not enforced by decoding, so out-of-range values are clamped after parsing.
    confidence: float = Field(description="Confidence in the overall sentiment, 0.0 to 1.0.")
    news_signal: Signal = Field(description="Sentiment from the news headlines; Unavailable if no usable data.")
    technical_signal: Signal = Field(description="Sentiment from the technical indicators; Unavailable if no usable data.")
    analyst_signal: Signal = Field(description="Sentiment from analyst ratings; Unavailable if no usable data.")

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return min(max(v, 0.0), 1.0)
