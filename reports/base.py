"""Base class for every report a task can produce."""
from typing import get_args

from pydantic import BaseModel, Field


class BaseReport(BaseModel):
    summary: str = Field(description="One-sentence justification or answer, grounded in the tool results.")

    @classmethod
    def response_schema(cls) -> dict:
        """JSON schema for constrained decoding, with `summary` generated last.

        Pydantic always lists inherited fields first, but the model should commit to
        the structured fields before it writes the free-text summary.
        """
        schema = cls.model_json_schema()
        schema["properties"]["summary"] = schema["properties"].pop("summary")
        return schema


def signal_fields(model: type[BaseModel]) -> list[str]:
    """Field names on `model` typed as a Bullish/Bearish-style directional label.

    Used by backtest.py to find what it can score, without backtest.py having to
    know each report's field names. A field counts if its declared type is a
    Literal including both "Bullish" and "Bearish" (see reports/sentiment.py).
    """
    names = []
    for name, field in model.model_fields.items():
        args = get_args(field.annotation)
        if "Bullish" in args and "Bearish" in args:
            names.append(name)
    return names
