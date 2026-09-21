"""Base class for every report a task can produce."""
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
