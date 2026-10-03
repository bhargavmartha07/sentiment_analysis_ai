"""Pydantic request/response models.

These schemas are the single source of truth for validation *and* for the
generated OpenAPI document (requirement 12). Field descriptions and examples
here surface directly in Swagger UI at ``/docs``.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import get_settings

SentimentLabel = Literal["positive", "negative"]
JobStatus = Literal["processing"]

# Pydantic emits "Z" for UTC timestamps, which matches the documented contract
# of GET /api/sentiment/results/{job_id}.


class SentimentRequest(BaseModel):
    """Payload for synchronous and asynchronous analysis."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"text": "I love this product!"},
            "examples": [
                {"text": "I love this product!"},
                {"text": "This movie was terrible."},
            ],
        }
    )

    text: Annotated[
        str,
        Field(
            description=(
                "Raw text to analyse. Must be a non-empty string of at most "
                "SYNC_MAX_TEXT_LENGTH characters. Blank/whitespace-only input is rejected."
            ),
            examples=["I love this product!"],
        ),
    ]

    @field_validator("text")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        settings = get_settings()
        if not isinstance(value, str):
            raise ValueError("Field 'text' must be a string.")
        text = value.strip()
        if not text:
            raise ValueError("Text field is required and must not be empty.")
        if len(text) > settings.sync_max_text_length:
            raise ValueError(
                f"Text must not exceed {settings.sync_max_text_length} characters "
                f"(received {len(text)})."
            )
        return text


class SyncSentimentResponse(BaseModel):
    """Immediate result of a synchronous analysis."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"text": "I love this product!", "sentiment": "positive", "score": 0.985}}
    )

    text: str = Field(description="Echo of the analysed text (whitespace trimmed).")
    sentiment: SentimentLabel = Field(description="Predicted sentiment label.")
    score: float = Field(ge=0.0, le=1.0, description="Confidence that the text carries the returned sentiment.")


class AsyncJobResponse(BaseModel):
    """Acknowledgement that an asynchronous job was accepted by the broker."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"job_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef", "status": "processing"}
        }
    )

    job_id: uuid.UUID = Field(description="Unique identifier used to poll for the result.")
    status: JobStatus = Field(description="Current job state. Always 'processing' on acceptance.")


class JobResultResponse(BaseModel):
    """Persisted result of an asynchronous analysis."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
                "text": "This movie was terrible.",
                "sentiment": "negative",
                "score": 0.052,
                "timestamp": "2023-10-27T10:30:00Z",
            }
        }
    )

    job_id: str = Field(description="Identifier returned when the job was submitted.")
    text: str = Field(description="Original input text.")
    sentiment: SentimentLabel = Field(description="Predicted sentiment label.")
    score: float = Field(ge=0.0, le=1.0, description="Confidence score in [0, 1].")
    timestamp: datetime = Field(description="UTC completion time of the analysis.")


class BatchSentimentRequest(BaseModel):
    """Bonus endpoint: analyse several texts in a single forward pass."""

    model_config = ConfigDict(json_schema_extra={"example": {"texts": ["Great!", "Awful waste of money."]}})

    texts: Annotated[
        list[str],
        Field(min_length=1, max_length=32, description="Between 1 and 32 texts to analyse."),
    ]

    @field_validator("texts")
    @classmethod
    def _validate_texts(cls, value: list[str]) -> list[str]:
        settings = get_settings()
        if len(value) > settings.sync_batch_max_items:
            raise ValueError(f"At most {settings.sync_batch_max_items} texts may be submitted per batch.")
        cleaned: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, str):
                raise ValueError(f"Item at index {index} must be a string.")
            stripped = item.strip()
            if not stripped:
                raise ValueError(f"Item at index {index} must not be empty.")
            if len(stripped) > settings.sync_max_text_length:
                raise ValueError(
                    f"Item at index {index} exceeds the {settings.sync_max_text_length} character limit."
                )
            cleaned.append(stripped)
        return cleaned


class BatchSentimentResponse(BaseModel):
    """Per-item results for a batch request, in submission order."""

    model_config = ConfigDict(json_schema_extra={"example": {"count": 1, "results": [{"text": "Great!", "sentiment": "positive", "score": 0.981}]}})

    count: int = Field(description="Number of analysed texts.")
    results: list[SyncSentimentResponse] = Field(description="Results in the same order as the request texts.")


class ErrorResponse(BaseModel):
    """Uniform error envelope used by every endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"detail": "Text field is required and must not be empty.", "error_code": "INVALID_INPUT"}
        }
    )

    detail: str = Field(description="Human readable explanation of the failure.")
    error_code: str = Field(
        description="Stable machine readable code: INVALID_INPUT, JOB_NOT_FOUND, QUEUE_UNAVAILABLE, "
        "PERSISTENCE_ERROR, MODEL_UNAVAILABLE, SERVICE_UNAVAILABLE, INTERNAL_ERROR."
    )


class DependencyStatus(BaseModel):
    """Health of a single downstream dependency."""

    status: Literal["up", "down"] = Field(description="Whether the dependency answered its probe.")
    detail: str | None = Field(default=None, description="Diagnostic message when the dependency is down.")


class HealthResponse(BaseModel):
    """Liveness / readiness payload."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"status": "ok", "service": "api", "version": "1.0.0", "environment": "development"}}
    )

    status: Literal["ok", "degraded", "error"] = Field(description="Overall service state.")
    service: str = Field(description="Service name.")
    version: str = Field(description="Deployed application version.")
    environment: str = Field(description="Deployment environment.")
    model_loaded: bool | None = Field(default=None, description="Whether the Keras model is resident in memory.")
    dependencies: dict[str, DependencyStatus] | None = Field(
        default=None, description="Downstream dependency probes (readiness endpoint only)."
    )
