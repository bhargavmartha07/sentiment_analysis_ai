"""Application error hierarchy.

Every failure the client can observe maps to a stable machine readable
``error_code`` and an HTTP status code, so responses are predictable for
consumers and easy to assert on in tests.

All errors share the same JSON envelope::

    {"detail": "human readable message", "error_code": "JOB_NOT_FOUND"}
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Stable error identifiers exposed in API responses."""

    INVALID_INPUT = "INVALID_INPUT"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    QUEUE_UNAVAILABLE = "QUEUE_UNAVAILABLE"
    PERSISTENCE_ERROR = "PERSISTENCE_ERROR"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ServiceError(Exception):
    """Base class for all expected, client visible failures."""

    status_code: int = 500
    error_code: ErrorCode = ErrorCode.INTERNAL_ERROR

    def __init__(
        self,
        detail: str,
        *,
        error_code: ErrorCode | None = None,
        status_code: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        if error_code is not None:
            self.error_code = error_code
        if status_code is not None:
            self.status_code = status_code
        self.context = context or {}

    def to_payload(self) -> dict[str, Any]:
        """Serialise to the client-facing error envelope."""
        return {"detail": self.detail, "error_code": self.error_code.value}


class InvalidInputError(ServiceError):
    """Request payload failed validation (HTTP 400)."""

    status_code = 400
    error_code = ErrorCode.INVALID_INPUT


class JobNotFoundError(ServiceError):
    """No stored result exists for the supplied job id (HTTP 404)."""

    status_code = 404
    error_code = ErrorCode.JOB_NOT_FOUND

    @classmethod
    def for_job(cls, job_id: str) -> "JobNotFoundError":
        return cls(f"Job ID '{job_id}' not found.", context={"job_id": job_id})


class QueueUnavailableError(ServiceError):
    """The broker rejected, or was unable to accept, a publish (HTTP 503)."""

    status_code = 503
    error_code = ErrorCode.QUEUE_UNAVAILABLE


class PersistenceError(ServiceError):
    """MongoDB read/write failure (HTTP 500)."""

    status_code = 500
    error_code = ErrorCode.PERSISTENCE_ERROR


class ModelUnavailableError(ServiceError):
    """The Keras model is not loaded or failed during inference (HTTP 503)."""

    status_code = 503
    error_code = ErrorCode.MODEL_UNAVAILABLE


class ServiceUnavailableError(ServiceError):
    """A dependency the request depends on is not ready (HTTP 503)."""

    status_code = 503
    error_code = ErrorCode.SERVICE_UNAVAILABLE
