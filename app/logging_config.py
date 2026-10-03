"""Structured logging shared by the API and the worker.

Every log line is emitted as a single JSON object so it can be shipped into a
log aggregator without a regex parser. The shape is stable:

    {"timestamp": "...Z", "level": "INFO", "service": "api",
     "logger": "app.services.model_service", "message": "...",
     "<any extra= keys passed by the caller>"}

Requirements 13 of the brief (consistent, timestamped, structured logs for both
services) is satisfied by :func:`configure_logging`, which both entry points call
before importing any heavy dependency.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

# Third-party loggers that are noisy or redundant at INFO level.
_NOISY_LOGGERS = (
    "pika",
    "tensorflow",
    "urllib3",
    "pymongo",
    "asyncio",
    "multipart",
)

# Attributes present on every LogRecord; anything else was passed via extra=.
_RESERVED_RECORD_KEYS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName", "taskName",
    }
)


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of arbitrary ``extra`` values into JSON types."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return str(value)


class JsonFormatter(logging.Formatter):
    """Renders a :class:`logging.LogRecord` as one line of JSON."""

    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "service": getattr(record, "service", self.service),
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED_RECORD_KEYS or key in payload or key.startswith("_"):
                continue
            payload[key] = _jsonable(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human readable formatter for local development."""

    def __init__(self, service: str) -> None:
        super().__init__(
            fmt=f"%(asctime)s %(levelname)-8s [{service}] %(name)s - %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )


def configure_logging(
    *,
    service: str,
    level: str = "INFO",
    log_format: str = "json",
) -> None:
    """Install the root log handler. Idempotent: safe to call more than once."""
    formatter: logging.Formatter = (
        JsonFormatter(service=service) if log_format == "json" else TextFormatter(service=service)
    )

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(getattr(logging, level, logging.INFO))

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # Uvicorn installs its own handlers; route them through ours so access logs
    # share the same JSON envelope as application logs.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = [handler]
        uvicorn_logger.propagate = False


def get_logger(name: str, *, service: str = "api") -> logging.Logger:
    """Return a logger whose records carry the service name."""
    logger = logging.getLogger(name)
    if not hasattr(logger, "service"):
        logger.service = service  # type: ignore[attr-defined]
    return logger
