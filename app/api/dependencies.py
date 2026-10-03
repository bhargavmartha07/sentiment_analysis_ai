"""FastAPI dependency providers.

Keeping service lookups behind ``Depends`` means tests can substitute a fake
model, publisher or repository without patching module globals.
"""

from __future__ import annotations

from app.services.db_service import SentimentRepository, get_repository
from app.services.model_service import SentimentModelService, get_model_service
from app.services.queue_service import QueuePublisher, get_publisher


def get_model() -> SentimentModelService:
    return get_model_service()


def get_publisher_dependency() -> QueuePublisher:
    return get_publisher()


def get_repository_dependency() -> SentimentRepository:
    return get_repository()


__all__ = [
    "get_model",
    "get_publisher_dependency",
    "get_repository",
    "get_repository_dependency",
    "SentimentModelService",
    "QueuePublisher",
    "SentimentRepository",
]
