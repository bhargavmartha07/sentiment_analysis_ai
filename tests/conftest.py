"""Shared fixtures and test doubles for the test-suite."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Sequence

import pytest

from app.config import Settings
from app.services.model_service import Prediction


class FakeModelService:
    """Stand-in for :class:`SentimentModelService` (no TensorFlow required)."""

    def __init__(
        self,
        score: float = 0.9,
        sentiment: str = "positive",
        *,
        is_loaded: bool = True,
        fail_with: Exception | None = None,
    ) -> None:
        self.score = score
        self.sentiment = sentiment
        self.is_loaded = is_loaded
        self.fail_with = fail_with
        self.calls: list[Sequence[str]] = []

    def predict_batch(self, texts: Sequence[str]) -> list[Prediction]:
        self.calls.append(list(texts))
        if self.fail_with is not None:
            raise self.fail_with
        return [Prediction(score=self.score, sentiment=self.sentiment) for _ in texts]

    def predict(self, text: str) -> Prediction:
        return self.predict_batch([text])[0]

    def stats(self) -> dict[str, Any]:
        return {"model_loaded": self.is_loaded, "architecture": "fake"}

    @property
    def metadata(self) -> dict[str, Any]:
        return {"architecture": "fake", "metrics": {"test_accuracy": 0.87}}


class FakePublisher:
    """Records published jobs instead of contacting RabbitMQ."""

    def __init__(self, *, connected: bool = True, fail_with: Exception | None = None) -> None:
        self.is_connected = connected
        self.fail_with = fail_with
        self.published: list[Any] = []
        self.keepalive_calls = 0
        self.ensure_connected_calls = 0

    def publish(self, message: Any) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        self.published.append(message)

    def keepalive(self) -> bool:
        self.keepalive_calls += 1
        return self.is_connected

    def ensure_connected(self) -> bool:
        self.ensure_connected_calls += 1
        return self.is_connected

    def stats(self) -> dict[str, Any]:
        return {"connected": self.is_connected, "published_total": len(self.published)}


class FakeRepository:
    """In-memory stand-in for :class:`SentimentRepository`."""

    def __init__(self, *, documents: dict[str, dict[str, Any]] | None = None, connected: bool = True) -> None:
        self.documents: dict[str, dict[str, Any]] = documents or {}
        self._connected = connected
        self.saved: list[dict[str, Any]] = []
        self.indexes_ensured = 0

    def get_by_job_id(self, job_id: str) -> dict[str, Any] | None:
        return self.documents.get(job_id)

    def save_result(self, **kwargs: Any) -> dict[str, Any]:
        self.saved.append(kwargs)
        document = {
            "job_id": kwargs["job_id"],
            "text": kwargs["text"],
            "sentiment": kwargs["prediction"].sentiment,
            "score": kwargs["prediction"].score,
            "timestamp": kwargs.get("timestamp") or datetime.now(timezone.utc),
        }
        self.documents[kwargs["job_id"]] = document
        return document

    def ensure_indexes(self) -> None:
        self.indexes_ensured += 1

    def ping(self) -> bool:
        return self._connected


class FakeDelivery:
    """Records ack/nack/retry calls made by the worker."""

    def __init__(self, job: Any, *, retry_count: int = 0, redelivered: bool = False) -> None:
        self.message = job
        self.delivery_tag = 1
        self.redelivered = redelivered
        self.retry_count = retry_count
        self.raw_body = job.to_json()
        self.acknowledged = False
        self.rejected = False
        self.requeue: bool | None = None
        self.retried_with: int | None = None

    def acknowledge(self) -> None:
        self.acknowledged = True

    def reject(self, *, requeue: bool = False) -> None:
        self.rejected = True
        self.requeue = requeue

    def requeue_for_retry(self, *, retry_count: int, exchange: str, routing_key: str) -> None:
        self.retried_with = retry_count
        self.acknowledged = True


@pytest.fixture()
def settings(tmp_path: Any) -> Settings:
    """Test settings with deterministic, dependency-free values.

    Model paths are redirected into ``tmp_path`` so unit tests never read or
    write the real artefacts in ``app/models``.
    """
    return Settings(
        environment="test",
        log_level="WARNING",
        log_format="text",
        rabbitmq_host="localhost",
        rabbitmq_port=5672,
        mongo_uri="mongodb://localhost:27017",
        mongo_db_name="sentiment_db_test",
        mongo_collection="sentiment_results",
        positive_threshold=0.5,
        model_path=str(tmp_path / "sentiment_model.h5"),
        word_index_path=str(tmp_path / "word_index.json"),
        model_metadata_path=str(tmp_path / "model_metadata.json"),
    )


@pytest.fixture()
def fake_model() -> FakeModelService:
    return FakeModelService()


@pytest.fixture()
def fake_publisher() -> FakePublisher:
    return FakePublisher()


@pytest.fixture()
def fake_repository() -> FakeRepository:
    return FakeRepository()


@pytest.fixture()
def model_artifacts(tmp_path: Any) -> dict[str, str]:
    """Vocabulary + metadata files on disk, without a Keras model file."""
    word_index = tmp_path / "word_index.json"
    word_index.write_text(json.dumps({"great": 3, "movie": 4, "terrible": 5}), encoding="utf-8")
    metadata = tmp_path / "model_metadata.json"
    metadata.write_text(
        json.dumps({"architecture": "test", "max_len": 4, "metrics": {"test_accuracy": 0.5}}),
        encoding="utf-8",
    )
    return {
        "word_index": str(word_index),
        "metadata": str(metadata),
        "model": str(tmp_path / "sentiment_model.h5"),
    }
