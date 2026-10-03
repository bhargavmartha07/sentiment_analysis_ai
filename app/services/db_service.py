"""MongoDB persistence for sentiment analysis results.

Schema (``sentiment_results``)::

    {
      "_id":       ObjectId,   # MongoDB generated
      "job_id":    "<uuid4>",  # unique, indexed - the API lookup key
      "text":      "<str>",    # original input
      "sentiment": "positive" | "negative",
      "score":     <float>,    # confidence in [0, 1]
      "timestamp": ISODate,    # UTC completion time
      "model":     {           # model card snapshot for reproducibility
         "name": "...", "version": "...", "framework": "..."
      },
      "latency_ms": <int>,     # end-to-end worker latency for this job
      "attempt":    <int>      # delivery attempt that produced the result
    }

``job_id`` carries a unique index, which both documents the lookup contract and
makes the worker's upsert idempotent under at-least-once redelivery.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from bson.codec_options import CodecOptions
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.collection import Collection
from pymongo.errors import CollectionInvalid, PyMongoError

from app.config import Settings, get_settings
from app.errors import PersistenceError
from app.logging_config import get_logger
from app.services.model_service import Prediction

logger = get_logger(__name__)


class SentimentRepository:
    """Thin repository over the results collection.

    ``MongoClient`` maintains its own internal connection pool and is safe to
    share across threads, so one client per process is created lazily and reused.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: MongoClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._injected_client = client is not None
        self._lock = threading.RLock()
        self._next_connect_attempt: float = 0.0

    # ------------------------------------------------------------- connection
    @property
    def is_connected(self) -> bool:
        try:
            if self._client is None:
                return False
            self._client.admin.command("ping")
            return True
        except PyMongoError:
            return False

    @property
    def collection(self) -> Collection:
        """Return the (tz-aware) results collection, connecting on first use."""
        return self._collection()

    def _collection(self) -> Collection:
        client = self._ensure_client()
        # tz_aware keeps BSON datetimes timezone-aware on the way out, so the API
        # serialises a correct UTC timestamp instead of a naive local value.
        return client[self._settings.mongo_db_name].get_collection(
            self._settings.mongo_collection,
            codec_options=CodecOptions(tz_aware=True),
        )

    def _ensure_client(self) -> MongoClient:
        with self._lock:
            if self._client is not None:
                return self._client
            now = time.monotonic()
            if now < self._next_connect_attempt:
                raise PersistenceError(
                    "Database connection is in backoff after a recent failure.",
                    context={"retry_in_seconds": round(self._next_connect_attempt - now, 2)},
                )
            try:
                client: MongoClient = MongoClient(
                    self._settings._secret_value(self._settings.mongo_uri),
                    serverSelectionTimeoutMS=self._settings.mongo_server_selection_timeout_ms,
                    connectTimeoutMS=self._settings.mongo_server_selection_timeout_ms,
                    socketTimeoutMS=10_000,
                    maxPoolSize=50,
                    retryWrites=True,
                    appname=self._settings.service_name,
                )
                client.admin.command("ping")
            except PyMongoError as exc:
                self._next_connect_attempt = now + self._settings.mongo_connect_retry_seconds
                logger.error(
                    "Failed to connect to MongoDB",
                    extra={
                        "event": "db.connect.failed",
                        "error": str(exc),
                        "uri": self._settings.mongo_uri_plain,
                    },
                )
                raise PersistenceError(
                    f"Unable to connect to the results database: {exc}",
                    context={"mongo_uri": self._settings.mongo_uri_plain},
                ) from exc

            self._client = client
            self._next_connect_attempt = 0.0
            logger.info(
                "Connected to MongoDB",
                extra={
                    "event": "db.connect.success",
                    "database": self._settings.mongo_db_name,
                    "collection": self._settings.mongo_collection,
                    "uri": self._settings.mongo_uri_plain,
                },
            )
            return client

    def ensure_indexes(self) -> None:
        """Create the indexes this service depends on. Idempotent."""
        try:
            collection = self._collection()
            collection.create_index([("job_id", ASCENDING)], unique=True, name="job_id_unique")
            collection.create_index([("timestamp", DESCENDING)], name="timestamp_desc")
            collection.create_index([("sentiment", ASCENDING), ("timestamp", DESCENDING)], name="sentiment_timestamp")
        except (PyMongoError, CollectionInvalid) as exc:  # pragma: no cover - startup path
            logger.warning(
                "Could not create MongoDB indexes",
                extra={"event": "db.index.warning", "error": str(exc)},
            )

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None
                logger.info("Closed MongoDB client", extra={"event": "db.close"})

    # ------------------------------------------------------------------ writes
    def save_result(
        self,
        *,
        job_id: str,
        text: str,
        prediction: Prediction,
        model_card: dict[str, Any] | None = None,
        latency_ms: int | None = None,
        attempt: int = 1,
        timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """Upsert the result for ``job_id``. Idempotent under redelivery."""
        document = {
            "job_id": job_id,
            "text": text,
            "sentiment": prediction.sentiment,
            "score": prediction.score,
            "timestamp": timestamp or datetime.now(timezone.utc),
            "model": model_card or {},
            "latency_ms": latency_ms,
            "attempt": attempt,
        }
        try:
            collection = self._collection()
            stored = collection.find_one_and_update(
                {"job_id": job_id},
                {"$set": document},
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
        except PyMongoError as exc:
            logger.error(
                "Failed to persist sentiment result",
                extra={"event": "db.save.failed", "job_id": job_id, "error": str(exc)},
            )
            raise PersistenceError(
                f"Failed to persist the analysis result: {exc}", context={"job_id": job_id}
            ) from exc

        logger.info(
            "Result persisted",
            extra={
                "event": "db.save.success",
                "job_id": job_id,
                "sentiment": prediction.sentiment,
                "score": prediction.score,
                "latency_ms": latency_ms,
            },
        )
        return stored or document

    # ------------------------------------------------------------------- reads
    def get_by_job_id(self, job_id: str) -> dict[str, Any] | None:
        """Return the stored result for ``job_id`` or ``None``."""
        try:
            return self._collection().find_one({"job_id": job_id}, {"_id": 0, "job_id": 1, "text": 1, "sentiment": 1, "score": 1, "timestamp": 1})
        except PyMongoError as exc:
            logger.error(
                "Failed to read sentiment result",
                extra={"event": "db.get.failed", "job_id": job_id, "error": str(exc)},
            )
            raise PersistenceError(
                f"Failed to retrieve the analysis result: {exc}", context={"job_id": job_id}
            ) from exc

    def ping(self) -> bool:
        """Readiness probe for MongoDB."""
        return self.is_connected

    # ------------------------------------------------------------------ extras
    def count(self, query: dict[str, Any] | None = None) -> int:
        return self._collection().count_documents(query or {})

    def latest(self, limit: int = 10) -> list[dict[str, Any]]:
        """Most recent results, newest first. Used by the smoke-test script."""
        return list(
            self._collection()
            .find({}, {"_id": 0, "job_id": 1, "sentiment": 1, "score": 1, "text": 1, "timestamp": 1})
            .sort("timestamp", DESCENDING)
            .limit(limit)
        )

    def delete_job(self, job_id: str) -> bool:
        """Remove a single result. Used by tests and operational cleanup."""
        return self._collection().delete_one({"job_id": job_id}).deleted_count > 0


def document_to_api(document: dict[str, Any] | None) -> dict[str, Any] | None:
    """Map a MongoDB document onto the ``JobResultResponse`` shape."""
    if document is None:
        return None
    timestamp = document.get("timestamp")
    if isinstance(timestamp, datetime) and timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return {
        "job_id": document["job_id"],
        "text": document["text"],
        "sentiment": document["sentiment"],
        "score": document["score"],
        "timestamp": timestamp,
    }


# --------------------------------------------------------------------- singleton
_repository: SentimentRepository | None = None
_repository_lock = threading.Lock()


def get_repository() -> SentimentRepository:
    """Return the process-wide repository singleton."""
    global _repository
    if _repository is None:
        with _repository_lock:
            if _repository is None:
                _repository = SentimentRepository()
    return _repository


def reset_repository() -> None:
    """Drop the repository singleton (tests)."""
    global _repository
    with _repository_lock:
        if _repository is not None:
            _repository.close()
        _repository = None


__all__ = [
    "ObjectId",
    "SentimentRepository",
    "document_to_api",
    "get_repository",
    "reset_repository",
]
