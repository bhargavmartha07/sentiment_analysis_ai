"""End-to-end integration tests.

These run against the **live** stack (API + worker + RabbitMQ + MongoDB) and
are the only place where the asynchronous pipeline is exercised for real:

    docker compose exec api python -m pytest tests/integration/integration_tests.py -v

Configuration comes from the environment, so the same file works locally
(``BASE_URL=http://localhost:8000 ... python -m pytest ...``) and in CI.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pika
import pytest
import requests
from pymongo import MongoClient

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB_NAME", "sentiment_db")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "sentiment_results")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "guest")
QUEUE = os.getenv("RABBITMQ_QUEUE", "sentiment.jobs")
DEAD_LETTER_QUEUE = os.getenv("RABBITMQ_DEAD_LETTER_QUEUE", "sentiment.jobs.dlq")

RESULT_POLL_INTERVAL_SECONDS = 0.5
RESULT_POLL_TIMEOUT_SECONDS = 45.0

pytestmark = pytest.mark.integration


class HttpClient:
    """Session wrapper that resolves relative paths against ``base_url``.

    ``requests.Session`` has no notion of a base URL, so tests must not rely on
    one; this keeps the call sites readable.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self._session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def get(self, path: str, **kwargs: Any) -> Any:
        return self._session.get(self._url(path), **kwargs)

    def post(self, path: str, **kwargs: Any) -> Any:
        return self._session.post(self._url(path), **kwargs)

    def close(self) -> None:
        self._session.close()


@pytest.fixture(scope="module")
def http_client():
    try:
        client = HttpClient(BASE_URL)
        response = client.get("/health", timeout=10)
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"API is not reachable at {BASE_URL}: {exc}")
    yield client
    client.close()


@pytest.fixture(scope="module")
def mongo_collection():
    # tz_aware mirrors the service's own codec options, so assertions see the
    # same UTC-aware datetimes the API reads back.
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000, tz_aware=True)
    try:
        client.admin.command("ping")
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"MongoDB is not reachable at {MONGO_URI}: {exc}")
    yield client[MONGO_DB][MONGO_COLLECTION]
    client.close()


@pytest.fixture()
def amqp_channel():
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    parameters = pika.ConnectionParameters(
        host=RABBITMQ_HOST, port=RABBITMQ_PORT, credentials=credentials, socket_timeout=10
    )
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    try:
        yield channel
    finally:
        connection.close()


def wait_for_result(http_client, job_id: str, timeout: float = RESULT_POLL_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Poll the result endpoint until the worker has persisted the outcome."""
    deadline = time.monotonic() + timeout
    last_status: int | None = None
    while time.monotonic() < deadline:
        response = http_client.get(f"/api/sentiment/results/{job_id}", timeout=10)
        last_status = response.status_code
        if response.status_code == 200:
            return response.json()
        assert response.status_code == 404, f"Unexpected status {response.status_code}: {response.text}"
        time.sleep(RESULT_POLL_INTERVAL_SECONDS)
    pytest.fail(f"Result for {job_id} was not stored within {timeout}s (last status {last_status})")


# ==================================================================== health
class TestHealth:
    def test_liveness(self, http_client):
        response = http_client.get("/health", timeout=10)

        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_readiness_reports_dependencies_up(self, http_client):
        response = http_client.get("/health/ready", timeout=10)

        assert response.status_code == 200
        dependencies = response.json()["dependencies"]
        assert dependencies["model"]["status"] == "up"
        assert dependencies["mongodb"]["status"] == "up"
        assert dependencies["rabbitmq"]["status"] == "up"

    def test_model_is_loaded_and_documented(self, http_client):
        root = http_client.get("/", timeout=10).json()

        assert root["model"]["model_loaded"] is True
        assert root["model"]["architecture"]

    def test_openapi_schema_is_served(self, http_client):
        schema = http_client.get("/openapi.json", timeout=10).json()

        assert schema["info"]["title"] == "Sentiment Analysis API"
        for path in ("/api/sentiment/sync", "/api/sentiment/async", "/api/sentiment/results/{job_id}"):
            assert path in schema["paths"]


# ====================================================================== sync
class TestSyncAnalysis:
    @pytest.mark.parametrize(
        "text",
        [
            "I love this product! It works perfectly and arrived early.",
            "A wonderful, heartwarming film with an excellent cast.",
            "This movie was terrible. Waste of time and money.",
            "Awful product. It broke on the first day and support ignored me.",
        ],
    )
    def test_classifies_real_world_text(self, http_client, text):
        response = http_client.post("/api/sentiment/sync", json={"text": text}, timeout=30)

        assert response.status_code == 200
        body = response.json()
        assert body["text"] == text
        assert body["sentiment"] in ("positive", "negative")
        assert 0.0 <= body["score"] <= 1.0

    def test_positive_text_scores_above_threshold(self, http_client):
        body = http_client.post(
            "/api/sentiment/sync",
            json={"text": "Absolutely fantastic! I am delighted with this purchase."},
            timeout=30,
        ).json()

        assert body["sentiment"] == "positive"
        assert body["score"] > 0.5

    def test_negative_text_scores_below_threshold(self, http_client):
        body = http_client.post(
            "/api/sentiment/sync",
            json={"text": "Disgusting and useless. The worst experience of my life."},
            timeout=30,
        ).json()

        assert body["sentiment"] == "negative"
        assert body["score"] < 0.5

    def test_returns_timing_header(self, http_client):
        response = http_client.post("/api/sentiment/sync", json={"text": "hello"}, timeout=30)

        assert float(response.headers["X-Process-Time-Ms"]) >= 0

    @pytest.mark.parametrize(
        "payload",
        [{}, {"text": ""}, {"text": "    "}, {"text": None}, {"text": 42}, {"text": ["a"]}],
    )
    def test_invalid_payloads_return_400(self, http_client, payload):
        response = http_client.post("/api/sentiment/sync", json=payload, timeout=30)

        assert response.status_code == 400
        assert response.json()["error_code"] == "INVALID_INPUT"
        assert response.json()["detail"]

    def test_repeated_calls_are_consistent(self, http_client):
        text = "This is a wonderfully sad and bittersweet story."
        scores = {
            http_client.post("/api/sentiment/sync", json={"text": text}, timeout=30).json()["score"]
            for _ in range(3)
        }

        assert len(scores) == 1

    def test_batch_endpoint_returns_one_result_per_text(self, http_client):
        texts = ["Great!", "Terrible.", "It was okay I guess."]

        response = http_client.post("/api/sentiment/batch", json={"texts": texts}, timeout=30)

        assert response.status_code == 200
        assert response.json()["count"] == 3
        assert [item["text"] for item in response.json()["results"]] == texts


# ===================================================================== async
class TestAsyncPipeline:
    def test_submission_returns_202_and_a_job_id(self, http_client):
        response = http_client.post(
            "/api/sentiment/async", json={"text": "This movie was terrible."}, timeout=30
        )

        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "processing"
        assert str(uuid.UUID(body["job_id"])) == body["job_id"]

    def test_result_is_eventually_persisted_and_retrievable(self, http_client, mongo_collection):
        text = "I absolutely adored this book, it changed my life."

        job_id = http_client.post("/api/sentiment/async", json={"text": text}, timeout=30).json()["job_id"]
        result = wait_for_result(http_client, job_id)

        assert result["job_id"] == job_id
        assert result["text"] == text
        assert result["sentiment"] in ("positive", "negative")
        assert 0.0 <= result["score"] <= 1.0

        document = mongo_collection.find_one({"job_id": job_id})
        assert document is not None, "worker did not write the result to MongoDB"
        assert document["text"] == text
        assert document["sentiment"] == result["sentiment"]
        assert document["score"] == result["score"]
        assert document["_id"] is not None
        assert isinstance(document["timestamp"], datetime)
        # PyMongo returns bson.tz_util.FixedOffset(0) for UTC, so compare the
        # offset rather than the tzinfo object identity.
        assert document["timestamp"].utcoffset() == timedelta(0)
        assert document["model"], "model card should be snapshotted with the result"
        assert document["latency_ms"] >= 0

    def test_result_timestamp_is_serialised_as_utc_iso8601(self, http_client):
        job_id = http_client.post("/api/sentiment/async", json={"text": "nice"}, timeout=30).json()["job_id"]

        result = wait_for_result(http_client, job_id)

        assert result["timestamp"].endswith("Z") or "+00:00" in result["timestamp"]

    def test_async_and_sync_paths_agree(self, http_client):
        text = "A dreadful, boring, pointless waste of an evening."

        sync_body = http_client.post("/api/sentiment/sync", json={"text": text}, timeout=30).json()
        job_id = http_client.post("/api/sentiment/async", json={"text": text}, timeout=30).json()["job_id"]
        async_body = wait_for_result(http_client, job_id)

        assert sync_body["sentiment"] == async_body["sentiment"]
        assert sync_body["score"] == pytest.approx(async_body["score"], abs=1e-6)

    def test_many_jobs_are_all_persisted(self, http_client, mongo_collection):
        texts = [f"Review number {index}: {'great' if index % 2 else 'terrible'} experience" for index in range(10)]

        job_ids = [
            http_client.post("/api/sentiment/async", json={"text": text}, timeout=30).json()["job_id"]
            for text in texts
        ]

        for job_id in job_ids:
            wait_for_result(http_client, job_id)

        assert mongo_collection.count_documents({"job_id": {"$in": job_ids}}) == 10

    def test_invalid_text_is_rejected_before_publishing(self, http_client):
        response = http_client.post("/api/sentiment/async", json={"text": ""}, timeout=30)

        assert response.status_code == 400
        assert response.json()["error_code"] == "INVALID_INPUT"


# ================================================================== results
class TestResultLookup:
    def test_unknown_job_id_returns_404(self, http_client):
        job_id = str(uuid.uuid4())

        response = http_client.get(f"/api/sentiment/results/{job_id}", timeout=30)

        assert response.status_code == 404
        body = response.json()
        assert body["error_code"] == "JOB_NOT_FOUND"
        assert job_id in body["detail"]

    def test_malformed_job_id_returns_400(self, http_client):
        response = http_client.get("/api/sentiment/results/definitely-not-a-uuid", timeout=30)

        assert response.status_code == 400
        assert response.json()["error_code"] == "INVALID_INPUT"

    def test_lookup_does_not_hit_the_worker(self, http_client):
        """A result that was never submitted stays 404 - retrieval is DB-only."""
        response = http_client.get(f"/api/sentiment/results/{uuid.uuid4()}", timeout=30)

        assert response.status_code == 404


# ============================================== reliability of the transport
class TestQueueReliability:
    def test_redelivered_job_does_not_duplicate_the_document(self, http_client, mongo_collection, amqp_channel):
        job_id = str(uuid.uuid4())
        payload = json.dumps(
            {
                "job_id": job_id,
                "text": "Absolutely brilliant and heartfelt.",
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "attempt": 1,
                "source": "integration-test",
            }
        )
        properties = pika.BasicProperties(
            content_type="application/json", delivery_mode=pika.DeliveryMode.Persistent, message_id=job_id
        )
        exchange = os.getenv("RABBITMQ_EXCHANGE", "sentiment.exchange")
        routing_key = os.getenv("RABBITMQ_ROUTING_KEY", "sentiment.analyze")

        amqp_channel.basic_publish(exchange=exchange, routing_key=routing_key, body=payload, properties=properties)
        amqp_channel.basic_publish(exchange=exchange, routing_key=routing_key, body=payload, properties=properties)

        result = wait_for_result(http_client, job_id)
        assert result["job_id"] == job_id
        assert mongo_collection.count_documents({"job_id": job_id}) == 1

    def test_malformed_message_is_dead_lettered(self, amqp_channel):
        amqp_channel.queue_purge(DEAD_LETTER_QUEUE)
        exchange = os.getenv("RABBITMQ_EXCHANGE", "sentiment.exchange")
        routing_key = os.getenv("RABBITMQ_ROUTING_KEY", "sentiment.analyze")

        amqp_channel.basic_publish(
            exchange=exchange,
            routing_key=routing_key,
            body=b"this-is-not-json",
            properties=pika.BasicProperties(content_type="application/json", delivery_mode=2),
        )

        deadline = time.monotonic() + 30
        message = None
        while time.monotonic() < deadline and message is None:
            method, properties, body = amqp_channel.basic_get(DEAD_LETTER_QUEUE, auto_ack=False)
            if method is not None:
                message = body
                amqp_channel.basic_nack(method.delivery_tag, requeue=False)
            else:
                time.sleep(0.5)

        assert message is not None, "malformed message never reached the dead letter queue"
        assert message == b"this-is-not-json"
