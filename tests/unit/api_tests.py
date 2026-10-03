"""Unit tests for the HTTP layer: routing, validation, error mapping, docs.

These tests deliberately avoid TensorFlow, RabbitMQ and MongoDB: the service
dependencies are injected through FastAPI's dependency overrides, which is the
same seam the application uses at runtime.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_model,
    get_publisher_dependency,
    get_repository_dependency,
)
from app.errors import ModelUnavailableError, PersistenceError, QueueUnavailableError
from app.main import create_app
from app.schemas import SentimentRequest
from tests.conftest import FakeModelService, FakePublisher, FakeRepository

VALID_UUID = "a1b2c3d4-e5f6-7890-1234-567890abcdef"
SYNC_URL = "/api/sentiment/sync"
ASYNC_URL = "/api/sentiment/async"
BATCH_URL = "/api/sentiment/batch"
RESULT_URL = f"/api/sentiment/results/{VALID_UUID}"


def build_client(
    settings,
    *,
    model: FakeModelService | None = None,
    publisher: FakePublisher | None = None,
    repository: FakeRepository | None = None,
) -> TestClient:
    application = create_app(settings)
    application.dependency_overrides[get_model] = lambda: model or FakeModelService()
    application.dependency_overrides[get_publisher_dependency] = lambda: publisher or FakePublisher()
    application.dependency_overrides[get_repository_dependency] = lambda: repository or FakeRepository()
    # TestClient without a context manager skips the lifespan, so the real
    # model/queue/database are never touched.
    return TestClient(application, raise_server_exceptions=False)


# --------------------------------------------------------------- sync endpoint
class TestSyncEndpoint:
    def test_returns_200_with_expected_payload(self, settings, fake_model):
        client = build_client(settings, model=fake_model)

        response = client.post(SYNC_URL, json={"text": "I love this product!"})

        assert response.status_code == 200
        assert response.json() == {
            "text": "I love this product!",
            "sentiment": "positive",
            "score": 0.9,
        }
        assert fake_model.calls == [["I love this product!"]]

    def test_negative_sentiment_is_reported(self, settings):
        model = FakeModelService(score=0.04, sentiment="negative")
        client = build_client(settings, model=model)

        body = client.post(SYNC_URL, json={"text": "This movie was terrible."}).json()

        assert body["sentiment"] == "negative"
        assert body["score"] == 0.04

    def test_trims_surrounding_whitespace(self, settings, fake_model):
        client = build_client(settings, model=fake_model)

        response = client.post(SYNC_URL, json={"text": "  spaced out  "})

        assert response.status_code == 200
        assert response.json()["text"] == "spaced out"
        assert fake_model.calls == [["spaced out"]]

    def test_response_carries_timing_and_request_id_headers(self, settings):
        client = build_client(settings)

        response = client.post(SYNC_URL, json={"text": "hello"})

        assert float(response.headers["X-Process-Time-Ms"]) >= 0
        assert response.headers["X-Request-ID"]

    def test_rejects_empty_text_with_400(self, settings):
        client = build_client(settings)

        response = client.post(SYNC_URL, json={"text": ""})

        assert response.status_code == 400
        assert response.json()["error_code"] == "INVALID_INPUT"
        assert "must not be empty" in response.json()["detail"]

    def test_rejects_whitespace_only_text_with_400(self, settings):
        client = build_client(settings)

        response = client.post(SYNC_URL, json={"text": "   \t\n  "})

        assert response.status_code == 400
        assert response.json()["error_code"] == "INVALID_INPUT"

    def test_rejects_missing_field_with_400(self, settings):
        client = build_client(settings)

        response = client.post(SYNC_URL, json={})

        assert response.status_code == 400
        assert response.json()["error_code"] == "INVALID_INPUT"
        assert "text" in response.json()["detail"]

    def test_rejects_wrong_type_with_400(self, settings):
        client = build_client(settings)

        response = client.post(SYNC_URL, json={"text": 12345})

        assert response.status_code == 400
        assert response.json()["error_code"] == "INVALID_INPUT"

    def test_rejects_non_json_body_with_400(self, settings):
        client = build_client(settings)

        response = client.post(SYNC_URL, content="not json", headers={"content-type": "application/json"})

        assert response.status_code == 400
        assert response.json()["error_code"] == "INVALID_INPUT"

    def test_rejects_oversized_text_with_400(self, settings):
        client = build_client(settings)

        response = client.post(SYNC_URL, json={"text": "x" * (settings.sync_max_text_length + 1)})

        assert response.status_code == 400
        assert response.json()["error_code"] == "INVALID_INPUT"

    def test_model_failure_maps_to_503(self, settings):
        model = FakeModelService(fail_with=ModelUnavailableError("model is not loaded"))
        client = build_client(settings, model=model)

        response = client.post(SYNC_URL, json={"text": "hello"})

        assert response.status_code == 503
        assert response.json()["error_code"] == "MODEL_UNAVAILABLE"

    def test_unexpected_failure_maps_to_500(self, settings):
        model = FakeModelService(fail_with=RuntimeError("kernel exploded"))
        client = build_client(settings, model=model)

        response = client.post(SYNC_URL, json={"text": "hello"})

        assert response.status_code == 500
        assert response.json()["error_code"] == "INTERNAL_ERROR"
        assert "kernel exploded" not in response.json()["detail"]


# -------------------------------------------------------------- async endpoint
class TestAsyncEndpoint:
    def test_returns_202_and_publishes_to_the_queue(self, settings, fake_publisher):
        client = build_client(settings, publisher=fake_publisher)

        response = client.post(ASYNC_URL, json={"text": "This movie was terrible."})

        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "processing"
        assert str(uuid.UUID(body["job_id"])) == body["job_id"]

        assert len(fake_publisher.published) == 1
        published = fake_publisher.published[0]
        assert published.job_id == body["job_id"]
        assert published.text == "This movie was terrible."

    def test_generates_unique_job_ids(self, settings, fake_publisher):
        client = build_client(settings, publisher=fake_publisher)

        first = client.post(ASYNC_URL, json={"text": "a"}).json()["job_id"]
        second = client.post(ASYNC_URL, json={"text": "b"}).json()["job_id"]

        assert first != second

    def test_rejects_invalid_text_with_400(self, settings, fake_publisher):
        client = build_client(settings, publisher=fake_publisher)

        response = client.post(ASYNC_URL, json={"text": "   "})

        assert response.status_code == 400
        assert response.json()["error_code"] == "INVALID_INPUT"
        assert fake_publisher.published == []

    def test_broker_outage_maps_to_503(self, settings):
        publisher = FakePublisher(fail_with=QueueUnavailableError("broker unreachable"))
        client = build_client(settings, publisher=publisher)

        response = client.post(ASYNC_URL, json={"text": "hello"})

        assert response.status_code == 503
        assert response.json()["error_code"] == "QUEUE_UNAVAILABLE"


# ------------------------------------------------------------- results lookup
class TestResultsEndpoint:
    def test_returns_200_with_stored_result(self, settings):
        from datetime import datetime, timezone

        repository = FakeRepository()
        repository.documents[VALID_UUID] = {
            "job_id": VALID_UUID,
            "text": "This movie was terrible.",
            "sentiment": "negative",
            "score": 0.052,
            "timestamp": datetime(2023, 10, 27, 10, 30, tzinfo=timezone.utc),
        }
        client = build_client(settings, repository=repository)

        response = client.get(RESULT_URL)

        assert response.status_code == 200
        assert response.json() == {
            "job_id": VALID_UUID,
            "text": "This movie was terrible.",
            "sentiment": "negative",
            "score": 0.052,
            "timestamp": "2023-10-27T10:30:00Z",
        }

    def test_unknown_job_id_returns_404(self, settings):
        client = build_client(settings)

        response = client.get(f"/api/sentiment/results/{uuid.uuid4()}")

        assert response.status_code == 404
        body = response.json()
        assert body["error_code"] == "JOB_NOT_FOUND"
        assert "not found" in body["detail"]

    def test_malformed_job_id_returns_400(self, settings):
        client = build_client(settings)

        response = client.get("/api/sentiment/results/not-a-uuid")

        assert response.status_code == 400
        assert response.json()["error_code"] == "INVALID_INPUT"

    def test_database_failure_maps_to_500(self, settings):
        class BrokenRepository(FakeRepository):
            def get_by_job_id(self, job_id):
                raise PersistenceError("mongo is down")

        client = build_client(settings, repository=BrokenRepository())

        response = client.get(RESULT_URL)

        assert response.status_code == 500
        assert response.json()["error_code"] == "PERSISTENCE_ERROR"


# ------------------------------------------------------------------- batching
class TestBatchEndpoint:
    def test_returns_results_in_submission_order(self, settings, fake_model):
        client = build_client(settings, model=fake_model)

        response = client.post(BATCH_URL, json={"texts": ["first", "second", "third"]})

        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 3
        assert [item["text"] for item in body["results"]] == ["first", "second", "third"]

    def test_rejects_empty_list(self, settings):
        client = build_client(settings)

        response = client.post(BATCH_URL, json={"texts": []})

        assert response.status_code == 400
        assert response.json()["error_code"] == "INVALID_INPUT"

    def test_rejects_blank_item(self, settings):
        client = build_client(settings)

        response = client.post(BATCH_URL, json={"texts": ["ok", "   "]})

        assert response.status_code == 400
        assert "index 1" in response.json()["detail"]


# ------------------------------------------------------------ health and docs
class TestHealthAndDocs:
    def test_health_is_ok(self, settings):
        client = build_client(settings)

        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["service"]

    def test_readiness_reports_all_dependencies_up(self, settings):
        client = build_client(
            settings,
            model=FakeModelService(is_loaded=True),
            publisher=FakePublisher(connected=True),
            repository=FakeRepository(connected=True),
        )

        response = client.get("/health/ready")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["dependencies"]["mongodb"]["status"] == "up"
        assert response.json()["dependencies"]["rabbitmq"]["status"] == "up"

    def test_readiness_returns_503_when_a_dependency_is_down(self, settings):
        client = build_client(settings, repository=FakeRepository(connected=False))

        response = client.get("/health/ready")

        assert response.status_code == 503
        assert response.json()["status"] == "degraded"
        assert response.json()["dependencies"]["mongodb"]["status"] == "down"

    def test_readiness_reconnects_the_publisher_instead_of_just_reading_it(self, settings):
        publisher = FakePublisher(connected=True)
        client = build_client(settings, publisher=publisher)

        assert client.get("/health/ready").status_code == 200
        # An idle connection can be reaped by the broker heartbeat; readiness
        # must attempt a reconnect rather than report a false 503.
        assert publisher.ensure_connected_calls == 1

    def test_readiness_is_503_when_the_publisher_cannot_reconnect(self, settings):
        client = build_client(settings, publisher=FakePublisher(connected=False))

        response = client.get("/health/ready")

        assert response.status_code == 503
        assert response.json()["dependencies"]["rabbitmq"]["status"] == "down"

    def test_openapi_documents_every_contract(self, settings):
        client = build_client(settings)

        schema = client.get("/openapi.json").json()

        assert "/api/sentiment/sync" in schema["paths"]
        assert "/api/sentiment/async" in schema["paths"]
        assert "/api/sentiment/results/{job_id}" in schema["paths"]
        assert "400" in schema["paths"]["/api/sentiment/sync"]["post"]["responses"]
        assert "404" in schema["paths"]["/api/sentiment/results/{job_id}"]["get"]["responses"]
        assert "202" in schema["paths"]["/api/sentiment/async"]["post"]["responses"]
        assert "ErrorResponse" in schema["components"]["schemas"]
        assert "SyncSentimentResponse" in schema["components"]["schemas"]
        assert "JobResultResponse" in schema["components"]["schemas"]

    def test_swagger_ui_is_served(self, settings):
        client = build_client(settings)

        assert client.get("/docs").status_code == 200

    def test_root_advertises_endpoints(self, settings):
        client = build_client(settings)

        body = client.get("/").json()

        assert body["endpoints"]["sync"] == "POST /api/sentiment/sync"
        assert body["docs"] == "/docs"


# ------------------------------------------------------------------- schemas
class TestSchemas:
    @pytest.mark.parametrize("value", ["hello", " hello ", "hëllo wörld", "12345"])
    def test_accepts_non_empty_strings(self, value):
        assert SentimentRequest(text=value).text == value.strip()

    @pytest.mark.parametrize("value", ["", "   ", "\n\t"])
    def test_rejects_blank_text(self, value):
        with pytest.raises(ValueError):
            SentimentRequest(text=value)
