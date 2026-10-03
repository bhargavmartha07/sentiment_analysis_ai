"""Unit tests for the service layer: model, queue, database, worker, logging.

Covers the three concerns named in the brief:

* **model_service** - text encoding, label/score mapping, artefact loading errors
* **queue_service** - message serialisation, topology declaration, publisher
  error mapping (both producer and consumer side)
* **db_service** - document schema, upsert idempotency, lookup and index setup

plus the worker's acknowledgement/retry/dead-letter policy.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import mongomock
import pika
import pytest
from pika.exceptions import AMQPConnectionError, UnroutableError

from app.config import Settings
from app.errors import ModelUnavailableError, PersistenceError, QueueUnavailableError
from app.logging_config import JsonFormatter, configure_logging
from app.services.db_service import SentimentRepository, document_to_api
from app.services.model_service import Prediction, SentimentModelService
from app.services.queue_service import (
    Delivery,
    QueueConsumer,
    QueuePublisher,
    SentimentJobMessage,
    declare_topology,
)
from app.worker.consumer import SentimentWorker
from tests.conftest import FakeDelivery, FakeModelService, FakeRepository

# ===========================================================================
# model_service
# ===========================================================================


class _FakeKerasModel:
    """Minimal object satisfying the inference contract of the real model."""

    def __init__(self, scores=(0.9,)) -> float | None:
        self._scores = list(scores)
        self.seen: list[list[list[int]]] = []
        self.input_shape = (None, 4)
        self.count_params = lambda: 1234

    def predict(self, batch, batch_size=None, verbose=0):
        self.seen.append(batch.tolist())
        size = len(batch)
        scores = []
        for index in range(size):
            scores.append([self._scores[index % len(self._scores)]])
        return scores


def _loaded_service(settings: Settings, artifacts: dict[str, str] | None = None, *, scores=(0.9,)):
    service = SentimentModelService(settings)
    service._word_index = {"great": 3, "movie": 4, "terrible": 5}
    service._metadata = {"architecture": "test", "max_len": 4}
    service._max_len = 4
    service._model = _FakeKerasModel(scores)
    return service


class TestModelServiceEncoding:
    def test_known_words_map_to_vocabulary_ids(self, settings):
        service = _loaded_service(settings)

        assert service.tokenize("great movie") == [3, 4, 0, 0]

    def test_unknown_words_map_to_oov(self, settings):
        service = _loaded_service(settings)

        # Row 2 is the out-of-vocabulary row reserved by keras.datasets.imdb.
        assert service.tokenize("great zebrafish") == [3, 2, 0, 0]

    def test_text_is_lowercased_and_punctuation_split(self, settings):
        service = _loaded_service(settings)

        assert service.tokenize("GREAT, movie!") == [3, 4, 0, 0]

    def test_contractions_stay_single_tokens(self, settings):
        service = _loaded_service(settings)

        # "don't" is not in the tiny fixture vocabulary, so it becomes one OOV.
        assert service.tokenize("don't") == [2, 0, 0, 0]

    def test_sequences_are_truncated_to_max_len(self, settings):
        service = _loaded_service(settings)

        assert service.tokenize("great great great great great") == [3, 3, 3, 3]

    def test_tokenize_batch_produces_a_rectangular_matrix(self, settings):
        service = _loaded_service(settings)

        matrix = service.tokenize_batch(["great", "great movie"])

        assert matrix.shape == (2, 4)
        assert matrix.tolist() == [[3, 0, 0, 0], [3, 4, 0, 0]]


class TestModelServiceInference:
    def test_positive_score_maps_to_positive_label(self, settings):
        service = _loaded_service(settings, scores=(0.985,))

        prediction = service.predict("I love this product!")

        assert isinstance(prediction, Prediction)
        assert prediction.sentiment == "positive"
        assert prediction.score == 0.985
        assert prediction.is_positive is True

    def test_low_score_maps_to_negative_label(self, settings):
        service = _loaded_service(settings, scores=(0.052,))

        prediction = service.predict("This movie was terrible.")

        assert prediction.sentiment == "negative"
        assert prediction.score == 0.052

    def test_threshold_is_configurable(self, model_artifacts):
        settings = Settings(positive_threshold=0.3)
        service = _loaded_service(settings, scores=(0.4,))

        assert service.predict("mixed").sentiment == "positive"

    def test_score_is_clamped_and_rounded(self, settings):
        service = _loaded_service(settings, scores=(1.4,))
        assert service.predict("x").score == 1.0

        service = _loaded_service(Settings(score_precision=2), scores=(0.987654,))
        assert service.predict("x").score == 0.99

    def test_batch_returns_one_prediction_per_input(self, settings):
        service = _loaded_service(settings, scores=(0.9, 0.1))

        predictions = service.predict_batch(["a", "b", "c"])

        assert [p.sentiment for p in predictions] == ["positive", "negative", "positive"]

    def test_empty_batch_short_circuits(self, settings):
        service = _loaded_service(settings)

        assert service.predict_batch([]) == []

    def test_predict_before_load_raises_service_error(self, settings, model_artifacts):
        service = SentimentModelService(settings)

        with pytest.raises(ModelUnavailableError) as excinfo:
            service.predict("hello")

        assert excinfo.value.error_code.value == "MODEL_UNAVAILABLE"
        assert excinfo.value.status_code == 503

    def test_load_missing_artifact_reports_the_path(self, settings, model_artifacts):
        broken = settings.model_copy(update={"model_path": f"{model_artifacts['model']}.missing"})
        service = SentimentModelService(broken)

        with pytest.raises(ModelUnavailableError) as excinfo:
            service.load()

        assert "sentiment_model.h5" in excinfo.value.detail
        assert excinfo.value.status_code == 503

    def test_load_missing_vocabulary_reports_the_path(self, settings, model_artifacts):
        from pathlib import Path

        # load() validates the vocabulary after the model, so point the model at
        # the real artefact to reach the vocabulary branch under test.
        shipped = Path("app/models/sentiment_model.h5")
        if not shipped.is_file():
            pytest.skip("shipped model artefact is not present in this image")
        broken = settings.model_copy(
            update={
                "model_path": str(shipped),
                "word_index_path": f"{model_artifacts['word_index']}.missing",
            }
        )
        service = SentimentModelService(broken)

        with pytest.raises(ModelUnavailableError, match="word_index"):
            service.load()


class TestModelServiceVocabularyLoading:
    def test_reads_word_index_mapping(self, settings, model_artifacts):
        from pathlib import Path

        service = SentimentModelService(settings)

        vocabulary = service._load_vocabulary(Path(model_artifacts["word_index"]))

        assert vocabulary == {"great": 3, "movie": 4, "terrible": 5}

    def test_accepts_list_form_vocabulary(self, settings, model_artifacts):
        from pathlib import Path

        service = SentimentModelService(settings)
        path = Path(model_artifacts["word_index"])
        # List form is positional: rows 0/1/2 are reserved, so "great" lands on 3.
        path.write_text(json.dumps(["pad", "start", "oov", "great"]), encoding="utf-8")

        vocabulary = service._load_vocabulary(path)

        assert vocabulary == {"great": 3}

    def test_invalid_metadata_is_ignored(self, settings, model_artifacts):
        from pathlib import Path

        service = SentimentModelService(settings)
        path = Path(model_artifacts["metadata"])
        path.write_text("{not json", encoding="utf-8")

        assert service._load_metadata() == {}


# ===========================================================================
# queue_service
# ===========================================================================


class RecordingChannel:
    """Captures every AMQP call for assertions."""

    def __init__(self) -> None:
        self.exchanges: list[dict] = []
        self.queues: list[dict] = []
        self.bindings: list[dict] = []
        self.published: list[dict] = []
        self.confirms_enabled = False
        self.qos: dict | None = None
        self.is_open = True

    def exchange_declare(self, **kwargs):
        self.exchanges.append(kwargs)

    def queue_declare(self, **kwargs):
        self.queues.append(kwargs)

    def queue_bind(self, **kwargs):
        self.bindings.append(kwargs)

    def confirm_delivery(self):
        self.confirms_enabled = True

    def basic_qos(self, **kwargs):
        self.qos = kwargs

    def basic_publish(self, **kwargs):
        self.published.append(kwargs)
        if getattr(self, "publish_error", None):
            raise self.publish_error

    def basic_ack(self, **kwargs):
        self.acks.append(kwargs)

    def basic_nack(self, **kwargs):
        self.nacks.append(kwargs)


class FakeConnection:
    def __init__(self, channel: RecordingChannel) -> None:
        self._channel = channel
        self.is_closed = False
        self.closed = False

    def channel(self):
        return self._channel

    def process_data_events(self, time_limit=None):
        time.sleep(min(time_limit or 0.01, 0.01))

    def close(self):
        self.closed = True
        self.is_closed = True


class TestJobMessage:
    def test_round_trip(self):
        message = SentimentJobMessage.create("job-1", "great movie")

        restored = SentimentJobMessage.from_json(message.to_json())

        assert restored.job_id == "job-1"
        assert restored.text == "great movie"
        assert restored.attempt == 1
        assert restored.source == "api"
        assert restored.submitted_at.endswith("Z")

    def test_payload_is_valid_json_with_required_fields(self):
        payload = json.loads(SentimentJobMessage.create("job-2", "text").to_json())

        assert set(payload) >= {"job_id", "text", "submitted_at", "attempt", "source"}

    def test_unicode_is_preserved(self):
        message = SentimentJobMessage.create("job-3", "café  screenplay \U0001f600")

        assert SentimentJobMessage.from_json(message.to_json()).text == message.text

    def test_rejects_missing_fields(self):
        with pytest.raises(ValueError, match="job_id"):
            SentimentJobMessage.from_json(json.dumps({"text": "hello"}))

    def test_rejects_non_object_payload(self):
        with pytest.raises(ValueError):
            SentimentJobMessage.from_json("[1, 2, 3]")

    def test_rejects_invalid_json(self):
        with pytest.raises(ValueError):
            SentimentJobMessage.from_json("not-json")


class TestTopology:
    def test_declares_durable_exchange_and_dead_letter_queue(self, settings):
        channel = RecordingChannel()

        declare_topology(channel, settings)

        assert {e["exchange"] for e in channel.exchanges} == {
            settings.rabbitmq_exchange,
            settings.rabbitmq_dead_letter_exchange,
        }
        assert all(e["durable"] is True and e["auto_delete"] is False for e in channel.exchanges)
        assert {q["queue"] for q in channel.queues} == {
            settings.rabbitmq_queue,
            settings.rabbitmq_dead_letter_queue,
        }

    def test_work_queue_routes_to_the_dead_letter_exchange(self, settings):
        channel = RecordingChannel()

        declare_topology(channel, settings)

        work_queue = next(q for q in channel.queues if q["queue"] == settings.rabbitmq_queue)
        assert work_queue["arguments"]["x-dead-letter-exchange"] == settings.rabbitmq_dead_letter_exchange

    def test_queue_is_bound_to_the_exchange(self, settings):
        channel = RecordingChannel()

        declare_topology(channel, settings)

        binding = next(b for b in channel.bindings if b["queue"] == settings.rabbitmq_queue)
        assert binding["exchange"] == settings.rabbitmq_exchange
        assert binding["routing_key"] == settings.rabbitmq_routing_key

    def test_declaration_is_idempotent_on_a_real_channel(self, settings, monkeypatch):
        channel = RecordingChannel()
        monkeypatch.setattr(pika, "BlockingConnection", lambda *a, **k: FakeConnection(channel))

        publisher = QueuePublisher(settings)
        publisher.publish(SentimentJobMessage.create("job-1", "hello"))
        publisher.publish(SentimentJobMessage.create("job-2", "hello"))

        assert len(channel.published) == 2


class TestQueuePublisher:
    def _publisher(self, settings, channel: RecordingChannel, monkeypatch) -> QueuePublisher:
        monkeypatch.setattr(pika, "BlockingConnection", lambda *a, **k: FakeConnection(channel))
        return QueuePublisher(settings)

    def test_publishes_persistent_message_with_confirms(self, settings, monkeypatch):
        channel = RecordingChannel()
        publisher = self._publisher(settings, channel, monkeypatch)

        publisher.publish(SentimentJobMessage.create("job-42", "great movie"))

        assert channel.confirms_enabled is True
        published = channel.published[0]
        assert published["exchange"] == settings.rabbitmq_exchange
        assert published["routing_key"] == settings.rabbitmq_routing_key
        assert published["mandatory"] is True
        assert published["properties"].delivery_mode == pika.DeliveryMode.Persistent.value == 2
        assert published["properties"].message_id == "job-42"
        assert json.loads(published["body"])["text"] == "great movie"

    def test_increments_publish_counter(self, settings, monkeypatch):
        publisher = self._publisher(settings, RecordingChannel(), monkeypatch)

        publisher.publish(SentimentJobMessage.create("a", "x"))
        publisher.publish(SentimentJobMessage.create("b", "x"))

        assert publisher.stats()["published_total"] == 2
        assert publisher.stats()["connected"] is True

    def test_unroutable_message_maps_to_queue_unavailable(self, settings, monkeypatch):
        channel = RecordingChannel()
        channel.publish_error = UnroutableError(messages=[])
        publisher = self._publisher(settings, channel, monkeypatch)

        with pytest.raises(QueueUnavailableError) as excinfo:
            publisher.publish(SentimentJobMessage.create("job-1", "hello"))

        assert excinfo.value.error_code.value == "QUEUE_UNAVAILABLE"
        assert excinfo.value.status_code == 503

    def test_broker_connection_failure_maps_to_queue_unavailable(self, settings, monkeypatch):
        def explode(*args, **kwargs):
            raise AMQPConnectionError("connection refused")

        monkeypatch.setattr(pika, "BlockingConnection", explode)
        publisher = QueuePublisher(settings)

        with pytest.raises(QueueUnavailableError):
            publisher.publish(SentimentJobMessage.create("job-1", "hello"))

        assert publisher.is_connected is False

    def test_enters_backoff_after_a_failure(self, settings, monkeypatch):
        def explode(*args, **kwargs):
            raise AMQPConnectionError("down")

        monkeypatch.setattr(pika, "BlockingConnection", explode)
        publisher = QueuePublisher(settings)

        with pytest.raises(QueueUnavailableError):
            publisher.publish(SentimentJobMessage.create("job-1", "hello"))
        with pytest.raises(QueueUnavailableError, match="backoff"):
            publisher.publish(SentimentJobMessage.create("job-2", "hello"))

    def test_close_is_safe_when_never_connected(self, settings):
        QueuePublisher(settings).close()

    def test_publish_is_thread_safe(self, settings, monkeypatch):
        import threading

        channel = RecordingChannel()
        publisher = self._publisher(settings, channel, monkeypatch)
        errors: list[Exception] = []

        def worker(index: int) -> None:
            try:
                publisher.publish(SentimentJobMessage.create(f"job-{index}", "text"))
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert len(channel.published) == 20
        assert publisher.stats()["published_total"] == 20


class TestDelivery:
    def test_acknowledge_uses_manual_ack(self, settings):
        channel = SimpleNamespace(basic_ack=lambda **kw: channel.acks.append(kw), acks=[])
        delivery = Delivery(
            message=SentimentJobMessage.create("job-1", "text"),
            delivery_tag=7,
            redelivered=False,
            retry_count=0,
            raw_body=b"{}",
            _channel=channel,
        )

        delivery.acknowledge()

        assert channel.acks == [{"delivery_tag": 7}]

    def test_reject_can_requeue_for_another_attempt(self, settings):
        channel = SimpleNamespace(basic_nack=lambda **kw: channel.nacks.append(kw), nacks=[])
        delivery = Delivery(
            message=SentimentJobMessage.create("job-1", "text"),
            delivery_tag=7,
            redelivered=True,
            retry_count=1,
            raw_body=b"{}",
            _channel=channel,
        )

        delivery.reject(requeue=True)

        assert channel.nacks == [{"delivery_tag": 7, "requeue": True}]

    def _retry_channel(self, *, confirmed=True):
        recorder = SimpleNamespace(published=[], acks=[], nacks=[])
        recorder.basic_publish = lambda **kw: recorder.published.append(kw) or confirmed
        recorder.basic_ack = lambda **kw: recorder.acks.append(kw)
        recorder.basic_nack = lambda **kw: recorder.nacks.append(kw)
        return recorder

    def _retry_delivery(self, channel):
        return Delivery(
            message=SentimentJobMessage.create("job-1", "text"),
            delivery_tag=7,
            redelivered=False,
            retry_count=0,
            raw_body=b'{"job_id":"job-1"}',
            _channel=channel,
        )

    def test_requeue_for_retry_republishes_with_an_incremented_header(self, settings):
        channel = self._retry_channel()

        self._retry_delivery(channel).requeue_for_retry(
            retry_count=2, exchange="sentiment.exchange", routing_key="sentiment.analyze"
        )

        published = channel.published[0]
        assert published["exchange"] == "sentiment.exchange"
        assert published["routing_key"] == "sentiment.analyze"
        assert published["mandatory"] is True
        assert published["properties"].headers["x-retry-count"] == 2
        assert published["body"] == b'{"job_id":"job-1"}'
        # Acked only after the broker confirmed the republish.
        assert channel.acks == [{"delivery_tag": 7}]
        assert channel.nacks == []

    def test_unconfirmed_republish_leaves_the_job_on_the_queue(self, settings):
        from pika.exceptions import ChannelError

        channel = self._retry_channel(confirmed=False)

        with pytest.raises(ChannelError):
            self._retry_delivery(channel).requeue_for_retry(
                retry_count=1, exchange="sentiment.exchange", routing_key="sentiment.analyze"
            )

        assert channel.acks == []
        assert channel.nacks == [{"delivery_tag": 7, "requeue": True}]


class TestQueueConsumer:
    def test_applies_prefetch_limit(self, settings, monkeypatch):
        channel = RecordingChannel()
        monkeypatch.setattr(pika, "BlockingConnection", lambda *a, **k: FakeConnection(channel))

        QueueConsumer(settings).connect()

        assert channel.qos == {"prefetch_count": settings.rabbitmq_prefetch_count}
        assert channel.exchanges  # topology declared on connect

    def test_reconnects_after_broker_outage(self, settings, monkeypatch):
        import threading

        attempts = {"count": 0}

        def flaky_connection(*args, **kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise AMQPConnectionError("broker restarted")
            return FakeConnection(RecordingChannel())

        monkeypatch.setattr(pika, "BlockingConnection", flaky_connection)
        resilient = settings.model_copy(update={"rabbitmq_connection_retry_seconds": 0.01})
        consumer = QueueConsumer(resilient)
        stop_event = threading.Event()

        def drain() -> None:
            for _ in consumer.messages(stop_event):
                pass

        worker_thread = threading.Thread(target=drain, daemon=True)
        worker_thread.start()
        time.sleep(0.4)
        stop_event.set()
        worker_thread.join(timeout=5)

        assert attempts["count"] >= 2, "consumer never re-established the connection"
        assert not worker_thread.is_alive()

    def test_connect_propagates_broker_failure(self, settings, monkeypatch):
        def explode(*args, **kwargs):
            raise AMQPConnectionError("down")

        monkeypatch.setattr(pika, "BlockingConnection", explode)

        with pytest.raises(AMQPConnectionError):
            QueueConsumer(settings).connect()


# ===========================================================================
# db_service
# ===========================================================================


def _repository(settings: Settings) -> SentimentRepository:
    return SentimentRepository(settings, client=mongomock.MongoClient())


class TestSentimentRepository:
    def test_saves_the_documented_schema(self, settings):
        repository = _repository(settings)

        document = repository.save_result(
            job_id="job-1",
            text="I love this product!",
            prediction=Prediction(score=0.985, sentiment="positive"),
            model_card={"architecture": "test"},
            latency_ms=12,
        )

        assert document["job_id"] == "job-1"
        assert document["text"] == "I love this product!"
        assert document["sentiment"] == "positive"
        assert document["score"] == 0.985
        assert isinstance(document["timestamp"], datetime)
        assert document["model"] == {"architecture": "test"}
        assert document["latency_ms"] == 12

    def test_timestamp_is_utc(self, settings):
        repository = _repository(settings)

        repository.save_result(
            job_id="job-1", text="x", prediction=Prediction(score=0.5, sentiment="positive")
        )

        stored = repository.collection.find_one({"job_id": "job-1"})
        assert stored["timestamp"].utcoffset() == timedelta(0)

    def test_timestamp_read_back_is_timezone_aware(self, settings):
        repository = _repository(settings)
        repository.save_result(
            job_id="job-1", text="x", prediction=Prediction(score=0.5, sentiment="positive")
        )

        found = repository.get_by_job_id("job-1")

        assert found["timestamp"].tzinfo is not None
        assert found["timestamp"].utcoffset() == timedelta(0)

    def test_get_by_job_id_returns_the_stored_result(self, settings):
        repository = _repository(settings)
        repository.save_result(job_id="job-1", text="x", prediction=Prediction(score=0.1, sentiment="negative"))

        found = repository.get_by_job_id("job-1")

        assert found is not None
        assert found["sentiment"] == "negative"
        assert set(found) == {"job_id", "text", "sentiment", "score", "timestamp"}

    def test_missing_job_returns_none(self, settings):
        assert _repository(settings).get_by_job_id("nope") is None

    def test_upsert_is_idempotent_under_redelivery(self, settings):
        repository = _repository(settings)

        for _ in range(3):
            repository.save_result(
                job_id="job-1", text="x", prediction=Prediction(score=0.5, sentiment="positive")
            )

        assert repository.count({"job_id": "job-1"}) == 1
        assert repository.get_by_job_id("job-1")["score"] == 0.5

    def test_unique_index_on_job_id_is_created(self, settings):
        repository = _repository(settings)

        repository.ensure_indexes()

        indexes = repository.collection.index_information()
        assert "job_id_unique" in indexes
        assert indexes["job_id_unique"]["unique"] is True

    def test_job_id_index_is_declared_unique(self, settings):
        """mongomock does not enforce uniqueness, so assert the index contract itself."""
        repository = _repository(settings)
        repository.ensure_indexes()

        options = repository.collection.index_information()["job_id_unique"]

        assert options["unique"] is True
        assert options["key"] == [("job_id", 1)]

    def test_ensure_indexes_is_idempotent(self, settings):
        repository = _repository(settings)

        repository.ensure_indexes()
        repository.ensure_indexes()

        assert set(repository.collection.index_information()) == {
            "_id_",
            "job_id_unique",
            "timestamp_desc",
            "sentiment_timestamp",
        }

    def test_latest_returns_newest_first(self, settings):
        repository = _repository(settings)
        for index in range(3):
            repository.save_result(
                job_id=f"job-{index}",
                text=f"text {index}",
                prediction=Prediction(score=0.5, sentiment="positive"),
                timestamp=datetime(2024, 1, index + 1, tzinfo=timezone.utc),
            )

        latest = repository.latest(limit=2)

        assert [item["job_id"] for item in latest] == ["job-2", "job-1"]

    def test_ping_reports_health(self, settings):
        assert _repository(settings).ping() is True

    def test_driver_errors_are_wrapped(self, settings, monkeypatch):
        from pymongo.errors import ConnectionFailure

        repository = _repository(settings)

        def explode(*args, **kwargs):
            raise ConnectionFailure("server selection timed out")

        monkeypatch.setattr(repository, "_collection", explode)

        with pytest.raises(PersistenceError) as excinfo:
            repository.get_by_job_id("job-1")
        assert excinfo.value.error_code.value == "PERSISTENCE_ERROR"

        with pytest.raises(PersistenceError):
            repository.save_result(job_id="job-1", text="x", prediction=Prediction(score=0.5, sentiment="positive"))

    def test_close_is_safe(self, settings):
        repository = _repository(settings)
        repository.close()
        repository.close()

    def test_delete_job_removes_document(self, settings):
        repository = _repository(settings)
        repository.save_result(job_id="job-1", text="x", prediction=Prediction(score=0.5, sentiment="positive"))

        assert repository.delete_job("job-1") is True
        assert repository.get_by_job_id("job-1") is None


class TestDocumentMapping:
    def test_maps_mongo_document_to_api_shape(self):
        payload = document_to_api(
            {
                "job_id": "job-1",
                "text": "hello",
                "sentiment": "positive",
                "score": 0.9,
                "timestamp": datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc),
            }
        )

        assert payload == {
            "job_id": "job-1",
            "text": "hello",
            "sentiment": "positive",
            "score": 0.9,
            "timestamp": datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc),
        }

    def test_naive_timestamps_are_treated_as_utc(self):
        payload = document_to_api(
            {
                "job_id": "job-1",
                "text": "hello",
                "sentiment": "negative",
                "score": 0.1,
                "timestamp": datetime(2024, 5, 1, 12, 0),
            }
        )

        assert payload["timestamp"].tzinfo == timezone.utc

    def test_none_document_maps_to_none(self):
        assert document_to_api(None) is None


# ===========================================================================
# worker
# ===========================================================================


class TestWorkerProcessing:
    def _worker(self, settings, model, repository, **overrides):
        settings = settings.model_copy(update=overrides)
        return SentimentWorker(settings, model=model, repository=repository, consumer=None)

    def test_successful_job_is_acknowledged_and_persisted(self, settings, tmp_path):
        repository = FakeRepository()
        worker = self._worker(settings, FakeModelService(), repository)
        delivery = FakeDelivery(SentimentJobMessage.create("job-1", "great movie"))

        worker._process(delivery)

        assert delivery.acknowledged is True
        assert delivery.rejected is False
        assert repository.documents["job-1"]["sentiment"] == "positive"
        assert repository.documents["job-1"]["text"] == "great movie"

    def test_persistence_failure_is_requeued_when_retries_remain(self, settings):
        class BrokenRepository(FakeRepository):
            def save_result(self, **kwargs):
                raise PersistenceError("mongo unreachable")

        worker = self._worker(settings, FakeModelService(), BrokenRepository())
        delivery = FakeDelivery(SentimentJobMessage.create("job-1", "text"), retry_count=0)

        worker._process(delivery)

        # The retry header must advance, otherwise the job is redelivered
        # forever and never reaches the dead-letter queue.
        assert delivery.retried_with == 1
        assert delivery.rejected is False
        assert delivery.acknowledged is True

    def test_retry_header_keeps_climbing_until_the_budget_is_spent(self, settings):
        class BrokenRepository(FakeRepository):
            def save_result(self, **kwargs):
                raise PersistenceError("mongo unreachable")

        worker = self._worker(
            settings, FakeModelService(), BrokenRepository(), worker_max_retries=3
        )

        for attempt in (1, 2):
            delivery = FakeDelivery(
                SentimentJobMessage.create("job-1", "text"), retry_count=attempt - 1
            )
            worker._process(delivery)
            assert delivery.retried_with == attempt
            assert worker._dead_lettered == 0

        final = FakeDelivery(SentimentJobMessage.create("job-1", "text"), retry_count=2)
        worker._process(final)

        assert final.retried_with is None
        assert final.rejected is True
        assert final.requeue is False
        assert worker._dead_lettered == 1

    def test_persistence_failure_is_dead_lettered_when_retries_exhausted(self, settings):
        class BrokenRepository(FakeRepository):
            def save_result(self, **kwargs):
                raise PersistenceError("mongo unreachable")

        worker = self._worker(
            settings, FakeModelService(), BrokenRepository(), worker_max_retries=2
        )
        delivery = FakeDelivery(SentimentJobMessage.create("job-1", "text"), retry_count=1)

        worker._process(delivery)

        assert delivery.rejected is True
        assert delivery.requeue is False
        assert worker._dead_lettered == 1

    def test_unexpected_error_never_acknowledges(self, settings):
        model = FakeModelService(fail_with=RuntimeError("kernel failure"))
        repository = FakeRepository()
        worker = self._worker(settings, model, repository, worker_max_retries=1)
        delivery = FakeDelivery(SentimentJobMessage.create("job-1", "text"))

        worker._process(delivery)

        assert delivery.acknowledged is False
        assert delivery.rejected is True
        assert delivery.requeue is False
        assert repository.saved == []


class TestWorkerHeartbeat:
    def test_healthcheck_passes_for_a_fresh_heartbeat(self, settings, tmp_path):
        path = tmp_path / "heartbeat.json"
        worker = SentimentWorker(settings, model=FakeModelService(), repository=FakeRepository())
        worker._settings = settings.model_copy(update={"worker_heartbeat_path": str(path)})

        worker._write_heartbeat("idle")

        assert worker.healthcheck() is True
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["state"] == "idle"
        assert payload["processed_total"] == 0

    def test_healthcheck_fails_when_heartbeat_is_missing(self, settings, tmp_path):
        worker = SentimentWorker(settings, model=FakeModelService(), repository=FakeRepository())
        worker._settings = settings.model_copy(update={"worker_heartbeat_path": str(tmp_path / "absent.json")})

        assert worker.healthcheck() is False

    def test_healthcheck_fails_for_a_stale_heartbeat(self, settings, tmp_path):
        path = tmp_path / "heartbeat.json"
        worker = SentimentWorker(settings, model=FakeModelService(), repository=FakeRepository())
        worker._settings = settings.model_copy(
            update={"worker_heartbeat_path": str(path), "worker_heartbeat_stale_seconds": 0.05}
        )
        worker._write_heartbeat("idle")
        time.sleep(0.2)

        assert worker.healthcheck() is False

    def test_heartbeat_records_processing_counters(self, settings, tmp_path):
        path = tmp_path / "heartbeat.json"
        worker = SentimentWorker(settings, model=FakeModelService(), repository=FakeRepository())
        worker._settings = settings.model_copy(update={"worker_heartbeat_path": str(path)})
        worker._processed = 5
        worker._failed = 2

        worker._write_heartbeat("idle")

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["processed_total"] == 5
        assert payload["failed_total"] == 2


class TestWorkerCli:
    def test_healthcheck_flag_is_parsed(self):
        from app.worker.consumer import parse_args

        assert parse_args(["--healthcheck"]).healthcheck is True
        assert parse_args([]).healthcheck is False


# ===========================================================================
# logging
# ===========================================================================


class TestStructuredLogging:
    def test_emits_json_with_timestamp_and_service(self, capsys):
        formatter = JsonFormatter(service="worker")
        record = logging.LogRecord(
            name="app.worker.consumer",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="Job completed",
            args=(),
            exc_info=None,
        )
        record.service = "worker"
        record.job_id = "job-1"

        payload = json.loads(formatter.format(record))

        assert payload["level"] == "INFO"
        assert payload["service"] == "worker"
        assert payload["message"] == "Job completed"
        assert payload["job_id"] == "job-1"
        assert payload["timestamp"].endswith("Z")

    def test_non_serialisable_extras_are_coerced(self, capsys):
        formatter = JsonFormatter(service="api")
        record = logging.LogRecord(
            name="x", level=logging.INFO, pathname=__file__, lineno=1, msg="m", args=(), exc_info=None
        )
        record.model = object()

        assert json.loads(formatter.format(record))["model"].startswith("<object")

    def test_configure_logging_installs_a_single_handler(self):
        configure_logging(service="api", level="DEBUG", log_format="json")

        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert root.level == logging.DEBUG

        configure_logging(service="api", level="INFO", log_format="json")
        assert len(logging.getLogger().handlers) == 1


# ===========================================================================
# settings
# ===========================================================================


class TestSettings:
    def test_rabbitmq_url_encodes_credentials(self, settings):
        configured = settings.model_copy(update={"rabbitmq_user": "user@corp", "rabbitmq_pass": "p/a ss"})

        assert configured.rabbitmq_url == "amqp://user%40corp:p%2Fa%20ss@localhost:5672/%2F"

    def test_mongo_uri_is_redacted_for_logging(self, settings):
        configured = settings.model_copy(update={"mongo_uri": "mongodb://user:secret@db:27017/app"})

        assert "secret" not in configured.mongo_uri_plain
        assert configured.mongo_uri_plain == "mongodb://***@db:27017/app"

    def test_invalid_log_level_is_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(log_level="LOUD")

    def test_positive_threshold_bounds_are_enforced(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(positive_threshold=1.5)
