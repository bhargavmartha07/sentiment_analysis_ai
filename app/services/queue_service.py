"""RabbitMQ transport: topology declaration, publishing and consuming.

The broker is the backbone of the asynchronous pipeline, so this module owns
every AMQP concern and the worker never touches ``pika`` directly:

* **Durable** exchange + queue, so a broker restart does not lose accepted jobs.
* **Publisher confirms** with ``mandatory=True`` - a job is only reported as
  accepted by the API once the broker has durably taken responsibility for it.
* **Dead-letter topology** - messages that exhaust their retries land in a DLQ
  for inspection instead of being silently dropped or looping forever.
* **Manual acknowledgements** - a job is acked only *after* the result is
  persisted in MongoDB, giving at-least-once delivery. The worker writes results
  with an upsert keyed on ``job_id``, making redelivery idempotent.
* **Automatic reconnection** with capped exponential backoff for both the
  publisher and the consumer.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.exceptions import ChannelError

from app.config import Settings, get_settings
from app.errors import QueueUnavailableError
from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SentimentJobMessage:
    """Wire format of a queued analysis job."""

    job_id: str
    text: str
    submitted_at: str
    attempt: int = 1
    source: str = "api"
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, job_id: str, text: str, *, source: str = "api") -> "SentimentJobMessage":
        return cls(
            job_id=job_id,
            text=text,
            submitted_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            source=source,
        )

    def to_json(self) -> bytes:
        return json.dumps(asdict(self), ensure_ascii=False).encode("utf-8")

    @classmethod
    def from_json(cls, raw: bytes | str) -> "SentimentJobMessage":
        """Parse a message body, raising ``ValueError`` on malformed input."""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Message body must be a JSON object.")
        missing = [key for key in ("job_id", "text") if not payload.get(key)]
        if missing:
            raise ValueError(f"Message body missing required field(s): {', '.join(missing)}.")
        return cls(
            job_id=str(payload["job_id"]),
            text=str(payload["text"]),
            submitted_at=str(payload.get("submitted_at") or datetime.now(timezone.utc).isoformat()),
            attempt=int(payload.get("attempt", 1)),
            source=str(payload.get("source", "api")),
            options=dict(payload.get("options") or {}),
        )


@dataclass(slots=True)
class Delivery:
    """A message handed to the worker by :meth:`QueueConsumer.messages`."""

    message: SentimentJobMessage
    delivery_tag: int
    redelivered: bool
    retry_count: int
    raw_body: bytes
    _channel: BlockingChannel
    _consumed_at: float = field(default_factory=time.monotonic)

    def requeue_for_retry(self, *, retry_count: int, exchange: str, routing_key: str) -> None:
        """Republish with an incremented retry header, then ack the original.

        ``basic_nack(requeue=True)`` redelivers the *same* envelope, so a counter
        carried in the headers would never advance and the job would loop
        forever. Publishing a fresh copy with ``x-retry-count`` bumped lets the
        retry budget actually advance towards the dead-letter queue. The original
        is only acked after the broker confirms the republish, so a failed
        publish leaves the job on the queue instead of dropping it.
        """
        headers = {"x-retry-count": retry_count, "x-retry-reason": "transient_failure"}
        confirmed = self._channel.basic_publish(
            exchange=exchange,
            routing_key=routing_key,
            body=self.raw_body,
            properties=pika.BasicProperties(
                content_type="application/json",
                delivery_mode=pika.DeliveryMode.Persistent,
                message_id=self.message.job_id,
                timestamp=int(time.time()),
                headers=headers,
            ),
            mandatory=True,
        )
        if confirmed is False:
            self._channel.basic_nack(delivery_tag=self.delivery_tag, requeue=True)
            raise ChannelError("retry republish was not confirmed by the broker")
        self._channel.basic_ack(delivery_tag=self.delivery_tag)
        logger.info(
            "Message requeued for retry",
            extra={
                "event": "queue.message.retry",
                "job_id": self.message.job_id,
                "retry_count": retry_count,
                "latency_ms": self._elapsed_ms(),
            },
        )

    def acknowledge(self) -> None:
        """Acknowledge the message (call only after durable persistence)."""
        self._channel.basic_ack(delivery_tag=self.delivery_tag)
        logger.debug(
            "Message acknowledged",
            extra={"event": "queue.message.ack", "job_id": self.message.job_id, "latency_ms": self._elapsed_ms()},
        )

    def reject(self, *, requeue: bool = False) -> None:
        """Reject the message, optionally requeueing it for another attempt."""
        self._channel.basic_nack(delivery_tag=self.delivery_tag, requeue=requeue)
        logger.warning(
            "Message rejected",
            extra={
                "event": "queue.message.nack",
                "job_id": self.message.job_id,
                "requeue": requeue,
                "retry_count": self.retry_count,
                "latency_ms": self._elapsed_ms(),
            },
        )

    def _elapsed_ms(self) -> int:
        return int((time.monotonic() - self._consumed_at) * 1000)


def build_connection_parameters(settings: Settings, *, heartbeat: int = 60) -> pika.ConnectionParameters:
    """Build AMQP connection parameters with sane resilience defaults."""
    return pika.ConnectionParameters(
        host=settings.rabbitmq_host,
        port=settings.rabbitmq_port,
        virtual_host=settings.rabbitmq_vhost,
        credentials=pika.PlainCredentials(
            settings.rabbitmq_user, settings._secret_value(settings.rabbitmq_pass)
        ),
        heartbeat=heartbeat,
        blocked_connection_timeout=settings.rabbitmq_publish_timeout_seconds,
        connection_attempts=1,
        retry_delay=1,
        socket_timeout=10,
        client_properties={"connection_name": f"{settings.service_name}-consumer"},
    )


def declare_topology(channel: BlockingChannel, settings: Settings) -> None:
    """Idempotently declare exchange, queue, DLX and DLQ, then bind.

    Safe to call from every process on every start-up: identical declarations
    are no-ops on the broker, which is what lets the API and the worker scale
    horizontally without coordination.
    """
    channel.exchange_declare(
        exchange=settings.rabbitmq_exchange,
        exchange_type="direct",
        durable=True,
        auto_delete=False,
    )
    channel.exchange_declare(
        exchange=settings.rabbitmq_dead_letter_exchange,
        exchange_type="direct",
        durable=True,
        auto_delete=False,
    )
    channel.queue_declare(
        queue=settings.rabbitmq_dead_letter_queue,
        durable=True,
        auto_delete=False,
    )
    channel.queue_bind(
        queue=settings.rabbitmq_dead_letter_queue,
        exchange=settings.rabbitmq_dead_letter_exchange,
        routing_key=settings.rabbitmq_routing_key,
    )
    channel.queue_declare(
        queue=settings.rabbitmq_queue,
        durable=True,
        auto_delete=False,
        arguments={
            "x-dead-letter-exchange": settings.rabbitmq_dead_letter_exchange,
            "x-dead-letter-routing-key": settings.rabbitmq_routing_key,
        },
    )
    channel.queue_bind(
        queue=settings.rabbitmq_queue,
        exchange=settings.rabbitmq_exchange,
        routing_key=settings.rabbitmq_routing_key,
    )


class QueuePublisher:
    """Thread-safe publisher for sentiment jobs.

    ``pika.BlockingConnection`` is not thread safe, so every channel interaction
    is serialised behind ``_lock``. The API calls :meth:`publish` from a
    threadpool (FastAPI is async), which keeps the event loop free while still
    allowing concurrent producers.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._connection: pika.BlockingConnection | None = None
        self._channel: BlockingChannel | None = None
        self._lock = threading.RLock()
        self._next_retry_at: float = 0.0
        self._published_count: int = 0

    # ------------------------------------------------------------- connection
    @property
    def is_connected(self) -> bool:
        return self._channel is not None and self._channel.is_open and not self._connection.is_closed

    def connect(self) -> None:
        """Open the channel and enable publisher confirms. Idempotent."""
        with self._lock:
            if self.is_connected:
                return
            now = time.monotonic()
            if now < self._next_retry_at:
                raise QueueUnavailableError(
                    "Message broker connection is in backoff after a recent failure.",
                    context={"retry_in_seconds": round(self._next_retry_at - now, 2)},
                )
            self._close_locked()
            try:
                self._connection = pika.BlockingConnection(
                    build_connection_parameters(self._settings)
                )
                self._channel = self._connection.channel()
                declare_topology(self._channel, self._settings)
                self._channel.confirm_delivery()
            except (pika.exceptions.AMQPError, OSError) as exc:
                self._close_locked()
                self._next_retry_at = now + self._settings.rabbitmq_connection_retry_seconds
                logger.error(
                    "Failed to connect to RabbitMQ",
                    extra={"event": "queue.connect.failed", "error": str(exc), "host": self._settings.rabbitmq_host},
                )
                raise QueueUnavailableError(
                    f"Unable to connect to the message broker: {exc}",
                    context={"rabbitmq_host": self._settings.rabbitmq_host},
                ) from exc

            logger.info(
                "Connected to RabbitMQ publisher",
                extra={
                    "event": "queue.connect.success",
                    "exchange": self._settings.rabbitmq_exchange,
                    "queue": self._settings.rabbitmq_queue,
                },
            )

    def _close_locked(self) -> None:
        for resource in (self._channel, self._connection):
            try:
                if resource is not None and getattr(resource, "is_open", False):
                    resource.close()
            except Exception:  # pragma: no cover - close is best effort
                logger.debug("Error while closing RabbitMQ resource", extra={"event": "queue.close.error"})
        self._channel = None
        self._connection = None

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def keepalive(self) -> bool:
        """Service AMQP heartbeats on an otherwise idle connection.

        ``pika.BlockingConnection`` only processes heartbeats while it is doing
        I/O, so a quiet API loses the broker's heartbeat timeout and the socket
        is reaped underneath us. The next ``/async`` call reconnects fine, but
        readiness would report a false 503 in the meantime. Returns the current
        connectivity without raising.
        """
        with self._lock:
            if not self.is_connected:
                return False
            try:
                self._connection.process_data_events(time_limit=0)
            except (pika.exceptions.AMQPError, OSError) as exc:
                logger.warning(
                    "Publisher keepalive failed; connection will be re-established on demand",
                    extra={"event": "queue.keepalive.failed", "error": str(exc)},
                )
                self._close_locked()
                return False
            return True

    def ensure_connected(self) -> bool:
        """Reconnect if needed and report whether publishing is possible."""
        try:
            self.connect()
        except QueueUnavailableError as exc:
            logger.warning(
                "Publisher unavailable during readiness probe",
                extra={"event": "queue.readiness.failed", "error": str(exc)},
            )
            return False
        return self.is_connected

    # ---------------------------------------------------------------- publish
    def publish(self, message: SentimentJobMessage) -> None:
        """Publish a job durably. Raises :class:`QueueUnavailableError` on failure."""
        with self._lock:
            try:
                self.connect()
                assert self._channel is not None
                self._channel.basic_publish(
                    exchange=self._settings.rabbitmq_exchange,
                    routing_key=self._settings.rabbitmq_routing_key,
                    body=message.to_json(),
                    properties=pika.BasicProperties(
                        content_type="application/json",
                        content_encoding="utf-8",
                        delivery_mode=pika.DeliveryMode.Persistent,
                        message_id=message.job_id,
                        timestamp=int(time.time()),
                        headers={"x-retry-count": 0, "x-source": message.source},
                    ),
                    mandatory=True,
                )
                self._published_count += 1
                self._next_retry_at = 0.0
                logger.info(
                    "Job published",
                    extra={
                        "event": "queue.publish.success",
                        "job_id": message.job_id,
                        "routing_key": self._settings.rabbitmq_routing_key,
                        "published_total": self._published_count,
                    },
                )
            except QueueUnavailableError:
                raise
            except (pika.exceptions.AMQPError, pika.exceptions.UnroutableError, OSError) as exc:
                self._close_locked()
                self._next_retry_at = time.monotonic() + self._settings.rabbitmq_connection_retry_seconds
                logger.error(
                    "Failed to publish job",
                    extra={"event": "queue.publish.failed", "job_id": message.job_id, "error": str(exc)},
                )
                raise QueueUnavailableError(
                    f"Could not enqueue the analysis job: {exc}",
                    context={"job_id": message.job_id},
                ) from exc

    def stats(self) -> dict[str, Any]:
        return {"connected": self.is_connected, "published_total": self._published_count}


class QueueConsumer:
    """Blocking consumer that survives broker outages.

    :meth:`messages` is a generator that yields :class:`Delivery` objects and
    transparently rebuilds the connection with capped exponential backoff when
    the broker goes away. Callbacks are invoked while the channel is open, so
    the caller can acknowledge immediately.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._connection: pika.BlockingConnection | None = None
        self._channel: BlockingChannel | None = None

    @property
    def is_connected(self) -> bool:
        return self._channel is not None and self._channel.is_open and not self._connection.is_closed

    def connect(self) -> BlockingChannel:
        """Open a fresh channel, declare topology and apply QoS."""
        self.close()
        connection = pika.BlockingConnection(build_connection_parameters(self._settings))
        channel = connection.channel()
        declare_topology(channel, self._settings)
        channel.basic_qos(prefetch_count=self._settings.rabbitmq_prefetch_count)
        self._connection, self._channel = connection, channel
        logger.info(
            "Connected to RabbitMQ as consumer",
            extra={
                "event": "queue.consumer.connect",
                "queue": self._settings.rabbitmq_queue,
                "prefetch_count": self._settings.rabbitmq_prefetch_count,
            },
        )
        return channel

    def close(self) -> None:
        for resource in (self._channel, self._connection):
            try:
                if resource is not None and getattr(resource, "is_open", False):
                    resource.close()
            except Exception:  # pragma: no cover
                pass
        self._channel = None
        self._connection = None

    def messages(self, stop_event: threading.Event) -> Iterator[Delivery]:
        """Yield deliveries until ``stop_event`` is set or the broker drops."""
        backoff = self._settings.rabbitmq_connection_retry_seconds
        while not stop_event.is_set():
            try:
                channel = self.connect()
                backoff = self._settings.rabbitmq_connection_retry_seconds  # reset after success

                def on_message(
                    ch: BlockingChannel,
                    method: pika.spec.Basic.Deliver,
                    properties: pika.BasicProperties,
                    body: bytes,
                ) -> None:
                    retry_count = int((properties.headers or {}).get("x-retry-count", 0)) if properties.headers else 0
                    try:
                        job = SentimentJobMessage.from_json(body)
                    except (ValueError, UnicodeDecodeError):
                        logger.error(
                            "Discarding unparseable message",
                            extra={"event": "queue.message.malformed", "delivery_tag": method.delivery_tag},
                        )
                        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                        return
                    # Park the delivery and let the consumer loop hand it over.
                    self._inbox.append(
                        Delivery(
                            message=job,
                            delivery_tag=method.delivery_tag,
                            redelivered=bool(method.redelivered),
                            retry_count=retry_count,
                            raw_body=body,
                            _channel=ch,
                        )
                    )

                self._inbox: list[Delivery] = []
                channel.basic_consume(
                    queue=self._settings.rabbitmq_queue,
                    on_message_callback=on_message,
                    auto_ack=False,
                )
                logger.info(
                    "Worker waiting for messages",
                    extra={"event": "queue.consumer.listening", "queue": self._settings.rabbitmq_queue},
                )

                while not stop_event.is_set() and self.is_connected:
                    if self._inbox:
                        yield self._inbox.pop(0)
                        continue
                    connection_heartbeat = 0
                    if stop_event.wait(0.1):
                        break
                    # process_data_events drives heartbeats and IO callbacks.
                    self._connection.process_data_events(time_limit=0.5)  # type: ignore[union-attr]
                    connection_heartbeat += 1
                    if connection_heartbeat > 1200:  # ~10 minutes
                        logger.info("Connection healthy, still waiting", extra={"event": "queue.consumer.heartbeat"})
                        connection_heartbeat = 0

                if stop_event.is_set():
                    logger.info("Consumer stopping on shutdown signal", extra={"event": "queue.consumer.stopping"})
                    return
            except (pika.exceptions.AMQPError, OSError) as exc:
                logger.error(
                    "Consumer connection lost; reconnecting",
                    extra={
                        "event": "queue.consumer.reconnecting",
                        "error": str(exc),
                        "backoff_seconds": round(backoff, 1),
                    },
                )
                self.close()
                if stop_event.wait(backoff):
                    return
                backoff = min(backoff * 2, self._settings.rabbitmq_connection_retry_max_seconds)
            except Exception:  # pragma: no cover - unexpected, must not kill the worker
                logger.critical(
                    "Unexpected error in consumer loop",
                    extra={"event": "queue.consumer.crashed"},
                    exc_info=True,
                )
                self.close()
                if stop_event.wait(backoff):
                    return
                backoff = min(backoff * 2, self._settings.rabbitmq_connection_retry_max_seconds)

        self.close()


# --------------------------------------------------------------------- singleton
_publisher: QueuePublisher | None = None
_publisher_lock = threading.Lock()


def get_publisher() -> QueuePublisher:
    """Return the process-wide publisher singleton used by the API."""
    global _publisher
    if _publisher is None:
        with _publisher_lock:
            if _publisher is None:
                _publisher = QueuePublisher()
    return _publisher


def reset_publisher() -> None:
    """Drop the publisher singleton (tests)."""
    global _publisher
    with _publisher_lock:
        if _publisher is not None:
            _publisher.close()
        _publisher = None


def new_job_id() -> str:
    """Generate the job identifier used end-to-end (API -> queue -> Mongo -> API)."""
    return str(uuid.uuid4())


PublishCallback = Callable[[SentimentJobMessage], None]
