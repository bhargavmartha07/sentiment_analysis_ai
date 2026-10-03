"""Queue worker: consumes sentiment jobs, runs inference, persists results.

Delivery contract
-----------------
* Messages are acknowledged **only after** the result is durably written to
  MongoDB, giving at-least-once processing.
* Because the write is an upsert keyed on ``job_id``, redelivery is idempotent -
  a crash between write and ack cannot corrupt the record.
* Failures are classified:

  - **permanent** (malformed payload, invalid job id) -> ``nack(requeue=False)``,
    which routes the message to the dead-letter queue for inspection.
  - **transient** (MongoDB or broker hiccup) -> requeue with an incremented
    ``x-retry-count`` header, up to ``WORKER_MAX_RETRIES``, then dead-letter.

Health
------
The worker is a batch process, so it cannot answer HTTP. Instead it writes a
heartbeat file (``WORKER_HEARTBEAT_PATH``) after every processed message and at
least every few seconds while idle; ``--healthcheck`` exits 0 only if that file
is fresh. Docker then reports the worker as healthy/unhealthy.

Usage::

    python -m app.worker.consumer            # run the worker
    python -m app.worker.consumer --healthcheck
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.errors import PersistenceError, ServiceError
from app.logging_config import configure_logging, get_logger
from app.services.db_service import SentimentRepository, get_repository
from app.services.model_service import SentimentModelService, get_model_service
from app.services.queue_service import Delivery, QueueConsumer

logger = get_logger(__name__, service="worker")

WORKER_ID = f"worker-{uuid.uuid4().hex[:8]}"
_HEARTBEAT_INTERVAL_SECONDS = 5.0


class SentimentWorker:
    """Owns the consume -> infer -> persist -> acknowledge loop."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model: SentimentModelService | None = None,
        repository: SentimentRepository | None = None,
        consumer: QueueConsumer | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._model = model or get_model_service()
        self._repository = repository or get_repository()
        self._consumer = consumer or QueueConsumer(self._settings)
        self._stop_event = threading.Event()
        self._processed = 0
        self._failed = 0
        self._dead_lettered = 0

    # ------------------------------------------------------------- heartbeat
    def _heartbeat_file(self) -> Path:
        return Path(self._settings.worker_heartbeat_path)

    def _write_heartbeat(self, state: str) -> None:
        payload = {
            "worker_id": WORKER_ID,
            "state": state,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "processed_total": self._processed,
            "failed_total": self._failed,
            "dead_lettered_total": self._dead_lettered,
            "queue": self._settings.rabbitmq_queue,
        }
        path = self._heartbeat_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a probe never observes a half-written file.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)

    def healthcheck(self) -> bool:
        """Exit-code helper used by the Docker HEALTHCHECK instruction."""
        path = self._heartbeat_file()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            age = time.time() - path.stat().st_mtime
        except (OSError, json.JSONDecodeError):
            logger.warning(
                "Worker heartbeat missing or unreadable",
                extra={"event": "worker.healthcheck.no_heartbeat", "path": str(path)},
            )
            return False

        max_age = self._settings.worker_heartbeat_stale_seconds
        healthy = age <= max_age
        logger.info(
            "Worker healthcheck evaluated",
            extra={
                "event": "worker.healthcheck",
                "healthy": healthy,
                "age_seconds": round(age, 1),
                "max_age_seconds": max_age,
                "worker_id": payload.get("worker_id"),
                "state": payload.get("state"),
                "processed_total": payload.get("processed_total"),
            },
        )
        return healthy

    # ------------------------------------------------------------ processing
    def _process(self, delivery: Delivery) -> None:
        job = delivery.message
        started = time.perf_counter()
        logger.info(
            "Processing job",
            extra={
                "event": "worker.job.started",
                "job_id": job.job_id,
                "attempt": delivery.retry_count + 1,
                "redelivered": delivery.redelivered,
                "text_length": len(job.text),
            },
        )

        try:
            model_card = self._model.metadata
            prediction = self._model.predict(job.text)
            self._repository.save_result(
                job_id=job.job_id,
                text=job.text,
                prediction=prediction,
                model_card=model_card,
                latency_ms=int((time.perf_counter() - started) * 1000),
                attempt=delivery.retry_count + 1,
            )
        except (PersistenceError, ServiceError) as exc:
            self._failed += 1
            self._handle_transient_failure(delivery, exc)
            return
        except Exception as exc:
            self._failed += 1
            logger.exception(
                "Unexpected error while processing job",
                extra={
                    "event": "worker.job.error",
                    "job_id": job.job_id,
                    "exception_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            self._handle_transient_failure(delivery, exc, unexpected=True)
            return

        delivery.acknowledge()
        self._processed += 1
        logger.info(
            "Job completed",
            extra={
                "event": "worker.job.completed",
                "job_id": job.job_id,
                "sentiment": prediction.sentiment,
                "score": prediction.score,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "processed_total": self._processed,
            },
        )

    def _handle_transient_failure(self, delivery: Delivery, exc: BaseException, *, unexpected: bool = False) -> None:
        """Requeue with an incremented retry count, or dead-letter when exhausted."""
        attempts = delivery.retry_count + 1
        if attempts < self._settings.worker_max_retries:
            delay = min(2 ** attempts, 30)
            logger.warning(
                "Job failed; scheduling retry",
                extra={
                    "event": "worker.job.retry",
                    "job_id": delivery.message.job_id,
                    "attempt": attempts,
                    "max_retries": self._settings.worker_max_retries,
                    "retry_in_seconds": delay,
                    "error": str(exc),
                    "unexpected": unexpected,
                },
            )
            delivery.requeue_for_retry(
                retry_count=attempts,
                exchange=self._settings.rabbitmq_exchange,
                routing_key=self._settings.rabbitmq_routing_key,
            )
            time.sleep(delay)
            return

        self._dead_lettered += 1
        logger.error(
            "Job exhausted retries; dead-lettering",
            extra={
                "event": "worker.job.dead_lettered",
                "job_id": delivery.message.job_id,
                "attempts": attempts,
                "dead_letter_queue": self._settings.rabbitmq_dead_letter_queue,
                "error": str(exc),
            },
        )
        delivery.reject(requeue=False)

    # ------------------------------------------------------------------ run
    def request_stop(self, signum: int | None = None, _frame: Any = None) -> None:
        """Signal handler: stop consuming and drain in-flight work."""
        if not self._stop_event.is_set():
            logger.info(
                "Shutdown signal received",
                extra={"event": "worker.signal", "signal": int(signum) if signum else None},
            )
        self._stop_event.set()

    def run(self) -> int:
        """Consume until stopped. Returns a process exit code."""
        self._install_signal_handlers()
        logger.info(
            "Sentiment worker starting",
            extra={
                "event": "worker.start",
                "worker_id": WORKER_ID,
                "queue": self._settings.rabbitmq_queue,
                "prefetch_count": self._settings.rabbitmq_prefetch_count,
                "max_retries": self._settings.worker_max_retries,
                "mongo_uri": self._settings.mongo_uri_plain,
                "model": self._settings.model_path,
            },
        )

        try:
            self._model.load()
            self._repository.ensure_indexes()
        except Exception:
            logger.critical("Worker initialisation failed", extra={"event": "worker.init.failed"}, exc_info=True)
            return 1

        heartbeat_stop = threading.Event()
        heartbeater = threading.Thread(
            target=self._heartbeat_loop,
            args=(heartbeat_stop,),
            name="heartbeat",
            daemon=True,
        )
        heartbeater.start()

        self._write_heartbeat("starting")
        try:
            for delivery in self._consumer.messages(self._stop_event):
                self._write_heartbeat("processing")
                self._process(delivery)
                self._write_heartbeat("idle")
        except KeyboardInterrupt:  # pragma: no cover - interactive path
            self.request_stop(signal.SIGINT)
        finally:
            heartbeat_stop.set()
            self._consumer.close()
            self._write_heartbeat("stopped")
            logger.info(
                "Sentiment worker stopped",
                extra={
                    "event": "worker.stop",
                    "worker_id": WORKER_ID,
                    "processed_total": self._processed,
                    "failed_total": self._failed,
                    "dead_lettered_total": self._dead_lettered,
                },
            )
        return 0

    def _heartbeat_loop(self, stop_event: threading.Event) -> None:
        """Keep the heartbeat fresh while idle so Docker sees a live worker."""
        while not stop_event.is_set():
            try:
                self._write_heartbeat("idle")
            except OSError:  # pragma: no cover - filesystem issue
                logger.warning("Could not write heartbeat", extra={"event": "worker.heartbeat.failed"})
            stop_event.wait(_HEARTBEAT_INTERVAL_SECONDS)

    def _install_signal_handlers(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, self.request_stop)
            except ValueError:  # pragma: no cover - not on the main thread
                logger.debug(
                    "Could not install signal handler",
                    extra={"event": "worker.signal_handler.failed", "signal": int(sig)},
                )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RabbitMQ consumer for the sentiment analysis service")
    parser.add_argument(
        "--healthcheck",
        action="store_true",
        help="Exit 0 if the worker heartbeat is fresh, 1 otherwise. Used by Docker HEALTHCHECK.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(service="worker", level=settings.log_level, log_format=settings.log_format)
    worker = SentimentWorker(settings)
    if args.healthcheck:
        return 0 if worker.healthcheck() else 1
    return worker.run()


if __name__ == "__main__":
    sys.exit(main())
