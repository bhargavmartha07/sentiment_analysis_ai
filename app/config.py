"""Centralised, environment-driven configuration.

Every deployment-specific value (credentials, hosts, ports, model location)
lives here and is read from environment variables, with ``.env`` support for
local development. Nothing else in the codebase reads ``os.environ`` directly,
which keeps configuration auditable and makes the service trivially
12-factor compliant.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogFormat = Literal["json", "text"]

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Runtime settings for both the API and the worker process."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ service
    service_name: str = Field(default="sentiment-microservice", description="Service identifier used in logs and health payloads.")
    environment: str = Field(default="development", description="Deployment environment name (development/test/production).")
    version: str = Field(default="1.0.0", description="Application version reported by the health and root endpoints.")

    # --------------------------------------------------------------- observability
    log_level: str = Field(default="INFO", description="Python logging level name.")
    log_format: LogFormat = Field(default="json", description="Structured JSON logs (production) or human readable text (local dev).")
    log_request_body: bool = Field(default=False, description="Log incoming request bodies. Keep off in production - user text may be sensitive.")

    # ----------------------------------------------------------------- http api
    api_host: str = Field(default="0.0.0.0", description="Interface the API binds to.")
    api_port: int = Field(default=8000, description="Port the API binds to.")
    api_workers: int = Field(default=1, description="Uvicorn worker processes. >1 multiplies model memory.")
    sync_max_text_length: int = Field(default=10_000, description="Maximum accepted characters per request text.")
    sync_batch_max_items: int = Field(default=32, description="Maximum items accepted by the batch endpoint.")

    # --------------------------------------------------------------- rabbitmq
    rabbitmq_host: str = Field(default="localhost", description="RabbitMQ hostname.")
    rabbitmq_port: int = Field(default=5672, description="RabbitMQ AMQP port.")
    rabbitmq_user: str = Field(default="guest", description="RabbitMQ username.")
    rabbitmq_pass: SecretStr = Field(default=SecretStr("guest"), description="RabbitMQ password.")
    rabbitmq_vhost: str = Field(default="/", description="RabbitMQ virtual host.")
    rabbitmq_exchange: str = Field(default="sentiment.exchange", description="Durable direct exchange used for sentiment jobs.")
    rabbitmq_queue: str = Field(default="sentiment.jobs", description="Durable work queue bound to the exchange.")
    rabbitmq_routing_key: str = Field(default="sentiment.analyze", description="Routing key used when publishing jobs.")
    rabbitmq_dead_letter_exchange: str = Field(default="sentiment.dlx", description="Dead letter exchange for poison messages.")
    rabbitmq_dead_letter_queue: str = Field(default="sentiment.jobs.dlq", description="Queue that collects messages that exhausted retries.")
    rabbitmq_prefetch_count: int = Field(default=8, description="Unacknowledged messages allowed per worker channel.")
    rabbitmq_publish_timeout_seconds: float = Field(default=5.0, description="Timeout for confirm_delivery() publisher confirms.")
    rabbitmq_connection_retry_seconds: float = Field(default=5.0, description="Base delay for the consumer's exponential reconnect backoff.")
    rabbitmq_connection_retry_max_seconds: float = Field(default=60.0, description="Ceiling for the consumer's reconnect backoff.")
    worker_max_retries: int = Field(default=3, description="Delivery attempts before a message is dead lettered.")
    worker_heartbeat_path: str = Field(default="/tmp/worker-heartbeat.json", description="File the worker touches so Docker can health-check it.")
    worker_heartbeat_stale_seconds: float = Field(default=60.0, description="Heartbeat age after which the worker is considered unhealthy.")

    # ----------------------------------------------------------------- mongodb
    mongo_uri: SecretStr = Field(default=SecretStr("mongodb://localhost:27017"), description="MongoDB connection string.")
    mongo_db_name: str = Field(default="sentiment_db", description="Database name.")
    mongo_collection: str = Field(default="sentiment_results", description="Collection holding sentiment analysis results.")
    mongo_server_selection_timeout_ms: int = Field(default=3000, description="Server selection timeout for the MongoDB driver.")
    mongo_connect_retry_seconds: float = Field(default=5.0, description="Delay between MongoDB reconnection attempts.")

    # -------------------------------------------------------------------- model
    model_path: str = Field(default="app/models/sentiment_model.h5", description="Path to the pre-trained Keras .h5 model.")
    word_index_path: str = Field(default="app/models/word_index.json", description="Path to the vocabulary used to encode text.")
    model_metadata_path: str = Field(default="app/models/model_metadata.json", description="Optional model card with training metrics.")
    positive_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Score at or above which a sample is labelled positive.")
    score_precision: int = Field(default=6, ge=0, le=12, description="Decimal places kept for the confidence score.")

    # ---------------------------------------------------------------- tf tuning
    tf_intra_op_threads: int = Field(default=0, ge=0, description="TF intra-op threads. 0 keeps the TensorFlow default.")
    tf_inter_op_threads: int = Field(default=0, ge=0, description="TF inter-op threads. 0 keeps the TensorFlow default.")
    tf_enable_gpu: bool = Field(default=False, description="Attempt to place the model on GPU when available.")

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        level = value.strip().upper()
        if level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}:
            raise ValueError(f"Unsupported LOG_LEVEL '{value}'")
        return level

    def _secret_value(self, secret: SecretStr | str) -> str:
        """Read a secret that may be a SecretStr or a plain str.

        Constructing ``Settings`` via ``model_copy(update=...)`` skips
        validation and can leave a plain string in place of a SecretStr, so every
        read goes through this helper instead of ``get_secret_value()`` directly.
        """
        return secret.get_secret_value() if isinstance(secret, SecretStr) else str(secret)

    @property
    def rabbitmq_url(self) -> str:
        """AMQP URL for publisher and consumer connections."""
        from urllib.parse import quote

        user = quote(self.rabbitmq_user, safe="")
        password = quote(self._secret_value(self.rabbitmq_pass), safe="")
        vhost = quote(self.rabbitmq_vhost, safe="")
        return f"amqp://{user}:{password}@{self.rabbitmq_host}:{self.rabbitmq_port}/{vhost}"

    @property
    def mongo_uri_plain(self) -> str:
        """Connection string safe to log (credentials redacted)."""
        raw = self._secret_value(self.mongo_uri)
        if "@" in raw:
            scheme, _, host = raw.rpartition("@")
            return f"{scheme.split('://', 1)[0]}://***@{host}"
        return raw


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
