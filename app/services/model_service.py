"""TensorFlow Keras model loading and inference.

Design notes
------------
* The model is loaded **once** per process (guarded by a re-entrant lock) and
  served from memory for the lifetime of the API or worker container. Loading a
  Keras model per request is the single most common cause of high p95 latency in
  naive inference services.
* Text encoding is implemented in pure Python/Numpy against a ``word_index.json``
  vocabulary rather than relying on a pickled ``keras.preprocessing.Tokenizer``.
  Pickled Keras tokenizers couple the artifact to the exact Keras version that
  created it - a portability trap that silently breaks inference after a library
  upgrade. A plain JSON vocabulary is version independent and diff-able in git.
* TensorFlow is imported lazily inside :meth:`SentimentModelService.load` so unit
  tests (and CI linting) can exercise the encoding logic without the ~600 MB
  dependency, and so the thread-count environment variables can be applied
  before TensorFlow initialises its pools.
* A warm-up forward pass runs during load. TensorFlow initialises CUDA/CPU
  kernels lazily on the first call, which would otherwise put a multi-second
  spike on the very first production request.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from app.config import Settings, get_settings
from app.errors import ModelUnavailableError
from app.logging_config import get_logger

logger = get_logger(__name__)

# Anything that is not a letter or digit separates tokens. Keeping apostrophes
# ("don't", "it's") means negations stay a single token.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_DEFAULT_MAX_LEN = 200
# Reserved rows of the embedding table, matching keras.datasets.imdb:
# 0 = padding, 1 = start-of-sequence, 2 = out-of-vocabulary. Word ids start at 3
# (see scripts/train_model.py, which persists the shifted vocabulary).
_PAD_ID = 0
_START_ID = 1
_OOV_ID = 2


@dataclass(frozen=True, slots=True)
class Prediction:
    """A single inference result."""

    score: float
    sentiment: str

    @property
    def is_positive(self) -> bool:
        return self.sentiment == "positive"


class SentimentModelService:
    """Thread-safe façade over the pre-trained Keras sentiment classifier."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        batch_size: int = 32,
    ) -> None:
        self._settings = settings or get_settings()
        self._batch_size = batch_size
        self._model: Any | None = None
        self._word_index: dict[str, int] = {}
        self._metadata: dict[str, Any] = {}
        self._max_len = _DEFAULT_MAX_LEN
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ state
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def metadata(self) -> dict[str, Any]:
        """Model card: architecture, dataset and evaluation metrics."""
        return dict(self._metadata)

    @property
    def max_len(self) -> int:
        return self._max_len

    # ------------------------------------------------------------------ setup
    def load(self) -> None:
        """Load the .h5 model and vocabulary. Idempotent and thread safe."""
        with self._lock:
            if self._model is not None:
                return

            model_path = Path(self._settings.model_path)
            word_index_path = Path(self._settings.word_index_path)
            if not model_path.is_file():
                raise ModelUnavailableError(
                    f"Sentiment model not found at '{model_path}'. "
                    "Run 'python scripts/train_model.py' or mount the artifact into the container.",
                    context={"model_path": str(model_path)},
                )
            if not word_index_path.is_file():
                raise ModelUnavailableError(
                    f"Vocabulary not found at '{word_index_path}'. "
                    "The model artifact must ship together with its word_index.json.",
                    context={"word_index_path": str(word_index_path)},
                )

            self._configure_tensorflow_runtime()

            try:
                import tensorflow as tf  # noqa: PLC0415 - intentional lazy import
            except ImportError as exc:  # pragma: no cover - dependency is pinned in requirements
                raise ModelUnavailableError(f"TensorFlow is not installed: {exc}") from exc

            self._word_index = self._load_vocabulary(word_index_path)
            self._metadata = self._load_metadata()

            logger.info(
                "Loading Keras sentiment model",
                extra={
                    "event": "model.load.start",
                    "model_path": str(model_path),
                    "vocabulary_size": len(self._word_index),
                    "tensorflow_version": tf.__version__,
                },
            )

            model = tf.keras.models.load_model(model_path, compile=False)

            self._max_len = self._resolve_max_len(model)
            if self._settings.tf_enable_gpu:
                device = self._select_device(tf)
                model = model.to(device)
                logger.info("Model placed on device", extra={"event": "model.device", "device": str(device)})

            self._model = model
            self._warm_up()

            logger.info(
                "Sentiment model ready",
                extra={
                    "event": "model.load.success",
                    "max_len": self._max_len,
                    "parameters": int(getattr(model, "count_params", lambda: 0)()),
                    "architecture": self._metadata.get("architecture"),
                    "test_accuracy": (self._metadata.get("metrics") or {}).get("test_accuracy"),
                },
            )

    def _configure_tensorflow_runtime(self) -> None:
        """Cap TensorFlow's thread pools. Must run before the first TF import."""
        settings = self._settings
        if not settings.tf_enable_gpu:
            # Hide GPU devices entirely unless explicitly enabled: on CPU-only hosts
            # TF still probes CUDA and adds ~2s to start-up plus noisy warnings.
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        if settings.tf_intra_op_threads:
            os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(settings.tf_intra_op_threads))
        if settings.tf_inter_op_threads:
            os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(settings.tf_inter_op_threads))
        # Keep logs from third-party C++ layers out of our structured stream.
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("AUTOGRAPH_VERBOSITY", "0")

    @staticmethod
    def _select_device(tf: Any) -> Any:
        gpus = tf.config.list_physical_devices("GPU")
        return gpus[0] if gpus else "cpu"

    def _load_vocabulary(self, path: Path) -> dict[str, int]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        # Accept either a {"word": id} mapping or a list ordered by id.
        if isinstance(raw, list):
            raw = {word: index for index, word in enumerate(raw)}
        # Ids 0/1/2 are reserved (pad/start/oov), so real words start at 3.
        return {
            str(word): int(index)
            for word, index in raw.items()
            if int(index) >= _OOV_ID + 1
        }

    def _load_metadata(self) -> dict[str, Any]:
        path = Path(self._settings.model_metadata_path)
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning(
                "Model metadata is not valid JSON and will be ignored",
                extra={"event": "model.metadata.invalid", "path": str(path)},
            )
            return {}

    def _resolve_max_len(self, model: Any) -> int:
        """Prefer the model card, then the model's own input shape."""
        declared = self._metadata.get("max_len")
        if isinstance(declared, int) and declared > 0:
            return declared
        shape = getattr(model, "input_shape", None)
        if shape and isinstance(shape, tuple) and shape[-1]:
            return int(shape[-1])
        return _DEFAULT_MAX_LEN

    def _warm_up(self) -> None:
        """Force lazy TensorFlow initialisation before serving traffic."""
        assert self._model is not None
        sample = "The quick brown fox jumps over the lazy dog and it was wonderful."
        try:
            self.predict_batch([sample])
        except Exception:  # pragma: no cover - warm-up must never block start-up
            logger.warning(
                "Model warm-up inference failed; continuing to serve",
                extra={"event": "model.warmup.failed", "error": "see exception log"},
                exc_info=True,
            )

    # -------------------------------------------------------------- inference
    def tokenize(self, text: str) -> list[int]:
        """Encode one string into fixed-length token ids (PAD filled, post-truncated)."""
        tokens = _TOKEN_RE.findall(text.lower())
        limit = self._max_len
        encoded = [self._word_index.get(token, _OOV_ID) for token in tokens[:limit]]
        return encoded + [_PAD_ID] * (limit - len(encoded))

    def tokenize_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Encode several strings into a ``(len(texts), max_len)`` int32 matrix."""
        matrix = np.zeros((len(texts), self._max_len), dtype=np.int32)
        for row, text in enumerate(texts):
            encoded = self.tokenize(text)
            matrix[row, : len(encoded)] = encoded
        return matrix

    def predict_batch(self, texts: Sequence[str]) -> list[Prediction]:
        """Run inference over one or more texts in a single forward pass."""
        if not texts:
            return []
        with self._lock:
            model = self._require_model()
            batch = self.tokenize_batch(texts)
            probabilities = model.predict(
                batch,
                batch_size=min(self._batch_size, len(texts)),
                verbose=0,
            )
        scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if scores.shape[0] != len(texts):  # pragma: no cover - defensive
            raise ModelUnavailableError(
                f"Model returned {scores.shape[0]} scores for {len(texts)} inputs."
            )
        return [self._to_prediction(float(score)) for score in scores]

    def predict(self, text: str) -> Prediction:
        """Convenience wrapper for a single text."""
        return self.predict_batch([text])[0]

    def _to_prediction(self, raw_score: float) -> Prediction:
        score = round(min(max(raw_score, 0.0), 1.0), self._settings.score_precision)
        threshold = self._settings.positive_threshold
        return Prediction(score=score, sentiment="positive" if raw_score >= threshold else "negative")

    def _require_model(self) -> Any:
        if self._model is None:
            raise ModelUnavailableError(
                "Sentiment model is not loaded. The service is still starting up or failed to initialise."
            )
        return self._model

    def stats(self) -> dict[str, Any]:
        """Operational counters surfaced on the readiness endpoint."""
        return {
            "model_loaded": self.is_loaded,
            "max_len": self._max_len,
            "vocabulary_size": len(self._word_index),
            "architecture": self._metadata.get("architecture"),
            "dataset": self._metadata.get("dataset"),
            "test_accuracy": (self._metadata.get("metrics") or {}).get("test_accuracy"),
        }


# --------------------------------------------------------------------- module singleton
_model_service: SentimentModelService | None = None
_singleton_lock = threading.Lock()


def get_model_service() -> SentimentModelService:
    """Return the process-wide model service singleton."""
    global _model_service
    if _model_service is None:
        with _singleton_lock:
            if _model_service is None:
                _model_service = SentimentModelService()
                logger.debug("Created model service singleton", extra={"event": "model.singleton.created"})
    return _model_service


def reset_model_service() -> None:
    """Drop the singleton. Used by tests to isolate state."""
    global _model_service
    with _singleton_lock:
        _model_service = None


def iter_batched(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    """Yield ``size``-length slices of ``items``. Kept for worker batching."""
    for start in range(0, len(items), size):
        yield items[start : start + size]
