"""Train the pre-trained Keras sentiment model shipped with this service.

This script is NOT part of the serving path. It is run once (or re-run to refresh
the artifact) to produce:

    app/models/sentiment_model.h5   - Keras Sequential model, sigmoid output
    app/models/word_index.json      - word -> integer id map used at inference
    app/models/model_metadata.json  - training metrics + provenance

Dataset: IMDB Large Movie Review Dataset (25k train / 25k test, balanced binary
sentiment) via ``keras.datasets.imdb``.

Usage (locally):
    python scripts/train_model.py --output-dir app/models --epochs 3

Usage (inside Docker, recommended so no local TF install is needed):
    docker run --rm -v "$PWD:/work" -w /work python:3.11-slim \
        bash -c "pip install -q -r requirements.txt && python scripts/train_model.py"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

MAX_LEN = 200
NUM_WORDS = 20000
EMBEDDING_DIM = 32
# keras.datasets.imdb reserves the first three embedding rows for
# pad/start/oov and shifts every word id by INDEX_FROM, so the vocabulary we
# persist for inference must be shifted the same way (see build_vocabulary).
PAD_ID = 0
START_ID = 1
OOV_ID = 2
INDEX_FROM = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="app/models", help="Where to write artifacts")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("TRAIN_EPOCHS", "3")))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_vocabulary(word_index: dict[str, int]) -> dict[str, int]:
    """Shift keras word ids into the embedding table used for training.

    ``keras.datasets.imdb.get_word_index()`` returns raw ids that still start at
    0, while ``load_data`` reserves rows 0/1/2 for pad/start/oov and emits every
    word at ``raw_id + 3``. Serving must therefore persist the *shifted* ids,
    otherwise every input reaches the model three rows too low and predictions
    become meaningless (this silently shipped once already).
    """
    return {
        word: idx + INDEX_FROM
        for word, idx in word_index.items()
        if PAD_ID <= idx + INDEX_FROM < NUM_WORDS
    }


def verify_serving_path(
    vocabulary: dict[str, int], x_test: Any, y_test: Any, samples: int = 250
) -> float:
    """Score the real inference path and return accuracy against true labels.

    This exercises decode -> ``tokenize`` -> ``predict_batch`` exactly as the API
    does, so a vocabulary/model mismatch fails the training run instead of
    quietly shipping an inverted classifier.
    """
    from app.services.model_service import SentimentModelService

    inverse = {idx: word for word, idx in vocabulary.items()}
    texts, labels = [], []
    for sequence, label in zip(x_test[:samples], y_test[:samples]):
        texts.append(" ".join(inverse.get(token, "") for token in sequence if token > OOV_ID))
        labels.append(int(label))

    service = SentimentModelService()
    service.load()
    predictions = service.predict_batch(texts)
    correct = sum(1 for p, y in zip(predictions, labels) if p.is_positive == bool(y))
    return correct / len(labels)


def build_model() -> Any:
    """The served architecture. ``mask_zero=True`` is essential, not cosmetic.

    IMDB reviews are far longer than MAX_LEN, so ~20% of every encoded batch is
    padding. With masking disabled the LSTM integrates over those PAD tokens and
    the final hidden state is dominated by noise - measured test accuracy was
    0.54 unmasked versus 0.87 masked on identical data. Masking makes short
    reviews behave exactly like long ones, truncated.
    """
    import tensorflow as tf
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Input
    from tensorflow.keras.models import Sequential

    return Sequential(
        [
            Input(shape=(MAX_LEN,), name="token_ids"),
            Embedding(
                NUM_WORDS,
                EMBEDDING_DIM,
                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.05),
                name="embedding",
                mask_zero=True,
            ),
            LSTM(64, dropout=0.2, name="lstm"),
            Dropout(0.5, name="dropout"),
            Dense(16, activation="relu", name="hidden"),
            Dense(1, activation="sigmoid", name="sentiment_probability"),
        ],
        name="imdb_sentiment_lstm",
    )


def main() -> int:
    args = parse_args()

    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.datasets import imdb

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading IMDB dataset (num_words=%d, max_len=%d)...", NUM_WORDS, MAX_LEN)
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=NUM_WORDS)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(
        x_train, maxlen=MAX_LEN, padding="post", truncating="post", value=PAD_ID
    )
    x_test = tf.keras.preprocessing.sequence.pad_sequences(
        x_test, maxlen=MAX_LEN, padding="post", truncating="post", value=PAD_ID
    )
    log.info(
        "Train shape=%s  Test shape=%s  pos_train_rate=%.3f",
        x_train.shape,
        x_test.shape,
        float(np.mean(y_train)),
    )

    # ---------------------------------------------------------------- architecture
    # A single sigmoid output keeps the contract "score in [0, 1]" that the API
    # response schema promises, and lets us store one float in MongoDB.
    model = build_model()
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=log.info)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=2, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5),
    ]

    started = time.time()
    history = model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=2,
    )
    train_seconds = time.time() - started

    loss, accuracy = model.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=0)
    # Per-class metrics make the README table honest.
    y_prob = model.predict(x_test, batch_size=args.batch_size, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype("int32")
    tp = int(np.sum((y_pred == 1) & (y_test == 1)))
    fp = int(np.sum((y_pred == 1) & (y_test == 0)))
    fn = int(np.sum((y_pred == 0) & (y_test == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0

    log.info("Test accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f", accuracy, precision, recall, f1)

    # ------------------------------------------------------------- save artifacts
    # Re-materialise the graph without optimiser state: the shipped artefact is
    # inference-only, which cuts the .h5 from ~8 MB to ~2.6 MB and keeps training
    # state out of the serving image.
    inference_model = build_model()
    inference_model.build((None, MAX_LEN))
    inference_model.set_weights(model.get_weights())

    train_probe = model.predict(x_test[:256], verbose=0).ravel()
    inference_probe = inference_model.predict(x_test[:256], verbose=0).ravel()
    if not np.allclose(train_probe, inference_probe, atol=1e-5):
        log.warning("Weight transfer changed predictions; saving the trained graph instead")
        inference_model = model

    model_path = output_dir / "sentiment_model.h5"
    inference_model.save(model_path)
    log.info("Saved Keras model -> %s (%.1f KiB)", model_path, model_path.stat().st_size / 1024)

    word_index = imdb.get_word_index()
    trimmed = build_vocabulary(word_index)
    word_index_path = output_dir / "word_index.json"
    word_index_path.write_text(json.dumps(trimmed, ensure_ascii=False), encoding="utf-8")
    log.info("Saved vocabulary -> %s (%d words)", word_index_path, len(trimmed))

    serving_accuracy = verify_serving_path(trimmed, x_test, y_test)
    log.info("Serving-path accuracy (decoded text -> tokenize -> predict): %.4f", serving_accuracy)
    if serving_accuracy < 0.70:
        raise SystemExit(
            f"serving-path accuracy {serving_accuracy:.4f} is far below the trained "
            f"accuracy {accuracy:.4f}; the persisted vocabulary is misaligned with the "
            f"model and must not be published"
        )

    metadata = {
        "model_file": model_path.name,
        "framework": f"tensorflow {tf.__version__}",
        "architecture": f"Embedding({EMBEDDING_DIM}, mask_zero) -> LSTM(64) -> Dense(16, relu) -> Dense(1, sigmoid)",
        "dataset": "IMDB Large Movie Review Dataset (keras.datasets.imdb)",
        "vocab_size": NUM_WORDS,
        "max_len": MAX_LEN,
        "pad_id": PAD_ID,
        "start_id": START_ID,
        "oov_id": OOV_ID,
        "index_from": INDEX_FROM,
        "serving_path_accuracy": round(serving_accuracy, 4),
        "epochs_completed": len(history.history["loss"]),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "train_seconds": round(train_seconds, 1),
        "metrics": {
            "test_loss": round(float(loss), 4),
            "test_accuracy": round(float(accuracy), 4),
            "test_precision": round(precision, 4),
            "test_recall": round(recall, 4),
            "test_f1": round(f1, 4),
            "best_val_accuracy": round(float(max(history.history["val_accuracy"])), 4),
        },
        "parameter_count": int(model.count_params()),
        "positive_label": "positive (label 1)",
        "negative_label": "negative (label 0)",
    }
    metadata_path = output_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log.info("Saved metadata -> %s", metadata_path)
    log.info("Done in %.1fs", train_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
