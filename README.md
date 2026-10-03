# Sentiment Analysis Microservice

A production-shaped sentiment classification service: a FastAPI front door, a
RabbitMQ-backed asynchronous worker, MongoDB persistence, and a real Keras
classifier trained on the IMDB dataset.

The asynchronous path is genuinely asynchronous. `POST /api/sentiment/async`
validates input, publishes a durable message, and returns `202` in single-digit
milliseconds; classification and persistence happen in a separate container. The
worker acks only *after* MongoDB confirms the write, so a crash mid-processing
redelivers the job rather than losing it.

```
POST /api/sentiment/sync   ->  200 {"text", "sentiment", "score"}     (inference inline)
POST /api/sentiment/async  ->  202 {"job_id", "status"}               (queue + worker)
GET  /api/sentiment/results/{job_id}  ->  200 result | 404 JOB_NOT_FOUND
POST /api/sentiment/batch  ->  200 one result per text                (bonus)
GET  /health  /health/ready  /health/live  /                            (bonus)
GET  /docs    /openapi.json                                            (bonus)
```

## Quick start

Prerequisites: Docker Desktop with Compose v2. Nothing else — no Python, no
TensorFlow, no broker or database installation on the host.

```bash
git clone https://github.com/bhargavmartha07/sentiment-analysis-microservice.git
cd sentiment-analysis-microservice
docker compose up -d --build      # first build downloads ~600 MB of wheels
docker compose ps                 # wait until all four services are healthy
```

Then:

```bash
curl -s localhost:8000/api/sentiment/sync \
  -H 'Content-Type: application/json' \
  -d '{"text":"I love this product! It arrived early and works perfectly."}'
# {"text":"...","sentiment":"positive","score":0.977017}

curl -s -X POST localhost:8000/api/sentiment/async \
  -H 'Content-Type: application/json' \
  -d '{"text":"This movie was terrible. A complete waste of time and money."}'
# {"job_id":"a8d4efcd-...","status":"processing"}

curl -s localhost:8000/api/sentiment/results/a8d4efcd-8a32-4dd9-9fe7-f4f8a76ac55b
# {"job_id":"...","sentiment":"negative","score":0.002898,"timestamp":"2026-10-03T09:19:42.834Z"}
```

Interactive docs: <http://localhost:8000/docs>. RabbitMQ management:
<http://localhost:15672> (`guest` / `guest`).

Stop it with `docker compose down`; add `-v` to also drop the MongoDB volume.

## What the model actually is

Not a stub. Trained on `keras.datasets.imdb` (25,000 labelled reviews), embedded
and shipped in the repo.

| Property | Value |
|---|---|
| Architecture | `Embedding(20000, 32)` → `LSTM(64)` → `Dropout(0.5)` → `Dense(16)` → `Dense(1, sigmoid)` |
| Parameters | 665,889 (2.6 MB `.h5`, inference-only weights) |
| Sequence length | 200 tokens, pre-truncated reviews |
| Test accuracy | **0.8413** (precision 0.859, recall 0.817, F1 0.837) |
| Best validation accuracy | 0.8653 |
| Serving-path accuracy | 0.8360 (see below) |
| Training time | ~145 s on CPU, 5 epochs, batch 64 |

`app/models/model_metadata.json` carries these numbers, and the service reports
them through `GET /health`.

Two details were worth real debugging, and both are now guarded by code:

**`mask_zero=True` is load-bearing.** IMDB reviews are much longer than 200
tokens, so roughly 20% of every batch is padding. With masking off, the LSTM
integrates over those pad tokens and the final hidden state is dominated by
noise. Measured on identical data: **0.54 accuracy unmasked vs 0.87 masked.**

**Vocabulary ids are shifted by 3.** `keras.datasets.imdb.get_word_index()`
returns raw ids starting at 0, but `load_data()` reserves embedding rows
`0/1/2` for pad/start/oov and emits each word at `raw_id + 3`. Serving an
unshifted vocabulary trains to 0.85 and then serves *inverted* predictions,
which is exactly the bug this repo shipped and then fixed. `train_model.py` now
re-reads its own artifacts through the real `SentimentModelService` after
saving and **aborts** if serving-path accuracy drops materially below training
accuracy, and `tests/integration` asserts that plainly positive text scores
positive.

## Architecture

```
                      ┌──────────────────────────────┐
  client ──POST──────▶│  api  (FastAPI, :8000)      │
                      │  validates → predicts inline │
                      └───────┬──────────────┬───────┘
                              │ sync          │ async: publish (publisher confirms)
                              ▼               ▼
                        (returns 200)   ┌────────────────────────────┐
                                        │ RabbitMQ                   │
                                        │ durable direct exchange   │
                                        │ sentiment.jobs  ──DLX──▶   │
                                        │                    .dlq    │
                                        └───────────┬────────────────┘
                                                    │ manual ack, prefetch 8
                                                    ▼
                                      ┌────────────────────────────┐
                                      │ worker  (separate image)   │
                                      │ load model → predict →     │
                                      │ upsert MongoDB → ack       │
                                      └───────────┬────────────────┘
                                                  ▼
                                        ┌────────────────────────────┐
                                        │ MongoDB  sentiment_results │
                                        │ unique job_id, ts index    │
                                        └────────────────────────────┘
```

`api` and `worker` are separate images with separate processes — the API never
consumes from the queue, and the worker serves no traffic. Scaling is
`docker compose up -d --scale worker=3`.

Design rationale, failure modes, and the alternatives rejected are in
[ARCHITECTURE.md](ARCHITECTURE.md).

### Delivery and failure semantics

| Situation | Behaviour |
|---|---|
| Malformed payload | `nack(requeue=False)` → dead-letter queue, never retried |
| Transient failure (Mongo/broker) | republish with `x-retry-count++`, exponential backoff, then ack |
| Retries exhausted | `nack(requeue=False)` → dead-letter queue |
| Unconfirmed republish | original nacked back onto the queue — delayed, never dropped |
| Worker crash mid-job | message unacked → redelivered; upsert on `job_id` keeps it idempotent |

`basic_nack(requeue=True)` alone would redeliver the *same* envelope, so a
header-carried retry counter never advances and a poison job loops forever.
The worker republishes a fresh copy and only acks the original after the broker
confirms, which is why the retry budget actually terminates.

One more broker subtlety: `pika.BlockingConnection` only services AMQP
heartbeats while it is doing I/O, so a quiet API has its connection reaped by
the broker after the heartbeat timeout. Requests still succeed — publishing
reconnects on demand — but a naive readiness probe would report a false `503`
and a load balancer would pull a healthy instance out of rotation. The API runs
a background keepalive (`QueuePublisher.keepalive`) and the readiness probe
reconnects rather than merely inspecting (`ensure_connected`), so "ready" means
"can serve traffic now".

## API

### `POST /api/sentiment/sync`

```bash
curl -s localhost:8000/api/sentiment/sync \
  -H 'Content-Type: application/json' -d '{"text":"a wonderful film"}'
```

```json
{"text": "a wonderful film", "sentiment": "positive", "score": 0.912004}
```

### `POST /api/sentiment/async`

Returns `202` once the broker confirms the message, with a generated UUID.

```json
{"job_id": "a8d4efcd-8a32-4dd9-9fe7-f4f8a76ac55b", "status": "processing"}
```

### `GET /api/sentiment/results/{job_id}`

`200` with the stored result once the worker has finished. Results are
idempotent and persisted, so this keeps working after the job leaves the queue.

### Errors

| Status | `error_code` | When |
|---|---|---|
| 400 | `INVALID_INPUT` | blank/oversized text, malformed UUID |
| 404 | `JOB_NOT_FOUND` | well-formed UUID with no result |
| 503 | `BROKER_UNAVAILABLE` | RabbitMQ unreachable on submit |
| 503 | `DATABASE_UNAVAILABLE` | MongoDB unreachable on lookup |
| 500 | `INTERNAL_ERROR` | anything unhandled |

```json
{
  "detail": "text: Text field is required and must not be empty.",
  "error_code": "INVALID_INPUT"
}
```

## Tests

```bash
docker compose exec api    python -m pytest tests/unit/api_tests.py -v
docker compose exec worker python -m pytest tests/unit/service_tests.py -v
docker compose exec api    python -m pytest tests/integration -v
```

| Suite | Count | Needs a live stack? |
|---|---|---|
| `tests/unit/api_tests.py` | endpoint, validation, error-mapping, OpenAPI | no |
| `tests/unit/service_tests.py` | model, queue, Mongo, worker, retry/DLQ, logging, settings | no |
| `tests/integration/integration_tests.py` | real API → RabbitMQ → worker → MongoDB | yes |

Current status: **114 unit tests pass, 30 integration tests pass**, and
`python -m ruff check app tests scripts --select=F,E9` is clean.

The integration suite exercises the parts that unit tests cannot: that a job
really traverses the broker and lands in MongoDB, that a redelivered job does
not duplicate its document, that a malformed message reaches the DLQ, that
sync and async agree, and that positive text scores positive.

Two end-to-end scripts:

```bash
docker compose exec api python scripts/smoke_test.py    # 14 assertions, exits non-zero on failure
docker compose exec api python scripts/benchmark.py     # latency/throughput, indicative numbers only
```

## Measured behaviour

Single request, `docker compose` on an ordinary laptop CPU, TensorFlow CPU:

```
POST /sync (c=1)              p50   82 ms   p90   93 ms   p99  122 ms
POST /sync (c=8)              p50  667 ms   p90  754 ms   p99  919 ms
POST /batch (32 texts)        146 texts/s
async submit -> result        p50  831 ms   (end-to-end, includes queue round trip)
async throughput              9.2 jobs/s
```

The concurrency figure is honest about a real property: one TensorFlow session
holds the GIL, so `/sync` requests serialise inside the interpreter. `/sync` is
for interactive single calls; `/async` is the path that scales, because the work
moves to worker processes that scale independently. These are indicative
numbers from one machine, not a published benchmark.

## Configuration

Copy `.env.example` to `.env` to override anything; Compose reads it
automatically and every value has a working default, so the stack runs with no
`.env` at all.

| Variable | Default | Purpose |
|---|---|---|
| `ENVIRONMENT` | `development` | Surfaced by `/health` |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` / `json` | Structured JSON logs by default |
| `MODEL_PATH` | `app/models/sentiment_model.h5` | Inference artifact |
| `WORD_INDEX_PATH` | `app/models/word_index.json` | Vocabulary (ids shifted by 3) |
| `POSITIVE_THRESHOLD` | `0.5` | Score at/above which text is positive |
| `WORKER_MAX_RETRIES` | `3` | Attempts before dead-lettering |
| `RABBITMQ_PREFETCH_COUNT` | `8` | Unacked messages per worker |
| `MONGO_URI` / `MONGO_DB_NAME` / `MONGO_COLLECTION` | see `.env.example` | Persistence target |
| `TF_ENABLE_GPU` | `false` | CPU by default; no GPU required |

## Retraining

```bash
make train        # or: docker compose run --rm --no-deps \
                  #      -v "$PWD/app/models:/app/app/models" \
                  #      api python scripts/train_model.py --epochs 5
```

Downloads IMDB, trains for ~2.5 minutes on CPU, and rewrites `sentiment_model.h5`,
`word_index.json` and `model_metadata.json` in place. The run fails loudly if the
artifacts it just wrote do not reproduce training accuracy through the real
inference path.

## Project layout

```
app/
  main.py                  FastAPI app, lifespan, error handlers, health
  config.py                env-driven Settings (pydantic-settings)
  schemas.py               request/response contracts and validators
  errors.py                typed service errors and their HTTP mapping
  logging_config.py        structured JSON logging
  api/endpoints.py         sync / async / results / batch routes
  services/model_service.py   lazy TF load, tokenize, predict, warmup
  services/queue_service.py    topology, publisher confirms, consumer, Delivery
  services/db_service.py       Mongo repository, indexes, idempotent upsert
  worker/consumer.py       worker loop, retries, DLQ, heartbeat healthcheck
  models/                  sentiment_model.h5, word_index.json, model_metadata.json
scripts/                   train_model.py, smoke_test.py, benchmark.py
tests/unit, tests/integration
Dockerfile.api, Dockerfile.worker, docker-compose.yml, Makefile
ARCHITECTURE.md, postman_collection.json
```

## Notes and limitations

- **Domain.** Trained on movie reviews; product and support text will score
  worse than the IMDB numbers suggest. Retrain on in-domain data before
  treating the score as calibrated.
- **English only**, lowercase-tokenised, 200-token window.
- **Single-replica MongoDB.** The compose file runs a standalone `mongod` with
  no replica set, so transactions are unavailable; the design relies on atomic
  single-document upserts instead, which is sufficient for idempotency.
- **Credentials default to `guest`/`guest`** for local use. Anything beyond a
  laptop needs real secrets and TLS.
- No authentication or rate limiting is implemented — both belong in front of
  this service, not inside it.
