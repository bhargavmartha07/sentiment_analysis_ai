# Architecture

> Companion document to [`README.md`](README.md). It explains *why* the system is
> shaped the way it is; the README explains how to run it.

## 1. System context

The service ingests short free-form text and returns a binary sentiment label
with a confidence score. The same model serves two very different traffic
profiles, which is the central design tension:

| Path | Latency requirement | Volume | Backpressure |
|---|---|---|---|
| **Synchronous** | tens of milliseconds | low | the API itself |
| **Asynchronous** | seconds to minutes | high, bursty | the queue |

Rather than pick one, the service exposes both and shares every component except
the transport. Both paths run the *same* model artifact through the *same*
`SentimentModelService`, so a client cannot observe different behaviour between
them (asserted by `test_async_and_sync_paths_agree`).

```
                       ┌──────────────────────────────────────────┐
   interactive client ─┤  POST /api/sentiment/sync                 │
                       │  POST /api/sentiment/batch                │
                       └───────────────┬──────────────────────────┘
                                       │ in-process inference
                                       ▼
                       ┌──────────────────────────────────────────┐
   batch client     ──┤  POST /api/sentiment/async  (202 + job_id)│
                       └───────────────┬──────────────────────────┘
                                       │ publisher confirms
                                       ▼
                       ┌──────────────────────────────────────────┐
                       │  RabbitMQ  sentiment.exchange             │
                       │    └── sentiment.jobs  (durable)         │
                       │          └── sentiment.jobs.dlq (DLX)     │
                       └───────────────┬──────────────────────────┘
                                       │ manual ack after persist
                                       ▼
                       ┌──────────────────────────────────────────┐
                       │  worker  (N replicas, same image)        │
                       │  load .h5 once → infer → upsert → ack    │
                       └───────────────┬──────────────────────────┘
                                       │
                                       ▼
                       ┌──────────────────────────────────────────┐
   polling client    ──┤  GET /api/sentiment/results/{job_id}     │
                       └───────────────┬──────────────────────────┘
                                       ▼
                       ┌──────────────────────────────────────────┐
                       │  MongoDB  sentiment_db.sentiment_results │
                       └──────────────────────────────────────────┘
```

## 2. Components and responsibilities

| Module | Owns | Deliberately does **not** own |
|---|---|---|
| `app/api/endpoints.py` | HTTP contract, status codes, response shaping | AMQP, MongoDB, TensorFlow internals |
| `app/services/model_service.py` | loading the `.h5`, text encoding, inference | HTTP, persistence |
| `app/services/queue_service.py` | AMQP topology, publishing, consuming, reconnects | business logic, persistence |
| `app/services/db_service.py` | MongoDB connection, indexes, documents | inference, HTTP |
| `app/worker/consumer.py` | job lifecycle: infer → persist → ack/nack | transport details |
| `app/config.py` | every environment-driven value | business logic |

The rule: **transport and business logic never mix.** `queue_service` never
knows what sentiment means; `consumer` never imports `pika`. That is what allows
the worker's ack/nack policy to be unit-tested with plain fakes, and the
publisher's error mapping to be tested without a broker.

## 3. Key decisions

### 3.1 Model loaded once per process, warm-started

`SentimentModelService.load()` is idempotent, guarded by an `RLock`, and runs
in the FastAPI lifespan so the first user request never pays the load cost. A
warm-up forward pass follows immediately, because TensorFlow initialises its
kernels lazily and would otherwise put a multi-second spike on whichever unlucky
request arrived first.

Loading is a **fatal** start-up error. A service that cannot do its primary job
should crash and be restarted, not accept traffic it will fail.

### 3.2 Vocabulary as JSON, not a pickled Keras `Tokenizer`

The obvious approach is `pickle`d `keras.preprocessing.Tokenizer`. It is a trap:
the pickle embeds the exact Keras version that created it, so a routine library
upgrade can silently break inference in production. This service ships
`word_index.json` and does the encoding in ~20 lines of Python:

```python
_TOKENS = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
```

Consequences: the artifact is diff-able in git, portable across Keras versions,
and testable without TensorFlow installed (which is exactly what
`tests/unit/service_tests.py` does).

`max_len` is resolved from the model card, falling back to the model's own
`input_shape[-1]` — the model, not a constant, is the source of truth.

One subtlety is worth stating explicitly, because getting it wrong produces a
model that trains to 0.85 accuracy and then serves *inverted* predictions.
`keras.datasets.imdb.get_word_index()` returns raw word ids starting at 0,
while `load_data()` reserves embedding rows `0/1/2` for pad/start/oov and emits
every word at `raw_id + 3`. The vocabulary we persist must therefore be shifted
by the same `INDEX_FROM = 3`, and unknown words must map to row `2`, not row
`1`. `scripts/train_model.py` re-reads its own artifacts through the real
`SentimentModelService` after saving and aborts the run if serving-path accuracy
falls more than a few points below training accuracy, so this class of bug fails
loudly instead of shipping.

### 3.3 At-least-once delivery with an idempotent write

The worker acks **after** the MongoDB write. A crash in that window means the
message is redelivered, so the write must be idempotent:

```python
collection.find_one_and_update({"job_id": job_id}, {"$set": document}, upsert=True)
```

with a **unique index on `job_id`**. Redelivery therefore converges on exactly
one document — verified end-to-end by
`test_redelivered_job_does_not_duplicate_the_document`, which publishes the same
job twice and asserts a single stored record.

The alternative (ack before write) is at-most-once and silently loses results.

### 3.4 Retry classification, not blanket retries

| Failure | Action | Rationale |
|---|---|---|
| Malformed payload | `nack(requeue=False)` → DLQ | retrying cannot fix bad data |
| MongoDB/broker transient | requeue with `x-retry-count++`, exponential sleep | self-healing |
| Retries exhausted | `nack(requeue=False)` → DLQ | stop poison-message loops |

Without this, one poison message re-enters the queue forever and starves real
work — the classic RabbitMQ failure mode. `prefetch_count=8` bounds how much
unfinished work a single worker can hold.

The retry counter lives in the message header, which has a sharp edge:
`basic_nack(requeue=True)` redelivers the *same* envelope, so `x-retry-count`
would never advance and the job would loop until the worker was restarted. The
worker therefore republishes a fresh copy with the header incremented and only
acks the original once the broker confirms the republish; if the confirm does
not arrive, the original is nacked back onto the queue so the job is delayed
rather than lost.

### 3.5 Blocking I/O is offloaded from the event loop

FastAPI is async, but TensorFlow inference, `pika` publishing and `pymongo`
reads are blocking. Every call goes through
`starlette.concurrency.run_in_threadpool`, so a slow MongoDB query cannot stall
health checks or other in-flight requests.

`QueuePublisher` additionally serialises channel access behind a lock —
`pika.BlockingConnection` is explicitly not thread-safe. Under concurrency each
publish may wait its turn, but the invariant holds (asserted by
`test_publish_is_thread_safe`, 20 concurrent publishes).

### 3.6 Publisher confirms make `202 Accepted` truthful

`mandatory=True` plus `confirm_delivery()` means a 202 is only returned once
RabbitMQ has durably accepted the message. Without confirms, a broker-side
failure can surface as an accepted job that is never processed.

If the broker is down, the publisher enters a capped backoff instead of
hot-looping, and `/api/sentiment/async` returns **503 `QUEUE_UNAVAILABLE`**
rather than accepting work it cannot guarantee.

### 3.7 Errors are a stable contract

Every failure maps to `(status_code, error_code)` in one place
(`app/errors.py`), and FastAPI's default **422** is deliberately remapped to
**400** because the brief specifies 400 for invalid input:

```json
{ "detail": "text: Text field is required and must not be empty.", "error_code": "INVALID_INPUT" }
```

Client-visible `detail` strings are curated and never leak internals; stack
traces go to the structured log instead. `tests/unit/api_tests.py` asserts a
model crash returns 500 *without* echoing the underlying exception message.

For `GET /results/{job_id}` the split is deliberate: a well-formed UUID with no
record is **404 `JOB_NOT_FOUND`**, while a syntactically invalid id is **400
`INVALID_INPUT`** (input validation, per the brief) rather than a misleading 404.

### 3.8 Health, readiness, liveness — and how a batch process gets one

`/health` is a pure liveness probe: no dependencies touched, never fails, safe as
a Docker `HEALTHCHECK`. `/health/ready` actively probes MongoDB (`ping`), RabbitMQ
(channel state) and the model, returning **503** with a per-dependency breakdown
so an orchestrator can tell *which* dependency is unhealthy.

The worker has no HTTP server, so its health is a **heartbeat file** it rewrites
(at minimum) every 5 seconds while idle. `python -m app.worker.consumer
--healthcheck` exits 0 only if that file is younger than
`WORKER_HEARTBEAT_STALE_SECONDS`. The file is written to a temp path and
`os.replace`d so a probe never reads a half-written document. This gives Docker a
real health signal without inventing an HTTP server just to satisfy the check.

### 3.9 Structured logging everywhere

One JSON object per line, same envelope in both services:

```json
{"timestamp":"2024-05-01T10:30:00.412Z","level":"INFO","service":"worker",
 "logger":"app.worker.consumer","message":"Job completed","event":"worker.job.completed",
 "job_id":"a1b2...","sentiment":"negative","score":0.052,"latency_ms":41}
```

`event` is a stable, greppable key; ad-hoc context arrives via `extra=`. Uvicorn's
handlers are re-pointed at the same formatter so access logs share the envelope.
MongoDB URIs are logged with credentials redacted
(`Settings.mongo_uri_plain`). Request logs are emitted by a middleware that also
injects `X-Process-Time-Ms` and `X-Request-ID`.

### 3.10 Configuration is 12-factor

`app/config.py` is the only module that reads configuration; nothing else calls
`os.environ`. Secrets are `pydantic.SecretStr` so they cannot leak through
`repr()` or an accidental log line. `docker-compose.yml` supplies `${VAR:-default}`
for everything, so the stack boots with **zero** configuration while still
honouring a local `.env` (which Compose reads automatically). The MongoDB password
in an example file is a local dev default that is never used in production.

### 3.11 One connection per process, shared

`MongoClient` and the RabbitMQ publisher are process-wide singletons with lazy
connect and explicit backoff. Both maintain their own internal connection pools,
so per-request client creation — the most common cause of connection storms —
never happens. `reset_*()` helpers exist so tests can isolate state.

## 4. Data model

```json
{
  "_id":       "ObjectId(...)",
  "job_id":    "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "text":      "This movie was terrible.",
  "sentiment": "negative",
  "score":     0.052,
  "timestamp": ISODate("2024-05-01T10:30:00.412Z"),
  "model":     { "architecture": "...", "metrics": { "test_accuracy": 0.87 } },
  "latency_ms": 41,
  "attempt":    1
}
```

Indexes: `job_id` **unique** (lookup + idempotency), `timestamp` desc (recent
activity), `(sentiment, timestamp)` (label analytics).

`model` snapshots the model card with each result, so historical results remain
interpretable after a model upgrade. `timestamp` is written timezone-aware and
read back with `tz_aware=True` so the API serialises `...Z` exactly as documented
rather than a naive local time.

## 5. Model

`app/models/sentiment_model.h5` — an LSTM classifier trained on the IMDB Large
Movie Review Dataset (25k balanced labelled reviews) via
`scripts/train_model.py`:

```
Input(200) → Embedding(20000, 32) → LSTM(64) → Dropout → Dense(16, relu) → Dense(1, sigmoid)
```

A single sigmoid output keeps the published contract (`score ∈ [0, 1]`, one float
in MongoDB) trivially honest. Trained weights, the vocabulary and
`model_metadata.json` (metrics, seed, dataset, framework version) are committed
so `docker compose up` needs no training step. Retraining is one command:

```bash
make train        # or: docker compose run --rm api python scripts/train_model.py
```

Measured metrics are in `app/models/model_metadata.json` and quoted in the
README.

## 6. Failure modes considered

| Failure | Behaviour |
|---|---|
| Broker down at start-up | API still serves `/sync`; `/async` → 503; publisher backs off and reconnects |
| Broker dies mid-run | Worker reconnects with capped exponential backoff, unacked messages are redelivered |
| MongoDB down | Worker retries with backoff, then dead-letters; API returns 500/503 |
| Model artifact missing | Start-up fails loudly rather than serving wrong answers |
| Poison message | Dead-lettered after N attempts, never blocks the queue |
| API pod dies mid-request | Client retries with the same `job_id`; upsert keeps one record |
| Poison input (blank text) | Rejected at the edge with 400; never reaches the queue |

## 7. Scaling

- **API** — stateless; scale horizontally. Each replica holds its own model copy,
  so memory (≈300 MB resident) is the per-replica cost to plan for.
- **Worker** — scale by replica count; `prefetch_count` controls per-worker
  in-flight work. Because results are keyed by `job_id`, order and placement are
  irrelevant.
- **Queue** — durable messages survive broker restarts; add competing consumers
  with no code change.
- **Batching** — `/batch` amortises interpreter overhead across up to 32 texts.

## 8. What I would add next

1. **Idempotency keys** on `/async` so a client retry cannot create duplicate
   jobs (today the `job_id` is server-generated).
2. **Dead-letter replay tooling** — inspect the DLQ and re-drive entries.
3. **Rate limiting** (per-token bucket) to protect the model from abusive bursts.
4. **Prometheus metrics** (`/metrics`) — request latency histogram, queue depth,
   worker throughput; the counters already exist in the heartbeat file.
5. **Model drift monitoring** — track the score distribution in MongoDB and alert
   when it shifts.
6. **A transformer encoder** (e.g. DistilBERT) behind the same `Prediction`
   interface; `model_service` is the only module that would change.
