"""HTTP endpoints for sentiment analysis.

Three documented contracts:

* ``POST /api/sentiment/sync``     -> infer inline, return the result (200)
* ``POST /api/sentiment/async``    -> enqueue a job, return a job id (202)
* ``GET  /api/sentiment/results/{job_id}`` -> read the persisted result (200/404)

Blocking work (TensorFlow inference, AMQP publish, MongoDB reads) is dispatched
to a threadpool via :func:`starlette.concurrency.run_in_threadpool` so the event
loop keeps serving other requests.
"""

from __future__ import annotations

import time
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, status
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import (
    get_model,
    get_publisher_dependency,
    get_repository_dependency,
)
from app.errors import InvalidInputError, JobNotFoundError
from app.logging_config import get_logger
from app.schemas import (
    AsyncJobResponse,
    BatchSentimentRequest,
    BatchSentimentResponse,
    ErrorResponse,
    JobResultResponse,
    SentimentRequest,
    SyncSentimentResponse,
)
from app.services.db_service import SentimentRepository, document_to_api
from app.services.model_service import Prediction, SentimentModelService
from app.services.queue_service import QueuePublisher, SentimentJobMessage, new_job_id

logger = get_logger(__name__)

router = APIRouter(prefix="/api/sentiment", tags=["sentiment"])

# Shared error catalogue so every endpoint documents the same failure modes.
_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"description": "Invalid input (validation failed)", "model": ErrorResponse},
    404: {"description": "Job not found", "model": ErrorResponse},
    500: {"description": "Internal server error", "model": ErrorResponse},
    503: {"description": "A dependency is unavailable", "model": ErrorResponse},
}


def _to_response(text: str, prediction: Prediction) -> SyncSentimentResponse:
    return SyncSentimentResponse(text=text, sentiment=prediction.sentiment, score=prediction.score)


@router.post(
    "/sync",
    response_model=SyncSentimentResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse text synchronously",
    description=(
        "Runs inference in-process and returns the sentiment immediately. "
        "Use this for interactive traffic; switch to `/async` for bulk or latency-tolerant workloads."
    ),
    responses=_ERROR_RESPONSES,
)
async def analyze_sync(
    payload: SentimentRequest,
    model: Annotated[SentimentModelService, Depends(get_model)],
) -> SyncSentimentResponse:
    started = time.perf_counter()
    predictions = await run_in_threadpool(model.predict_batch, [payload.text])
    prediction = predictions[0]
    logger.info(
        "Synchronous analysis completed",
        extra={
            "event": "sentiment.sync.completed",
            "text_length": len(payload.text),
            "sentiment": prediction.sentiment,
            "score": prediction.score,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        },
    )
    return _to_response(payload.text, prediction)


@router.post(
    "/batch",
    response_model=BatchSentimentResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse several texts in one forward pass",
    description=(
        "Bonus endpoint. Amortises interpreter overhead across up to 32 texts, "
        "which is markedly faster than issuing one `/sync` call per text."
    ),
    responses=_ERROR_RESPONSES,
)
async def analyze_batch(
    payload: BatchSentimentRequest,
    model: Annotated[SentimentModelService, Depends(get_model)],
) -> BatchSentimentResponse:
    started = time.perf_counter()
    predictions = await run_in_threadpool(model.predict_batch, payload.texts)
    logger.info(
        "Batch analysis completed",
        extra={
            "event": "sentiment.batch.completed",
            "count": len(predictions),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        },
    )
    return BatchSentimentResponse(
        count=len(predictions),
        results=[_to_response(text, prediction) for text, prediction in zip(payload.texts, predictions)],
    )


@router.post(
    "/async",
    response_model=AsyncJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue text for asynchronous analysis",
    description=(
        "Publishes the text to RabbitMQ and returns immediately with a job id. "
        "The worker performs inference and persists the result to MongoDB; "
        "poll `GET /api/sentiment/results/{job_id}` to collect it. "
        "Returns 503 if the broker cannot durably accept the job."
    ),
    responses=_ERROR_RESPONSES,
)
async def submit_async(
    payload: SentimentRequest,
    publisher: Annotated[QueuePublisher, Depends(get_publisher_dependency)],
) -> AsyncJobResponse:
    job_id = new_job_id()
    message = SentimentJobMessage.create(job_id, payload.text)

    # Publisher confirms mean this only returns once RabbitMQ has taken
    # responsibility for the message, so a 202 is never a lie.
    await run_in_threadpool(publisher.publish, message)

    logger.info(
        "Async job accepted",
        extra={"event": "sentiment.async.accepted", "job_id": job_id, "text_length": len(payload.text)},
    )
    return AsyncJobResponse(job_id=uuid.UUID(job_id), status="processing")


@router.get(
    "/results/{job_id}",
    response_model=JobResultResponse,
    status_code=status.HTTP_200_OK,
    summary="Fetch a persisted analysis result",
    description=(
        "Returns the stored result for an asynchronous job. "
        "A well-formed id with no stored result yields 404 `JOB_NOT_FOUND`; "
        "a malformed id yields 400 `INVALID_INPUT`."
    ),
    responses=_ERROR_RESPONSES,
)
async def get_result(
    job_id: str,
    repository: Annotated[SentimentRepository, Depends(get_repository_dependency)],
) -> JobResultResponse:
    canonical_id = _canonicalise_job_id(job_id)
    document = await run_in_threadpool(repository.get_by_job_id, canonical_id)
    if document is None:
        logger.info(
            "Result lookup miss",
            extra={"event": "sentiment.result.miss", "job_id": canonical_id},
        )
        raise JobNotFoundError.for_job(job_id)
    return JobResultResponse(**(document_to_api(document) or {}))


def _canonicalise_job_id(job_id: str) -> str:
    """Validate the job id is a UUID and return its canonical form."""
    try:
        return str(uuid.UUID(job_id))
    except (ValueError, AttributeError, TypeError) as exc:
        raise InvalidInputError(
            f"Job ID '{job_id}' is not a valid UUID.",
            context={"job_id": job_id},
        ) from exc


__all__ = ["router"]
