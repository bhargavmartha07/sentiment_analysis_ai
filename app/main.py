"""FastAPI application factory and entry point.

Run directly with ``python -m app.main`` or via uvicorn:

    uvicorn app.main:app --host 0.0.0.0 --port 8000

Startup order (lifespan):

1. configure structured logging - first, so every later step is observable
2. load the Keras model once into memory (fail fast if the artifact is broken)
3. ensure MongoDB indexes (best effort; a warning does not block traffic)
4. warm the RabbitMQ publisher connection (best effort; ``/async`` returns a
   descriptive 503 until it is available)

Only step 2 is fatal by design: a service that cannot serve its primary function
should exit and be restarted rather than accept traffic it cannot answer.
"""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncIterator

from fastapi import Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.dependencies import (
    get_model,
    get_publisher_dependency,
    get_repository_dependency,
)
from app.api.endpoints import router as sentiment_router
from app.config import Settings, get_settings
from app.errors import ErrorCode, ServiceError
from app.logging_config import configure_logging, get_logger
from app.schemas import DependencyStatus, HealthResponse
from app.services.db_service import SentimentRepository, get_repository
from app.services.model_service import SentimentModelService, get_model_service
from app.services.queue_service import (
    QueuePublisher,
    build_connection_parameters,
    get_publisher,
)

logger = get_logger(__name__)

DESCRIPTION = """
Event-driven sentiment analysis service.

* **`/api/sentiment/sync`** - inline inference, immediate response.
* **`/api/sentiment/async`** - durable hand-off to RabbitMQ; poll for the result.
* **`/api/sentiment/results/{job_id}`** - result lookup backed by MongoDB.

The same TensorFlow Keras model is loaded once per process by both the API and
the worker, so behaviour is identical on both paths.
"""


def _error_payload(detail: str, code: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"detail": detail, "error_code": code}
    if extra:
        payload["context"] = extra
    return payload


def _publisher_keepalive_loop(publisher: Any, stop_event: threading.Event) -> None:
    """Keep the AMQP heartbeat alive while the API is idle.

    Without this, a quiet service has its broker connection reaped after the
    heartbeat timeout, and readiness then reports a false 503 for a service that
    would reconnect and serve the request perfectly well.
    """
    interval = max(5.0, build_connection_parameters(get_settings()).heartbeat / 3)
    while not stop_event.wait(interval):
        try:
            await_free = publisher.keepalive()
        except Exception:  # pragma: no cover - keepalive is best effort
            logger.debug("Publisher keepalive raised", extra={"event": "queue.keepalive.error"})
            continue
        if not await_free:
            logger.info(
                "Publisher connection dropped while idle; it will reconnect on demand",
                extra={"event": "queue.keepalive.idle_disconnect"},
            )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load resources on start-up, release them on shutdown."""
    settings: Settings = app.state.settings
    configure_logging(
        service="api",
        level=settings.log_level,
        log_format=settings.log_format,
    )
    logger.info(
        "Starting sentiment API",
        extra={
            "event": "service.start",
            "version": settings.version,
            "environment": settings.environment,
            "model_path": settings.model_path,
            "mongo_uri": settings.mongo_uri_plain,
            "rabbitmq_host": settings.rabbitmq_host,
        },
    )

    model_service = get_model_service()
    try:
        model_service.load()
    except Exception:
        logger.critical(
            "Model initialisation failed; aborting start-up",
            extra={"event": "service.model_load.failed", "model_path": settings.model_path},
            exc_info=True,
        )
        raise
    app.state.model_service = model_service

    try:
        get_repository().ensure_indexes()
    except Exception:
        logger.warning(
            "MongoDB indexes could not be ensured at start-up",
            extra={"event": "service.index_ensure.failed"},
            exc_info=True,
        )

    try:
        get_publisher().connect()
    except Exception:
        logger.warning(
            "RabbitMQ publisher not connected at start-up; /async will return 503 until it recovers",
            extra={"event": "service.publisher_connect.failed"},
        )

    stop_keepalive = threading.Event()
    keepalive_thread = threading.Thread(
        target=_publisher_keepalive_loop,
        args=(get_publisher(), stop_keepalive),
        name="rabbitmq-keepalive",
        daemon=True,
    )
    keepalive_thread.start()

    logger.info("Sentiment API ready", extra={"event": "service.ready"})
    try:
        yield
    finally:
        stop_keepalive.set()
        keepalive_thread.join(timeout=2.0)
        logger.info("Shutting down sentiment API", extra={"event": "service.shutdown.start"})
        try:
            get_publisher().close()
        except Exception:
            logger.debug("Publisher close failed", extra={"event": "service.publisher_close.failed"})
        try:
            get_repository().close()
        except Exception:
            logger.debug("Repository close failed", extra={"event": "service.repository_close.failed"})
        logger.info("Shutdown complete", extra={"event": "service.shutdown.complete"})


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI application. Split out so tests can build isolated apps."""
    settings = settings or get_settings()

    application = FastAPI(
        title="Sentiment Analysis API",
        description=DESCRIPTION,
        version=settings.version,
        openapi_tags=[
            {"name": "sentiment", "description": "Synchronous and asynchronous sentiment analysis."},
            {"name": "health", "description": "Liveness and readiness probes for orchestrators."},
        ],
        contact={"name": "Sentiment Microservice"},
        license_info={"name": "MIT"},
        lifespan=lifespan,
    )
    application.state.settings = settings

    # ------------------------------------------------------------ middleware
    @application.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Any:
        """Attach a request id, log the exchange and time the handler."""
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Registered exception handlers own logging and rendering of these
            # failures; re-raise untouched so they stay the single source.
            raise
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers["X-Process-Time-Ms"] = f"{duration_ms}"
        response.headers["X-Request-ID"] = request_id
        if request.url.path not in ("/health", "/health/live"):
            logger.info(
                "Request handled",
                extra={
                    "event": "http.request.completed",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": duration_ms,
                    "client": request.client.host if request.client else None,
                },
            )
        return response

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time-Ms", "X-Request-ID"],
    )

    # ------------------------------------------------------ error handlers
    @application.exception_handler(ServiceError)
    async def handle_service_error(request: Request, exc: ServiceError) -> JSONResponse:
        log = logger.warning if exc.status_code < 500 else logger.error
        log(
            "Request rejected",
            extra={
                "event": "http.error.service_error",
                "request_id": getattr(request.state, "request_id", None),
                "path": request.url.path,
                "status_code": exc.status_code,
                "error_code": exc.error_code.value,
                "detail": exc.detail,
                **exc.context,
            },
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(exc.detail, exc.error_code.value, **exc.context),
        )

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Convert pydantic/fastapi validation failures into 400 + error_code.

        FastAPI defaults to 422; the API contract requires 400 for invalid input,
        so the mapping is done explicitly here.
        """
        detail = _humanise_validation_errors(exc.errors())
        logger.info(
            "Request validation failed",
            extra={
                "event": "http.error.validation",
                "request_id": getattr(request.state, "request_id", None),
                "path": request.url.path,
                "status_code": status.HTTP_400_BAD_REQUEST,
                "error_code": ErrorCode.INVALID_INPUT.value,
                "detail": detail,
            },
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=_error_payload(detail, ErrorCode.INVALID_INPUT.value),
        )

    @application.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {
            400: ErrorCode.INVALID_INPUT.value,
            404: ErrorCode.JOB_NOT_FOUND.value,
            503: ErrorCode.SERVICE_UNAVAILABLE.value,
        }.get(exc.status_code, ErrorCode.INTERNAL_ERROR.value)
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(str(exc.detail), code),
            headers=getattr(exc, "headers", None),
        )

    @application.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unexpected application error",
            extra={
                "event": "http.error.unhandled",
                "request_id": getattr(request.state, "request_id", None),
                "path": request.url.path,
                "exception_type": type(exc).__name__,
            },
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_payload(
                "An unexpected internal error occurred. The incident has been logged.",
                ErrorCode.INTERNAL_ERROR.value,
            ),
        )

    # ------------------------------------------------------------- routes
    application.include_router(sentiment_router)
    _register_health_routes(application)
    _register_root_routes(application)
    return application


def _humanise_validation_errors(errors: list[dict[str, Any]]) -> str:
    """Turn pydantic error objects into a single readable sentence."""
    messages: list[str] = []
    for error in errors:
        location = [str(part) for part in error.get("loc", []) if part not in ("body", "query", "path")]
        field = ".".join(location) or "request"
        message = error.get("msg", "invalid value")
        message = message.removeprefix("Value error, ")
        messages.append(f"{field}: {message}")
    return "; ".join(messages) or "Request validation failed."


def _register_health_routes(application: FastAPI) -> None:
    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["health"],
        summary="Liveness probe",
        description="Returns 200 as long as the process is serving. Does not touch dependencies.",
    )
    async def health_check(
        request: Request,
        model: Annotated[SentimentModelService, Depends(get_model)],
    ) -> HealthResponse:
        settings: Settings = request.app.state.settings
        return HealthResponse(
            status="ok",
            service=settings.service_name,
            version=settings.version,
            environment=settings.environment,
            model_loaded=model.is_loaded,
        )

    @application.get(
        "/health/live",
        response_model=HealthResponse,
        tags=["health"],
        summary="Liveness probe (Kubernetes style)",
    )
    async def liveness(
        request: Request,
        model: Annotated[SentimentModelService, Depends(get_model)],
    ) -> HealthResponse:
        return await health_check(request, model)

    @application.get(
        "/health/ready",
        response_model=HealthResponse,
        tags=["health"],
        summary="Readiness probe",
        description="Probes the model, MongoDB and RabbitMQ. Returns 503 if any dependency is down.",
        responses={503: {"description": "One or more dependencies are unavailable", "model": HealthResponse}},    )
    async def readiness(
        request: Request,
        model: Annotated[SentimentModelService, Depends(get_model)],
        repository: Annotated[SentimentRepository, Depends(get_repository_dependency)],
        publisher: Annotated[QueuePublisher, Depends(get_publisher_dependency)],
    ) -> Any:
        settings: Settings = request.app.state.settings
        dependencies: dict[str, DependencyStatus] = {}

        if model.is_loaded:
            dependencies["model"] = DependencyStatus(status="up")
        else:
            dependencies["model"] = DependencyStatus(status="down", detail="Keras model is not loaded")

        try:
            mongo_up = await run_in_threadpool(repository.ping)
            dependencies["mongodb"] = DependencyStatus(
                status="up" if mongo_up else "down",
                detail=None if mongo_up else "ping failed",
            )
        except Exception as exc:
            dependencies["mongodb"] = DependencyStatus(status="down", detail=str(exc))

        rabbit_up = await run_in_threadpool(publisher.ensure_connected)
        dependencies["rabbitmq"] = DependencyStatus(
            status="up" if rabbit_up else "down",
            detail=None if rabbit_up else "publisher channel is closed",
        )

        all_up = all(dep.status == "up" for dep in dependencies.values())
        payload = HealthResponse(
            status="ok" if all_up else "degraded",
            service=settings.service_name,
            version=settings.version,
            environment=settings.environment,
            model_loaded=model.is_loaded,
            dependencies=dependencies,
        )
        if not all_up:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=payload.model_dump(mode="json"),
            )
        return payload


def _register_root_routes(application: FastAPI) -> None:
    @application.get("/", tags=["health"], summary="Service index", include_in_schema=False)
    async def index(request: Request) -> dict[str, Any]:
        settings: Settings = request.app.state.settings
        return {
            "service": settings.service_name,
            "version": settings.version,
            "environment": settings.environment,
            "docs": "/docs",
            "openapi": "/openapi.json",
            "endpoints": {
                "sync": "POST /api/sentiment/sync",
                "batch": "POST /api/sentiment/batch",
                "async": "POST /api/sentiment/async",
                "results": "GET /api/sentiment/results/{job_id}",
                "health": "GET /health",
                "readiness": "GET /health/ready",
            },
            "model": getattr(request.app.state, "model_service", None).stats()
            if getattr(request.app.state, "model_service", None)
            else None,
        }


app = create_app()


def main() -> None:
    """Run the API with uvicorn (used by the Docker CMD and `python -m app.main`)."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers if settings.api_workers > 1 else None,
        log_config=None,  # logging is owned by app.logging_config
        access_log=False,  # request logging is done by the timing middleware
    )


if __name__ == "__main__":
    main()
