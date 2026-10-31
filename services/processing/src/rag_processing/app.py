"""FastAPI application for the processing service.

Owns the resource lifecycle (S3/SQS clients, OCR engine, extractor registry,
content store, service) and optionally runs the SQS worker in-process. Exposes
health/readiness/metrics plus an on-demand ``/v1/process`` for replay/testing.

Deployment note: in production you typically run the *worker* as its own
horizontally-scaled deployment (PROCESSING_RUN_WORKER=true, API disabled) and a
small *API* deployment for health/replay (PROCESSING_RUN_WORKER=false). Same image.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from rag_core.config import get_settings
from rag_core.di import Container
from rag_core.logging import configure_logging, get_logger
from rag_core.observability import REGISTRY, setup_tracing
from rag_core.resilience import NonRetryableError
from rag_core.utils import document_id
from rag_processing.config import ProcessingSettings
from rag_processing.dependencies import get_content_store, get_service
from rag_processing.extractors import build_default_registry
from rag_processing.models import ProcessRequest, ProcessResult
from rag_processing.ocr import build_ocr_engine
from rag_processing.service import ProcessingService
from rag_processing.store import ContentStore, S3ContentStore
from rag_processing.worker import ProcessingWorker

_log = get_logger("rag_processing.app")
SERVICE_NAME = "processing"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(
        level=settings.log_level, json_logs=settings.log_json, service_name=SERVICE_NAME
    )
    setup_tracing(settings.otel, service_name=SERVICE_NAME)

    ps = ProcessingSettings()
    container = Container(settings)
    ocr_engine = build_ocr_engine(ps, container.aws_factory())
    registry = build_default_registry()
    content_store: ContentStore = S3ContentStore(
        container.s3(), bucket=settings.s3.bucket, prefix=ps.content_prefix
    )
    service = ProcessingService(
        s3=container.s3(),
        sqs=container.sqs(),
        registry=registry,
        ocr_engine=ocr_engine,
        content_store=content_store,
        core_settings=settings,
        settings=ps,
    )

    app.state.settings = settings
    app.state.container = container
    app.state.ocr_engine = ocr_engine
    app.state.content_store = content_store
    app.state.service = service

    worker_task: asyncio.Task[None] | None = None
    worker: ProcessingWorker | None = None
    if ps.run_worker and settings.sqs.ingestion_queue_url:
        worker = ProcessingWorker(
            sqs=container.sqs(), service=service, core_settings=settings, settings=ps
        )
        worker_task = asyncio.create_task(worker.run(), name="processing-worker")
    app.state.worker = worker

    _log.info(
        "processing_startup_complete",
        supported=[t.value for t in registry.supported_types],
        ocr=ocr_engine.name,
        worker=bool(worker_task),
    )
    try:
        yield
    finally:
        if worker is not None:
            await worker.stop()
        if worker_task is not None:
            worker_task.cancel()
            await asyncio.gather(worker_task, return_exceptions=True)
        await ocr_engine.aclose()
        await container.aclose()
        _log.info("processing_shutdown_complete")


ops = APIRouter(tags=["ops"])


@ops.get("/", summary="Service info")
async def root() -> dict[str, str]:
    return {"service": SERVICE_NAME, "status": "ok"}


@ops.get("/healthz", summary="Liveness probe")
async def healthz() -> dict[str, str]:
    return {"status": "alive"}


@ops.get("/readyz", summary="Readiness probe")
async def readyz(
    response: Response, store: ContentStore = Depends(get_content_store)
) -> dict[str, object]:
    store_ok = await store.health()
    if not store_ok:
        response.status_code = 503
    return {"ready": store_ok, "checks": {"content_store": store_ok}}


@ops.get("/metrics", summary="Prometheus metrics")
async def metrics() -> Response:
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


v1 = APIRouter(prefix="/v1", tags=["processing"])


@v1.post("/process", response_model=ProcessResult, summary="Process one document on demand")
async def process(
    body: ProcessRequest, service: ProcessingService = Depends(get_service)
) -> ProcessResult:
    """Process a single document (replay/testing). Provide ``s3_uri`` or
    ``bucket`` + ``key``."""
    if body.s3_uri:
        from rag_processing.service import parse_s3_uri

        bucket, key = parse_s3_uri(body.s3_uri)
    elif body.bucket and body.key:
        bucket, key = body.bucket, body.key
    else:
        raise HTTPException(status_code=422, detail="provide s3_uri or bucket+key")

    doc_id = body.document_id or document_id(bucket, key)
    try:
        return await service.process_object(
            bucket=bucket,
            key=key,
            document_id=doc_id,
            namespace=body.namespace,
            doc_type=body.doc_type,
        )
    except NonRetryableError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Processing Service",
        version="0.1.0",
        description="Text/OCR/tables/layout extraction into structured ExtractedContent.",
        lifespan=lifespan,
    )
    app.include_router(ops)
    app.include_router(v1)
    return app


app = create_app()
