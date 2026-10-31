"""FastAPI application for the ingestion service.

Exposes: a health/readiness/metrics surface (for K8s probes + Prometheus) and a
sync trigger. The app owns resource lifecycle via the lifespan context: build the
DI container + ledger + service on startup, tear them down on shutdown.

Note on scale: the synchronous ``/v1/sync`` endpoint is right for incremental
scans and tests. Full 1M-document backfills are driven by a Temporal workflow
(durable, resumable) that calls the same ``IngestionService`` — see
``docs/ROADMAP.md``. The business logic is identical; only the trigger differs.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, Query, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from rag_core.config import get_settings
from rag_core.di import Container
from rag_core.logging import configure_logging, get_logger
from rag_core.observability import REGISTRY, setup_tracing
from rag_ingestion.config import IngestionSettings
from rag_ingestion.dedup import DedupLedger, RedisDedupLedger
from rag_ingestion.dependencies import get_ledger, get_service
from rag_ingestion.models import SyncRequest, SyncSummary
from rag_ingestion.service import IngestionService

_log = get_logger("rag_ingestion.app")
SERVICE_NAME = "ingestion"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build and tear down shared resources."""
    settings = get_settings()
    configure_logging(
        level=settings.log_level, json_logs=settings.log_json, service_name=SERVICE_NAME
    )
    setup_tracing(settings.otel, service_name=SERVICE_NAME)

    ingestion_settings = IngestionSettings()
    container = Container(settings)
    ledger: DedupLedger = RedisDedupLedger(settings.redis.url)

    service = IngestionService(
        s3=container.s3(),
        sqs=container.sqs(),
        ledger=ledger,
        core_settings=settings,
        settings=ingestion_settings,
    )

    app.state.settings = settings
    app.state.container = container
    app.state.ledger = ledger
    app.state.service = service
    app.state.sync_lock = asyncio.Lock()

    _log.info("ingestion_startup_complete", bucket=settings.s3.bucket, prefix=settings.s3.prefix)
    try:
        yield
    finally:
        await container.aclose()
        await ledger.aclose()
        _log.info("ingestion_shutdown_complete")


# ---- Routers ---------------------------------------------------------------
ops = APIRouter(tags=["ops"])


@ops.get("/", summary="Service info")
async def root() -> dict[str, str]:
    return {"service": SERVICE_NAME, "status": "ok"}


@ops.get("/healthz", summary="Liveness probe")
async def healthz() -> dict[str, str]:
    """Liveness: process is up. Does NOT check dependencies (K8s would restart
    the pod for a transient Redis blip, which is wrong)."""
    return {"status": "alive"}


@ops.get("/readyz", summary="Readiness probe")
async def readyz(response: Response, ledger: DedupLedger = Depends(get_ledger)) -> dict[str, object]:
    """Readiness: can we actually serve? Checks the dedup ledger. Returns 503 if
    not ready so K8s pulls the pod from the load-balancer rotation."""
    ledger_ok = await ledger.health()
    ready = ledger_ok
    if not ready:
        response.status_code = 503
    return {"ready": ready, "checks": {"ledger": ledger_ok}}


@ops.get("/metrics", summary="Prometheus metrics")
async def metrics() -> Response:
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


v1 = APIRouter(prefix="/v1", tags=["ingestion"])


@v1.post("/sync", response_model=SyncSummary, summary="Trigger an incremental S3 sync")
async def sync(
    body: SyncRequest,
    request: Request,
    response: Response,
    wait: bool = Query(default=True, description="Run synchronously and return the summary."),
    service: IngestionService = Depends(get_service),
) -> SyncSummary:
    """Run an incremental sync over the configured (or requested) S3 prefix.

    A process-wide lock prevents overlapping syncs from double-emitting work.
    """
    lock: asyncio.Lock = request.app.state.sync_lock
    if lock.locked() and not wait:
        response.status_code = 409  # a sync is already running
        s3_cfg = request.app.state.settings.s3
        return SyncSummary(
            bucket=s3_cfg.bucket,
            prefix=body.prefix or s3_cfg.prefix,
            namespace=body.namespace,
            dry_run=body.dry_run,
            error_samples=["a sync is already in progress"],
        )
    async with lock:
        summary = await service.sync(body)
    if summary.errors:
        response.status_code = 207  # multi-status: completed with per-object errors
    return summary


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="RAG Ingestion Service",
        version="0.1.0",
        description="S3 incremental sync, deduplication, version detection, SQS emit.",
        lifespan=lifespan,
    )
    app.include_router(ops)
    app.include_router(v1)
    return app


app = create_app()
