"""FastAPI dependency providers.

Resources are built once in the app lifespan and stashed on ``app.state``; these
providers expose them to handlers. Tests override them via
``app.dependency_overrides`` to inject fakes — no real S3/SQS/Redis needed.
"""

from __future__ import annotations

from fastapi import Request

from rag_ingestion.dedup import DedupLedger
from rag_ingestion.service import IngestionService


def get_service(request: Request) -> IngestionService:
    """Return the singleton ingestion service from app state."""
    return request.app.state.service


def get_ledger(request: Request) -> DedupLedger:
    """Return the dedup ledger (used by readiness checks)."""
    return request.app.state.ledger
