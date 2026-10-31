"""FastAPI dependency providers for the processing service."""

from __future__ import annotations

from fastapi import Request

from rag_processing.service import ProcessingService
from rag_processing.store import ContentStore


def get_service(request: Request) -> ProcessingService:
    return request.app.state.service


def get_content_store(request: Request) -> ContentStore:
    return request.app.state.content_store
