"""API DTOs for the processing service."""

from __future__ import annotations

from pydantic import Field

from rag_core.models.base import DomainModel
from rag_core.models.enums import DocumentClass, DocumentType, Language


class ProcessRequest(DomainModel):
    """Process a single document on demand (testing / replay).

    Either provide ``s3_uri`` or ``bucket`` + ``key``. ``document_id`` is optional;
    when omitted it's derived deterministically from the source location.
    """

    s3_uri: str | None = None
    bucket: str | None = None
    key: str | None = None
    document_id: str | None = None
    namespace: str = "default"
    doc_type: DocumentType | None = None


class ProcessResult(DomainModel):
    """Outcome of processing one document."""

    document_id: str
    namespace: str
    doc_type: DocumentType
    page_count: int = 0
    table_count: int = 0
    language: Language = Language.UNKNOWN
    has_ocr: bool = False
    document_class: DocumentClass = DocumentClass.OTHER
    pii_findings: int = 0
    content_uri: str | None = None
    chars_extracted: int = 0
    duration_s: float = 0.0
    emitted: bool = False
    error: str | None = Field(default=None)
