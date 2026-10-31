"""Document identity, source provenance, and lifecycle state.

A ``Document`` is the unit the write path tracks from S3 discovery to indexed.
It is intentionally separate from its extracted *content* (``content.py``) and its
*metadata* (``metadata.py``): those are produced by different services and updated
on different cadences. The ``Document`` record is the durable spine they hang off.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from rag_core.models.base import TimestampedModel, utcnow
from rag_core.models.enums import DocumentStatus, DocumentType, Language


class SourceRef(TimestampedModel):
    """Where a document came from. For S3 this captures everything needed to
    re-fetch the exact bytes and to detect change (etag/version/size)."""

    bucket: str
    key: str
    version_id: str | None = None  # S3 object version (if bucket versioning on)
    etag: str | None = None        # S3 ETag — cheap change signal
    size_bytes: int = Field(default=0, ge=0)
    last_modified: datetime | None = None
    content_type: str | None = None

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


class Document(TimestampedModel):
    """The tracked document record.

    ``id`` is deterministic (UUIDv5 of bucket+key) so the same object always maps
    to the same document, making the whole write path idempotent on re-runs.
    ``content_checksum`` (sha256 of bytes) is the change/dedup signal: two objects
    with the same checksum are byte-identical regardless of key.
    """

    id: str
    source: SourceRef
    doc_type: DocumentType = DocumentType.UNKNOWN
    status: DocumentStatus = DocumentStatus.DISCOVERED

    # Change detection & dedup
    content_checksum: str | None = None  # sha256 hex of raw bytes

    # Versioning — monotonic per logical document (same source key)
    version: int = Field(default=1, ge=1)
    is_latest: bool = True
    supersedes_id: str | None = None  # previous Document.id this version replaces

    # Namespace isolation (e.g. per-tenant / per-business-unit Qdrant collection)
    namespace: str = "default"

    # Content signals (populated by processing)
    page_count: int = Field(default=0, ge=0)
    language: Language = Language.UNKNOWN

    # Failure bookkeeping
    attempts: int = Field(default=0, ge=0)
    last_error: str | None = None

    def mark(self, status: DocumentStatus, *, error: str | None = None) -> None:
        """Transition lifecycle state, stamping the update and any error."""
        self.status = status
        if error is not None:
            self.last_error = error
        if status in (DocumentStatus.FAILED, DocumentStatus.PROCESSING):
            self.attempts += 1
        self.touch()

    def as_new_version(self, *, new_id: str, checksum: str) -> Document:
        """Produce the next version of this document (content changed in S3)."""
        return Document(
            id=new_id,
            source=self.source,
            doc_type=self.doc_type,
            status=DocumentStatus.DISCOVERED,
            content_checksum=checksum,
            version=self.version + 1,
            is_latest=True,
            supersedes_id=self.id,
            namespace=self.namespace,
            created_at=utcnow(),
        )
