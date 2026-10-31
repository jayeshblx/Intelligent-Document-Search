"""Source discovery helpers: type inference and Document construction.

Type inference is by extension here (cheap, runs over millions of keys). True
format detection (e.g. is a .pdf actually scanned/image-only?) is deferred to the
processing service, which has the bytes in hand. Keeping this layer cheap is what
makes scanning a 1M-object prefix affordable.
"""

from __future__ import annotations

from rag_core.aws import S3ObjectSummary
from rag_core.models import Document, DocumentStatus, SourceRef
from rag_core.utils import document_id, infer_document_type

# Re-exported so callers/tests can import it from the ingestion namespace; the
# canonical implementation lives in rag_core (shared with the processing service).
__all__ = ["build_document", "infer_document_type"]


def build_document(
    summary: S3ObjectSummary,
    *,
    doc_id: str | None = None,
    namespace: str = "default",
    version: int = 1,
    checksum: str | None = None,
    supersedes_id: str | None = None,
) -> Document:
    """Construct a ``Document`` record from an S3 listing entry.

    ``doc_id`` is the *physical* (per-version) id; when omitted it falls back to
    the logical key-derived id. The caller (ingestion service) passes a
    version-aware id so each content version is a distinct, independently
    indexable document while the logical id ties the version chain together.
    """
    resolved_id = doc_id or document_id(summary.bucket, summary.key)
    return Document(
        id=resolved_id,
        source=SourceRef(
            bucket=summary.bucket,
            key=summary.key,
            etag=summary.etag,
            size_bytes=summary.size_bytes,
            last_modified=summary.last_modified,
            content_type=None,
        ),
        doc_type=infer_document_type(summary.key),
        status=DocumentStatus.DISCOVERED,
        content_checksum=checksum,
        version=version,
        is_latest=True,
        supersedes_id=supersedes_id,
        namespace=namespace,
    )
