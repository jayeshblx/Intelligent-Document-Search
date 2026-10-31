"""Pipeline message envelopes carried over SQS / Temporal between services.

A uniform envelope (``PipelineMessage``) gives every stage the same handle on
correlation id, attempt count, idempotency key, and trace context — which is what
makes retries, DLQ routing, and end-to-end tracing uniform across services.
"""

from __future__ import annotations

from pydantic import Field

from rag_core.models.base import DomainModel, utcnow
from rag_core.models.enums import DocumentStatus


class TraceContext(DomainModel):
    """W3C trace propagation across async hops (SQS doesn't carry OTel context)."""

    traceparent: str | None = None
    tracestate: str | None = None


class PipelineMessage[PayloadT: DomainModel](DomainModel):
    """Generic envelope for a pipeline event/command.

    ``idempotency_key`` lets every consumer dedupe naturally (e.g. document_id +
    target stage), so at-least-once delivery from SQS becomes effectively
    exactly-once at the application layer.
    """

    message_id: str
    correlation_id: str  # one document's journey across all stages
    stage: str           # producing/target stage name
    idempotency_key: str
    attempt: int = Field(default=1, ge=1)
    occurred_at: str = Field(default_factory=lambda: utcnow().isoformat())
    trace: TraceContext = Field(default_factory=TraceContext)
    payload: PayloadT


class DocumentEvent(DomainModel):
    """Lightweight per-document event payload moved between write-path stages."""

    document_id: str
    namespace: str = "default"
    status: DocumentStatus
    s3_uri: str | None = None
    content_checksum: str | None = None
    reason: str | None = None
