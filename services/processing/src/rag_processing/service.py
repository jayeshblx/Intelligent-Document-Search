"""ProcessingService — extract + enrich + persist + emit for one document.

Pipeline per document:
    fetch bytes (S3) -> select extractor (registry) -> extract (text/tables/layout,
    OCR fallback) -> enrich (language/PII/classification) -> persist content (store)
    -> emit PROCESSED event (pointing at the content URI) for chunking.

Idempotent: deterministic content keys mean re-processing overwrites in place; the
emitted event's idempotency key dedupes downstream.
"""

from __future__ import annotations

import time

from rag_core.aws import S3Client, SQSClient
from rag_core.config import Settings
from rag_core.logging import bind_context, get_logger
from rag_core.models import DocumentEvent, DocumentStatus, PipelineMessage
from rag_core.models.enums import DocumentClass, DocumentType
from rag_core.observability import counter, histogram
from rag_core.resilience import NonRetryableError
from rag_core.utils import infer_document_type, new_message_id
from rag_processing.config import ProcessingSettings
from rag_processing.enrich import classify_document, detect_language, detect_pii
from rag_processing.extractors import ExtractionContext, ExtractorRegistry
from rag_processing.models import ProcessResult
from rag_processing.ocr.base import OCREngine
from rag_processing.store import ContentStore

_log = get_logger("rag_processing.service")

_DOCS = counter("processing_documents_total", "Documents processed", ("doc_type", "outcome"))
_PAGES = counter("processing_pages_total", "Pages extracted")
_OCR = counter("processing_ocr_documents_total", "Documents that required OCR")
_LATENCY = histogram("processing_seconds", "Processing latency per document", ("doc_type",))


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key`` into (bucket, key)."""
    if not uri.startswith("s3://"):
        raise NonRetryableError(f"not an s3 uri: {uri!r}")
    without = uri[len("s3://") :]
    bucket, _, key = without.partition("/")
    if not bucket or not key:
        raise NonRetryableError(f"malformed s3 uri: {uri!r}")
    return bucket, key


class ProcessingService:
    def __init__(
        self,
        *,
        s3: S3Client,
        sqs: SQSClient,
        registry: ExtractorRegistry,
        ocr_engine: OCREngine,
        content_store: ContentStore,
        core_settings: Settings,
        settings: ProcessingSettings,
    ) -> None:
        self._s3 = s3
        self._sqs = sqs
        self._registry = registry
        self._ocr = ocr_engine
        self._store = content_store
        self._core = core_settings
        self._settings = settings

    async def process_event(self, event: DocumentEvent) -> ProcessResult:
        """Process a document referenced by a DISCOVERED event."""
        if not event.s3_uri:
            raise NonRetryableError("event missing s3_uri")
        bucket, key = parse_s3_uri(event.s3_uri)
        return await self.process_object(
            bucket=bucket,
            key=key,
            document_id=event.document_id,
            namespace=event.namespace,
        )

    async def process_object(
        self,
        *,
        bucket: str,
        key: str,
        document_id: str,
        namespace: str = "default",
        doc_type: DocumentType | None = None,
    ) -> ProcessResult:
        """Fetch + process a single S3 object end to end."""
        started = time.perf_counter()
        resolved_type = doc_type or infer_document_type(key)
        bind_context(document_id=document_id, doc_type=resolved_type.value, key=key)

        extractor = self._registry.get(resolved_type)
        if extractor is None:
            _DOCS.labels(doc_type=resolved_type.value, outcome="unsupported").inc()
            raise NonRetryableError(f"no extractor registered for {resolved_type.value}")

        data = await self._s3.get_bytes(bucket, key)

        ctx = ExtractionContext(
            document_id=document_id,
            key=key,
            doc_type=resolved_type,
            bucket=bucket,
            heading_max_words=self._settings.heading_max_words,
            detect_tables=self._settings.detect_tables,
            max_pages=self._settings.max_pages,
            pdf_engine=self._settings.pdf_engine,
            ocr_engine=self._ocr,
        )
        content = await extractor.extract(data, ctx)

        # --- Enrichment ---
        full_text = content.full_text
        if self._settings.detect_language:
            content.language = detect_language(full_text)
        pii = detect_pii(full_text) if self._settings.detect_pii else []
        doc_class = (
            classify_document(full_text, filename=key)
            if self._settings.classify
            else DocumentClass.OTHER
        )

        # --- Persist + emit ---
        content_uri = await self._store.put(content, namespace=namespace)
        emitted = await self._emit(document_id, namespace, content_uri)

        # --- Metrics / result ---
        _PAGES.inc(content.page_count)
        if content.has_ocr:
            _OCR.inc()
        _DOCS.labels(doc_type=resolved_type.value, outcome="ok").inc()
        _LATENCY.labels(doc_type=resolved_type.value).observe(time.perf_counter() - started)

        result = ProcessResult(
            document_id=document_id,
            namespace=namespace,
            doc_type=resolved_type,
            page_count=content.page_count,
            table_count=sum(len(p.tables) for p in content.pages),
            language=content.language,
            has_ocr=content.has_ocr,
            document_class=doc_class,
            pii_findings=sum(f.count for f in pii),
            content_uri=content_uri,
            chars_extracted=len(full_text),
            duration_s=round(time.perf_counter() - started, 3),
            emitted=emitted,
        )
        _log.info(
            "document_processed",
            pages=result.page_count,
            tables=result.table_count,
            language=result.language.value,
            has_ocr=result.has_ocr,
            doc_class=doc_class.value,
            pii=result.pii_findings,
            chars=result.chars_extracted,
            duration_s=result.duration_s,
        )
        return result

    async def _emit(self, document_id: str, namespace: str, content_uri: str) -> bool:
        """Emit a PROCESSED event whose s3_uri points at the extracted content."""
        queue = self._settings.output_queue_url
        if not queue:
            return False
        event = DocumentEvent(
            document_id=document_id,
            namespace=namespace,
            status=DocumentStatus.PROCESSED,
            s3_uri=content_uri,
        )
        message: PipelineMessage[DocumentEvent] = PipelineMessage(
            message_id=new_message_id(),
            correlation_id=document_id,
            stage="processing",
            idempotency_key=f"{document_id}:chunking",
            payload=event,
        )
        await self._sqs.send(queue, message.model_dump_json())
        return True
