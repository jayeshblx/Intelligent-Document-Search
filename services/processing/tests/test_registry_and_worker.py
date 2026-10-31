"""Tests for the extractor registry and the SQS worker's message handling."""

from __future__ import annotations

from rag_core.config import Settings, SQSSettings
from rag_core.models import DocumentEvent, DocumentStatus, PipelineMessage
from rag_core.models.enums import DocumentType
from rag_processing.config import ProcessingSettings
from rag_processing.extractors import build_default_registry
from rag_processing.worker import ProcessingWorker


def test_default_registry_covers_all_supported_types() -> None:
    reg = build_default_registry()
    for dt in (
        DocumentType.PDF,
        DocumentType.DOCX,
        DocumentType.PPTX,
        DocumentType.XLSX,
        DocumentType.XML,
        DocumentType.HTML,
        DocumentType.TXT,
        DocumentType.IMAGE,
    ):
        assert reg.get(dt) is not None, dt
    assert reg.get(DocumentType.UNKNOWN) is None


# ---- Worker handling -------------------------------------------------------
class _FakeSQS:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.dlq: list[str] = []

    async def delete(self, queue_url: str, receipt_handle: str) -> None:
        self.deleted.append(receipt_handle)

    async def send(self, queue_url: str, body: str, *, attributes: dict | None = None) -> str:
        self.dlq.append(body)
        return "x"


class _Msg:
    def __init__(self, body: str, receipt: str = "r1", receive_count: int = 1) -> None:
        self.body = body
        self.receipt_handle = receipt
        self.receive_count = receive_count


class _StubService:
    def __init__(self) -> None:
        self.processed: list[str] = []

    async def process_event(self, event: DocumentEvent) -> None:
        self.processed.append(event.document_id)


def _settings() -> tuple[Settings, ProcessingSettings]:
    core = Settings(
        sqs=SQSSettings(
            ingestion_queue_url="http://localhost/in", ingestion_dlq_url="http://localhost/dlq"
        )
    )
    return core, ProcessingSettings(run_worker=False)


def _make_worker(sqs: _FakeSQS, service: _StubService) -> ProcessingWorker:
    core, ps = _settings()
    return ProcessingWorker(sqs=sqs, service=service, core_settings=core, settings=ps)  # type: ignore[arg-type]


async def test_worker_handles_valid_message() -> None:
    sqs, service = _FakeSQS(), _StubService()
    worker = _make_worker(sqs, service)
    event = DocumentEvent(
        document_id="d1", status=DocumentStatus.DISCOVERED, s3_uri="s3://b/k.txt"
    )
    envelope: PipelineMessage[DocumentEvent] = PipelineMessage(
        message_id="m1", correlation_id="d1", stage="ingestion", idempotency_key="d1:p", payload=event
    )
    await worker._handle(_Msg(envelope.model_dump_json()))
    assert service.processed == ["d1"]
    assert sqs.deleted == ["r1"]  # acked after success


async def test_worker_routes_poison_to_dlq() -> None:
    sqs, service = _FakeSQS(), _StubService()
    worker = _make_worker(sqs, service)
    await worker._handle(_Msg("not-json"))
    assert sqs.dlq  # unparseable message sent to DLQ
    assert sqs.deleted == ["r1"]  # and removed from the main queue
    assert service.processed == []
