"""Tests for ProcessingService orchestration."""

from __future__ import annotations

from collections.abc import Callable
from io import BytesIO

import pytest

from rag_core.models import DocumentEvent, DocumentStatus, PipelineMessage
from rag_core.models.enums import DocumentType, Language
from rag_core.resilience import NonRetryableError
from rag_processing.extractors.pdf import PdfExtractor
from rag_processing.extractors.registry import ExtractorRegistry
from rag_processing.extractors.text import TextExtractor
from rag_processing.service import ProcessingService, parse_s3_uri

BuildService = Callable[..., tuple[ProcessingService, object, object]]


def _txt_registry() -> ExtractorRegistry:
    reg = ExtractorRegistry()
    reg.register(TextExtractor())
    return reg


def _blank_pdf() -> bytes:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_parse_s3_uri() -> None:
    assert parse_s3_uri("s3://bucket/a/b.pdf") == ("bucket", "a/b.pdf")
    with pytest.raises(NonRetryableError):
        parse_s3_uri("https://bucket/x")


async def test_process_txt_end_to_end(build_service: BuildService) -> None:
    body = b"Information Security Policy. Contact admin@example.com for access."
    service, _, sqs = build_service(
        objects={"Test/policy.txt": body}, registry=_txt_registry()
    )

    result = await service.process_object(
        bucket="test-bucket", key="Test/policy.txt", document_id="doc-1", namespace="default"
    )

    assert result.doc_type == DocumentType.TXT
    assert result.page_count >= 1
    assert result.chars_extracted > 0
    assert result.language == Language.EN
    assert result.pii_findings >= 1  # the email
    assert result.content_uri is not None
    assert result.emitted is True

    # A PROCESSED event was emitted pointing at the content URI.
    assert len(sqs.sent) == 1
    emitted = PipelineMessage[DocumentEvent].model_validate_json(sqs.sent[0])
    assert emitted.payload.status == DocumentStatus.PROCESSED
    assert emitted.payload.s3_uri == result.content_uri
    assert emitted.stage == "processing"


async def test_process_event_parses_uri(build_service: BuildService) -> None:
    service, _, _ = build_service(
        objects={"Test/a.txt": b"hello world content"}, registry=_txt_registry()
    )
    event = DocumentEvent(
        document_id="doc-1",
        namespace="default",
        status=DocumentStatus.DISCOVERED,
        s3_uri="s3://test-bucket/Test/a.txt",
    )
    result = await service.process_event(event)
    assert result.document_id == "doc-1"
    assert result.page_count >= 1


async def test_unsupported_type_raises(build_service: BuildService) -> None:
    service, _, _ = build_service(objects={"Test/a.txt": b"x"}, registry=ExtractorRegistry())
    with pytest.raises(NonRetryableError):
        await service.process_object(
            bucket="test-bucket", key="Test/a.txt", document_id="d", doc_type=DocumentType.TXT
        )


async def test_scanned_pdf_falls_back_to_ocr(build_service: BuildService) -> None:
    reg = ExtractorRegistry()
    reg.register(PdfExtractor())
    service, _, _ = build_service(objects={"Test/scan.pdf": _blank_pdf()}, registry=reg)

    result = await service.process_object(
        bucket="test-bucket", key="Test/scan.pdf", document_id="doc-pdf", doc_type=DocumentType.PDF
    )
    # Blank page -> no native text -> OCR fallback (FakeOCREngine) kicks in.
    assert result.has_ocr is True
    assert result.page_count == 1
