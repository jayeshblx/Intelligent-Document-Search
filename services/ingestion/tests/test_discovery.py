"""Tests for type inference and Document construction."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from rag_core.aws import S3ObjectSummary
from rag_core.models import DocumentStatus, DocumentType
from rag_core.utils import document_id
from rag_ingestion.discovery import build_document, infer_document_type


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("a/b/policy.pdf", DocumentType.PDF),
        ("report.DOCX", DocumentType.DOCX),
        ("deck.pptx", DocumentType.PPTX),
        ("data.xlsx", DocumentType.XLSX),
        ("schema.xml", DocumentType.XML),
        ("scan.TIFF", DocumentType.IMAGE),
        ("notes.txt", DocumentType.TXT),
        ("page.html", DocumentType.HTML),
        ("archive.zip", DocumentType.UNKNOWN),
        ("no_extension", DocumentType.UNKNOWN),
    ],
)
def test_infer_document_type(key: str, expected: DocumentType) -> None:
    assert infer_document_type(key) == expected


def test_build_document_defaults_to_logical_id(
    make_summary: Callable[..., S3ObjectSummary],
) -> None:
    summary = make_summary("Test/policy.pdf", etag="e1", size=10)
    doc = build_document(summary, namespace="ns1")
    assert doc.id == document_id("test-bucket", "Test/policy.pdf")
    assert doc.namespace == "ns1"
    assert doc.doc_type == DocumentType.PDF
    assert doc.status == DocumentStatus.DISCOVERED
    assert doc.source.uri == "s3://test-bucket/Test/policy.pdf"


def test_build_document_uses_explicit_physical_id(
    make_summary: Callable[..., S3ObjectSummary],
) -> None:
    summary = make_summary("Test/policy.pdf", etag="e1", size=10)
    doc = build_document(
        summary, doc_id="phys-123", version=2, checksum="abc", supersedes_id="phys-122"
    )
    assert doc.id == "phys-123"
    assert doc.version == 2
    assert doc.content_checksum == "abc"
    assert doc.supersedes_id == "phys-122"
