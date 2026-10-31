"""Tests for text/HTML/XML extractors (byte inputs, no files needed)."""

from __future__ import annotations

import pytest

from rag_core.models.content import BlockType
from rag_core.models.enums import DocumentType
from rag_processing.extractors.base import ExtractionContext
from rag_processing.extractors.text import HtmlExtractor, TextExtractor, XmlExtractor


def _ctx(doc_type: DocumentType) -> ExtractionContext:
    return ExtractionContext(document_id="d1", key=f"x.{doc_type.value}", doc_type=doc_type)


async def test_text_extractor_basic() -> None:
    content = await TextExtractor().extract(b"line one\nline two", _ctx(DocumentType.TXT))
    assert content.document_id == "d1"
    assert content.page_count >= 1
    assert "line one" in content.full_text


async def test_html_extractor_strips_markup_and_marks_headings() -> None:
    html = b"<html><head><style>x{}</style></head><body><h1>Safety Policy</h1><p>Wear gloves.</p><script>1</script></body></html>"
    content = await HtmlExtractor().extract(html, _ctx(DocumentType.HTML))
    text = content.full_text
    assert "Safety Policy" in text
    assert "Wear gloves." in text
    assert "x{}" not in text  # style stripped
    headings = [b for p in content.pages for b in p.blocks if b.role == BlockType.HEADING]
    assert any(h.text == "Safety Policy" for h in headings)


async def test_xml_extractor_extracts_text_nodes() -> None:
    xml = b"<doc><title>Spec</title><body>Requirement A</body></doc>"
    content = await XmlExtractor().extract(xml, _ctx(DocumentType.XML))
    assert "Spec" in content.full_text
    assert "Requirement A" in content.full_text


async def test_xml_extractor_recovers_from_malformed() -> None:
    # Unclosed tag should not raise; recover mode salvages text.
    content = await XmlExtractor().extract(b"<doc><a>hello</doc>", _ctx(DocumentType.XML))
    assert "hello" in content.full_text


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16", "latin-1"])
async def test_text_extractor_handles_encodings(encoding: str) -> None:
    content = await TextExtractor().extract("café déjà".encode(encoding), _ctx(DocumentType.TXT))
    assert content.full_text.strip() != ""
