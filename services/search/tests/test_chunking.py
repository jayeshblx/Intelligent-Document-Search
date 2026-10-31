"""Tests for the parent-child chunker."""

from __future__ import annotations

from rag_search.chunking import Chunker

from rag_core.models.content import BlockType, ExtractedContent, Page, TextBlock
from rag_core.models.enums import ChunkStrategy


def _content() -> ExtractedContent:
    body = " ".join(f"word{i}" for i in range(100))
    return ExtractedContent(
        document_id="doc-1",
        pages=[
            Page(
                number=1,
                text="",
                blocks=[
                    TextBlock(text="Section One", role=BlockType.HEADING, page=1, heading_level=1),
                    TextBlock(text=body, role=BlockType.PARAGRAPH, page=1),
                ],
            ),
            Page(
                number=2,
                text="",
                blocks=[
                    TextBlock(text="Section Two", role=BlockType.HEADING, page=2, heading_level=1),
                    TextBlock(text="short paragraph here", role=BlockType.PARAGRAPH, page=2),
                ],
            ),
        ],
    )


def test_chunker_produces_parents_and_children() -> None:
    chunks = Chunker(child_words=30, child_overlap=5).chunk(_content(), namespace="ns")
    parents = [c for c in chunks if c.is_parent]
    children = [c for c in chunks if not c.is_parent]

    assert len(parents) >= 2  # one per heading section
    assert children, "expected child chunks"
    # Every child references a real parent.
    parent_ids = {p.id for p in parents}
    assert all(c.parent_id in parent_ids for c in children)
    # Provenance carried through.
    assert any(c.section_path == ["Section One"] for c in children)
    assert all(c.strategy == ChunkStrategy.CHILD for c in children)
    assert all(c.namespace == "ns" for c in chunks)


def test_child_windows_overlap() -> None:
    chunker = Chunker(child_words=30, child_overlap=10)
    chunks = chunker.chunk(_content())
    section_one_children = [
        c for c in chunks if not c.is_parent and c.section_path == ["Section One"]
    ]
    # 100 words, window 30, step 20 -> multiple overlapping windows.
    assert len(section_one_children) >= 3


def test_base_payload_propagates() -> None:
    chunks = Chunker().chunk(_content(), base_payload={"source_key": "foxtrot/a.docx", "language": "en"})
    assert all(c.metadata_payload.get("source_key") == "foxtrot/a.docx" for c in chunks)
