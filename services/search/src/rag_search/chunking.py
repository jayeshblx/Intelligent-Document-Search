"""Parent-child chunking (platform capability #3).

Why parent-child: we search *small* child chunks (precise semantic/keyword match)
but return the *parent* section as context so the answer/citation has enough
surrounding text to be trustworthy. Children carry ``parent_id`` so retrieval can
expand to the parent when needed.

Sectioning is heading-aware: a new heading block starts a new section (parent),
so chunks respect document structure and each carries a ``section_path`` and page
range for citations. Tables are emitted as their own child (their markdown is
highly searchable and shouldn't be split mid-row).
"""

from __future__ import annotations

from rag_core.models.chunk import Chunk
from rag_core.models.content import BlockType, ExtractedContent
from rag_core.models.enums import ChunkStrategy
from rag_core.utils import chunk_id


class Chunker:
    def __init__(
        self, *, parent_max_words: int = 350, child_words: int = 180, child_overlap: int = 40
    ) -> None:
        self._parent_max = parent_max_words
        self._child_words = child_words
        self._overlap = min(child_overlap, child_words - 1)

    def chunk(
        self,
        content: ExtractedContent,
        *,
        namespace: str = "default",
        base_payload: dict[str, object] | None = None,
    ) -> list[Chunk]:
        """Produce parent + child chunks for one document."""
        sections = self._sections(content)
        payload = base_payload or {}
        chunks: list[Chunk] = []
        parent_idx = 0
        child_idx = 0

        for section in sections:
            text = section["text"].strip()
            if not text:
                continue
            heading = section["heading"]
            section_path = [heading] if heading else []
            page_start = section["page_start"]
            page_end = section["page_end"]

            parent_id = chunk_id(content.document_id, ChunkStrategy.PARENT, parent_idx)
            parent = Chunk(
                id=parent_id,
                document_id=content.document_id,
                namespace=namespace,
                is_parent=True,
                strategy=ChunkStrategy.PARENT,
                order_index=parent_idx,
                text=text,
                page_start=page_start,
                page_end=page_end,
                section_path=section_path,
                heading=heading,
                token_count=len(text.split()),
                metadata_payload=payload,
            )
            chunks.append(parent)
            parent_idx += 1

            for window in self._windows(text):
                child = Chunk(
                    id=chunk_id(content.document_id, ChunkStrategy.CHILD, child_idx),
                    document_id=content.document_id,
                    namespace=namespace,
                    parent_id=parent_id,
                    is_parent=False,
                    strategy=ChunkStrategy.CHILD,
                    order_index=child_idx,
                    text=window,
                    page_start=page_start,
                    page_end=page_end,
                    section_path=section_path,
                    heading=heading,
                    token_count=len(window.split()),
                    metadata_payload=payload,
                )
                chunks.append(child)
                child_idx += 1

        return chunks

    def _sections(self, content: ExtractedContent) -> list[dict]:
        """Group blocks + tables into heading-delimited sections across pages."""
        sections: list[dict] = []
        current: dict | None = None

        def start(heading: str | None, page: int) -> dict:
            return {"heading": heading, "text": "", "page_start": page, "page_end": page}

        for page in content.pages:
            # Interleave blocks and tables roughly in page order (blocks first,
            # then tables — extractors already order blocks; tables are appended).
            for block in page.blocks:
                if block.role == BlockType.HEADING:
                    if current and current["text"].strip():
                        sections.append(current)
                    current = start(block.text, page.number)
                    continue
                if current is None:
                    current = start(None, page.number)
                current["text"] += block.text + "\n"
                current["page_end"] = page.number
                # Split overly long sections into a new parent to bound size.
                if len(current["text"].split()) >= self._parent_max:
                    sections.append(current)
                    current = start(current["heading"], page.number)
            for table in page.tables:
                if current is None:
                    current = start(None, page.number)
                current["text"] += table.markdown + "\n"
                current["page_end"] = page.number

        if current and current["text"].strip():
            sections.append(current)

        # A document with no blocks (e.g. empty OCR) still yields nothing; caller
        # handles the empty case.
        return sections

    def _windows(self, text: str) -> list[str]:
        """Sliding word windows over a section's text (child chunks)."""
        words = text.split()
        if len(words) <= self._child_words:
            return [text]
        step = self._child_words - self._overlap
        windows: list[str] = []
        for start in range(0, len(words), step):
            window = words[start : start + self._child_words]
            if not window:
                break
            windows.append(" ".join(window))
            if start + self._child_words >= len(words):
                break
        return windows
