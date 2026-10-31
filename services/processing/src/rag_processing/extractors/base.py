"""Extractor interface + per-call context.

An extractor converts raw bytes into ``ExtractedContent``. Heavy parsing is CPU
bound; extractors run it via ``asyncio.to_thread`` so they don't block the event
loop while the worker handles other documents concurrently. Extractors that need
OCR (PDF scanned pages, images) reach it through ``ExtractionContext.ocr_engine``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_core.models.content import ExtractedContent
from rag_core.models.enums import DocumentType

if TYPE_CHECKING:
    from rag_processing.ocr.base import OCREngine


@dataclass(slots=True)
class ExtractionContext:
    """Inputs an extractor needs beyond the raw bytes."""

    document_id: str
    key: str
    doc_type: DocumentType
    bucket: str | None = None
    heading_max_words: int = 12
    detect_tables: bool = True
    max_pages: int = 0  # 0 = no cap
    pdf_engine: str = "pymupdf"  # pymupdf (markdown) | pypdf (fast text) | pdfplumber (rich)
    ocr_engine: OCREngine | None = None


class Extractor(ABC):
    """Strategy interface for one or more document formats."""

    #: Formats this extractor handles (used by the registry).
    supported: tuple[DocumentType, ...] = ()

    @abstractmethod
    async def extract(self, data: bytes, ctx: ExtractionContext) -> ExtractedContent:
        """Parse ``data`` into structured content. Must not raise on empty input —
        return an empty ``ExtractedContent`` so the pipeline records a result."""
