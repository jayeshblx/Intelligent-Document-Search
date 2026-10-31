"""OCR engine interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_core.models.content import Page


class OCREngine(ABC):
    """Converts image bytes / scanned PDFs into structured ``Page`` objects.

    Returning the shared ``Page`` model (text + tables + blocks) means OCR output
    composes directly into ``ExtractedContent`` alongside native-text extraction.
    """

    name: str = "base"

    @abstractmethod
    async def ocr_image(self, data: bytes) -> Page:
        """OCR a single image into one page."""

    @abstractmethod
    async def ocr_pdf(self, *, bucket: str, key: str, max_pages: int = 200) -> list[Page]:
        """OCR a scanned PDF (located in S3) into pages."""

    async def ocr_pdf_bytes(self, data: bytes, *, max_pages: int = 200) -> list[Page]:
        """OCR a scanned PDF from in-memory bytes. Engines that can rasterise
        locally (e.g. Tesseract via PyMuPDF) override this; the default returns
        nothing so the caller falls back to the S3-based ``ocr_pdf``."""
        return []

    async def health(self) -> bool:  # pragma: no cover - overridden where meaningful
        return True

    async def aclose(self) -> None:  # pragma: no cover - optional
        return None
