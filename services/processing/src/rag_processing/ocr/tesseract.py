"""Local Tesseract OCR engine (dev/offline).

Requires the system ``tesseract`` binary + ``pytesseract``. Scanned-PDF OCR
rasterises pages with **PyMuPDF** (no poppler needed), then OCRs each page image
with Tesseract. ``pytesseract`` and ``pymupdf`` are imported lazily so they stay
optional when another provider is configured.
"""

from __future__ import annotations

import asyncio
import os
from io import BytesIO

from rag_core.logging import get_logger
from rag_core.models.content import BlockType, Page, TextBlock
from rag_processing.ocr.base import OCREngine

_log = get_logger("rag_processing.ocr.tesseract")

# Common Windows install location (UB-Mannheim build) used when not on PATH.
_WINDOWS_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class TesseractOCREngine(OCREngine):
    name = "tesseract"

    def __init__(self, *, tesseract_cmd: str = "", dpi: int = 150, page_cap: int = 15) -> None:
        self._dpi = dpi
        self._page_cap = page_cap  # OCR is ~1-2s/page; cap to keep per-doc time bounded
        # Resolve the binary: explicit path > known Windows path > rely on PATH.
        self._cmd = tesseract_cmd or (_WINDOWS_DEFAULT if os.path.exists(_WINDOWS_DEFAULT) else "")

    def _configure(self) -> None:
        import pytesseract

        if self._cmd:
            pytesseract.pytesseract.tesseract_cmd = self._cmd

    async def ocr_image(self, data: bytes) -> Page:
        def _run() -> str:
            import pytesseract
            from PIL import Image

            self._configure()
            return pytesseract.image_to_string(Image.open(BytesIO(data)))

        text = await asyncio.to_thread(_run)
        blocks = [
            TextBlock(text=line, role=BlockType.PARAGRAPH, page=1, from_ocr=True)
            for line in text.splitlines()
            if line.strip()
        ]
        return Page(number=1, text=text, blocks=blocks, is_scanned=True)

    async def ocr_pdf_bytes(self, data: bytes, *, max_pages: int = 200) -> list[Page]:
        return await asyncio.to_thread(self._ocr_pdf_sync, data, max_pages)

    def _ocr_pdf_sync(self, data: bytes, max_pages: int) -> list[Page]:
        import pymupdf
        import pytesseract
        from PIL import Image

        self._configure()
        pages: list[Page] = []
        doc = pymupdf.open(stream=data, filetype="pdf")
        try:
            limit = min(max_pages or doc.page_count, doc.page_count, self._page_cap)
            for i in range(limit):
                pix = doc.load_page(i).get_pixmap(dpi=self._dpi)
                img = Image.open(BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img)
                blocks = [
                    TextBlock(text=line, role=BlockType.PARAGRAPH, page=i + 1, from_ocr=True)
                    for line in text.splitlines()
                    if line.strip()
                ]
                pages.append(Page(number=i + 1, text=text, blocks=blocks, is_scanned=True))
        finally:
            doc.close()
        _log.info("tesseract_pdf_ocr", pages=len(pages))
        return pages

    async def ocr_pdf(self, *, bucket: str, key: str, max_pages: int = 200) -> list[Page]:
        # PDF OCR goes through ocr_pdf_bytes (the extractor already has the bytes);
        # nothing to do from an S3 reference alone.
        return []
