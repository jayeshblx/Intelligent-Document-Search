"""No-op OCR engine — graceful degradation when OCR is disabled.

Scanned content is flagged (``is_scanned=True``) with empty text rather than
crashing the pipeline, so the document still flows through and is visible as
"needs OCR" in metrics/review.
"""

from __future__ import annotations

from rag_core.logging import get_logger
from rag_core.models.content import Page
from rag_processing.ocr.base import OCREngine

_log = get_logger("rag_processing.ocr.noop")


class NoOpOCREngine(OCREngine):
    name = "none"

    async def ocr_image(self, data: bytes) -> Page:
        _log.warning("ocr_disabled_image")
        return Page(number=1, text="", is_scanned=True)

    async def ocr_pdf(self, *, bucket: str, key: str, max_pages: int = 200) -> list[Page]:
        _log.warning("ocr_disabled_pdf", key=key)
        return []
