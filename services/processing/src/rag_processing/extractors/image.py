"""Image extractor — routes images (PNG/JPG/TIFF/flowcharts/certificates) to OCR.

Images carry no text layer, so extraction *is* OCR. The engine choice (Textract
etc.) is injected via the context, keeping this extractor backend-agnostic.
"""

from __future__ import annotations

from rag_core.models.content import ExtractedContent, Page
from rag_core.models.enums import DocumentType
from rag_processing.extractors.base import ExtractionContext, Extractor


class ImageExtractor(Extractor):
    supported = (DocumentType.IMAGE,)

    async def extract(self, data: bytes, ctx: ExtractionContext) -> ExtractedContent:
        if ctx.ocr_engine is None:
            # OCR disabled: emit an empty, flagged page rather than failing.
            return ExtractedContent(
                document_id=ctx.document_id,
                pages=[Page(number=1, text="", is_scanned=True)],
            )
        page = await ctx.ocr_engine.ocr_image(data)
        return ExtractedContent(document_id=ctx.document_id, pages=[page], has_ocr=True)
