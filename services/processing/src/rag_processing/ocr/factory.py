"""OCR engine factory — maps the configured provider to an engine instance."""

from __future__ import annotations

from rag_core.aws.session import AwsSessionFactory
from rag_processing.config import OCRProvider, ProcessingSettings
from rag_processing.ocr.base import OCREngine
from rag_processing.ocr.noop import NoOpOCREngine


def build_ocr_engine(
    settings: ProcessingSettings, factory: AwsSessionFactory
) -> OCREngine:
    """Construct the OCR engine for the configured provider."""
    if settings.ocr_provider is OCRProvider.TEXTRACT:
        from rag_processing.ocr.textract import TextractOCREngine

        return TextractOCREngine(
            factory,
            poll_interval_s=settings.textract_poll_interval_s,
            timeout_s=settings.textract_timeout_s,
            detect_tables=settings.detect_tables,
        )
    if settings.ocr_provider is OCRProvider.TESSERACT:
        from rag_processing.ocr.tesseract import TesseractOCREngine

        return TesseractOCREngine(tesseract_cmd=settings.tesseract_cmd)
    return NoOpOCREngine()
