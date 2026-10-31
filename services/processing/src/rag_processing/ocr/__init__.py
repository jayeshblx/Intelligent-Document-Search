"""OCR engines for scanned PDFs and images.

Pluggable behind ``OCREngine`` so the choice (Textract / Tesseract / none) is a
config switch, not a code change. Default is Textract: managed, AWS-native, no
poppler/tesseract system binaries to ship in the image, and it extracts tables.
"""

from __future__ import annotations

from rag_processing.ocr.base import OCREngine
from rag_processing.ocr.factory import build_ocr_engine
from rag_processing.ocr.noop import NoOpOCREngine

__all__ = ["NoOpOCREngine", "OCREngine", "build_ocr_engine"]
