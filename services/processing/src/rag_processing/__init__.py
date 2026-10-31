"""Document Processing Agent (platform service #2).

Consumes ``DISCOVERED`` events from SQS, fetches the raw bytes from S3, and turns
each document into structured ``ExtractedContent`` — text, layout (headings /
footers / TOC), tables, page numbers, OCR for scanned pages/images, plus language
detection, PII findings, and a functional classification.

Design stance:
* Extraction is an *extractor-per-format* strategy selected by a registry, so a
  new format is one new class + one registration — open/closed.
* OCR sits behind a pluggable ``OCREngine`` (Textract by default — AWS-native, no
  system binaries) so scanned PDFs/images work in prod without poppler/tesseract.
* The output (``ExtractedContent``) is persisted to a content store and a
  ``PROCESSED`` event is emitted for the chunking service. The service is
  idempotent: re-processing the same document overwrites the same content key.
"""

from __future__ import annotations

__version__ = "0.1.0"
