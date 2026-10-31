"""Processing-service settings (env prefix ``PROCESSING_``)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OCRProvider(StrEnum):
    """Which OCR backend to use for scanned PDFs and images.

    TEXTRACT — Amazon Textract (managed, AWS-native, handles tables/forms; no
               system binaries). Recommended for prod.
    TESSERACT — local Tesseract via pytesseract (requires the tesseract binary;
               dev/offline only). Lazily imported.
    NONE — disable OCR; scanned pages are flagged for review with empty text.
    """

    TEXTRACT = "textract"
    TESSERACT = "tesseract"
    NONE = "none"


class ProcessingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PROCESSING_", env_file=".env", extra="ignore", case_sensitive=False
    )

    # SQS worker
    run_worker: bool = True  # start the SQS consumer in-process (disable for API-only pods)
    worker_concurrency: int = Field(default=8, ge=1, le=128)
    output_queue_url: str = ""  # next stage (chunking); empty => don't emit
    poison_max_receive: int = Field(default=5, ge=1)  # -> DLQ after this many tries

    # Where extracted content is persisted (S3 key prefix within the source bucket)
    content_prefix: str = "processed/content/"

    # OCR
    ocr_provider: OCRProvider = OCRProvider.TEXTRACT
    ocr_max_pages: int = Field(default=200, ge=1)  # cap Textract async jobs
    # Path to the tesseract binary (blank = auto-detect / rely on PATH).
    tesseract_cmd: str = ""
    textract_poll_interval_s: float = Field(default=2.0, ge=0.5)
    textract_timeout_s: float = Field(default=300.0, ge=10.0)

    # Layout/extraction tuning
    heading_max_words: int = Field(default=12, ge=1)  # short lines -> heading candidate
    detect_tables: bool = True
    # PDF engine: pymupdf converts PDFs to Markdown (reliable, fast, no stalls) —
    # the default; pypdf = plain fast text; pdfplumber = rich word geometry + tables.
    pdf_engine: str = "pymupdf"

    # Enrichment toggles
    detect_language: bool = True
    detect_pii: bool = True
    classify: bool = True
    # Long documents are expensive; cap pages processed (0 = no cap).
    max_pages: int = Field(default=0, ge=0)
