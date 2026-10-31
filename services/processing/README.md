# Processing Service

Platform service **#2**. Turns raw documents into structured `ExtractedContent`.

## Responsibilities

- Consume `DISCOVERED` events from SQS; fetch bytes from S3.
- **Extract** text, **tables**, layout (**headings/footers/TOC**), **page numbers**
  via a format-specific extractor (PDF, DOCX, PPTX, XLSX, XML, HTML, TXT, images).
- **OCR** scanned PDFs and images via a pluggable engine (Textract by default).
- Enrich: **language detection**, **PII detection**, **document classification**.
- Persist `ExtractedContent` to the content store (S3) and emit a `PROCESSED`
  event pointing at it for the chunking service.

## Format → extractor

| Type | Extractor | Notes |
|---|---|---|
| PDF | `pdf.PdfExtractor` | pdfplumber text+tables+geometry; pypdf TOC; **scanned→OCR fallback**. |
| DOCX | `office.DocxExtractor` | document-order paragraphs+tables; Word heading styles. |
| PPTX | `office.PptxExtractor` | one slide = one page; titles→headings; tables. |
| XLSX | `office.XlsxExtractor` | one sheet = one page rendered as a table. |
| HTML | `text.HtmlExtractor` | lxml strip + heading detection. |
| XML | `text.XmlExtractor` | recover-mode parse, text nodes. |
| TXT | `text.TextExtractor` | synthetic pagination. |
| Image | `image.ImageExtractor` | routes to OCR engine. |

## OCR

`OCREngine` is pluggable (`PROCESSING_OCR_PROVIDER`):

- **textract** (default) — Amazon Textract; managed, AWS-native, extracts tables,
  **no system binaries**. Images use sync `analyze_document`; scanned PDFs use the
  async job API against the S3 object.
- **tesseract** — local Tesseract (needs the `tesseract` binary; PDF needs poppler).
  Dev/offline only.
- **none** — OCR disabled; scanned pages flagged for review (graceful degradation).

## Reliability

- N concurrent SQS consumers; long-poll receive.
- **Visibility heartbeat** keeps long OCR jobs from being redelivered mid-flight.
- Success → delete. Retryable failure → SQS redelivers (→ DLQ via redrive policy).
  Permanent/poison → routed to DLQ explicitly so one bad doc never blocks the queue.
- Idempotent: deterministic content keys; re-processing overwrites in place.

## Why content goes to S3, not the SQS message

Extracted content can be large (a 500-page manual). Messages stay small and the
pipeline stays replayable by passing a **content URI**, not the payload.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` `/readyz` `/metrics` | probes + Prometheus |
| POST | `/v1/process` | process one document on demand (`s3_uri` or `bucket`+`key`) |

## Config (`PROCESSING_` prefix)

| Var | Default | Meaning |
|---|---|---|
| `PROCESSING_RUN_WORKER` | true | Run the SQS consumer in-process. |
| `PROCESSING_WORKER_CONCURRENCY` | 8 | Concurrent consumers. |
| `PROCESSING_OUTPUT_QUEUE_URL` | "" | Chunking queue; empty disables emit. |
| `PROCESSING_OCR_PROVIDER` | textract | textract / tesseract / none. |
| `PROCESSING_DETECT_TABLES` | true | Table extraction. |
| `PROCESSING_MAX_PAGES` | 0 | Cap pages (0 = no cap). |

## Tests

```bash
uv run pytest services/processing -q
```
