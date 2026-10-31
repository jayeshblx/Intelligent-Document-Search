"""PDF extractor.

Strategy:
* Native text + word geometry + tables via ``pdfplumber`` (no system binaries).
* Document outline (TOC) via ``pypdf``.
* Heading detection from font-size geometry: lines whose font is meaningfully
  larger than the body font and that are short are tagged as headings — this is
  what lets chunking respect section hierarchy and citations name the section.
* **Scanned-PDF detection**: if native text is near-empty, the page layer is an
  image, so we fall back to OCR (``ctx.ocr_engine.ocr_pdf`` against the S3 object).

Parsing is CPU-bound and runs in a worker thread; OCR fallback is awaited async.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from io import BytesIO
from typing import Any

import pdfplumber
from pypdf import PdfReader

from rag_core.logging import get_logger
from rag_core.models.content import (
    BlockType,
    BoundingBox,
    ExtractedContent,
    Page,
    Table,
    TextBlock,
    TOCEntry,
)
from rag_core.models.enums import DocumentType
from rag_processing.extractors.base import ExtractionContext, Extractor
from rag_processing.extractors.layout import grid_to_markdown

_log = get_logger("rag_processing.extractors.pdf")

# Average chars/page below this => treat the PDF as scanned (image-only).
_SCANNED_AVG_CHARS = 10
_LINE_TOP_TOLERANCE = 3.0  # points; words within this top-delta share a line


class PdfExtractor(Extractor):
    supported = (DocumentType.PDF, DocumentType.SCANNED_PDF)

    async def extract(self, data: bytes, ctx: ExtractionContext) -> ExtractedContent:
        content, total_chars = await asyncio.to_thread(_parse_native, data, ctx)

        page_count = max(content.page_count, 1)
        looks_scanned = (total_chars / page_count) < _SCANNED_AVG_CHARS
        if looks_scanned and ctx.ocr_engine is not None:
            _log.info("pdf_scanned_ocr_fallback", document_id=ctx.document_id, key=ctx.key)
            # Prefer local rasterised OCR (Tesseract via PyMuPDF) on the bytes we
            # already have; fall back to the S3-based engine (Textract) otherwise.
            ocr_pages = await ctx.ocr_engine.ocr_pdf_bytes(data, max_pages=ctx.max_pages or 200)
            if not ocr_pages and ctx.bucket:
                ocr_pages = await ctx.ocr_engine.ocr_pdf(
                    bucket=ctx.bucket, key=ctx.key, max_pages=ctx.max_pages or 200
                )
            if ocr_pages:
                return ExtractedContent(
                    document_id=ctx.document_id,
                    pages=ocr_pages,
                    toc=content.toc,
                    has_ocr=True,
                )
        return content


def _parse_native(data: bytes, ctx: ExtractionContext) -> tuple[ExtractedContent, int]:
    """Synchronous native parse. Returns (content, total_text_chars).

    Three paths:
    * **pymupdf** (default): fast PyMuPDF text -> lightweight Markdown (headings
      via font size). C-speed (~85ms/page), no stalls. Falls back to pypdf if
      PyMuPDF errors on a file. Scanned pages route to OCR.
    * **fast**: pypdf ``extract_text`` — plain text, robust and quick.
    * **rich**: pdfplumber word geometry (heading detection) + table extraction.
    """
    if ctx.pdf_engine == "pymupdf":
        try:
            return _parse_pymupdf(data, ctx)
        except Exception as exc:  # never let PDF parsing break the pipeline
            _log.warning("pymupdf_failed_fallback", key=ctx.key, error=str(exc))
            return _parse_fast(data, ctx)
    if ctx.detect_tables and ctx.pdf_engine == "pdfplumber":
        return _parse_rich(data, ctx)
    return _parse_fast(data, ctx)


def _parse_pymupdf(data: bytes, ctx: ExtractionContext) -> tuple[ExtractedContent, int]:
    """Fast PyMuPDF text extraction rendered as lightweight Markdown.

    Uses ``get_text("dict")`` (C-speed, ~85ms/page) with font-size heading
    detection — headings become ``# ...`` lines. We deliberately avoid
    pymupdf4llm's layout/OCR model here: it is ~20x slower (84s vs 4s for a
    47-page PDF) and would time out large documents. Scanned PDFs (no text
    layer) fall through to the OCR path in ``extract``.
    """
    import pymupdf  # lazy import: only needed when parsing PDFs

    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        toc: list[TOCEntry] = []
        for level, title, page_no in doc.get_toc() or []:
            if str(title).strip() and page_no and page_no > 0:
                toc.append(TOCEntry(title=str(title).strip(), level=max(int(level), 1), page=page_no))

        limit = min(ctx.max_pages or doc.page_count, doc.page_count)
        pages: list[Page] = []
        total_chars = 0
        for i in range(limit):
            text, blocks = _pymupdf_page(doc.load_page(i), i + 1, ctx.heading_max_words)
            total_chars += len(text)
            pages.append(Page(number=i + 1, text=text, blocks=blocks))
    finally:
        doc.close()

    return ExtractedContent(document_id=ctx.document_id, pages=pages, toc=toc), total_chars


def _pymupdf_page(page: Any, page_no: int, heading_max_words: int) -> tuple[str, list[TextBlock]]:
    """One page -> (markdown text, blocks) with cheap font-size heading detection."""
    data = page.get_text("dict")
    spans_sizes = [
        round(float(s.get("size", 0)))
        for blk in data.get("blocks", [])
        for line in blk.get("lines", [])
        for s in line.get("spans", [])
        if s.get("text", "").strip()
    ]
    if not spans_sizes:
        return "", []
    body = Counter(spans_sizes).most_common(1)[0][0] or 1

    md_lines: list[str] = []
    blocks: list[TextBlock] = []
    for blk in data.get("blocks", []):
        for line in blk.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(s.get("text", "") for s in spans).strip()
            if not text:
                continue
            max_size = max((float(s.get("size", 0)) for s in spans), default=float(body))
            is_heading = max_size > body * 1.2 and len(text.split()) <= heading_max_words
            if is_heading:
                level = _heading_level(max_size, body)
                md_lines.append("#" * level + " " + text)
                blocks.append(
                    TextBlock(text=text, role=BlockType.HEADING, page=page_no, heading_level=level)
                )
            else:
                md_lines.append(text)
                blocks.append(TextBlock(text=text, role=BlockType.PARAGRAPH, page=page_no))
    return "\n".join(md_lines), blocks


def _parse_fast(data: bytes, ctx: ExtractionContext) -> tuple[ExtractedContent, int]:
    toc = _extract_toc(data)
    pages: list[Page] = []
    total_chars = 0
    reader = PdfReader(BytesIO(data))
    limit = ctx.max_pages or len(reader.pages)
    for idx, page in enumerate(reader.pages[:limit], start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            _log.warning("pdf_page_extract_error", page=idx, error=str(exc))
            text = ""
        blocks = [
            TextBlock(text=line.strip(), role=BlockType.PARAGRAPH, page=idx)
            for line in text.splitlines()
            if line.strip()
        ]
        total_chars += len(text)
        pages.append(Page(number=idx, text=text, blocks=blocks))
    return ExtractedContent(document_id=ctx.document_id, pages=pages, toc=toc), total_chars


def _parse_rich(data: bytes, ctx: ExtractionContext) -> tuple[ExtractedContent, int]:
    """pdfplumber-based parse with heading geometry + table extraction."""
    toc = _extract_toc(data)
    pages: list[Page] = []
    total_chars = 0

    with pdfplumber.open(BytesIO(data)) as pdf:
        limit = ctx.max_pages or len(pdf.pages)
        for idx, page in enumerate(pdf.pages[:limit], start=1):
            try:
                blocks = _page_blocks(page, idx, ctx.heading_max_words)
            except Exception as exc:
                _log.warning("pdf_page_parse_error", page=idx, error=str(exc))
                blocks = []
            text = "\n".join(b.text for b in blocks)
            total_chars += len(text)

            tables: list[Table] = []
            if ctx.detect_tables:
                tables = _page_tables(page, idx)

            pages.append(Page(number=idx, text=text, blocks=blocks, tables=tables))

    return ExtractedContent(document_id=ctx.document_id, pages=pages, toc=toc), total_chars


def _page_blocks(page: Any, page_no: int, heading_max_words: int) -> list[TextBlock]:
    words = page.extract_words(extra_attrs=["size"], use_text_flow=True)
    if not words:
        return []

    width = float(page.width or 1) or 1.0
    height = float(page.height or 1) or 1.0

    # Body font = most common rounded size; headings stand out above it.
    sizes = Counter(round(float(w.get("size", 0)) or 0) for w in words)
    body_size = sizes.most_common(1)[0][0] or 1

    # Group words into lines by vertical position.
    lines: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    last_top: float | None = None
    for w in sorted(words, key=lambda w: (round(float(w["top"]), 1), float(w["x0"]))):
        top = float(w["top"])
        if last_top is None or abs(top - last_top) <= _LINE_TOP_TOLERANCE:
            current.append(w)
        else:
            lines.append(current)
            current = [w]
        last_top = top
    if current:
        lines.append(current)

    blocks: list[TextBlock] = []
    for line in lines:
        ordered = sorted(line, key=lambda w: float(w["x0"]))
        text = " ".join(w["text"] for w in ordered).strip()
        if not text:
            continue
        max_size = max(float(w.get("size", body_size)) for w in ordered)
        is_heading = max_size > body_size * 1.15 and len(ordered) <= heading_max_words
        bbox = BoundingBox(
            x0=_clamp(min(float(w["x0"]) for w in ordered) / width),
            y0=_clamp(min(float(w["top"]) for w in ordered) / height),
            x1=_clamp(max(float(w["x1"]) for w in ordered) / width),
            y1=_clamp(max(float(w["bottom"]) for w in ordered) / height),
        )
        blocks.append(
            TextBlock(
                text=text,
                role=BlockType.HEADING if is_heading else BlockType.PARAGRAPH,
                page=page_no,
                heading_level=_heading_level(max_size, body_size) if is_heading else None,
                bbox=bbox,
            )
        )
    return blocks


def _page_tables(page: Any, page_no: int) -> list[Table]:
    tables: list[Table] = []
    try:
        raw = page.extract_tables()
    except Exception as exc:
        _log.warning("pdf_table_parse_error", page=page_no, error=str(exc))
        return tables
    for raw_table in raw or []:
        grid = [[(cell or "").strip() for cell in row] for row in raw_table if row]
        # Drop rows whose cells are all blank — pdfplumber emits [None, None] rows for
        # ruling lines, which survive the `if row` filter above (a non-empty list) and
        # would otherwise be rendered as empty "|  |  |" markdown noise.
        grid = [r for r in grid if any(c for c in r)]
        if grid:
            tables.append(Table(page=page_no, rows=grid, markdown=grid_to_markdown(grid)))
    return tables


def _extract_toc(data: bytes) -> list[TOCEntry]:
    try:
        reader = PdfReader(BytesIO(data))
        outline = reader.outline
    except Exception as exc:
        _log.debug("pdf_toc_unavailable", error=str(exc))
        return []

    entries: list[TOCEntry] = []

    def _walk(items: Any, level: int) -> None:
        for item in items:
            if isinstance(item, list):
                _walk(item, level + 1)
                continue
            try:
                title = str(getattr(item, "title", "")).strip()
                page_index = reader.get_destination_page_number(item)
                if title and page_index is not None and page_index >= 0:
                    entries.append(TOCEntry(title=title, level=level, page=page_index + 1))
            except Exception:
                continue

    try:
        _walk(outline, 1)
    except Exception as exc:
        _log.debug("pdf_toc_walk_error", error=str(exc))
    return entries


def _heading_level(size: float, body_size: float) -> int:
    ratio = size / max(body_size, 1)
    if ratio >= 1.6:
        return 1
    if ratio >= 1.3:
        return 2
    return 3


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))
