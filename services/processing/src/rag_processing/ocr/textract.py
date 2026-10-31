"""Amazon Textract OCR engine.

Textract is the production default: managed OCR with table extraction, no system
binaries. Images use the synchronous ``analyze_document``; multi-page scanned PDFs
use the asynchronous job API (start -> poll -> paginate) against the object in S3.

Textract's flat block list (PAGE/LINE/WORD/TABLE/CELL with id relationships) is
reshaped here into the platform's ``Page``/``Table`` models so OCR output is
indistinguishable downstream from natively-extracted content.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from rag_core.aws.session import AwsSessionFactory
from rag_core.logging import get_logger
from rag_core.models.content import BlockType, Page, Table, TextBlock
from rag_core.resilience import NonRetryableError, RetryableError
from rag_processing.extractors.layout import grid_to_markdown
from rag_processing.ocr.base import OCREngine

_log = get_logger("rag_processing.ocr.textract")


class TextractOCREngine(OCREngine):
    name = "textract"

    def __init__(
        self,
        factory: AwsSessionFactory,
        *,
        poll_interval_s: float = 2.0,
        timeout_s: float = 300.0,
        detect_tables: bool = True,
    ) -> None:
        self._factory = factory
        self._poll = poll_interval_s
        self._timeout = timeout_s
        self._tables = detect_tables
        self._client: Any | None = None
        self._cm: Any | None = None

    async def _ensure(self) -> Any:
        if self._client is None:
            self._cm = self._factory.client("textract")
            self._client = await self._cm.__aenter__()
        return self._client

    async def aclose(self) -> None:
        if self._cm is not None:
            await self._cm.__aexit__(None, None, None)
            self._cm = None
            self._client = None

    async def ocr_image(self, data: bytes) -> Page:
        client = await self._ensure()
        try:
            if self._tables:
                resp = await client.analyze_document(
                    Document={"Bytes": data}, FeatureTypes=["TABLES"]
                )
            else:
                resp = await client.detect_document_text(Document={"Bytes": data})
        except Exception as exc:
            raise RetryableError(f"Textract image OCR failed: {exc}") from exc
        pages = _blocks_to_pages(resp.get("Blocks", []))
        return pages[0] if pages else Page(number=1, text="", is_scanned=True)

    async def ocr_pdf(self, *, bucket: str, key: str, max_pages: int = 200) -> list[Page]:
        client = await self._ensure()
        location = {"S3Object": {"Bucket": bucket, "Name": key}}
        try:
            if self._tables:
                start = await client.start_document_analysis(
                    DocumentLocation=location, FeatureTypes=["TABLES"]
                )
            else:
                start = await client.start_document_text_detection(DocumentLocation=location)
        except Exception as exc:
            raise RetryableError(f"Textract start job failed: {exc}") from exc

        blocks = await self._collect(client, start["JobId"], with_tables=self._tables)
        pages = _blocks_to_pages(blocks)
        return pages[:max_pages] if max_pages else pages

    async def _collect(self, client: Any, job_id: str, *, with_tables: bool) -> list[dict[str, Any]]:
        getter = (
            client.get_document_analysis if with_tables else client.get_document_text_detection
        )
        deadline = time.monotonic() + self._timeout
        while True:
            resp = await getter(JobId=job_id)
            status = resp.get("JobStatus")
            if status == "SUCCEEDED":
                break
            if status == "FAILED":
                raise NonRetryableError(f"Textract job failed: {resp.get('StatusMessage')}")
            if time.monotonic() > deadline:
                raise RetryableError(f"Textract job {job_id} timed out")
            await asyncio.sleep(self._poll)

        blocks: list[dict[str, Any]] = list(resp.get("Blocks", []))
        token = resp.get("NextToken")
        while token:
            resp = await getter(JobId=job_id, NextToken=token)
            blocks.extend(resp.get("Blocks", []))
            token = resp.get("NextToken")
        return blocks

    async def health(self) -> bool:
        try:
            await self._ensure()
            return True
        except Exception:
            return False


# ---- Block reshaping --------------------------------------------------------
def _blocks_to_pages(blocks: list[dict[str, Any]]) -> list[Page]:
    """Reshape a Textract block list into ordered ``Page`` objects."""
    if not blocks:
        return []
    id_map = {b["Id"]: b for b in blocks if "Id" in b}

    # Group line text and tables per page (Textract pages are 1-based; sync image
    # responses omit Page, so default to 1).
    lines_by_page: dict[int, list[str]] = {}
    tables_by_page: dict[int, list[Table]] = {}

    for b in blocks:
        page_no = int(b.get("Page", 1))
        btype = b.get("BlockType")
        if btype == "LINE":
            lines_by_page.setdefault(page_no, []).append(b.get("Text", ""))
        elif btype == "TABLE":
            grid = _table_grid(b, id_map)
            if grid:
                tables_by_page.setdefault(page_no, []).append(
                    Table(page=page_no, rows=grid, markdown=grid_to_markdown(grid))
                )

    page_numbers = sorted(set(lines_by_page) | set(tables_by_page)) or [1]
    pages: list[Page] = []
    for n in page_numbers:
        lines = lines_by_page.get(n, [])
        text = "\n".join(lines)
        text_blocks = [
            TextBlock(text=line, role=BlockType.PARAGRAPH, page=n, from_ocr=True)
            for line in lines
            if line
        ]
        pages.append(
            Page(
                number=n,
                text=text,
                blocks=text_blocks,
                tables=tables_by_page.get(n, []),
                is_scanned=True,
            )
        )
    return pages


def _table_grid(table_block: dict[str, Any], id_map: dict[str, dict[str, Any]]) -> list[list[str]]:
    cells = [
        cb
        for rel in table_block.get("Relationships", []) or []
        if rel.get("Type") == "CHILD"
        for cid in rel.get("Ids", [])
        if (cb := id_map.get(cid)) is not None and cb.get("BlockType") == "CELL"
    ]
    if not cells:
        return []
    max_row = max(c.get("RowIndex", 1) for c in cells)
    max_col = max(c.get("ColumnIndex", 1) for c in cells)
    grid = [["" for _ in range(max_col)] for _ in range(max_row)]
    for c in cells:
        r, col = c.get("RowIndex", 1) - 1, c.get("ColumnIndex", 1) - 1
        grid[r][col] = _cell_text(c, id_map)
    return grid


def _cell_text(cell: dict[str, Any], id_map: dict[str, dict[str, Any]]) -> str:
    words: list[str] = []
    for rel in cell.get("Relationships", []) or []:
        if rel.get("Type") != "CHILD":
            continue
        for cid in rel.get("Ids", []):
            wb = id_map.get(cid)
            if not wb:
                continue
            if wb.get("BlockType") == "WORD":
                words.append(wb.get("Text", ""))
            elif wb.get("BlockType") == "SELECTION_ELEMENT" and wb.get("SelectionStatus") == "SELECTED":
                words.append("[x]")
    return " ".join(words).strip()
