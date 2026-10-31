"""Extended unit tests for Textract OCR engine — focused on coverage gaps.

Priority: Error handling (job failures, timeouts), multi-page pagination,
confidence scoring, block relationships, table reconstruction, async job polling.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from rag_core.models.content import BlockType, Page
from rag_core.resilience import NonRetryableError, RetryableError
from rag_processing.ocr.textract import (
    TextractOCREngine,
    _blocks_to_pages,
    _cell_text,
    _table_grid,
)


class FakeAwsSessionFactory:
    """Minimal session factory for testing."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def client(self, service: str):
        return _FakeClientContext(self._client)


class _FakeClientContext:
    def __init__(self, client: Any) -> None:
        self._client = client

    async def __aenter__(self):
        return self._client

    async def __aexit__(self, *args):
        pass


class FakeTextractClient:
    """Stub Textract client for testing without AWS."""

    def __init__(
        self,
        *,
        analyze_response: dict[str, Any] | None = None,
        detect_response: dict[str, Any] | None = None,
        job_status: str = "SUCCEEDED",
        job_blocks: list[dict[str, Any]] | None = None,
        raise_on_start: Exception | None = None,
        raise_on_get: Exception | None = None,
        multi_page_job: bool = False,
    ) -> None:
        self.analyze_response = analyze_response or {"Blocks": []}
        self.detect_response = detect_response or {"Blocks": []}
        self.job_status = job_status
        self.job_blocks = job_blocks or []
        self.raise_on_start = raise_on_start
        self.raise_on_get = raise_on_get
        self.multi_page = multi_page_job
        self._get_call_count = 0

    async def analyze_document(self, *, Document: dict, FeatureTypes: list[str]) -> dict:
        if "Bytes" in Document:
            return self.analyze_response
        raise RuntimeError("analyze_document: unexpected Document format")

    async def detect_document_text(self, *, Document: dict) -> dict:
        if "Bytes" in Document:
            return self.detect_response
        raise RuntimeError("detect_document_text: unexpected Document format")

    async def start_document_analysis(
        self, *, DocumentLocation: dict, FeatureTypes: list[str]
    ) -> dict:
        if self.raise_on_start:
            raise self.raise_on_start
        return {"JobId": "job-123"}

    async def start_document_text_detection(self, *, DocumentLocation: dict) -> dict:
        if self.raise_on_start:
            raise self.raise_on_start
        return {"JobId": "job-456"}

    async def get_document_analysis(self, *, JobId: str, NextToken: str | None = None) -> dict:
        if self.raise_on_get:
            raise self.raise_on_get
        self._get_call_count += 1
        if self.multi_page and not NextToken:
            return {
                "JobStatus": self.job_status,
                "Blocks": self.job_blocks[:2],
                "NextToken": "page2",
            }
        if self.multi_page and NextToken == "page2":
            return {"JobStatus": self.job_status, "Blocks": self.job_blocks[2:]}
        return {"JobStatus": self.job_status, "Blocks": self.job_blocks}

    async def get_document_text_detection(
        self, *, JobId: str, NextToken: str | None = None
    ) -> dict:
        if self.raise_on_get:
            raise self.raise_on_get
        return {"JobStatus": self.job_status, "Blocks": self.job_blocks}


def _make_engine(
    client: FakeTextractClient,
    *,
    poll_interval_s: float = 0.01,
    timeout_s: float = 1.0,
    detect_tables: bool = True,
) -> TextractOCREngine:
    factory = FakeAwsSessionFactory(client)
    return TextractOCREngine(
        factory, poll_interval_s=poll_interval_s, timeout_s=timeout_s, detect_tables=detect_tables
    )


# ---- OCR image tests ---------------------------------------------------------
async def test_ocr_image_with_tables() -> None:
    blocks = [
        {"Id": "l1", "BlockType": "LINE", "Text": "Header", "Page": 1},
    ]
    client = FakeTextractClient(analyze_response={"Blocks": blocks})
    engine = _make_engine(client, detect_tables=True)

    page = await engine.ocr_image(b"fake-image-data")
    assert page.number == 1
    assert page.text == "Header"
    assert page.is_scanned is True


async def test_ocr_image_without_tables() -> None:
    blocks = [{"Id": "l1", "BlockType": "LINE", "Text": "Simple text", "Page": 1}]
    client = FakeTextractClient(detect_response={"Blocks": blocks})
    engine = _make_engine(client, detect_tables=False)

    page = await engine.ocr_image(b"fake-image-data")
    assert page.text == "Simple text"


async def test_ocr_image_empty_response() -> None:
    client = FakeTextractClient(analyze_response={"Blocks": []})
    engine = _make_engine(client)

    page = await engine.ocr_image(b"fake-image-data")
    assert page.number == 1
    assert page.text == ""
    assert page.is_scanned is True


async def test_ocr_image_raises_retryable_on_error() -> None:
    client = FakeTextractClient()
    engine = _make_engine(client)

    # Patch the client to raise an exception.
    async def _fail(**kwargs):
        raise RuntimeError("Textract service error")

    client.analyze_document = _fail

    with pytest.raises(RetryableError, match="Textract image OCR failed"):
        await engine.ocr_image(b"data")


# ---- OCR PDF (async job) tests -----------------------------------------------
async def test_ocr_pdf_with_tables_job_succeeds() -> None:
    blocks = [
        {"Id": "l1", "BlockType": "LINE", "Text": "Page one", "Page": 1},
        {"Id": "l2", "BlockType": "LINE", "Text": "Page two", "Page": 2},
    ]
    client = FakeTextractClient(job_status="SUCCEEDED", job_blocks=blocks)
    engine = _make_engine(client, detect_tables=True)

    pages = await engine.ocr_pdf(bucket="test-bucket", key="doc.pdf", max_pages=200)
    assert len(pages) == 2
    assert pages[0].text == "Page one"
    assert pages[1].text == "Page two"


async def test_ocr_pdf_without_tables() -> None:
    blocks = [{"Id": "l1", "BlockType": "LINE", "Text": "Text only", "Page": 1}]
    client = FakeTextractClient(job_status="SUCCEEDED", job_blocks=blocks)
    engine = _make_engine(client, detect_tables=False)

    pages = await engine.ocr_pdf(bucket="b", key="k", max_pages=200)
    assert len(pages) == 1
    assert pages[0].text == "Text only"


async def test_ocr_pdf_job_failed_status() -> None:
    client = FakeTextractClient(job_status="FAILED", job_blocks=[])
    engine = _make_engine(client)

    with pytest.raises(NonRetryableError, match="Textract job failed"):
        await engine.ocr_pdf(bucket="b", key="k")


async def test_ocr_pdf_job_timeout() -> None:
    """Job stays IN_PROGRESS past timeout -> RetryableError."""

    class _SlowClient(FakeTextractClient):
        async def get_document_analysis(self, *, JobId: str, NextToken: str | None = None):
            await asyncio.sleep(0.1)  # Delay to trigger timeout.
            return {"JobStatus": "IN_PROGRESS", "Blocks": []}

    client = _SlowClient()
    engine = _make_engine(client, poll_interval_s=0.05, timeout_s=0.2)

    with pytest.raises(RetryableError, match="timed out"):
        await engine.ocr_pdf(bucket="b", key="k")


async def test_ocr_pdf_start_job_fails() -> None:
    client = FakeTextractClient(raise_on_start=RuntimeError("Start failed"))
    engine = _make_engine(client)

    with pytest.raises(RetryableError, match="Textract start job failed"):
        await engine.ocr_pdf(bucket="b", key="k")


async def test_ocr_pdf_get_job_fails() -> None:
    client = FakeTextractClient(raise_on_get=RuntimeError("Get failed"))
    engine = _make_engine(client)

    with pytest.raises(RuntimeError, match="Get failed"):
        await engine.ocr_pdf(bucket="b", key="k")


async def test_ocr_pdf_pagination() -> None:
    """Test NextToken pagination for large results."""
    blocks = [
        {"Id": "l1", "BlockType": "LINE", "Text": "Page 1a", "Page": 1},
        {"Id": "l2", "BlockType": "LINE", "Text": "Page 1b", "Page": 1},
        {"Id": "l3", "BlockType": "LINE", "Text": "Page 2a", "Page": 2},
    ]
    client = FakeTextractClient(
        job_status="SUCCEEDED", job_blocks=blocks, multi_page_job=True
    )
    engine = _make_engine(client)

    pages = await engine.ocr_pdf(bucket="b", key="k")
    assert len(pages) == 2
    assert "Page 1a" in pages[0].text
    assert "Page 2a" in pages[1].text


async def test_ocr_pdf_max_pages_truncates() -> None:
    blocks = [
        {"Id": f"l{i}", "BlockType": "LINE", "Text": f"Page {i}", "Page": i}
        for i in range(1, 11)
    ]
    client = FakeTextractClient(job_status="SUCCEEDED", job_blocks=blocks)
    engine = _make_engine(client)

    pages = await engine.ocr_pdf(bucket="b", key="k", max_pages=5)
    assert len(pages) == 5


# ---- Engine lifecycle tests --------------------------------------------------
async def test_engine_ensure_creates_client_once() -> None:
    client = FakeTextractClient()
    engine = _make_engine(client)

    c1 = await engine._ensure()
    c2 = await engine._ensure()
    assert c1 is c2  # Same client instance.


async def test_engine_aclose_releases_client() -> None:
    client = FakeTextractClient()
    engine = _make_engine(client)

    await engine._ensure()
    await engine.aclose()
    assert engine._client is None
    assert engine._cm is None


async def test_health_returns_true_when_client_ready() -> None:
    client = FakeTextractClient()
    engine = _make_engine(client)

    assert await engine.health() is True


async def test_health_returns_false_on_client_error() -> None:
    class _BrokenFactory:
        def client(self, service: str):
            raise RuntimeError("Factory error")

    engine = TextractOCREngine(_BrokenFactory())  # type: ignore
    assert await engine.health() is False


# ---- Block reshaping tests ---------------------------------------------------
def test_blocks_to_pages_empty() -> None:
    assert _blocks_to_pages([]) == []


def test_blocks_to_pages_single_line() -> None:
    blocks = [{"Id": "l1", "BlockType": "LINE", "Text": "Hello", "Page": 1}]
    pages = _blocks_to_pages(blocks)
    assert len(pages) == 1
    assert pages[0].text == "Hello"
    assert pages[0].number == 1
    assert pages[0].is_scanned is True


def test_blocks_to_pages_multiple_lines_same_page() -> None:
    blocks = [
        {"Id": "l1", "BlockType": "LINE", "Text": "Line 1", "Page": 1},
        {"Id": "l2", "BlockType": "LINE", "Text": "Line 2", "Page": 1},
    ]
    pages = _blocks_to_pages(blocks)
    assert len(pages) == 1
    assert pages[0].text == "Line 1\nLine 2"


def test_blocks_to_pages_multiple_pages() -> None:
    blocks = [
        {"Id": "l1", "BlockType": "LINE", "Text": "Page 1", "Page": 1},
        {"Id": "l2", "BlockType": "LINE", "Text": "Page 2", "Page": 2},
    ]
    pages = _blocks_to_pages(blocks)
    assert len(pages) == 2
    assert pages[0].number == 1
    assert pages[1].number == 2


def test_blocks_to_pages_no_page_defaults_to_1() -> None:
    """Image responses omit Page; default to 1."""
    blocks = [{"Id": "l1", "BlockType": "LINE", "Text": "Image text"}]
    pages = _blocks_to_pages(blocks)
    assert pages[0].number == 1


def test_blocks_to_pages_text_blocks_marked_from_ocr() -> None:
    blocks = [{"Id": "l1", "BlockType": "LINE", "Text": "OCR line", "Page": 1}]
    pages = _blocks_to_pages(blocks)
    assert pages[0].blocks[0].from_ocr is True
    assert pages[0].blocks[0].role == BlockType.PARAGRAPH


# ---- Table grid reconstruction tests -----------------------------------------
def test_table_grid_basic() -> None:
    id_map = {
        "t1": {
            "BlockType": "TABLE",
            "Relationships": [{"Type": "CHILD", "Ids": ["c1", "c2"]}],
        },
        "c1": {"BlockType": "CELL", "RowIndex": 1, "ColumnIndex": 1},
        "c2": {"BlockType": "CELL", "RowIndex": 1, "ColumnIndex": 2},
    }
    grid = _table_grid(id_map["t1"], id_map)
    assert grid == [["", ""]]


def test_table_grid_with_word_relationships() -> None:
    id_map = {
        "t1": {
            "BlockType": "TABLE",
            "Relationships": [{"Type": "CHILD", "Ids": ["c1"]}],
        },
        "c1": {
            "BlockType": "CELL",
            "RowIndex": 1,
            "ColumnIndex": 1,
            "Relationships": [{"Type": "CHILD", "Ids": ["w1"]}],
        },
        "w1": {"BlockType": "WORD", "Text": "Data"},
    }
    grid = _table_grid(id_map["t1"], id_map)
    assert grid == [["Data"]]


def test_table_grid_multiple_rows_and_columns() -> None:
    id_map = {
        "t1": {
            "BlockType": "TABLE",
            "Relationships": [{"Type": "CHILD", "Ids": ["c1", "c2", "c3", "c4"]}],
        },
        "c1": {
            "BlockType": "CELL",
            "RowIndex": 1,
            "ColumnIndex": 1,
            "Relationships": [{"Type": "CHILD", "Ids": ["w1"]}],
        },
        "c2": {
            "BlockType": "CELL",
            "RowIndex": 1,
            "ColumnIndex": 2,
            "Relationships": [{"Type": "CHILD", "Ids": ["w2"]}],
        },
        "c3": {
            "BlockType": "CELL",
            "RowIndex": 2,
            "ColumnIndex": 1,
            "Relationships": [{"Type": "CHILD", "Ids": ["w3"]}],
        },
        "c4": {
            "BlockType": "CELL",
            "RowIndex": 2,
            "ColumnIndex": 2,
            "Relationships": [{"Type": "CHILD", "Ids": ["w4"]}],
        },
        "w1": {"BlockType": "WORD", "Text": "A"},
        "w2": {"BlockType": "WORD", "Text": "B"},
        "w3": {"BlockType": "WORD", "Text": "1"},
        "w4": {"BlockType": "WORD", "Text": "2"},
    }
    grid = _table_grid(id_map["t1"], id_map)
    assert grid == [["A", "B"], ["1", "2"]]


def test_table_grid_no_cells() -> None:
    id_map = {
        "t1": {"BlockType": "TABLE", "Relationships": [{"Type": "CHILD", "Ids": []}]},
    }
    grid = _table_grid(id_map["t1"], id_map)
    assert grid == []


def test_table_grid_missing_cell_ids() -> None:
    """Handles missing cell references gracefully."""
    id_map = {
        "t1": {
            "BlockType": "TABLE",
            "Relationships": [{"Type": "CHILD", "Ids": ["missing"]}],
        },
    }
    grid = _table_grid(id_map["t1"], id_map)
    assert grid == []


# ---- Cell text extraction tests ----------------------------------------------
def test_cell_text_single_word() -> None:
    id_map = {
        "c1": {
            "BlockType": "CELL",
            "Relationships": [{"Type": "CHILD", "Ids": ["w1"]}],
        },
        "w1": {"BlockType": "WORD", "Text": "Hello"},
    }
    text = _cell_text(id_map["c1"], id_map)
    assert text == "Hello"


def test_cell_text_multiple_words() -> None:
    id_map = {
        "c1": {
            "BlockType": "CELL",
            "Relationships": [{"Type": "CHILD", "Ids": ["w1", "w2"]}],
        },
        "w1": {"BlockType": "WORD", "Text": "Hello"},
        "w2": {"BlockType": "WORD", "Text": "World"},
    }
    text = _cell_text(id_map["c1"], id_map)
    assert text == "Hello World"


def test_cell_text_selection_element() -> None:
    """SELECTION_ELEMENT with SELECTED -> [x]."""
    id_map = {
        "c1": {
            "BlockType": "CELL",
            "Relationships": [{"Type": "CHILD", "Ids": ["s1"]}],
        },
        "s1": {"BlockType": "SELECTION_ELEMENT", "SelectionStatus": "SELECTED"},
    }
    text = _cell_text(id_map["c1"], id_map)
    assert text == "[x]"


def test_cell_text_selection_element_not_selected() -> None:
    id_map = {
        "c1": {
            "BlockType": "CELL",
            "Relationships": [{"Type": "CHILD", "Ids": ["s1"]}],
        },
        "s1": {"BlockType": "SELECTION_ELEMENT", "SelectionStatus": "NOT_SELECTED"},
    }
    text = _cell_text(id_map["c1"], id_map)
    assert text == ""


def test_cell_text_no_relationships() -> None:
    id_map = {"c1": {"BlockType": "CELL"}}
    text = _cell_text(id_map["c1"], id_map)
    assert text == ""


def test_cell_text_missing_word_ids() -> None:
    id_map = {
        "c1": {
            "BlockType": "CELL",
            "Relationships": [{"Type": "CHILD", "Ids": ["missing"]}],
        },
    }
    text = _cell_text(id_map["c1"], id_map)
    assert text == ""


# ---- Edge cases & integration ------------------------------------------------
def test_blocks_to_pages_with_table_integration() -> None:
    """Full integration: lines + table on same page."""
    blocks = [
        {"Id": "l1", "BlockType": "LINE", "Text": "Header", "Page": 1},
        {
            "Id": "t1",
            "BlockType": "TABLE",
            "Page": 1,
            "Relationships": [{"Type": "CHILD", "Ids": ["c1"]}],
        },
        {
            "Id": "c1",
            "BlockType": "CELL",
            "RowIndex": 1,
            "ColumnIndex": 1,
            "Relationships": [{"Type": "CHILD", "Ids": ["w1"]}],
        },
        {"Id": "w1", "BlockType": "WORD", "Text": "Data"},
    ]
    pages = _blocks_to_pages(blocks)
    assert len(pages) == 1
    assert pages[0].text == "Header"
    assert len(pages[0].tables) == 1
    assert pages[0].tables[0].rows == [["Data"]]


async def test_ocr_pdf_empty_blocks() -> None:
    """Job succeeds but returns no blocks."""
    client = FakeTextractClient(job_status="SUCCEEDED", job_blocks=[])
    engine = _make_engine(client)

    pages = await engine.ocr_pdf(bucket="b", key="k")
    assert pages == []
