"""Unit tests for the custom PDF layout logic and Textract block reshaping."""

from __future__ import annotations

from typing import Any

from rag_core.models.content import BlockType
from rag_processing.extractors.pdf import _page_blocks
from rag_processing.ocr.textract import _blocks_to_pages


class _FakePdfPage:
    """Duck-typed pdfplumber page exposing only what _page_blocks needs."""

    def __init__(self, words: list[dict[str, Any]]) -> None:
        self.width = 600
        self.height = 800
        self._words = words

    def extract_words(self, extra_attrs: list[str] | None = None, use_text_flow: bool = False):
        return self._words


def test_page_blocks_detects_headings_and_lines() -> None:
    words = [
        # Heading line (large font, few words) at top.
        {"text": "SECTION", "x0": 10, "x1": 90, "top": 10, "bottom": 34, "size": 20},
        {"text": "ONE", "x0": 95, "x1": 140, "top": 10, "bottom": 34, "size": 20},
        # Body line (smaller, more frequent font) below.
        {"text": "this", "x0": 10, "x1": 40, "top": 60, "bottom": 70, "size": 10},
        {"text": "is", "x0": 45, "x1": 60, "top": 60, "bottom": 70, "size": 10},
        {"text": "body", "x0": 65, "x1": 95, "top": 60, "bottom": 70, "size": 10},
        {"text": "text", "x0": 100, "x1": 130, "top": 60, "bottom": 70, "size": 10},
    ]
    blocks = _page_blocks(_FakePdfPage(words), 1, heading_max_words=12)
    assert len(blocks) == 2
    heading, body = blocks
    assert heading.text == "SECTION ONE"
    assert heading.role == BlockType.HEADING
    assert heading.heading_level == 1
    assert body.text == "this is body text"
    assert body.role == BlockType.PARAGRAPH
    # bbox is normalised to [0,1].
    assert 0.0 <= heading.bbox.x0 <= 1.0


def test_textract_blocks_to_pages_with_table() -> None:
    blocks = [
        {"Id": "l1", "BlockType": "LINE", "Text": "Hello", "Page": 1},
        {"Id": "l2", "BlockType": "LINE", "Text": "World", "Page": 1},
        {"Id": "t1", "BlockType": "TABLE", "Page": 1,
         "Relationships": [{"Type": "CHILD", "Ids": ["c1", "c2", "c3", "c4"]}]},
        {"Id": "c1", "BlockType": "CELL", "RowIndex": 1, "ColumnIndex": 1,
         "Relationships": [{"Type": "CHILD", "Ids": ["w1"]}]},
        {"Id": "c2", "BlockType": "CELL", "RowIndex": 1, "ColumnIndex": 2,
         "Relationships": [{"Type": "CHILD", "Ids": ["w2"]}]},
        {"Id": "c3", "BlockType": "CELL", "RowIndex": 2, "ColumnIndex": 1,
         "Relationships": [{"Type": "CHILD", "Ids": ["w3"]}]},
        {"Id": "c4", "BlockType": "CELL", "RowIndex": 2, "ColumnIndex": 2,
         "Relationships": [{"Type": "CHILD", "Ids": ["w4"]}]},
        {"Id": "w1", "BlockType": "WORD", "Text": "A"},
        {"Id": "w2", "BlockType": "WORD", "Text": "B"},
        {"Id": "w3", "BlockType": "WORD", "Text": "1"},
        {"Id": "w4", "BlockType": "WORD", "Text": "2"},
    ]
    pages = _blocks_to_pages(blocks)
    assert len(pages) == 1
    page = pages[0]
    assert page.text == "Hello\nWorld"
    assert page.is_scanned is True
    assert page.tables[0].rows == [["A", "B"], ["1", "2"]]
    assert "| A | B |" in page.tables[0].markdown


def test_textract_empty_blocks() -> None:
    assert _blocks_to_pages([]) == []
