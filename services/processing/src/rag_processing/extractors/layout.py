"""Shared layout helpers: table rendering + pagination.

Centralised so every extractor renders tables and lays content onto synthetic
pages identically (DRY across pdf/office/text/textract).
"""

from __future__ import annotations

from rag_core.models.content import BlockType, Page, Table, TextBlock

CHARS_PER_PAGE = 3000


def grid_to_markdown(grid: list[list[str]]) -> str:
    """Render a row grid as a GitHub-flavoured markdown table."""
    if not grid:
        return ""
    header = grid[0]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in grid[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def paginate_text(text: str, *, chars_per_page: int = CHARS_PER_PAGE) -> list[Page]:
    """Split flat text into synthetic pages on line boundaries by char budget."""
    pages: list[Page] = []
    buf: list[str] = []
    size = 0
    page_no = 1
    for line in text.split("\n"):
        buf.append(line)
        size += len(line) + 1
        if size >= chars_per_page:
            pages.append(_page_from_lines(page_no, buf))
            page_no += 1
            buf, size = [], 0
    if buf:
        pages.append(_page_from_lines(page_no, buf))
    return pages or [Page(number=1, text="")]


def _page_from_lines(page_no: int, lines: list[str]) -> Page:
    text = "\n".join(lines)
    blocks = [
        TextBlock(text=ln, role=BlockType.PARAGRAPH, page=page_no) for ln in lines if ln.strip()
    ]
    return Page(number=page_no, text=text, blocks=blocks)


def paginate_items(
    items: list[TextBlock | Table], *, chars_per_page: int = CHARS_PER_PAGE
) -> list[Page]:
    """Lay an ordered stream of blocks/tables onto synthetic pages by char budget."""
    pages: list[Page] = []
    blocks: list[TextBlock] = []
    tables: list[Table] = []
    size = 0
    page_no = 1

    def flush() -> None:
        nonlocal blocks, tables, size
        if blocks or tables:
            text = "\n".join(b.text for b in blocks)
            pages.append(Page(number=page_no, text=text, blocks=list(blocks), tables=list(tables)))
            blocks, tables, size = [], [], 0

    for item in items:
        if isinstance(item, Table):
            item.page = page_no
            tables.append(item)
            size += sum(len(c) for row in item.rows for c in row)
        else:
            item.page = page_no
            blocks.append(item)
            size += len(item.text) + 1
        if size >= chars_per_page:
            flush()
            page_no += 1
    flush()
    return pages or [Page(number=1, text="")]
