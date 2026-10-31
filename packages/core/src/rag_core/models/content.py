"""Extracted content artifacts produced by the Document Processing Agent.

This is the structured intermediate between raw bytes and chunks. Keeping layout
(blocks, headings, footers, tables, TOC, page numbers) explicit — rather than
collapsing to a flat string — is what lets downstream chunking honour document
hierarchy and lets citations point at exact pages/sections.
"""

from __future__ import annotations

from pydantic import Field

from rag_core.models.base import DomainModel
from rag_core.models.enums import Language


class BoundingBox(DomainModel):
    """Normalised [0,1] page coordinates for a layout element (citations/UI)."""

    x0: float = Field(ge=0.0, le=1.0)
    y0: float = Field(ge=0.0, le=1.0)
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)


class BlockType:
    """String constants for text-block roles.

    A plain constants holder (not a Pydantic model/enum) so ``role`` stays a free
    ``str`` — extractors from different libraries emit varied role names and we
    don't want validation to reject an unforeseen one.
    """

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    HEADER = "header"
    FOOTER = "footer"
    CAPTION = "caption"


class TextBlock(DomainModel):
    """A contiguous text region with layout role and provenance."""

    text: str
    role: str = BlockType.PARAGRAPH  # one of BlockType.*
    page: int = Field(ge=1)
    heading_level: int | None = None  # 1..6 when role == heading
    bbox: BoundingBox | None = None
    from_ocr: bool = False


class Table(DomainModel):
    """An extracted table preserved as structured rows plus rendered forms.

    ``markdown`` is what gets embedded/searched; ``rows`` preserves structure for
    precise cell-level downstream use."""

    page: int = Field(ge=1)
    caption: str | None = None
    rows: list[list[str]] = Field(default_factory=list)
    markdown: str = ""
    bbox: BoundingBox | None = None


class ImageRef(DomainModel):
    """A figure/flowchart/certificate image, with OCR'd text if any."""

    page: int = Field(ge=1)
    ocr_text: str | None = None
    caption: str | None = None
    bbox: BoundingBox | None = None
    storage_uri: str | None = None  # where the cropped image is persisted


class TOCEntry(DomainModel):
    title: str
    level: int = Field(ge=1)
    page: int = Field(ge=1)


class Page(DomainModel):
    """A single page's extracted content."""

    number: int = Field(ge=1)
    text: str = ""
    blocks: list[TextBlock] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    images: list[ImageRef] = Field(default_factory=list)
    is_scanned: bool = False


class ExtractedContent(DomainModel):
    """The full processing output for one document."""

    document_id: str
    pages: list[Page] = Field(default_factory=list)
    toc: list[TOCEntry] = Field(default_factory=list)
    language: Language = Language.UNKNOWN
    has_ocr: bool = False

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        """Concatenated page text — convenience for whole-doc operations
        (language detection, classification). Chunking uses structure, not this."""
        return "\n\n".join(p.text for p in self.pages if p.text)
