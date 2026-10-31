"""Chunk model — the retrievable unit.

Parent-child is first-class: small *child* chunks are embedded and searched
(precise matching), but at answer time we expand to their *parent* chunk (richer
context for the LLM). ``parent_id`` wires the relationship; the indexer stores
children as searchable points and parents as retrievable-but-not-searched context.

``metadata_payload`` is the denormalised, filterable facet set copied from
``DocumentMetadata`` so a single Qdrant query can both vector-match and filter.
"""

from __future__ import annotations

from pydantic import Field

from rag_core.models.base import TimestampedModel
from rag_core.models.enums import ChunkStrategy


class Chunk(TimestampedModel):
    """A unit of retrievable text with full provenance back to its document."""

    id: str
    document_id: str
    namespace: str = "default"

    # Parent-child relationship (None for top-level / parent chunks)
    parent_id: str | None = None
    is_parent: bool = False

    strategy: ChunkStrategy
    order_index: int = Field(default=0, ge=0)  # position within its strategy stream

    text: str

    # Provenance for citations
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    section_path: list[str] = Field(
        default_factory=list,
        description="Ordered heading trail, e.g. ['3. Safety', '3.2 Handling'].",
    )
    heading: str | None = None
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)

    token_count: int = Field(default=0, ge=0)

    # Denormalised filterable metadata (see DocumentMetadata.filterable_payload)
    metadata_payload: dict[str, object] = Field(default_factory=dict)

    @property
    def citation(self) -> str:
        """Human-readable source locator for responses."""
        loc = self.section_path[-1] if self.section_path else self.heading
        pages = (
            f"p.{self.page_start}"
            if self.page_start and self.page_start == self.page_end
            else f"pp.{self.page_start}-{self.page_end}"
            if self.page_start and self.page_end
            else None
        )
        parts = [self.document_id]
        if loc:
            parts.append(loc)
        if pages:
            parts.append(pages)
        return " · ".join(parts)
