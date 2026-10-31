"""Document metadata — the structured facets used for filtering and graph build.

Metadata is stored separately (its own collection/index) AND denormalised onto
each chunk's Qdrant payload. Why both: the metadata service owns the canonical
record (single source of truth, independently updatable), while denormalisation
onto chunks makes metadata filtering a single vector-store query rather than a
join across systems — essential at 100M chunks.

Every field here corresponds to a supported filter facet in the query API.
"""

from __future__ import annotations

from datetime import date

from pydantic import Field

from rag_core.models.base import DomainModel
from rag_core.models.enums import DocumentClass, Language, PIIType


class Entity(DomainModel):
    """A named entity extracted from the document (feeds the knowledge graph)."""

    text: str
    type: str  # e.g. ORG, PRODUCT, REGULATION, COUNTRY, GPE, DATE
    salience: float = Field(default=0.0, ge=0.0, le=1.0)
    mentions: int = Field(default=1, ge=1)


class PIIFinding(DomainModel):
    """A detected PII span — recorded for redaction/audit, never the raw value."""

    type: PIIType
    count: int = Field(default=1, ge=1)
    sample_redacted: str | None = None  # e.g. "j***@***.com"


class DocumentMetadata(DomainModel):
    """Canonical structured metadata for a document.

    All facets are optional because real-world compliance corpora are messy;
    extraction is best-effort and confidence-scored. Required structure lives on
    ``Document`` itself, not here.
    """

    document_id: str

    # --- Bibliographic ---
    title: str | None = None
    author: str | None = None
    document_class: DocumentClass = DocumentClass.OTHER

    # --- Organisational facets (filterable) ---
    country: str | None = None
    region: str | None = None
    department: str | None = None
    business_unit: str | None = None
    product: str | None = None
    supplier: str | None = None

    # --- Regulatory facets (filterable) ---
    regulation: str | None = None
    version: str | None = None
    revision: str | None = None
    category: str | None = None
    sub_category: str | None = None

    # --- Temporal validity (filterable; drives "currently in force" queries) ---
    effective_date: date | None = None
    expiry_date: date | None = None

    # --- Content signals ---
    language: Language = Language.UNKNOWN
    keywords: list[str] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    pii: list[PIIFinding] = Field(default_factory=list)

    # --- Extraction provenance ---
    extraction_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    extra: dict[str, str] = Field(
        default_factory=dict,
        description="Open-ended facets not yet promoted to first-class fields.",
    )

    def filterable_payload(self) -> dict[str, str | list[str] | None]:
        """Flatten to the subset denormalised onto each chunk for store-side
        metadata filtering. Dates are ISO strings for range filters."""
        return {
            "country": self.country,
            "region": self.region,
            "department": self.department,
            "business_unit": self.business_unit,
            "product": self.product,
            "supplier": self.supplier,
            "regulation": self.regulation,
            "version": self.version,
            "revision": self.revision,
            "category": self.category,
            "sub_category": self.sub_category,
            "document_class": self.document_class.value,
            "language": self.language.value,
            "effective_date": self.effective_date.isoformat() if self.effective_date else None,
            "expiry_date": self.expiry_date.isoformat() if self.expiry_date else None,
            "keywords": self.keywords,
        }
