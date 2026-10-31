"""Enumerations shared across the domain model.

StrEnum is used throughout so values serialise to plain strings in JSON (SQS
messages, Qdrant payloads, OpenSearch docs) and remain human-readable in logs and
Neo4j, while still being type-checked in Python.
"""

from __future__ import annotations

from enum import StrEnum


class DocumentType(StrEnum):
    """Source document format. Drives which processing extractor is selected."""

    PDF = "pdf"
    SCANNED_PDF = "scanned_pdf"  # image-only PDF requiring OCR
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    XML = "xml"
    IMAGE = "image"
    HTML = "html"
    TXT = "txt"
    UNKNOWN = "unknown"


class DocumentStatus(StrEnum):
    """Lifecycle state of a document as it moves through the write path.

    The progression is monotonic on the happy path; FAILED/QUARANTINED are
    terminal-until-retried, DELETED is terminal (tombstone retained for audit).
    """

    DISCOVERED = "discovered"   # found in S3, not yet queued
    QUEUED = "queued"           # enqueued for processing
    PROCESSING = "processing"   # extraction in progress
    PROCESSED = "processed"     # text/tables/layout extracted
    CHUNKED = "chunked"
    METADATA_EXTRACTED = "metadata_extracted"
    EMBEDDED = "embedded"
    INDEXED = "indexed"         # searchable end-to-end
    FAILED = "failed"           # retriable failure
    QUARANTINED = "quarantined" # repeated failure; needs human review
    DELETED = "deleted"         # tombstoned


class ChunkStrategy(StrEnum):
    """Chunking strategy that produced a chunk. A single document yields chunks
    under several strategies simultaneously (e.g. parent + child + semantic)."""

    PARENT = "parent"
    CHILD = "child"
    SEMANTIC = "semantic"
    PAGE = "page"
    SECTION = "section"
    HEADING = "heading"
    WINDOW = "window"


class EmbeddingModel(StrEnum):
    """Embedding model identity. Each maps to a named vector in Qdrant."""

    TITAN = "titan"        # amazon.titan-embed-text-v2 (dense, Bedrock)
    BGE_M3 = "bge_m3"      # BAAI/bge-m3 (dense, multilingual)
    COLBERT = "colbert"    # late-interaction multi-vector


class Language(StrEnum):
    """ISO-639-1 subset commonly seen in compliance corpora; EXTEND as needed."""

    EN = "en"
    FR = "fr"
    DE = "de"
    ES = "es"
    IT = "it"
    PT = "pt"
    ZH = "zh"
    JA = "ja"
    UNKNOWN = "unknown"


class PIIType(StrEnum):
    """Categories of detected personally identifiable information."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    PERSON = "person"
    ADDRESS = "address"
    IBAN = "iban"
    PASSPORT = "passport"
    DATE_OF_BIRTH = "date_of_birth"


class DocumentClass(StrEnum):
    """High-level functional classification used for routing and filtering."""

    POLICY = "policy"
    REGULATION = "regulation"
    CERTIFICATE = "certificate"
    TECHNICAL_MANUAL = "technical_manual"
    DESIGN_DOCUMENT = "design_document"
    WORKFLOW = "workflow"
    STANDARD = "standard"
    REPORT = "report"
    SPECIFICATION = "specification"
    OTHER = "other"


class NodeType(StrEnum):
    """Knowledge-graph node labels."""

    DOCUMENT = "Document"
    SECTION = "Section"
    PRODUCT = "Product"
    INGREDIENT = "Ingredient"
    COMPANY = "Company"
    SUPPLIER = "Supplier"
    COUNTRY = "Country"
    REGULATION = "Regulation"
    STANDARD = "Standard"
    CERTIFICATE = "Certificate"


class EdgeType(StrEnum):
    """Knowledge-graph relationship types."""

    REFERENCES = "REFERENCES"
    BELONGS_TO = "BELONGS_TO"
    CONTAINS = "CONTAINS"
    REGULATES = "REGULATES"
    SUPPLIED_BY = "SUPPLIED_BY"
    RELATED_TO = "RELATED_TO"
    DERIVED_FROM = "DERIVED_FROM"
