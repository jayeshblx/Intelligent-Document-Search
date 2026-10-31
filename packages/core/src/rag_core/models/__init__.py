"""Domain models — the typed contracts shared across every service.

Importing from ``rag_core.models`` (rather than submodules) gives services a
stable surface even if internal module layout changes.
"""

from __future__ import annotations

from rag_core.models.base import DomainModel, TimestampedModel, utcnow
from rag_core.models.chunk import Chunk
from rag_core.models.content import (
    BlockType,
    BoundingBox,
    ExtractedContent,
    ImageRef,
    Page,
    Table,
    TextBlock,
    TOCEntry,
)
from rag_core.models.document import Document, SourceRef
from rag_core.models.embedding import ChunkEmbedding, DenseVector, MultiVector
from rag_core.models.enums import (
    ChunkStrategy,
    DocumentClass,
    DocumentStatus,
    DocumentType,
    EdgeType,
    EmbeddingModel,
    Language,
    NodeType,
    PIIType,
)
from rag_core.models.messages import DocumentEvent, PipelineMessage, TraceContext
from rag_core.models.metadata import DocumentMetadata, Entity, PIIFinding

__all__ = [
    # base
    "DomainModel",
    "TimestampedModel",
    "utcnow",
    # document
    "Document",
    "SourceRef",
    # content
    "BlockType",
    "BoundingBox",
    "ExtractedContent",
    "ImageRef",
    "Page",
    "Table",
    "TextBlock",
    "TOCEntry",
    # chunk
    "Chunk",
    # embedding
    "ChunkEmbedding",
    "DenseVector",
    "MultiVector",
    # metadata
    "DocumentMetadata",
    "Entity",
    "PIIFinding",
    # messages
    "DocumentEvent",
    "PipelineMessage",
    "TraceContext",
    # enums
    "ChunkStrategy",
    "DocumentClass",
    "DocumentStatus",
    "DocumentType",
    "EdgeType",
    "EmbeddingModel",
    "Language",
    "NodeType",
    "PIIType",
]
