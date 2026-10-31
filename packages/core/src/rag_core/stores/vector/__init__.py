"""Vector store interface + Qdrant adapter."""

from __future__ import annotations

from rag_core.stores.vector.base import (
    CollectionConfig,
    FieldCondition,
    MatchOp,
    MetadataFilter,
    SearchHit,
    VectorParams,
    VectorPoint,
    VectorStore,
)
from rag_core.stores.vector.qdrant_store import QdrantVectorStore

__all__ = [
    "CollectionConfig",
    "FieldCondition",
    "MatchOp",
    "MetadataFilter",
    "QdrantVectorStore",
    "SearchHit",
    "VectorParams",
    "VectorPoint",
    "VectorStore",
]
