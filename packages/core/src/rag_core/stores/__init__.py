"""Store adapters: vector (Qdrant), full-text (OpenSearch), graph (Neo4j).

Each store sits behind a store-agnostic interface so the retrieval/indexing layers
depend on the *capability*, not the vendor. This is what makes the "no Pinecone"
constraint enforceable and the backend swappable: only the adapter changes.

This module currently ships the vector-store interface + Qdrant implementation;
full-text and graph adapters land with the read path.
"""

from __future__ import annotations

from rag_core.stores.vector import (
    CollectionConfig,
    FieldCondition,
    MatchOp,
    MetadataFilter,
    QdrantVectorStore,
    SearchHit,
    VectorParams,
    VectorPoint,
    VectorStore,
)

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
