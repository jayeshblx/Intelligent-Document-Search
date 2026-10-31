"""Vendor-neutral vector-store interface and a small filter DSL.

Why an interface: the retrieval and indexing layers must not import qdrant-client
types directly, or swapping/augmenting the backend would ripple everywhere. The
DSL here (``MetadataFilter`` / ``FieldCondition``) expresses the metadata filters
the platform needs (eq/in/range/exists) independent of any backend's query model;
each adapter translates it natively.

Multi-vector is first-class: a ``VectorPoint`` carries several named vectors
(titan/bge_m3/colbert), matching how chunks are embedded.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum
from typing import Any

from pydantic import Field

from rag_core.models.base import DomainModel


class VectorDistance(StrEnum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLID = "euclid"


class MatchOp(StrEnum):
    """Per-field match operators supported by the filter DSL."""

    EQ = "eq"
    IN = "in"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    EXISTS = "exists"
    TEXT = "text"  # full-text-ish match on a keyword/text payload field


class FieldCondition(DomainModel):
    """A single metadata predicate, e.g. country == 'DE' or effective_date >= X."""

    field: str
    op: MatchOp
    value: Any = None


class MetadataFilter(DomainModel):
    """Boolean combination of conditions (Qdrant/OpenSearch bool-query shaped).

    ``must`` = AND, ``should`` = OR, ``must_not`` = negation. An empty filter
    matches everything.
    """

    must: list[FieldCondition] = Field(default_factory=list)
    should: list[FieldCondition] = Field(default_factory=list)
    must_not: list[FieldCondition] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.must or self.should or self.must_not)

    @classmethod
    def eq(cls, **fields: Any) -> MetadataFilter:
        """Convenience: AND of equality conditions. ``MetadataFilter.eq(country='DE')``."""
        return cls(must=[FieldCondition(field=k, op=MatchOp.EQ, value=v) for k, v in fields.items()])


class VectorParams(DomainModel):
    """Configuration for one named vector space within a collection."""

    name: str
    dim: int = Field(ge=1)
    distance: VectorDistance = VectorDistance.COSINE
    # True for ColBERT-style late interaction (multiple vectors per point).
    multivector: bool = False
    # Store vectors on disk (mmap) to keep RAM bounded at 100M scale.
    on_disk: bool = True


class CollectionConfig(DomainModel):
    """Full collection definition: named vectors + payload indexes.

    ``payload_indexes`` are critical for performance: filtering 100M points on
    ``country``/``effective_date`` without an index is a full scan. Each entry is
    ``(field_name, schema_type)`` where schema_type ∈ {keyword,integer,float,
    datetime,bool,text}.
    """

    name: str
    vectors: list[VectorParams]
    payload_indexes: list[tuple[str, str]] = Field(default_factory=list)
    # HNSW + quantization knobs surfaced for prod tuning.
    hnsw_m: int = 16
    hnsw_ef_construct: int = 128
    shard_number: int = 1
    replication_factor: int = 1
    use_scalar_quantization: bool = True


class VectorPoint(DomainModel):
    """A point to upsert: id + one-or-more named vectors + filterable payload."""

    id: str
    # name -> dense vector (list[float]) or multivector (list[list[float]])
    vectors: dict[str, list[float] | list[list[float]]]
    payload: dict[str, Any] = Field(default_factory=dict)


class SearchHit(DomainModel):
    """A search result: point id, similarity score, and payload."""

    id: str
    score: float
    payload: dict[str, Any] = Field(default_factory=dict)


class VectorStore(ABC):
    """Abstract async vector store. Adapters implement against a real backend."""

    @abstractmethod
    async def ensure_collection(self, config: CollectionConfig) -> None:
        """Create the collection + payload indexes if absent (idempotent)."""

    @abstractmethod
    async def upsert(self, collection: str, points: Sequence[VectorPoint], *, batch_size: int = 256) -> int:
        """Upsert points (idempotent on id). Returns the number written."""

    @abstractmethod
    async def search(
        self,
        collection: str,
        *,
        using: str,
        query: list[float] | list[list[float]],
        limit: int = 10,
        query_filter: MetadataFilter | None = None,
        with_payload: bool = True,
    ) -> list[SearchHit]:
        """Nearest-neighbour search in the ``using`` vector space, with optional
        metadata filtering."""

    @abstractmethod
    async def retrieve(
        self, collection: str, ids: Sequence[str], *, with_payload: bool = True
    ) -> list[VectorPoint]:
        """Fetch points by id (e.g. parent-chunk expansion)."""

    @abstractmethod
    async def delete(self, collection: str, ids: Sequence[str]) -> None:
        """Delete points by id."""

    @abstractmethod
    async def delete_by_filter(self, collection: str, query_filter: MetadataFilter) -> None:
        """Delete all points matching a filter (e.g. all chunks of a document)."""

    @abstractmethod
    async def count(self, collection: str, query_filter: MetadataFilter | None = None) -> int:
        """Count points (optionally filtered)."""

    @abstractmethod
    async def create_snapshot(self, collection: str) -> str:
        """Snapshot the collection for versioning/backup. Returns snapshot name."""

    @abstractmethod
    async def health(self) -> bool:
        """Liveness probe for readiness checks."""

    @abstractmethod
    async def aclose(self) -> None:
        """Release the underlying connection."""
