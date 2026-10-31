"""Qdrant implementation of the ``VectorStore`` interface.

Qdrant was chosen for: named vectors (Titan + BGE-M3 + ColBERT on one point),
payload-indexed metadata filtering, on-disk vectors + scalar quantization (RAM
stays bounded at 100M points), collections as namespaces, and snapshots for
versioning. This adapter is the only place qdrant-client types appear.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from rag_core.config import QdrantSettings
from rag_core.logging import get_logger
from rag_core.resilience import RetryableError, async_retry
from rag_core.stores.vector.base import (
    CollectionConfig,
    FieldCondition,
    MatchOp,
    MetadataFilter,
    SearchHit,
    VectorDistance,
    VectorParams,
    VectorPoint,
    VectorStore,
)

_log = get_logger("rag_core.stores.qdrant")

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ].*)?$")

_DISTANCE_MAP = {
    VectorDistance.COSINE: models.Distance.COSINE,
    VectorDistance.DOT: models.Distance.DOT,
    VectorDistance.EUCLID: models.Distance.EUCLID,
}

# Map our payload schema-type strings to Qdrant index schema types.
_SCHEMA_MAP = {
    "keyword": models.PayloadSchemaType.KEYWORD,
    "integer": models.PayloadSchemaType.INTEGER,
    "float": models.PayloadSchemaType.FLOAT,
    "datetime": models.PayloadSchemaType.DATETIME,
    "bool": models.PayloadSchemaType.BOOL,
    "text": models.PayloadSchemaType.TEXT,
}


class QdrantVectorStore(VectorStore):
    """Async Qdrant adapter. One instance is shared across a service."""

    def __init__(self, settings: QdrantSettings) -> None:
        self._settings = settings
        self._client = AsyncQdrantClient(
            url=settings.url,
            api_key=settings.api_key or None,
            prefer_grpc=settings.prefer_grpc,
            timeout=settings.timeout_s,
        )

    # ---- schema ----------------------------------------------------------
    def _vector_config(self, vectors: list[VectorParams]) -> dict[str, models.VectorParams]:
        cfg: dict[str, models.VectorParams] = {}
        for v in vectors:
            multi = (
                models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM)
                if v.multivector
                else None
            )
            cfg[v.name] = models.VectorParams(
                size=v.dim,
                distance=_DISTANCE_MAP[v.distance],
                on_disk=v.on_disk,
                multivector_config=multi,
            )
        return cfg

    async def ensure_collection(self, config: CollectionConfig) -> None:
        if await self._client.collection_exists(config.name):
            _log.debug("collection_exists", collection=config.name)
        else:
            quantization = (
                models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8, always_ram=True
                    )
                )
                if config.use_scalar_quantization
                else None
            )
            await self._client.create_collection(
                collection_name=config.name,
                vectors_config=self._vector_config(config.vectors),
                hnsw_config=models.HnswConfigDiff(
                    m=config.hnsw_m, ef_construct=config.hnsw_ef_construct
                ),
                quantization_config=quantization,
                shard_number=config.shard_number,
                replication_factor=config.replication_factor,
                on_disk_payload=True,
            )
            _log.info("collection_created", collection=config.name, vectors=len(config.vectors))

        # Payload indexes are idempotent to (re)create and essential for filtering.
        for field, schema in config.payload_indexes:
            schema_type = _SCHEMA_MAP.get(schema)
            if schema_type is None:
                raise ValueError(f"unknown payload schema type {schema!r} for field {field!r}")
            try:
                await self._client.create_payload_index(
                    collection_name=config.name, field_name=field, field_schema=schema_type
                )
            except Exception as exc:
                _log.debug("payload_index_skip", field=field, reason=str(exc))

    # ---- writes ----------------------------------------------------------
    @async_retry(max_attempts=4)
    async def upsert(
        self, collection: str, points: Sequence[VectorPoint], *, batch_size: int = 256
    ) -> int:
        if not points:
            return 0
        written = 0
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            structs = [
                models.PointStruct(id=p.id, vector=p.vectors, payload=p.payload) for p in batch
            ]
            try:
                await self._client.upsert(collection_name=collection, points=structs, wait=True)
            except Exception as exc:
                raise RetryableError(f"qdrant upsert failed: {exc}") from exc
            written += len(structs)
        return written

    @async_retry(max_attempts=4)
    async def delete(self, collection: str, ids: Sequence[str]) -> None:
        if not ids:
            return
        try:
            await self._client.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(points=list(ids)),
                wait=True,
            )
        except Exception as exc:
            raise RetryableError(f"qdrant delete failed: {exc}") from exc

    @async_retry(max_attempts=4)
    async def delete_by_filter(self, collection: str, query_filter: MetadataFilter) -> None:
        try:
            await self._client.delete(
                collection_name=collection,
                points_selector=models.FilterSelector(filter=self._to_qdrant_filter(query_filter)),
                wait=True,
            )
        except Exception as exc:
            raise RetryableError(f"qdrant delete_by_filter failed: {exc}") from exc

    # ---- reads -----------------------------------------------------------
    @async_retry(max_attempts=4)
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
        qfilter = (
            self._to_qdrant_filter(query_filter)
            if query_filter and not query_filter.is_empty
            else None
        )
        try:
            resp = await self._client.query_points(
                collection_name=collection,
                query=query,
                using=using,
                limit=limit,
                query_filter=qfilter,
                with_payload=with_payload,
            )
        except Exception as exc:
            raise RetryableError(f"qdrant search failed: {exc}") from exc
        return [
            SearchHit(id=str(p.id), score=p.score, payload=dict(p.payload or {}))
            for p in resp.points
        ]

    @async_retry(max_attempts=4)
    async def retrieve(
        self, collection: str, ids: Sequence[str], *, with_payload: bool = True
    ) -> list[VectorPoint]:
        if not ids:
            return []
        try:
            records = await self._client.retrieve(
                collection_name=collection,
                ids=list(ids),
                with_payload=with_payload,
                with_vectors=False,
            )
        except Exception as exc:
            raise RetryableError(f"qdrant retrieve failed: {exc}") from exc
        return [
            VectorPoint(id=str(r.id), vectors={}, payload=dict(r.payload or {})) for r in records
        ]

    @async_retry(max_attempts=4)
    async def count(self, collection: str, query_filter: MetadataFilter | None = None) -> int:
        qfilter = (
            self._to_qdrant_filter(query_filter)
            if query_filter and not query_filter.is_empty
            else None
        )
        try:
            result = await self._client.count(
                collection_name=collection, count_filter=qfilter, exact=True
            )
        except Exception as exc:
            raise RetryableError(f"qdrant count failed: {exc}") from exc
        return result.count

    async def create_snapshot(self, collection: str) -> str:
        desc = await self._client.create_snapshot(collection_name=collection)
        name = desc.name if desc else ""
        _log.info("snapshot_created", collection=collection, snapshot=name)
        return name

    async def health(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    async def aclose(self) -> None:
        await self._client.close()

    # ---- filter translation ----------------------------------------------
    def _to_qdrant_filter(self, mf: MetadataFilter) -> models.Filter:
        must: list[Any] = []
        should: list[Any] = []
        must_not: list[Any] = []

        for cond in mf.must:
            self._route(cond, must, must_not)
        for cond in mf.should:
            # EXISTS in a should-clause has no clean single-condition mapping;
            # all other ops translate to a positive condition.
            should.append(self._positive(cond))
        for cond in mf.must_not:
            self._route(cond, must_not, must)

        return models.Filter(
            must=must or None, should=should or None, must_not=must_not or None
        )

    def _route(self, cond: FieldCondition, positive: list[Any], negative: list[Any]) -> None:
        """Place a condition into the right clause. EXISTS flips into IsEmpty on
        the opposite clause; everything else is a positive field condition."""
        if cond.op is MatchOp.EXISTS:
            negative.append(models.IsEmptyCondition(is_empty=models.PayloadField(key=cond.field)))
        else:
            positive.append(self._positive(cond))

    def _positive(self, cond: FieldCondition) -> models.FieldCondition:
        op, key, value = cond.op, cond.field, cond.value
        if op is MatchOp.EQ:
            return models.FieldCondition(key=key, match=models.MatchValue(value=value))
        if op is MatchOp.IN:
            return models.FieldCondition(key=key, match=models.MatchAny(any=list(value)))
        if op is MatchOp.TEXT:
            return models.FieldCondition(key=key, match=models.MatchText(text=str(value)))
        if op in (MatchOp.GT, MatchOp.GTE, MatchOp.LT, MatchOp.LTE):
            return self._range_condition(key, op, value)
        raise ValueError(f"unsupported match op {op} for positive clause")

    def _range_condition(self, key: str, op: MatchOp, value: Any) -> models.FieldCondition:
        bound = {"gt": None, "gte": None, "lt": None, "lte": None}
        bound[op.value] = value
        # Datetime fields need DatetimeRange (RFC3339/ISO); numbers use Range.
        if isinstance(value, str) and _ISO_DATE_RE.match(value):
            return models.FieldCondition(key=key, range=models.DatetimeRange(**bound))
        return models.FieldCondition(key=key, range=models.Range(**bound))
