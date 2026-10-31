"""Embedded (server-less) Qdrant vector index.

Qdrant client's *local mode* (`QdrantClient(path=...)`) runs the whole engine
in-process, persisting to a directory on disk — no server, no Docker, no cost.
This satisfies the "free & local vector DB" requirement while keeping the exact
Qdrant semantics (cosine ANN, payload filtering) the server path uses.

Qdrant's local client is synchronous, so calls run in a worker thread to avoid
blocking the event loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from pydantic import Field

from rag_core.logging import get_logger
from rag_core.models.base import DomainModel
from rag_search.facet_filters import FacetFilters, accepted_values

_log = get_logger("rag_search.vector_local")


class VectorMatch(DomainModel):
    chunk_id: str
    score: float
    payload: dict[str, Any] = Field(default_factory=dict)


class LocalVectorIndex:
    def __init__(self, path: str, *, collection: str, dim: int) -> None:
        from qdrant_client import QdrantClient

        # Embedded mode — persists to `path`, single-process (file-locked).
        try:
            self._client = QdrantClient(path=path)
        except RuntimeError as exc:
            if "already accessed" in str(exc).lower():
                raise RuntimeError(
                    f"The local index at '{path}' is already in use by another process. "
                    "Embedded Qdrant allows only ONE process at a time — stop the other "
                    "search server or indexing script and retry. If nothing is running, "
                    f"delete the stale lock file '{path}/.lock' and retry."
                ) from exc
            raise
        self._collection = collection
        self._dim = dim

    def _ensure_sync(self) -> None:
        from qdrant_client import models

        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=self._dim, distance=models.Distance.COSINE
                ),
            )
            _log.info("local_collection_created", collection=self._collection, dim=self._dim)

    async def ensure(self) -> None:
        await asyncio.to_thread(self._ensure_sync)

    async def reset(self) -> None:
        def _reset() -> None:
            from qdrant_client import models

            # In local mode, delete_collection + recreate can retain storage, so
            # clear points explicitly with a match-all filter (works in both modes).
            self._ensure_sync()
            self._client.delete(
                collection_name=self._collection,
                points_selector=models.FilterSelector(filter=models.Filter()),
            )

        await asyncio.to_thread(_reset)

    async def delete_document(self, document_id: str) -> None:
        """Remove every chunk/vector belonging to one document (e.g. soft-deleted in
        the system of record). No-op if the document was never indexed."""

        def _delete() -> None:
            from qdrant_client import models

            if not self._client.collection_exists(self._collection):
                return
            self._client.delete(
                collection_name=self._collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id", match=models.MatchValue(value=document_id)
                            )
                        ]
                    )
                ),
            )

        await asyncio.to_thread(_delete)

    async def set_document_payload(
        self, document_id: str, fields: Mapping[str, Any]
    ) -> None:
        """Merge facet fields into every point of one document, keeping its vectors.

        The counterpart to ``KeywordIndex.update_payload`` — lets an enrichment pass add
        a facet to an already-built index without re-embedding anything. A ``None`` value
        deletes that key from the points instead of storing a null.
        """
        if not fields:
            return

        def _set() -> None:
            from qdrant_client import models

            if not self._client.collection_exists(self._collection):
                return
            selector = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id)
                    )
                ]
            )
            if setting := {k: v for k, v in fields.items() if v is not None}:
                self._client.set_payload(
                    collection_name=self._collection, payload=setting, points=selector
                )
            if clearing := [k for k, v in fields.items() if v is None]:
                self._client.delete_payload(
                    collection_name=self._collection, keys=clearing, points=selector
                )

        await asyncio.to_thread(_set)

    async def upsert(self, points: list[tuple[str, list[float], dict[str, Any]]]) -> int:
        if not points:
            return 0

        def _upsert() -> int:
            from qdrant_client import models

            structs = [
                models.PointStruct(id=cid, vector=vec, payload=payload)
                for cid, vec, payload in points
            ]
            self._client.upsert(collection_name=self._collection, points=structs)
            return len(structs)

        return await asyncio.to_thread(_upsert)

    async def search(
        self, vector: list[float], *, limit: int, filters: FacetFilters | None = None
    ) -> list[VectorMatch]:
        facets = {k: vals for k, v in (filters or {}).items() if (vals := accepted_values(v))}

        def _search() -> list[VectorMatch]:
            from qdrant_client import models

            qfilter = (
                models.Filter(
                    must=[
                        # One value -> equality; several -> any-of, so a request
                        # scoped to "INGREDIENTSPEC,PIF" sees both.
                        models.FieldCondition(
                            key=k,
                            match=(
                                models.MatchValue(value=vals[0])
                                if len(vals) == 1
                                else models.MatchAny(any=list(vals))
                            ),
                        )
                        for k, vals in facets.items()
                    ]
                )
                if facets
                else None
            )
            resp = self._client.query_points(
                collection_name=self._collection,
                query=vector,
                limit=limit,
                with_payload=True,
                query_filter=qfilter,
            )
            return [
                VectorMatch(chunk_id=str(p.id), score=float(p.score), payload=dict(p.payload or {}))
                for p in resp.points
            ]

        return await asyncio.to_thread(_search)

    async def fetch_document(
        self, document_id: str, *, namespace: str = "default", limit: int = 60
    ) -> list[dict[str, Any]]:
        """All chunk payloads for one document, ordered by page — for expanding a
        retrieved hit to full-document context so answers don't miss info that lives
        in a neighbouring chunk."""

        def _scroll() -> list[dict[str, Any]]:
            from qdrant_client import models

            if not self._client.collection_exists(self._collection):
                return []
            points, _ = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=document_id)
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
            )
            payloads = []
            for p in points:
                pl = dict(p.payload or {})
                pl["_chunk_id"] = str(p.id)
                payloads.append(pl)
            payloads.sort(key=lambda pl: (pl.get("page_start") or 0, pl.get("_chunk_id", "")))
            return payloads

        return await asyncio.to_thread(_scroll)

    async def count(self) -> int:
        def _count() -> int:
            if not self._client.collection_exists(self._collection):
                return 0
            return self._client.count(self._collection, exact=True).count

        return await asyncio.to_thread(_count)

    async def aclose(self) -> None:
        await asyncio.to_thread(self._client.close)
