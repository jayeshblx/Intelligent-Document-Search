"""Hybrid retrieval (platform capabilities #9 + #10).

Runs dense (embedded Qdrant) and sparse (SQLite FTS5) retrieval concurrently,
fuses them with RRF, and hydrates the top results into ``Passage`` objects that
tell the UI *which document*, *which page*, and *why it matched* (vector/keyword/
both).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from rag_core.logging import get_logger
from rag_search.config import SearchSettings
from rag_search.embedder import Embedder
from rag_search.facet_filters import FacetFilters
from rag_search.fusion import reciprocal_rank_fusion
from rag_search.models import Passage, SearchResponse
from rag_search.stores import KeywordSearchIndex, VectorIndex

_log = get_logger("rag_search.retrieval")

# Signs a source S3 key into a downloadable URL (async, local crypto).
Presigner = Callable[[str], Awaitable[str | None]]


class SearchService:
    def __init__(
        self,
        *,
        embedder: Embedder,
        vector: VectorIndex,
        keyword: KeywordSearchIndex,
        settings: SearchSettings,
        presign: Presigner | None = None,
    ) -> None:
        self._embedder = embedder
        self._vector = vector
        self._keyword = keyword
        self._settings = settings
        self._presign = presign

    async def search(
        self,
        query: str,
        *,
        namespace: str = "default",
        top_k: int | None = None,
        filters: FacetFilters | None = None,
    ) -> SearchResponse:
        started = time.perf_counter()
        query = query.strip()
        if not query:
            return SearchResponse(query=query, took_ms=0.0, results=[])

        k = top_k or self._settings.top_k
        cand = self._settings.candidate_k

        # Dense + sparse retrieval concurrently (both honour the metadata filter).
        query_vector = await self._embedder.embed_query(query)
        vector_res, keyword_res = await asyncio.gather(
            self._vector.search(query_vector, limit=cand, filters=filters),
            self._keyword.search(query, limit=cand, filters=filters),
            return_exceptions=True,
        )
        # One retriever failing (a timed-out or cold Atlas index, say) degrades the
        # answer to the other one instead of failing the whole request.
        if isinstance(vector_res, BaseException) and isinstance(keyword_res, BaseException):
            raise vector_res
        vector_hits: list[Any] = []
        keyword_hits: list[tuple[str, float]] = []
        if isinstance(vector_res, BaseException):
            _log.warning("vector_search_failed_keyword_only", error=str(vector_res)[:300])
        else:
            vector_hits = vector_res
        if isinstance(keyword_res, BaseException):
            _log.warning("keyword_search_failed_vector_only", error=str(keyword_res)[:300])
        else:
            keyword_hits = keyword_res

        vector_ids = [h.chunk_id for h in vector_hits]
        keyword_ids = [cid for cid, _ in keyword_hits]
        vector_set, keyword_set = set(vector_ids), set(keyword_ids)
        vector_payloads = {h.chunk_id: h.payload for h in vector_hits}

        fused = reciprocal_rank_fusion([vector_ids, keyword_ids], k=self._settings.rrf_k)[:k]
        top_ids = [cid for cid, _ in fused]

        hydrated = await self._keyword.get(top_ids)

        results: list[Passage] = []
        for chunk_id, score in fused:
            if chunk_id in hydrated:
                text, payload = hydrated[chunk_id]
            elif chunk_id in vector_payloads:  # fallback to vector-stored payload
                payload = vector_payloads[chunk_id]
                text = str(payload.get("text", ""))
            else:
                continue
            results.append(_to_passage(chunk_id, score, text, payload, vector_set, keyword_set))

        await self._attach_download_urls(results)

        return SearchResponse(
            query=query,
            took_ms=round((time.perf_counter() - started) * 1000, 2),
            results=results,
        )

    async def exact_lookup(self, identifiers: list[str], *, limit: int = 8) -> list[Passage]:
        """Documents that literally contain a distinctive identifier (content or filename).

        Bypasses ranked retrieval, which buries exact IDs, so "Delivery No 1032658" or an
        invoice number in a filename reliably surfaces its document.
        """
        hits = await self._keyword.exact_lookup(identifiers, limit=limit)
        results = [
            _to_passage(cid, 1.0, text, payload, set(), {cid})  # keyword-only, top score
            for cid, text, payload in hits
        ]
        await self._attach_download_urls(results)
        return results

    async def expand_document(self, document_id: str, seed_text: str, *, cap: int = 4000) -> str:
        """Full-document context for one hit: the matched chunk first, then its
        sibling chunks (up to ``cap`` chars). Fixes answers that miss information
        living in a neighbouring chunk (e.g. an approval value under a header)."""
        seed = seed_text.strip()
        parts, total = ([seed], len(seed)) if seed else ([], 0)
        try:
            chunks = await self._vector.fetch_document(document_id, limit=60)
        except Exception as exc:  # never let expansion break the answer
            _log.warning("expand_document_failed", document_id=document_id, error=str(exc))
            return seed
        # Payloads carry no text (it lives in the keyword store), so hydrate the
        # sibling chunks by id. Falls back to any text a payload does carry, so an
        # index published before that change still expands.
        sibling_ids = [str(p.get("_chunk_id") or "") for p in chunks]
        hydrated = await self._keyword.get([cid for cid in sibling_ids if cid])
        for payload in chunks:
            cid = str(payload.get("_chunk_id") or "")
            text = (hydrated[cid][0] if cid in hydrated else str(payload.get("text", ""))).strip()
            if not text or text in parts:
                continue
            if total + len(text) + 1 > cap:
                break
            parts.append(text)
            total += len(text) + 1
        return "\n".join(parts)

    async def _attach_download_urls(self, results: list[Passage]) -> None:
        """Sign one presigned URL per distinct source document and attach it."""
        if not self._presign:
            return
        keys = {r.source_key for r in results if r.source_key}
        if not keys:
            return
        key_list = list(keys)
        try:
            urls = await asyncio.gather(*(self._presign(k) for k in key_list))
        except Exception as exc:
            _log.warning("presign_failed", error=str(exc))
            return
        by_key = dict(zip(key_list, urls, strict=True))
        for r in results:
            r.download_url = by_key.get(r.source_key)


def _to_passage(
    chunk_id: str,
    score: float,
    text: str,
    payload: dict[str, Any],
    vector_set: set[str],
    keyword_set: set[str],
) -> Passage:
    if chunk_id in vector_set and chunk_id in keyword_set:
        matched = "both"
    elif chunk_id in vector_set:
        matched = "vector"
    else:
        matched = "keyword"
    return Passage(
        chunk_id=chunk_id,
        document_id=str(payload.get("document_id", "")),
        source_key=str(payload.get("source_key", "")),
        score=round(score, 6),
        text=text,
        page_start=payload.get("page_start"),
        page_end=payload.get("page_end"),
        section=payload.get("section"),
        doc_class=payload.get("doc_class"),
        language=payload.get("language"),
        matched_by=matched,
    )
