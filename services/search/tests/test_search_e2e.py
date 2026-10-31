"""End-to-end: index real documents through the full pipeline, then search.

Proves the core promise — a natural-language query returns the *correct document*
— using the fake (offline) embedder + real chunking, embedded Qdrant, and FTS5.
"""

from __future__ import annotations

from collections.abc import Callable

from rag_search.indexer import IndexingService
from rag_search.models import IndexRequest
from rag_search.retrieval import SearchService

Stack = Callable[[dict[str, bytes]], tuple[IndexingService, SearchService, object, list]]


async def test_index_then_search_returns_correct_document(
    build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    indexer, search, _meta, closeables = build_stack(sample_objects)
    try:
        summary = await indexer.index(IndexRequest(reset=True))
        assert summary.documents_indexed == 2
        assert summary.documents_failed == 0
        assert summary.chunks_indexed > 0

        # Semantic-ish + keyword query about encryption -> the policy doc.
        res = await search.search("how must confidential data be encrypted at rest")
        assert res.results, "expected search results"
        assert res.results[0].source_key == "foxtrot/security_policy.docx"
        assert res.took_ms >= 0

        # A different query -> the supplier doc.
        res2 = await search.search("which supplier is located in Germany with ISO certification")
        assert res2.results
        assert res2.results[0].source_key == "foxtrot/supplier_register.html"

        # Passages carry provenance for the UI.
        top = res.results[0]
        assert top.document_id
        assert top.matched_by in {"vector", "keyword", "both"}
    finally:
        for c in closeables:
            await c.aclose()


async def test_search_before_index_is_empty(build_stack: Stack) -> None:
    indexer, search, _meta, closeables = build_stack({})
    try:
        await indexer.index(IndexRequest(reset=True))  # nothing to index
        res = await search.search("anything")
        assert res.results == []
    finally:
        for c in closeables:
            await c.aclose()


async def test_one_retriever_failing_degrades_to_the_other(
    build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    """A timed-out vector (or keyword) query must not fail the whole search."""
    import pytest

    indexer, search, _meta, closeables = build_stack(sample_objects)
    try:
        await indexer.index(IndexRequest(reset=True))

        async def boom(*_args: object, **_kwargs: object) -> list:
            raise TimeoutError("operation exceeded time limit")

        real_vector_search = search._vector.search
        search._vector.search = boom  # type: ignore[method-assign]
        res = await search.search("confidential data encrypted at rest")
        assert res.results, "keyword-only results expected when vector search fails"
        assert {r.matched_by for r in res.results} == {"keyword"}

        search._vector.search = real_vector_search  # type: ignore[method-assign]
        search._keyword.search = boom  # type: ignore[method-assign]
        res = await search.search("confidential data encrypted at rest")
        assert res.results, "vector-only results expected when keyword search fails"

        search._vector.search = boom  # type: ignore[method-assign]
        with pytest.raises(TimeoutError):
            await search.search("confidential data encrypted at rest")
    finally:
        for c in closeables:
            await c.aclose()
