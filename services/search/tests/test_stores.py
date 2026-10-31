"""Tests for the local keyword (FTS5) and vector (embedded Qdrant) indexes."""

from __future__ import annotations

from rag_search.stores import KeywordIndex, LocalVectorIndex


async def test_keyword_index_add_search_get(tmp_path) -> None:
    kw = KeywordIndex(str(tmp_path / "kw.db"))
    try:
        await kw.add(
            [
                ("c1", "data encryption at rest using AES-256", {"document_id": "d1", "source_key": "a"}),
                ("c2", "supplier ACME located in Germany", {"document_id": "d2", "source_key": "b"}),
            ]
        )
        assert await kw.count() == 2

        hits = await kw.search("encryption AES", limit=5)
        assert hits and hits[0][0] == "c1"

        hydrated = await kw.get(["c1"])
        text, payload = hydrated["c1"]
        assert "encryption" in text
        assert payload["source_key"] == "a"

        # No matching tokens -> no hits.
        assert await kw.search("zzzznomatch", limit=5) == []
    finally:
        await kw.aclose()


async def test_vector_index_roundtrip(tmp_path) -> None:
    vec = LocalVectorIndex(str(tmp_path / "qdrant"), collection="t", dim=3)
    try:
        await vec.ensure()
        await vec.upsert(
            [
                ("11111111-1111-5111-8111-111111111111", [1.0, 0.0, 0.0], {"text": "x"}),
                ("22222222-2222-5222-8222-222222222222", [0.0, 1.0, 0.0], {"text": "y"}),
            ]
        )
        assert await vec.count() == 2
        matches = await vec.search([0.9, 0.1, 0.0], limit=2)
        assert matches[0].chunk_id == "11111111-1111-5111-8111-111111111111"
        assert matches[0].payload["text"] == "x"
    finally:
        await vec.aclose()


async def test_vector_index_reset(tmp_path) -> None:
    vec = LocalVectorIndex(str(tmp_path / "qdrant"), collection="t", dim=2)
    try:
        await vec.ensure()
        await vec.upsert([("33333333-3333-5333-8333-333333333333", [1.0, 0.0], {})])
        assert await vec.count() == 1
        await vec.reset()
        assert await vec.count() == 0
    finally:
        await vec.aclose()
