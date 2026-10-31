"""Exact-identifier retrieval: distinctive IDs must surface their document.

Ranked search buries an exact identifier — a number is noise to the embedder and dilutes
under BM25 — so a literal lookup over content AND filename is the only reliable path.
"""

from __future__ import annotations

import pytest
from rag_search.answer import _distinctive_identifiers
from rag_search.stores import KeywordIndex


def test_extracts_only_distinctive_identifiers() -> None:
    f = _distinctive_identifiers
    # Long numbers and letter+digit codes qualify.
    assert f("Search Document in which Delivery No.: 1032658") == ["1032658"]
    assert f("find invoice 0123456789") == ["0123456789"]
    assert f("anything about STD3163") == ["STD3163"]
    assert f("Project Name A017550 Acme Foods") == ["A017550"]
    assert f("object code SUPPLIER0005") == ["SUPPLIER0005"]
    # Ordinary words and a 4-digit year must NOT — they'd match half the corpus.
    assert f("list all certificates") == []
    assert f("what does the Supplierco statement say") == []
    assert f("documents from 2024") == []


async def test_exact_lookup_finds_content_and_filename(tmp_path) -> None:
    kw = KeywordIndex(str(tmp_path / "kw.db"))
    try:
        await kw.add([
            # (chunk_id, text, payload) — identifier in BODY text.
            ("c1", "Delivery reference 1032658 was short by one box.",
             {"document_id": "d1", "source_key": "foxtrot/hash_Northwind Invoice~pdf"}),
            # identifier only in the FILENAME, not the body.
            ("c2", "Tax invoice for goods supplied.",
             {"document_id": "d2", "source_key": "foxtrot/hash_INV 0123456789 XYZ~pdf"}),
            # unrelated noise that must not match.
            ("c3", "A certificate of conformance for tomatoes.",
             {"document_id": "d3", "source_key": "foxtrot/hash_cert~pdf"}),
        ])

        # Content match.
        hits = await kw.exact_lookup(["1032658"])
        assert [h[0] for h in hits] == ["c1"]

        # Filename match — body says nothing about the number.
        hits = await kw.exact_lookup(["0123456789"])
        assert [h[0] for h in hits] == ["c2"]

        # A missing identifier returns nothing (not everything).
        assert await kw.exact_lookup(["9999999"]) == []

        # Too-short tokens are ignored (would match too much).
        assert await kw.exact_lookup(["ab", "1"]) == []
    finally:
        await kw.aclose()


@pytest.mark.parametrize("backend_note", ["fts5-or-like-fallback"])
async def test_exact_lookup_matches_multiple_identifiers(tmp_path, backend_note) -> None:
    kw = KeywordIndex(str(tmp_path / "kw2.db"))
    try:
        await kw.add([
            ("a", "code STD3163 investigation", {"document_id": "d1", "source_key": "k1"}),
            ("b", "account 115036 balance", {"document_id": "d2", "source_key": "k2"}),
        ])
        hits = await kw.exact_lookup(["STD3163", "115036"])
        assert {h[0] for h in hits} == {"a", "b"}
    finally:
        await kw.aclose()
