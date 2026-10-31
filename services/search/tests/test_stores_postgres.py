"""Tests for the Postgres backend (vectors + keyword + metadata in one server).

These need a live Postgres with the ``vector`` and ``pg_trgm`` extensions
available — ``docker compose up -d postgres`` brings one up. The whole module
skips when that server is unreachable, so the default (local) test run is
unaffected.

Point them elsewhere with ``SEARCH_POSTGRES_DSN``.

The tests that matter most are the PARITY ones at the bottom: they run the same
scenario through the SQLite/Qdrant stack and the Postgres stack and assert the
two agree, which is what makes the swap safe.
"""

from __future__ import annotations

import datetime as dt
import os

import pytest
from rag_search.metadata_store import MetadataStore
from rag_search.models import IndexRequest
from rag_search.stores import KeywordIndex, LocalVectorIndex
from rag_search.stores.postgres import PgDatabase, PgKeywordIndex, PgVectorIndex
from rag_search.stores.postgres_metadata import PgMetadataStore

from rag_core.models.metadata import DocumentMetadata

psycopg = pytest.importorskip("psycopg", reason="psycopg not installed")

# A DEDICATED database, never the one the service indexes into: these tests drop
# and recreate tables, which would destroy a real index. Created on demand.
ADMIN_DSN = os.environ.get("SEARCH_POSTGRES_DSN", "postgresql://rag:rag@localhost:5432/rag")
TEST_DB = os.environ.get("SEARCH_POSTGRES_TEST_DB", "rag_test")
DSN = ADMIN_DSN.rsplit("/", 1)[0] + "/" + TEST_DB


def _reachable() -> bool:
    """True once the test database exists (creating it if the server is up)."""
    try:
        with psycopg.connect(ADMIN_DSN, connect_timeout=5) as conn:
            conn.autocommit = True  # CREATE DATABASE cannot run in a transaction
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)
            ).fetchone()
            if not exists:
                conn.execute(f'CREATE DATABASE "{TEST_DB}"')
        with psycopg.connect(DSN, connect_timeout=5):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _reachable(), reason=f"no Postgres at {ADMIN_DSN} (try: docker compose up -d postgres)"
)


def _fresh_db(dim: int) -> PgDatabase:
    """A PgDatabase over empty tables. Dropping first also lets each test pick its
    own embedding dimension, which is baked into the ``chunks`` column type."""
    with psycopg.connect(DSN, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS chunks, document_metadata, document_facet")
        conn.commit()
    return PgDatabase(DSN, dim=dim)


@pytest.fixture
async def pg3():
    db = _fresh_db(3)
    try:
        yield db
    finally:
        await db.aclose()


# --------------------------------------------------------------------------
# Chunk index: vectors + keyword over the one shared table
# --------------------------------------------------------------------------


async def test_keyword_index_add_search_get(pg3: PgDatabase) -> None:
    kw = PgKeywordIndex(pg3)
    await kw.add(
        [
            ("c1", "data encryption at rest using AES-256", {"document_id": "d1", "source_key": "a"}),
            ("c2", "supplier ACME located in Germany", {"document_id": "d2", "source_key": "b"}),
        ]
    )
    assert await kw.count() == 2

    hits = await kw.search("encryption AES", limit=5)
    assert hits and hits[0][0] == "c1"

    text, payload = (await kw.get(["c1"]))["c1"]
    assert "encryption" in text
    assert payload["source_key"] == "a"

    assert await kw.search("zzzznomatch", limit=5) == []


async def test_keyword_index_stems_like_fts5(pg3: PgDatabase) -> None:
    """'english' config stems, matching FTS5's porter tokenizer."""
    kw = PgKeywordIndex(pg3)
    await kw.add([("c1", "keys are rotated annually", {"document_id": "d1"})])
    assert [c for c, _ in await kw.search("rotate", limit=5)] == ["c1"]


async def test_vector_index_roundtrip(pg3: PgDatabase) -> None:
    vec = PgVectorIndex(pg3)
    await vec.ensure()
    await vec.upsert([("c1", [1.0, 0.0, 0.0], {"text": "x"}), ("c2", [0.0, 1.0, 0.0], {"text": "y"})])
    assert await vec.count() == 2

    matches = await vec.search([0.9, 0.1, 0.0], limit=2)
    assert matches[0].chunk_id == "c1"
    assert matches[0].payload["text"] == "x"
    # Cosine SIMILARITY, matching what Qdrant reports (not distance).
    assert 0.9 < matches[0].score <= 1.0


async def test_vector_index_reset(pg3: PgDatabase) -> None:
    vec = PgVectorIndex(pg3)
    await vec.ensure()
    await vec.upsert([("c1", [1.0, 0.0, 0.0], {})])
    assert await vec.count() == 1
    await vec.reset()
    assert await vec.count() == 0


async def test_both_facades_share_one_row(pg3: PgDatabase) -> None:
    """The vector upsert and the keyword add write the SAME row without clobbering
    each other — this is what makes the single-table design safe."""
    vec, kw = PgVectorIndex(pg3), PgKeywordIndex(pg3)
    await vec.ensure()
    payload = {"document_id": "d1", "source_key": "a.pdf"}
    await vec.upsert([("c1", [1.0, 0.0, 0.0], payload)])
    await kw.add([("c1", "encryption policy", payload)])

    assert await vec.count() == 1  # embedding survived the keyword write
    assert await kw.count() == 1
    assert (await kw.get(["c1"]))["c1"][0] == "encryption policy"
    assert (await vec.search([1.0, 0.0, 0.0], limit=1))[0].chunk_id == "c1"


async def test_facet_filter_and_payload_merge(pg3: PgDatabase) -> None:
    vec, kw = PgVectorIndex(pg3), PgKeywordIndex(pg3)
    await vec.ensure()
    p1 = {"document_id": "d1", "source_key": "policy.pdf", "page_start": 2}
    p2 = {"document_id": "d2", "source_key": "other.pdf"}
    await vec.upsert([("c1", [1.0, 0.0, 0.0], p1), ("c2", [0.0, 1.0, 0.0], p2)])
    await kw.add([("c1", "encryption policy", p1), ("c2", "encryption elsewhere", p2)])

    assert [c for c, _ in await kw.search("encryption", limit=5, filters={"document_id": "d1"})] == ["c1"]
    assert [m.chunk_id for m in await vec.search([0.0, 1.0, 0.0], limit=5, filters={"document_id": "d2"})] == ["c2"]

    # Enrichment reaches every chunk of the document; None removes the key.
    await vec.set_document_payload("d1", {"supplier": "ACME", "page_start": None})
    merged = (await kw.get(["c1"]))["c1"][1]
    assert merged["supplier"] == "ACME"
    assert "page_start" not in merged
    assert await kw.update_payload("d1", {"status": "Draft"}) == 1


async def test_exact_lookup_matches_text_and_filename(pg3: PgDatabase) -> None:
    kw = PgKeywordIndex(pg3)
    await kw.add(
        [
            ("c1", "see delivery note 1032658 for details", {"document_id": "d1", "source_key": "a.pdf"}),
            ("c2", "nothing distinctive here", {"document_id": "d2", "source_key": "inv-998877.pdf"}),
        ]
    )
    assert {c for c, _, _ in await kw.exact_lookup(["1032658"], limit=5)} == {"c1"}
    assert {c for c, _, _ in await kw.exact_lookup(["998877"], limit=5)} == {"c2"}
    assert await kw.exact_lookup(["12"], limit=5) == []  # under the 4-char floor


async def test_fetch_document_orders_by_page(pg3: PgDatabase) -> None:
    vec = PgVectorIndex(pg3)
    await vec.ensure()
    await vec.upsert(
        [
            ("c2", [1.0, 0.0, 0.0], {"document_id": "d1", "page_start": 5}),
            ("c1", [1.0, 0.0, 0.0], {"document_id": "d1", "page_start": 1}),
        ]
    )
    assert [c["_chunk_id"] for c in await vec.fetch_document("d1")] == ["c1", "c2"]


async def test_delete_document_removes_chunks(pg3: PgDatabase) -> None:
    vec, kw = PgVectorIndex(pg3), PgKeywordIndex(pg3)
    await vec.ensure()
    await kw.add([("c1", "a", {"document_id": "d1"}), ("c2", "b", {"document_id": "d2"})])
    assert await kw.delete_document("d1") == 1
    assert await kw.count() == 1


async def test_embedding_dim_mismatch_is_explicit(pg3: PgDatabase) -> None:
    """A changed SEARCH_EMBED_DIM must fail with an actionable message, not a
    confusing insert error later on."""
    await PgVectorIndex(pg3).ensure()  # creates chunks at dim 3
    other = PgDatabase(DSN, dim=1024)
    try:
        with pytest.raises(RuntimeError, match="3-dimensional vectors"):
            await other.fetchone("SELECT 1")
    finally:
        await other.aclose()


async def test_aclose_is_idempotent(pg3: PgDatabase) -> None:
    """All three stores share one pool and each calls aclose() on shutdown."""
    db = _fresh_db(3)
    await PgVectorIndex(db).ensure()
    await db.aclose()
    await db.aclose()


# --------------------------------------------------------------------------
# Metadata store
# --------------------------------------------------------------------------


async def _seed_metadata(md) -> None:
    await md.put(
        DocumentMetadata(document_id="d1", title="AMS Encryption Policy", expiry_date=dt.date(2026, 5, 1)),
        namespace="default",
        source_key="policy-1032658.pdf",
        facets={"status": "Draft", "object_code": "SUPPLIER0005", "business_object": "Policy"},
    )
    await md.put(
        DocumentMetadata(document_id="d2", title="ACME Supplier Record"),
        namespace="default",
        source_key="supplier-acme.pdf",
        facets={"status": "Active", "object_code": "SUPPLIER0009", "business_object": "Supplier"},
    )


async def test_metadata_put_get_count(pg3: PgDatabase) -> None:
    md = PgMetadataStore(pg3)
    await _seed_metadata(md)
    assert await md.count() == 2
    got = await md.get("d1")
    assert got is not None and got.title == "AMS Encryption Policy"
    assert await md.get("missing") is None
    assert sorted(await md.all_documents()) == [
        ("d1", "policy-1032658.pdf"),
        ("d2", "supplier-acme.pdf"),
    ]


async def test_metadata_facets_cascade(pg3: PgDatabase) -> None:
    md = PgMetadataStore(pg3)
    await _seed_metadata(md)
    assert (await md.facets("default"))["status"] == ["Active", "Draft"]

    cascaded = await md.facets("default", filters={"status": "Draft"})
    assert cascaded["business_object"] == ["Policy"]      # narrowed by the filter
    assert cascaded["status"] == ["Active", "Draft"]      # never narrowed by itself


async def test_metadata_match_facet_values_is_case_insensitive(pg3: PgDatabase) -> None:
    md = PgMetadataStore(pg3)
    await _seed_metadata(md)
    assert await md.match_facet_values(["supplier0005"]) == {"object_code": "SUPPLIER0005"}
    assert await md.match_facet_values(["nope"]) == {}


async def test_metadata_list_documents_filter_and_expiry(pg3: PgDatabase) -> None:
    md = PgMetadataStore(pg3)
    await _seed_metadata(md)
    total, rows = await md.list_documents("default", filters={"status": "Draft"})
    assert total == 1 and rows[0]["document_id"] == "d1"

    total, rows = await md.list_documents("default", expiry_from="2026-01-01", expiry_to="2026-12-31")
    assert total == 1 and rows[0]["expiry_date"] == "2026-05-01"

    total, rows = await md.list_documents("default", limit=1, offset=1)
    assert total == 2 and len(rows) == 1


async def test_metadata_search_documents(pg3: PgDatabase) -> None:
    md = PgMetadataStore(pg3)
    await _seed_metadata(md)
    assert (await md.search_documents("AMS encryption policy"))[0]["document_id"] == "d1"
    # Facet values are searchable, not just the title.
    assert (await md.search_documents("SUPPLIER0009"))[0]["document_id"] == "d2"
    # Only question scaffolding -> no terms -> no results.
    assert await md.search_documents("what is the") == []


async def test_metadata_apply_facets(pg3: PgDatabase) -> None:
    md = PgMetadataStore(pg3)
    await _seed_metadata(md)
    assert await md.apply_facets("d1", {"supplier_name": "ACME Corp", "status": None}) is True
    assert await md.apply_facets("missing", {"x": "y"}) is False

    facets = await md.facets("default")
    assert facets["supplier_name"] == ["ACME Corp"]
    assert facets["status"] == ["Active"]  # d1's status was retracted

    # A facet that is also a DocumentMetadata field reaches the canonical payload.
    await md.apply_facets("d1", {"title": "Renamed Policy"})
    got = await md.get("d1")
    assert got is not None and got.title == "Renamed Policy"


async def test_metadata_delete_and_backfill(pg3: PgDatabase) -> None:
    md = PgMetadataStore(pg3)
    await _seed_metadata(md)
    assert await md.backfill_expiry(lambda _t: dt.date(2030, 1, 1)) == 2
    total, _ = await md.list_documents("default", expiry_from="2030-01-01", expiry_to="2030-12-31")
    assert total == 2

    await md.delete_document("d2")
    assert await md.count() == 1
    assert (await md.facets("default")).get("business_object") == ["Policy"]


# --------------------------------------------------------------------------
# Parity: the same scenario through both backends must agree
# --------------------------------------------------------------------------


async def test_end_to_end_index_and_search_on_postgres(sample_objects, build_stack) -> None:
    """The real pipeline — extract, chunk, embed, index, hybrid search — with every
    store on Postgres and nothing else changed."""
    db = _fresh_db(256)
    try:
        vec, kw, md = PgVectorIndex(db), PgKeywordIndex(db), PgMetadataStore(db)
        indexer, search, _, _ = build_stack(sample_objects, stores=(vec, kw, md))
        report = await indexer.index(IndexRequest(reset=True))
        assert report.documents_indexed == 2
        assert await vec.count() > 0
        assert await kw.count() == await vec.count()  # one row, both facades

        resp = await search.search("encryption at rest AES-256", top_k=5)
        assert resp.results
        assert resp.results[0].source_key == "foxtrot/security_policy.docx"
        # Hybrid actually fused: at least one hit found by both retrievers.
        assert any(r.matched_by == "both" for r in resp.results)

        assert await md.count() == 2
        assert (await md.facets("default"))
    finally:
        await db.aclose()


async def test_parity_search_results_match_local_backend(
    sample_objects, build_stack, tmp_path
) -> None:
    """Index the same corpus into both backends and compare what search returns.

    Ranking is not asserted score-for-score — Postgres ranks sparse hits with
    ts_rank_cd and SQLite with BM25 — but the retrieved DOCUMENTS and the top hit
    must agree, which is what the UI and the answer service consume.
    """
    db = _fresh_db(256)
    try:
        pg_stores = (PgVectorIndex(db), PgKeywordIndex(db), PgMetadataStore(db))
        local_stores = (
            LocalVectorIndex(str(tmp_path / "qdrant"), collection="t", dim=256),
            KeywordIndex(str(tmp_path / "kw.db")),
            MetadataStore(str(tmp_path / "meta.db")),
        )
        try:
            pg_index, pg_search, _, _ = build_stack(sample_objects, stores=pg_stores)
            lo_index, lo_search, _, _ = build_stack(sample_objects, stores=local_stores)
            await pg_index.index(IndexRequest(reset=True))
            await lo_index.index(IndexRequest(reset=True))

            assert await pg_stores[0].count() == await local_stores[0].count()
            assert await pg_stores[2].count() == await local_stores[2].count()

            for query in (
                "encryption at rest AES-256",
                "supplier ACME Germany",
                "multi-factor authentication",
            ):
                pg_res = await pg_search.search(query, top_k=5)
                lo_res = await lo_search.search(query, top_k=5)
                assert pg_res.results and lo_res.results, query
                assert pg_res.results[0].source_key == lo_res.results[0].source_key, query
                assert {r.source_key for r in pg_res.results} == {
                    r.source_key for r in lo_res.results
                }, query

            # Facet vocabulary must be identical, since the UI dropdowns read it.
            assert await pg_stores[2].facets("default") == await local_stores[2].facets("default")
        finally:
            for store in local_stores:
                await store.aclose()
    finally:
        await db.aclose()


# --------------------------------------------------------------------------
# NUL bytes: the one place Postgres and SQLite genuinely disagree
# --------------------------------------------------------------------------


async def test_nul_bytes_in_text_do_not_fail_the_document(pg3: PgDatabase) -> None:
    """Extracted PDF/OCR text sometimes carries embedded NULs. SQLite stores them;
    Postgres rejects the statement outright, which failed 17 real documents on the
    first full index run. They must be stripped, not propagated.
    """
    vec, kw = PgVectorIndex(pg3), PgKeywordIndex(pg3)
    await vec.ensure()
    dirty_text = "ready to start using your Welcomecard!\x00 continue"
    dirty_payload = {
        "document_id": "d1",
        "source_key": "welcome\x00card.pdf",
        "section": "intro\x00",
    }

    await kw.add([("c1", dirty_text, dirty_payload)])
    await vec.upsert([("c1", [1.0, 0.0, 0.0], dirty_payload)])

    text, payload = (await kw.get(["c1"]))["c1"]
    assert "\x00" not in text
    assert text == "ready to start using your Welcomecard! continue"
    assert payload["source_key"] == "welcomecard.pdf"
    assert payload["section"] == "intro"
    # Still fully searchable after scrubbing.
    assert [c for c, _ in await kw.search("welcomecard", limit=5)] == ["c1"]


async def test_nul_bytes_survive_payload_merge_and_metadata(pg3: PgDatabase) -> None:
    vec, kw, md = PgVectorIndex(pg3), PgKeywordIndex(pg3), PgMetadataStore(pg3)
    await vec.ensure()
    await kw.add([("c1", "clean text", {"document_id": "d1"})])

    # Enrichment paths must scrub too, not just the initial write.
    await vec.set_document_payload("d1", {"supplier": "ACME\x00 Corp"})
    assert (await kw.get(["c1"]))["c1"][1]["supplier"] == "ACME Corp"
    await kw.update_payload("d1", {"status": "Dra\x00ft"})
    assert (await kw.get(["c1"]))["c1"][1]["status"] == "Draft"

    # The metadata store serialises to JSON first, so a NUL arrives escaped.
    await md.put(
        DocumentMetadata(document_id="d1", title="Welcome\x00card Guide"),
        namespace="default",
        source_key="welcome\x00card.pdf",
        facets={"status": "Act\x00ive"},
    )
    got = await md.get("d1")
    assert got is not None and got.title == "Welcomecard Guide"
    assert (await md.facets("default"))["status"] == ["Active"]

    assert await md.apply_facets("d1", {"supplier_name": "ACME\x00 Ltd"}) is True
    assert (await md.facets("default"))["supplier_name"] == ["ACME Ltd"]


async def test_vector_index_rebuild_keeps_search_working(pg3: PgDatabase) -> None:
    """Runs single-process, so it also works under a container's small /dev/shm."""
    vec = PgVectorIndex(pg3)
    await vec.ensure()
    await vec.upsert([("c1", [1.0, 0.0, 0.0], {"document_id": "d1"}),
                      ("c2", [0.0, 1.0, 0.0], {"document_id": "d2"})])
    await vec.rebuild_index()
    assert (await vec.search([0.9, 0.1, 0.0], limit=1))[0].chunk_id == "c1"
    assert await vec.count() == 2
