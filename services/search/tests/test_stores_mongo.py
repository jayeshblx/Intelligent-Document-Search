"""Tests for the MongoDB Atlas backend (vectors + keyword + metadata in one collection).

Two halves:

* Pure helpers (filter building, facet terms, index definitions) -- always run.
* Live store tests against a real Atlas cluster -- Atlas Vector Search and Atlas
  Search cannot be emulated locally. They run only when ``SEARCH_MONGO_TEST_URI``
  is set, in a throwaway database (``SEARCH_MONGO_TEST_DB``, default
  ``DocumentSearch_TEST``) that is dropped afterwards. Never point them at the
  real ``DocumentSearch`` database.

Atlas indexes asynchronously, so live assertions on search results wait for the
search indexes to catch up with the writes first.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import uuid

import pytest
from rag_search.stores.mongo import (
    TEXT_ANALYZER,
    MongoIndexDatabase,
    MongoKeywordIndex,
    MongoVectorIndex,
    facet_terms,
    text_filter,
    text_index_definition,
    vector_filter,
    vector_index_definition,
)
from rag_search.stores.mongo_metadata import MongoMetadataStore, document_record

from rag_core.models.metadata import DocumentMetadata

# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


def test_facet_terms_cover_scalars_and_list_elements() -> None:
    terms = facet_terms(
        {
            "status": "Draft",
            "page_start": 3,
            "flag": True,
            "tags": ["a", "b"],
            "text": "chunk body is never a facet",
            "missing": None,
            "nested": {"x": 1},
            "blank": "",
        }
    )
    assert terms == ["flag=true", "page_start=3", "status=Draft", "tags=a", "tags=b"]


def test_facet_terms_skip_prose_length_values() -> None:
    assert facet_terms({"title": "x" * 600}) == []


def test_vector_filter_any_of_within_and_across_facets() -> None:
    assert vector_filter(None) is None
    assert vector_filter({"status": "  "}) is None  # empty = no constraint
    assert vector_filter({"status": "Draft"}) == {"facet_kv": {"$in": ["status=Draft"]}}
    both = vector_filter({"business_object": ["INGREDIENTSPEC", "PIF"], "tenant": "FOXTROT"})
    assert both == {
        "$and": [
            {"facet_kv": {"$in": ["business_object=INGREDIENTSPEC", "business_object=PIF"]}},
            {"facet_kv": {"$in": ["tenant=FOXTROT"]}},
        ]
    }


def test_text_filter_always_restricts_to_chunks() -> None:
    assert text_filter(None) == [{"equals": {"path": "kind", "value": "chunk"}}]
    assert text_filter({"status": ["Draft", "Active"]})[1] == {
        "in": {"path": "facet_kv", "value": ["status=Draft", "status=Active"]}
    }


def test_index_definitions_declare_the_filter_field() -> None:
    vec = vector_index_definition(1024)
    assert {
        "type": "vector",
        "path": "embedding",
        "numDimensions": 1024,
        "similarity": "cosine",
        "quantization": "scalar",
    } in vec["fields"]
    assert {"type": "filter", "path": "facet_kv"} in vec["fields"]
    fields = text_index_definition()["mappings"]["fields"]
    assert fields["text"]["analyzer"] == TEXT_ANALYZER
    analyzer = text_index_definition()["analyzers"][0]
    assert analyzer["name"] == TEXT_ANALYZER
    assert [f["type"] for f in analyzer["tokenFilters"]][-1] == "porterStemming"
    assert fields["facet_kv"]["type"] == fields["kind"]["type"] == "token"


def test_document_record_shape() -> None:
    rec = document_record(
        document_id="d1",
        namespace="default",
        source_key="foxtrot/Policy-1032658.pdf",
        payload={"title": "AMS Encryption Policy", "effective": dt.date(2026, 1, 1)},
        facets={"status": "Draft", "tags": ["A", "B"]},
    )
    assert rec["_id"] == "meta::d1" and rec["kind"] == "document"
    assert rec["payload"]["effective"] == "2026-01-01"  # BSON-safe
    assert {"k": "tags", "v": "B", "lv": "b"} in rec["facet_pairs"]
    assert "ams encryption policy" in rec["hay"] and "1032658" in rec["hay"]
    # Verbatim pairs (the migration path) win over derived ones.
    copied = document_record(
        document_id="d1",
        namespace="default",
        source_key="",
        payload={},
        facets={"a": "x"},
        pairs=[("a", "Y")],
    )
    assert copied["facet_pairs"] == [{"k": "a", "v": "Y", "lv": "y"}]


def test_database_requires_a_target() -> None:
    with pytest.raises(ValueError, match="connection string"):
        MongoIndexDatabase(database="d", collection="c", dim=3)
    with pytest.raises(ValueError, match="collection"):
        MongoIndexDatabase("mongodb://x", database="d", collection="", dim=3)


# --------------------------------------------------------------------------
# Live Atlas tests
# --------------------------------------------------------------------------

TEST_URI = os.environ.get("SEARCH_MONGO_TEST_URI", "")
TEST_DB = os.environ.get("SEARCH_MONGO_TEST_DB", "DocumentSearch_TEST")

live = pytest.mark.skipif(not TEST_URI, reason="set SEARCH_MONGO_TEST_URI to run Atlas tests")


@pytest.fixture(scope="module")
def atlas_collection():
    """One collection (and its two search indexes) for the whole module; each test
    starts from an empty collection, the indexes stay. Dropped at the end."""
    from pymongo import MongoClient

    name = f"t_{uuid.uuid4().hex[:10]}"
    yield name
    client = MongoClient(TEST_URI)
    try:
        # MongoDB removes a database once its last collection is gone, so dropping
        # the collection is enough (and needs no dropDatabase privilege).
        client[TEST_DB][name].drop()
    finally:
        client.close()


@pytest.fixture
async def atlas(atlas_collection):
    db = MongoIndexDatabase(TEST_URI, database=TEST_DB, collection=atlas_collection, dim=3)
    await db.ensure_schema()
    await db.run(lambda coll: coll.delete_many({}))
    try:
        yield db
    finally:
        await db.aclose()


async def _searchable(db: MongoIndexDatabase) -> None:
    await db.wait_until_searchable(timeout_s=600, poll_s=2)


@live
async def test_live_keyword_add_search_get_and_stemming(atlas: MongoIndexDatabase) -> None:
    kw = MongoKeywordIndex(atlas)
    await kw.add(
        [
            (
                "c1",
                "data encryption at rest using AES-256",
                {"document_id": "d1", "source_key": "a"},
            ),
            ("c2", "supplier ACME located in Germany", {"document_id": "d2", "source_key": "b"}),
            ("c3", "keys are rotated annually", {"document_id": "d3", "source_key": "c"}),
        ]
    )
    await _searchable(atlas)
    assert await kw.count() == 3
    hits = await kw.search("encryption AES", limit=5)
    assert hits and hits[0][0] == "c1"
    assert [c for c, _ in await kw.search("rotate", limit=5)] == ["c3"]  # stemmed
    assert await kw.search("zzzznomatch", limit=5) == []
    text, payload = (await kw.get(["c1"]))["c1"]
    assert "encryption" in text and payload["source_key"] == "a"


@live
async def test_live_vector_roundtrip_and_shared_record(atlas: MongoIndexDatabase) -> None:
    vec, kw = MongoVectorIndex(atlas), MongoKeywordIndex(atlas)
    payload = {"document_id": "d1", "source_key": "a.pdf"}
    await vec.upsert(
        [("c1", [1.0, 0.0, 0.0], payload), ("c2", [0.0, 1.0, 0.0], {"document_id": "d2"})]
    )
    await kw.add([("c1", "encryption policy", payload)])
    await _searchable(atlas)

    assert await vec.count() == 2  # embedding survived the keyword write
    matches = await vec.search([0.9, 0.1, 0.0], limit=2)
    assert matches[0].chunk_id == "c1"
    assert 0.9 < matches[0].score <= 1.0  # plain cosine, like Qdrant
    assert (await kw.get(["c1"]))["c1"][0] == "encryption policy"


@live
async def test_live_facet_filters_and_payload_merge(atlas: MongoIndexDatabase) -> None:
    vec, kw = MongoVectorIndex(atlas), MongoKeywordIndex(atlas)
    p1 = {"document_id": "d1", "source_key": "policy.pdf", "page_start": 2, "status": "Draft"}
    p2 = {"document_id": "d2", "source_key": "other.pdf", "status": "Active"}
    await vec.upsert([("c1", [1.0, 0.0, 0.0], p1), ("c2", [0.0, 1.0, 0.0], p2)])
    await kw.add([("c1", "encryption policy", p1), ("c2", "encryption elsewhere", p2)])
    await _searchable(atlas)

    assert [
        c for c, _ in await kw.search("encryption", limit=5, filters={"document_id": "d1"})
    ] == ["c1"]
    assert [
        m.chunk_id
        for m in await vec.search([1.0, 0.0, 0.0], limit=5, filters={"document_id": "d2"})
    ] == ["c2"]
    any_of = await kw.search("encryption", limit=5, filters={"status": ["Draft", "Active"]})
    assert {c for c, _ in any_of} == {"c1", "c2"}

    # Enrichment reaches every chunk of the document; None removes the key.
    await vec.set_document_payload("d1", {"supplier": "ACME", "page_start": None})
    merged = (await kw.get(["c1"]))["c1"][1]
    assert merged["supplier"] == "ACME" and "page_start" not in merged
    # An in-place update reaches the search index a moment later (Atlas is
    # eventually consistent), so poll rather than assert once.
    found: list[str] = []
    for _ in range(60):
        found = [c for c, _ in await kw.search("encryption", limit=5, filters={"supplier": "ACME"})]
        if found:
            break
        await asyncio.sleep(1)
    assert found == ["c1"]


@live
async def test_live_exact_lookup_fetch_and_delete(atlas: MongoIndexDatabase) -> None:
    vec, kw = MongoVectorIndex(atlas), MongoKeywordIndex(atlas)
    await kw.add(
        [
            (
                "c1",
                "see delivery note 1032658 for details",
                {"document_id": "d1", "source_key": "a.pdf"},
            ),
            (
                "c2",
                "nothing distinctive here",
                {"document_id": "d2", "source_key": "INV-998877.pdf"},
            ),
        ]
    )
    await vec.upsert(
        [
            ("c4", [1.0, 0.0, 0.0], {"document_id": "d9", "page_start": 5}),
            ("c3", [1.0, 0.0, 0.0], {"document_id": "d9", "page_start": 1}),
        ]
    )
    assert {c for c, _, _ in await kw.exact_lookup(["1032658"], limit=5)} == {"c1"}
    assert {c for c, _, _ in await kw.exact_lookup(["inv-998877"], limit=5)} == {"c2"}
    assert await kw.exact_lookup(["12"], limit=5) == []  # under the 4-char floor
    assert [c["_chunk_id"] for c in await vec.fetch_document("d9")] == ["c3", "c4"]
    assert await kw.delete_document("d1") == 1
    assert await kw.count() == 3


async def _seed_metadata(md: MongoMetadataStore) -> None:
    await md.put(
        DocumentMetadata(
            document_id="d1", title="AMS Encryption Policy", expiry_date=dt.date(2026, 5, 1)
        ),
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


@live
async def test_live_metadata_reads(atlas: MongoIndexDatabase) -> None:
    md = MongoMetadataStore(atlas)
    await _seed_metadata(md)
    assert await md.count() == 2
    got = await md.get("d1")
    assert got is not None and got.title == "AMS Encryption Policy"
    assert await md.get("missing") is None
    assert sorted(await md.all_documents()) == [
        ("d1", "policy-1032658.pdf"),
        ("d2", "supplier-acme.pdf"),
    ]
    assert await md.indexed_ids() == {"d1", "d2"}

    assert (await md.facets("default"))["status"] == ["Active", "Draft"]
    cascaded = await md.facets("default", filters={"status": "Draft"})
    assert cascaded["business_object"] == ["Policy"]  # narrowed by the filter
    assert cascaded["status"] == ["Active", "Draft"]  # never narrowed by itself
    assert await md.facet_values("object_code", filters={"status": "Active"}) == ["SUPPLIER0009"]

    assert await md.match_facet_values(["supplier0005"]) == {"object_code": "SUPPLIER0005"}
    assert await md.match_facet_values(["nope"]) == {}

    total, rows = await md.list_documents("default", filters={"status": "Draft"})
    assert total == 1 and rows[0]["document_id"] == "d1"
    total, rows = await md.list_documents(
        "default", expiry_from="2026-01-01", expiry_to="2026-12-31"
    )
    assert total == 1 and rows[0]["expiry_date"] == "2026-05-01"
    total, rows = await md.list_documents("default", limit=1, offset=1)
    assert total == 2 and len(rows) == 1

    assert (await md.search_documents("AMS encryption policy"))[0]["document_id"] == "d1"
    assert (await md.search_documents("SUPPLIER0009"))[0]["document_id"] == "d2"
    assert await md.search_documents("what is the") == []
    assert (await md.describe(["d2"]))["d2"]["title"] == "ACME Supplier Record"


@live
async def test_live_metadata_writes(atlas: MongoIndexDatabase) -> None:
    md = MongoMetadataStore(atlas)
    await _seed_metadata(md)
    assert await md.apply_facets("d1", {"supplier_name": "ACME Corp", "status": None}) is True
    assert await md.apply_facets("missing", {"x": "y"}) is False
    facets = await md.facets("default")
    assert facets["supplier_name"] == ["ACME Corp"]
    assert facets["status"] == ["Active"]  # d1's status was retracted

    await md.apply_facets("d1", {"title": "Renamed Policy"})
    got = await md.get("d1")
    assert got is not None and got.title == "Renamed Policy"

    assert await md.backfill_expiry(lambda _t: dt.date(2030, 1, 1)) == 2
    total, _ = await md.list_documents("default", expiry_from="2030-01-01", expiry_to="2030-12-31")
    assert total == 2
    await md.delete_document("d2")
    assert await md.count() == 1
    assert (await md.facets("default")).get("business_object") == ["Policy"]


@live
async def test_live_manifest_roundtrip(atlas: MongoIndexDatabase) -> None:
    assert await atlas.read_manifest() == {}
    await atlas.write_manifest({"tenant_codes": ["FOXTROT"], "embed_dim": 3})
    assert await atlas.read_manifest() == {"tenant_codes": ["FOXTROT"], "embed_dim": 3}
    # The manifest is not a chunk or a document.
    assert (await atlas.counts()) == {"chunks": 0, "embedded_chunks": 0, "document_metadata": 0}


def test_bm25_clauses_apply_fts5_idf_and_count_repeats() -> None:
    import math

    from rag_search.stores.mongo import bm25_clauses

    total = 1000
    clauses = bm25_clauses(["the", "frozen", "frozen", "zzz"], total, {"the": 900, "frozen": 10, "zzz": 0})
    boosts = {c["text"]["query"]: c["text"]["score"]["boost"]["value"] for c in clauses}
    assert "zzz" not in boosts  # occurs nowhere: no clause
    # A term in most chunks is worth ~nothing in FTS5 (IDF floored at 1e-6).
    assert boosts["the"] < 1e-5
    ratio = (total - 10 + 0.5) / 10.5
    assert math.isclose(boosts["frozen"], 2 * math.log(ratio) / math.log(1 + ratio))


def test_rescore_orders_by_exact_cosine_and_keeps_tie_order() -> None:
    from rag_search.stores.mongo import _rescore, embedding_value

    rows = [
        {"_id": "far", "embedding": embedding_value([0.0, 1.0, 0.0])},
        {"_id": "tie_a", "embedding": embedding_value([1.0, 1.0, 0.0])},
        {"_id": "best", "embedding": embedding_value([1.0, 0.0, 0.0])},
        {"_id": "tie_b", "embedding": embedding_value([2.0, 2.0, 0.0])},  # same direction
    ]
    scored = _rescore([1.0, 0.0, 0.0], rows)
    assert [r["_id"] for _, r in scored] == ["best", "tie_a", "tie_b", "far"]
    assert abs(scored[0][0] - 1.0) < 1e-6 and abs(scored[-1][0]) < 1e-6
