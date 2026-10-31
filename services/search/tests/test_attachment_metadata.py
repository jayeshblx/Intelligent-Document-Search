"""A catalogue record's MetaData values are searchable, not only the file's text.

``AttachmentMetaData.MetaData`` holds values that exist only in the catalogue -- a
complaint's description, its complaint number -- so a question about them could
never match the file. Shapes here mirror real FOXTROT_DEV COMPLAINTSMETA records.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rag_search.indexer import METADATA_CHUNK_SECTION, METADATA_CHUNK_STRATEGY
from rag_search.models import IndexRequest
from rag_search.mongo_source import MongoMetadataSource, attachment_metadata_text

from rag_core.utils import chunk_id, document_id

Stack = Callable[..., Any]


def _complaint(**overrides: Any) -> dict[str, Any]:
    return {
        "TenantCode": "FOXTROT",
        "BusinessObject": "COMPLAINTS",
        "ObjectCode": "STD1647",
        "Status": "Active",
        "MetaDataName": "COMPLAINTSMETA",
        "OriginalFileName": "security_policy.docx",
        "DeleteMark": 0,
        "MetaData": [
            {"FieldName": "ComplaintDescription", "DisplayName": "ComplaintDescription",
             "Value": "Customer reported a rancid oil smell in the fryer batch"},
            {"FieldName": "ObjectCode", "DisplayName": "Complaint Number", "Value": "STD1647"},
            {"FieldName": "ModifiedDate", "DisplayName": "ModifiedDate", "Value": None},
            {"FieldName": "Resolution", "DisplayName": "Resolution", "Value": "   "},
        ],
        **overrides,
    }


def _mongo(mapping: dict[str, dict[str, Any]]) -> MongoMetadataSource:
    mongo = MongoMetadataSource("mongodb://x", database="FOXTROT_DEV", collection="c", tenant_codes="FOXTROT")
    mongo.seed(mapping)
    return mongo


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------


def test_metadata_text_keeps_values_and_both_names() -> None:
    text = attachment_metadata_text(_complaint())

    assert text.startswith("Attachment metadata for security_policy.docx")
    assert "Business object: COMPLAINTS" in text
    assert "ComplaintDescription: Customer reported a rancid oil smell" in text
    # Display and field name differ: both kept, so either word matches.
    assert "Complaint Number (ObjectCode): STD1647" in text
    # Empty values carry nothing searchable.
    assert "ModifiedDate" not in text and "Resolution" not in text


def test_no_metadata_values_means_no_text() -> None:
    assert attachment_metadata_text(_complaint(MetaData=[])) == ""
    assert attachment_metadata_text(_complaint(MetaData=None)) == ""
    only_empty = [{"FieldName": "X", "DisplayName": "X", "Value": None}]
    assert attachment_metadata_text(_complaint(MetaData=only_empty)) == ""


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------


async def test_metadata_values_are_searchable_after_indexing(
    build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    mongo = _mongo({"security_policy.docx": _complaint()})
    indexer, search, _meta, closeables = build_stack(sample_objects, mongo=mongo)
    try:
        await indexer.index(IndexRequest(reset=True, mongo_only=True))

        # A value that is NOT in the file's text: only the metadata chunk can match it.
        hits = (await search.search("rancid oil smell fryer")).results
        assert hits and hits[0].source_key == "foxtrot/security_policy.docx"
        assert hits[0].section == METADATA_CHUNK_SECTION

        # The complaint number is found by the literal identifier lookup too.
        exact = await search.exact_lookup(["STD1647"])
        assert "foxtrot/security_policy.docx" in {p.source_key for p in exact}
    finally:
        for c in closeables:
            await c.aclose()


async def test_documents_without_metadata_values_get_no_extra_chunk(
    build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    mongo = _mongo({"security_policy.docx": _complaint(MetaData=[])})
    indexer, _search, _meta, closeables = build_stack(sample_objects, mongo=mongo)
    try:
        await indexer.index(IndexRequest(reset=True, mongo_only=True))
        doc_id = document_id("demo-bucket", "foxtrot/security_policy.docx")
        assert await indexer._keyword.get([chunk_id(doc_id, METADATA_CHUNK_STRATEGY, 0)]) == {}
    finally:
        for c in closeables:
            await c.aclose()


async def test_backfill_adds_metadata_to_an_existing_index_once(
    build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    """An index built before this existed gains the chunk without re-indexing files."""
    old = _mongo({"security_policy.docx": _complaint(MetaData=[])})
    indexer, search, _meta, closeables = build_stack(sample_objects, mongo=old)
    try:
        await indexer.index(IndexRequest(reset=True, mongo_only=True))
        before = await indexer._keyword.count()
        assert not [
            p for p in (await search.search("rancid oil smell fryer")).results
            if p.section == METADATA_CHUNK_SECTION
        ]

        indexer._mongo = _mongo({"security_policy.docx": _complaint()})
        embedded: list[int] = []
        real_embed = indexer._embedder.embed

        async def counting_embed(texts: list[str]) -> list[list[float]]:
            embedded.append(len(texts))
            return await real_embed(texts)

        indexer._embedder.embed = counting_embed  # type: ignore[method-assign]

        assert await indexer.backfill_attachment_metadata() == 1
        assert embedded == [1]  # only the metadata chunk was embedded
        assert await indexer._keyword.count() == before + 1
        hits = (await search.search("rancid oil smell fryer")).results
        assert hits[0].section == METADATA_CHUNK_SECTION
        # The chunk carries the document's facets, so filtered search still finds it.
        filtered = await search.search("rancid oil", filters={"business_object": "COMPLAINTS"})
        assert filtered.results

        # Safe to re-run: nothing is added twice.
        assert await indexer.backfill_attachment_metadata() == 0
        assert await indexer._keyword.count() == before + 1
    finally:
        for c in closeables:
            await c.aclose()


async def test_backfill_ignores_other_tenants_records(
    build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    """A document with no metadata chunk yet, whose only MetaData-bearing record now
    belongs to another tenant: the backfill must not attach that tenant's values."""
    mongo = _mongo({"security_policy.docx": _complaint(MetaData=[])})
    indexer, _search, _meta, closeables = build_stack(sample_objects, mongo=mongo)
    try:
        await indexer.index(IndexRequest(reset=True, mongo_only=True))
        indexer._mongo = _mongo({"security_policy.docx": _complaint(TenantCode="ACMEFOODS")})
        assert await indexer.backfill_attachment_metadata() == 0

        # Control: the same record under this tenant WOULD be added.
        indexer._mongo = _mongo({"security_policy.docx": _complaint()})
        assert await indexer.backfill_attachment_metadata() == 1
    finally:
        for c in closeables:
            await c.aclose()


async def test_vector_index_is_rebuilt_after_rows_are_added(
    build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    """Late inserts into a large HNSW graph lost the best match in the FOXTROT dev index
    until a rebuild, so both write paths rebuild where the store supports it."""
    mongo = _mongo({"security_policy.docx": _complaint(MetaData=[])})
    indexer, _search, _meta, closeables = build_stack(sample_objects, mongo=mongo)
    rebuilds: list[str] = []

    async def rebuild_index() -> None:
        rebuilds.append("rebuild")

    indexer._vector.rebuild_index = rebuild_index  # type: ignore[attr-defined]
    try:
        await indexer.index(IndexRequest(reset=True, mongo_only=True))
        assert rebuilds == ["rebuild"]

        indexer._mongo = _mongo({"security_policy.docx": _complaint()})
        await indexer.backfill_attachment_metadata()
        assert rebuilds == ["rebuild", "rebuild"]

        # Nothing added -> nothing to rebuild.
        await indexer.backfill_attachment_metadata()
        assert rebuilds == ["rebuild", "rebuild"]
    finally:
        for c in closeables:
            await c.aclose()
