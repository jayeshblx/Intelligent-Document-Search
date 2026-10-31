"""An index is built for one TenantCode, even though tenants share storage.

Measured in dev: ``FOXTROT_DEV.AttachmentMetaData`` holds 30,270 ACMEFOODS records and
5,155 FOXTROT ones, and ``example-docs-bucket/foxtrot/`` holds 5,636 ACMEFOODS files beside 1,647
FOXTROT files. Neither the folder nor the database separates tenants, and an index built
before this filter existed was 498 ACMEFOODS documents to 186 FOXTROT.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from rag_search.models import IndexRequest
from rag_search.mongo_source import MongoMetadataSource, parse_tenant_codes

Stack = Callable[..., Any]


def _record(tenant: str, **extra: Any) -> dict[str, Any]:
    return {
        "TenantCode": tenant,
        "BusinessObject": "SUPPLIERDOCUMENTS",
        "Status": "Active",
        "DeleteMark": 0,
        **extra,
    }


def _source(tenant_codes: str | None, mapping: dict[str, dict[str, Any]]) -> MongoMetadataSource:
    mongo = MongoMetadataSource("mongodb://x", database="d", collection="c", tenant_codes=tenant_codes)
    mongo.seed(mapping)
    return mongo


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


def test_other_tenants_records_never_enrich_and_are_reported_foreign() -> None:
    mongo = _source(
        "FOXTROT",
        {"foxtrot_policy.pdf": _record("FOXTROT"), "af_kids_meal_box.pdf": _record("ACMEFOODS")},
    )
    assert mongo.lookup("foxtrot/foxtrot_policy.pdf") is not None
    assert mongo.is_foreign("foxtrot/foxtrot_policy.pdf") is False

    assert mongo.lookup("foxtrot/af_kids_meal_box.pdf") is None
    assert mongo.is_foreign("foxtrot/af_kids_meal_box.pdf") is True


def test_tenant_codes_are_case_insensitive() -> None:
    """AF_DEV holds both "AF" and "af"."""
    mongo = _source(" af , ACMEFOODS", {"a.pdf": _record("af"), "b.pdf": _record("AF")})
    assert mongo.lookup("foxtrot/a.pdf") and mongo.lookup("foxtrot/b.pdf")
    assert parse_tenant_codes("foxtrot, ,AF") == frozenset({"FOXTROT", "AF"})


def test_a_file_this_tenant_catalogues_is_not_foreign() -> None:
    mongo = MongoMetadataSource("mongodb://x", database="d", collection="c", tenant_codes="FOXTROT")
    mongo._classify(
        [
            (_record("ACMEFOODS"), ["shared.pdf"]),
            (_record("FOXTROT"), ["shared.pdf"]),
        ]
    )
    assert mongo.lookup("foxtrot/shared.pdf") is not None
    assert mongo.is_foreign("foxtrot/shared.pdf") is False


def test_another_tenants_deletion_does_not_delete_ours() -> None:
    mongo = MongoMetadataSource("mongodb://x", database="d", collection="c", tenant_codes="FOXTROT")
    mongo._classify(
        [
            (_record("FOXTROT"), ["shared.pdf"]),
            (_record("ACMEFOODS", DeleteMark=1), ["shared.pdf"]),
        ]
    )
    assert mongo.is_deleted("foxtrot/shared.pdf") is False
    assert mongo.lookup("foxtrot/shared.pdf") is not None


def test_without_a_filter_nothing_is_foreign() -> None:
    """Existing single-tenant setups keep working unchanged."""
    mongo = _source(None, {"x.pdf": _record("ACMEFOODS")})
    assert mongo.filters_tenants is False
    assert mongo.is_foreign("foxtrot/x.pdf") is False
    assert mongo.lookup("foxtrot/x.pdf") is not None


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mongo_only", [True, False])
async def test_indexing_skips_other_tenants_files(
    mongo_only: bool, build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    """Skipped with or without mongo_only: without metadata a foreign file would still
    be indexed and searchable."""
    mongo = _source(
        "FOXTROT",
        {
            "security_policy.docx": _record("FOXTROT"),
            "supplier_register.html": _record("ACMEFOODS"),
        },
    )
    indexer, search, meta_store, closeables = build_stack(sample_objects, mongo=mongo)
    try:
        summary = await indexer.index(IndexRequest(reset=True, mongo_only=mongo_only))

        assert summary.documents_indexed == 1
        assert summary.documents_other_tenant == 1
        sources = {p.source_key for p in (await search.search("ACME supplier Germany")).results}
        assert "foxtrot/supplier_register.html" not in sources
        assert (await meta_store.facets("default")).get("tenant") == ["FOXTROT"]
    finally:
        for c in closeables:
            await c.aclose()


async def test_append_run_purges_other_tenants_leftovers(
    build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    """An index built before the filter existed is cleaned by the next filtered run."""
    unfiltered = _source(
        None,
        {
            "security_policy.docx": _record("FOXTROT"),
            "supplier_register.html": _record("ACMEFOODS"),
        },
    )
    indexer, search, meta_store, closeables = build_stack(sample_objects, mongo=unfiltered)
    try:
        await indexer.index(IndexRequest(reset=True))
        assert sorted((await meta_store.facets("default"))["tenant"]) == ["ACMEFOODS", "FOXTROT"]

        indexer._mongo = _source(
            "FOXTROT",
            {
                "security_policy.docx": _record("FOXTROT"),
                "supplier_register.html": _record("ACMEFOODS"),
            },
        )
        await indexer.index(IndexRequest(reset=False))

        assert (await meta_store.facets("default"))["tenant"] == ["FOXTROT"]
        sources = {p.source_key for p in (await search.search("ACME supplier Germany")).results}
        assert "foxtrot/supplier_register.html" not in sources
    finally:
        for c in closeables:
            await c.aclose()


async def test_mongo_outage_fails_a_tenant_filtered_run(
    build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    """Without the catalogue tenants cannot be told apart: index nothing, loudly."""

    class Down(MongoMetadataSource):
        async def preload(self) -> int:
            raise ConnectionError("mongo unreachable")

    mongo = Down("mongodb://x", database="d", collection="c", tenant_codes="FOXTROT")
    indexer, _search, meta_store, closeables = build_stack(sample_objects, mongo=mongo)
    try:
        with pytest.raises(ConnectionError):
            await indexer.index(IndexRequest(reset=True))
        assert await meta_store.count() == 0
    finally:
        for c in closeables:
            await c.aclose()


async def test_mongo_outage_without_a_filter_still_indexes(
    build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    """Unchanged behaviour when no tenant filter is configured."""

    class Down(MongoMetadataSource):
        async def preload(self) -> int:
            raise ConnectionError("mongo unreachable")

    indexer, _search, _meta, closeables = build_stack(
        sample_objects, mongo=Down("mongodb://x", database="d", collection="c")
    )
    try:
        summary = await indexer.index(IndexRequest(reset=True))
        assert summary.documents_indexed == 2
    finally:
        for c in closeables:
            await c.aclose()
