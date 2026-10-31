"""Incremental indexing + DeleteMark purge (the scheduled Lambda jobs), fully offline."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable, Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from rag_search.incremental import (
    AlreadyRunning,
    DeleteMarkPurger,
    IncrementalIndexer,
    InMemoryIndexState,
    PendingFile,
)
from rag_search.mongo_source import MongoMetadataSource

from rag_core.aws import S3ObjectSummary

Stack = Callable[..., Any]
BUCKET = "demo-bucket"


class TimedS3:
    """Lists ``objects`` with a LastModified per key (default: just now)."""

    def __init__(self, objects: dict[str, bytes], modified: dict[str, datetime]) -> None:
        self._objects, self._modified = objects, modified

    async def list_objects(self, bucket: str, prefix: str = "") -> AsyncIterator[S3ObjectSummary]:
        for key, body in self._objects.items():
            if key.startswith(prefix):
                yield S3ObjectSummary(
                    bucket=bucket,
                    key=key,
                    size_bytes=len(body),
                    etag="e",
                    last_modified=self._modified.get(key, datetime.now(UTC)),
                )


def _matches(doc: dict[str, Any], query: dict[str, Any]) -> bool:
    for field, cond in query.items():
        if field == "$or":
            if not any(_matches(doc, q) for q in cond):
                return False
        elif isinstance(cond, dict) and "$in" in cond:
            if doc.get(field) not in cond["$in"]:
                return False
        elif doc.get(field) != cond:
            return False
    return True


class Catalogue(MongoMetadataSource):
    """AttachmentMetaData in memory; records the queries it was asked."""

    def __init__(self, records: list[dict[str, Any]], tenant_codes: str = "FOXTROT") -> None:
        super().__init__("mongodb://x", database="d", collection="c", tenant_codes=tenant_codes)
        self.records = records
        self.queries: list[dict[str, Any]] = []

    def _find_sync(self, query: dict[str, Any]) -> Iterable[dict[str, Any]]:
        self.queries.append(query)
        return [d for d in self.records if _matches(d, query)]


def _record(name: str, tenant: str = "FOXTROT", **extra: Any) -> dict[str, Any]:
    return {
        "GeneratedFileName": name,
        "TenantCode": tenant,
        "BusinessObject": "SUPPLIERDOCUMENTS",
        "Status": "Active",
        "DeleteMark": 0,
        **extra,
    }


def _incremental(
    indexer: Any, meta_store: Any, mongo: Catalogue, objects: dict[str, bytes], **kw: Any
) -> tuple[IncrementalIndexer, InMemoryIndexState]:
    state = kw.pop("state", None) or InMemoryIndexState()
    modified = kw.pop("modified", {})
    inc = IncrementalIndexer(
        indexer=indexer,
        metadata_store=meta_store,
        mongo=mongo,
        state=state,
        s3=TimedS3(objects, modified),
        bucket=BUCKET,
        prefix="foxtrot/",
        **kw,
    )
    return inc, state


async def _indexed_keys(meta_store: Any) -> set[str]:
    return {key for _, key in await meta_store.all_documents()}


async def test_new_file_with_metadata_is_indexed_and_one_without_waits(
    build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    mongo = Catalogue([_record("security_policy.docx")])
    indexer, search, meta_store, closeables = build_stack(sample_objects, mongo=mongo)
    try:
        inc, state = _incremental(indexer, meta_store, mongo, sample_objects)
        first = await inc.run()

        assert first.indexed == 1 and first.awaiting_metadata == 1
        assert await _indexed_keys(meta_store) == {"foxtrot/security_policy.docx"}
        assert [p.key for p in await state.pending()] == ["foxtrot/supplier_register.html"]
        # Only the candidates were looked up -- never the whole catalogue.
        assert {} not in mongo.queries

        # The catalogue record arrives after the file: the next run picks it up.
        mongo.records.append(_record("supplier_register.html"))
        second = await inc.run()
        assert second.indexed == 1 and second.already_indexed == 1
        assert await state.pending() == []
        sources = {p.source_key for p in (await search.search("ACME supplier Germany")).results}
        assert "foxtrot/supplier_register.html" in sources
    finally:
        for c in closeables:
            await c.aclose()


async def test_files_older_than_the_watermark_are_not_reconsidered(
    build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    mongo = Catalogue([_record("security_policy.docx"), _record("supplier_register.html")])
    indexer, _, meta_store, closeables = build_stack(sample_objects, mongo=mongo)
    try:
        old = datetime.now(UTC) - timedelta(days=30)
        inc, state = _incremental(
            indexer, meta_store, mongo, sample_objects, modified=dict.fromkeys(sample_objects, old)
        )
        summary = await inc.run()
        assert summary.listed == 2 and summary.candidates == 0 and summary.indexed == 0
        assert await state.watermark() is not None
    finally:
        for c in closeables:
            await c.aclose()


async def test_other_tenants_and_deleted_files_are_never_indexed(
    build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    mongo = Catalogue(
        [
            _record("security_policy.docx", DeleteMark=1),
            _record("supplier_register.html", tenant="ACMEFOODS"),
        ]
    )
    indexer, _, meta_store, closeables = build_stack(sample_objects, mongo=mongo)
    try:
        inc, state = _incremental(indexer, meta_store, mongo, sample_objects)
        summary = await inc.run()
        assert (summary.deleted, summary.other_tenant, summary.indexed) == (1, 1, 0)
        assert await _indexed_keys(meta_store) == set()
        assert await state.pending() == []
    finally:
        for c in closeables:
            await c.aclose()


async def test_a_file_that_never_gets_metadata_expires(
    build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    mongo = Catalogue([])
    indexer, _, meta_store, closeables = build_stack(sample_objects, mongo=mongo)
    try:
        state = InMemoryIndexState()
        stale = datetime.now(UTC) - timedelta(days=8)
        await state.put_pending([PendingFile(k, stale) for k in sample_objects])
        inc, _ = _incremental(
            indexer, meta_store, mongo, sample_objects, state=state,
            pending_max_age=timedelta(days=7),
        )
        summary = await inc.run()
        assert summary.pending_expired == 2
        assert await state.pending() == []
    finally:
        for c in closeables:
            await c.aclose()


async def test_out_of_time_files_are_deferred_to_the_next_run(
    build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    mongo = Catalogue([_record("security_policy.docx"), _record("supplier_register.html")])
    indexer, _, meta_store, closeables = build_stack(sample_objects, mongo=mongo)
    try:
        inc, state = _incremental(indexer, meta_store, mongo, sample_objects)
        summary = await inc.run(deadline=time.monotonic() - 1)
        assert summary.deferred == 2 and summary.indexed == 0
        assert len(await state.pending()) == 2

        # Deferred files are resumed even though S3 has not changed since.
        inc2, _ = _incremental(
            indexer, meta_store, mongo, sample_objects, state=state,
            modified=dict.fromkeys(sample_objects, datetime.now(UTC) - timedelta(days=30)),
        )
        resumed = await inc2.run()
        assert resumed.indexed == 2
        assert await state.pending() == []
    finally:
        for c in closeables:
            await c.aclose()


async def test_a_second_concurrent_run_is_refused(
    build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    mongo = Catalogue([])
    indexer, _, meta_store, closeables = build_stack(sample_objects, mongo=mongo)
    try:
        state = InMemoryIndexState()
        assert await state.acquire("someone-else", timedelta(minutes=5))
        inc, _ = _incremental(indexer, meta_store, mongo, sample_objects, state=state)
        with pytest.raises(AlreadyRunning):
            await inc.run()
    finally:
        for c in closeables:
            await c.aclose()


@pytest.mark.parametrize("dry_run", [True, False])
async def test_purge_removes_only_this_tenants_delete_marked_documents(
    dry_run: bool, build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    mongo = Catalogue([_record("security_policy.docx"), _record("supplier_register.html")])
    indexer, search, meta_store, closeables = build_stack(sample_objects, mongo=mongo)
    try:
        inc, _ = _incremental(indexer, meta_store, mongo, sample_objects)
        assert (await inc.run()).indexed == 2

        # Upstream: FOXTROT soft-deletes one file; another tenant's DeleteMark on the
        # other file's name is not ours to act on.
        mongo.records[0]["DeleteMark"] = 1
        mongo.records.append(_record("supplier_register.html", tenant="ACMEFOODS", DeleteMark=1))

        purger = DeleteMarkPurger(indexer=indexer, metadata_store=meta_store, mongo=mongo)
        summary = await purger.run(dry_run=dry_run)

        assert summary.purged == 1 and summary.sample == ["security_policy.docx"]
        remaining = await _indexed_keys(meta_store)
        if dry_run:
            assert remaining == {"foxtrot/security_policy.docx", "foxtrot/supplier_register.html"}
        else:
            assert remaining == {"foxtrot/supplier_register.html"}
            hits = {p.source_key for p in (await search.search("encrypted AES-256")).results}
            assert "foxtrot/security_policy.docx" not in hits
    finally:
        for c in closeables:
            await c.aclose()


# --------------------------------------------------------------------------
# Lambda target configuration
# --------------------------------------------------------------------------


async def test_targets_resolve_from_config_and_refuse_without_a_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rag_search.config import SearchSettings
    from rag_search.index_jobs import load_targets, resolve_target

    monkeypatch.setenv(
        "INDEX_TARGETS",
        '{"foxtrot": {"collection": "FOXTROT_DocumentSearch_DEV", "bucket": "b", "prefix": "foxtrot/",'
        ' "tenant_codes": ["FOXTROT"], "metadata_database": "FOXTROT_DEV"}}',
    )
    targets = load_targets()
    assert list(targets) == ["FOXTROT"]  # names are case-insensitive

    search = SearchSettings(mongo_collection="AttachmentMetaData")
    target = await resolve_target("FOXTROT", targets["FOXTROT"], search)
    assert (target.bucket, target.prefix, target.tenant_codes) == ("b", "foxtrot/", "FOXTROT")
    assert (target.metadata_database, target.metadata_collection) == ("FOXTROT_DEV", "AttachmentMetaData")

    # Tenants share S3 folders: an unfiltered target would index everyone's files.
    unfiltered = {"collection": "X", "bucket": "b", "prefix": "", "tenant_codes": ""}
    monkeypatch.setattr("rag_search.index_jobs.build_mongo_index", lambda *a, **k: _NoManifest())
    with pytest.raises(ValueError, match="tenant_codes"):
        await resolve_target("X", unfiltered, search)


class _NoManifest:
    async def read_manifest(self) -> dict[str, Any]:
        return {}

    async def aclose(self) -> None:
        return None
