"""The embedded PostgreSQL server for real: initdb, pg_ctl, pg_dump, pg_restore.

Needs the PostgreSQL 16 binaries and a non-root user -- i.e. the agent image -- and
skips anywhere else. Deliberately NOT gated on the docker-compose database the other
Postgres tests use: inside the image there is no such database, and a gate on it
would silently skip the one test that proves the S3-only path works.

Run inside the image:

    docker run --rm -v "$PWD/services/agentic/tests:/app/services/agentic/tests" \
        exampleapp-agent:local-amd64 python -m pytest services/agentic/tests/test_embedded_pg.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from rag_agentic.config import AgenticSettings, PostgresSource
from rag_agentic.embedded_pg import (
    SUPERUSER,
    EmbeddedPostgres,
    EmbeddedPostgresError,
    find_bin_dir,
)
from rag_agentic.index_store import IndexUnavailableError, TenantIndexStore
from rag_agentic.tenancy import TenantContext
from rag_search.stores import PgDatabase, PgKeywordIndex, PgMetadataStore, PgVectorIndex

from rag_core.models.metadata import DocumentMetadata

BUCKET = "test-index-bucket"
PREFIX = "DocumentSearch/foxtrot-Dev"
DUMP_KEY = "rag-20260917T000000Z.dump"
DIM = 8


def _pg_available() -> bool:
    try:
        find_bin_dir(os.environ.get("AGENTIC_POSTGRES_BIN_DIR", ""))
    except EmbeddedPostgresError:
        return False
    return hasattr(os, "geteuid") and os.geteuid() != 0


pytestmark = pytest.mark.skipif(
    not _pg_available(), reason="PostgreSQL binaries not installed (or running as root)"
)


class _S3:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects

    async def get_bytes(self, bucket: str, key: str, *, version_id: str | None = None) -> bytes:
        return self.objects[key]

    async def head_object(self, bucket: str, key: str) -> dict[str, Any]:
        return {"ETag": '"v1"'}

    async def list_objects(self, bucket: str, prefix: str = "", *, page_size: int = 1000) -> Any:
        raise AssertionError("a postgres snapshot must not mirror files")
        yield  # pragma: no cover


@pytest.fixture
def context() -> TenantContext:
    return TenantContext(environment="foxtrot-Dev", tenant="foxtrot", base_url="foxtrot-dev.example.io/foxtrot")


async def _build_dump(root: Path) -> bytes:
    """A real index in a real database, dumped exactly as the publisher dumps it."""
    source = EmbeddedPostgres(root / "source")
    await source.ensure_started()
    try:
        await source._admin('CREATE DATABASE "src"')
        db = PgDatabase(source.dsn("src"), dim=DIM)
        vec, kw, md = PgVectorIndex(db), PgKeywordIndex(db), PgMetadataStore(db)
        await vec.ensure()
        payload = {"document_id": "d1", "source_key": "foxtrot/policy.pdf", "page_start": 1}
        await vec.upsert([("c1", [1.0] + [0.0] * (DIM - 1), payload)])
        await kw.add([("c1", "encryption at rest policy", payload)])
        await md.put(
            DocumentMetadata(document_id="d1", title="Encryption Policy"),
            namespace="default",
            source_key="foxtrot/policy.pdf",
            facets={"business_object": "SUPPLIERDOCUMENTS"},
        )
        await db.aclose()
        dump_file = root / DUMP_KEY
        await asyncio.to_thread(
            subprocess.run,
            [
                str(find_bin_dir() / "pg_dump"), "-h", str(source._socket_dir), "-U", SUPERUSER,
                "-d", "src", "--format=custom", "--compress=9", "--no-owner", "--no-privileges",
                "-f", str(dump_file),
            ],
            check=True,
        )
    finally:
        await source.aclose()
    return await asyncio.to_thread(dump_file.read_bytes)


def _published(dump: bytes, **manifest: Any) -> _S3:
    body = {
        "manifest_version": 2,
        "backend": "postgres",
        "embed_dim": DIM,
        "namespace": "default",
        "counts": {"chunks": 1},
        "dump": {"key": DUMP_KEY, "bytes": len(dump), "sha256": hashlib.sha256(dump).hexdigest()},
        **manifest,
    }
    return _S3({f"{PREFIX}/manifest.json": json.dumps(body).encode(), f"{PREFIX}/{DUMP_KEY}": dump})


def _settings(tmp_path: Path) -> AgenticSettings:
    return AgenticSettings(
        index_bucket=BUCKET,
        index_prefix_override=PREFIX,
        index_prefix_override_tenant="foxtrot-Dev/foxtrot",
        postgres_source=PostgresSource.S3,
        cache_dir=str(tmp_path / "cache"),
        cache_ttl_s=0.0,
        embed_dim=DIM,
    )


async def test_restores_a_real_dump_from_s3_and_serves_queries(
    tmp_path: Path, context: TenantContext
) -> None:
    dump = await _build_dump(tmp_path)
    store = TenantIndexStore(_settings(tmp_path), _published(dump))  # type: ignore[arg-type]
    try:
        index = await store.acquire(context)
        assert await index.vector.count() == 1
        assert [c for c, _ in await index.keyword.search("encryption", limit=5)] == ["c1"]
        assert (await index.vector.search([1.0] + [0.0] * (DIM - 1), limit=1))[0].chunk_id == "c1"
        assert (await index.metadata.facets("default"))["business_object"] == ["SUPPLIERDOCUMENTS"]
        # The dump was deleted once restored; a second acquire is served warm.
        assert not (index.root / DUMP_KEY).exists()
        assert await store.acquire(context) is index
    finally:
        await store.aclose()


async def test_a_dump_that_is_not_a_postgres_archive_is_refused(
    tmp_path: Path, context: TenantContext
) -> None:
    """Checksum-valid garbage must fail the restore loudly, not yield an empty index."""
    garbage = b"this is not a pg_dump archive"
    store = TenantIndexStore(_settings(tmp_path), _published(garbage))  # type: ignore[arg-type]
    try:
        # IndexUnavailableError -> the runtime answers 503, like a missing snapshot.
        with pytest.raises(IndexUnavailableError, match="could not be restored"):
            await store.acquire(context)
    finally:
        await store.aclose()


async def test_eviction_drops_the_tenant_database(tmp_path: Path, context: TenantContext) -> None:
    import psycopg

    dump = await _build_dump(tmp_path)
    store = TenantIndexStore(_settings(tmp_path), _published(dump))  # type: ignore[arg-type]
    try:
        await store.acquire(context)
        pg = store._embedded()

        def databases() -> set[str]:
            with psycopg.connect(pg.dsn("postgres")) as conn:
                rows = conn.execute("SELECT datname FROM pg_database").fetchall()
            return {r[0] for r in rows if r[0].startswith("t_")}

        assert len(await asyncio.to_thread(databases)) == 1
        await store._evict(context.key)
        assert await asyncio.to_thread(databases) == set()
    finally:
        await store.aclose()

