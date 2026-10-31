"""Opening a tenant snapshot whose manifest declares the Postgres backend.

Such a snapshot has no files in S3 — only ``manifest.json`` (the version marker)
and a ``.dump`` restore artifact — so the store must read the live database via
``AGENTIC_POSTGRES_DSN`` and download nothing.

Needs a live Postgres; the module skips when none is reachable. A DEDICATED test
database is used, never the one a real index is published from.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
from rag_agentic.config import AgenticSettings, PostgresSource
from rag_agentic.index_store import IndexUnavailableError, TenantIndexStore
from rag_agentic.tenancy import TenantContext
from rag_search.stores import PgDatabase, PgKeywordIndex, PgMetadataStore, PgVectorIndex

from rag_core.models.metadata import DocumentMetadata

psycopg = pytest.importorskip("psycopg", reason="psycopg not installed")

ADMIN_DSN = os.environ.get("AGENTIC_POSTGRES_DSN", "postgresql://rag:rag@localhost:5432/rag")
TEST_DB = os.environ.get("AGENTIC_POSTGRES_TEST_DB", "rag_test")
DSN = ADMIN_DSN.rsplit("/", 1)[0] + "/" + TEST_DB

BUCKET = "test-index-bucket"
PREFIX = "DocumentSearch/foxtrot-Dev"
DIM = 8


def _reachable() -> bool:
    try:
        with psycopg.connect(ADMIN_DSN, connect_timeout=5) as conn:
            conn.autocommit = True
            if not conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)
            ).fetchone():
                conn.execute(f'CREATE DATABASE "{TEST_DB}"')
        with psycopg.connect(DSN, connect_timeout=5):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _reachable(), reason=f"no Postgres at {ADMIN_DSN} (try: docker compose up -d postgres)"
)


class _ManifestOnlyS3:
    """S3 holding just a manifest — no snapshot files.

    ``list_objects`` records that it was reached and yields nothing, so tests can
    assert the file-mirroring path was skipped (postgres) or entered (local).
    """

    def __init__(self, objects: dict[str, bytes]) -> None:
        self._objects = objects
        self.gets: list[str] = []
        self.listed = False

    async def get_bytes(self, bucket: str, key: str, *, version_id: str | None = None) -> bytes:
        self.gets.append(key)
        if key not in self._objects:
            raise FileNotFoundError(key)
        return self._objects[key]

    async def head_object(self, bucket: str, key: str) -> dict[str, Any]:
        if key not in self._objects:
            raise FileNotFoundError(key)
        return {"ETag": f'"{hash(self._objects[key]) & 0xFFFFFF:06x}"'}

    async def list_objects(self, bucket: str, prefix: str = "", *, page_size: int = 1000) -> Any:
        self.listed = True
        return
        yield  # pragma: no cover - makes this an async generator


def _manifest(**overrides: Any) -> bytes:
    body = {
        "manifest_version": 2,
        "backend": "postgres",
        "tenant": "foxtrot",
        "environment": "foxtrot-Dev",
        "collection": "compliance",
        "embed_dim": DIM,
        "namespace": "default",
        "dump": {"key": "rag-20260910T000000Z.dump", "bytes": 123},
    }
    body.update(overrides)
    return json.dumps(body).encode()


@pytest.fixture
def context() -> TenantContext:
    return TenantContext(environment="foxtrot-Dev", tenant="foxtrot", base_url="foxtrot-dev.example.io/foxtrot")


@pytest.fixture
def settings(tmp_path: Path) -> AgenticSettings:
    return AgenticSettings(
        index_bucket=BUCKET,
        index_prefix_override=PREFIX,  # the literal prefix the snapshot is published to
        index_prefix_override_tenant="foxtrot-Dev/foxtrot",
        # These tests exercise the live-database mode; S3 restore is tested below.
        postgres_source=PostgresSource.DSN,
        postgres_dsn=DSN,
        cache_dir=str(tmp_path / "cache"),
        cache_ttl_s=0.0,
        embed_dim=DIM,
        collection="compliance",
    )


@pytest.fixture
async def seeded() -> Any:
    """A small real index in the test database."""
    with psycopg.connect(ADMIN_DSN, connect_timeout=10) as conn:
        conn.autocommit = True
        if not conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)
        ).fetchone():
            conn.execute(f'CREATE DATABASE "{TEST_DB}"')
    with psycopg.connect(DSN, connect_timeout=10) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks, document_metadata, document_facet")
        conn.commit()

    db = PgDatabase(DSN, dim=DIM)
    vec, kw, md = PgVectorIndex(db), PgKeywordIndex(db), PgMetadataStore(db)
    await vec.ensure()
    payload = {"document_id": "d1", "source_key": "policy.pdf", "page_start": 1}
    await vec.upsert([("c1", [1.0] + [0.0] * (DIM - 1), payload)])
    await kw.add([("c1", "encryption at rest policy", payload)])
    await md.put(
        DocumentMetadata(document_id="d1", title="Encryption Policy"),
        namespace="default",
        source_key="policy.pdf",
        facets={"status": "Active"},
    )
    try:
        yield db
    finally:
        await db.aclose()


async def test_opens_postgres_snapshot_without_downloading(
    settings: AgenticSettings, context: TenantContext, seeded: Any
) -> None:
    s3 = _ManifestOnlyS3({f"{PREFIX}/manifest.json": _manifest()})
    store = TenantIndexStore(settings, s3)  # type: ignore[arg-type]
    try:
        index = await store.acquire(context)

        # Real data, read straight from Postgres.
        assert await index.keyword.count() == 1
        assert await index.vector.count() == 1
        assert await index.metadata.count() == 1
        assert index.namespace == "default"
        assert [c for c, _ in await index.keyword.search("encryption", limit=5)] == ["c1"]

        # Nothing was materialised on disk, no listing, only the manifest fetched.
        assert not index.root.exists()
        assert s3.listed is False
        assert s3.gets == [f"{PREFIX}/manifest.json"]
    finally:
        await store.aclose()


async def test_prefix_override_is_used_verbatim(
    settings: AgenticSettings, context: TenantContext, seeded: Any
) -> None:
    """Without the override the store would look under
    ``{index_prefix}/{environment}/{tenant}``; the snapshot lives at a flat prefix."""
    s3 = _ManifestOnlyS3({f"{PREFIX}/manifest.json": _manifest()})
    store = TenantIndexStore(settings, s3)  # type: ignore[arg-type]
    try:
        await store.acquire(context)
        assert all(key.startswith(f"{PREFIX}/") for key in s3.gets)
    finally:
        await store.aclose()


async def test_missing_dsn_is_an_explicit_error(
    settings: AgenticSettings, context: TenantContext
) -> None:
    settings = settings.model_copy(update={"postgres_dsn": ""})
    s3 = _ManifestOnlyS3({f"{PREFIX}/manifest.json": _manifest()})
    store = TenantIndexStore(settings, s3)  # type: ignore[arg-type]
    # Names both settings now: a multi-tenant deployment configures the template.
    with pytest.raises(IndexUnavailableError, match="AGENTIC_POSTGRES_DSN"):
        await store.acquire(context)


async def test_embed_dim_comes_from_the_manifest(
    settings: AgenticSettings, context: TenantContext, seeded: Any
) -> None:
    """The snapshot's dimension wins over the service default, so one deployment can
    serve tenants embedded at different dimensions."""
    settings = settings.model_copy(update={"embed_dim": 1024})  # deliberately wrong
    s3 = _ManifestOnlyS3({f"{PREFIX}/manifest.json": _manifest(embed_dim=DIM)})
    store = TenantIndexStore(settings, s3)  # type: ignore[arg-type]
    try:
        index = await store.acquire(context)
        assert await index.vector.count() == 1  # would raise on a dimension mismatch
    finally:
        await store.aclose()


async def test_local_backend_manifest_still_mirrors_files(
    settings: AgenticSettings, context: TenantContext
) -> None:
    """A manifest that is not "postgres" keeps the original file-mirroring path, so
    the branch turns on the manifest value and existing snapshots are unaffected."""
    s3 = _ManifestOnlyS3({f"{PREFIX}/manifest.json": _manifest(backend="local")})
    store = TenantIndexStore(settings, s3)  # type: ignore[arg-type]
    with pytest.raises(IndexUnavailableError, match="no index snapshot"):
        await store.acquire(context)
    assert s3.listed is True


# --------------------------------------------------------------------------
# AGENTIC_POSTGRES_SOURCE=s3: the dump is pulled from S3 and restored locally
# --------------------------------------------------------------------------

DUMP_KEY = "rag-20260917T000000Z.dump"
DUMP_BYTES = b"PGDMP-fake-archive-bytes"


def _sha(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


class _FakeEmbedded:
    """Stands in for EmbeddedPostgres. "Restoring" points at the pre-seeded test
    database, so the logic around a restore is exercised with real stores."""

    def __init__(self) -> None:
        self.restored: list[tuple[str, bytes]] = []
        self.dropped: list[str] = []
        self.closed = False

    async def ensure_started(self) -> None:
        return None

    async def restore(self, database: str, dump_path: Path) -> None:
        self.restored.append((database, await asyncio.to_thread(dump_path.read_bytes)))

    async def drop(self, database: str) -> None:
        self.dropped.append(database)

    def dsn(self, database: str) -> str:
        return DSN

    async def aclose(self) -> None:
        self.closed = True


class _DumpS3(_ManifestOnlyS3):
    """Manifest plus the dump object it names."""


def _s3_settings(settings: AgenticSettings) -> AgenticSettings:
    return settings.model_copy(update={"postgres_source": PostgresSource.S3, "postgres_dsn": ""})


def _s3_bucket(**manifest_overrides: Any) -> _DumpS3:
    dump = {"key": DUMP_KEY, "bytes": len(DUMP_BYTES), "sha256": _sha(DUMP_BYTES)}
    manifest = {"dump": dump, "counts": {"chunks": 1}, **manifest_overrides}
    return _DumpS3(
        {
            f"{PREFIX}/manifest.json": _manifest(**manifest),
            f"{PREFIX}/{DUMP_KEY}": DUMP_BYTES,
        }
    )


async def test_s3_source_downloads_verifies_and_restores_the_dump(
    settings: AgenticSettings, context: TenantContext, seeded: Any
) -> None:
    fake = _FakeEmbedded()
    s3 = _s3_bucket()
    store = TenantIndexStore(_s3_settings(settings), s3, embedded_postgres=fake)  # type: ignore[arg-type]
    try:
        index = await store.acquire(context)

        # Restored exactly the bytes that are in S3, into one tenant database.
        assert len(fake.restored) == 1
        database, restored_bytes = fake.restored[0]
        assert restored_bytes == DUMP_BYTES
        assert database.startswith("t_")
        # Queryable afterwards, and the dump no longer occupies /tmp.
        assert await index.keyword.count() == 1
        assert not (index.root / DUMP_KEY).exists()
        assert s3.listed is False
    finally:
        await store.aclose()

    # Closing the store drops the tenant database and stops the server.
    assert fake.dropped == [database]
    assert fake.closed is True


async def test_s3_source_rejects_a_corrupted_dump_before_restoring(
    settings: AgenticSettings, context: TenantContext
) -> None:
    fake = _FakeEmbedded()
    s3 = _s3_bucket(dump={"key": DUMP_KEY, "bytes": 1, "sha256": "0" * 64})
    store = TenantIndexStore(_s3_settings(settings), s3, embedded_postgres=fake)  # type: ignore[arg-type]
    with pytest.raises(IndexUnavailableError, match="failed its checksum"):
        await store.acquire(context)
    assert fake.restored == []


async def test_s3_source_refuses_a_restore_that_does_not_match_the_manifest(
    settings: AgenticSettings, context: TenantContext, seeded: Any
) -> None:
    """The seeded database has 1 chunk; a manifest promising 500 means a bad restore."""
    fake = _FakeEmbedded()
    s3 = _s3_bucket(counts={"chunks": 500})
    store = TenantIndexStore(_s3_settings(settings), s3, embedded_postgres=fake)  # type: ignore[arg-type]
    with pytest.raises(IndexUnavailableError, match="has 1 chunks; its manifest says 500"):
        await store.acquire(context)
    assert len(fake.dropped) == 1  # the half-trusted database is not left behind


@pytest.mark.parametrize("bad_key", ["../globex/rag.dump", "other/rag.dump", "", ".hidden"])
async def test_s3_source_only_restores_a_dump_beside_the_manifest(
    bad_key: str, settings: AgenticSettings, context: TenantContext
) -> None:
    fake = _FakeEmbedded()
    s3 = _s3_bucket(dump={"key": bad_key, "sha256": _sha(DUMP_BYTES)})
    store = TenantIndexStore(_s3_settings(settings), s3, embedded_postgres=fake)  # type: ignore[arg-type]
    with pytest.raises(IndexUnavailableError, match=r"no usable dump\.key"):
        await store.acquire(context)
    assert fake.restored == []


async def test_s3_source_is_the_default(settings: AgenticSettings) -> None:
    assert AgenticSettings().postgres_source is PostgresSource.S3


# ---- a tenant database that is missing or unloaded must fail, not answer empty ----

async def test_an_unloaded_tenant_database_is_unavailable(
    settings: AgenticSettings, context: TenantContext
) -> None:
    """Connection pools open lazily, so an empty tenant database would otherwise
    serve a confident answer with no documents behind it -- a 200 that reads like a
    real result. It must be a 503 instead."""
    empty_db = "rag_empty_tenant_test"
    admin = ADMIN_DSN.rsplit("/", 1)[0]
    with psycopg.connect(ADMIN_DSN, connect_timeout=5) as conn:
        conn.autocommit = True
        if not conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (empty_db,)
        ).fetchone():
            conn.execute(f'CREATE DATABASE "{empty_db}"')

    settings = settings.model_copy(update={"postgres_dsn": f"{admin}/{empty_db}"})
    s3 = _ManifestOnlyS3({f"{PREFIX}/manifest.json": _manifest()})
    store = TenantIndexStore(settings, s3)  # type: ignore[arg-type]

    with pytest.raises(IndexUnavailableError, match=r"no chunks|not been loaded"):
        await store.acquire(context)
    await store.aclose()

