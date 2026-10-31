"""Per-tenant index snapshots, pulled from S3 and opened locally.

AgentCore Runtime containers are ephemeral and hold no state between deployments,
so a tenant's index cannot live in the image. It lives in S3 under

    s3://{bucket}/{prefix}/{environment}/{tenant}/
        manifest.json   -- snapshot version marker (its ETag is the cache key)
        qdrant/**       -- embedded Qdrant collection (vectors)
        keyword.db      -- SQLite FTS5 (keyword)
        metadata.db     -- SQLite (facets: business_object, status, object_code, ...)

and is materialised into the container's writable ``/tmp`` on first use, then held
open in a small LRU.

A snapshot whose manifest says ``"backend": "postgres"`` is a ``pg_dump`` beside
the manifest instead of a file set. By default (``AGENTIC_POSTGRES_SOURCE=s3``) the
dump is downloaded, checked against the manifest's sha256, and restored into a
PostgreSQL server private to this container (``embedded_pg``) -- one database per
tenant -- so S3 stays the only storage. ``AGENTIC_POSTGRES_SOURCE=dsn`` connects to
an already-running database instead.

Either way a warm tenant costs one HEAD per ``cache_ttl_s``; a changed manifest ETag
triggers a re-pull (and, for Postgres, a fresh restore). Snapshots are immutable
from this service's point of view -- indexing writes them, the agent only reads.

A tenant listed in ``AGENTIC_MONGO_COLLECTIONS`` skips all of the above: its index
is a live MongoDB Atlas collection, queried in place. Nothing is downloaded, so
there is no cold start; the manifest is a record in the collection, re-read every
``cache_ttl_s``; and a re-index is visible immediately, with no republish.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import random
import shutil
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from rag_search.stores import (
    DocumentMetadataStore,
    KeywordIndex,
    KeywordSearchIndex,
    LocalVectorIndex,
    MetadataStore,
    MongoIndexDatabase,
    MongoKeywordIndex,
    MongoMetadataStore,
    MongoVectorIndex,
    PgDatabase,
    PgKeywordIndex,
    PgMetadataStore,
    PgVectorIndex,
    VectorIndex,
)
from rag_search.stores.mongo import open_client

from rag_agentic.config import AgenticSettings, PostgresSource
from rag_agentic.embedded_pg import EmbeddedPostgres, EmbeddedPostgresError
from rag_agentic.tenancy import TenancyError, TenantContext
from rag_core.logging import get_logger

_log = get_logger("rag_agentic.index_store")

MANIFEST_KEY = "manifest.json"
# manifest["backend"] value that means "read the live database, don't download files".
POSTGRES_BACKEND = "postgres"
MONGO_BACKEND = "mongo"
# Bound the fan-out when pulling a snapshot: enough to saturate the link, not so
# many that a large tenant exhausts the connection pool other calls share.
_DOWNLOAD_CONCURRENCY = 8
# A snapshot is dominated by ONE object (Qdrant's storage.sqlite is ~80% of it), so
# object-level concurrency does nothing for the part that actually takes the time.
# Anything at least this large is fetched as parallel byte ranges instead.
_RANGED_DOWNLOAD_MIN_BYTES = 32 * 1024 * 1024
_RANGE_PART_BYTES = 16 * 1024 * 1024


class ObjectStore(Protocol):
    """The slice of ``rag_core.aws.S3Client`` this module needs (fakeable in tests)."""

    def list_objects(self, bucket: str, prefix: str = "", *, page_size: int = 1000) -> Any: ...

    async def get_bytes(self, bucket: str, key: str, *, version_id: str | None = None) -> bytes: ...

    async def head_object(self, bucket: str, key: str) -> dict[str, Any]: ...


def _presize(path: Path, size: int) -> None:
    """Create ``path`` at its final length so ranges can be written at any offset."""
    with path.open("wb") as handle:
        handle.truncate(size)


def _write_at(handle: Any, offset: int, data: bytes) -> None:
    handle.seek(offset)
    handle.write(data)


def _fresh_directory(path: Path) -> None:
    """An empty directory at ``path``: nothing from a previous, possibly failed, load."""
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


class IndexUnavailableError(RuntimeError):
    """No usable snapshot exists for this tenant."""


@dataclass(slots=True)
class TenantIndex:
    """One tenant's opened, read-only retrieval stack."""

    context: TenantContext
    vector: VectorIndex
    keyword: KeywordSearchIndex
    metadata: DocumentMetadataStore
    # Where the snapshot was materialised. Never created for a postgres snapshot,
    # which has nothing to download.
    root: Path
    version: str = ""
    loaded_at: float = field(default_factory=time.monotonic)
    checked_at: float = field(default_factory=time.monotonic)
    manifest: dict[str, Any] = field(default_factory=dict)
    # Extra teardown once the stores are closed (drops a restored tenant database).
    cleanup: Callable[[], Awaitable[None]] | None = None
    # Live indexes (Mongo) re-read their manifest in place instead of being reloaded.
    refresh_manifest: Callable[[], Awaitable[dict[str, Any]]] | None = None

    @property
    def namespace(self) -> str:
        """Namespace the snapshot was indexed under (metadata/facet scoping)."""
        return str(self.manifest.get("namespace") or "default")

    async def aclose(self) -> None:
        for resource in (self.vector, self.keyword, self.metadata):
            with contextlib.suppress(Exception):  # best-effort; the dir is going away
                await resource.aclose()
        if self.cleanup is not None:
            with contextlib.suppress(Exception):
                await self.cleanup()


class TenantIndexStore:
    """Opens (and caches) tenant index snapshots.

    Concurrency: one lock per tenant key, so N simultaneous requests for a cold
    tenant trigger exactly one download, while requests for *different* tenants
    never block each other.
    """

    def __init__(
        self,
        settings: AgenticSettings,
        s3: ObjectStore,
        *,
        embedded_postgres: EmbeddedPostgres | None = None,
    ) -> None:
        self._settings = settings
        self._s3 = s3
        # Started lazily, on the first Postgres snapshot restored from S3.
        self._embedded_pg = embedded_postgres
        self._cache: OrderedDict[str, TenantIndex] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}
        self._guard = asyncio.Lock()
        # Which tenant a single (non-templated) postgres_dsn has been bound to.
        # One database holds one tenant; see _dsn_for.
        self._dsn_tenant: str | None = None
        # One pymongo client (one connection pool) shared by every Mongo tenant.
        self._mongo_client: Any = None
        # Background pre-warms in flight, one per tenant. Held so the event loop
        # does not garbage-collect a task nobody is awaiting, and so a request
        # that needs the index warm can wait on the probe already running.
        self._prewarming: dict[str, asyncio.Task[None]] = {}
        # Monotonic deadline per tenant, until which its index counts as resident.
        self._warm_until: dict[str, float] = {}

    async def _lock_for(self, key: str) -> asyncio.Lock:
        async with self._guard:
            return self._locks.setdefault(key, asyncio.Lock())

    async def acquire(self, context: TenantContext) -> TenantIndex:
        """The tenant's index, downloading or refreshing it if needed."""
        if not self._settings.index_bucket and self._mongo_collection(context) is None:
            raise IndexUnavailableError("AGENTIC_INDEX_BUCKET is not set")

        key = context.key
        async with await self._lock_for(key):
            cached = self._cache.get(key)
            if cached is not None:
                if not await self._is_stale(cached):
                    self._cache.move_to_end(key)
                    return cached
                _log.info("tenant_index_stale", tenant=key, version=cached.version)
                await self._evict(key)
            return await self._load(context)

    # ---- staleness ----------------------------------------------------------

    async def _is_stale(self, index: TenantIndex) -> bool:
        ttl = self._settings.cache_ttl_s
        if ttl <= 0:
            return False  # never re-check; snapshot is pinned for the container's life
        if (time.monotonic() - index.checked_at) < ttl:
            return False
        index.checked_at = time.monotonic()
        if index.refresh_manifest is not None:
            # A live index never goes stale -- only its manifest (tenant codes, source
            # bucket) can change, and that is re-read in place. A failed read keeps
            # the current one rather than dropping a working tenant.
            try:
                if manifest := await index.refresh_manifest():
                    index.manifest = manifest
                    index.version = str(manifest.get("built_at") or index.version)
            except Exception as exc:
                _log.warning(
                    "tenant_manifest_refresh_failed", tenant=index.context.key, error=str(exc)
                )
            return False
        version = await self._remote_version(index.context)
        # An unreadable manifest keeps the warm copy: a transient S3 blip should
        # not drop a working tenant into a cold re-download.
        return bool(version) and version != index.version

    def _prefix_for(self, context: TenantContext) -> str:
        """S3 prefix holding this tenant's snapshot.

        Normally ``{index_prefix}/{environment}/{tenant}``; ``index_prefix_override``
        replaces that with one literal prefix for single-tenant deployments.
        """
        if override := self._settings.index_prefix_override.strip("/"):
            bound = self._settings.index_prefix_override_tenant.strip()
            if context.key != bound:
                # Another tenant resolved while one flat prefix is configured: serving
                # it would hand it the bound tenant's documents.
                raise TenancyError(
                    f"tenant {context.key!r} has no index here: "
                    f"AGENTIC_INDEX_PREFIX_OVERRIDE serves only {bound!r}"
                )
            return override
        return context.s3_prefix(self._settings.index_prefix)

    async def _remote_version(self, context: TenantContext) -> str:
        prefix = self._prefix_for(context)
        key = f"{prefix}/{MANIFEST_KEY}"
        try:
            head = await self._s3.head_object(self._settings.index_bucket, key)
        except Exception as exc:  # missing manifest, throttling, transient 5xx
            _log.debug("tenant_manifest_head_failed", tenant=context.key, error=str(exc))
            return ""
        etag = head.get("ETag") or head.get("etag") or ""
        return str(etag).strip('"')

    # ---- loading ------------------------------------------------------------

    async def _fetch_manifest(self, prefix: str) -> dict[str, Any]:
        """Read the snapshot manifest before deciding how to open it. A missing or
        unreadable manifest yields {} — the file backend tolerates that and falls
        back to configured defaults, exactly as before."""
        try:
            raw = await self._s3.get_bytes(self._settings.index_bucket, f"{prefix}/{MANIFEST_KEY}")
        except Exception as exc:
            _log.debug("tenant_manifest_get_failed", prefix=prefix, error=str(exc))
            return {}
        try:
            loaded = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            _log.warning("tenant_manifest_unreadable", prefix=prefix, error=str(exc))
            return {}
        return loaded if isinstance(loaded, dict) else {}

    async def _load(self, context: TenantContext) -> TenantIndex:
        if (collection := self._mongo_collection(context)) is not None:
            return await self._load_mongo(context, collection)
        prefix = self._prefix_for(context)
        manifest = await self._fetch_manifest(prefix)
        if str(manifest.get("backend") or "").lower() == POSTGRES_BACKEND:
            return await self._load_postgres(context, prefix, manifest)
        return await self._load_files(context, prefix, manifest)

    # ---- live MongoDB tenants ------------------------------------------------

    def _mongo_collection(self, context: TenantContext) -> str | None:
        """The Atlas collection serving this tenant, or None to use S3 snapshots."""
        if not self._settings.mongo_uri:
            return None
        return self._settings.mongo_collections.get(context.key)

    def _mongo(self) -> Any:
        if self._mongo_client is None:
            self._mongo_client = open_client(
                self._settings.mongo_uri,
                tls_ca_file=self._settings.mongo_tls_ca_file,
                pool_max=self._settings.mongo_pool_max,
            )
        return self._mongo_client

    async def _load_mongo(self, context: TenantContext, collection: str) -> TenantIndex:
        """Open the tenant's live collection. Nothing is downloaded or restored."""
        db = MongoIndexDatabase(
            database=self._settings.mongo_database,
            collection=collection,
            dim=self._settings.embed_dim,
            client=await asyncio.to_thread(self._mongo),
            exact_vector_search=self._settings.mongo_exact_vectors,
            query_timeout_ms=self._settings.mongo_query_timeout_ms,
        )
        try:
            manifest = await db.read_manifest()
            populated = await db.has_chunks()
        except Exception as exc:
            raise IndexUnavailableError(
                f"tenant {context.key!r} has no reachable index collection {collection!r}: {exc}"
            ) from exc
        # Same guard as a live Postgres DSN: an empty collection must be a loud 503,
        # never a confident answer with no documents behind it.
        if not populated:
            raise IndexUnavailableError(
                f"tenant {context.key!r} maps to collection {collection!r} but it holds no "
                "chunks -- it has not been indexed yet"
            )
        dim = int(manifest.get("embed_dim") or self._settings.embed_dim)
        if dim != self._settings.embed_dim:
            raise IndexUnavailableError(
                f"collection {collection!r} holds {dim}-dimensional vectors but this agent "
                f"embeds at {self._settings.embed_dim} (AGENTIC_EMBED_DIM)"
            )
        namespace = str(manifest.get("namespace") or "default")
        index = TenantIndex(
            context=context,
            vector=MongoVectorIndex(db),
            keyword=MongoKeywordIndex(db),
            metadata=MongoMetadataStore(db, namespace=namespace),
            # Never created: there is nothing to download.
            root=Path(self._settings.cache_dir) / context.environment / context.tenant,
            version=str(manifest.get("built_at") or ""),
            manifest=manifest,
            refresh_manifest=db.read_manifest,
        )
        self._cache[context.key] = index
        self._cache.move_to_end(context.key)
        await self._trim()
        _log.info(
            "tenant_index_loaded",
            tenant=context.key,
            backend=MONGO_BACKEND,
            collection=collection,
            chunks=manifest.get("counts", {}).get("chunks"),
            namespace=namespace,
        )
        return index

    async def _probe(self, index: TenantIndex) -> None:
        """One query against each index, to page it into Atlas's memory.

        A random direction walks a different part of the HNSW graph on every
        pass, so repeated probes keep the whole graph resident rather than one
        path through it.
        """
        probe = [random.gauss(0.0, 1.0) for _ in range(self._settings.embed_dim)]
        await index.vector.search(probe, limit=30)
        await index.keyword.search("document", limit=1)

    def _warm_ttl(self) -> float:
        """How long one probe keeps a tenant counted as warm.

        Tied to the keep-warm interval: the background loop re-probes on that
        cadence, so a tenant that stays mapped never falls out of it. Slightly
        longer than the interval so a probe that runs a little late does not
        leave a gap where every request pays to warm the index again.
        """
        interval = self._settings.mongo_keep_warm_s or 240.0
        return float(interval) * 1.5

    def _mark_warm(self, key: str) -> None:
        self._warm_until[key] = time.monotonic() + self._warm_ttl()

    def is_warm(self, context: TenantContext) -> bool:
        return time.monotonic() < self._warm_until.get(context.key, 0.0)

    async def ensure_warm(self, context: TenantContext) -> None:
        """Hold the caller until this tenant's search index is resident.

        A cold ``$vectorSearch`` on a large collection can exceed its time limit,
        and the search then quietly falls back to keyword-only: the request
        succeeds, the answer looks reasonable, and half the retrieval never ran.
        Waiting is the better trade -- the request was going to spend that time
        inside the timing-out search anyway, and it ends with a correct answer
        instead of a plausible one.

        Single-flight: concurrent requests for a cold tenant wait on one probe
        rather than thrashing the cluster with several at once, which is what
        made the cold case worse than it needed to be.

        Never fatal. If the probe fails or outlives ``mongo_warm_wait_s`` the
        request proceeds on a cold index -- the outcome it would have had anyway.
        """
        wait_s = float(self._settings.mongo_warm_wait_s or 0.0)
        if wait_s <= 0 or self._mongo_collection(context) is None:
            return
        if self.is_warm(context):
            return

        key = context.key
        started = time.monotonic()
        task = self._prewarming.get(key)
        if task is None or task.done():
            self.prewarm(context)
            task = self._prewarming.get(key)
        if task is None:  # prewarm declined (not a Mongo tenant)
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=wait_s)
        except TimeoutError:
            _log.warning(
                "tenant_index_warm_wait_timed_out",
                tenant=key, waited_s=round(time.monotonic() - started, 2),
            )
            return
        except Exception as exc:  # the probe's own failure is already logged
            _log.warning("tenant_index_warm_wait_failed", tenant=key, error=str(exc))
            return
        _log.info(
            "tenant_index_warm_waited",
            tenant=key, seconds=round(time.monotonic() - started, 2),
        )

    def prewarm(self, context: TenantContext) -> None:
        """Start paging this tenant's index in, without waiting for it.

        Called on the phase-1 object-code lookup, which touches only metadata and
        so leaves the vector index cold. ExampleApp then spends a few seconds
        filtering the returned codes against the user's ACL before asking the
        real question -- time the search index would otherwise spend idle, and
        long enough to absorb most of a cold ``$vectorSearch``.

        Deliberately fire-and-forget: phase 1 answers in milliseconds and must
        keep doing so, and a probe that fails changes nothing -- the next real
        query simply pays the cost it would have paid anyway.
        """
        key = context.key
        if self._mongo_collection(context) is None:
            return  # file and Postgres snapshots are local; there is nothing to page in
        existing = self._prewarming.get(key)
        if existing is not None and not existing.done():
            return  # one in flight already; a second would queue behind it

        async def _run() -> None:
            started = time.monotonic()
            try:
                await self._probe(await self.acquire(context))
                self._mark_warm(key)
                _log.info(
                    "tenant_index_prewarmed",
                    tenant=key,
                    seconds=round(time.monotonic() - started, 2),
                )
            except Exception as exc:
                _log.warning("tenant_index_prewarm_failed", tenant=key, error=str(exc))
            finally:
                self._prewarming.pop(key, None)

        self._prewarming[key] = asyncio.create_task(_run())

    async def warm(self) -> None:
        """Open every mapped Mongo tenant and run one query against each index.

        Called in the background at startup: the TLS handshake, authentication and
        the first ``$vectorSearch``/``$search`` all happen before any user asks
        anything. Failures are logged, never raised -- a tenant that cannot be
        warmed is simply opened on first use instead.

        The runtime repeats it every ``mongo_keep_warm_s``: Atlas pages an idle
        search index out of memory, and the next real query would read it from disk.
        """
        async def _warm_one(key: str) -> None:
            environment, _, tenant = key.partition("/")
            context = TenantContext(environment=environment, tenant=tenant, base_url="")
            started = time.monotonic()
            try:
                await self._probe(await self.acquire(context))
                self._mark_warm(key)
                _log.info(
                    "tenant_index_warmed",
                    tenant=key,
                    seconds=round(time.monotonic() - started, 2),
                )
            except Exception as exc:
                _log.warning("tenant_index_warm_failed", tenant=key, error=str(exc))

        # All tenants at once: each one's first vector query is the slow part, and
        # they share nothing but the connection pool.
        await asyncio.gather(*(_warm_one(key) for key in self._settings.mongo_collections))

    def _dsn_for(self, context: TenantContext) -> str:
        """The database this tenant's index lives in.

        One database per tenant, rendered from ``postgres_dsn_template``. Isolation
        is physical: the ``chunks`` table carries no tenant column, so two tenants
        sharing a database would see each other's documents.

        A plain ``postgres_dsn`` names ONE database and is therefore valid for exactly
        one tenant. Serving a second from it is refused here rather than silently
        leaking -- this is the last check before any query runs.
        """
        template = self._settings.postgres_dsn_template.strip()
        if template:
            return template.format(environment=context.environment, tenant=context.tenant)

        dsn = self._settings.postgres_dsn.strip()
        if not dsn:
            raise IndexUnavailableError(
                f"snapshot for tenant {context.key!r} is postgres-backed and "
                "AGENTIC_POSTGRES_SOURCE=dsn, but neither AGENTIC_POSTGRES_DSN nor "
                "AGENTIC_POSTGRES_DSN_TEMPLATE is set"
            )
        if self._dsn_tenant is None:
            self._dsn_tenant = context.key
        elif self._dsn_tenant != context.key:
            raise TenancyError(
                "AGENTIC_POSTGRES_DSN names a single database, already serving tenant "
                f"{self._dsn_tenant!r}; tenant {context.key!r} cannot share it. Set "
                "AGENTIC_POSTGRES_DSN_TEMPLATE containing '{tenant}' so each tenant "
                "gets its own database."
            )
        return dsn

    async def _load_postgres(
        self, context: TenantContext, prefix: str, manifest: dict[str, Any]
    ) -> TenantIndex:
        """Open the tenant's stores on Postgres: restored from S3, or a live DSN."""
        dim = int(manifest.get("embed_dim") or self._settings.embed_dim)
        namespace = str(manifest.get("namespace") or "default")
        root = Path(self._settings.cache_dir) / context.environment / context.tenant
        cleanup: Callable[[], Awaitable[None]] | None = None

        if self._settings.postgres_source is PostgresSource.DSN:
            dsn = self._dsn_for(context)
        else:
            dsn, cleanup = await self._restore_from_s3(context, prefix, manifest, root)

        db = PgDatabase(dsn, dim=dim)
        index = TenantIndex(
            context=context,
            vector=PgVectorIndex(db),
            keyword=PgKeywordIndex(db),
            metadata=PgMetadataStore(db, namespace=namespace),
            root=root,
            version=await self._remote_version(context),
            manifest=manifest,
            cleanup=cleanup,
        )

        if self._settings.postgres_source is PostgresSource.S3:
            # The checksum proves the bytes; the row count proves the restore produced
            # the index the publisher described.
            expected = (manifest.get("counts") or {}).get("chunks")
            if expected is not None:
                actual = await index.keyword.count()
                if actual != int(expected):
                    await index.aclose()
                    raise IndexUnavailableError(
                        f"restored snapshot for tenant {context.key!r} has {actual} chunks; "
                        f"its manifest says {expected}"
                    )
        else:
            # Pools connect lazily, so an unreachable or not-yet-loaded tenant
            # database would otherwise surface as a confident answer with no
            # documents behind it -- a 200 that reads like a real result. Prove
            # the database is there and populated before serving a single query.
            try:
                chunks = await index.keyword.count()
            except Exception as exc:
                await index.aclose()
                raise IndexUnavailableError(
                    f"tenant {context.key!r} has no reachable index database: {exc}"
                ) from exc
            if chunks == 0:
                await index.aclose()
                raise IndexUnavailableError(
                    f"tenant {context.key!r} has an index database but it holds no "
                    "chunks -- it has not been loaded yet"
                )

        self._cache[context.key] = index
        self._cache.move_to_end(context.key)
        await self._trim()
        _log.info(
            "tenant_index_loaded",
            tenant=context.key,
            backend=POSTGRES_BACKEND,
            source=self._settings.postgres_source.value,
            version=index.version,
            namespace=index.namespace,
            dim=dim,
        )
        return index

    async def _restore_from_s3(
        self,
        context: TenantContext,
        prefix: str,
        manifest: dict[str, Any],
        root: Path,
    ) -> tuple[str, Callable[[], Awaitable[None]]]:
        """Download the tenant's dump, verify it, restore it into the embedded server.

        Returns the DSN of the restored database and a callable that drops it.
        """
        dump = manifest.get("dump") or {}
        key = str(dump.get("key") or "")
        # A bare file name beside manifest.json, so a manifest cannot point the agent
        # at an object under some other tenant's prefix.
        if not key or "/" in key or "\\" in key or key.startswith("."):
            raise IndexUnavailableError(
                f"postgres snapshot for tenant {context.key!r} has no usable dump.key in its "
                f"manifest (got {key!r}); expected a file name next to manifest.json"
            )

        await asyncio.to_thread(_fresh_directory, root)
        dump_path = root / key

        started = time.monotonic()
        digest = await self._download_to(f"{prefix}/{key}", dump_path)
        downloaded_s = time.monotonic() - started

        expected_sha = str(dump.get("sha256") or "").lower()
        if expected_sha and digest != expected_sha:
            await asyncio.to_thread(dump_path.unlink, True)
            raise IndexUnavailableError(
                f"dump for tenant {context.key!r} failed its checksum "
                f"(manifest {expected_sha[:12]}..., downloaded {digest[:12]}...)"
            )
        if not expected_sha:
            _log.warning("tenant_dump_unverified_no_sha256", tenant=context.key)

        pg = self._embedded()
        # Outside the restore's error mapping on purpose: a server that cannot start
        # (missing binaries, running as root) is a deployment fault, not this tenant's.
        await pg.ensure_started()
        database = "t_" + hashlib.sha1(context.key.encode("utf-8")).hexdigest()[:24]
        started = time.monotonic()
        try:
            await pg.restore(database, dump_path)
        except EmbeddedPostgresError as exc:
            raise IndexUnavailableError(
                f"snapshot for tenant {context.key!r} could not be restored: {exc}"
            ) from exc
        finally:
            # The restored database is what gets queried; the dump is dead weight in /tmp.
            await asyncio.to_thread(dump_path.unlink, True)
        _log.info(
            "tenant_postgres_restored",
            tenant=context.key,
            database=database,
            bytes=dump.get("bytes"),
            download_s=round(downloaded_s, 2),
            restore_s=round(time.monotonic() - started, 2),
        )

        async def drop() -> None:
            await pg.drop(database)

        return pg.dsn(database), drop

    def _embedded(self) -> EmbeddedPostgres:
        if self._embedded_pg is None:
            # Beside the tenant directories, never inside one: eviction removes a
            # tenant's directory and must not take the server's data dir with it.
            # Tenant environments must start alphanumeric, so "_postgres" cannot clash.
            self._embedded_pg = EmbeddedPostgres(
                Path(self._settings.cache_dir) / "_postgres",
                bin_dir=self._settings.postgres_bin_dir,
                restore_jobs=self._settings.postgres_restore_jobs,
            )
        return self._embedded_pg

    async def _download_to(self, key: str, destination: Path) -> str:
        """Stream an object to disk and return its sha256. Streams when the client
        supports it, so a large dump never has to fit in memory."""
        bucket = self._settings.index_bucket
        digest = hashlib.sha256()
        stream = getattr(self._s3, "stream", None)
        with destination.open("wb") as sink:
            if stream is not None:
                async for block in stream(bucket, key):
                    digest.update(block)
                    await asyncio.to_thread(sink.write, block)
            else:
                data = await self._s3.get_bytes(bucket, key)
                digest.update(data)
                await asyncio.to_thread(sink.write, data)
        return digest.hexdigest()

    async def _load_files(
        self, context: TenantContext, prefix: str, manifest: dict[str, Any]
    ) -> TenantIndex:
        root = Path(self._settings.cache_dir) / context.environment / context.tenant
        # Always start from an empty directory: a half-written snapshot from a
        # previous failed pull would otherwise be silently mixed with the new one.
        if root.exists():
            await asyncio.to_thread(shutil.rmtree, root, True)
        root.mkdir(parents=True, exist_ok=True)

        downloaded = await self._download(prefix, root)
        if not downloaded:
            raise IndexUnavailableError(
                f"no index snapshot at s3://{self._settings.index_bucket}/{prefix}/ "
                f"for tenant {context.key!r}"
            )

        # The manifest was already fetched from S3; fall back to the downloaded copy
        # only if that read failed (e.g. a transient error on the HEAD/GET).
        manifest_path = root / MANIFEST_KEY
        if not manifest and manifest_path.exists():
            with contextlib.suppress(json.JSONDecodeError, UnicodeDecodeError):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        vector = LocalVectorIndex(
            str(root / "qdrant"),
            collection=str(manifest.get("collection") or self._settings.collection),
            dim=int(manifest.get("embed_dim") or self._settings.embed_dim),
        )
        index = TenantIndex(
            context=context,
            vector=vector,
            keyword=KeywordIndex(str(root / "keyword.db")),
            metadata=MetadataStore(str(root / "metadata.db")),
            root=root,
            version=await self._remote_version(context),
            manifest=manifest,
        )

        self._cache[context.key] = index
        self._cache.move_to_end(context.key)
        await self._trim()
        _log.info(
            "tenant_index_loaded",
            tenant=context.key,
            backend="local",
            objects=downloaded,
            version=index.version,
            namespace=index.namespace,
        )
        return index

    async def _download(self, prefix: str, root: Path) -> int:
        """Mirror ``s3://bucket/prefix/**`` into ``root``. Returns the object count."""
        bucket = self._settings.index_bucket
        semaphore = asyncio.Semaphore(_DOWNLOAD_CONCURRENCY)
        count = 0

        async def _one(key: str) -> None:
            relative = key[len(prefix) :].lstrip("/")
            if not relative:
                return  # the prefix "folder" marker itself
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)

            size = 0
            if hasattr(self._s3, "get_range"):
                with contextlib.suppress(Exception):
                    size = int((await self._s3.head_object(bucket, key)).get("ContentLength", 0))
            if size >= _RANGED_DOWNLOAD_MIN_BYTES:
                await self._download_ranged(bucket, key, destination, size)
                return

            async with semaphore:
                data = await self._s3.get_bytes(bucket, key)
            await asyncio.to_thread(destination.write_bytes, data)

        async with asyncio.TaskGroup() as group:
            async for summary in self._s3.list_objects(bucket, f"{prefix}/"):
                group.create_task(_one(summary.key))
                count += 1
        return count

    async def _download_ranged(
        self, bucket: str, key: str, destination: Path, size: int
    ) -> None:
        """Fetch one large object as parallel byte ranges.

        Parts land at their own offsets in a pre-sized file. The writes are
        serialised behind a lock -- they are memory-to-disk and fast, while the
        fetches, which are what actually costs, run concurrently.
        """
        await asyncio.to_thread(_presize, destination, size)
        semaphore = asyncio.Semaphore(_DOWNLOAD_CONCURRENCY)
        write_lock = asyncio.Lock()

        with destination.open("r+b") as sink:

            async def _part(start: int) -> None:
                end = min(start + _RANGE_PART_BYTES, size) - 1
                async with semaphore:
                    data = await self._s3.get_range(bucket, key, start, end)  # type: ignore[attr-defined]
                async with write_lock:
                    await asyncio.to_thread(_write_at, sink, start, data)

            async with asyncio.TaskGroup() as group:
                for start in range(0, size, _RANGE_PART_BYTES):
                    group.create_task(_part(start))

        _log.info(
            "ranged_download",
            key=key,
            mib=round(size / 1_048_576, 1),
            parts=-(-size // _RANGE_PART_BYTES),
        )

    # ---- eviction -----------------------------------------------------------

    async def _trim(self) -> None:
        while len(self._cache) > self._settings.cache_max_tenants:
            oldest = next(iter(self._cache))
            await self._evict(oldest)

    async def _evict(self, key: str) -> None:
        # A reopened index is cold again, whatever the old deadline said.
        self._warm_until.pop(key, None)
        index = self._cache.pop(key, None)
        if index is None:
            return
        await index.aclose()
        # Reclaim the disk too -- /tmp in a runtime container is small and shared.
        await asyncio.to_thread(shutil.rmtree, index.root, True)
        _log.info("tenant_index_evicted", tenant=key)

    async def aclose(self) -> None:
        for task in list(self._prewarming.values()):
            task.cancel()
        for task in list(self._prewarming.values()):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._prewarming.clear()
        for key in list(self._cache):
            await self._evict(key)
        if self._embedded_pg is not None:
            await self._embedded_pg.aclose()
        if self._mongo_client is not None:
            client, self._mongo_client = self._mongo_client, None
            await asyncio.to_thread(client.close)
