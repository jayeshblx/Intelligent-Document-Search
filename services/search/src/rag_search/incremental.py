"""Incremental indexing and DeleteMark purging for a live tenant index.

Two scheduled jobs (``lambda_handler.py`` runs them from EventBridge; the CLI is
``scripts/incremental_index.py``), one tenant collection each:

``IncrementalIndexer.run``
    1. List the tenant's S3 prefix. Candidates are objects modified since the last
       run's watermark, plus objects from earlier runs still waiting for metadata.
    2. Skip images, already-indexed documents and other tenants' files.
    3. Look up ONLY the candidates in ``AttachmentMetaData`` (no whole-collection load).
    4. With metadata: embed + write them to the index. Without: park them as pending
       and retry next run -- the catalogue record often lands after the file. A file
       that never gets a record is dropped after ``pending_max_age``.
    5. Advance the watermark only once every candidate is indexed or parked, so a run
       cut off by the Lambda timeout repeats safely (indexed documents are skipped).

``DeleteMarkPurger.run``
    Find this tenant's ``DeleteMark=1`` records and remove those documents' chunks,
    vectors and metadata from the index. No S3, no Bedrock.

State (watermark, pending files, a run lock so overlapping invocations cannot
double-embed) lives in ``<collection>_IndexState`` beside the index collection --
never in the index collection itself, whose Atlas search indexes would cover it.
"""

from __future__ import annotations

import socket
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from rag_core.logging import get_logger
from rag_core.utils import document_id, infer_document_type
from rag_search.indexer import NON_TEXT_TYPES, IndexingService
from rag_search.mongo_source import MongoMetadataSource, basename
from rag_search.stores import DocumentMetadataStore

_log = get_logger("rag_search.incremental")

WATERMARK_ID = "_watermark"
LOCK_ID = "_lock"
PENDING = "pending"
# S3 LastModified is the upload START; a large upload still in flight when one run
# lists the prefix is seen by the next only if the windows overlap.
_WATERMARK_OVERLAP = timedelta(minutes=10)


class AlreadyRunning(RuntimeError):
    """Another run holds this index's lock."""


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------


@dataclass(slots=True)
class PendingFile:
    key: str
    first_seen: datetime
    attempts: int = 0


class IndexState(Protocol):
    async def acquire(self, owner: str, ttl: timedelta) -> bool: ...
    async def release(self, owner: str) -> None: ...
    async def watermark(self) -> datetime | None: ...
    async def set_watermark(self, value: datetime) -> None: ...
    async def pending(self) -> list[PendingFile]: ...
    async def put_pending(self, files: list[PendingFile]) -> None: ...
    async def drop_pending(self, keys: list[str]) -> None: ...


class InMemoryIndexState:
    """For tests and dry runs."""

    def __init__(self) -> None:
        self._watermark: datetime | None = None
        self._pending: dict[str, PendingFile] = {}
        self._lock: tuple[str, datetime] | None = None

    async def acquire(self, owner: str, ttl: timedelta) -> bool:
        now = datetime.now(UTC)
        if self._lock and self._lock[0] != owner and self._lock[1] > now:
            return False
        self._lock = (owner, now + ttl)
        return True

    async def release(self, owner: str) -> None:
        if self._lock and self._lock[0] == owner:
            self._lock = None

    async def watermark(self) -> datetime | None:
        return self._watermark

    async def set_watermark(self, value: datetime) -> None:
        self._watermark = value

    async def pending(self) -> list[PendingFile]:
        return list(self._pending.values())

    async def put_pending(self, files: list[PendingFile]) -> None:
        self._pending.update({f.key: f for f in files})

    async def drop_pending(self, keys: list[str]) -> None:
        for key in keys:
            self._pending.pop(key, None)


def _aware(value: Any) -> datetime | None:
    """pymongo returns naive UTC datetimes unless the client is tz_aware."""
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


class MongoIndexState:
    """State in ``<collection>_IndexState`` of the index database."""

    def __init__(self, db: Any) -> None:  # rag_search.stores.mongo.MongoIndexDatabase
        self._db = db
        self._name = f"{db.collection_name}_IndexState"

    async def _run(self, fn: Any) -> Any:
        return await self._db.run(lambda coll: fn(coll.database[self._name]))

    async def acquire(self, owner: str, ttl: timedelta) -> bool:
        from pymongo.errors import DuplicateKeyError

        now = datetime.now(UTC)

        def _op(coll: Any) -> bool:
            try:
                # Matches only a free or expired lock; otherwise the upsert collides
                # with the live lock's _id and raises.
                coll.update_one(
                    {"_id": LOCK_ID, "$or": [{"expires": {"$lt": now}}, {"owner": owner}]},
                    {"$set": {"owner": owner, "expires": now + ttl, "host": socket.gethostname()}},
                    upsert=True,
                )
                return True
            except DuplicateKeyError:
                return False

        return bool(await self._run(_op))

    async def release(self, owner: str) -> None:
        await self._run(lambda coll: coll.delete_one({"_id": LOCK_ID, "owner": owner}))

    async def watermark(self) -> datetime | None:
        doc = await self._run(lambda coll: coll.find_one({"_id": WATERMARK_ID}))
        return _aware((doc or {}).get("value"))

    async def set_watermark(self, value: datetime) -> None:
        await self._run(
            lambda coll: coll.update_one(
                {"_id": WATERMARK_ID}, {"$set": {"value": value}}, upsert=True
            )
        )

    async def pending(self) -> list[PendingFile]:
        rows = await self._run(lambda coll: list(coll.find({"kind": PENDING})))
        return [
            PendingFile(
                key=r["key"],
                first_seen=_aware(r.get("first_seen")) or datetime.now(UTC),
                attempts=int(r.get("attempts") or 0),
            )
            for r in rows
        ]

    async def put_pending(self, files: list[PendingFile]) -> None:
        if not files:
            return
        from pymongo import UpdateOne

        ops = [
            UpdateOne(
                {"_id": f"pending::{f.key}"},
                {
                    "$set": {"kind": PENDING, "key": f.key, "attempts": f.attempts},
                    "$setOnInsert": {"first_seen": f.first_seen},
                },
                upsert=True,
            )
            for f in files
        ]
        await self._run(lambda coll: coll.bulk_write(ops, ordered=False))

    async def drop_pending(self, keys: list[str]) -> None:
        if keys:
            ids = [f"pending::{k}" for k in keys]
            await self._run(lambda coll: coll.delete_many({"_id": {"$in": ids}}))


# --------------------------------------------------------------------------
# Incremental indexing
# --------------------------------------------------------------------------


@dataclass(slots=True)
class IncrementalSummary:
    collection: str
    bucket: str
    prefix: str
    listed: int = 0
    candidates: int = 0
    already_indexed: int = 0
    non_text: int = 0
    other_tenant: int = 0
    deleted: int = 0
    awaiting_metadata: int = 0
    pending_expired: int = 0
    indexed: int = 0
    chunks: int = 0
    failed: int = 0
    deferred: int = 0  # out of time this run; picked up by the next
    watermark: str | None = None
    duration_s: float = 0.0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


class IncrementalIndexer:
    def __init__(
        self,
        *,
        indexer: IndexingService,
        metadata_store: DocumentMetadataStore,
        mongo: MongoMetadataSource,
        state: IndexState,
        s3: Any,
        bucket: str,
        prefix: str,
        collection: str = "",
        namespace: str = "default",
        initial_lookback: timedelta = timedelta(days=1),
        pending_max_age: timedelta = timedelta(days=7),
        lock_ttl: timedelta = timedelta(minutes=20),
    ) -> None:
        self._indexer = indexer
        self._metadata = metadata_store
        self._mongo = mongo
        self._state = state
        self._s3 = s3
        self._bucket = bucket
        self._prefix = prefix
        self._collection = collection
        self._namespace = namespace
        self._initial_lookback = initial_lookback
        self._pending_max_age = pending_max_age
        self._lock_ttl = lock_ttl

    async def run(self, *, deadline: float | None = None) -> IncrementalSummary:
        """One sweep. ``deadline`` is a ``time.monotonic()`` instant to stop embedding by."""
        owner = uuid.uuid4().hex
        if not await self._state.acquire(owner, self._lock_ttl):
            raise AlreadyRunning(f"{self._collection or 'index'}: another run holds the lock")
        try:
            return await self._sweep(deadline)
        finally:
            await self._state.release(owner)

    async def _sweep(self, deadline: float | None) -> IncrementalSummary:
        started = time.perf_counter()
        now = datetime.now(UTC)
        out = IncrementalSummary(self._collection, self._bucket, self._prefix)

        # 1. Candidates: new since the watermark, plus files parked for metadata.
        watermark = await self._state.watermark() or now - self._initial_lookback
        since = watermark - _WATERMARK_OVERLAP
        pending = {p.key: p for p in await self._state.pending()}
        candidates: dict[str, None] = dict.fromkeys(pending)
        async for obj in self._s3.list_objects(self._bucket, self._prefix):
            out.listed += 1
            modified = obj.last_modified
            if modified is None or (_aware(modified) or now) > since:
                candidates.setdefault(obj.key)
        out.candidates = len(candidates)

        # 2. Cheap skips before any metadata lookup.
        keys: list[str] = []
        for key in candidates:
            if infer_document_type(key) in NON_TEXT_TYPES:
                out.non_text += 1
                continue
            keys.append(key)
        ids = [document_id(self._bucket, k) for k in keys]
        indexed: set[str] = set()
        for start in range(0, len(ids), 1000):
            indexed.update(await self._metadata.describe(ids[start : start + 1000], self._namespace))
        fresh = [k for k in keys if document_id(self._bucket, k) not in indexed]
        out.already_indexed = len(keys) - len(fresh)
        resolved = [k for k in pending if k not in fresh]  # indexed or skipped since

        # 3. Metadata for just these files.
        await self._mongo.load_keys(fresh)
        to_index: list[str] = []
        waiting: list[PendingFile] = []
        expired: list[str] = []
        for key in fresh:
            if self._mongo.is_deleted(key):
                out.deleted += 1
                resolved.append(key)
            elif self._mongo.is_foreign(key):
                out.other_tenant += 1
                resolved.append(key)
            elif self._mongo.lookup(key) is not None:
                to_index.append(key)
            else:
                parked = pending.get(key) or PendingFile(key=key, first_seen=now)
                if now - parked.first_seen > self._pending_max_age:
                    expired.append(key)
                else:
                    parked.attempts += 1
                    waiting.append(parked)
        out.awaiting_metadata = len(waiting)
        out.pending_expired = len(expired)
        if expired:
            _log.info("pending_expired_without_metadata", files=len(expired), sample=expired[:5])

        # 4. Embed + write.
        summary, unattempted = await self._indexer.index_keys(
            to_index, self._namespace, deadline=deadline
        )
        out.indexed, out.chunks = summary.documents_indexed, summary.chunks_indexed
        out.failed, out.errors = summary.documents_failed, summary.error_samples
        out.deferred = len(unattempted)
        done = [k for k in to_index if k not in set(unattempted)]

        # 5. Persist. Failed files are retried only if they change again in S3 (a
        # broken PDF would otherwise be re-downloaded every run); deferred ones are
        # parked so the next run resumes them without waiting for S3 to change.
        await self._state.drop_pending([*resolved, *done, *expired])
        await self._state.put_pending(
            [*waiting, *(pending.get(k) or PendingFile(key=k, first_seen=now) for k in unattempted)]
        )
        await self._state.set_watermark(now)
        out.watermark = now.isoformat()
        out.duration_s = round(time.perf_counter() - started, 3)
        _log.info("incremental_index_complete", **out.as_dict())
        return out


# --------------------------------------------------------------------------
# DeleteMark purge
# --------------------------------------------------------------------------


@dataclass(slots=True)
class PurgeSummary:
    collection: str
    deleted_in_catalogue: int = 0
    indexed_documents: int = 0
    purged: int = 0
    dry_run: bool = False
    sample: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


class DeleteMarkPurger:
    def __init__(
        self,
        *,
        indexer: IndexingService,
        metadata_store: DocumentMetadataStore,
        mongo: MongoMetadataSource,
        collection: str = "",
        namespace: str = "default",
    ) -> None:
        self._indexer = indexer
        self._metadata = metadata_store
        self._mongo = mongo
        self._collection = collection
        self._namespace = namespace

    async def run(self, *, dry_run: bool = False) -> PurgeSummary:
        started = time.perf_counter()
        out = PurgeSummary(self._collection, dry_run=dry_run)
        deleted = await self._mongo.deleted_names()
        out.deleted_in_catalogue = len(deleted)
        indexed = await self._metadata.all_documents()
        out.indexed_documents = len(indexed)
        doomed = [key for _, key in indexed if basename(key) in deleted]
        out.sample = [basename(k) for k in doomed[:10]]
        if not dry_run:
            for key in doomed:
                await self._indexer.purge_document(key, self._namespace)
        out.purged = len(doomed)
        out.duration_s = round(time.perf_counter() - started, 3)
        _log.info("delete_mark_purge_complete", **out.as_dict())
        return out
