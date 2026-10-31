"""Deduplication + version ledger.

The ledger answers two questions during a sync:
  1. Has this object (by id) changed since we last saw it?  -> version detection
  2. Have we already indexed these exact bytes (by checksum)? -> deduplication

Two impls share one interface so tests run in-memory and prod runs on Redis (fast,
shared across the ingestion fleet, survives restarts). The ledger is intentionally
small — it stores change-detection state, not document content.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

from rag_core.models.base import DomainModel


class DocRecord(DomainModel):
    """Last-known state of a tracked document, used for change detection.

    ``document_id`` is the *logical* (key-derived) id and is the ledger key;
    ``physical_id`` is the id of the most recent content version (so a new version
    can record what it supersedes).
    """

    document_id: str
    physical_id: str | None = None
    etag: str | None = None
    checksum: str | None = None
    version: int = 1
    status: str = "discovered"


class DedupLedger(ABC):
    """Abstract change-detection + dedup store."""

    @abstractmethod
    async def get(self, document_id: str) -> DocRecord | None: ...

    @abstractmethod
    async def put(self, record: DocRecord) -> None: ...

    @abstractmethod
    async def find_by_checksum(self, checksum: str) -> str | None:
        """Return the document_id that first registered this checksum, if any."""

    @abstractmethod
    async def register_checksum(self, checksum: str, document_id: str) -> None: ...

    @abstractmethod
    async def health(self) -> bool: ...

    async def aclose(self) -> None:  # pragma: no cover - optional override
        return None


class InMemoryDedupLedger(DedupLedger):
    """Process-local ledger for tests/dev. NOT for multi-replica production."""

    def __init__(self) -> None:
        self._docs: dict[str, DocRecord] = {}
        self._checksums: dict[str, str] = {}

    async def get(self, document_id: str) -> DocRecord | None:
        return self._docs.get(document_id)

    async def put(self, record: DocRecord) -> None:
        self._docs[record.document_id] = record

    async def find_by_checksum(self, checksum: str) -> str | None:
        return self._checksums.get(checksum)

    async def register_checksum(self, checksum: str, document_id: str) -> None:
        self._checksums.setdefault(checksum, document_id)

    async def health(self) -> bool:
        return True


class RedisDedupLedger(DedupLedger):
    """Redis-backed ledger shared across the ingestion fleet.

    Keys:
      ``ingest:doc:{document_id}``    -> JSON DocRecord
      ``ingest:chk:{checksum}``       -> document_id  (SETNX for first-writer-wins)
    """

    def __init__(self, url: str, *, key_prefix: str = "ingest") -> None:
        # Imported lazily so the in-memory ledger has no redis dependency.
        from redis.asyncio import Redis

        self._redis = Redis.from_url(url, decode_responses=True)
        self._p = key_prefix

    def _doc_key(self, document_id: str) -> str:
        return f"{self._p}:doc:{document_id}"

    def _chk_key(self, checksum: str) -> str:
        return f"{self._p}:chk:{checksum}"

    async def get(self, document_id: str) -> DocRecord | None:
        raw = await self._redis.get(self._doc_key(document_id))
        return DocRecord(**json.loads(raw)) if raw else None

    async def put(self, record: DocRecord) -> None:
        await self._redis.set(self._doc_key(record.document_id), record.model_dump_json())

    async def find_by_checksum(self, checksum: str) -> str | None:
        value = await self._redis.get(self._chk_key(checksum))
        return None if value is None else str(value)

    async def register_checksum(self, checksum: str, document_id: str) -> None:
        # SETNX => first object with these bytes wins; later ones are duplicates.
        await self._redis.set(self._chk_key(checksum), document_id, nx=True)

    async def health(self) -> bool:
        try:
            return bool(await self._redis.ping())
        except Exception:
            return False

    async def aclose(self) -> None:
        await self._redis.aclose()
