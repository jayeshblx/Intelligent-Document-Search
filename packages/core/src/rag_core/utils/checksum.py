"""Content checksums for change detection and deduplication.

sha256 of the raw bytes is the canonical fingerprint: if two S3 objects share a
checksum they are byte-identical (skip re-processing); if an object's checksum
changes, its content changed (trigger re-indexing as a new version). We stream
rather than read-whole-file so multi-hundred-MB technical manuals don't blow the
worker's memory budget.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Iterable
from pathlib import Path

_CHUNK_BYTES = 1024 * 1024  # 1 MiB streaming window


def sha256_bytes(data: bytes) -> str:
    """sha256 hex digest of an in-memory buffer."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Streaming sha256 of a file on disk."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_stream(chunks: Iterable[bytes]) -> str:
    """sha256 over a synchronous iterable of byte chunks (e.g. an S3 body)."""
    digest = hashlib.sha256()
    for block in chunks:
        digest.update(block)
    return digest.hexdigest()


async def sha256_astream(chunks: AsyncIterator[bytes]) -> str:
    """sha256 over an async byte stream (e.g. aioboto3 StreamingBody)."""
    digest = hashlib.sha256()
    async for block in chunks:
        digest.update(block)
    return digest.hexdigest()
