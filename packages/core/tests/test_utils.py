"""Tests for checksum + deterministic id utilities."""

from __future__ import annotations

import hashlib

from rag_core.utils import chunk_id, document_id, sha256_astream, sha256_bytes, sha256_stream


def test_sha256_bytes_matches_hashlib() -> None:
    data = b"compliance"
    assert sha256_bytes(data) == hashlib.sha256(data).hexdigest()


def test_sha256_stream_equals_whole() -> None:
    data = b"a" * 5000
    chunks = [data[i : i + 1000] for i in range(0, len(data), 1000)]
    assert sha256_stream(chunks) == sha256_bytes(data)


async def test_sha256_astream_equals_whole() -> None:
    data = b"streamed-bytes" * 100

    async def gen():
        for i in range(0, len(data), 17):
            yield data[i : i + 17]

    assert await sha256_astream(gen()) == sha256_bytes(data)


def test_document_id_is_deterministic() -> None:
    a = document_id("bucket", "key.pdf")
    b = document_id("bucket", "key.pdf")
    assert a == b


def test_document_id_changes_with_version() -> None:
    base = document_id("bucket", "key.pdf")
    v2 = document_id("bucket", "key.pdf", version_id="etag-2")
    assert base != v2


def test_chunk_id_is_deterministic_and_unique_per_index() -> None:
    c0 = chunk_id("doc-1", "child", 0)
    c0_again = chunk_id("doc-1", "child", 0)
    c1 = chunk_id("doc-1", "child", 1)
    assert c0 == c0_again
    assert c0 != c1
