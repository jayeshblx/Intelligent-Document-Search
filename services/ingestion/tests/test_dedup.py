"""Tests for the in-memory dedup ledger."""

from __future__ import annotations

import pytest

from rag_ingestion.dedup import DocRecord, InMemoryDedupLedger


async def test_get_put_roundtrip() -> None:
    ledger = InMemoryDedupLedger()
    assert await ledger.get("doc-1") is None

    rec = DocRecord(document_id="doc-1", physical_id="phys-1", etag="e1", checksum="c1", version=1)
    await ledger.put(rec)

    fetched = await ledger.get("doc-1")
    assert fetched is not None
    assert fetched.etag == "e1"
    assert fetched.physical_id == "phys-1"


async def test_checksum_first_writer_wins() -> None:
    ledger = InMemoryDedupLedger()
    await ledger.register_checksum("chk-1", "doc-1")
    # A second registration for the same bytes must not overwrite the owner.
    await ledger.register_checksum("chk-1", "doc-2")
    assert await ledger.find_by_checksum("chk-1") == "doc-1"


async def test_health_is_true() -> None:
    assert await InMemoryDedupLedger().health() is True


@pytest.mark.parametrize("missing", ["", "nope"])
async def test_find_by_checksum_missing(missing: str) -> None:
    assert await InMemoryDedupLedger().find_by_checksum(missing) is None
