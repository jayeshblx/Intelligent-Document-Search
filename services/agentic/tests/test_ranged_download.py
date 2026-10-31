"""Large snapshot objects are fetched as parallel byte ranges.

A snapshot is dominated by one object -- Qdrant's ``storage.sqlite`` is ~80% of it --
so the existing per-object concurrency does nothing for the part that actually takes
the time. Restoring FOXTROT measured ~3 MB/s against a 446 MB object: one connection's
worth.

The risk in ranged writes is silent corruption: parts landing at the wrong offset
produce a file of exactly the right SIZE that is garbage inside. These assert the
bytes, not the length.

Part size is shrunk per-test so the timing assertions measure fetch concurrency
rather than the disk writing a hundred megabytes.
"""

from __future__ import annotations

import asyncio
import hashlib
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
from rag_agentic import index_store
from rag_agentic.config import AgenticSettings
from rag_agentic.index_store import _RANGED_DOWNLOAD_MIN_BYTES, TenantIndexStore

PART = 1024
# Deterministic and incompressible, spanning several parts with a short final one.
BODY = (hashlib.sha256(b"snapshot").digest() * 400)[: PART * 9 + 377]


@pytest.fixture(autouse=True)
def _small_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(index_store, "_RANGE_PART_BYTES", PART)


class _RangeS3:
    """S3 that serves byte ranges and records how it was asked for them."""

    def __init__(self, body: bytes, *, delay: float = 0.0) -> None:
        self._body = body
        self._delay = delay
        self.ranges: list[tuple[int, int]] = []
        self.peak_in_flight = 0
        self._in_flight = 0

    async def head_object(self, bucket: str, key: str) -> dict[str, Any]:
        return {"ContentLength": len(self._body), "ETag": '"v1"'}

    async def get_range(self, bucket: str, key: str, start: int, end: int) -> bytes:
        self._in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
            self.ranges.append((start, end))
            return self._body[start : end + 1]
        finally:
            self._in_flight -= 1

    async def get_bytes(self, bucket: str, key: str, **_: Any) -> bytes:
        return self._body


def _store(s3: Any) -> TenantIndexStore:
    return TenantIndexStore(AgenticSettings(index_bucket="b"), s3)


async def test_ranged_download_reassembles_the_exact_bytes(tmp_path: Path) -> None:
    s3 = _RangeS3(BODY)
    destination = tmp_path / "storage.sqlite"

    await _store(s3)._download_ranged("b", "k", destination, len(BODY))

    assert destination.read_bytes() == BODY
    assert len(s3.ranges) > 1, "a multi-part object must be fetched in several parts"


async def test_every_byte_is_covered_exactly_once(tmp_path: Path) -> None:
    """Overlapping or missing ranges are how this corrupts a file quietly."""
    s3 = _RangeS3(BODY)

    await _store(s3)._download_ranged("b", "k", tmp_path / "f", len(BODY))

    covered = sorted(s3.ranges)
    assert covered[0][0] == 0
    assert covered[-1][1] == len(BODY) - 1
    for (_, prev_end), (next_start, _) in pairwise(covered):
        assert next_start == prev_end + 1


async def test_a_size_that_is_not_a_multiple_of_the_part_size(tmp_path: Path) -> None:
    """The final part is short; an off-by-one here truncates or over-reads."""
    body = BODY[: PART * 2 + 7]
    s3 = _RangeS3(body)
    destination = tmp_path / "odd"

    await _store(s3)._download_ranged("b", "k", destination, len(body))

    assert destination.read_bytes() == body
    assert destination.stat().st_size == len(body)


async def test_parts_are_fetched_concurrently(tmp_path: Path) -> None:
    """The whole point: parts overlap in time instead of queueing on one connection."""
    s3 = _RangeS3(BODY, delay=0.05)

    started = asyncio.get_running_loop().time()
    await _store(s3)._download_ranged("b", "k", tmp_path / "f", len(BODY))
    elapsed = asyncio.get_running_loop().time() - started

    assert s3.peak_in_flight > 1, "fetches never overlapped"
    # 10 parts at 50ms each: ~0.5s serial, ~0.1s at a concurrency of 8.
    assert elapsed < 0.3, f"fetches appear serial ({elapsed:.2f}s)"


def test_the_threshold_leaves_small_objects_on_a_single_get() -> None:
    """Ranging a small object adds a HEAD and several round trips for nothing."""
    assert _RANGED_DOWNLOAD_MIN_BYTES >= 8 * 1024 * 1024
