"""Phase 1 starts paging the search index in, without waiting for it.

The object-code lookup touches metadata only, so the vector index is still cold
when the real question arrives seconds later. `prewarm` uses that gap. What
matters is that it never slows phase 1 down and never turns a probe failure
into a failed request -- the probe is an optimisation, and an optimisation that
can break the request is worse than none.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from rag_agentic.config import AgenticSettings
from rag_agentic.index_store import TenantIndexStore
from rag_agentic.tenancy import TenantContext


class FakeVector:
    def __init__(self, on_search: Any = None) -> None:
        self.searches: list[int] = []
        self._on_search = on_search

    async def search(self, vector: list[float], *, limit: int = 10) -> list[Any]:
        if self._on_search is not None:
            await self._on_search()
        self.searches.append(len(vector))
        return []


class FakeKeyword:
    def __init__(self) -> None:
        self.searches: list[str] = []

    async def search(self, text: str, *, limit: int = 10) -> list[Any]:
        self.searches.append(text)
        return []


class FakeIndex:
    def __init__(self, on_search: Any = None) -> None:
        self.vector = FakeVector(on_search)
        self.keyword = FakeKeyword()

    async def aclose(self) -> None:
        return None


def _store(index: FakeIndex, *, mongo: bool = True) -> TenantIndexStore:
    settings = AgenticSettings(
        mongo_uri="mongodb://localhost:27017" if mongo else "",
        mongo_collections={"acmedev/bravo": "BRAVO_DocumentSearch_DEV"} if mongo else {},
        embed_dim=8,
    )
    store = TenantIndexStore(settings, s3=object())  # type: ignore[arg-type]
    store.acquire = _returning(index)  # type: ignore[method-assign]
    return store


def _returning(index: FakeIndex) -> Any:
    async def _acquire(context: TenantContext) -> FakeIndex:
        return index

    return _acquire


def _context(tenant: str = "bravo") -> TenantContext:
    return TenantContext(environment="acmedev", tenant=tenant, base_url="")


async def _settle() -> None:
    """Let the background task run to completion."""
    for _ in range(50):
        await asyncio.sleep(0)


async def test_prewarm_probes_both_indexes() -> None:
    index = FakeIndex()
    store = _store(index)

    store.prewarm(_context())
    await _settle()

    assert index.vector.searches == [8], "one probe, at the configured embedding width"
    assert index.keyword.searches, "the keyword index is paged in too"


async def test_prewarm_returns_before_the_probe_finishes() -> None:
    """Phase 1 answers in milliseconds; it must not wait on Atlas."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def _block() -> None:
        started.set()
        await release.wait()

    index = FakeIndex(on_search=_block)
    store = _store(index)

    store.prewarm(_context())  # returns immediately even though the probe blocks
    await asyncio.wait_for(started.wait(), timeout=1)
    assert index.vector.searches == [], "the probe is still in flight"

    release.set()
    await _settle()
    assert index.vector.searches == [8]


async def test_a_failing_probe_is_swallowed() -> None:
    async def _boom() -> None:
        raise RuntimeError("atlas unreachable")

    store = _store(FakeIndex(on_search=_boom))

    store.prewarm(_context())
    await _settle()  # must not raise, and must not leave the task pending


async def test_a_second_prewarm_does_not_queue_behind_the_first() -> None:
    release = asyncio.Event()

    async def _block() -> None:
        await release.wait()

    index = FakeIndex(on_search=_block)
    store = _store(index)

    store.prewarm(_context())
    store.prewarm(_context())
    store.prewarm(_context())
    release.set()
    await _settle()

    assert index.vector.searches == [8], "three calls, one probe"


async def test_local_snapshots_are_not_probed() -> None:
    """A file or Postgres snapshot is already local; there is nothing to page in."""
    index = FakeIndex()
    store = _store(index, mongo=False)

    store.prewarm(_context())
    await _settle()

    assert index.vector.searches == []


async def test_closing_cancels_a_probe_in_flight() -> None:
    release = asyncio.Event()

    async def _block() -> None:
        await release.wait()

    store = _store(FakeIndex(on_search=_block))
    store.prewarm(_context())
    await _settle()

    await store.aclose()  # must not hang waiting for the probe


@pytest.mark.parametrize("tenant", ["bravo", "foxtrot"])
async def test_each_tenant_warms_independently(tenant: str) -> None:
    index = FakeIndex()
    settings = AgenticSettings(
        mongo_uri="mongodb://localhost:27017",
        mongo_collections={"acmedev/bravo": "BRAVO", "acmedev/foxtrot": "FOXTROT"},
        embed_dim=8,
    )
    store = TenantIndexStore(settings, s3=object())  # type: ignore[arg-type]
    store.acquire = _returning(index)  # type: ignore[method-assign]

    store.prewarm(_context(tenant))
    await _settle()

    assert index.vector.searches == [8]
