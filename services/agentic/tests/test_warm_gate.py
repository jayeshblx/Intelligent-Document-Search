"""A question waits for a cold index rather than searching half of it.

A cold `$vectorSearch` on a large Atlas collection can exceed its own time
limit. The search then falls back to keyword-only and the request still
succeeds, so nobody sees that the semantic half never ran. Waiting costs time
the request was going to spend inside that timing-out search anyway, and ends
with a correct answer instead of a plausible one.

The properties that matter: concurrent callers share one probe, a warm tenant
is not waited on at all, and nothing here can make a request fail -- a probe
that breaks or overruns leaves the request exactly where it would have been.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from rag_agentic.config import AgenticSettings
from rag_agentic.index_store import TenantIndexStore
from rag_agentic.tenancy import TenantContext


class FakeVector:
    def __init__(self, gate: asyncio.Event | None = None, fail: bool = False) -> None:
        self.calls = 0
        self._gate = gate
        self._fail = fail

    async def search(self, vector: list[float], *, limit: int = 10) -> list[Any]:
        if self._gate is not None:
            await self._gate.wait()
        if self._fail:
            raise RuntimeError("atlas unreachable")
        self.calls += 1
        return []


class FakeKeyword:
    async def search(self, text: str, *, limit: int = 10) -> list[Any]:
        return []


class FakeIndex:
    def __init__(self, gate: asyncio.Event | None = None, fail: bool = False) -> None:
        self.vector = FakeVector(gate, fail)
        self.keyword = FakeKeyword()

    async def aclose(self) -> None:
        return None


def _store(index: FakeIndex, **over: Any) -> TenantIndexStore:
    settings = AgenticSettings(
        mongo_uri="mongodb://localhost:27017",
        mongo_collections={"acmedev/bravo": "BRAVO_DocumentSearch_DEV"},
        embed_dim=8,
        **over,
    )
    store = TenantIndexStore(settings, s3=object())  # type: ignore[arg-type]

    async def _acquire(context: TenantContext) -> FakeIndex:
        return index

    store.acquire = _acquire  # type: ignore[method-assign]
    return store


def _context(tenant: str = "bravo") -> TenantContext:
    return TenantContext(environment="acmedev", tenant=tenant, base_url="")


async def test_a_cold_tenant_is_warmed_before_the_question_runs() -> None:
    index = FakeIndex()
    store = _store(index)

    assert not store.is_warm(_context())
    await store.ensure_warm(_context())

    assert index.vector.calls == 1
    assert store.is_warm(_context()), "the next question must not wait again"


async def test_a_warm_tenant_is_not_waited_on() -> None:
    index = FakeIndex()
    store = _store(index)

    await store.ensure_warm(_context())
    await store.ensure_warm(_context())
    await store.ensure_warm(_context())

    assert index.vector.calls == 1, "one probe covers the whole warm window"


async def test_concurrent_questions_share_one_probe() -> None:
    """Several cold requests must not each thrash the cluster."""
    gate = asyncio.Event()
    index = FakeIndex(gate)
    store = _store(index)

    waiters = [asyncio.create_task(store.ensure_warm(_context())) for _ in range(5)]
    await asyncio.sleep(0)
    gate.set()
    await asyncio.gather(*waiters)

    assert index.vector.calls == 1, "five callers, one probe"


async def test_a_slow_probe_does_not_hold_the_request_forever() -> None:
    gate = asyncio.Event()  # never set
    index = FakeIndex(gate)
    store = _store(index, mongo_warm_wait_s=0.05)

    await asyncio.wait_for(store.ensure_warm(_context()), timeout=2)

    assert not store.is_warm(_context()), "it timed out; the index is still cold"
    gate.set()
    await store.aclose()


async def test_a_failing_probe_lets_the_request_through() -> None:
    store = _store(FakeIndex(fail=True))

    await store.ensure_warm(_context())  # must not raise

    assert not store.is_warm(_context())


async def test_the_wait_can_be_switched_off() -> None:
    index = FakeIndex()
    store = _store(index, mongo_warm_wait_s=0)

    await store.ensure_warm(_context())

    assert index.vector.calls == 0, "disabled means the old behaviour, exactly"


async def test_a_local_snapshot_is_never_waited_on() -> None:
    """File and Postgres snapshots are already local; there is nothing to page in."""
    index = FakeIndex()
    settings = AgenticSettings(embed_dim=8)
    store = TenantIndexStore(settings, s3=object())  # type: ignore[arg-type]

    async def _acquire(context: TenantContext) -> FakeIndex:
        return index

    store.acquire = _acquire  # type: ignore[method-assign]
    await store.ensure_warm(_context())

    assert index.vector.calls == 0


async def test_an_evicted_tenant_counts_as_cold_again() -> None:
    index = FakeIndex()
    store = _store(index)

    await store.ensure_warm(_context())
    assert store.is_warm(_context())

    await store._evict(_context().key)

    assert not store.is_warm(_context())
    await store.ensure_warm(_context())
    assert index.vector.calls == 2, "a reopened index is paged in again"


@pytest.mark.parametrize("keep_warm", [60.0, 240.0])
async def test_the_warm_window_outlasts_the_keepalive_interval(keep_warm: float) -> None:
    """A probe that runs slightly late must not leave every request warming."""
    store = _store(FakeIndex(), mongo_keep_warm_s=keep_warm)

    assert store._warm_ttl() > keep_warm
