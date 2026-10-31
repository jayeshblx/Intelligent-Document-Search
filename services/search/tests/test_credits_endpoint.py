"""Integration tests for the credit-metered endpoints (no AWS — fake LLM).

Drives the real FastAPI app + router + dependency wiring over ASGI, with a
lightweight ``app.state`` (a real ``CreditStore`` + a fake LLM), so nothing here
touches Bedrock, Qdrant, S3 or Mongo.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from httpx import ASGITransport, AsyncClient
from rag_search.answer import LLM
from rag_search.app import create_app
from rag_search.config import SearchSettings
from rag_search.credit_store import CreditStore
from rag_search.dependencies import USER_COOKIE

from rag_core.aws import GenerationResult


class FixedLLM(LLM):
    """Returns a constant reply + fixed token usage, so charges are deterministic."""

    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self._i, self._o = input_tokens, output_tokens

    async def complete(
        self, messages: list[dict[str, Any]], *, system: str, max_tokens: int, temperature: float, cache_system: bool = False
    ) -> GenerationResult:
        return GenerationResult(
            text="ok", input_tokens=self._i, output_tokens=self._o, model="fake-llm"
        )


class BoomLLM(LLM):
    """Simulates an Anthropic/Bedrock error."""

    async def complete(self, *args: Any, **kwargs: Any) -> GenerationResult:
        raise RuntimeError("bedrock unavailable")


def _make_app(tmp_path: Path, llm: LLM, *, plan_total: float = 2000.0) -> tuple[Any, CreditStore]:
    app = create_app()
    app.state.search_settings = SearchSettings(data_dir=str(tmp_path), mongo_uri="")
    store = CreditStore(str(tmp_path / "credits.db"), default_plan_total=Decimal(str(plan_total)))
    app.state.components = SimpleNamespace(credits=store, llm=llm)
    return app, store


def _client(app: Any) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_run_deducts_per_formula_and_returns_spec_shape(tmp_path) -> None:
    app, store = _make_app(tmp_path, FixedLLM(1000, 500), plan_total=2000.0)
    try:
        async with _client(app) as c:
            r = await c.post("/api/credits/run", json={"prompt": "hi"}, headers={"X-User-Id": "u1"})
            assert r.status_code == 200
            d = r.json()
            # i=1000,o=500 -> 7.0 credits (no overhead); balance 2000 - 7 = 1993
            assert d["credits_used"] == 7.0
            assert d["balance"] == 1993.0
            assert d["plan_total"] == 2000.0
            assert d["usage"]["input_tokens"] == 1000
            assert d["usage"]["output_tokens"] == 500
            assert d["reply"] == "ok"
            assert 99.6 <= d["percent_remaining"] <= 99.7

            # Balance persists and is server-authoritative on reload.
            b = await c.get("/api/credits/balance", headers={"X-User-Id": "u1"})
            assert b.json()["balance"] == 1993.0
    finally:
        await store.aclose()


async def test_run_returns_402_when_out_of_credits_without_calling_model(tmp_path) -> None:
    # plan_total 5 < one run's 7.0 credits, so the first run exhausts it.
    app, store = _make_app(tmp_path, FixedLLM(1000, 500), plan_total=5.0)
    try:
        async with _client(app) as c:
            r1 = await c.post("/api/credits/run", json={"prompt": "hi"}, headers={"X-User-Id": "u1"})
            assert r1.status_code == 200
            assert r1.json()["balance"] == 0.0  # clamped, never negative

            r2 = await c.post("/api/credits/run", json={"prompt": "hi"}, headers={"X-User-Id": "u1"})
            assert r2.status_code == 402
            detail = r2.json()["detail"]
            assert detail["error"] == "out_of_credits"
            assert "administrator" in detail["message"].lower()  # ask an admin to reset
    finally:
        await store.aclose()


async def test_reset_restores_pool_to_full(tmp_path) -> None:
    """The admin reset endpoint refills the shared pool to its full starting balance."""
    app, store = _make_app(tmp_path, FixedLLM(1000, 500), plan_total=5.0)
    try:
        async with _client(app) as c:
            await c.post("/api/credits/run", json={"prompt": "hi"}, headers={"X-User-Id": "u1"})
            assert (await c.get("/api/credits/balance", headers={"X-User-Id": "u1"})).json()["balance"] == 0.0

            r = await c.post("/api/credits/reset")
            assert r.status_code == 200
            body = r.json()
            assert body["balance"] == 5.0 and body["plan_total"] == 5.0
            assert body["percent_remaining"] == 100.0

            # Access is restored — the balance endpoint agrees and a run works again.
            assert (await c.get("/api/credits/balance", headers={"X-User-Id": "u1"})).json()["balance"] == 5.0
            assert (await c.post("/api/credits/run", json={"prompt": "hi"},
                                 headers={"X-User-Id": "u1"})).status_code == 200
    finally:
        await store.aclose()


async def test_failed_model_call_deducts_nothing(tmp_path) -> None:
    app, store = _make_app(tmp_path, BoomLLM(), plan_total=2000.0)
    try:
        async with _client(app) as c:
            r = await c.post("/api/credits/run", json={"prompt": "hi"}, headers={"X-User-Id": "u1"})
            assert r.status_code == 502
            b = await c.get("/api/credits/balance", headers={"X-User-Id": "u1"})
            assert b.json()["balance"] == 2000.0  # unchanged
    finally:
        await store.aclose()


async def test_balance_endpoint_seeds_new_user(tmp_path) -> None:
    app, store = _make_app(tmp_path, FixedLLM(1, 1), plan_total=2000.0)
    try:
        async with _client(app) as c:
            b = await c.get("/api/credits/balance", headers={"X-User-Id": "fresh"})
            assert b.status_code == 200
            assert b.json() == {"balance": 2000.0, "plan_total": 2000.0, "percent_remaining": 100.0}
    finally:
        await store.aclose()


async def test_shared_pool_across_users(tmp_path) -> None:
    """Two different users draw down ONE pool — u2 sees the balance u1 reduced."""
    app, store = _make_app(tmp_path, FixedLLM(1000, 500), plan_total=2000.0)
    try:
        async with _client(app) as c:
            await c.post("/api/credits/run", json={"prompt": "hi"}, headers={"X-User-Id": "u1"})
            b1 = await c.get("/api/credits/balance", headers={"X-User-Id": "u1"})
            b2 = await c.get("/api/credits/balance", headers={"X-User-Id": "u2"})
            assert b1.json()["balance"] == 1993.0
            assert b2.json()["balance"] == 1993.0  # shared — reflects u1's spend
    finally:
        await store.aclose()


async def test_cookie_tags_actor_and_pool_is_shared(tmp_path) -> None:
    """No login: a per-browser cookie tags who spent (the actor), but the balance is
    one shared pool — a second browser sees the same reduced balance."""
    app, store = _make_app(tmp_path, FixedLLM(1000, 500), plan_total=2000.0)
    try:
        async with _client(app) as browser1:  # no X-User-Id — actor comes from a cookie
            r1 = await browser1.post("/api/credits/run", json={"prompt": "hi"})
            assert r1.status_code == 200
            assert browser1.cookies.get(USER_COOKIE) is not None  # cookie was minted + set
            assert r1.json()["balance"] == 1993.0
            # Same cookie jar (== same browser, incl. after a refresh) keeps deducting.
            r2 = await browser1.post("/api/credits/run", json={"prompt": "hi"})
            assert r2.json()["balance"] == 1986.0  # 1993 - 7

        async with _client(app) as browser2:  # fresh cookie jar → different actor, SAME pool
            b = await browser2.get("/api/credits/balance")
            assert b.json()["balance"] == 1986.0  # shared — sees the reduced pool (2 runs)
    finally:
        await store.aclose()


async def test_upgrade_refills_to_pro(tmp_path) -> None:
    app, store = _make_app(tmp_path, FixedLLM(1000, 500), plan_total=5.0)
    try:
        async with _client(app) as c:
            await c.post("/api/credits/run", json={"prompt": "hi"}, headers={"X-User-Id": "u1"})
            up = await c.post("/api/credits/upgrade", headers={"X-User-Id": "u1"})
            assert up.status_code == 200
            assert up.json()["plan_total"] == 25.0   # 5 * 5
            assert up.json()["balance"] == 25.0
    finally:
        await store.aclose()
