"""Tests for the MongoDB query audit log (offline — Mongo is never contacted).

Uses ``MongoQueryLog.capture()``, the same fake-injection convention as
``MongoMetadataSource.seed()``: the network path is short-circuited, so the dummy URI
stays inert.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from httpx import ASGITransport, AsyncClient
from rag_search.answer import LLM
from rag_search.app import create_app
from rag_search.config import SearchSettings
from rag_search.credit_store import CreditStore
from rag_search.models import Citation, UsageInfo
from rag_search.query_log import MongoQueryLog

from rag_core.aws import GenerationResult


def _log(**kw: Any) -> MongoQueryLog:
    log = MongoQueryLog("mongodb://inert", database="db", collection="coll", **kw)
    log.capture()
    return log


def _usage() -> UsageInfo:
    return UsageInfo(
        input_tokens=1000,
        output_tokens=500,
        embed_tokens=42,
        total_tokens=1542,
        input_cost_usd=0.002,
        output_cost_usd=0.005,
        embed_cost_usd=0.000001,
        total_cost_usd=0.007001,
        model="claude-sonnet-4-5",
    )


# ---- document shape --------------------------------------------------------------
def test_document_captures_everything_asked_for() -> None:
    log = _log()
    doc = log.build_document(
        question="which certificates expire in 2026?",
        answer="Three certificates expire in 2026 [1].",
        sources=log.sources_from_citations(
            [
                Citation(
                    index=1,
                    document_id="d1",
                    source_key="foxtrot/cert.pdf",
                    snippet="expires 2026-03-01",
                    page_start=2,
                    page_end=2,
                    score=0.87,
                    matched_by="both",
                )
            ]
        ),
        usage=_usage(),
        credit=SimpleNamespace(
            base_cost_usd=0.007, charged_usd=0.0105, credits_used=10.5, balance=989.5
        ),
        session_id="s1",
        actor="u1",
        took_ms=1234.56,
    )

    # question / answer
    assert doc["Question"] == "which certificates expire in 2026?"
    assert doc["Answer"] == "Three certificates expire in 2026 [1]."
    # tokens, incl. embedding
    assert doc["InputTokens"] == 1000
    assert doc["OutputTokens"] == 500
    assert doc["EmbedTokens"] == 42
    assert doc["TotalTokens"] == 1542
    # cost
    assert doc["TotalCostUsd"] == 0.007001
    assert doc["BaseCostUsd"] == 0.007
    assert doc["ChargedUsd"] == 0.0105
    assert doc["CreditsUsed"] == 10.5
    assert doc["BalanceAfter"] == 989.5
    # sources
    assert doc["SourceCount"] == 1
    src = doc["Sources"][0]
    assert src["SourceKey"] == "foxtrot/cert.pdf"
    assert src["DocumentId"] == "d1"
    assert src["PageStart"] == 2
    assert src["Score"] == 0.87
    assert src["MatchedBy"] == "both"
    # context
    assert doc["Model"] == "claude-sonnet-4-5"
    assert doc["SessionId"] == "s1"
    assert doc["Actor"] == "u1"
    assert doc["TookMs"] == 1234.6
    assert doc["DateCreated"].tzinfo is not None  # tz-aware UTC, not naive


def test_document_survives_a_zero_usage_answer_with_no_credit() -> None:
    """The 'couldn't find anything' path never calls the model: no credit, no sources."""
    log = _log()
    doc = log.build_document(
        question="q", answer="I could not find that.", sources=[], usage=UsageInfo(), credit=None
    )
    assert doc["InputTokens"] == 0
    assert doc["TotalCostUsd"] == 0.0
    assert doc["SourceCount"] == 0
    assert "CreditsUsed" not in doc  # nothing was charged, so nothing is claimed


def test_long_answer_is_truncated_and_flagged() -> None:
    """A runaway answer must not push the document toward Mongo's 16MB ceiling."""
    log = _log(max_answer_chars=500)
    doc = log.build_document(
        question="q", answer="x" * 5000, sources=[], usage=UsageInfo(), credit=None
    )
    assert len(doc["Answer"]) == 500
    assert doc["AnswerTruncated"] is True


def test_short_answer_is_not_flagged_as_truncated() -> None:
    log = _log(max_answer_chars=500)
    doc = log.build_document(question="q", answer="fine", sources=[], usage=UsageInfo(), credit=None)
    assert doc["AnswerTruncated"] is False


def test_snippets_are_capped() -> None:
    log = _log()
    sources = log.sources_from_citations(
        [Citation(index=1, document_id="d", source_key="k", snippet="y" * 5000)]
    )
    assert len(sources[0]["Snippet"]) == 400


# ---- safety: logging must never break answering ----------------------------------
async def test_record_never_raises_when_mongo_is_unreachable() -> None:
    """The whole point: a dead Mongo must not surface to the caller."""
    log = MongoQueryLog("mongodb://127.0.0.1:1", database="db", collection="c")
    doc = log.build_document(question="q", answer="a", sources=[], usage=UsageInfo())
    log.record(doc)  # dispatches to the background; must not raise here
    await log.aclose()  # swallows the connection failure while flushing


async def test_record_returns_immediately_without_awaiting_the_insert() -> None:
    """record() must not block the response on a slow Mongo."""
    log = MongoQueryLog("mongodb://127.0.0.1:1", database="db", collection="c")
    slow_started = asyncio.Event()

    def _slow(_doc: dict[str, Any]) -> None:
        slow_started.set()

    log._insert_sync = _slow  # type: ignore[method-assign]
    doc = log.build_document(question="q", answer="a", sources=[], usage=UsageInfo())
    log.record(doc)
    assert not slow_started.is_set()  # nothing ran synchronously
    await log.aclose()
    assert slow_started.is_set()  # ...but the flush completed it


# ---- wiring: the chat endpoint actually logs -------------------------------------
class FixedLLM(LLM):
    def __init__(self, i: int, o: int) -> None:
        self._i, self._o = i, o

    async def complete(
        self, messages: list[dict[str, Any]], *, system: str, max_tokens: int, temperature: float, cache_system: bool = False
    ) -> GenerationResult:
        return GenerationResult(text="ok", input_tokens=self._i, output_tokens=self._o, model="m")


class StubAnswerer:
    """Stands in for AnswerService so the test targets the logging hook, not retrieval."""

    def __init__(self, resp: Any) -> None:
        self._resp = resp

    async def answer(self, message: str, **kw: Any) -> Any:
        return self._resp


def _chat_app(tmp_path: Path, resp: Any, query_log: Any) -> tuple[Any, CreditStore]:
    from rag_search.models import ChatResponse

    assert isinstance(resp, ChatResponse)
    app = create_app()
    app.state.search_settings = SearchSettings(data_dir=str(tmp_path), mongo_uri="")
    store = CreditStore(str(tmp_path / "credits.db"), default_plan_total=Decimal("1000"))
    app.state.components = SimpleNamespace(
        credits=store,
        llm=FixedLLM(10, 5),
        answer=StubAnswerer(resp),
        conversations=SimpleNamespace(new_session=lambda: "s-new"),
        query_log=query_log,
    )
    return app, store


def _chat_response() -> Any:
    from rag_search.models import ChatResponse

    return ChatResponse(
        session_id="s1",
        answer="Grounded answer [1].",
        citations=[Citation(index=1, document_id="d1", source_key="foxtrot/a.pdf", snippet="snip")],
        usage=UsageInfo(
            input_tokens=1000, output_tokens=500, embed_tokens=7, total_tokens=1507, model="m"
        ),
        took_ms=12.0,
    )


async def test_chat_writes_one_log_document_per_answer(tmp_path: Path) -> None:
    log = _log()
    app, store = _chat_app(tmp_path, _chat_response(), log)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.post("/v1/chat", json={"message": "hi"}, headers={"X-User-Id": "u1"})
            assert r.status_code == 200
    finally:
        await store.aclose()

    assert len(log.captured) == 1
    doc = log.captured[0]
    assert doc["Question"] == "hi"
    assert doc["Answer"] == "Grounded answer [1]."
    assert doc["Sources"][0]["SourceKey"] == "foxtrot/a.pdf"
    assert doc["InputTokens"] == 1000
    assert doc["EmbedTokens"] == 7
    assert doc["Actor"] == "u1"
    # The credit that was actually settled on this turn is recorded, not an estimate.
    assert doc["CreditsUsed"] > 0
    assert doc["BalanceAfter"] == 1000 - doc["CreditsUsed"]


async def test_chat_still_answers_when_the_log_write_blows_up(tmp_path: Path) -> None:
    """A broken audit log must never turn a good answer into a 500."""

    class Exploding:
        def build_document(self, **kw: Any) -> dict[str, Any]:
            raise RuntimeError("mongo exploded")

        def sources_from_citations(self, c: list[Any]) -> list[Any]:
            return []

        def record(self, doc: dict[str, Any]) -> None:
            raise RuntimeError("never reached")

    app, store = _chat_app(tmp_path, _chat_response(), Exploding())
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.post("/v1/chat", json={"message": "hi"}, headers={"X-User-Id": "u1"})
            assert r.status_code == 200
            assert r.json()["answer"] == "Grounded answer [1]."
    finally:
        await store.aclose()


async def test_chat_works_when_logging_is_disabled(tmp_path: Path) -> None:
    """query_log=None (SEARCH_QUERY_LOG_URI unset) is the default and must be inert."""
    app, store = _chat_app(tmp_path, _chat_response(), None)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.post("/v1/chat", json={"message": "hi"}, headers={"X-User-Id": "u1"})
            assert r.status_code == 200
    finally:
        await store.aclose()
