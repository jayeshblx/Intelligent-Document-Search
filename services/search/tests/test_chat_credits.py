"""Integration tests for /v1/chat credit metering (no AWS — fake LLM).

Tests credit-metering behavior in the chat endpoint, complementing test_credits_endpoint.py
which tests /api/credits/run. Both endpoints share the same metering logic.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from httpx import ASGITransport, AsyncClient
from rag_search.answer import LLM, AnswerService
from rag_search.app import create_app
from rag_search.config import SearchSettings
from rag_search.conversation import ConversationStore
from rag_search.credit_store import CreditStore
from rag_search.metadata_store import MetadataStore

from rag_core.aws import GenerationResult
from rag_core.models.enums import Language
from rag_core.models.metadata import DocumentMetadata


class FixedLLM(LLM):
    """Returns a constant reply + fixed token usage, so charges are deterministic."""

    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self._i, self._o = input_tokens, output_tokens

    async def complete(
        self, messages: list[dict[str, Any]], *, system: str, max_tokens: int, temperature: float, cache_system: bool = False
    ) -> GenerationResult:
        return GenerationResult(
            text="The policy requires AES-256 encryption [1]",
            input_tokens=self._i,
            output_tokens=self._o,
            model="fake-llm",
        )


class BoomLLM(LLM):
    """Simulates an Anthropic/Bedrock error."""

    async def complete(self, *args: Any, **kwargs: Any) -> GenerationResult:
        raise RuntimeError("bedrock unavailable")


def _make_app(tmp_path: Path, llm: LLM, *, plan_total: float = 2000.0) -> tuple[Any, CreditStore]:
    """Build minimal app with credits + chat dependencies."""
    app = create_app()
    app.state.search_settings = SearchSettings(data_dir=str(tmp_path), mongo_uri="")
    
    credit_store = CreditStore(
        str(tmp_path / "credits.db"),
        default_plan_total=Decimal(str(plan_total)),
    )
    metadata_store = MetadataStore(str(tmp_path / "meta.db"))
    conversations = ConversationStore(ttl_minutes=30)
    search = _FakeSearch()

    # Minimal components for chat endpoint. ``get_answer`` reads
    # ``components.answer`` straight off app.state — there is no DI construction —
    # so the AnswerService must be built here, as the real factory does.
    app.state.components = SimpleNamespace(
        credits=credit_store,
        llm=llm,
        conversations=conversations,
        metadata=metadata_store,
        search=search,
        answer=AnswerService(
            search=search,  # type: ignore[arg-type]
            llm=llm,
            conversations=conversations,
            metadata=metadata_store,
        ),
    )
    return app, credit_store


class _FakeSearch:
    """Minimal search service that returns one passage."""
    
    async def search(self, query: str, *, namespace: str = "default", top_k: int = 6, filters: Any = None):
        from rag_search.models import Passage, SearchResponse
        
        passage = Passage(
            chunk_id="c1",
            document_id="d1",
            source_key="foxtrot/security_policy.docx",
            score=1.0,
            text="Confidential data must be encrypted at rest using AES-256 encryption.",
        )
        return SearchResponse(query=query, took_ms=1.0, results=[passage])
    
    async def expand_document(self, document_id: str, text: str, *, cap: int) -> str:
        return text


def _client(app: Any) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_chat_deducts_credits_on_success(tmp_path) -> None:
    """Chat endpoint deducts credits using the same formula as /api/credits/run."""
    app, store = _make_app(tmp_path, FixedLLM(1000, 500), plan_total=2000.0)
    try:
        async with _client(app) as c:
            r = await c.post(
                "/v1/chat",
                json={"message": "how is confidential data encrypted", "namespace": "default"},
                headers={"X-User-Id": "u1"},
            )
            assert r.status_code == 200
            d = r.json()

            # Same formula as /api/credits/run (credit_config.compute_charge):
            #   base_cost    = 1000/1e6*$2 + 500/1e6*$10 = $0.007  (+ a negligible
            #                  fraction of a cent for the query's embed tokens)
            #   charged      = base_cost (overhead disabled)     = $0.007
            #   credits_used = charged / $0.001 = 7.0 credits     (1000 credits = $1)
            assert d["credit"]["credits_used"] == 7.0
            assert d["credit"]["balance"] == 1993.0
            assert d["credit"]["plan_total"] == 2000.0
            # Token usage is a sibling of ``credit`` on ChatResponse, not nested inside it.
            assert d["usage"]["input_tokens"] == 1000
            assert d["usage"]["output_tokens"] == 500
            assert 99.6 <= d["credit"]["percent_remaining"] <= 99.7

            # Verify balance persists
            b = await c.get("/api/credits/balance", headers={"X-User-Id": "u1"})
            assert b.json()["balance"] == 1993.0
    finally:
        await store.aclose()


async def test_chat_returns_402_when_out_of_credits(tmp_path) -> None:
    """Chat refuses with HTTP 402 when account exhausted (before calling LLM)."""
    app, store = _make_app(tmp_path, FixedLLM(1000, 500), plan_total=5.0)
    try:
        async with _client(app) as c:
            # First call exhausts the pool (plan_total 5.0 < one turn's 7.0 credits)
            r1 = await c.post(
                "/v1/chat",
                json={"message": "test", "namespace": "default"},
                headers={"X-User-Id": "u1"},
            )
            assert r1.status_code == 200
            assert r1.json()["credit"]["balance"] == 0.0  # clamped, never negative

            # Second call should be rejected with 402
            r2 = await c.post(
                "/v1/chat",
                json={"message": "test", "namespace": "default"},
                headers={"X-User-Id": "u1"},
            )
            assert r2.status_code == 402
            # The 402 payload (_out_of_credits) carries the standing + an admin-reset
            # instruction; there is no self-serve upgrade offer in it.
            detail = r2.json()["detail"]
            assert detail["error"] == "out_of_credits"
            assert "administrator" in detail["message"].lower()
            assert detail["balance"] == 0.0
            assert detail["plan_total"] == 5.0
            assert detail["percent_remaining"] == 0.0
    finally:
        await store.aclose()


async def test_chat_failed_model_deducts_nothing(tmp_path) -> None:
    """When LLM fails, chat returns 502 and deducts no credits.

    A model failure must not be dressed up as a 200 with a fallback answer — the
    caller cannot tell that apart from a real, grounded answer."""
    app, store = _make_app(tmp_path, BoomLLM(), plan_total=2000.0)
    try:
        async with _client(app) as c:
            r = await c.post(
                "/v1/chat",
                json={"message": "test", "namespace": "default"},
                headers={"X-User-Id": "u1"},
            )
            assert r.status_code == 502


            # Balance should be unchanged
            b = await c.get("/api/credits/balance", headers={"X-User-Id": "u1"})
            assert b.json()["balance"] == 2000.0
    finally:
        await store.aclose()


async def test_chat_credit_info_shape_matches_spec(tmp_path) -> None:
    """Verify ChatResponse.credit field matches CreditInfo spec."""
    app, store = _make_app(tmp_path, FixedLLM(100, 50), plan_total=2000.0)
    try:
        async with _client(app) as c:
            r = await c.post(
                "/v1/chat",
                json={"message": "test", "namespace": "default"},
                headers={"X-User-Id": "u1"},
            )
            assert r.status_code == 200
            body = r.json()
            credit = body["credit"]

            # Required CreditInfo fields — the credit accounting only.
            assert "credits_used" in credit
            assert "base_cost_usd" in credit
            assert "charged_usd" in credit
            assert "balance" in credit
            assert "plan_total" in credit
            assert "percent_remaining" in credit

            # Token usage is NOT nested in CreditInfo — it's a sibling field on
            # ChatResponse (ChatResponse.usage: UsageInfo).
            assert "usage" not in credit
            usage = body["usage"]
            assert "input_tokens" in usage
            assert "output_tokens" in usage
            assert "total_tokens" in usage
            assert usage["total_tokens"] == 100 + 50
    finally:
        await store.aclose()


async def test_chat_shared_pool_with_run_endpoint(tmp_path) -> None:
    """Chat and /api/credits/run draw from the same shared pool."""
    app, store = _make_app(tmp_path, FixedLLM(1000, 500), plan_total=2000.0)
    try:
        async with _client(app) as c:
            # Use /api/credits/run first (deducts 7.0)
            r1 = await c.post(
                "/api/credits/run",
                json={"prompt": "hi"},
                headers={"X-User-Id": "u1"},
            )
            assert r1.status_code == 200
            assert r1.json()["balance"] == 1993.0

            # Chat sees the reduced balance
            r2 = await c.post(
                "/v1/chat",
                json={"message": "test", "namespace": "default"},
                headers={"X-User-Id": "u1"},
            )
            assert r2.status_code == 200
            # The same 1000/500 usage costs 7.0 credits here too (plus a negligible
            # embed charge for the query, which rounds away): 1993 - 7 = 1986.0
            assert r2.json()["credit"]["balance"] == 1986.0
    finally:
        await store.aclose()


async def test_chat_with_conversation_context(tmp_path) -> None:
    """Chat maintains conversation history and credits accumulate across turns."""
    app, store = _make_app(tmp_path, FixedLLM(100, 50), plan_total=2000.0)
    try:
        async with _client(app) as c:
            # First turn creates session
            r1 = await c.post(
                "/v1/chat",
                json={"message": "what is encryption", "namespace": "default"},
                headers={"X-User-Id": "u1"},
            )
            assert r1.status_code == 200
            session_id = r1.json()["session_id"]
            balance1 = r1.json()["credit"]["balance"]
            
            # Second turn uses same session
            r2 = await c.post(
                "/v1/chat",
                json={
                    "message": "tell me more",
                    "namespace": "default",
                    "session_id": session_id,
                },
                headers={"X-User-Id": "u1"},
            )
            assert r2.status_code == 200
            assert r2.json()["session_id"] == session_id
            balance2 = r2.json()["credit"]["balance"]
            
            # Credits continue to deduct
            assert balance2 < balance1
    finally:
        await store.aclose()


async def test_chat_with_metadata_filters(tmp_path) -> None:
    """Chat endpoint respects facet filters and still meters credits."""
    app, store = _make_app(tmp_path, FixedLLM(100, 50), plan_total=2000.0)
    
    # Seed metadata for facet detection
    metadata_store = app.state.components.metadata
    try:
        await metadata_store.put(
            DocumentMetadata(document_id="d1", language=Language.EN),
            namespace="default",
            source_key="foxtrot/security_policy.docx",
            facets={"status": "Active"},
        )
        
        async with _client(app) as c:
            r = await c.post(
                "/v1/chat",
                json={
                    "message": "which active documents mention encryption",
                    "namespace": "default",
                },
                headers={"X-User-Id": "u1"},
            )
            assert r.status_code == 200
            # Credits still deducted even with filters
            assert r.json()["credit"]["balance"] < 2000.0
    finally:
        await store.aclose()
        await metadata_store.aclose()
