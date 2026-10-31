"""FastAPI app for local hybrid search + the search UI.

Serves a single-page UI at ``/`` and a small API:
    POST /v1/index    build the local index from S3 (foxtrot/)
    GET  /v1/search   hybrid search -> ranked passages
    GET  /v1/stats    index sizes
Plus /healthz /readyz /metrics.

All storage is local/free (embedded Qdrant + SQLite FTS5); embeddings use Bedrock
Titan (configurable to a local fake for offline demos via SEARCH_EMBED_PROVIDER).
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from rag_core.config import get_settings
from rag_core.logging import configure_logging, get_logger
from rag_core.models.base import utcnow
from rag_core.observability import REGISTRY, setup_tracing
from rag_search.agent_client import (
    AgentRuntimeClient,
    AgentUnavailableError,
    new_agent_session,
)
from rag_search.answer import LLM, AnswerService, AnswerUnavailableError
from rag_search.config import SearchSettings, StoreBackend
from rag_search.conversation import ConversationStore
from rag_search.credit_config import ChargeBreakdown
from rag_search.credit_store import Account, CreditStore
from rag_search.dependencies import (
    get_answer,
    get_conversations,
    get_credits,
    get_current_user,
    get_feedback,
    get_indexer,
    get_llm,
    get_query_log,
    get_search,
)
from rag_search.downloads import DownloadNotFound, DownloadTokenError
from rag_search.factory import build_components
from rag_search.feedback import FeedbackStore
from rag_search.indexer import IndexingService
from rag_search.models import (
    AgentChatRequest,
    AgentChatResponse,
    AgentCitation,
    AgentStatus,
    AgentTenantChoice,
    ChatRequest,
    ChatResponse,
    CreditBalance,
    CreditInfo,
    CreditRunRequest,
    CreditRunResponse,
    DocumentListItem,
    DocumentListResponse,
    FeedbackRequest,
    IndexRequest,
    IndexSummary,
    NewSessionResponse,
    SearchResponse,
    TopUpRequest,
    UsageInfo,
)
from rag_search.retrieval import SearchService

_log = get_logger("rag_search.app")
SERVICE_NAME = "search"
_UI_PATH = Path(__file__).resolve().parent / "web" / "index.html"
_ADMIN_PATH = Path(__file__).resolve().parent / "web" / "admin.html"
_FAVICON_PATH = Path(__file__).resolve().parent / "web" / "favicon.svg"

# System prompt for the raw metered run endpoint (/api/credits/run).
_RUN_SYSTEM = "You are a helpful assistant. Answer the user's prompt concisely and accurately."


def _credit_info(account: Account, breakdown: ChargeBreakdown | None = None) -> CreditInfo:
    """Build the client-facing credit readout — full precision in the ledger,
    rounded here only for display (credits/balance to 2 dp, USD to 6 dp)."""
    return CreditInfo(
        credits_used=float(round(breakdown.credits_used, 2)) if breakdown else 0.0,
        base_cost_usd=float(round(breakdown.base_cost, 6)) if breakdown else 0.0,
        charged_usd=float(round(breakdown.charged, 6)) if breakdown else 0.0,
        balance=float(round(account.balance, 2)),
        plan_total=float(round(account.plan_total, 2)),
        percent_remaining=account.percent_remaining,
    )


def _out_of_credits(account: Account) -> HTTPException:
    """402 payload shown when the shared pool is exhausted — the user must ask an
    administrator to reset the limit (via POST /api/credits/reset)."""
    return HTTPException(
        status_code=402,
        detail={
            "error": "out_of_credits",
            "message": (
                "You have exhausted your credit limit. Please contact your administrator "
                "to reset the limit."
            ),
            "balance": float(round(account.balance, 2)),
            "plan_total": float(round(account.plan_total, 2)),
            "percent_remaining": account.percent_remaining,
        },
    )


async def _keep_index_warm(components: Any, search_settings: SearchSettings) -> None:
    """Probe both search indexes now, then every ``index_mongo_keep_warm_s``.

    Atlas pages a search index out of memory when nobody queries it; the next
    query then reads it from disk and takes seconds. A random probe vector walks a
    different part of the HNSW graph each time, so the whole graph stays resident.
    """
    interval = search_settings.index_mongo_keep_warm_s
    while True:
        started = time.monotonic()
        try:
            probe = [random.gauss(0.0, 1.0) for _ in range(search_settings.embed_dim)]
            await components.vector.search(probe, limit=search_settings.candidate_k)
            await components.keyword.search("document", limit=1)
            _log.info("search_index_warmed", seconds=round(time.monotonic() - started, 2))
        except Exception as exc:  # warming must never take the service down
            _log.warning("search_index_warm_failed", error=str(exc)[:300])
        if interval <= 0:
            return
        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(
        level=settings.log_level, json_logs=settings.log_json, service_name=SERVICE_NAME
    )
    setup_tracing(settings.otel, service_name=SERVICE_NAME)

    search_settings = SearchSettings()
    components = build_components(settings, search_settings)
    await components.vector.ensure()

    # Reset the shared credit pool on restart (survives refreshes, resets on restart).
    if search_settings.credit_reset_on_start:
        pool = await components.credits.reset()
        _log.info("credit_pool_reset_on_start", balance=str(pool.balance),
                  plan_total=str(pool.plan_total))

    app.state.settings = settings
    app.state.search_settings = search_settings
    app.state.components = components
    # The deployed orchestrator, for demonstrating what ExampleApp actually gets.
    # Absent unless an ARN is configured, and unmetered when present.
    app.state.agent = (
        AgentRuntimeClient(
            search_settings.agent_runtime_arn,
            region=search_settings.agent_region,
            read_timeout_s=search_settings.agent_read_timeout_s,
            ca_bundle=search_settings.agent_tls_ca_file,
        )
        if search_settings.agent_runtime_arn
        else None
    )
    if app.state.agent is not None:
        _log.info(
            "deployed_agent_enabled",
            arn=search_settings.agent_runtime_arn, region=search_settings.agent_region,
        )
    _log.info(
        "search_startup_complete",
        provider=search_settings.embed_provider.value,
        data_dir=search_settings.data_dir,
    )
    # Live Atlas index: warm the pool and both search indexes in the background so
    # startup is not blocked and the first user does not pay the cold read.
    warming: asyncio.Task[None] | None = None
    if search_settings.store_backend is StoreBackend.MONGO:
        warming = asyncio.create_task(_keep_index_warm(components, search_settings))
    try:
        yield
    finally:
        if warming is not None and not warming.done():
            warming.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await warming
        await components.aclose()
        _log.info("search_shutdown_complete")


ops = APIRouter(tags=["ops"])


@ops.get("/", response_class=HTMLResponse, summary="Search UI")
async def ui() -> HTMLResponse:
    return HTMLResponse(_UI_PATH.read_text(encoding="utf-8"))


@ops.get("/admin", response_class=HTMLResponse, summary="Admin console")
async def admin() -> HTMLResponse:
    """Admin console: user feedback + credit top-up / reset."""
    return HTMLResponse(_ADMIN_PATH.read_text(encoding="utf-8"))


@ops.get("/favicon.svg", include_in_schema=False, summary="Favicon")
async def favicon() -> Response:
    return Response(_FAVICON_PATH.read_bytes(), media_type="image/svg+xml")


@ops.get("/healthz", summary="Liveness probe")
async def healthz() -> dict[str, str]:
    return {"status": "alive"}


@ops.get("/readyz", summary="Readiness probe")
async def readyz(search: SearchService = Depends(get_search)) -> dict[str, bool]:
    return {"ready": True}


@ops.get("/metrics", summary="Prometheus metrics")
async def metrics() -> Response:
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


v1 = APIRouter(prefix="/v1", tags=["search"])


@v1.post("/index", response_model=IndexSummary, summary="Build the local index from S3")
async def index(
    body: IndexRequest, indexer: IndexingService = Depends(get_indexer)
) -> IndexSummary:
    return await indexer.index(body)


@v1.get("/search", response_model=SearchResponse, summary="Hybrid search")
async def search(
    q: str = Query(..., description="Search query"),
    k: int | None = Query(default=None, ge=1, le=50),
    namespace: str = Query(default="default"),
    svc: SearchService = Depends(get_search),
) -> SearchResponse:
    return await svc.search(q, namespace=namespace, top_k=k)


@v1.get("/files/{token}/{filename}", include_in_schema=False)
async def download_file(token: str, filename: str, request: Request) -> Response:
    """Serve a source document to the browser behind a credential-free URL.

    The URL ends in the original filename and carries only a signed token — never the
    bucket, the storage key, or an AWS access-key id. The app fetches from S3 with its own
    credentials and returns the bytes. ``filename`` is cosmetic (it drives the browser's
    "Save as"); the token is what authorises and identifies the object.
    """
    downloads = request.app.state.components.downloads
    try:
        dl = await downloads.open(token)
    except (DownloadTokenError, DownloadNotFound) as exc:
        raise HTTPException(
            status_code=404, detail="This download link is invalid or has expired."
        ) from exc
    return Response(
        content=dl.data,
        media_type=dl.content_type,
        headers={"Content-Disposition": dl.content_disposition, "Cache-Control": "private, no-store"},
    )


@v1.get("/stats", summary="Index sizes")
async def stats(request: Request) -> dict[str, int]:
    components = request.app.state.components
    return {
        "vector_chunks": await components.vector.count(),
        "keyword_chunks": await components.keyword.count(),
    }


@v1.post("/session/new", response_model=NewSessionResponse, summary="Start a new chat session")
async def new_session(
    request: Request, conversations: ConversationStore = Depends(get_conversations)
) -> NewSessionResponse:
    ttl = request.app.state.search_settings.session_ttl_minutes
    return NewSessionResponse(session_id=conversations.new_session(), ttl_minutes=ttl)


@v1.post("/chat", response_model=ChatResponse, summary="Ask a question (conversational RAG)")
async def chat(
    body: ChatRequest,
    conversations: ConversationStore = Depends(get_conversations),
    answerer: AnswerService = Depends(get_answer),
    credits: CreditStore = Depends(get_credits),
    user: str = Depends(get_current_user),
    query_log: Any | None = Depends(get_query_log),
) -> ChatResponse:
    """Answer a question grounded in the indexed documents, maintaining context
    within the session's 30-minute window. Omit ``session_id`` to start fresh.
    Optional ``filters`` restrict retrieval to matching metadata facets.

    Credit-metered against the shared pool: refuses with HTTP 402 when the pool is
    empty (the model is never called), and otherwise deducts the real token cost +
    50% overhead after a successful answer. A failed model call deducts nothing and
    surfaces as HTTP 502 (matching POST /api/credits/run).

    Every answered question is also written to the Mongo audit log when configured
    (best-effort + off the response path — it can neither fail nor slow the answer)."""
    account = await credits.get_or_create()
    if account.is_exhausted:
        raise _out_of_credits(account)

    session_id = body.session_id or conversations.new_session()
    try:
        resp = await answerer.answer(
            body.message, session_id=session_id, filters=body.filters, namespace=body.namespace
        )
    except AnswerUnavailableError as exc:
        # 502, never a 200 carrying a fallback answer: a fallback is indistinguishable
        # from a real grounded answer to the caller, so the failure must be explicit.
        # We raise before charging, so the balance is untouched.
        _log.warning("chat_llm_failed", user=user, session=session_id, error=str(exc))
        raise HTTPException(
            status_code=502, detail="The model is temporarily unavailable — please try again."
        ) from exc

    # Deduct only when the model actually ran (real token usage). A zero-usage answer
    # (e.g. the canned "couldn't find anything" reply) never called the model → no charge.
    if resp.usage.input_tokens or resp.usage.output_tokens:
        result = await credits.charge(
            actor=user,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            embed_tokens=resp.usage.embed_tokens,
            cache_read_tokens=resp.usage.cache_read_tokens,
            cache_write_tokens=resp.usage.cache_write_tokens,
            model=resp.usage.model or "",
        )
        resp.credit = _credit_info(result.account, result.breakdown)
    else:
        resp.credit = _credit_info(account)

    # Audit trail. Last thing before returning: the answer, citations, usage and the
    # settled credit/balance only all exist here. Wrapped because an audit write must
    # never turn a good answer into an error.
    if query_log is not None:
        try:
            query_log.record(
                query_log.build_document(
                    question=body.message,
                    answer=resp.answer,
                    sources=query_log.sources_from_citations(resp.citations),
                    usage=resp.usage,
                    credit=resp.credit,
                    session_id=resp.session_id,
                    actor=user,
                    took_ms=resp.took_ms,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            _log.warning("query_log_record_failed", error=str(exc))
    return resp


@v1.get("/agent/status", response_model=AgentStatus, summary="Is the deployed agent reachable?")
async def agent_status(request: Request) -> AgentStatus:
    """What the UI needs to decide whether to offer the deployed agent at all.

    Returns ``enabled: false`` rather than an error when no ARN is configured, so
    a deployment without one simply never shows the option.
    """
    search_settings: SearchSettings = request.app.state.search_settings
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        return AgentStatus()
    return AgentStatus(
        enabled=True,
        runtime_arn=agent.runtime_arn,
        region=search_settings.agent_region,
        tenants=[
            AgentTenantChoice(label=label, base_url=url)
            for label, url in search_settings.agent_tenant_choices()
        ],
        metered=False,
    )


@v1.post("/agent/chat", response_model=AgentChatResponse, summary="Ask the DEPLOYED agent")
async def agent_chat(body: AgentChatRequest, request: Request) -> AgentChatResponse:
    """Put the question to the deployed orchestrator instead of this service.

    Same code path ExampleApp uses, so what is demonstrated is what they receive --
    including feature-code routing and the sub-agent delegation.

    Unmetered on both sides: no reservation is sent, so the agent never calls
    Licensing and nothing is written to ExampleApp's credit ledger; and the local
    credit pool is untouched, because the model call happened in AWS rather than
    here. A demo can therefore run beside a live integration without touching it.
    """
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="No deployed agent is configured. Set SEARCH_AGENT_RUNTIME_ARN to enable it.",
        )
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    try:
        answer = await agent.ask(
            body.message,
            base_url=body.base_url,
            session_id=body.session_id or new_agent_session(),
            business_object=body.business_object,
            object_codes=body.object_codes,
        )
    except AgentUnavailableError as exc:
        # 502, never a 200 carrying a local fallback: in a demo an answer from the
        # wrong pipeline is worse than a visible failure, because nobody can tell.
        _log.warning("agent_chat_failed", base_url=body.base_url, error=str(exc)[:300])
        raise HTTPException(
            status_code=502, detail=f"The deployed agent did not answer: {exc}"
        ) from exc

    return AgentChatResponse(
        session_id=answer.session_id,
        answer=answer.answer,
        citations=[AgentCitation.from_agent(c) for c in answer.citations],
        documents=[AgentCitation.from_agent(d) for d in answer.documents],
        documents_total=answer.documents_total,
        route=answer.route,
        tenant=answer.tenant,
        environment=answer.environment,
        business_object=answer.business_object,
        object_codes=answer.object_codes,
        usage=UsageInfo(
            input_tokens=answer.input_tokens,
            output_tokens=answer.output_tokens,
            embed_tokens=answer.embedding_tokens,
            model=answer.model_id,
        ),
        took_ms=answer.took_ms,
        metered=False,
    )


@v1.post("/feedback", summary="Record thumbs up/down feedback on an answer")
async def feedback(
    body: FeedbackRequest,
    user: str = Depends(get_current_user),
    fb: FeedbackStore = Depends(get_feedback),
) -> dict[str, bool]:
    """Append one feedback entry to the likes/dislikes log (by rating). A 👎 (down)
    requires a comment; a 👍 (up) carries none. Non-blocking: the write runs off the
    event loop."""
    rating = body.rating.strip().lower()
    if rating not in ("up", "down"):
        raise HTTPException(status_code=422, detail="rating must be 'up' or 'down'")
    comment = body.comment.strip()
    if rating == "down" and not comment:
        raise HTTPException(status_code=422, detail="a comment is required for negative feedback")
    entry: dict[str, object] = {
        "ts": utcnow().isoformat(),
        "rating": rating,
        "actor": user,
        "session_id": body.session_id,
        "question": body.question,
        "answer": body.answer,
    }
    if rating == "down":
        entry["comment"] = comment
    await fb.record(entry)
    return {"ok": True}


@v1.get("/feedback", summary="Recent user feedback (admin)")
async def feedback_list(
    limit: int = Query(default=200, ge=1, le=1000),
    fb: FeedbackStore = Depends(get_feedback),
) -> dict[str, Any]:
    """All recorded feedback (newest first) plus like/dislike totals — powers /admin."""
    entries = await fb.load()
    likes = sum(1 for e in entries if e.get("rating") == "up")
    return {
        "total": len(entries),
        "likes": likes,
        "dislikes": len(entries) - likes,
        "entries": entries[:limit],
    }


@v1.get("/facets", summary="Available metadata filter facets")
async def facets(request: Request, namespace: str = "default") -> dict[str, list[str]]:
    # Any extra query params are treated as active facet selections, so the returned
    # options are cascaded (interdependent): ?status=Draft narrows the other facets to
    # values that co-occur on Draft documents.
    active = {k: v for k, v in request.query_params.items() if k != "namespace" and v}
    return await request.app.state.components.metadata.facets(namespace, filters=active)


@v1.get("/documents", response_model=DocumentListResponse, summary="Browse documents by facet")
async def documents(
    request: Request,
    namespace: str = "default",
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> DocumentListResponse:
    """List documents filtered by any metadata facet (e.g. ``?status=Draft``),
    paginated. All query params other than namespace/limit/offset are treated as
    facet filters (ANDed). ``total`` drives the Next/Prev controls."""
    import asyncio

    reserved = {"namespace", "limit", "offset", "expiry_from", "expiry_to"}
    filters = {k: v for k, v in request.query_params.items() if k not in reserved and v}
    qp = request.query_params
    components = request.app.state.components
    total, docs = await components.metadata.list_documents(
        namespace, filters=filters, expiry_from=qp.get("expiry_from"),
        expiry_to=qp.get("expiry_to"), limit=limit, offset=offset,
    )
    urls = await asyncio.gather(*(components.presign(d["source_key"]) for d in docs))
    items = [
        DocumentListItem(
            document_id=d["document_id"],
            source_key=d["source_key"],
            title=d["title"],
            expiry_date=d.get("expiry_date"),
            download_url=url,
            facets=d["facets"],
        )
        for d, url in zip(docs, urls, strict=True)
    ]
    return DocumentListResponse(total=total, limit=limit, offset=offset, documents=items)


@v1.post("/admin/backfill-dates", summary="Backfill expiry dates from titles (no re-index)")
async def backfill_dates(request: Request) -> dict[str, int]:
    from rag_search.metadata_extract import parse_expiry

    updated = await request.app.state.components.metadata.backfill_expiry(parse_expiry)
    return {"updated": updated}


@v1.get("/metadata/{document_id}", summary="Canonical metadata for a document")
async def metadata(request: Request, document_id: str) -> Response:
    from fastapi.responses import JSONResponse

    meta = await request.app.state.components.metadata.get(document_id)
    if meta is None:
        return JSONResponse({"detail": "not found"}, status_code=404)
    return JSONResponse(meta.model_dump(mode="json"))


credits_api = APIRouter(prefix="/api/credits", tags=["credits"])


@credits_api.get("/balance", response_model=CreditBalance, summary="Current credit balance")
async def credit_balance(
    user: str = Depends(get_current_user), credits: CreditStore = Depends(get_credits)
) -> CreditBalance:
    """The shared pool's server-authoritative balance — every browser renders its
    meter from this (and polls it), so a refresh shows the stored balance and all
    users see the same number drop as anyone spends."""
    account = await credits.get_or_create()
    return CreditBalance(
        balance=float(round(account.balance, 2)),
        plan_total=float(round(account.plan_total, 2)),
        percent_remaining=account.percent_remaining,
    )


@credits_api.post("/run", response_model=CreditRunResponse, summary="Run a prompt (metered)")
async def credit_run(
    body: CreditRunRequest,
    request: Request,
    user: str = Depends(get_current_user),
    credits: CreditStore = Depends(get_credits),
    llm: LLM = Depends(get_llm),
) -> CreditRunResponse:
    """Send the prompt to Claude (Bedrock, Sonnet) and deduct credits from the real
    token usage + 50% overhead, inside a row-locked transaction. Returns 402 without
    calling the model when the balance is already spent; on a model error, deducts
    nothing and surfaces a clear error."""
    from rag_search.pricing import compute_cost

    account = await credits.get_or_create()
    if account.is_exhausted:
        raise _out_of_credits(account)

    settings: SearchSettings = request.app.state.search_settings
    messages = [{"role": "user", "content": [{"text": body.prompt}]}]
    try:
        result = await llm.complete(
            messages,
            system=_RUN_SYSTEM,
            max_tokens=settings.answer_max_tokens,
            temperature=settings.answer_temperature,
        )
    except Exception as exc:  # any Anthropic/Bedrock error → deduct nothing
        _log.warning("credit_run_llm_failed", user=user, error=str(exc))
        raise HTTPException(
            status_code=502, detail="The model is temporarily unavailable — please try again."
        ) from exc

    charge = await credits.charge(
        actor=user, input_tokens=result.input_tokens, output_tokens=result.output_tokens,
        model=result.model,
    )
    cost = compute_cost(
        model=result.model, input_tokens=result.input_tokens, output_tokens=result.output_tokens
    )
    return CreditRunResponse(
        reply=result.text,
        usage=UsageInfo(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.input_tokens + result.output_tokens,
            input_cost_usd=cost.input_cost_usd,
            output_cost_usd=cost.output_cost_usd,
            total_cost_usd=cost.total_cost_usd,
            model=result.model,
        ),
        credits_used=float(round(charge.breakdown.credits_used, 2)),
        base_cost_usd=float(round(charge.breakdown.base_cost, 6)),
        charged_usd=float(round(charge.breakdown.charged, 6)),
        balance=float(round(charge.account.balance, 2)),
        plan_total=float(round(charge.account.plan_total, 2)),
        percent_remaining=charge.account.percent_remaining,
    )


@credits_api.post("/topup", response_model=CreditBalance, summary="Top up the credit pool (admin)")
async def credit_topup(
    body: TopUpRequest, credits: CreditStore = Depends(get_credits)
) -> CreditBalance:
    """**Admin action.** Add ``amount`` credits to the shared pool (the plan total is
    raised to match if the balance would exceed it)."""
    account = await credits.top_up(Decimal(str(body.amount)))
    return CreditBalance(
        balance=float(round(account.balance, 2)),
        plan_total=float(round(account.plan_total, 2)),
        percent_remaining=account.percent_remaining,
    )


@credits_api.post("/reset", response_model=CreditBalance, summary="Reset the credit pool to full (admin)")
async def credit_reset(credits: CreditStore = Depends(get_credits)) -> CreditBalance:
    """**Admin action.** Reset the shared credit pool back to its full starting balance
    (``SEARCH_CREDIT_PLAN_TOTAL``, default 1000) — used to restore access after users
    hit the limit. Returns the refreshed ``{balance, plan_total, percent_remaining}``."""
    account = await credits.reset()
    return CreditBalance(
        balance=float(round(account.balance, 2)),
        plan_total=float(round(account.plan_total, 2)),
        percent_remaining=account.percent_remaining,
    )


@credits_api.post("/upgrade", response_model=CreditBalance, summary="Upgrade to the Pro plan")
async def credit_upgrade(credits: CreditStore = Depends(get_credits)) -> CreditBalance:
    """Move the shared pool to the Pro plan (5x credits) and refill."""
    account = await credits.upgrade()
    return CreditBalance(
        balance=float(round(account.balance, 2)),
        plan_total=float(round(account.plan_total, 2)),
        percent_remaining=account.percent_remaining,
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Search",
        version="0.1.0",
        description="Local hybrid (vector + keyword) search over compliance documents.",
        lifespan=lifespan,
    )
    app.include_router(ops)
    app.include_router(v1)
    app.include_router(credits_api)
    return app


app = create_app()
