"""FastAPI dependency providers for the search service."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import Request, Response

from rag_search.answer import LLM, AnswerService
from rag_search.conversation import ConversationStore
from rag_search.credit_store import CreditStore
from rag_search.feedback import FeedbackStore
from rag_search.indexer import IndexingService
from rag_search.retrieval import SearchService

# Stable anonymous-identity cookie. Credits are keyed by user, and this app has no
# login, so a per-browser cookie gives each user their own balance that survives a
# refresh — while an explicit ``X-User-Id`` / ``X-User-Email`` header (used by the
# API and tests) overrides it.
USER_COOKIE = "sscredits_uid"
_COOKIE_MAX_AGE = 60 * 60 * 24 * 365  # one year


def get_search(request: Request) -> SearchService:
    return request.app.state.components.search


def get_indexer(request: Request) -> IndexingService:
    return request.app.state.components.indexer


def get_answer(request: Request) -> AnswerService:
    return request.app.state.components.answer


def get_conversations(request: Request) -> ConversationStore:
    return request.app.state.components.conversations


def get_credits(request: Request) -> CreditStore:
    return request.app.state.components.credits


def get_feedback(request: Request) -> FeedbackStore:
    return request.app.state.components.feedback


def get_llm(request: Request) -> LLM:
    return request.app.state.components.llm


def get_query_log(request: Request) -> Any | None:
    """The Mongo audit log, or None when SEARCH_QUERY_LOG_URI is unset.

    ``getattr`` rather than attribute access: logging is optional, so a component graph
    assembled without it (partial wiring in tests) must still serve requests.
    """
    return getattr(request.app.state.components, "query_log", None)


def get_current_user(request: Request, response: Response) -> str:
    """Resolve the caller's stable user id for credit accounting.

    Priority: explicit ``X-User-Id`` / ``X-User-Email`` header (API/tests) → the
    per-browser cookie (minted + set on first visit, so a refresh keeps the same
    balance and two different browsers are two different users)."""
    explicit = request.headers.get("x-user-id") or request.headers.get("x-user-email")
    if explicit and explicit.strip():
        return explicit.strip()
    uid = request.cookies.get(USER_COOKIE)
    if not uid:
        uid = "anon-" + uuid.uuid4().hex
        response.set_cookie(
            USER_COOKIE, uid, max_age=_COOKIE_MAX_AGE, samesite="lax", httponly=True, path="/"
        )
    return uid
