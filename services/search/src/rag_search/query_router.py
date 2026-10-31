"""Reads a question's intent with the model instead of regexes.

The query path used to decide what a question *meant* with eight hand-written patterns
(``_LISTING_START``, ``_FOLLOWUP_RE``, ``_DETECTABLE_FACETS``, …). Each worked for the
phrasings someone anticipated and failed silently for the rest — "Find all Documents who
has Grade A BRC Global **Standard**" listed 132 unrelated standards because a regex
grabbed the word "standard"; "documents with StoreUser1" found nothing because
``created_by`` was left off an allowlist. Every fix meant another rule, and English does
not run out of phrasings.

So the model reads the question and returns a *plan*; the existing (working) retrieval
machinery executes it.

Two properties keep this safe:

* **The model cannot invent a filter.** Every key and value it proposes is validated
  against the corpus's real facet vocabulary and dropped if it doesn't exist. The worst
  a bad plan can do is retrieve less well — never fabricate a filter that shows wrong
  documents as if they matched.
* **It cannot take the app down.** Any failure — bad JSON, a model outage, a nonsense
  plan — returns ``None``, and the caller keeps its previous behaviour.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

import structlog

_log = structlog.get_logger(__name__)

# Facet values are unbounded (13k object codes), so the vocabulary is NOT sent to the
# model. It copies candidate values out of the question; validation against the real
# vocabulary is what makes them true. Only the keys and a couple of examples go in the
# prompt, to convey each field's shape.
_EXAMPLES_PER_FACET = 3
_MAX_HISTORY_TURNS = 6

_ROUTER_SYSTEM = """You route questions for a document-search system. Return ONLY a JSON object.

Fields:
- "intent": "list" when the user wants the SET of documents matching a property
  ("how many certificates", "documents by storeuser1", "all expired ones"), or
  "content" when they want information FROM documents ("what does X say", "which
  suppliers have Grade A", "we're missing a box of Soda").
  If unsure, use "content" — it still searches everything and cites sources.
- "filters": {facet_key: value} ONLY when the question names a value that plausibly
  belongs to that field. Copy the value EXACTLY as written in the question. Use {} if
  none. Never guess a value the user did not write.
- "expiry_from"/"expiry_to": ISO dates (YYYY-MM-DD), only for expiry questions
  ("expired" -> expiry_to = today; "expiring in future" -> expiry_from = today). Else null.
- "continues_thread": true only when the question cannot be understood alone and needs
  the previous turns ("what about it?", "tell me more", "find the store for this").
  A question that introduces its own subject is false, even mid-conversation.
- "search_query": what to search for. Usually the question itself. When
  continues_thread is true, rewrite it into a STANDALONE query by pulling the subject in
  from the conversation.

Judge intent from meaning, not keywords. "Find all Documents who has Grade A BRC Global
Standard for Food Safety" asks about documents whose CONTENT mentions a Grade A BRC
certification: intent "content", no filter — "Standard" there is part of a scheme name,
not a document type.

Output JSON only. No prose, no code fences."""


class QueryRouter:
    """Turns a question into a validated retrieval plan."""

    def __init__(self, llm: Any, *, max_tokens: int = 400) -> None:
        self._llm = llm
        self._max_tokens = max_tokens
        # Tokens spent by the most recent route() call, so the answer service can fold the
        # router's cost into the turn's credit charge. Reset at the start of every route().
        self.last_input_tokens = 0
        self.last_output_tokens = 0

    async def route(
        self,
        question: str,
        *,
        facet_values: dict[str, list[str]],
        history: list[Any] | None = None,
    ) -> dict[str, Any] | None:
        """Return a validated plan, or None to fall back to the caller's own logic."""
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        try:
            raw = await self._ask(question, facet_values, history or [])
        except Exception as exc:  # a router failure must never fail the question
            _log.warning("query_router_llm_failed", error=str(exc))
            return None
        plan = _parse(raw)
        if plan is None:
            _log.warning("query_router_unparseable", raw=raw[:200])
            return None
        return _validate(plan, facet_values)

    async def _ask(
        self, question: str, facet_values: dict[str, list[str]], history: list[Any]
    ) -> str:
        parts: list[str] = [f"Today is {datetime.now(UTC).date().isoformat()}."]
        vocab = _vocabulary_block(facet_values)
        if vocab:
            parts.append("Facet fields you may filter on (with example values):\n" + vocab)
        convo = _history_block(history)
        if convo:
            parts.append("Conversation so far:\n" + convo)
        parts.append(f"Question: {question}")
        result = await self._llm.complete(
            [{"role": "user", "content": [{"text": "\n\n".join(parts)}]}],
            system=_ROUTER_SYSTEM,
            max_tokens=self._max_tokens,
            temperature=0.0,
        )
        self.last_input_tokens = int(getattr(result, "input_tokens", 0) or 0)
        self.last_output_tokens = int(getattr(result, "output_tokens", 0) or 0)
        return str(result.text or "")


def _vocabulary_block(facet_values: dict[str, list[str]]) -> str:
    lines = []
    for key, values in sorted(facet_values.items()):
        if not values:
            continue
        sample = ", ".join(str(v) for v in values[:_EXAMPLES_PER_FACET])
        lines.append(f"- {key} (e.g. {sample})")
    return "\n".join(lines)


def _history_block(history: list[Any]) -> str:
    lines = []
    for msg in history[-_MAX_HISTORY_TURNS:]:
        role = getattr(msg, "role", "")
        content = str(getattr(msg, "content", ""))[:300]
        if role and content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


_JSON_RE = re.compile(r"\{.*\}", re.S)


def _parse(raw: str) -> dict[str, Any] | None:
    """Pull the JSON object out of the reply (models like to add fences or prose)."""
    m = _JSON_RE.search(raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate(plan: dict[str, Any], facet_values: dict[str, list[str]]) -> dict[str, Any]:
    """Keep only what the corpus can actually support.

    This is the load-bearing safety step: a filter the model invented would silently
    show the wrong documents as though they matched, which is worse than no filter at
    all. So every value must exist in the real vocabulary, matched case-insensitively.
    """
    intent = plan.get("intent")
    out: dict[str, Any] = {
        "intent": intent if intent in ("list", "content") else "content",
        "filters": {},
        "expiry_from": None,
        "expiry_to": None,
        "continues_thread": bool(plan.get("continues_thread")),
        "search_query": "",
    }

    raw_filters = plan.get("filters")
    if isinstance(raw_filters, dict):
        for key, value in raw_filters.items():
            if key not in facet_values or not isinstance(value, str) or not value.strip():
                continue
            # Case-insensitive match against the REAL values; keep the corpus's spelling
            # ("StoreUser1" in the question -> "storeuser1" as stored).
            canon = {str(v).lower(): str(v) for v in facet_values[key]}
            real = canon.get(value.strip().lower())
            if real is not None:
                out["filters"][key] = real
            else:
                _log.info("query_router_dropped_filter", key=key, value=value[:60])

    for field in ("expiry_from", "expiry_to"):
        v = plan.get(field)
        if isinstance(v, str) and _ISO_DATE.match(v.strip()):
            out[field] = v.strip()

    sq = plan.get("search_query")
    out["search_query"] = sq.strip() if isinstance(sq, str) and sq.strip() else ""
    return out
