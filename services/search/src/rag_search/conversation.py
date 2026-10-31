"""Conversation memory for the chat RAG — production-grade, multi-user, durable.

The model itself is stateless; "context" is entirely what this store chooses to keep
and what the answer service chooses to re-send each turn. Three layers, so a session
can run for hundreds of turns across a day without losing the thread or exploding cost:

* Tier 1 — durable store + realistic window. Every turn is persisted to MongoDB (survives
  restarts, day-long idle, any number of concurrent users) and only *active* sessions are
  held in a bounded in-process LRU cache, so RAM can't grow without limit. Instead of a
  fixed "last N turns", the recent window is chosen by a **token budget**, so short turns
  keep many and long turns keep fewer.
* Tier 2 — rolling summary. When a session grows past a hard cap the oldest turns are
  compressed into a running summary (one LLM call, only for very long sessions) so the
  gist of the earliest questions survives forever without re-sending them.
* Tier 3 — semantic recall. Each user turn is embedded; when answering, the turns most
  relevant to the *current* question (beyond the recent window) are pulled back, so
  "what did I ask 40 messages ago?" is answered precisely without sending everything.

If MongoDB is not configured (or is unreachable) the store degrades to in-memory only —
the chat keeps working, it just isn't durable. All Mongo I/O runs in a worker thread
(``pymongo`` is synchronous) so the event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from rag_core.logging import get_logger

_log = get_logger("rag_search.conversation")


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) — matches the rest of the cost accounting."""
    return max(1, len(text) // 4)


@dataclass(slots=True)
class Message:
    role: str  # "user" | "assistant"
    content: str
    created_at: float = field(default_factory=time.time)
    # Embedding of a user message, for Tier-3 semantic recall (None for assistant turns
    # and when semantic recall is disabled).
    embedding: list[float] | None = None


@dataclass(slots=True)
class Conversation:
    session_id: str
    messages: list[Message] = field(default_factory=list)
    summary: str = ""  # Tier-2 rolling summary of turns dropped past the hard cap
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)


@dataclass(slots=True)
class ContextBundle:
    """What the answer service should feed the model for continuity."""

    summary: str  # rolling summary of the earliest (dropped) turns
    recalled: list[Message]  # older, semantically-relevant turns (Tier 3)
    recent: list[Message]  # recent verbatim window (Tier 1)

    def is_empty(self) -> bool:
        return not (self.summary or self.recalled or self.recent)


@dataclass(slots=True)
class UsageDelta:
    """Tokens spent by memory-maintenance model calls, folded into the turn's charge."""

    input_tokens: int = 0
    output_tokens: int = 0
    embed_tokens: int = 0

    def add(self, other: "UsageDelta") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.embed_tokens += other.embed_tokens


_SUMMARY_SYSTEM = (
    "You compress the older part of a support conversation about the user's documents "
    "into a compact factual summary (<=180 words). Preserve concrete details the user may "
    "refer back to: what they asked, document names/codes/statuses discussed, and any "
    "conclusions. Merge with the prior summary if given. Write plain prose, no preamble."
)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return -1.0
    return dot / (na * nb)


class _MongoBackend:
    """Thin persistence layer: load/save one session document. Sync pymongo, called
    from worker threads. One document per session keyed by ``_id = session_id``."""

    def __init__(self, uri: str, *, database: str, collection: str, tls_ca_file: str | None) -> None:
        self._uri = uri
        self._database = database
        self._collection = collection
        self._tls_ca_file = tls_ca_file
        self._client: Any = None

    def _coll(self) -> Any:
        from pymongo import MongoClient  # lazy: only when conversations are Mongo-backed

        if self._client is None:
            kwargs: dict[str, Any] = {"serverSelectionTimeoutMS": 8000}
            if self._tls_ca_file:
                kwargs["tlsCAFile"] = self._tls_ca_file
            self._client = MongoClient(self._uri, **kwargs)
        return self._client[self._database][self._collection]

    def load(self, session_id: str) -> Conversation | None:
        doc = self._coll().find_one({"_id": session_id})
        if not doc:
            return None
        msgs = [
            Message(
                role=m["role"],
                content=m.get("content", ""),
                created_at=m.get("created_at", time.time()),
                embedding=m.get("embedding"),
            )
            for m in doc.get("messages", [])
        ]
        return Conversation(
            session_id=session_id,
            messages=msgs,
            summary=doc.get("summary", ""),
            created_at=doc.get("created_at", time.time()),
            last_activity=doc.get("last_activity", time.time()),
        )

    def save(self, conv: Conversation) -> None:
        self._coll().replace_one(
            {"_id": conv.session_id},
            {
                "_id": conv.session_id,
                "messages": [
                    {
                        "role": m.role,
                        "content": m.content,
                        "created_at": m.created_at,
                        "embedding": m.embedding,
                    }
                    for m in conv.messages
                ],
                "summary": conv.summary,
                "created_at": conv.created_at,
                "last_activity": conv.last_activity,
            },
            upsert=True,
        )

    def delete_messages(self, session_id: str) -> None:
        self._coll().update_one(
            {"_id": session_id},
            {"$set": {"messages": [], "summary": "", "last_activity": time.time()}},
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


class ConversationStore:
    def __init__(
        self,
        *,
        ttl_minutes: float = 0.0,
        token_budget: int = 30000,
        hard_message_cap: int = 200,
        summarize_batch: int = 40,
        semantic_recall: bool = True,
        recall_k: int = 4,
        cache_capacity: int = 1000,
        embedder: Any = None,
        llm: Any = None,
        backend: _MongoBackend | None = None,
    ) -> None:
        self._ttl_s = ttl_minutes * 60.0
        self._token_budget = token_budget
        self._hard_cap = hard_message_cap
        self._summarize_batch = summarize_batch
        self._semantic_recall = semantic_recall and embedder is not None
        self._recall_k = recall_k
        self._embedder = embedder
        self._llm = llm
        self._backend = backend
        self._degraded = False  # set once if Mongo goes unreachable → in-memory only
        # Bounded LRU of loaded sessions. Mongo (when present) is the source of truth, so
        # eviction just frees RAM; the session reloads on next access.
        self._cache: "OrderedDict[str, Conversation]" = OrderedDict()
        self._cache_capacity = cache_capacity
        self._lock = asyncio.Lock()

    # ---- lifecycle -------------------------------------------------------------
    def new_session(self) -> str:
        """Mint a fresh session id. The document is created lazily on first append."""
        return uuid.uuid4().hex

    async def aclose(self) -> None:
        if self._backend is not None:
            await asyncio.to_thread(self._backend.close)

    # ---- internal cache/persistence -------------------------------------------
    def _cache_put(self, conv: Conversation) -> None:
        self._cache[conv.session_id] = conv
        self._cache.move_to_end(conv.session_id)
        while len(self._cache) > self._cache_capacity:
            self._cache.popitem(last=False)  # evict least-recently-used (already persisted)

    async def _load(self, session_id: str) -> Conversation:
        conv = self._cache.get(session_id)
        if conv is not None:
            self._cache.move_to_end(session_id)
            return conv
        if self._backend is not None and not self._degraded:
            try:
                loaded = await asyncio.to_thread(self._backend.load, session_id)
            except Exception as exc:  # Mongo down → degrade, keep serving from memory
                self._degrade(exc)
                loaded = None
            if loaded is not None:
                self._cache_put(loaded)
                return loaded
        conv = Conversation(session_id=session_id)
        self._cache_put(conv)
        return conv

    async def _persist(self, conv: Conversation) -> None:
        if self._backend is None or self._degraded:
            return
        try:
            await asyncio.to_thread(self._backend.save, conv)
        except Exception as exc:
            self._degrade(exc)

    def _degrade(self, exc: Exception) -> None:
        if not self._degraded:
            self._degraded = True
            _log.warning("conversation_store_degraded_to_memory", error=str(exc))

    def _expired(self, conv: Conversation) -> bool:
        return self._ttl_s > 0 and (time.time() - conv.last_activity) > self._ttl_s

    # ---- writes ----------------------------------------------------------------
    async def append(self, session_id: str, role: str, content: str) -> UsageDelta:
        """Record a turn. User turns are embedded (Tier-3 recall) when enabled; the
        returned :class:`UsageDelta` reports any embedding tokens so the caller charges
        for them. Triggers rolling summarization when the session grows past the cap."""
        usage = UsageDelta()
        async with self._lock:
            conv = await self._load(session_id)
            if self._expired(conv):
                conv.messages.clear()
                conv.summary = ""
                conv.created_at = time.time()
            embedding: list[float] | None = None
            if role == "user" and self._semantic_recall:
                try:
                    embedding = await self._embedder.embed_query(content)
                    usage.embed_tokens += estimate_tokens(content)
                except Exception as exc:  # recall is best-effort, never blocks a turn
                    _log.warning("conversation_embed_failed", error=str(exc))
            conv.messages.append(Message(role=role, content=content, embedding=embedding))
            conv.last_activity = time.time()
            summarize_usage = await self._maybe_summarize(conv)
            usage.add(summarize_usage)
            self._cache_put(conv)
            await self._persist(conv)
        return usage

    async def _maybe_summarize(self, conv: Conversation) -> UsageDelta:
        """Compress the oldest turns into the rolling summary once the retained message
        count exceeds the hard cap (rare — only very long sessions). Caller holds the lock.
        """
        usage = UsageDelta()
        if self._llm is None or len(conv.messages) <= self._hard_cap:
            return usage
        drop = conv.messages[: self._summarize_batch]
        keep = conv.messages[self._summarize_batch :]
        transcript = "\n".join(f"{m.role}: {m.content}" for m in drop)
        prompt = transcript if not conv.summary else f"Prior summary:\n{conv.summary}\n\nNewer turns:\n{transcript}"
        try:
            result = await self._llm.complete(
                [{"role": "user", "content": [{"text": prompt}]}],
                system=_SUMMARY_SYSTEM,
                max_tokens=320,
                temperature=0.0,
            )
            conv.summary = (result.text or conv.summary).strip()
            conv.messages = keep
            usage.input_tokens += int(getattr(result, "input_tokens", 0) or 0)
            usage.output_tokens += int(getattr(result, "output_tokens", 0) or 0)
        except Exception as exc:  # keep the turns rather than lose them on a summary failure
            _log.warning("conversation_summarize_failed", error=str(exc))
        return usage

    # ---- reads -----------------------------------------------------------------
    async def recent(self, session_id: str) -> list[Message]:
        """The recent verbatim window (token-budgeted) — for routing, follow-up detection,
        and the 'is this a fresh, empty session?' guard."""
        async with self._lock:
            conv = await self._load(session_id)
            if self._expired(conv):
                return []
            return self._recent_window(conv)

    def _recent_window(self, conv: Conversation) -> list[Message]:
        window: list[Message] = []
        budget = self._token_budget
        for m in reversed(conv.messages):
            cost = estimate_tokens(m.content)
            if window and cost > budget:
                break
            budget -= cost
            window.append(m)
        window.reverse()
        return window

    async def context(self, session_id: str, question: str) -> tuple[ContextBundle, UsageDelta]:
        """Everything to feed the model for continuity: rolling summary + semantically
        recalled older turns + the recent verbatim window. Returns any embedding tokens
        spent resolving recall so the caller charges for them."""
        usage = UsageDelta()
        async with self._lock:
            conv = await self._load(session_id)
            if self._expired(conv):
                return ContextBundle("", [], []), usage
            recent = self._recent_window(conv)
            recent_ids = {id(m) for m in recent}
            older = [m for m in conv.messages if id(m) not in recent_ids]
            recalled: list[Message] = []
            if self._semantic_recall and older and question.strip():
                try:
                    qvec = await self._embedder.embed_query(question)
                    usage.embed_tokens += estimate_tokens(question)
                    recalled = self._recall(older, conv.messages, qvec)
                except Exception as exc:
                    _log.warning("conversation_recall_failed", error=str(exc))
            return ContextBundle(conv.summary, recalled, recent), usage

    def _recall(
        self, older: list[Message], all_msgs: list[Message], qvec: list[float]
    ) -> list[Message]:
        """Top-K older USER turns by similarity to the question, each paired with the
        assistant reply that followed it, returned in chronological order."""
        scored = [
            (_cosine(qvec, m.embedding), i)
            for i, m in enumerate(all_msgs)
            if m.role == "user" and m.embedding and m in older
        ]
        scored.sort(reverse=True)
        pick_idx = sorted(i for _s, i in scored[: self._recall_k])
        out: list[Message] = []
        for i in pick_idx:
            out.append(all_msgs[i])
            if i + 1 < len(all_msgs) and all_msgs[i + 1].role == "assistant":
                out.append(all_msgs[i + 1])
        return out

    async def reset(self, session_id: str) -> None:
        """Clear a session's context (keep the id)."""
        async with self._lock:
            conv = await self._load(session_id)
            conv.messages.clear()
            conv.summary = ""
            conv.created_at = time.time()
            conv.last_activity = time.time()
            self._cache_put(conv)
            if self._backend is not None and not self._degraded:
                try:
                    await asyncio.to_thread(self._backend.delete_messages, session_id)
                except Exception as exc:
                    self._degrade(exc)

    async def seconds_remaining(self, session_id: str) -> float:
        """Seconds until this session's context expires (for the UI). -1 when expiry is
        disabled (ttl <= 0) — context is kept until 'New chat'."""
        if self._ttl_s <= 0:
            return -1.0
        conv = self._cache.get(session_id)
        if conv is None:
            return self._ttl_s
        return max(0.0, self._ttl_s - (time.time() - conv.last_activity))
