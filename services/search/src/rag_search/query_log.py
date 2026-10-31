"""Audit log of every answered question, written to MongoDB.

Separate from :mod:`rag_search.mongo_source` on purpose: that one is a read-only,
index-time metadata source pointed at a different database. This one is write-only and
sits on the live request path, which drives the two rules it is built around:

* **It never fails a request.** Every failure path is swallowed and logged. A question
  that was answered must still be answered even if the audit write dies.
* **It never delays a request.** The insert is dispatched as a background task, so the
  user's answer returns immediately; a Mongo outage stalls the log, not the reply.
  (``serverSelectionTimeoutMS`` alone would add seconds to every answer during an
  outage — hence fire-and-forget rather than merely a short timeout.)

Field names are PascalCase to match the existing log documents in this database
(``DateCreated``/``Question``/``InputTokens``/…) rather than this repo's snake_case.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import structlog

_log = structlog.get_logger(__name__)

# Cap how much of each source snippet is kept: enough to identify the passage, not so
# much that a many-source answer bloats the document.
_SNIPPET_CHARS = 400


class MongoQueryLog:
    """Best-effort, non-blocking audit trail of answered questions."""

    def __init__(
        self,
        uri: str,
        *,
        database: str,
        collection: str,
        tls_ca_file: str | None = None,
        max_answer_chars: int = 20000,
    ) -> None:
        self._uri = uri
        self._database = database
        self._collection = collection
        self._tls_ca_file = tls_ca_file
        self._max_answer_chars = max_answer_chars
        self._client: Any = None
        # Strong refs to in-flight inserts: asyncio only holds weak refs to tasks, so
        # without this a task can be garbage-collected mid-write.
        self._pending: set[asyncio.Task[None]] = set()
        self._captured: list[dict[str, Any]] = []
        self._capture = False

    # -- test seam ---------------------------------------------------------------
    def capture(self) -> None:
        """Record documents in memory instead of connecting (tests).

        Mirrors ``MongoMetadataSource.seed``: the network is never touched, so a dummy
        URI stays inert.
        """
        self._capture = True

    @property
    def captured(self) -> list[dict[str, Any]]:
        """Documents that would have been inserted (only populated after capture())."""
        return self._captured

    # -- connection --------------------------------------------------------------
    def _connect(self) -> Any:
        from pymongo import MongoClient  # lazy: only needed when the log is configured

        if self._client is None:
            kwargs: dict[str, Any] = {"serverSelectionTimeoutMS": 8000}
            if self._tls_ca_file:
                kwargs["tlsCAFile"] = self._tls_ca_file
            self._client = MongoClient(self._uri, **kwargs)
        return self._client

    # -- writing -----------------------------------------------------------------
    def build_document(
        self,
        *,
        question: str,
        answer: str,
        sources: list[dict[str, Any]],
        usage: Any,
        credit: Any = None,
        session_id: str = "",
        actor: str = "",
        took_ms: float = 0.0,
    ) -> dict[str, Any]:
        """Shape one log document. Pure — split out so tests can assert the shape."""
        answer = answer or ""
        truncated = len(answer) > self._max_answer_chars
        doc: dict[str, Any] = {
            "DateCreated": datetime.now(UTC),
            "Kind": "document_search",
            "Question": question,
            "Answer": answer[: self._max_answer_chars],
            "AnswerTruncated": truncated,
            "Sources": sources,
            "SourceCount": len(sources),
            "InputTokens": int(getattr(usage, "input_tokens", 0) or 0),
            "OutputTokens": int(getattr(usage, "output_tokens", 0) or 0),
            "EmbedTokens": int(getattr(usage, "embed_tokens", 0) or 0),
            "TotalTokens": int(getattr(usage, "total_tokens", 0) or 0),
            "InputCostUsd": float(getattr(usage, "input_cost_usd", 0.0) or 0.0),
            "OutputCostUsd": float(getattr(usage, "output_cost_usd", 0.0) or 0.0),
            "EmbedCostUsd": float(getattr(usage, "embed_cost_usd", 0.0) or 0.0),
            "TotalCostUsd": float(getattr(usage, "total_cost_usd", 0.0) or 0.0),
            "Model": str(getattr(usage, "model", "") or ""),
            "SessionId": session_id,
            "Actor": actor,
            "TookMs": round(float(took_ms or 0.0), 1),
        }
        if credit is not None:
            # What the user was actually charged: base cost + overhead, in credits.
            doc["BaseCostUsd"] = float(getattr(credit, "base_cost_usd", 0.0) or 0.0)
            doc["ChargedUsd"] = float(getattr(credit, "charged_usd", 0.0) or 0.0)
            doc["CreditsUsed"] = float(getattr(credit, "credits_used", 0.0) or 0.0)
            doc["BalanceAfter"] = float(getattr(credit, "balance", 0.0) or 0.0)
        return doc

    @staticmethod
    def sources_from_citations(citations: list[Any]) -> list[dict[str, Any]]:
        """Project citations down to the fields worth keeping in the audit trail."""
        out: list[dict[str, Any]] = []
        for c in citations:
            snippet = str(getattr(c, "snippet", "") or "")
            out.append(
                {
                    "Index": int(getattr(c, "index", 0) or 0),
                    "DocumentId": str(getattr(c, "document_id", "") or ""),
                    "SourceKey": str(getattr(c, "source_key", "") or ""),
                    "Snippet": snippet[:_SNIPPET_CHARS],
                    "PageStart": getattr(c, "page_start", None),
                    "PageEnd": getattr(c, "page_end", None),
                    "Section": getattr(c, "section", None),
                    "Score": round(float(getattr(c, "score", 0.0) or 0.0), 4),
                    "MatchedBy": str(getattr(c, "matched_by", "") or ""),
                }
            )
        return out

    def record(self, doc: dict[str, Any]) -> None:
        """Dispatch one insert in the background. Returns immediately; never raises."""
        if self._capture:
            self._captured.append(doc)
            return
        try:
            task = asyncio.create_task(asyncio.to_thread(self._insert_sync, doc))
        except RuntimeError:  # no running loop (e.g. called outside the server)
            return
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    def _insert_sync(self, doc: dict[str, Any]) -> None:
        try:
            client = self._connect()
            client[self._database][self._collection].insert_one(doc)
        except Exception as exc:  # an audit write must never break answering
            _log.warning("query_log_write_failed", error=str(exc))

    async def aclose(self) -> None:
        """Flush in-flight writes (briefly), then drop the client."""
        if self._pending:
            # Bounded: shutdown must not hang on an unreachable Mongo.
            await asyncio.wait(set(self._pending), timeout=5.0)
        if self._client is not None:
            await asyncio.to_thread(self._client.close)
            self._client = None
