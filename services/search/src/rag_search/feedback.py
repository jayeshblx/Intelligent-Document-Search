"""Per-answer user feedback, appended to JSONL files (likes and dislikes kept apart).

Positive (thumbs-up) and negative (thumbs-down, which always carries a required
comment) feedback are written to *separate* append-only logs — ``feedback_likes.jsonl``
and ``feedback_dislikes.jsonl`` — so each can be reviewed on its own. Writes are
serialised through an ``asyncio.Lock`` and run off the event loop, so recording
feedback never blocks the request that triggered it.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from rag_core.logging import get_logger

_log = get_logger("rag_search.feedback")


class FeedbackStore:
    def __init__(self, positive_path: str, negative_path: str) -> None:
        self._positive = Path(positive_path)
        self._negative = Path(negative_path)
        for p in (self._positive, self._negative):
            p.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def record(self, entry: dict[str, Any]) -> None:
        """Append the entry to the likes or dislikes log, by its ``rating``."""
        path = self._negative if entry.get("rating") == "down" else self._positive
        async with self._lock:
            await asyncio.to_thread(self._append, path, entry)

    def _append(self, path: Path, entry: dict[str, Any]) -> None:
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:  # feedback must never break a request
            _log.warning("feedback_write_failed", error=str(exc))

    async def load(self) -> list[dict[str, Any]]:
        """Every feedback entry from both logs, newest first — powers the admin view."""
        async with self._lock:
            return await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for path in (self._positive, self._negative):
            if not path.exists():
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                _log.warning("feedback_read_failed", path=str(path), error=str(exc))
                continue
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # skip a corrupt line rather than fail the whole view
        entries.sort(key=lambda e: str(e.get("ts", "")), reverse=True)
        return entries

    @property
    def positive_path(self) -> Path:
        return self._positive

    @property
    def negative_path(self) -> Path:
        return self._negative
