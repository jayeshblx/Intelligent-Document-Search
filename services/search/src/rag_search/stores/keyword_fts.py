"""Keyword index over SQLite FTS5 (platform capability #7 — free/local BM25).

SQLite's FTS5 gives real BM25 full-text ranking from the Python standard library
— no server, no dependency, one file on disk. It also serves as the canonical
*chunk store* (text + payload) so fused results can be hydrated by id.

If a SQLite build lacks the FTS5 module, we transparently fall back to a LIKE
scan (slower, still correct) so the service runs everywhere. All SQLite calls run
in a worker thread behind a lock (the connection is shared, sync).
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rag_core.logging import get_logger
from rag_search.facet_filters import FacetFilters, accepted_values

_log = get_logger("rag_search.keyword_fts")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class KeywordIndex:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = asyncio.Lock()
        self._fts = True
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS chunks ("
            "chunk_id TEXT PRIMARY KEY, text TEXT, payload TEXT)"
        )
        try:
            cur.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5("
                "chunk_id UNINDEXED, text, tokenize='porter unicode61')"
            )
        except sqlite3.OperationalError as exc:
            self._fts = False
            _log.warning("fts5_unavailable_using_like", error=str(exc))
        self._conn.commit()

    async def reset(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._reset_sync)

    def _reset_sync(self) -> None:
        cur = self._conn.cursor()
        cur.execute("DROP TABLE IF EXISTS chunks")
        if self._fts:
            cur.execute("DROP TABLE IF EXISTS chunk_fts")
        self._conn.commit()
        self._init_schema()

    async def add(self, entries: list[tuple[str, str, dict[str, Any]]]) -> int:
        if not entries:
            return 0
        async with self._lock:
            return await asyncio.to_thread(self._add_sync, entries)

    async def delete_document(self, document_id: str) -> int:
        """Delete every chunk of one document (soft-deleted upstream). Returns the row
        count removed. No-op if the document isn't present."""
        async with self._lock:
            return await asyncio.to_thread(self._delete_document_sync, document_id)

    def _delete_document_sync(self, document_id: str) -> int:
        cur = self._conn.cursor()
        rows = cur.execute(
            "SELECT chunk_id FROM chunks WHERE json_extract(payload, '$.document_id') = ?",
            (document_id,),
        ).fetchall()
        ids = [r["chunk_id"] for r in rows]
        if not ids:
            return 0
        qmarks = ",".join("?" for _ in ids)
        cur.execute(f"DELETE FROM chunks WHERE chunk_id IN ({qmarks})", ids)
        if self._fts:
            cur.execute(f"DELETE FROM chunk_fts WHERE chunk_id IN ({qmarks})", ids)
        self._conn.commit()
        return len(ids)

    async def update_payload(self, document_id: str, fields: Mapping[str, Any]) -> int:
        """Merge facet fields into every chunk payload of one document.

        The chunk payload is what query-time facet filtering reads, so an enrichment
        pass has to reach it as well as the metadata store. A ``None`` value removes the
        key. Text and FTS rows are untouched — no re-tokenising, no re-embedding.
        """
        if not fields:
            return 0
        async with self._lock:
            return await asyncio.to_thread(self._update_payload_sync, document_id, fields)

    def _update_payload_sync(self, document_id: str, fields: Mapping[str, Any]) -> int:
        cur = self._conn.cursor()
        rows = cur.execute(
            "SELECT chunk_id, payload FROM chunks "
            "WHERE json_extract(payload, '$.document_id') = ?",
            (document_id,),
        ).fetchall()
        updates = []
        for r in rows:
            try:
                payload = json.loads(r["payload"]) or {}
            except (json.JSONDecodeError, TypeError):
                continue
            for key, value in fields.items():
                if value is None:
                    payload.pop(key, None)
                else:
                    payload[key] = value
            updates.append((json.dumps(payload), r["chunk_id"]))
        if not updates:
            return 0
        cur.executemany("UPDATE chunks SET payload = ? WHERE chunk_id = ?", updates)
        self._conn.commit()
        return len(updates)

    def _add_sync(self, entries: list[tuple[str, str, dict[str, Any]]]) -> int:
        cur = self._conn.cursor()
        cur.executemany(
            "INSERT OR REPLACE INTO chunks(chunk_id, text, payload) VALUES (?, ?, ?)",
            [(cid, text, json.dumps(payload)) for cid, text, payload in entries],
        )
        if self._fts:
            cur.executemany(
                "INSERT INTO chunk_fts(chunk_id, text) VALUES (?, ?)",
                [(cid, text) for cid, text, _ in entries],
            )
        self._conn.commit()
        return len(entries)

    async def search(
        self, query: str, *, limit: int, filters: FacetFilters | None = None
    ) -> list[tuple[str, float]]:
        """Return (chunk_id, score) best-first, optionally filtered by facet payload."""
        tokens = _TOKEN_RE.findall(query.lower())
        if not tokens:
            return []
        facets = {k: vals for k, v in (filters or {}).items() if (vals := accepted_values(v))}
        async with self._lock:
            return await asyncio.to_thread(self._search_sync, tokens, limit, facets)

    def _search_sync(
        self, tokens: list[str], limit: int, facets: dict[str, list[str]]
    ) -> list[tuple[str, float]]:
        cur = self._conn.cursor()
        # Facet predicates apply to the chunk payload JSON in both branches. A facet
        # with several accepted values becomes an IN list (OR within the facet, AND
        # across facets).
        facet_sql = "".join(
            f" AND json_extract(chunks.payload, '$.{k}') IN "
            f"({','.join('?' * len(vals))})"
            for k, vals in facets.items()
        )
        facet_params = [v for vals in facets.values() for v in vals]

        if self._fts:
            match = " OR ".join(f'"{t}"' for t in tokens)
            rows = cur.execute(
                "SELECT chunk_fts.chunk_id AS chunk_id, bm25(chunk_fts) AS b "
                "FROM chunk_fts JOIN chunks ON chunks.chunk_id = chunk_fts.chunk_id "
                f"WHERE chunk_fts MATCH ?{facet_sql} ORDER BY b LIMIT ?",
                (match, *facet_params, limit),
            ).fetchall()
            # bm25 returns lower=better (often negative); flip to higher=better.
            return [(r["chunk_id"], -float(r["b"])) for r in rows]
        # LIKE fallback: score by number of distinct matched tokens.
        like_clauses = " OR ".join("text LIKE ?" for _ in tokens)
        params = [f"%{t}%" for t in tokens]
        rows = cur.execute(
            f"SELECT chunk_id, text FROM chunks WHERE ({like_clauses}){facet_sql} LIMIT ?",
            (*params, *facet_params, limit * 5),
        ).fetchall()
        scored = []
        for r in rows:
            text = (r["text"] or "").lower()
            scored.append((r["chunk_id"], float(sum(text.count(t) for t in tokens))))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    async def get(self, chunk_ids: list[str]) -> dict[str, tuple[str, dict[str, Any]]]:
        if not chunk_ids:
            return {}
        async with self._lock:
            return await asyncio.to_thread(self._get_sync, chunk_ids)

    def _get_sync(self, chunk_ids: list[str]) -> dict[str, tuple[str, dict[str, Any]]]:
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self._conn.execute(
            f"SELECT chunk_id, text, payload FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        return {r["chunk_id"]: (r["text"], json.loads(r["payload"])) for r in rows}

    async def exact_lookup(
        self, identifiers: list[str], *, limit: int = 8
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """Find chunks whose text OR source filename literally contains an identifier.

        Ranking-based search buries exact identifiers: a number like ``1032658`` is noise
        to the embedder and dilutes under BM25 when the query has other words, so the one
        document that actually contains it never reaches the top. This is a literal
        substring match — no ranking, no fusion — so a distinctive code always surfaces
        its document. Also matches the filename (source_key), since some identifiers live
        only in the document's name, not its body.
        """
        ids = [i for i in {i.strip() for i in identifiers} if len(i.strip()) >= 4]
        if not ids:
            return []
        async with self._lock:
            return await asyncio.to_thread(self._exact_lookup_sync, ids, limit)

    def _exact_lookup_sync(
        self, ids: list[str], limit: int
    ) -> list[tuple[str, str, dict[str, Any]]]:
        clauses, params = [], []
        for i in ids:
            clauses.append("(text LIKE ? OR json_extract(payload,'$.source_key') LIKE ?)")
            params += [f"%{i}%", f"%{i}%"]
        rows = self._conn.execute(
            f"SELECT chunk_id, text, payload FROM chunks WHERE {' OR '.join(clauses)} LIMIT ?",
            (*params, limit),
        ).fetchall()
        return [(r["chunk_id"], r["text"], json.loads(r["payload"])) for r in rows]

    async def count(self) -> int:
        async with self._lock:
            return await asyncio.to_thread(
                lambda: self._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
            )

    async def aclose(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._conn.close)
