"""Canonical metadata store (SQLite) with dynamic facets.

Kept separate from the chunk index — it's the single source of truth per document
(full ``DocumentMetadata`` as JSON) plus a denormalised ``(key, value)`` facet
table so the filter UI can enumerate available values cheaply via ``facets()``.

Facets are **dynamic**: whatever keys the indexer stores become filterable
dropdowns. That matters because the authoritative business facets come from the
external MongoDB system of record (``business_object``, ``object_code``,
``status``, ``tenant``, …) and are not fixed fields on ``DocumentMetadata``.

The *denormalised* copy that powers query-time filtering lives on the chunk
payloads in Qdrant + FTS (written by the indexer with the same facet dict).
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rag_core.models.metadata import DocumentMetadata
from rag_search.facet_filters import FacetFilters, accepted_values

# Facets excluded from dropdown enumeration: range-only or high-cardinality.
FACET_DENYLIST = {"effective_date", "expiry_date", "version", "revision", "keywords"}

# Words ignored when matching a question against document titles/metadata — question
# scaffolding and generic nouns that would over-match. Distinctive terms (AMS, EHO,
# supplier names, object codes) survive and drive the match.
_TITLE_SEARCH_STOP = frozenset({
    "the", "and", "for", "are", "is", "was", "were", "does", "did", "can", "use", "used",
    "using", "instead", "of", "an", "in", "on", "to", "me", "my", "show", "list", "which",
    "what", "when", "how", "who", "why", "where", "document", "documents", "doc", "docs",
    "file", "files", "form", "forms", "its", "it", "information", "info", "explain", "tell",
    "about", "across", "raised", "have", "has", "give", "get", "all", "any", "with", "from",
    "that", "this", "their", "there", "please", "full", "based", "detail", "details", "data",
})


def _flatten_facets(facets: dict[str, Any]) -> list[tuple[str, str]]:
    """(key, value) pairs for enumeration; list values expand to one row each."""
    out: list[tuple[str, str]] = []
    for key, value in facets.items():
        if key in FACET_DENYLIST or value in (None, "", []):
            continue
        values = value if isinstance(value, list) else [value]
        for v in values:
            s = str(v).strip()
            if s:
                out.append((key, s))
    return out


class MetadataStore:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = asyncio.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS document_metadata (
                document_id TEXT PRIMARY KEY, namespace TEXT, source_key TEXT,
                payload TEXT, facets TEXT, effective_date TEXT, expiry_date TEXT);
            CREATE TABLE IF NOT EXISTS document_facet (
                namespace TEXT, document_id TEXT, key TEXT, value TEXT);
            CREATE INDEX IF NOT EXISTS ix_facet_ns_key
                ON document_facet(namespace, key, value);
            CREATE INDEX IF NOT EXISTS ix_facet_doc ON document_facet(document_id);
            """
        )
        # Migrate older DBs that predate the date columns (must precede the index below).
        existing = {r["name"] for r in self._conn.execute("PRAGMA table_info(document_metadata)")}
        for col in ("effective_date", "expiry_date"):
            if col not in existing:
                self._conn.execute(f"ALTER TABLE document_metadata ADD COLUMN {col} TEXT")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_expiry ON document_metadata(namespace, expiry_date)"
        )
        self._conn.commit()

    async def reset(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._reset_sync)

    def _reset_sync(self) -> None:
        self._conn.executescript(
            "DROP TABLE IF EXISTS document_metadata; DROP TABLE IF EXISTS document_facet;"
        )
        self._conn.commit()
        self._init_schema()

    async def delete_document(self, document_id: str) -> None:
        """Drop one document's metadata + facet rows (soft-deleted upstream)."""
        async with self._lock:
            await asyncio.to_thread(self._delete_document_sync, document_id)

    def _delete_document_sync(self, document_id: str) -> None:
        self._conn.execute(
            "DELETE FROM document_metadata WHERE document_id = ?", (document_id,)
        )
        self._conn.execute(
            "DELETE FROM document_facet WHERE document_id = ?", (document_id,)
        )
        self._conn.commit()

    async def put(
        self,
        meta: DocumentMetadata,
        *,
        namespace: str,
        source_key: str,
        facets: dict[str, Any] | None = None,
    ) -> None:
        # Default to the standard DocumentMetadata payload; the indexer passes the
        # merged dict (standard facets + authoritative Mongo facets).
        merged = facets if facets is not None else dict(meta.filterable_payload())
        async with self._lock:
            await asyncio.to_thread(self._put_sync, meta, namespace, source_key, merged)

    def _put_sync(
        self, meta: DocumentMetadata, namespace: str, source_key: str, facets: dict[str, Any]
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO document_metadata "
            "(document_id, namespace, source_key, payload, facets, effective_date, expiry_date) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                meta.document_id,
                namespace,
                source_key,
                meta.model_dump_json(),
                json.dumps(facets, default=str),
                meta.effective_date.isoformat() if meta.effective_date else None,
                meta.expiry_date.isoformat() if meta.expiry_date else None,
            ),
        )
        self._conn.execute(
            "DELETE FROM document_facet WHERE document_id = ?", (meta.document_id,)
        )
        rows = [(namespace, meta.document_id, k, v) for k, v in _flatten_facets(facets)]
        if rows:
            self._conn.executemany(
                "INSERT INTO document_facet (namespace, document_id, key, value) "
                "VALUES (?,?,?,?)",
                rows,
            )
        self._conn.commit()

    async def get(self, document_id: str) -> DocumentMetadata | None:
        async with self._lock:
            return await asyncio.to_thread(self._get_sync, document_id)

    def _get_sync(self, document_id: str) -> DocumentMetadata | None:
        row = self._conn.execute(
            "SELECT payload FROM document_metadata WHERE document_id = ?", (document_id,)
        ).fetchone()
        return DocumentMetadata.model_validate_json(row["payload"]) if row else None

    async def facets(
        self,
        namespace: str = "default",
        *,
        filters: FacetFilters | None = None,
        per_facet_limit: int = 200,
    ) -> dict[str, list[str]]:
        """Distinct values per facet (most common first) — powers filter dropdowns.

        When ``filters`` are supplied the result is CASCADED (interdependent): each
        facet's values are restricted to those that co-occur on documents matching the
        OTHER active filters. So selecting ``status=Draft`` narrows object_code /
        business_object / category to their Draft-only values. A facet is never narrowed
        by a filter on itself, so its own options stay switchable.
        """
        async with self._lock:
            return await asyncio.to_thread(
                self._facets_sync, namespace, per_facet_limit, filters or {}
            )

    def _facets_sync(
        self, namespace: str, limit: int, filters: FacetFilters
    ) -> dict[str, list[str]]:
        active = {k: vals for k, v in filters.items() if (vals := accepted_values(v))}
        keys = [
            r["key"]
            for r in self._conn.execute(
                "SELECT DISTINCT key FROM document_facet WHERE namespace = ? ORDER BY key",
                (namespace,),
            ).fetchall()
        ]
        out: dict[str, list[str]] = {}
        for key in keys:
            where = ["namespace = ?", "key = ?"]
            params: list[Any] = [namespace, key]
            for fk, fv in active.items():
                if fk == key:
                    continue  # don't narrow a facet by its own selection
                where.append(
                    "document_id IN (SELECT document_id FROM document_facet "
                    f"WHERE namespace = ? AND key = ? AND value IN ({','.join('?' * len(fv))}))"
                )
                params.extend([namespace, fk, *fv])
            rows = self._conn.execute(
                f"SELECT value, COUNT(*) AS n FROM document_facet "
                f"WHERE {' AND '.join(where)} GROUP BY value ORDER BY n DESC, value",
                params,
            ).fetchall()
            out[key] = [r["value"] for r in rows[:limit]]
        return out

    async def facet_values(
        self,
        key: str,
        namespace: str = "default",
        *,
        filters: FacetFilters | None = None,
    ) -> list[str]:
        """EVERY value of one facet, unpaginated.

        ``facets()`` returns the most common 200 values per key, which is right for
        a filter dropdown. This is for callers that need the complete set -- the
        ACL hand-off, where a missing value makes documents invisible rather than
        merely unlisted.
        """
        async with self._lock:
            return await asyncio.to_thread(
                self._facet_values_sync, key, namespace, filters or {}
            )

    def _facet_values_sync(
        self, key: str, namespace: str, filters: FacetFilters
    ) -> list[str]:
        where = ["namespace = ?", "key = ?"]
        params: list[Any] = [namespace, key]
        for fk, raw in filters.items():
            if fk == key or not (vals := accepted_values(raw)):
                continue
            where.append(
                "document_id IN (SELECT document_id FROM document_facet "
                f"WHERE namespace = ? AND key = ? AND value IN ({','.join('?' * len(vals))}))"
            )
            params.extend([namespace, fk, *vals])
        rows = self._conn.execute(
            f"SELECT DISTINCT value FROM document_facet WHERE {' AND '.join(where)} "
            "ORDER BY value",
            params,
        ).fetchall()
        return [r["value"] for r in rows]

    async def match_facet_values(
        self, candidates: list[str], namespace: str = "default"
    ) -> dict[str, str]:
        """Resolve candidate tokens against the FULL facet vocabulary (case-insensitive).

        ``facets()`` caps each key at the 200 most common values for the UI dropdowns, so
        a rare object code (there are ~600, most appearing once) is invisible to any logic
        that scans that capped list. This is an indexed point query over every value, so a
        code the user pastes verbatim always resolves — however rare. Returns {facet_key:
        canonical_value}; on a value that belongs to two keys, the rarer key wins (a code
        is more specific than a category).
        """
        cands = [c.strip() for c in candidates if c and len(c.strip()) >= 2]
        if not cands:
            return {}
        async with self._lock:
            return await asyncio.to_thread(self._match_facet_values_sync, cands, namespace)

    def _match_facet_values_sync(self, cands: list[str], namespace: str) -> dict[str, str]:
        placeholders = ",".join("?" for _ in cands)
        rows = self._conn.execute(
            f"SELECT key, value, COUNT(*) n FROM document_facet "
            f"WHERE namespace = ? AND lower(value) IN ({placeholders}) "
            f"GROUP BY key, value",
            (namespace, *(c.lower() for c in cands)),
        ).fetchall()
        # Each matched value picks its rarest (most specific) facet key.
        by_value: dict[str, tuple[str, str, int]] = {}
        for r in rows:
            lv = str(r["value"]).lower()
            if lv not in by_value or r["n"] < by_value[lv][2]:
                by_value[lv] = (r["key"], str(r["value"]), r["n"])
        out: dict[str, str] = {}
        for key, canon, _n in by_value.values():
            out.setdefault(key, canon)  # one value per facet key
        return out

    async def list_documents(
        self,
        namespace: str = "default",
        *,
        filters: FacetFilters | None = None,
        expiry_from: str | None = None,
        expiry_to: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[int, list[dict[str, Any]]]:
        """Browse documents by facet (e.g. status=Draft) and/or an expiry date range
        (ISO ``YYYY-MM-DD`` bounds), paginated. Returns ``(total_matching, page_rows)``
        so the UI can render Next/Prev controls."""
        async with self._lock:
            return await asyncio.to_thread(
                self._list_sync, namespace, filters or {}, expiry_from, expiry_to, limit, offset
            )

    def _list_sync(
        self,
        namespace: str,
        filters: FacetFilters,
        expiry_from: str | None,
        expiry_to: str | None,
        limit: int,
        offset: int,
    ) -> tuple[int, list[dict[str, Any]]]:
        where = ["namespace = ?"]
        params: list[Any] = [namespace]
        for key, value in filters.items():
            if not (values := accepted_values(value)):
                continue
            where.append(
                "document_id IN (SELECT document_id FROM document_facet "
                f"WHERE namespace = ? AND key = ? AND value IN ({','.join('?' * len(values))}))"
            )
            params.extend([namespace, key, *values])
        if expiry_from and expiry_to:
            where.append("expiry_date IS NOT NULL AND expiry_date BETWEEN ? AND ?")
            params.extend([expiry_from, expiry_to])
        where_sql = " AND ".join(where)
        order = "expiry_date, source_key" if (expiry_from and expiry_to) else "source_key"

        total = self._conn.execute(
            f"SELECT COUNT(*) AS n FROM document_metadata WHERE {where_sql}", params
        ).fetchone()["n"]
        rows = self._conn.execute(
            f"SELECT document_id, source_key, payload, facets, expiry_date FROM document_metadata "
            f"WHERE {where_sql} ORDER BY {order} LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()

        docs: list[dict[str, Any]] = []
        for r in rows:
            facets = json.loads(r["facets"] or "{}")
            try:
                title = json.loads(r["payload"]).get("title")
            except (json.JSONDecodeError, TypeError):
                title = None
            docs.append(
                {
                    "document_id": r["document_id"],
                    "source_key": r["source_key"],
                    "title": title,
                    "expiry_date": r["expiry_date"],
                    "facets": {k: v for k, v in facets.items() if isinstance(v, str)},
                }
            )
        return total, docs

    async def search_documents(
        self, query: str, namespace: str = "default", *, limit: int = 6
    ) -> list[dict[str, Any]]:
        """Find documents by matching the question against their TITLE + metadata
        facet values (the authoritative names/types that come from MongoDB, which the
        body-text index doesn't cover). Returns the best matches (title + facets), so a
        question that names a document or type ("AMS form", "EHO audit", a supplier, an
        object code) surfaces the right document even when its body text is generic."""
        async with self._lock:
            return await asyncio.to_thread(self._search_documents_sync, query, namespace, limit)

    def _search_documents_sync(
        self, query: str, namespace: str, limit: int
    ) -> list[dict[str, Any]]:
        terms = {
            t for t in re.findall(r"[a-z0-9]{3,}", query.lower()) if t not in _TITLE_SEARCH_STOP
        }
        if not terms:
            return []
        rows = self._conn.execute(
            "SELECT document_id, source_key, payload, facets FROM document_metadata "
            "WHERE namespace = ?",
            (namespace,),
        ).fetchall()
        scored: list[tuple[int, dict[str, Any]]] = []
        for r in rows:
            try:
                title = json.loads(r["payload"] or "{}").get("title") or ""
            except (json.JSONDecodeError, TypeError):
                title = ""
            try:
                facet_map = json.loads(r["facets"] or "{}")
            except (json.JSONDecodeError, TypeError):
                facet_map = {}
            values = " ".join(str(v) for v in facet_map.values() if isinstance(v, str))
            hay = f"{title} {r['source_key'] or ''} {values}".lower()
            score = sum(1 for t in terms if t in hay)
            if score:
                scored.append((
                    score,
                    {
                        "document_id": r["document_id"],
                        "source_key": r["source_key"],
                        "title": title or None,
                        "facets": {k: v for k, v in facet_map.items() if isinstance(v, str)},
                    },
                ))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:limit]]

    async def describe(
        self, document_ids: list[str], namespace: str = "default"
    ) -> dict[str, dict[str, Any]]:
        """Title + string facets for the given documents — used to label grounded
        passages with what each document IS (name/type/status/supplier/dates)."""
        ids = [d for d in document_ids if d]
        if not ids:
            return {}
        async with self._lock:
            return await asyncio.to_thread(self._describe_sync, ids)

    def _describe_sync(self, document_ids: list[str]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        qmarks = ",".join("?" for _ in document_ids)
        rows = self._conn.execute(
            f"SELECT document_id, payload, facets FROM document_metadata "
            f"WHERE document_id IN ({qmarks})",
            document_ids,
        ).fetchall()
        for r in rows:
            try:
                title = json.loads(r["payload"] or "{}").get("title") or ""
            except (json.JSONDecodeError, TypeError):
                title = ""
            try:
                facet_map = json.loads(r["facets"] or "{}")
            except (json.JSONDecodeError, TypeError):
                facet_map = {}
            out[r["document_id"]] = {
                "title": title or None,
                "facets": {k: v for k, v in facet_map.items() if isinstance(v, str)},
            }
        return out

    async def backfill_expiry(self, parse_fn: Any) -> int:
        """Re-derive expiry_date for existing rows from their title + source_key
        (no re-download). ``parse_fn(text) -> date|None``. Returns rows updated."""
        async with self._lock:
            return await asyncio.to_thread(self._backfill_sync, parse_fn)

    def _backfill_sync(self, parse_fn: Any) -> int:
        rows = self._conn.execute(
            "SELECT document_id, source_key, payload FROM document_metadata"
        ).fetchall()
        updated = 0
        for r in rows:
            try:
                title = json.loads(r["payload"]).get("title") or ""
            except (json.JSONDecodeError, TypeError):
                title = ""
            exp = parse_fn(f"{title} {r['source_key']}")
            if exp is not None:
                self._conn.execute(
                    "UPDATE document_metadata SET expiry_date = ? WHERE document_id = ?",
                    (exp.isoformat(), r["document_id"]),
                )
                updated += 1
        self._conn.commit()
        return updated

    async def apply_facets(
        self, document_id: str, facets: Mapping[str, str | None]
    ) -> bool:
        """Merge extra facets into an existing row (no re-download, no re-embed).

        Used by enrichment passes — e.g. writing the supplier name extracted from a
        document onto every document sharing its supplier code. Existing keys are
        overwritten and a ``None`` value REMOVES the facet, which is how a value that
        later proves invalid gets retracted. A facet key that is also a
        ``DocumentMetadata`` field is written to the canonical payload too, so
        ``/v1/metadata/{id}`` and the facet dict never disagree. Returns True when the
        document existed.
        """
        if not facets:
            return False
        async with self._lock:
            return await asyncio.to_thread(self._apply_facets_sync, document_id, facets)

    def _apply_facets_sync(self, document_id: str, facets: Mapping[str, str | None]) -> bool:
        row = self._conn.execute(
            "SELECT namespace, facets, payload FROM document_metadata WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        if row is None:
            return False
        try:
            merged = json.loads(row["facets"]) or {}
        except (json.JSONDecodeError, TypeError):
            merged = {}
        for key, value in facets.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value
        try:
            payload = json.loads(row["payload"]) or {}
        except (json.JSONDecodeError, TypeError):
            payload = {}
        for key, value in facets.items():
            if key in DocumentMetadata.model_fields:
                payload[key] = value  # None clears the canonical field too
        self._conn.execute(
            "UPDATE document_metadata SET facets = ?, payload = ? WHERE document_id = ?",
            (json.dumps(merged, default=str), json.dumps(payload, default=str), document_id),
        )
        # Re-flatten only the changed keys so the filter dropdowns pick them up.
        self._conn.executemany(
            "DELETE FROM document_facet WHERE document_id = ? AND key = ?",
            [(document_id, k) for k in facets],
        )
        rows = [
            (row["namespace"], document_id, k, v)
            for k, v in _flatten_facets({k: v for k, v in facets.items() if v is not None})
        ]
        if rows:
            self._conn.executemany(
                "INSERT INTO document_facet (namespace, document_id, key, value) VALUES (?,?,?,?)",
                rows,
            )
        self._conn.commit()
        return True

    async def indexed_ids(self, namespace: str = "default") -> set[str]:
        """Every document_id already in this index -- what a resumed run may skip."""
        async with self._lock:
            return await asyncio.to_thread(self._indexed_ids_sync, namespace)

    def _indexed_ids_sync(self, namespace: str) -> set[str]:
        rows = self._conn.execute(
            "SELECT document_id FROM document_metadata WHERE namespace = ?", (namespace,)
        ).fetchall()
        return {r["document_id"] for r in rows}

    async def count(self) -> int:
        async with self._lock:
            return await asyncio.to_thread(
                lambda: self._conn.execute(
                    "SELECT COUNT(*) AS n FROM document_metadata"
                ).fetchone()["n"]
            )

    async def all_documents(self) -> list[tuple[str, str]]:
        """Every indexed ``(document_id, source_key)`` — used to reconcile the index
        against the system of record (e.g. purge documents soft-deleted in Mongo)."""
        async with self._lock:
            return await asyncio.to_thread(self._all_documents_sync)

    def _all_documents_sync(self) -> list[tuple[str, str]]:
        rows = self._conn.execute(
            "SELECT document_id, source_key FROM document_metadata"
        ).fetchall()
        return [(r["document_id"], r["source_key"]) for r in rows]

    async def aclose(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._conn.close)
