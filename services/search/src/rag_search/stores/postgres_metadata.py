"""Canonical metadata store on Postgres — mirrors ``MetadataStore`` (SQLite).

Same two-table shape as the SQLite original (a canonical row per document plus a
denormalised ``document_facet`` table powering the filter dropdowns), with
``payload``/``facets`` held as ``jsonb`` instead of TEXT + ``json_extract``. The
tables themselves are created by ``PgDatabase`` so there is one lazy-init path
for the whole backend.

Writes that touch both tables run through ``PgDatabase.run`` as a single
callable, so the canonical row and its facet rows commit or roll back together —
the SQLite version issued them as separate statements.

Two queries get materially better than the SQLite versions:

* ``facets()`` ran one query per facet key. The cascade rule ("a facet is never
  narrowed by a filter on itself") means there are only ``len(active_filters) + 1``
  distinct document sets to count over, not one per key — so with no filters
  applied this is now a single grouped query.
* ``search_documents()`` read *every* metadata row into Python and scored it
  there. The candidate filter now runs in SQL and only matching rows come back;
  the scoring itself stays in Python so ranking is byte-identical to SQLite.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from rag_core.models.metadata import DocumentMetadata
from rag_search.facet_filters import FacetFilters, accepted_values
from rag_search.metadata_store import _TITLE_SEARCH_STOP, _flatten_facets
from rag_search.stores.postgres import PgDatabase, scrub_json_text, scrub_nuls

# Lowercased haystack for title search: document title + filename + string facet
# values. Mirrors what ``MetadataStore._search_documents_sync`` builds in Python.
_HAY_SQL = """lower(
    coalesce(m.payload->>'title', '') || ' ' ||
    coalesce(m.source_key, '') || ' ' ||
    coalesce((
        SELECT string_agg(e.value #>> '{}', ' ')
        FROM jsonb_each(m.facets) e
        WHERE jsonb_typeof(e.value) = 'string'
    ), '')
)"""


class PgMetadataStore:
    def __init__(self, db: PgDatabase, *, namespace: str = "default") -> None:
        self._db = db
        self._default_namespace = namespace

    # ---- writes -----------------------------------------------------------

    async def reset(self) -> None:
        await self._db.execute("TRUNCATE TABLE document_metadata, document_facet")

    async def delete_document(self, document_id: str) -> None:
        def _op(cur: Any) -> None:
            cur.execute("DELETE FROM document_metadata WHERE document_id = %s", (document_id,))
            cur.execute("DELETE FROM document_facet WHERE document_id = %s", (document_id,))

        await self._db.run(_op)

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
        merged = scrub_nuls(facets if facets is not None else dict(meta.filterable_payload()))
        rows = [(namespace, meta.document_id, k, v) for k, v in _flatten_facets(merged)]
        params = (
            meta.document_id,
            namespace,
            scrub_nuls(source_key),
            # Already-serialised JSON, so a NUL arrives as the escape .
            scrub_json_text(meta.model_dump_json()),
            json.dumps(merged, default=str),
            meta.effective_date.isoformat() if meta.effective_date else None,
            meta.expiry_date.isoformat() if meta.expiry_date else None,
        )

        def _op(cur: Any) -> None:
            cur.execute(
                "INSERT INTO document_metadata (document_id, namespace, source_key, payload, "
                "facets, effective_date, expiry_date) "
                "VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s) "
                "ON CONFLICT (document_id) DO UPDATE SET "
                "namespace = EXCLUDED.namespace, source_key = EXCLUDED.source_key, "
                "payload = EXCLUDED.payload, facets = EXCLUDED.facets, "
                "effective_date = EXCLUDED.effective_date, expiry_date = EXCLUDED.expiry_date",
                params,
            )
            cur.execute("DELETE FROM document_facet WHERE document_id = %s", (meta.document_id,))
            if rows:
                cur.executemany(
                    "INSERT INTO document_facet (namespace, document_id, key, value) "
                    "VALUES (%s, %s, %s, %s)",
                    rows,
                )

        await self._db.run(_op)

    async def apply_facets(self, document_id: str, facets: Mapping[str, str | None]) -> bool:
        """Merge extra facets into an existing row — see ``MetadataStore.apply_facets``."""
        if not facets:
            return False
        setting = scrub_nuls({k: v for k, v in facets.items() if v is not None})
        clearing = [k for k, v in facets.items() if v is None]
        # A facet key that is also a DocumentMetadata field is written to the canonical
        # payload too; None writes an explicit null there (clearing the field) rather
        # than removing the key, matching the SQLite behaviour.
        payload_patch = scrub_nuls(
            {k: v for k, v in facets.items() if k in DocumentMetadata.model_fields}
        )
        changed_keys = list(facets)

        def _op(cur: Any) -> bool:
            cur.execute(
                "UPDATE document_metadata SET "
                "facets = (facets || %s::jsonb) - %s::text[], "
                "payload = payload || %s::jsonb "
                "WHERE document_id = %s RETURNING namespace",
                (
                    json.dumps(setting, default=str),
                    clearing,
                    json.dumps(payload_patch, default=str),
                    document_id,
                ),
            )
            row = cur.fetchone()
            if row is None:
                return False
            namespace = row["namespace"]
            # Re-flatten only the changed keys so the filter dropdowns pick them up.
            cur.executemany(
                "DELETE FROM document_facet WHERE document_id = %s AND key = %s",
                [(document_id, k) for k in changed_keys],
            )
            rows = [(namespace, document_id, k, v) for k, v in _flatten_facets(dict(setting))]
            if rows:
                cur.executemany(
                    "INSERT INTO document_facet (namespace, document_id, key, value) "
                    "VALUES (%s, %s, %s, %s)",
                    rows,
                )
            return True

        return await self._db.run(_op)

    async def backfill_expiry(self, parse_fn: Any) -> int:
        """Re-derive expiry_date from title + source_key. ``parse_fn(text) -> date|None``."""
        rows = await self._db.fetch(
            "SELECT document_id, source_key, payload FROM document_metadata"
        )
        updates = []
        for r in rows:
            title = (r["payload"] or {}).get("title") or ""
            exp = parse_fn(f"{title} {r['source_key']}")
            if exp is not None:
                updates.append((exp.isoformat(), r["document_id"]))
        if updates:
            await self._db.executemany(
                "UPDATE document_metadata SET expiry_date = %s WHERE document_id = %s", updates
            )
        return len(updates)

    # ---- reads ------------------------------------------------------------

    async def get(self, document_id: str) -> DocumentMetadata | None:
        row = await self._db.fetchone(
            "SELECT payload FROM document_metadata WHERE document_id = %s", (document_id,)
        )
        return DocumentMetadata.model_validate(row["payload"]) if row else None

    async def facets(
        self,
        namespace: str = "default",
        *,
        filters: FacetFilters | None = None,
        per_facet_limit: int = 200,
    ) -> dict[str, list[str]]:
        """Distinct values per facet (most common first) — powers filter dropdowns.

        CASCADED when ``filters`` are supplied: each facet's values are restricted to
        those co-occurring on documents matching the OTHER active filters, so a facet
        is never narrowed by a filter on itself.
        """
        active = {k: vals for k, v in (filters or {}).items() if (vals := accepted_values(v))}
        key_rows = await self._db.fetch(
            "SELECT DISTINCT key FROM document_facet WHERE namespace = %s ORDER BY key",
            (namespace,),
        )
        keys = [r["key"] for r in key_rows]
        out: dict[str, list[str]] = {k: [] for k in keys}
        if not keys:
            return out

        # The cascade rule means the only distinct document sets are `active` and
        # `active - {k}` for each filtered key k. Group the keys by which applies and
        # run one grouped query per group (exactly one when nothing is filtered).
        groups: dict[tuple[str, ...], list[str]] = {}
        for key in keys:
            applied = tuple(sorted(k for k in active if k != key))
            groups.setdefault(applied, []).append(key)

        for applied, group_keys in groups.items():
            where = ["namespace = %s", "key = ANY(%s)"]
            params: list[Any] = [namespace, group_keys]
            for fk in applied:
                where.append(
                    "document_id IN (SELECT document_id FROM document_facet "
                    "WHERE namespace = %s AND key = %s AND value = ANY(%s))"
                )
                params.extend([namespace, fk, list(active[fk])])
            rows = await self._db.fetch(
                f"SELECT key, value, COUNT(*) AS n FROM document_facet "
                f"WHERE {' AND '.join(where)} GROUP BY key, value ORDER BY n DESC, value",
                params,
            )
            for r in rows:
                bucket = out[r["key"]]
                if len(bucket) < per_facet_limit:
                    bucket.append(r["value"])
        return out

    async def facet_values(
        self,
        key: str,
        namespace: str = "default",
        *,
        filters: FacetFilters | None = None,
    ) -> list[str]:
        """EVERY value of one facet, unpaginated -- see MetadataStore.facet_values."""
        where = ["namespace = %s", "key = %s"]
        params: list[Any] = [namespace, key]
        for fk, raw in (filters or {}).items():
            if fk == key or not (vals := accepted_values(raw)):
                continue
            where.append(
                "document_id IN (SELECT document_id FROM document_facet "
                "WHERE namespace = %s AND key = %s AND value = ANY(%s))"
            )
            params.extend([namespace, fk, vals])
        rows = await self._db.fetch(
            f"SELECT DISTINCT value FROM document_facet WHERE {' AND '.join(where)} "
            "ORDER BY value",
            params,
        )
        return [str(r["value"]) for r in rows]

    async def match_facet_values(
        self, candidates: list[str], namespace: str = "default"
    ) -> dict[str, str]:
        """Resolve candidate tokens against the FULL facet vocabulary (case-insensitive)."""
        cands = [c.strip() for c in candidates if c and len(c.strip()) >= 2]
        if not cands:
            return {}
        rows = await self._db.fetch(
            "SELECT key, value, COUNT(*) AS n FROM document_facet "
            "WHERE namespace = %s AND lower(value) = ANY(%s) GROUP BY key, value",
            (namespace, [c.lower() for c in cands]),
        )
        # Each matched value picks its rarest (most specific) facet key.
        by_value: dict[str, tuple[str, str, int]] = {}
        for r in rows:
            lv = str(r["value"]).lower()
            if lv not in by_value or r["n"] < by_value[lv][2]:
                by_value[lv] = (r["key"], str(r["value"]), int(r["n"]))
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
        """Browse documents by facet and/or expiry range, paginated. Returns
        ``(total_matching, page_rows)``."""
        where = ["namespace = %s"]
        params: list[Any] = [namespace]
        for key, value in (filters or {}).items():
            if not (values := accepted_values(value)):
                continue
            where.append(
                "document_id IN (SELECT document_id FROM document_facet "
                "WHERE namespace = %s AND key = %s AND value = ANY(%s))"
            )
            params.extend([namespace, key, values])
        if expiry_from and expiry_to:
            where.append("expiry_date IS NOT NULL AND expiry_date BETWEEN %s AND %s")
            params.extend([expiry_from, expiry_to])
        where_sql = " AND ".join(where)
        order = "expiry_date, source_key" if (expiry_from and expiry_to) else "source_key"

        total_row = await self._db.fetchone(
            f"SELECT COUNT(*) AS n FROM document_metadata WHERE {where_sql}", params
        )
        total = int((total_row or {}).get("n", 0))
        rows = await self._db.fetch(
            f"SELECT document_id, source_key, payload, facets, expiry_date "
            f"FROM document_metadata WHERE {where_sql} ORDER BY {order} LIMIT %s OFFSET %s",
            [*params, limit, offset],
        )
        docs = [
            {
                "document_id": r["document_id"],
                "source_key": r["source_key"],
                "title": (r["payload"] or {}).get("title") or None,
                "expiry_date": r["expiry_date"],
                "facets": {k: v for k, v in (r["facets"] or {}).items() if isinstance(v, str)},
            }
            for r in rows
        ]
        return total, docs

    async def search_documents(
        self, query: str, namespace: str = "default", *, limit: int = 6
    ) -> list[dict[str, Any]]:
        """Find documents by matching the question against TITLE + filename + facet
        values — see ``MetadataStore.search_documents``."""
        terms = {
            t for t in re.findall(r"[a-z0-9]{3,}", query.lower()) if t not in _TITLE_SEARCH_STOP
        }
        if not terms:
            return []
        term_list = sorted(terms)
        # Candidate filter in SQL (at least one term present); scoring stays in Python
        # so the ranking matches the SQLite backend exactly.
        like_sql = " OR ".join("s.hay LIKE %s" for _ in term_list)
        rows = await self._db.fetch(
            f"SELECT * FROM ("
            f"  SELECT m.document_id, m.source_key, m.payload, m.facets, {_HAY_SQL} AS hay"
            f"  FROM document_metadata m WHERE m.namespace = %s"
            f") s WHERE {like_sql}",
            (namespace, *(f"%{t}%" for t in term_list)),
        )
        scored: list[tuple[int, dict[str, Any]]] = []
        for r in rows:
            hay = r["hay"] or ""
            score = sum(1 for t in terms if t in hay)
            if score:
                title = (r["payload"] or {}).get("title") or ""
                facet_map = r["facets"] or {}
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
        passages with what each document IS."""
        ids = [d for d in document_ids if d]
        if not ids:
            return {}
        rows = await self._db.fetch(
            "SELECT document_id, payload, facets FROM document_metadata "
            "WHERE document_id = ANY(%s)",
            (ids,),
        )
        return {
            r["document_id"]: {
                "title": ((r["payload"] or {}).get("title") or "") or None,
                "facets": {k: v for k, v in (r["facets"] or {}).items() if isinstance(v, str)},
            }
            for r in rows
        }

    async def indexed_ids(self, namespace: str = "default") -> set[str]:
        """Every document_id already in this index -- what a resumed run may skip."""
        rows = await self._db.fetch(
            "SELECT document_id FROM document_metadata WHERE namespace = %s", (namespace,)
        )
        return {str(r["document_id"]) for r in rows}

    async def count(self) -> int:
        row = await self._db.fetchone("SELECT COUNT(*) AS n FROM document_metadata")
        return int((row or {}).get("n", 0))

    async def all_documents(self) -> list[tuple[str, str]]:
        """Every indexed ``(document_id, source_key)`` — used to reconcile the index
        against the system of record."""
        rows = await self._db.fetch("SELECT document_id, source_key FROM document_metadata")
        return [(r["document_id"], r["source_key"]) for r in rows]

    async def aclose(self) -> None:
        await self._db.aclose()


__all__ = ["PgMetadataStore"]
