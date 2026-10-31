"""Canonical metadata store on MongoDB -- mirrors ``MetadataStore`` / ``PgMetadataStore``.

One ``kind="document"`` record per document, in the same collection as that
tenant's chunks (see ``stores/mongo.py``). The SQL backends' denormalised
``document_facet`` table becomes an embedded ``facet_pairs`` array of
``{k, v, lv}`` (``lv`` = lowercased value), produced by the same
``_flatten_facets`` so the filter dropdowns see identical vocabularies.

Search over titles keeps its scoring in Python exactly as the SQL backends do;
only the candidate filter runs in the database (against a precomputed ``hay``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from pymongo import UpdateOne

from rag_core.models.metadata import DocumentMetadata
from rag_search.facet_filters import FacetFilters, accepted_values
from rag_search.metadata_store import _TITLE_SEARCH_STOP, _flatten_facets
from rag_search.stores.mongo import DOCUMENT, MongoIndexDatabase


def _meta_id(document_id: str) -> str:
    # Chunk ids are UUIDs; the prefix keeps document records out of their keyspace.
    return f"meta::{document_id}"


def _jsonable(value: Any) -> Any:
    """Plain JSON types only -- BSON cannot encode ``datetime.date``."""
    return json.loads(json.dumps(value, default=str))


def _pairs(facets: Mapping[str, Any]) -> list[dict[str, str]]:
    return [{"k": k, "v": v, "lv": v.lower()} for k, v in _flatten_facets(dict(facets))]


def _string_facets(facets: Mapping[str, Any] | None) -> dict[str, str]:
    return {k: v for k, v in (facets or {}).items() if isinstance(v, str)}


def _hay(payload: Mapping[str, Any], source_key: str, facets: Mapping[str, Any]) -> str:
    """Title + filename + string facet values, lowercased -- what title search reads."""
    parts = [str(payload.get("title") or ""), source_key or ""]
    parts += [v for v in facets.values() if isinstance(v, str)]
    return " ".join(parts).lower()


def document_record(
    *,
    document_id: str,
    namespace: str,
    source_key: str,
    payload: Mapping[str, Any],
    facets: Mapping[str, Any],
    pairs: list[tuple[str, str]] | None = None,
    effective_date: str | None = None,
    expiry_date: str | None = None,
) -> dict[str, Any]:
    """The stored shape of one document's metadata (``_id`` included).

    ``pairs`` overrides the facet rows derived from ``facets`` -- the migration
    copies an existing index's rows verbatim so its dropdowns stay identical.
    """
    facets = _jsonable(dict(facets))
    payload = _jsonable(dict(payload))
    facet_pairs = (
        [{"k": k, "v": v, "lv": v.lower()} for k, v in pairs]
        if pairs is not None
        else _pairs(facets)
    )
    return {
        "_id": _meta_id(document_id),
        "kind": DOCUMENT,
        "document_id": document_id,
        "namespace": namespace,
        "source_key": source_key,
        "payload": payload,
        "facets": facets,
        "facet_pairs": facet_pairs,
        "hay": _hay(payload, source_key, facets),
        "effective_date": effective_date,
        "expiry_date": expiry_date,
    }


def _facet_match(namespace: str, filters: Mapping[str, list[str]]) -> dict[str, Any]:
    """Documents in ``namespace`` carrying every filtered facet (any-of its values)."""
    match: dict[str, Any] = {"kind": DOCUMENT, "namespace": namespace}
    clauses = [
        {"facet_pairs": {"$elemMatch": {"k": key, "v": {"$in": list(values)}}}}
        for key, values in filters.items()
        if values
    ]
    if clauses:
        match["$and"] = clauses
    return match


class MongoMetadataStore:
    def __init__(self, db: MongoIndexDatabase, *, namespace: str = "default") -> None:
        self._db = db
        self._default_namespace = namespace

    # ---- writes -----------------------------------------------------------

    async def reset(self) -> None:
        await self._db.ensure_schema()
        await self._db.run(lambda coll: coll.delete_many({"kind": DOCUMENT}))

    async def delete_document(self, document_id: str) -> None:
        await self._db.run(lambda coll: coll.delete_one({"_id": _meta_id(document_id)}))

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
        record = document_record(
            document_id=meta.document_id,
            namespace=namespace,
            source_key=source_key,
            payload=json.loads(meta.model_dump_json()),
            facets=facets if facets is not None else dict(meta.filterable_payload()),
            effective_date=meta.effective_date.isoformat() if meta.effective_date else None,
            expiry_date=meta.expiry_date.isoformat() if meta.expiry_date else None,
        )
        await self._db.run(
            lambda coll: coll.replace_one({"_id": record["_id"]}, record, upsert=True)
        )

    async def apply_facets(self, document_id: str, facets: Mapping[str, str | None]) -> bool:
        """Merge extra facets into an existing record -- see ``MetadataStore.apply_facets``."""
        if not facets:
            return False
        setting = _jsonable({k: v for k, v in facets.items() if v is not None})
        clearing = {k for k, v in facets.items() if v is None}
        # A facet key that is also a DocumentMetadata field is written to the canonical
        # payload too; None writes an explicit null there (clearing the field).
        payload_patch = _jsonable(
            {k: v for k, v in facets.items() if k in DocumentMetadata.model_fields}
        )
        changed = set(facets)

        def _op(coll: Any) -> bool:
            doc = coll.find_one({"_id": _meta_id(document_id)})
            if doc is None:
                return False
            merged = {k: v for k, v in (doc.get("facets") or {}).items() if k not in clearing}
            merged.update(setting)
            payload = {**(doc.get("payload") or {}), **payload_patch}
            # Re-flatten only the changed keys, as the SQL backends do.
            pairs = [p for p in doc.get("facet_pairs") or [] if p.get("k") not in changed]
            pairs += _pairs(setting)
            coll.update_one(
                {"_id": doc["_id"]},
                {
                    "$set": {
                        "facets": merged,
                        "payload": payload,
                        "facet_pairs": pairs,
                        "hay": _hay(payload, doc.get("source_key") or "", merged),
                    }
                },
            )
            return True

        return await self._db.run(_op)

    async def backfill_expiry(self, parse_fn: Any) -> int:
        """Re-derive expiry_date from title + source_key. ``parse_fn(text) -> date|None``."""
        rows = await self._db.run(
            lambda coll: list(coll.find({"kind": DOCUMENT}, {"source_key": 1, "payload.title": 1}))
        )
        updates = []
        for r in rows:
            title = (r.get("payload") or {}).get("title") or ""
            exp = parse_fn(f"{title} {r.get('source_key') or ''}")
            if exp is not None:
                updates.append(
                    UpdateOne({"_id": r["_id"]}, {"$set": {"expiry_date": exp.isoformat()}})
                )
        if updates:
            await self._db.run(lambda coll: coll.bulk_write(updates, ordered=False))
        return len(updates)

    # ---- reads ------------------------------------------------------------

    async def get(self, document_id: str) -> DocumentMetadata | None:
        doc = await self._db.run(
            lambda coll: coll.find_one({"_id": _meta_id(document_id)}, {"payload": 1})
        )
        return DocumentMetadata.model_validate(doc["payload"]) if doc else None

    async def facets(
        self,
        namespace: str = "default",
        *,
        filters: FacetFilters | None = None,
        per_facet_limit: int = 200,
    ) -> dict[str, list[str]]:
        """Distinct values per facet (most common first) -- powers filter dropdowns.

        CASCADED when ``filters`` are supplied: each facet's values are restricted to
        those co-occurring on documents matching the OTHER active filters, so a facet
        is never narrowed by a filter on itself.
        """
        active = {k: vals for k, v in (filters or {}).items() if (vals := accepted_values(v))}
        keys = sorted(
            await self._db.run(
                lambda coll: coll.distinct(
                    "facet_pairs.k", {"kind": DOCUMENT, "namespace": namespace}
                )
            )
        )
        out: dict[str, list[str]] = {k: [] for k in keys}
        if not keys:
            return out

        # Only `active` and `active - {k}` are distinct document sets; group the keys
        # by which applies and run one aggregation per group.
        groups: dict[tuple[str, ...], list[str]] = {}
        for key in keys:
            applied = tuple(sorted(k for k in active if k != key))
            groups.setdefault(applied, []).append(key)

        for applied, group_keys in groups.items():
            pipeline = [
                {"$match": _facet_match(namespace, {fk: active[fk] for fk in applied})},
                {"$unwind": "$facet_pairs"},
                {"$match": {"facet_pairs.k": {"$in": group_keys}}},
                {
                    "$group": {
                        "_id": {"k": "$facet_pairs.k", "v": "$facet_pairs.v"},
                        "n": {"$sum": 1},
                    }
                },
                {"$sort": {"n": -1, "_id.v": 1}},
            ]
            rows = await self._db.run(lambda coll, p=pipeline: list(coll.aggregate(p)))
            for r in rows:
                bucket = out[r["_id"]["k"]]
                if len(bucket) < per_facet_limit:
                    bucket.append(r["_id"]["v"])
        return out

    async def facet_values(
        self,
        key: str,
        namespace: str = "default",
        *,
        filters: FacetFilters | None = None,
    ) -> list[str]:
        """EVERY value of one facet, unpaginated -- see MetadataStore.facet_values."""
        others = {
            fk: vals
            for fk, raw in (filters or {}).items()
            if fk != key and (vals := accepted_values(raw))
        }
        pipeline = [
            {"$match": _facet_match(namespace, others)},
            {"$unwind": "$facet_pairs"},
            {"$match": {"facet_pairs.k": key}},
            {"$group": {"_id": "$facet_pairs.v"}},
            {"$sort": {"_id": 1}},
        ]
        rows = await self._db.run(lambda coll: list(coll.aggregate(pipeline)))
        return [str(r["_id"]) for r in rows]

    async def match_facet_values(
        self, candidates: list[str], namespace: str = "default"
    ) -> dict[str, str]:
        """Resolve candidate tokens against the FULL facet vocabulary (case-insensitive)."""
        lowered = [c.strip().lower() for c in candidates if c and len(c.strip()) >= 2]
        if not lowered:
            return {}
        pipeline = [
            {
                "$match": {
                    "kind": DOCUMENT,
                    "namespace": namespace,
                    "facet_pairs.lv": {"$in": lowered},
                }
            },
            {"$unwind": "$facet_pairs"},
            {"$match": {"facet_pairs.lv": {"$in": lowered}}},
            {"$group": {"_id": {"k": "$facet_pairs.k", "v": "$facet_pairs.v"}, "n": {"$sum": 1}}},
        ]
        rows = await self._db.run(lambda coll: list(coll.aggregate(pipeline)))
        # Each matched value picks its rarest (most specific) facet key.
        by_value: dict[str, tuple[str, str, int]] = {}
        for r in rows:
            key, value, n = r["_id"]["k"], str(r["_id"]["v"]), int(r["n"])
            lv = value.lower()
            if lv not in by_value or n < by_value[lv][2]:
                by_value[lv] = (key, value, n)
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
        active = {k: vals for k, v in (filters or {}).items() if (vals := accepted_values(v))}
        match = _facet_match(namespace, active)
        by_expiry = bool(expiry_from and expiry_to)
        if by_expiry:
            match["expiry_date"] = {"$ne": None, "$gte": expiry_from, "$lte": expiry_to}
        order = [("expiry_date", 1), ("source_key", 1)] if by_expiry else [("source_key", 1)]
        projection = {
            "document_id": 1,
            "source_key": 1,
            "payload.title": 1,
            "facets": 1,
            "expiry_date": 1,
        }

        def _op(coll: Any) -> tuple[int, list[dict[str, Any]]]:
            total = coll.count_documents(match)
            rows = list(coll.find(match, projection).sort(order).skip(offset).limit(limit))
            return total, rows

        total, rows = await self._db.run(_op)
        docs = [
            {
                "document_id": r["document_id"],
                "source_key": r.get("source_key"),
                "title": (r.get("payload") or {}).get("title") or None,
                "expiry_date": r.get("expiry_date"),
                "facets": _string_facets(r.get("facets")),
            }
            for r in rows
        ]
        return total, docs

    async def search_documents(
        self, query: str, namespace: str = "default", *, limit: int = 6
    ) -> list[dict[str, Any]]:
        """Find documents by matching the question against TITLE + filename + facet
        values -- see ``MetadataStore.search_documents``."""
        terms = {
            t for t in re.findall(r"[a-z0-9]{3,}", query.lower()) if t not in _TITLE_SEARCH_STOP
        }
        if not terms:
            return []
        match = {
            "kind": DOCUMENT,
            "namespace": namespace,
            "$or": [{"hay": {"$regex": re.escape(t)}} for t in sorted(terms)],
        }
        rows = await self._db.run(
            lambda coll: list(
                coll.find(
                    match,
                    {"document_id": 1, "source_key": 1, "payload.title": 1, "facets": 1, "hay": 1},
                )
            )
        )
        scored: list[tuple[int, dict[str, Any]]] = []
        for r in rows:
            hay = r.get("hay") or ""
            score = sum(1 for t in terms if t in hay)
            if score:
                scored.append(
                    (
                        score,
                        {
                            "document_id": r["document_id"],
                            "source_key": r.get("source_key"),
                            "title": (r.get("payload") or {}).get("title") or None,
                            "facets": _string_facets(r.get("facets")),
                        },
                    )
                )
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:limit]]

    async def describe(
        self, document_ids: list[str], namespace: str = "default"
    ) -> dict[str, dict[str, Any]]:
        """Title + string facets for the given documents -- used to label grounded
        passages with what each document IS."""
        ids = [_meta_id(d) for d in document_ids if d]
        if not ids:
            return {}
        rows = await self._db.run(
            lambda coll: list(
                coll.find(
                    {"_id": {"$in": ids}}, {"document_id": 1, "payload.title": 1, "facets": 1}
                )
            )
        )
        return {
            r["document_id"]: {
                "title": ((r.get("payload") or {}).get("title") or "") or None,
                "facets": _string_facets(r.get("facets")),
            }
            for r in rows
        }

    async def indexed_ids(self, namespace: str = "default") -> set[str]:
        """Every document_id already in this index -- what a resumed run may skip."""
        rows = await self._db.run(
            lambda coll: list(
                coll.find({"kind": DOCUMENT, "namespace": namespace}, {"document_id": 1})
            )
        )
        return {str(r["document_id"]) for r in rows}

    async def count(self) -> int:
        return await self._db.run(lambda coll: coll.count_documents({"kind": DOCUMENT}))

    async def all_documents(self) -> list[tuple[str, str]]:
        """Every indexed ``(document_id, source_key)`` -- used to reconcile the index
        against the system of record."""
        rows = await self._db.run(
            lambda coll: list(coll.find({"kind": DOCUMENT}, {"document_id": 1, "source_key": 1}))
        )
        return [(r["document_id"], r.get("source_key") or "") for r in rows]

    async def aclose(self) -> None:
        await self._db.aclose()


__all__ = ["MongoMetadataStore", "document_record"]
