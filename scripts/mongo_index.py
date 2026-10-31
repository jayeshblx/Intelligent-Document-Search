"""Manage the MongoDB Atlas search index: one collection per tenant.

Layout (see ``rag_search/stores/mongo.py``)::

    EX_<FEATURE>_<ENV>              default EX_DOCUMENTSEARCH_DEV
        <TenantCode>_DocumentSearch_<ENV>
          FOXTROT_DocumentSearch_DEV     chunks + vectors + metadata + manifest for FOXTROT
          BRAVO_DocumentSearch_DEV    ... for BRAVO

Subcommands
-----------
setup     Create the collection, its regular indexes and both Atlas search indexes.

migrate   Copy an existing LOCAL index (Qdrant + SQLite, e.g. ./data/search) into a
          collection. The stored embeddings are copied as-is -- nothing is
          re-embedded -- so vector results match the local index exactly, at no
          Bedrock cost. Writes the manifest and waits until Atlas has indexed it.

manifest  (Re)write the manifest the agent reads: tenant codes, source bucket and
          region, counts. Run after a fresh ``index_s3.py`` run into a collection.

move      Move an Atlas collection to a new database/collection name: a native
          rename, or (where Atlas refuses one) a server-side copy that is verified
          before the source is dropped. Embeddings move as-is; the search indexes
          are recreated on the new name.

status    Counts and search-index state.

compare   Run queries through the local index and the collection with the SAME
          query embedding and report how far the two rankings agree.

Examples::

    uv run python scripts/mongo_index.py setup   --collection BRAVO_DocumentSearch_DEV
    uv run python scripts/mongo_index.py migrate --collection FOXTROT_DocumentSearch_DEV \\
        --from-dir ./data/search --environment foxtrot-Dev --tenant foxtrot --tenant-codes FOXTROT
    uv run python scripts/mongo_index.py move    --collection BRAVO_DocumentSearch_DEV \\
        --from-database DocumentSearch --from-collection DocumentSearch_BRAVO_DEV
    uv run python scripts/mongo_index.py compare --collection FOXTROT_DocumentSearch_DEV \\
        --from-dir ./data/search
    uv run python scripts/mongo_index.py status  --collection FOXTROT_DocumentSearch_DEV

The connection string comes from SEARCH_INDEX_MONGO_URI, else SEARCH_MONGO_URI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_search.config import SearchSettings
from rag_search.stores.mongo import (
    CHUNK,
    MongoIndexDatabase,
    MongoKeywordIndex,
    MongoVectorIndex,
    embedding_value,
    facet_terms,
)
from rag_search.stores.mongo_metadata import document_record

from rag_core.config import get_settings
from rag_core.di import Container
from rag_core.logging import configure_logging

MANIFEST_VERSION = 3  # v1 local file set, v2 postgres dump, v3 live mongo collection
_PAGE = 1000  # ~8 MB per bulk write (1024-float vectors + text)

DEFAULT_QUERIES = (
    "allergen declaration for the product",
    "supplier certification expiry",
    "food safety audit corrective actions",
    "nutritional information per 100g",
    "packaging material specification",
    "country of origin of ingredients",
    "shelf life and storage conditions",
    "halal or kosher certificate",
    "microbiological testing limits",
    "GMO status statement",
)


def _db(args: argparse.Namespace, search: SearchSettings) -> MongoIndexDatabase:
    uri = search.index_mongo_uri or search.mongo_uri
    if not uri:
        raise SystemExit("Set SEARCH_INDEX_MONGO_URI (or SEARCH_MONGO_URI) first.")
    return MongoIndexDatabase(
        uri,
        database=args.database or search.index_mongo_database,
        collection=args.collection,
        dim=search.embed_dim,
        tls_ca_file=search.index_mongo_tls_ca_file or search.mongo_tls_ca_file,
        exact_vector_search=getattr(args, "exact", search.index_mongo_exact_vectors),
    )


# ---- setup ------------------------------------------------------------------


async def _setup(args: argparse.Namespace) -> int:
    db = _db(args, SearchSettings())
    try:
        await db.ensure_schema()
        print(f"  ready  {db.database_name}.{db.collection_name}")
        print(f"  search indexes: {await db.search_index_status()}")
    finally:
        await db.aclose()
    return 0


# ---- migrate ----------------------------------------------------------------


def _read_sqlite(path: Path, sql: str) -> list[sqlite3.Row]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _chunk_doc(
    chunk_id: str, text: str | None, payload: dict[str, Any], vector: Any
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "_id": chunk_id,
        "kind": CHUNK,
        "document_id": payload.get("document_id"),
        "payload": payload,
        "facet_kv": facet_terms(payload),
        "text": text or "",
    }
    if vector is not None:
        doc["embedding"] = embedding_value(vector)
    return doc


async def _migrate(args: argparse.Namespace) -> int:
    from pymongo import ReplaceOne

    search = SearchSettings()
    source = Path(args.from_dir)
    for name in ("keyword.db", "metadata.db", "qdrant"):
        if not (source / name).exists():
            print(f"  {source / name} not found", file=sys.stderr)
            return 2

    db = _db(args, search)
    try:
        await db.ensure_schema()
        existing = (await db.counts())["chunks"]
        if existing and not args.replace:
            print(
                f"  {db.collection_name} already holds {existing} chunks; pass --replace to "
                "overwrite it",
                file=sys.stderr,
            )
            return 2
        if existing:
            await db.run(lambda coll: coll.delete_many({"kind": {"$in": [CHUNK, "document"]}}))

        # Stream the points straight out of the embedded Qdrant's SQLite file. Opening
        # it with QdrantClient loads the whole collection into memory and its scroll
        # slows to a crawl past ~10k points (a 300k-point BRAVO run stalled twice).
        storage = (
            source
            / "qdrant"
            / "collection"
            / (args.qdrant_collection or search.collection)
            / "storage.sqlite"
        )
        if not storage.exists():
            print(f"  {storage} not found", file=sys.stderr)
            return 2
        points_db = sqlite3.connect(f"file:{storage}?mode=ro", uri=True, check_same_thread=False)
        keyword_db = sqlite3.connect(
            f"file:{source / 'keyword.db'}?mode=ro", uri=True, check_same_thread=False
        )
        total = points_db.execute("SELECT count(*) FROM points").fetchone()[0]
        print(f"  source  {total} vectors")

        def _next_batch(cursor: sqlite3.Cursor) -> list[Any]:
            # The file is one this project wrote (publish_file_snapshot / the indexer).
            points = [pickle.loads(blob) for (blob,) in cursor.fetchmany(_PAGE)]
            ids = [str(p.id) for p in points]
            marks = ",".join("?" * len(ids))
            # Keyword rows are the canonical text + payload (the keyword write lands
            # last in the indexer, so its payload is the one the app reads).
            texts = (
                {
                    cid: (text, json.loads(payload or "{}"))
                    for cid, text, payload in keyword_db.execute(
                        f"SELECT chunk_id, text, payload FROM chunks WHERE chunk_id IN ({marks})",
                        ids,
                    )
                }
                if ids
                else {}
            )
            ops = []
            for p, cid in zip(points, ids, strict=True):
                vec = p.vector
                if isinstance(vec, dict):  # named vectors: take the only one
                    vec = next(iter(vec.values()), None)
                text, payload = texts.get(cid, ("", dict(p.payload or {})))
                ops.append(
                    ReplaceOne({"_id": cid}, _chunk_doc(cid, text, payload, vec), upsert=True)
                )
            return ops

        started = time.monotonic()
        written = 0
        try:
            cursor = points_db.execute("SELECT point FROM points")
            while ops := await asyncio.to_thread(_next_batch, cursor):
                await db.run(lambda coll, o=ops: coll.bulk_write(o, ordered=False))
                written += len(ops)
                rate = written / max(1e-6, time.monotonic() - started)
                print(f"\r  chunks  {written}/{total}  ({rate:.0f}/s)", end="", flush=True)

            # Keyword-only chunks (no vector) still belong in the sparse index.
            vector_count = written
            keyword_count = keyword_db.execute("SELECT count(*) FROM chunks").fetchone()[0]
            orphans = 0
            if keyword_count > vector_count:
                seen = {
                    str(pickle.loads(blob).id)
                    for (blob,) in points_db.execute("SELECT point FROM points")
                }
                batch = []
                for cid, text, payload in keyword_db.execute(
                    "SELECT chunk_id, text, payload FROM chunks"
                ):
                    if cid in seen:
                        continue
                    batch.append(
                        ReplaceOne(
                            {"_id": cid},
                            _chunk_doc(cid, text, json.loads(payload or "{}"), None),
                            upsert=True,
                        )
                    )
                    if len(batch) == 500:
                        await db.run(lambda coll, b=batch: coll.bulk_write(b, ordered=False))
                        orphans, batch = orphans + len(batch), []
                if batch:
                    await db.run(lambda coll, b=batch: coll.bulk_write(b, ordered=False))
                    orphans += len(batch)
        finally:
            points_db.close()
            keyword_db.close()
        print(f"\r  chunks  {written} with vectors, {orphans} keyword-only")

        # Metadata: copy the rows and their facet pairs verbatim, so the filter
        # dropdowns are identical to the local index.
        meta_rows = await asyncio.to_thread(
            _read_sqlite,
            source / "metadata.db",
            "SELECT document_id, namespace, source_key, payload, facets, effective_date, "
            "expiry_date FROM document_metadata",
        )
        facet_rows = await asyncio.to_thread(
            _read_sqlite,
            source / "metadata.db",
            "SELECT document_id, key, value FROM document_facet",
        )
        pairs: dict[str, list[tuple[str, str]]] = {}
        for r in facet_rows:
            pairs.setdefault(r["document_id"], []).append((r["key"], str(r["value"])))
        docs = [
            ReplaceOne(
                {"_id": f"meta::{r['document_id']}"},
                document_record(
                    document_id=r["document_id"],
                    namespace=r["namespace"] or "default",
                    source_key=r["source_key"] or "",
                    payload=json.loads(r["payload"] or "{}"),
                    facets=json.loads(r["facets"] or "{}"),
                    pairs=pairs.get(r["document_id"], []),
                    effective_date=r["effective_date"],
                    expiry_date=r["expiry_date"],
                ),
                upsert=True,
            )
            for r in meta_rows
        ]
        for start in range(0, len(docs), 500):
            batch = docs[start : start + 500]
            await db.run(lambda coll, b=batch: coll.bulk_write(b, ordered=False))
        print(f"  docs    {len(docs)} metadata records")

        await _write_manifest(args, db, search)
        print("  waiting for Atlas to index everything ...")
        await db.wait_until_searchable(timeout_s=args.timeout)
        print(f"  done    {await db.counts()}  ({time.monotonic() - started:.0f}s)")
    finally:
        await db.aclose()
    return 0


# ---- manifest ---------------------------------------------------------------


async def _bucket_region(core: Any, bucket: str) -> str:
    container = Container(core)
    try:
        async with container.aws_factory().client("s3") as client:
            location = (await client.get_bucket_location(Bucket=bucket)).get("LocationConstraint")
    except Exception as exc:
        print(f"  (source region lookup failed: {exc})")
        return ""
    finally:
        await container.aclose()
    return {None: "us-east-1", "": "us-east-1", "EU": "eu-west-1"}.get(location, location)


async def _write_manifest(
    args: argparse.Namespace, db: MongoIndexDatabase, search: SearchSettings
) -> None:
    core = get_settings()
    codes = args.tenant_codes or search.mongo_tenant_codes
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "backend": "mongo",
        "tenant": args.tenant,
        "environment": args.environment,
        "database": db.database_name,
        "collection": db.collection_name,
        "embed_dim": search.embed_dim,
        "namespace": search.namespace,
        "built_at": datetime.now(UTC).isoformat(),
        "source_bucket": args.source_bucket or core.s3.bucket,
        "source_region": args.source_region
        or await _bucket_region(core, args.source_bucket or core.s3.bucket),
        "source_prefix": core.s3.prefix if args.source_prefix is None else args.source_prefix,
        "tenant_codes": [c.strip() for c in codes.split(",") if c.strip()],
        "counts": await db.counts(),
    }
    await db.write_manifest(manifest)
    print(
        "  manifest "
        + json.dumps(
            {
                k: manifest[k]
                for k in (
                    "environment",
                    "tenant",
                    "tenant_codes",
                    "source_bucket",
                    "source_region",
                    "counts",
                )
            }
        )
    )


async def _manifest(args: argparse.Namespace) -> int:
    search = SearchSettings()
    db = _db(args, search)
    try:
        await _write_manifest(args, db, search)
    finally:
        await db.aclose()
    return 0


# ---- copy -------------------------------------------------------------------


async def _move(args: argparse.Namespace) -> int:
    from pymongo.errors import OperationFailure

    search = SearchSettings()
    db = _db(args, search)
    src_db, src_coll = args.from_database, args.from_collection
    src_ns, dst_ns = f"{src_db}.{src_coll}", f"{db.database_name}.{db.collection_name}"
    if src_ns == dst_ns:
        print("  source and target are the same collection", file=sys.stderr)
        return 2

    def _sizes(coll: Any) -> tuple[int, int]:
        return coll.count_documents({}), coll.count_documents({"embedding": {"$exists": True}})

    try:

        def _preflight(coll: Any) -> tuple[tuple[int, int], int]:
            src = coll.database.client[src_db][src_coll]
            return _sizes(src), coll.estimated_document_count()

        (total, embedded), existing = await db.run(_preflight)
        if not total:
            print(f"  {src_ns} is empty or does not exist", file=sys.stderr)
            return 2
        if existing:
            print(f"  {dst_ns} already holds {existing} documents; drop it first", file=sys.stderr)
            return 2
        print(f"  source  {src_ns}: {total} documents, {embedded} with vectors")
        started = time.monotonic()

        def _rename(coll: Any) -> None:
            coll.database.client.admin.command("renameCollection", src_ns, to=dst_ns)

        try:
            await db.run(_rename)
            print(f"  renamed {src_ns} -> {dst_ns}")
        except OperationFailure as exc:
            # Not every Atlas tier allows a cross-database rename: copy server-side
            # instead, verify, and only then drop the source.
            print(f"  rename refused ({exc.code}: {exc.details.get('errmsg', exc)}); copying")

            def _copy(coll: Any) -> tuple[int, int]:
                src = coll.database.client[src_db][src_coll]
                src.aggregate(
                    [{"$merge": {"into": {"db": db.database_name, "coll": db.collection_name}}}]
                )
                return _sizes(coll)

            copied = await db.run(_copy)
            if copied != (total, embedded):
                print(
                    f"  copy mismatch: target has {copied}, source {(total, embedded)}; "
                    f"{src_ns} was NOT dropped",
                    file=sys.stderr,
                )
                return 1
            await db.run(lambda coll: coll.database.client[src_db].drop_collection(src_coll))
            print(f"  copied  {total} documents, verified, dropped {src_ns}")

        # Atlas search indexes do not travel with a cross-database move: (re)create
        # them on the new namespace and wait until both are queryable.
        await db.ensure_schema()
        manifest = await db.read_manifest()
        if manifest:
            manifest.update(
                database=db.database_name, collection=db.collection_name, counts=await db.counts()
            )
            await db.write_manifest(manifest)
        print("  waiting for Atlas to index everything ...")
        await db.wait_until_searchable(timeout_s=args.timeout)
        print(f"  done    {await db.counts()}  ({time.monotonic() - started:.0f}s)")
    finally:
        await db.aclose()
    return 0


# ---- status -----------------------------------------------------------------


async def _status(args: argparse.Namespace) -> int:
    db = _db(args, SearchSettings())
    try:
        print(f"  {db.database_name}.{db.collection_name}")
        print(f"  counts   {await db.counts()}")
        print(f"  indexes  {await db.search_index_status()}")
        manifest = await db.read_manifest()
        print(f"  manifest {json.dumps(manifest, default=str)[:600]}")
    finally:
        await db.aclose()
    return 0


# ---- compare ----------------------------------------------------------------


def _overlap(a: list[str], b: list[str]) -> float:
    return len(set(a) & set(b)) / max(1, len(set(a) | set(b)))


async def _compare(args: argparse.Namespace) -> int:
    from rag_search.embedder import TitanEmbedder
    from rag_search.fusion import reciprocal_rank_fusion
    from rag_search.stores import KeywordIndex, LocalVectorIndex

    search = SearchSettings()
    queries = DEFAULT_QUERIES
    if args.queries:
        raw = await asyncio.to_thread(Path(args.queries).read_text, encoding="utf-8")
        queries = tuple(q.strip() for q in raw.splitlines() if q.strip())

    core = get_settings()
    container = Container(core)
    embedder = TitanEmbedder(container.bedrock(), dim=search.embed_dim)
    source = Path(args.from_dir)
    local_vec = LocalVectorIndex(
        str(source / "qdrant"),
        collection=args.qdrant_collection or search.collection,
        dim=search.embed_dim,
    )
    local_kw = KeywordIndex(str(source / "keyword.db"))
    db = _db(args, search)
    mongo_vec, mongo_kw = MongoVectorIndex(db), MongoKeywordIndex(db)
    k, cand = args.top_k, search.candidate_k
    filters = json.loads(args.filters) if args.filters else None

    async def texts(kw: Any, chunk_ids: list[str]) -> list[str]:
        got = await kw.get(chunk_ids)
        return [" ".join(str(got.get(c, ("", {}))[0]).split()) for c in chunk_ids]

    async def doc_ids(kw: Any, chunk_ids: list[str]) -> list[str]:
        got = await kw.get(chunk_ids)
        return [str((got.get(c, ("", {}))[1]).get("document_id") or c) for c in chunk_ids]

    rows: list[dict[str, Any]] = []
    try:
        await db.wait_until_searchable(timeout_s=1800)
        await mongo_kw.search("warm up", limit=1)  # connection set-up is not a query cost
        print(
            f"  {'query':45} {'vecScore':>8} {'kw@' + str(cand):>6} {'docs@' + str(k):>7} "
            f"{'top doc':>7} {'mongo':>6}"
        )
        for q in queries:
            qv = await embedder.embed_query(q)
            l_vec = await local_vec.search(qv, limit=cand, filters=filters)
            lk = [c for c, _ in await local_kw.search(q, limit=cand, filters=filters)]
            started = time.perf_counter()
            m_vec, mk_hits = await asyncio.gather(
                mongo_vec.search(qv, limit=cand, filters=filters),
                mongo_kw.search(q, limit=cand, filters=filters),
            )
            elapsed = (time.perf_counter() - started) * 1000
            mk = [c for c, _ in mk_hits]
            # Tie-aware: identical vectors give identical scores, and engines order
            # tied chunks arbitrarily -- so compare the SCORE at every rank.
            pairs = list(zip(l_vec, m_vec, strict=False))
            vec_score = sum(abs(a.score - b.score) < 1e-4 for a, b in pairs) / max(
                1, max(len(l_vec), len(m_vec))
            )
            lf = [
                c
                for c, _ in reciprocal_rank_fusion(
                    [[h.chunk_id for h in l_vec], lk], k=search.rrf_k
                )[:k]
            ]
            mf = [
                c
                for c, _ in reciprocal_rank_fusion(
                    [[h.chunk_id for h in m_vec], mk], k=search.rrf_k
                )[:k]
            ]
            l_docs, m_docs = await doc_ids(local_kw, lf), await doc_ids(mongo_kw, mf)
            # Content-level: duplicate chunks (one template in many documents) tie,
            # and either engine may pick a different copy -- same text, other doc id.
            l_texts, m_texts = await texts(local_kw, lf), await texts(mongo_kw, mf)
            row = {
                "query": q,
                "vector_score_match": vec_score,
                "keyword_overlap": _overlap(lk, mk),
                "fused_doc_overlap": _overlap(l_docs, m_docs),
                "same_top_doc": l_docs[:1] == m_docs[:1],
                "fused_text_overlap": _overlap(l_texts, m_texts),
                "same_top_text": l_texts[:1] == m_texts[:1],
                "mongo_ms": elapsed,
            }
            rows.append(row)
            print(
                f"  {q[:45]:45} {vec_score:>8.0%} {row['keyword_overlap']:>6.0%} "
                f"{row['fused_doc_overlap']:>7.0%} {'yes' if row['same_top_doc'] else 'NO':>7} "
                f"{elapsed:5.0f}ms"
            )
        n = max(1, len(rows))
        ms = sorted(r["mongo_ms"] for r in rows)
        summary = {
            "queries": len(rows),
            "vector_score_match": sum(r["vector_score_match"] for r in rows) / n,
            "keyword_overlap": sum(r["keyword_overlap"] for r in rows) / n,
            "fused_doc_overlap": sum(r["fused_doc_overlap"] for r in rows) / n,
            "same_top_doc": sum(r["same_top_doc"] for r in rows),
            "fused_text_overlap": sum(r["fused_text_overlap"] for r in rows) / n,
            "same_top_text": sum(r["same_top_text"] for r in rows),
            "mongo_ms_median": ms[len(ms) // 2] if ms else 0.0,
            "mongo_ms_p95": ms[int(len(ms) * 0.95) - 1] if ms else 0.0,
        }
        print(
            f"\n  vector scores identical at {summary['vector_score_match']:.0%} of ranks, "
            f"keyword top-{cand} overlap {summary['keyword_overlap']:.0%}, "
            f"final top-{k} documents overlap {summary['fused_doc_overlap']:.0%}, "
            f"same top document {summary['same_top_doc']}/{len(rows)}; by content: "
            f"top-{k} overlap {summary['fused_text_overlap']:.0%}, "
            f"same top passage {summary['same_top_text']}/{len(rows)}; "
            f"mongo retrieval median {summary['mongo_ms_median']:.0f}ms "
            f"(p95 {summary['mongo_ms_p95']:.0f}ms)"
        )
        if args.out:
            payload = json.dumps({"summary": summary, "rows": rows}, indent=2)
            await asyncio.to_thread(Path(args.out).write_text, payload, encoding="utf-8")
    finally:
        await local_vec.aclose()
        await local_kw.aclose()
        await db.aclose()
        await container.aclose()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--collection", required=True, help="e.g. FOXTROT_DocumentSearch_DEV")
        p.add_argument("--database", default="", help="default SEARCH_INDEX_MONGO_DATABASE")

    def manifest_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--environment", required=True, help="agent environment, e.g. foxtrot-Dev")
        p.add_argument("--tenant", required=True, help="agent tenant, e.g. foxtrot")
        p.add_argument(
            "--tenant-codes", default="", help="e.g. FOXTROT (default SEARCH_MONGO_TENANT_CODES)"
        )
        p.add_argument("--source-bucket", default="", help="default S3__BUCKET")
        p.add_argument("--source-region", default="", help="default: looked up")
        p.add_argument("--source-prefix", default=None, help="default S3__PREFIX")

    p = sub.add_parser("setup", help="create collection + indexes")
    common(p)
    p.set_defaults(fn=_setup)

    p = sub.add_parser("migrate", help="copy a local Qdrant+SQLite index into a collection")
    common(p)
    manifest_args(p)
    p.add_argument("--from-dir", default="./data/search")
    p.add_argument("--qdrant-collection", default="", help="default SEARCH_COLLECTION")
    p.add_argument("--replace", action="store_true", help="overwrite a non-empty collection")
    p.add_argument("--timeout", type=float, default=1800.0, help="seconds to wait for Atlas")
    p.set_defaults(fn=_migrate)

    p = sub.add_parser("manifest", help="write the manifest the agent reads")
    common(p)
    manifest_args(p)
    p.set_defaults(fn=_manifest)

    p = sub.add_parser("move", help="rename (or copy + drop) a collection to a new name")
    common(p)
    p.add_argument("--from-collection", required=True, help="e.g. DocumentSearch_BRAVO_DEV")
    p.add_argument("--from-database", default="DocumentSearch")
    p.add_argument("--timeout", type=float, default=1800.0, help="seconds to wait for Atlas")
    p.set_defaults(fn=_move)

    p = sub.add_parser("status", help="counts + search index state")
    common(p)
    p.set_defaults(fn=_status)

    p = sub.add_parser("compare", help="rank agreement: local index vs collection")
    common(p)
    p.add_argument("--from-dir", default="./data/search")
    p.add_argument("--qdrant-collection", default="")
    p.add_argument("--queries", default="", help="file with one query per line")
    p.add_argument("--filters", default="", help='JSON facet filters, e.g. {"status": "Draft"}')
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument(
        "--approximate",
        dest="exact",
        action="store_false",
        default=True,
        help="approximate (quantized HNSW + rescoring) instead of exact vector search",
    )
    p.add_argument("--out", default="", help="write per-query results as JSON")
    p.set_defaults(fn=_compare)

    args = parser.parse_args()
    configure_logging()
    raise SystemExit(asyncio.run(args.fn(args)))


if __name__ == "__main__":
    main()
