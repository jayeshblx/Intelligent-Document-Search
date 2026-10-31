"""Postgres backend for the chunk index — vectors AND keyword search in one table.

Drop-in replacement for the embedded-Qdrant + SQLite-FTS5 pair. ``PgVectorIndex``
and ``PgKeywordIndex`` expose exactly the surfaces of ``LocalVectorIndex`` and
``KeywordIndex``, so ``SearchService``/``IndexingService`` are unchanged, but both
facades read and write ONE ``chunks`` table:

    chunk_id | text | payload(jsonb) | embedding(vector) | tsv | document_id

Why one table rather than two mirrored stores:

* The chunk payload was previously duplicated into Qdrant *and* SQLite, so every
  facet enrichment had to be written twice (``set_document_payload`` +
  ``update_payload``) and could disagree. Here it is one row, updated once.
* A document's vectors and its keyword rows now land in one transaction — a crash
  mid-document can no longer leave vectors without their text.
* Embedded Qdrant file-locks its directory to a single process, so the indexing
  script could not run while the search server was up. Postgres has no such limit.

Ranking note: Postgres full-text ranks with ``ts_rank_cd`` (cover density), not
BM25. That is fine here because results are fused by RRF (``fusion.py``), which
consumes *ranks*, never raw scores — the sparse list only has to be ordered
sensibly, not calibrated against the dense one.

Threading model: psycopg's SYNCHRONOUS pool driven through ``asyncio.to_thread``,
matching how ``KeywordIndex`` (sqlite3) and ``LocalVectorIndex`` (Qdrant local)
already work. This is deliberate — psycopg's *async* driver refuses to run on
Windows' default ProactorEventLoop, which is the loop uvicorn builds there, and
recent uvicorn passes an explicit ``loop_factory`` that an event-loop policy
cannot override. The sync pool has no such constraint, so the same code runs
under uvicorn, pytest and the CLI scripts on every platform.

Vectors are passed as ``'[1,2,3]'::vector`` literals, so no client-side type
registration is needed on pooled connections.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

from rag_core.logging import get_logger
from rag_search.facet_filters import FacetFilters, accepted_values
from rag_search.stores.vector_local import VectorMatch

_log = get_logger("rag_search.postgres")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_T = TypeVar("_T")

# Text-search configuration. 'english' applies a Porter stemmer, matching the
# `tokenize='porter unicode61'` the SQLite FTS5 index used.
_TS_CONFIG = "english"

# Seconds to wait for the pool to come up before failing with a clear error.
_CONNECT_TIMEOUT_S = 30.0


def _vector_literal(values: Sequence[float]) -> str:
    """pgvector input syntax — ``[0.1,0.2,...]``."""
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def scrub_nuls(value: Any) -> Any:
    """Strip NUL characters from strings, recursing into dicts and lists.

    Postgres accepts NUL in neither ``text`` nor ``jsonb``: a chunk containing one
    aborts the whole statement with *"unsupported Unicode escape sequence:
    \\u0000 cannot be converted to text"*, which fails the entire document. SQLite
    stores NULs without complaint, so this is the one place the two backends
    genuinely disagree about what they will accept.

    Extracted PDF/OCR text carries stray NULs often enough to matter — they cost
    17 documents on the first full run — and they are never meaningful content, so
    dropping them is lossless in practice and keeps the document indexable.
    """
    if isinstance(value, str):
        return value.replace("\x00", "") if "\x00" in value else value
    if isinstance(value, dict):
        return {k: scrub_nuls(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub_nuls(v) for v in value]
    return value


def scrub_json_text(serialized: str) -> str:
    """Same, for already-serialised JSON — a NUL arrives escaped as ``\\u0000``."""
    if "u0000" not in serialized and "\x00" not in serialized:
        return serialized
    return serialized.replace("\\u0000", "").replace("\x00", "")


def _facet_sql(facets: Mapping[str, list[str]], alias: str = "chunks") -> tuple[str, list[Any]]:
    """AND-predicates restricting chunks to a facet selection, as (sql, params).

    A facet with several accepted values matches any of them, so a request scoped
    to more than one BusinessObject sees all of them; facets still AND together.
    """
    parts: list[str] = []
    params: list[Any] = []
    for key, values in facets.items():
        if not values:
            continue
        if len(values) == 1:
            parts.append(f" AND {alias}.payload->>%s = %s")
            params.extend([key, values[0]])
        else:
            parts.append(f" AND {alias}.payload->>%s = ANY(%s)")
            params.extend([key, list(values)])
    return "".join(parts), params


class PgDatabase:
    """Owns the connection pool and the whole schema; shared by all three stores.

    Opening is lazy and idempotent so any store can be the first one touched (the
    app may serve ``/v1/facets`` before an index run ever calls ``ensure()``).
    """

    def __init__(self, dsn: str, *, dim: int, min_size: int = 1, max_size: int = 10) -> None:
        self._dsn = dsn
        self._dim = dim
        self._min_size = min_size
        self._max_size = max_size
        self._pool: Any = None
        self._init_lock = threading.Lock()
        self._closed = False

    @property
    def dim(self) -> int:
        return self._dim

    # ---- plumbing ---------------------------------------------------------

    def _get_pool(self) -> Any:
        """Open the pool and create the schema once. Called from a worker thread."""
        if self._pool is not None:
            return self._pool
        with self._init_lock:
            if self._pool is not None:  # another thread won the race
                return self._pool
            if self._closed:
                raise RuntimeError("PgDatabase is closed")
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool

            pool = ConnectionPool(
                self._dsn,
                min_size=self._min_size,
                max_size=self._max_size,
                kwargs={"row_factory": dict_row},
                open=False,
            )
            pool.open(wait=True, timeout=_CONNECT_TIMEOUT_S)
            self._create_schema(pool)
            self._pool = pool
            return pool

    async def run(self, fn: Callable[[Any], _T]) -> _T:
        """Run ``fn(cursor)`` in a worker thread inside ONE transaction.

        Multi-statement operations pass a single callable so they commit or roll
        back together — that is how a document's chunks, or a metadata row and its
        facet rows, stay consistent.
        """

        def _work() -> _T:
            pool = self._get_pool()
            with pool.connection() as conn, conn.cursor() as cur:
                return fn(cur)

        return await asyncio.to_thread(_work)

    async def fetch(self, sql: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        def _op(cur: Any) -> list[dict[str, Any]]:
            cur.execute(sql, params)
            return list(cur.fetchall())

        return await self.run(_op)

    async def fetchone(
        self, sql: str, params: Sequence[Any] | None = None
    ) -> dict[str, Any] | None:
        def _op(cur: Any) -> dict[str, Any] | None:
            cur.execute(sql, params)
            return cur.fetchone()

        return await self.run(_op)

    async def execute(self, sql: str, params: Sequence[Any] | None = None) -> int:
        """Run one statement, returning the affected row count."""

        def _op(cur: Any) -> int:
            cur.execute(sql, params)
            return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

        return await self.run(_op)

    async def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> int:
        if not rows:
            return 0

        def _op(cur: Any) -> int:
            cur.executemany(sql, rows)
            return len(rows)

        return await self.run(_op)

    # ---- schema -----------------------------------------------------------

    def _create_schema(self, pool: Any) -> None:
        """Create extensions, tables and indexes. Safe to run repeatedly.

        Both the chunk index and the metadata tables are created here so there is a
        single lazy-init path no matter which store is touched first.
        """
        with pool.connection() as conn, conn.cursor() as cur:
            # pgvector supplies the `vector` type; pg_trgm makes the substring
            # `exact_lookup` scan use an index instead of reading every chunk.
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id    text PRIMARY KEY,
                    text        text NOT NULL DEFAULT '',
                    payload     jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding   vector({self._dim}),
                    document_id text
                        GENERATED ALWAYS AS (payload->>'document_id') STORED,
                    tsv tsvector
                        GENERATED ALWAYS AS (
                            to_tsvector('{_TS_CONFIG}', coalesce(text, ''))
                        ) STORED
                )
                """
            )
            # The embedding dimension is baked into the column type, so a changed
            # SEARCH_EMBED_DIM against an existing table must fail loudly rather
            # than error obscurely on the first insert.
            cur.execute(
                "SELECT format_type(atttypid, atttypmod) AS t FROM pg_attribute "
                "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
            )
            declared = str((cur.fetchone() or {}).get("t") or "")
            if (m := re.search(r"\((\d+)\)", declared)) and int(m.group(1)) != self._dim:
                raise RuntimeError(
                    f"The 'chunks' table stores {m.group(1)}-dimensional vectors but this "
                    f"service embeds at {self._dim}. Either set SEARCH_EMBED_DIM="
                    f"{m.group(1)} or drop the table and re-index."
                )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS document_metadata (
                    document_id    text PRIMARY KEY,
                    namespace      text,
                    source_key     text,
                    payload        jsonb NOT NULL DEFAULT '{}'::jsonb,
                    facets         jsonb NOT NULL DEFAULT '{}'::jsonb,
                    effective_date text,
                    expiry_date    text
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS document_facet (
                    namespace   text,
                    document_id text,
                    key         text,
                    value       text
                )
                """
            )

            for ddl in (
                "CREATE INDEX IF NOT EXISTS ix_chunks_tsv ON chunks USING gin (tsv)",
                "CREATE INDEX IF NOT EXISTS ix_chunks_document ON chunks (document_id)",
                "CREATE INDEX IF NOT EXISTS ix_chunks_payload "
                "ON chunks USING gin (payload jsonb_path_ops)",
                "CREATE INDEX IF NOT EXISTS ix_chunks_text_trgm "
                "ON chunks USING gin (text gin_trgm_ops)",
                # HNSW: approximate cosine search. Built on an empty table it costs
                # nothing; rows are indexed as they arrive.
                "CREATE INDEX IF NOT EXISTS ix_chunks_embedding "
                "ON chunks USING hnsw (embedding vector_cosine_ops)",
                "CREATE INDEX IF NOT EXISTS ix_facet_ns_key "
                "ON document_facet (namespace, key, value)",
                "CREATE INDEX IF NOT EXISTS ix_facet_doc ON document_facet (document_id)",
                # match_facet_values() looks up by lower(value) across the whole
                # vocabulary — without this it is a full scan.
                "CREATE INDEX IF NOT EXISTS ix_facet_lower_value "
                "ON document_facet (namespace, lower(value))",
                "CREATE INDEX IF NOT EXISTS ix_expiry "
                "ON document_metadata (namespace, expiry_date)",
            ):
                cur.execute(ddl)
        _log.info("pg_schema_ready", dim=self._dim)

    async def aclose(self) -> None:
        """Idempotent — every store sharing this database calls it on shutdown."""
        with self._init_lock:
            self._closed = True
            pool, self._pool = self._pool, None
        if pool is not None:
            await asyncio.to_thread(pool.close)


class _ChunkStore:
    """Shared behaviour of the two facades over the ``chunks`` table."""

    def __init__(self, db: PgDatabase) -> None:
        self._db = db

    async def reset(self) -> None:
        await self._db.execute("TRUNCATE TABLE chunks")

    async def _merge_payload(self, document_id: str, fields: Mapping[str, Any]) -> int:
        """Merge facet fields into every chunk of one document. ``None`` removes a key."""
        if not fields:
            return 0
        from psycopg.types.json import Jsonb

        setting = scrub_nuls({k: v for k, v in fields.items() if v is not None})
        clearing = [k for k, v in fields.items() if v is None]
        return await self._db.execute(
            "UPDATE chunks SET payload = (payload || %s::jsonb) - %s::text[] "
            "WHERE document_id = %s",
            (Jsonb(setting), clearing, document_id),
        )

    async def delete_document(self, document_id: str) -> int:
        return await self._db.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))

    async def aclose(self) -> None:
        await self._db.aclose()


class PgVectorIndex(_ChunkStore):
    """Dense retrieval over pgvector — mirrors ``LocalVectorIndex``."""

    async def ensure(self) -> None:
        # Schema creation happens on first connection; this forces it eagerly so an
        # index run fails fast on a bad DSN rather than mid-document.
        await self._db.fetchone("SELECT 1 AS ok")

    async def upsert(self, points: list[tuple[str, list[float], dict[str, Any]]]) -> int:
        if not points:
            return 0
        from psycopg.types.json import Jsonb

        rows = [
            (chunk_id, _vector_literal(vec), Jsonb(scrub_nuls(payload)))
            for chunk_id, vec, payload in points
        ]
        return await self._db.executemany(
            "INSERT INTO chunks (chunk_id, embedding, payload) VALUES (%s, %s::vector, %s) "
            "ON CONFLICT (chunk_id) DO UPDATE SET "
            "embedding = EXCLUDED.embedding, payload = EXCLUDED.payload",
            rows,
        )

    async def set_document_payload(self, document_id: str, fields: Mapping[str, Any]) -> None:
        await self._merge_payload(document_id, fields)

    async def search(
        self, vector: list[float], *, limit: int, filters: FacetFilters | None = None
    ) -> list[VectorMatch]:
        facets = {k: vals for k, v in (filters or {}).items() if (vals := accepted_values(v))}
        facet_sql, facet_params = _facet_sql(facets)
        literal = _vector_literal(vector)
        # `<=>` is cosine DISTANCE (0 = identical); Qdrant reports cosine SIMILARITY,
        # so flip it to keep scores comparable with the local backend.
        rows = await self._db.fetch(
            "SELECT chunk_id, payload, 1 - (embedding <=> %s::vector) AS score "
            f"FROM chunks WHERE embedding IS NOT NULL{facet_sql} "
            "ORDER BY embedding <=> %s::vector LIMIT %s",
            (literal, *facet_params, literal, limit),
        )
        return [
            VectorMatch(
                chunk_id=r["chunk_id"], score=float(r["score"]), payload=dict(r["payload"] or {})
            )
            for r in rows
        ]

    async def fetch_document(
        self, document_id: str, *, namespace: str = "default", limit: int = 60
    ) -> list[dict[str, Any]]:
        rows = await self._db.fetch(
            "SELECT chunk_id, payload FROM chunks WHERE document_id = %s LIMIT %s",
            (document_id, limit),
        )
        payloads = []
        for r in rows:
            payload = dict(r["payload"] or {})
            payload["_chunk_id"] = r["chunk_id"]
            payloads.append(payload)
        payloads.sort(key=lambda pl: (pl.get("page_start") or 0, pl.get("_chunk_id", "")))
        return payloads

    async def count(self) -> int:
        row = await self._db.fetchone(
            "SELECT COUNT(*) AS n FROM chunks WHERE embedding IS NOT NULL"
        )
        return int((row or {}).get("n", 0))

    async def rebuild_index(self) -> None:
        """Rebuild the HNSW graph from every row currently in the table.

        Measured on the FOXTROT dev index: 250 vectors added after a 45,000-vector build
        were so poorly connected that HNSW returned a 0.432 match while the true
        best (0.659) was absent -- at ef_search 40 and 100 alike. After a rebuild
        (6 s) it ranked first at the default. So after adding rows to a large index,
        rebuild. A restored dump always builds the index fresh, so the agent's
        in-container copy never needs this.

        Single-process on purpose: a parallel build sizes a shared-memory segment
        well beyond the 64 MB /dev/shm containers default to, and fails.
        """

        def op(cur: Any) -> None:
            cur.execute("SET LOCAL max_parallel_maintenance_workers = 0")
            cur.execute("SET LOCAL maintenance_work_mem = '512MB'")
            cur.execute("REINDEX INDEX ix_chunks_embedding")

        started = time.perf_counter()
        await self._db.run(op)
        _log.info("pg_vector_index_rebuilt", seconds=round(time.perf_counter() - started, 2))


class PgKeywordIndex(_ChunkStore):
    """Sparse retrieval + the canonical chunk store — mirrors ``KeywordIndex``."""

    async def add(self, entries: list[tuple[str, str, dict[str, Any]]]) -> int:
        if not entries:
            return 0
        from psycopg.types.json import Jsonb

        rows = [
            (chunk_id, scrub_nuls(text), Jsonb(scrub_nuls(payload)))
            for chunk_id, text, payload in entries
        ]
        return await self._db.executemany(
            "INSERT INTO chunks (chunk_id, text, payload) VALUES (%s, %s, %s) "
            "ON CONFLICT (chunk_id) DO UPDATE SET "
            "text = EXCLUDED.text, payload = EXCLUDED.payload",
            rows,
        )

    async def update_payload(self, document_id: str, fields: Mapping[str, Any]) -> int:
        return await self._merge_payload(document_id, fields)

    async def search(
        self, query: str, *, limit: int, filters: FacetFilters | None = None
    ) -> list[tuple[str, float]]:
        tokens = _TOKEN_RE.findall(query.lower())
        if not tokens:
            return []
        facets = {k: vals for k, v in (filters or {}).items() if (vals := accepted_values(v))}
        facet_sql, facet_params = _facet_sql(facets)
        # Two tiers: chunks containing EVERY query term first, then any-term matches
        # to fill up. A plain OR query ranked by ts_rank_cd has no IDF, so common
        # words swamp it: on a seeded FOXTROT/BRAVO benchmark the passage a query was taken
        # from was among the top 30 only ~50% of the time (SQLite FTS5's BM25: 96%).
        # AND-first with length-normalised ts_rank: 92-96%, and faster.
        # Tokens are [A-Za-z0-9]+ by construction, so joining them into a tsquery
        # cannot inject operators.
        any_term = " | ".join(tokens)
        all_terms = " ".join(tokens)
        tier = (
            "SELECT chunk_id, {tier} AS tier, ts_rank(tsv, q, 1) AS score "
            f"FROM chunks, {{fn}}('{_TS_CONFIG}', %s) AS q "
            f"WHERE tsv @@ q{facet_sql} ORDER BY score DESC LIMIT %s"
        )
        rows = await self._db.fetch(
            f"SELECT chunk_id, min(tier) AS tier, max(score) AS score FROM ("
            f"({tier.format(tier=1, fn='plainto_tsquery')}) UNION ALL "
            f"({tier.format(tier=2, fn='to_tsquery')})"
            ") AS hits GROUP BY chunk_id ORDER BY tier, score DESC, chunk_id LIMIT %s",
            (all_terms, *facet_params, limit, any_term, *facet_params, limit, limit),
        )
        # ts_rank with normalisation 1 is < 1, so +1 keeps every all-terms match
        # scored above every any-term match for callers that sort by score.
        return [
            (r["chunk_id"], float(r["score"]) + (1.0 if r["tier"] == 1 else 0.0)) for r in rows
        ]

    async def get(self, chunk_ids: list[str]) -> dict[str, tuple[str, dict[str, Any]]]:
        if not chunk_ids:
            return {}
        rows = await self._db.fetch(
            "SELECT chunk_id, text, payload FROM chunks WHERE chunk_id = ANY(%s)",
            (list(chunk_ids),),
        )
        return {r["chunk_id"]: (r["text"], dict(r["payload"] or {})) for r in rows}

    async def exact_lookup(
        self, identifiers: list[str], *, limit: int = 8
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """Literal substring match on chunk text OR source filename — see ``KeywordIndex``."""
        ids = [i for i in {i.strip() for i in identifiers} if len(i.strip()) >= 4]
        if not ids:
            return []
        clauses, params = [], []
        for ident in ids:
            clauses.append("(text ILIKE %s OR payload->>'source_key' ILIKE %s)")
            params += [f"%{ident}%", f"%{ident}%"]
        rows = await self._db.fetch(
            f"SELECT chunk_id, text, payload FROM chunks "
            f"WHERE {' OR '.join(clauses)} LIMIT %s",
            (*params, limit),
        )
        return [(r["chunk_id"], r["text"], dict(r["payload"] or {})) for r in rows]

    async def count(self) -> int:
        row = await self._db.fetchone("SELECT COUNT(*) AS n FROM chunks")
        return int((row or {}).get("n", 0))


__all__ = ["PgDatabase", "PgKeywordIndex", "PgVectorIndex"]
