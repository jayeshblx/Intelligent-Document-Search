"""MongoDB Atlas backend for the chunk index — vectors AND keyword in one collection.

Drop-in replacement for the embedded-Qdrant + SQLite-FTS5 pair (and for Postgres).
``MongoVectorIndex`` and ``MongoKeywordIndex`` expose exactly the surfaces of
``LocalVectorIndex`` and ``KeywordIndex``, and ``MongoMetadataStore`` (in
``mongo_metadata``) mirrors ``MetadataStore``. All three read and write ONE
collection per tenant, e.g. ``EX_DOCUMENTSEARCH_DEV.FOXTROT_DocumentSearch_DEV``, whose
documents are told apart by ``kind``:

    kind="chunk"     _id=chunk_id, document_id, text, payload, facet_kv, embedding
    kind="document"  _id="meta::<document_id>", namespace, source_key, payload,
                     facets, facet_pairs[{k,v,lv}], hay, effective/expiry_date
    kind="manifest"  _id="_manifest" -- what the agent used to read from S3

Why Atlas rather than a snapshot the agent downloads: the index is LIVE. An agent
container opens a connection and queries -- no multi-hundred-MB download, no
Qdrant load, no pg_restore -- so there is no cold start to speak of, and a
re-index is visible to every running agent without a republish.

Retrieval:

* Dense  -- ``$vectorSearch`` over the ``chunk_vector`` index (cosine, HNSW).
  Atlas reports cosine as ``(1 + cos) / 2``; it is mapped back to plain cosine so
  scores stay comparable with Qdrant/pgvector.
* Sparse -- ``$search`` over the ``chunk_text`` index: Lucene BM25 with the
  ``lucene.english`` analyzer (Porter stemming), the same ranking family as the
  SQLite FTS5 ``bm25()`` it replaces.

Facet filters: Atlas only filters on fields declared in the search index, and the
facet keys are open-ended (Mongo source fields become facets), so every scalar
payload value is also written to ``facet_kv`` as ``"key=value"``. One declared
field then serves every facet, in both indexes. List values contribute one term
per element, matching Qdrant's any-element semantics.

Threading model: pymongo's synchronous client driven through ``asyncio.to_thread``,
matching how the SQLite, Qdrant-local and Postgres stores already work.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypeVar

from rag_core.logging import get_logger
from rag_search.facet_filters import FacetFilters, accepted_values
from rag_search.stores.vector_local import VectorMatch

_log = get_logger("rag_search.mongo")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_T = TypeVar("_T")

CHUNK = "chunk"
DOCUMENT = "document"
MANIFEST = "manifest"
MANIFEST_ID = "_manifest"

VECTOR_INDEX = "chunk_vector"
TEXT_INDEX = "chunk_text"

# Payload keys never worth filtering on: the chunk body is the big one.
_NOT_FACETS = frozenset({"text"})
# Longer values are prose, not facets, and would bloat both search indexes.
_MAX_FACET_CHARS = 512
# Vector candidates examined per result when searching approximately. Generous on
# purpose: an index of tens of thousands of vectors makes this cheap, and recall --
# not latency -- is what separates "same results as Qdrant" from "close".
_CANDIDATES_PER_RESULT = 20
_MIN_CANDIDATES = 400
_MAX_CANDIDATES = 10_000
# Candidates fetched per requested result and rescored on full-precision vectors.
_RESCORE_FACTOR = 4
_SERVER_SELECTION_TIMEOUT_MS = 30_000
_WRITE_BATCH = 500
# BM25 term statistics cache (see MongoIndexDatabase.term_stats).
_TERM_STATS_TTL_S = 600.0
_TERM_STATS_MAX = 50_000


def facet_terms(payload: Mapping[str, Any]) -> list[str]:
    """The ``"key=value"`` terms a chunk can be filtered by.

    Values are rendered the way a caller's filter arrives -- as text -- so the
    filter side can build the same term with :func:`facet_term`.
    """
    terms: set[str] = set()
    for key, value in payload.items():
        if key in _NOT_FACETS:
            continue
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, bool):
                text = "true" if item else "false"
            elif isinstance(item, (str, int, float)):
                text = str(item)
            else:
                continue
            if text and len(text) <= _MAX_FACET_CHARS:
                terms.add(facet_term(key, text))
    return sorted(terms)


def facet_term(key: str, value: str) -> str:
    return f"{key}={value}"


def _active_facets(filters: FacetFilters | None) -> dict[str, list[str]]:
    return {k: vals for k, v in (filters or {}).items() if (vals := accepted_values(v))}


def vector_filter(filters: FacetFilters | None) -> dict[str, Any] | None:
    """``$vectorSearch.filter`` for a facet selection: any-of within a facet, AND across."""
    clauses = [
        {"facet_kv": {"$in": [facet_term(k, v) for v in vals]}}
        for k, vals in _active_facets(filters).items()
    ]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def text_filter(filters: FacetFilters | None) -> list[dict[str, Any]]:
    """``$search`` compound ``filter`` clauses: chunks only, plus the facet selection."""
    clauses: list[dict[str, Any]] = [{"equals": {"path": "kind", "value": CHUNK}}]
    for key, vals in _active_facets(filters).items():
        clauses.append({"in": {"path": "facet_kv", "value": [facet_term(key, v) for v in vals]}})
    return clauses


def bm25_clauses(
    tokens: list[str], total: int, doc_freq: Mapping[str, int]
) -> list[dict[str, Any]]:
    """Per-term ``$search`` clauses whose weights reproduce SQLite FTS5's ``bm25()``.

    Lucene and FTS5 share BM25's term-frequency/length part (k1=1.2, b=0.75) but
    weigh TERMS differently:

    * IDF -- Lucene uses ``ln(1 + (N-n+0.5)/(n+0.5))``; FTS5 uses
      ``ln((N-n+0.5)/(n+0.5))``, floored at 1e-6. So a word in over half the
      chunks ("the", "document") is worth ~nothing in FTS5 but still counts in
      Lucene, which reorders every result list it touches.
    * Repeats -- FTS5 scores ``"a" OR "b" OR "a"`` with ``a`` twice.

    Boosting each term by ``count * idf_fts5 / idf_lucene`` cancels Lucene's IDF
    and applies FTS5's, so the two engines rank alike.
    """
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    clauses = []
    for token, count in counts.items():
        n = doc_freq.get(token, 0)
        if n <= 0 or total <= 0:
            continue
        ratio = (total - n + 0.5) / (n + 0.5)
        idf_fts5 = max(math.log(ratio), 1e-6) if ratio > 0 else 1e-6
        idf_lucene = math.log(1.0 + ratio)
        clauses.append(
            {
                "text": {
                    "query": token,
                    "path": "text",
                    "score": {"boost": {"value": count * idf_fts5 / idf_lucene}},
                }
            }
        )
    return clauses


def vector_index_definition(dim: int) -> dict[str, Any]:
    return {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": dim,
                "similarity": "cosine",
                # int8 copies for the HNSW graph: ~4x less memory, so the index stays
                # resident on a small cluster instead of being re-read from disk. Atlas
                # keeps the full-fidelity vectors for exact search.
                "quantization": "scalar",
            },
            {"type": "filter", "path": "facet_kv"},
        ]
    }


def _definition_fields(index: Mapping[str, Any]) -> list[Any]:
    definition = index.get("latestDefinition") or index.get("definition") or {}
    return list(definition.get("fields") or [])


# Tokenises exactly like SQLite FTS5's `porter unicode61` tokenizer, which the
# local index used: split on anything that is not a letter or digit, fold case and
# diacritics, Porter-stem, keep stop words. Lucene's stock analyzers differ in ways
# that change rankings -- `lucene.english` drops stop words and keeps
# "acmefoods.example.com" as ONE token, so a search for "acmefoods" missed it.
TEXT_ANALYZER = "fts5_porter_unicode61"


def text_index_definition() -> dict[str, Any]:
    return {
        "analyzers": [
            {
                "name": TEXT_ANALYZER,
                "charFilters": [],
                "tokenizer": {"type": "regexSplit", "pattern": "[^\\p{L}\\p{N}]+"},
                "tokenFilters": [
                    {"type": "icuFolding"},
                    {"type": "lowercase"},
                    {"type": "porterStemming"},
                ],
            }
        ],
        "mappings": {
            "dynamic": False,
            "fields": {
                "text": {"type": "string", "analyzer": TEXT_ANALYZER},
                "kind": {"type": "token"},
                "facet_kv": {"type": "token"},
            },
        },
    }


def _text_analyzer_of(index: Mapping[str, Any]) -> str | None:
    definition = index.get("latestDefinition") or index.get("definition") or {}
    return (((definition.get("mappings") or {}).get("fields") or {}).get("text") or {}).get(
        "analyzer"
    )


# Regular (B-tree) indexes. Every query names `kind` first, so it leads each one.
_REGULAR_INDEXES: tuple[tuple[str, list[tuple[str, int]]], ...] = (
    ("ix_kind_document", [("kind", 1), ("document_id", 1)]),
    ("ix_facet_pairs", [("kind", 1), ("namespace", 1), ("facet_pairs.k", 1), ("facet_pairs.v", 1)]),
    ("ix_facet_lower", [("kind", 1), ("namespace", 1), ("facet_pairs.lv", 1)]),
    ("ix_source_key", [("kind", 1), ("namespace", 1), ("source_key", 1)]),
    ("ix_expiry", [("kind", 1), ("namespace", 1), ("expiry_date", 1), ("source_key", 1)]),
)


def embedding_value(vector: Iterable[float]) -> Any:
    """BSON float32 vector: half the size of an array of doubles, natively indexed."""
    from bson.binary import Binary, BinaryVectorDtype

    return Binary.from_vector([float(v) for v in vector], BinaryVectorDtype.FLOAT32)


class MongoIndexDatabase:
    """One tenant's collection, shared by the vector, keyword and metadata stores.

    Pass ``client`` to share one connection pool across tenants (the agent does);
    otherwise a client is opened from ``uri`` and owned -- closed by ``aclose``.
    """

    def __init__(
        self,
        uri: str = "",
        *,
        database: str,
        collection: str,
        dim: int,
        client: Any = None,
        tls_ca_file: str | None = None,
        exact_vector_search: bool = False,
        pool_max: int = 20,
        query_timeout_ms: int = 10_000,
    ) -> None:
        if client is None and not uri:
            raise ValueError("MongoIndexDatabase needs a connection string or a client")
        if not database or not collection:
            raise ValueError("MongoIndexDatabase needs a database and a collection name")
        self._uri = uri
        self._database = database
        self._collection_name = collection
        self._dim = dim
        self._client = client
        self._owns_client = client is None
        self._tls_ca_file = tls_ca_file
        self._exact = exact_vector_search
        self._pool_max = pool_max
        self._query_timeout_ms = query_timeout_ms
        self._lock = threading.Lock()
        self._schema_ready = False
        self._closed = False
        # BM25 term statistics: token -> (fetched_at, doc frequency), plus the chunk
        # total. They drift only as fast as the index changes, so minutes-old values
        # rank identically; a query pays for a lookup only on a token it has not seen.
        self._df: dict[str, tuple[float, int]] = {}
        self._total: tuple[float, int] | None = None

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def database_name(self) -> str:
        return self._database

    @property
    def exact_vector_search(self) -> bool:
        return self._exact

    @property
    def query_timeout_ms(self) -> int:
        """Server-side cap on one search query: a stalled query fails fast, so the
        caller can fall back to the other retriever instead of timing the user out."""
        return self._query_timeout_ms

    # ---- plumbing ---------------------------------------------------------

    def _coll(self) -> Any:
        with self._lock:
            if self._closed:
                raise RuntimeError("MongoIndexDatabase is closed")
            if self._client is None:
                self._client = open_client(
                    self._uri, tls_ca_file=self._tls_ca_file, pool_max=self._pool_max
                )
            return self._client[self._database][self._collection_name]

    async def run(self, fn: Callable[[Any], _T]) -> _T:
        """Run ``fn(collection)`` in a worker thread."""
        return await asyncio.to_thread(lambda: fn(self._coll()))

    # ---- schema -----------------------------------------------------------

    async def ensure_schema(self) -> None:
        """Create the collection, its regular indexes and both search indexes.

        Idempotent and cheap once done. Only WRITERS call this (the indexer and the
        migration script); readers never create anything. Search indexes build
        asynchronously on Atlas -- see :meth:`wait_until_searchable`.
        """
        if self._schema_ready:
            return
        await self.run(self._ensure_schema_sync)
        self._schema_ready = True

    def _ensure_schema_sync(self, coll: Any) -> None:
        from pymongo.errors import CollectionInvalid
        from pymongo.operations import SearchIndexModel

        db = coll.database
        if self._collection_name not in db.list_collection_names(
            filter={"name": self._collection_name}
        ):
            # Search indexes can only be created on a collection that exists.
            with contextlib.suppress(CollectionInvalid):  # created concurrently
                db.create_collection(self._collection_name)

        for name, keys in _REGULAR_INDEXES:
            coll.create_index(keys, name=name)

        existing = {ix["name"]: ix for ix in coll.list_search_indexes()}
        vector_ix = existing.get(VECTOR_INDEX)
        if vector_ix is not None:
            declared = _declared_dim(vector_ix)
            if declared is not None and declared != self._dim:
                raise RuntimeError(
                    f"Collection {self._collection_name!r} indexes {declared}-dimensional "
                    f"vectors but this service embeds at {self._dim}. Either set "
                    f"SEARCH_EMBED_DIM={declared} or drop the collection and re-index."
                )
        models = []
        if vector_ix is None:
            models.append(
                SearchIndexModel(
                    definition=vector_index_definition(self._dim),
                    name=VECTOR_INDEX,
                    type="vectorSearch",
                )
            )
        elif _definition_fields(vector_ix) != vector_index_definition(self._dim)["fields"]:
            # Same dimension, different settings (e.g. quantization): rebuilt in place,
            # with the old index serving queries until the new one is ready.
            coll.update_search_index(VECTOR_INDEX, vector_index_definition(self._dim))
            _log.info("mongo_vector_index_updated", collection=self._collection_name)
        if TEXT_INDEX not in existing:
            models.append(
                SearchIndexModel(definition=text_index_definition(), name=TEXT_INDEX, type="search")
            )
        elif _text_analyzer_of(existing[TEXT_INDEX]) != TEXT_ANALYZER:
            # An index built with an older analyzer is rebuilt in place; Atlas keeps
            # serving the old one until the new build is ready.
            coll.update_search_index(TEXT_INDEX, text_index_definition())
            _log.info("mongo_text_index_updated", collection=self._collection_name)
        if models:
            coll.create_search_indexes(models)
            _log.info(
                "mongo_search_indexes_created",
                collection=self._collection_name,
                indexes=[m.document["name"] for m in models],
            )
        _log.info("mongo_schema_ready", collection=self._collection_name, dim=self._dim)

    async def search_index_status(self) -> dict[str, dict[str, Any]]:
        """``{name: {"status": ..., "queryable": ...}}`` for both search indexes."""

        def _op(coll: Any) -> dict[str, dict[str, Any]]:
            return {
                ix["name"]: {"status": ix.get("status"), "queryable": bool(ix.get("queryable"))}
                for ix in coll.list_search_indexes()
            }

        return await self.run(_op)

    async def wait_until_searchable(self, *, timeout_s: float = 900.0, poll_s: float = 5.0) -> None:
        """Block until both search indexes are built AND have caught up with writes.

        Atlas indexes asynchronously: a just-written chunk becomes searchable a
        moment later. "READY" means the index exists; caught up means
        ``$search`` sees as many chunks as the collection holds.
        """
        from pymongo.errors import OperationFailure

        deadline = time.monotonic() + timeout_s
        status: dict[str, dict[str, Any]] = {}
        while True:
            try:
                status = await self.search_index_status()
                ready = all(
                    status.get(name, {}).get("status") == "READY"
                    and status.get(name, {}).get("queryable")
                    for name in (VECTOR_INDEX, TEXT_INDEX)
                )
                if (
                    ready
                    and await self._text_index_caught_up()
                    and await self._vector_index_caught_up()
                ):
                    return
            except OperationFailure as exc:
                # A rebuilding index (e.g. after a rename) can report queryable while
                # still in INITIAL_SYNC and reject queries, and Atlas's search-index
                # service drops connections now and then: not ready yet, keep polling.
                _log.info("mongo_search_index_syncing", error=str(exc)[:200])
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"search indexes on {self._collection_name!r} not ready after "
                    f"{timeout_s:.0f}s: {status}"
                )
            await asyncio.sleep(poll_s)

    async def _text_index_caught_up(self) -> bool:
        def _op(coll: Any) -> bool:
            # Count chunks WITH text on both sides: a chunk is often written twice
            # (vector first, text second), and counting documents alone would call
            # the index caught up before the text write has reached it.
            stored = coll.count_documents({"kind": CHUNK, "text": {"$exists": True}})
            if stored == 0:
                return True
            meta = list(
                coll.aggregate(
                    [
                        {
                            "$searchMeta": {
                                "index": TEXT_INDEX,
                                "compound": {
                                    "filter": [
                                        {"equals": {"path": "kind", "value": CHUNK}},
                                        {"exists": {"path": "text"}},
                                    ]
                                },
                                "count": {"type": "total"},
                            }
                        }
                    ]
                )
            )
            indexed = int(((meta[0] if meta else {}).get("count") or {}).get("total", 0))
            return indexed >= stored

        return await self.run(_op)

    async def _vector_index_caught_up(self) -> bool:
        """An exact ``$vectorSearch`` must reach every embedded chunk.

        Vector Search has no count operator, so this asks for as many neighbours
        as there are embeddings (capped at Atlas's 10,000 limit) and counts them.
        A brute-force scan -- fine for a one-off wait, never on the query path.
        """

        def _op(coll: Any) -> bool:
            stored = coll.count_documents({"kind": CHUNK, "embedding": {"$exists": True}})
            if stored == 0:
                return True
            want = min(stored, _MAX_CANDIDATES)
            probe = [0.0] * self._dim
            probe[0] = 1.0
            rows = list(
                coll.aggregate(
                    [
                        {
                            "$vectorSearch": {
                                "index": VECTOR_INDEX,
                                "path": "embedding",
                                "queryVector": probe,
                                "exact": True,
                                "limit": want,
                            }
                        },
                        {"$project": {"_id": 1}},
                    ]
                )
            )
            return len(rows) >= want

        return await self.run(_op)

    # ---- manifest ---------------------------------------------------------

    async def read_manifest(self) -> dict[str, Any]:
        def _op(coll: Any) -> dict[str, Any]:
            doc = coll.find_one({"_id": MANIFEST_ID}) or {}
            doc.pop("_id", None)
            doc.pop("kind", None)
            return doc

        return await self.run(_op)

    async def write_manifest(self, manifest: Mapping[str, Any]) -> None:
        body = {k: v for k, v in manifest.items() if k not in ("_id", "kind")}

        def _op(coll: Any) -> None:
            coll.replace_one({"_id": MANIFEST_ID}, {"kind": MANIFEST, **body}, upsert=True)

        await self.run(_op)

    async def counts(self) -> dict[str, int]:
        def _op(coll: Any) -> dict[str, int]:
            return {
                "chunks": coll.count_documents({"kind": CHUNK}),
                "embedded_chunks": coll.count_documents(
                    {"kind": CHUNK, "embedding": {"$exists": True}}
                ),
                "document_metadata": coll.count_documents({"kind": DOCUMENT}),
            }

        return await self.run(_op)

    async def term_stats(self, tokens: list[str]) -> tuple[int, dict[str, int]]:
        """``(chunks with text, {token: chunks containing it})`` for BM25 weighting.

        Counted by the text index itself, so a token matches exactly what the query
        will match (same tokenizer, same stemming). Cached for ``_TERM_STATS_TTL_S``.
        """
        now = time.monotonic()
        distinct = list(dict.fromkeys(tokens))
        stale = [
            t
            for t in distinct
            if (hit := self._df.get(t)) is None or now - hit[0] > _TERM_STATS_TTL_S
        ]
        need_total = self._total is None or now - self._total[0] > _TERM_STATS_TTL_S

        if stale or need_total:
            jobs = [
                self.run(
                    lambda coll, t=t: _search_count(coll, {"text": {"query": t, "path": "text"}})
                )
                for t in stale
            ]
            if need_total:
                jobs.append(
                    self.run(lambda coll: _search_count(coll, {"exists": {"path": "text"}}))
                )
            counted = await asyncio.gather(*jobs)
            if len(self._df) > _TERM_STATS_MAX:
                self._df.clear()
            for token, n in zip(stale, counted, strict=False):
                self._df[token] = (now, n)
            if need_total:
                self._total = (now, counted[-1])
        assert self._total is not None
        return self._total[1], {t: self._df[t][1] for t in distinct}

    async def has_chunks(self) -> bool:
        """Whether the collection holds any chunk -- one indexed lookup.

        Use this, not :meth:`counts`, on any path a user waits on: counting
        embedded chunks has to read every chunk document off disk.
        """
        return await self.run(
            lambda coll: (
                coll.find_one({"kind": CHUNK}, {"_id": 1}, hint="ix_kind_document") is not None
            )
        )

    async def ping(self) -> None:
        """Open the connection pool now (TLS + auth), so the first query doesn't pay."""
        await self.run(lambda coll: coll.database.command("ping"))

    async def aclose(self) -> None:
        """Idempotent -- every store sharing this database calls it on shutdown."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            client, self._client = self._client, None
        if client is not None and self._owns_client:
            await asyncio.to_thread(client.close)


def open_client(uri: str, *, tls_ca_file: str | None = None, pool_max: int = 20) -> Any:
    """A pymongo client with the settings every index connection uses."""
    from pymongo import MongoClient

    kwargs: dict[str, Any] = {
        "serverSelectionTimeoutMS": _SERVER_SELECTION_TIMEOUT_MS,
        "maxPoolSize": pool_max,
        # Keep a connection open between requests: a fresh one costs a DNS SRV
        # lookup + TLS handshake + auth, which is what made first requests slow.
        "minPoolSize": 1,
        "appname": "rag-document-search",
    }
    if tls_ca_file:
        kwargs["tlsCAFile"] = tls_ca_file
    return MongoClient(uri, **kwargs)


def _search_count(coll: Any, operator: Mapping[str, Any]) -> int:
    """Exact number of chunks the text index matches for one operator."""
    rows = list(
        coll.aggregate(
            [
                {
                    "$searchMeta": {
                        "index": TEXT_INDEX,
                        "compound": {
                            "must": [dict(operator)],
                            "filter": [{"equals": {"path": "kind", "value": CHUNK}}],
                        },
                        "count": {"type": "total"},
                    }
                }
            ]
        )
    )
    return int(((rows[0] if rows else {}).get("count") or {}).get("total", 0))


def _rescore(query: list[float], rows: list[dict[str, Any]]) -> list[tuple[float, dict[str, Any]]]:
    """Exact cosine of each candidate's stored float32 vector, best first.

    Python's sort is stable, so candidates that tie exactly (duplicate chunks)
    keep Atlas's order. numpy when available (it is, via qdrant-client); a plain
    Python fallback otherwise -- ~120 x 1024 products either way.
    """
    if not rows:
        return []
    vectors = [r["embedding"].as_vector().data for r in rows]
    try:
        import numpy as np

        q = np.asarray(query, dtype=np.float32)
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(q) or 1.0)
        norms[norms == 0] = 1.0
        cosines = [float(c) for c in (matrix @ q) / norms]
    except ImportError:
        q_norm = math.sqrt(sum(x * x for x in query)) or 1.0
        cosines = []
        for v in vectors:
            v_norm = math.sqrt(sum(x * x for x in v)) or 1.0
            cosines.append(sum(a * b for a, b in zip(query, v, strict=False)) / (q_norm * v_norm))
    order = sorted(range(len(rows)), key=lambda i: -cosines[i])
    return [(cosines[i], rows[i]) for i in order]


def _declared_dim(index: Mapping[str, Any]) -> int | None:
    definition = index.get("latestDefinition") or index.get("definition") or {}
    for field in definition.get("fields") or []:
        if field.get("type") == "vector":
            return int(field.get("numDimensions") or 0) or None
    return None


class _ChunkStore:
    """Shared behaviour of the two facades over ``kind="chunk"`` documents."""

    def __init__(self, db: MongoIndexDatabase) -> None:
        self._db = db

    async def reset(self) -> None:
        await self._db.ensure_schema()
        await self._db.run(lambda coll: coll.delete_many({"kind": CHUNK}))

    async def _merge_payload(self, document_id: str, fields: Mapping[str, Any]) -> int:
        """Merge facet fields into every chunk of one document. ``None`` removes a key."""
        if not fields:
            return 0
        setting = {k: v for k, v in fields.items() if v is not None}
        clearing = [k for k, v in fields.items() if v is None]

        def _op(coll: Any) -> int:
            from pymongo import UpdateOne

            ops = []
            for doc in coll.find({"kind": CHUNK, "document_id": document_id}, {"payload": 1}):
                payload = dict(doc.get("payload") or {})
                payload.update(setting)
                for key in clearing:
                    payload.pop(key, None)
                ops.append(
                    UpdateOne(
                        {"_id": doc["_id"]},
                        {"$set": {"payload": payload, "facet_kv": facet_terms(payload)}},
                    )
                )
            if ops:
                coll.bulk_write(ops, ordered=False)
            return len(ops)

        return await self._db.run(_op)

    async def _write(self, ops: list[Any]) -> int:
        if not ops:
            return 0

        def _op(coll: Any) -> int:
            for start in range(0, len(ops), _WRITE_BATCH):
                coll.bulk_write(ops[start : start + _WRITE_BATCH], ordered=False)
            return len(ops)

        return await self._db.run(_op)

    async def delete_document(self, document_id: str) -> int:
        result = await self._db.run(
            lambda coll: coll.delete_many({"kind": CHUNK, "document_id": document_id})
        )
        return int(result.deleted_count)

    async def aclose(self) -> None:
        await self._db.aclose()


def _chunk_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": CHUNK,
        "document_id": payload.get("document_id"),
        "payload": dict(payload),
        "facet_kv": facet_terms(payload),
    }


class MongoVectorIndex(_ChunkStore):
    """Dense retrieval over Atlas Vector Search -- mirrors ``LocalVectorIndex``."""

    async def ensure(self) -> None:
        # Fails fast on a bad connection string rather than mid-document.
        await self._db.ensure_schema()

    async def upsert(self, points: list[tuple[str, list[float], dict[str, Any]]]) -> int:
        from pymongo import UpdateOne

        return await self._write(
            [
                UpdateOne(
                    {"_id": chunk_id},
                    {"$set": {**_chunk_fields(payload), "embedding": embedding_value(vec)}},
                    upsert=True,
                )
                for chunk_id, vec, payload in points
            ]
        )

    async def set_document_payload(self, document_id: str, fields: Mapping[str, Any]) -> None:
        await self._merge_payload(document_id, fields)

    async def search(
        self, vector: list[float], *, limit: int, filters: FacetFilters | None = None
    ) -> list[VectorMatch]:
        exact = self._db.exact_vector_search
        # The HNSW graph is int8-quantized (see vector_index_definition), so its
        # scores are approximate and would reorder near-ties. Fetch extra candidates
        # and rescore them on the full-precision vectors instead.
        fetch = limit if exact else min(_MAX_CANDIDATES, limit * _RESCORE_FACTOR)
        stage: dict[str, Any] = {
            "index": VECTOR_INDEX,
            "path": "embedding",
            "queryVector": [float(v) for v in vector],
            "limit": fetch,
        }
        if exact:
            stage["exact"] = True
        else:
            stage["numCandidates"] = min(
                _MAX_CANDIDATES, max(_MIN_CANDIDATES, fetch * _CANDIDATES_PER_RESULT)
            )
        if (flt := vector_filter(filters)) is not None:
            stage["filter"] = flt
        project: dict[str, Any] = {"payload": 1, "score": {"$meta": "vectorSearchScore"}}
        if not exact:
            project["embedding"] = 1
        pipeline = [{"$vectorSearch": stage}, {"$project": project}]
        timeout = self._db.query_timeout_ms
        rows = await self._db.run(lambda coll: list(coll.aggregate(pipeline, maxTimeMS=timeout)))
        if exact:
            # Atlas normalises cosine to (1 + cos) / 2; report plain cosine like Qdrant.
            scored = [(2.0 * float(r["score"]) - 1.0, r) for r in rows]
        else:
            scored = _rescore(vector, rows)
        return [
            VectorMatch(chunk_id=str(r["_id"]), score=score, payload=dict(r.get("payload") or {}))
            for score, r in scored[:limit]
        ]

    async def fetch_document(
        self, document_id: str, *, namespace: str = "default", limit: int = 60
    ) -> list[dict[str, Any]]:
        rows = await self._db.run(
            lambda coll: list(
                coll.find({"kind": CHUNK, "document_id": document_id}, {"payload": 1}).limit(limit)
            )
        )
        payloads = []
        for r in rows:
            payload = dict(r.get("payload") or {})
            payload["_chunk_id"] = str(r["_id"])
            payloads.append(payload)
        payloads.sort(key=lambda pl: (pl.get("page_start") or 0, pl.get("_chunk_id", "")))
        return payloads

    async def count(self) -> int:
        return await self._db.run(
            lambda coll: coll.count_documents({"kind": CHUNK, "embedding": {"$exists": True}})
        )


class MongoKeywordIndex(_ChunkStore):
    """Sparse retrieval (Atlas Search, BM25) + the canonical chunk store."""

    async def add(self, entries: list[tuple[str, str, dict[str, Any]]]) -> int:
        from pymongo import UpdateOne

        return await self._write(
            [
                UpdateOne(
                    {"_id": chunk_id},
                    {"$set": {**_chunk_fields(payload), "text": text or ""}},
                    upsert=True,
                )
                for chunk_id, text, payload in entries
            ]
        )

    async def update_payload(self, document_id: str, fields: Mapping[str, Any]) -> int:
        return await self._merge_payload(document_id, fields)

    async def search(
        self, query: str, *, limit: int, filters: FacetFilters | None = None
    ) -> list[tuple[str, float]]:
        tokens = _TOKEN_RE.findall(query.lower())
        if not tokens:
            return []
        total, doc_freq = await self._db.term_stats(tokens)
        clauses = bm25_clauses(tokens, total, doc_freq)
        if not clauses:
            return []  # no query term occurs anywhere
        pipeline = [
            {
                "$search": {
                    "index": TEXT_INDEX,
                    "compound": {
                        # One SHOULD clause per term = the FTS5 `"a" OR "b"` match.
                        "should": clauses,
                        "minimumShouldMatch": 1,
                        "filter": text_filter(filters),
                    },
                }
            },
            {"$limit": limit},
            {"$project": {"_id": 1, "score": {"$meta": "searchScore"}}},
        ]
        timeout = self._db.query_timeout_ms
        rows = await self._db.run(lambda coll: list(coll.aggregate(pipeline, maxTimeMS=timeout)))
        return [(str(r["_id"]), float(r["score"])) for r in rows]

    async def get(self, chunk_ids: list[str]) -> dict[str, tuple[str, dict[str, Any]]]:
        if not chunk_ids:
            return {}
        rows = await self._db.run(
            lambda coll: list(
                coll.find(
                    {"_id": {"$in": list(chunk_ids)}, "kind": CHUNK}, {"text": 1, "payload": 1}
                )
            )
        )
        return {
            str(r["_id"]): (str(r.get("text") or ""), dict(r.get("payload") or {})) for r in rows
        }

    async def exact_lookup(
        self, identifiers: list[str], *, limit: int = 8
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """Literal, case-insensitive substring match on chunk text OR source filename."""
        ids = [i for i in {i.strip() for i in identifiers} if len(i.strip()) >= 4]
        if not ids:
            return []
        clauses: list[dict[str, Any]] = []
        for ident in ids:
            pattern = {"$regex": re.escape(ident), "$options": "i"}
            clauses += [{"text": pattern}, {"payload.source_key": pattern}]
        rows = await self._db.run(
            lambda coll: list(
                coll.find({"kind": CHUNK, "$or": clauses}, {"text": 1, "payload": 1}).limit(limit)
            )
        )
        return [
            (str(r["_id"]), str(r.get("text") or ""), dict(r.get("payload") or {})) for r in rows
        ]

    async def count(self) -> int:
        return await self._db.run(lambda coll: coll.count_documents({"kind": CHUNK}))


__all__ = [
    "MANIFEST_ID",
    "TEXT_INDEX",
    "VECTOR_INDEX",
    "MongoIndexDatabase",
    "MongoKeywordIndex",
    "MongoVectorIndex",
    "facet_term",
    "facet_terms",
    "open_client",
    "text_filter",
    "vector_filter",
]
