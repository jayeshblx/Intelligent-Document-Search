"""Storage backends for the chunk index and document metadata.

Three interchangeable implementations, selected by ``SEARCH_STORE_BACKEND``:

* ``local``    — embedded Qdrant (vectors) + SQLite FTS5 (keyword). No server.
* ``postgres`` — one Postgres/pgvector server for vectors, keyword AND metadata.
* ``mongo``    — one MongoDB Atlas collection per tenant for all three (Atlas
  Vector Search + Atlas Search); read live, so agents have no snapshot to load.

The Postgres classes are import-light: ``psycopg`` is only imported when a
connection is actually opened, so the local backend needs neither driver.
"""

from __future__ import annotations

from rag_search.metadata_store import MetadataStore
from rag_search.stores.keyword_fts import KeywordIndex
from rag_search.stores.mongo import MongoIndexDatabase, MongoKeywordIndex, MongoVectorIndex
from rag_search.stores.mongo_metadata import MongoMetadataStore
from rag_search.stores.postgres import PgDatabase, PgKeywordIndex, PgVectorIndex
from rag_search.stores.postgres_metadata import PgMetadataStore
from rag_search.stores.vector_local import LocalVectorIndex, VectorMatch

# Structural aliases — the services depend on the shared surface, not the backend.
VectorIndex = LocalVectorIndex | PgVectorIndex | MongoVectorIndex
KeywordSearchIndex = KeywordIndex | PgKeywordIndex | MongoKeywordIndex
DocumentMetadataStore = MetadataStore | PgMetadataStore | MongoMetadataStore

__all__ = [
    "DocumentMetadataStore",
    "KeywordIndex",
    "KeywordSearchIndex",
    "LocalVectorIndex",
    "MetadataStore",
    "MongoIndexDatabase",
    "MongoKeywordIndex",
    "MongoMetadataStore",
    "MongoVectorIndex",
    "PgDatabase",
    "PgKeywordIndex",
    "PgMetadataStore",
    "PgVectorIndex",
    "VectorIndex",
    "VectorMatch",
]
