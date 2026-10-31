"""Local hybrid search over compliance documents (platform services #3,#5,#6,#7,#9,#10 — MVP).

A self-contained, free/local retrieval app that closes the loop from documents to
answers-to-a-query without any managed database:

    S3 (foxtrot/) -> process (extract) -> chunk (parent/child) -> embed (Titan)
             -> index: Qdrant (embedded, on-disk) + SQLite FTS5 (keyword)
    query    -> [vector search || keyword search] -> Reciprocal Rank Fusion
             -> ranked passages (with source document + page) -> web UI

Storage is 100% local and free: Qdrant runs in embedded mode as a directory on
disk (no server, no Docker), and full-text is SQLite FTS5 from the standard
library. No Pinecone, no OpenSearch Serverless.
"""

from __future__ import annotations

__version__ = "0.1.0"
