"""Deterministic identifier generation.

Idempotency is the foundation of a fault-tolerant pipeline: if the same S3 object
is re-discovered, or a chunk is re-embedded after a crash, it must map to the
*same* id so stores upsert rather than duplicate. We use UUIDv5 (namespaced
SHA-1) over stable inputs so ids are reproducible across processes and restarts.

Qdrant point ids must be UUIDs or unsigned ints; ``point_id`` returns the UUID
string form derived from the chunk id, keeping the vector store idempotent too.
"""

from __future__ import annotations

import uuid

# Fixed namespaces (generated once, hardcoded forever) so v5 ids are stable.
_NS_DOCUMENT = uuid.UUID("6f8d3b4e-2a1c-5d6e-9f0a-1b2c3d4e5f60")
_NS_CHUNK = uuid.UUID("7a9e4c5f-3b2d-6e7f-0a1b-2c3d4e5f6071")


def document_id(bucket: str, key: str, *, version_id: str | None = None) -> str:
    """Deterministic id for a source object.

    Including ``version_id`` (S3 object version) means a new S3 version yields a
    new document id, which is what we want for version tracking; omit it to treat
    the key as the logical document identity.
    """
    seed = f"{bucket}/{key}" + (f"@{version_id}" if version_id else "")
    return str(uuid.uuid5(_NS_DOCUMENT, seed))


def chunk_id(document_id: str, strategy: str, order_index: int) -> str:
    """Deterministic id for a chunk within a document+strategy stream."""
    return str(uuid.uuid5(_NS_CHUNK, f"{document_id}:{strategy}:{order_index}"))


def point_id(chunk_id: str) -> str:
    """Qdrant point id for a chunk. Identity mapping today (chunk ids are already
    UUID strings), wrapped so the coupling lives in one place."""
    return chunk_id


def new_message_id() -> str:
    """Random id for a transient message envelope (not idempotency-bearing)."""
    return uuid.uuid4().hex
