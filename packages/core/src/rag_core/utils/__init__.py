"""Small, dependency-free utilities: deterministic IDs and checksums."""

from __future__ import annotations

from rag_core.utils.checksum import (
    sha256_astream,
    sha256_bytes,
    sha256_file,
    sha256_stream,
)
from rag_core.utils.doctypes import infer_document_type
from rag_core.utils.ids import chunk_id, document_id, new_message_id, point_id

__all__ = [
    "chunk_id",
    "document_id",
    "infer_document_type",
    "new_message_id",
    "point_id",
    "sha256_astream",
    "sha256_bytes",
    "sha256_file",
    "sha256_stream",
]
