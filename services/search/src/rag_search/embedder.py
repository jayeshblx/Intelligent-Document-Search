"""Embedding providers (platform capability #5).

``TitanEmbedder`` is the production path (Bedrock Titan v2, batched with bounded
concurrency inside the Bedrock client). ``FakeEmbedder`` is a deterministic,
dependency-free hashing embedder used by tests and the offline demo — it gives
non-trivial similarity for texts that share vocabulary (so vector search is
meaningful without any model download or AWS call).

The collection's vector dimension is taken from ``Embedder.dim``, so swapping
providers needs no config change.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from contextvars import ContextVar

from rag_core.aws import BedrockClient

# Query-embedding tokens spent while serving ONE request. A contextvar, so
# concurrent requests on the same runtime cannot bleed into each other's count.
# Only `embed_query` is counted: bulk `embed()` is the indexer's batch cost, not
# something a caller pays per question.
_query_embedding_tokens: ContextVar[int] = ContextVar("query_embedding_tokens", default=0)


def reset_query_embedding_tokens() -> None:
    """Start counting afresh for a new request."""
    _query_embedding_tokens.set(0)


def query_embedding_tokens() -> int:
    """Query-embedding tokens spent since the last reset."""
    return _query_embedding_tokens.get()


def _record_query_embedding(tokens: int) -> None:
    _query_embedding_tokens.set(_query_embedding_tokens.get() + max(0, tokens))


class Embedder(ABC):
    dim: int

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed documents (batch)."""

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query (defaults to a 1-item batch)."""
        return (await self.embed([text]))[0]

    async def aclose(self) -> None:  # pragma: no cover - optional
        return None


class TitanEmbedder(Embedder):
    def __init__(self, bedrock: BedrockClient, *, dim: int = 1024, batch: int = 64) -> None:
        self._bedrock = bedrock
        self.dim = dim
        self._batch = batch

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            batch = texts[start : start + self._batch]
            results = await self._bedrock.embed_batch(batch, dimensions=self.dim)
            out.extend(r.vector for r in results)
        return out

    async def embed_query(self, text: str) -> list[float]:
        result = await self._bedrock.embed_text(text, dimensions=self.dim)
        # Billed per request: one embed call per search the agent runs.
        _record_query_embedding(result.input_tokens)
        return result.vector

    async def aclose(self) -> None:
        await self._bedrock.aclose()


class FakeEmbedder(Embedder):
    """Deterministic hashing embedder — no deps, no network, reproducible.

    Bag-of-hashed-tokens with signed buckets, L2-normalised. Two texts sharing
    words get a positive cosine similarity, which is enough for meaningful local
    vector search in tests/demos."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for token in text.lower().split():
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            v[idx] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]
