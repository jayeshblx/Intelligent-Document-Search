"""Reciprocal Rank Fusion (platform capability #10).

RRF combines several ranked lists without needing comparable scores: a chunk's
fused score is the sum over lists of ``1 / (k + rank)``. It's robust, tuning-free
(one constant ``k``), and the standard way to blend dense (vector) and sparse
(BM25) retrieval — each catches matches the other misses.
"""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]], *, k: int = 60
) -> list[tuple[str, float]]:
    """Fuse ranked id lists into one ranking (best-first).

    Args:
        ranked_lists: each inner list is chunk ids ordered best-first.
        k: damping constant (larger = flatter contribution from top ranks).

    Returns:
        ``(chunk_id, fused_score)`` sorted by score descending.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
