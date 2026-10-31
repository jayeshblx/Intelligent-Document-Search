"""Tests for Reciprocal Rank Fusion."""

from __future__ import annotations

from rag_search.fusion import reciprocal_rank_fusion


def test_rrf_rewards_agreement() -> None:
    # "b" appears high in both lists -> should win.
    vector = ["a", "b", "c"]
    keyword = ["b", "d", "a"]
    fused = reciprocal_rank_fusion([vector, keyword], k=60)
    ranked = [cid for cid, _ in fused]
    assert ranked[0] == "b"
    assert set(ranked) == {"a", "b", "c", "d"}


def test_rrf_single_list_preserves_order() -> None:
    fused = reciprocal_rank_fusion([["x", "y", "z"]], k=60)
    assert [cid for cid, _ in fused] == ["x", "y", "z"]


def test_rrf_scores_descending() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["a", "c"]])
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)
    assert fused[0][0] == "a"  # appears in both, rank 0 in both
