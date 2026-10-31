"""Tests for token cost accounting."""

from __future__ import annotations

from rag_search.pricing import compute_cost


def test_sonnet_rates() -> None:
    c = compute_cost(model="us.anthropic.claude-sonnet-4-5-v1:0",
                     input_tokens=1_000_000, output_tokens=1_000_000)
    assert c.input_cost_usd == 3.0
    assert c.output_cost_usd == 15.0
    assert c.total_cost_usd == 18.0


def test_haiku_and_opus_rates() -> None:
    haiku = compute_cost(model="anthropic.claude-haiku-4-5", input_tokens=1_000_000, output_tokens=0)
    assert haiku.input_cost_usd == 1.0
    opus = compute_cost(model="us.anthropic.claude-opus-4-8", input_tokens=1_000_000, output_tokens=1_000_000)
    assert opus.total_cost_usd == 30.0


def test_embed_cost_and_unknown_default() -> None:
    c = compute_cost(model="mystery-model", input_tokens=0, output_tokens=0, embed_tokens=1_000_000)
    assert c.embed_cost_usd == 0.02
    # unknown model falls back to sonnet-tier rates
    assert compute_cost(model="mystery", input_tokens=1_000_000, output_tokens=0).input_cost_usd == 3.0
