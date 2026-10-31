"""Token pricing for cost accounting (AWS Bedrock rates, USD per 1M tokens).

Rates are matched by substring of the model id so both first-party
(``anthropic.claude-sonnet-…``) and Bedrock inference-profile ids
(``us.anthropic.claude-sonnet-4-…-v1:0``) resolve. Update as pricing changes;
override per-deployment via ``SEARCH_*`` config if needed.
"""

from __future__ import annotations

from rag_core.models.base import DomainModel

# (input $/1M, output $/1M). Ordered most-specific first.
_LLM_RATES: list[tuple[str, tuple[float, float]]] = [
    ("opus", (5.0, 25.0)),
    ("sonnet", (3.0, 15.0)),
    ("haiku", (1.0, 5.0)),
]
_DEFAULT_LLM_RATE = (3.0, 15.0)  # assume Sonnet-tier if unknown

# Embedding $/1M input tokens.
_EMBED_RATES: list[tuple[str, float]] = [
    ("titan-embed-text-v2", 0.02),
    ("titan-embed", 0.02),
]
_DEFAULT_EMBED_RATE = 0.02


# Prompt-cache multipliers on the input rate (write costs a bit more; read ~90% cheaper).
_CACHE_WRITE_MULT = 1.25
_CACHE_READ_MULT = 0.10


class CostBreakdown(DomainModel):
    """Per-request cost accounting shown to the user."""

    input_tokens: int = 0
    output_tokens: int = 0
    embed_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    embed_cost_usd: float = 0.0
    cache_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    model: str = ""


def _llm_rate(model: str) -> tuple[float, float]:
    m = model.lower()
    for key, rate in _LLM_RATES:
        if key in m:
            return rate
    return _DEFAULT_LLM_RATE


def _embed_rate(model: str) -> float:
    m = model.lower()
    for key, rate in _EMBED_RATES:
        if key in m:
            return rate
    return _DEFAULT_EMBED_RATE


def compute_cost(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    embed_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    embed_model: str = "titan-embed-text-v2",
) -> CostBreakdown:
    """Compute a cost breakdown from token counts (incl. prompt-cache read/write)."""
    in_rate, out_rate = _llm_rate(model)
    emb_rate = _embed_rate(embed_model)
    input_cost = input_tokens / 1_000_000 * in_rate
    output_cost = output_tokens / 1_000_000 * out_rate
    embed_cost = embed_tokens / 1_000_000 * emb_rate
    cache_cost = (
        cache_read_tokens / 1_000_000 * in_rate * _CACHE_READ_MULT
        + cache_write_tokens / 1_000_000 * in_rate * _CACHE_WRITE_MULT
    )
    return CostBreakdown(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        embed_tokens=embed_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        input_cost_usd=round(input_cost, 6),
        output_cost_usd=round(output_cost, 6),
        embed_cost_usd=round(embed_cost, 6),
        cache_cost_usd=round(cache_cost, 6),
        total_cost_usd=round(input_cost + output_cost + embed_cost + cache_cost, 6),
        model=model,
    )
