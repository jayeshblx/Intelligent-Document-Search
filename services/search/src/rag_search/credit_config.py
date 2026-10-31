"""Credit-metering pricing — the single source of truth for the billing model.

Every rate, markup, and conversion the credit meter uses lives here (nothing is
scattered across the store, the service, or the endpoints). Given the real token
usage returned by Claude, ``compute_charge`` turns it into a credit charge:

    base_cost    = i/1e6 * INPUT_RATE + o/1e6 * OUTPUT_RATE   (USD)
    charged      = base_cost * (1 + OVERHEAD)                 (USD, +50% markup)
    credits_used = charged / CREDIT_VALUE                     (1000 credits = $1)

All arithmetic uses ``Decimal`` so credits are stored at full precision; callers
round to 2 decimals only for display. This is intentionally separate from
``pricing.py`` (which reports raw Bedrock USD cost for the answer's cost line) —
that is provider cost, this is the customer-facing *contract* billing.
"""

from __future__ import annotations

from decimal import Decimal

from rag_core.models.base import DomainModel

# ---- The billing model (contract rates; override per-deployment via SEARCH_*) --
INPUT_RATE = Decimal("2.00")     # USD per 1,000,000 input tokens  (Sonnet contract rate)
OUTPUT_RATE = Decimal("10.00")   # USD per 1,000,000 output tokens
EMBED_RATE = Decimal("0.02")     # USD per 1,000,000 embedding tokens (query embedding)
# Prompt-cache multipliers on INPUT_RATE (Anthropic pricing): writing the cache costs a
# little more than a normal read; reading it is heavily discounted.
CACHE_WRITE_MULT = Decimal("1.25")   # first call that writes the cached prefix
CACHE_READ_MULT = Decimal("0.10")    # later calls that read it (~90% cheaper)
OVERHEAD = Decimal("0.50")       # 50% markup on the raw token cost
CREDIT_VALUE = Decimal("0.001")  # USD per credit  (1000 credits = $1.00)

_MILLION = Decimal(1_000_000)

# ---- Plan + notification thresholds (fractions of plan_total) ------------------
# Starting/renewal size of the shared credit pool, in credits. 1000 credits = $1.
DEFAULT_PLAN_TOTAL = Decimal("1000")
# "Pro" upgrade: 5x the credits (matches the "5X search" offer shown on exhaustion).
PRO_PLAN_MULTIPLIER = Decimal("5")
# Warn the user once they've consumed this fraction of their plan (>= 90% used).
WARN_USED_FRACTION = Decimal("0.90")
# Below this fraction *remaining*, the meter bar shifts to a warning colour.
LOW_REMAINING_FRACTION = Decimal("0.15")


class ChargeBreakdown(DomainModel):
    """The credit charge for one request, at full precision.

    ``base_cost``/``charged`` are USD; ``credits_used`` is in credits. Kept as
    ``Decimal`` end-to-end; serialise to float only at the API/DB boundary.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    embed_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    base_cost: Decimal = Decimal(0)
    charged: Decimal = Decimal(0)
    credits_used: Decimal = Decimal(0)


def compute_charge(
    input_tokens: int,
    output_tokens: int,
    embed_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> ChargeBreakdown:
    """Turn real token usage into a credit charge (see module docstring).

    Covers every token the request spent: ordinary input + output, the query embedding,
    and — when prompt caching is active — the cache-write and (discounted) cache-read
    tokens. ``input_tokens`` here is the *uncached* input the provider billed at full
    rate; cached tokens are priced separately via the multipliers above."""
    i = Decimal(int(input_tokens))
    o = Decimal(int(output_tokens))
    e = Decimal(int(embed_tokens))
    cr = Decimal(int(cache_read_tokens))
    cw = Decimal(int(cache_write_tokens))
    base_cost = (
        (i / _MILLION) * INPUT_RATE
        + (o / _MILLION) * OUTPUT_RATE
        + (e / _MILLION) * EMBED_RATE
        + (cr / _MILLION) * INPUT_RATE * CACHE_READ_MULT
        + (cw / _MILLION) * INPUT_RATE * CACHE_WRITE_MULT
    )
    # Overhead disabled: users get the full $1 of credits with no markup, so the charge is
    # the raw token cost. Re-enable by uncommenting the line below (and commenting the next).
    # charged = base_cost * (Decimal(1) + OVERHEAD)
    charged = base_cost
    credits_used = charged / CREDIT_VALUE
    return ChargeBreakdown(
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        embed_tokens=int(embed_tokens),
        cache_read_tokens=int(cache_read_tokens),
        cache_write_tokens=int(cache_write_tokens),
        base_cost=base_cost,
        charged=charged,
        credits_used=credits_used,
    )


def percent_remaining(balance: Decimal, plan_total: Decimal) -> float:
    """Remaining balance as a 0-100 percentage of the plan (clamped, 2 dp)."""
    if plan_total <= 0:
        return 0.0
    pct = (balance / plan_total) * Decimal(100)
    pct = max(Decimal(0), min(Decimal(100), pct))
    return float(round(pct, 2))
