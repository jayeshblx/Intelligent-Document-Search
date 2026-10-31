"""Unit tests for the credit charge calculation (the billing formula)."""

from __future__ import annotations

from decimal import Decimal

import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_search.credit_config import (
    CREDIT_VALUE,
    INPUT_RATE,
    OUTPUT_RATE,
    OVERHEAD,
    compute_charge,
    percent_remaining,
)


def test_contract_rates_are_the_spec_values() -> None:
    assert Decimal("2.00") == INPUT_RATE
    assert Decimal("10.00") == OUTPUT_RATE
    assert Decimal("0.50") == OVERHEAD
    assert Decimal("0.001") == CREDIT_VALUE


def test_charge_for_known_token_counts() -> None:
    # Overhead disabled: the charge is the raw token cost, no markup.
    #   base    = 1000/1e6*2 + 500/1e6*10 = 0.002 + 0.005 = 0.007
    #   charged = base (no overhead)                       = 0.007
    #   credits = 0.007 / 0.001                            = 7.0
    b = compute_charge(1000, 500)
    assert b.base_cost == Decimal("0.007")
    assert b.charged == Decimal("0.007")
    assert b.credits_used == Decimal("7")


def test_charge_at_one_million_tokens_each() -> None:
    b = compute_charge(1_000_000, 1_000_000)
    assert b.base_cost == Decimal("12")        # 2 + 10
    assert b.charged == Decimal("12")          # no overhead
    assert b.credits_used == Decimal("12000")  # 12 / 0.001


def test_overhead_is_disabled() -> None:
    # Overhead is commented out, so the charge equals the raw base cost exactly.
    b = compute_charge(1000, 0)               # base 0.002 -> charged 0.002 -> 2.0 credits
    assert b.charged == b.base_cost
    assert b.credits_used == Decimal("2")


def test_zero_tokens_is_zero_charge() -> None:
    b = compute_charge(0, 0)
    assert b.base_cost == Decimal(0)
    assert b.credits_used == Decimal(0)


def test_embedding_cost_is_included() -> None:
    # 1M embed tokens at $0.02/1M = $0.02 base -> no overhead -> 20 credits.
    b = compute_charge(0, 0, 1_000_000)
    assert b.embed_tokens == 1_000_000
    assert b.base_cost == Decimal("0.02")
    assert b.credits_used == Decimal("20")
    # Embedding adds on top of input/output (1000/500 alone = 7.0 credits).
    assert compute_charge(1000, 500, 1000).credits_used == Decimal("7.02")


def test_percent_remaining_clamps_and_handles_edges() -> None:
    assert percent_remaining(Decimal("2000"), Decimal("2000")) == 100.0
    assert percent_remaining(Decimal("500"), Decimal("2000")) == 25.0
    assert percent_remaining(Decimal("0"), Decimal("2000")) == 0.0
    assert percent_remaining(Decimal("-5"), Decimal("2000")) == 0.0   # never below 0
    assert percent_remaining(Decimal("100"), Decimal("0")) == 0.0     # no plan
