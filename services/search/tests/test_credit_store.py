"""Tests for the authoritative shared credit ledger (SQLite)."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal

from rag_search.credit_store import CreditStore


async def test_charge_deducts_and_records_transaction(tmp_path) -> None:
    store = CreditStore(str(tmp_path / "c.db"), default_plan_total=Decimal("100"))
    try:
        acct = await store.get_or_create()
        assert acct.balance == Decimal("100")
        assert acct.plan_total == Decimal("100")

        res = await store.charge(actor="u1", input_tokens=1000, output_tokens=500, model="m")
        assert res.breakdown.credits_used == Decimal("7")   # no overhead
        assert res.account.balance == Decimal("93")
        assert res.account.percent_remaining == 93.0
    finally:
        await store.aclose()


async def test_balance_persists_across_reopen(tmp_path) -> None:
    """Reopening the DB (i.e. an app restart / a browser refresh) shows the stored
    balance, not a fresh plan."""
    path = str(tmp_path / "c.db")
    store = CreditStore(path, default_plan_total=Decimal("100"))
    await store.charge(actor="u1", input_tokens=1000, output_tokens=500, model="m")
    await store.aclose()

    reopened = CreditStore(path, default_plan_total=Decimal("100"))
    try:
        acct = await reopened.get_or_create()
        assert acct.balance == Decimal("93")  # not reset to 100
    finally:
        await reopened.aclose()


async def test_pool_is_shared_across_actors(tmp_path) -> None:
    """Every browser/user draws down ONE pool — a second actor sees the reduced
    balance the first actor left, not a fresh 100."""
    store = CreditStore(str(tmp_path / "c.db"), default_plan_total=Decimal("100"))
    try:
        await store.charge(actor="alice", input_tokens=1000, output_tokens=500, model="m")  # -7
        await store.charge(actor="bob", input_tokens=1000, output_tokens=500, model="m")    # -7
        acct = await store.get_or_create()
        assert acct.balance == Decimal("86.0")  # 100 - 7 - 7, shared
    finally:
        await store.aclose()


async def test_concurrent_charges_never_drive_below_zero(tmp_path) -> None:
    """Ten concurrent charges of 7 credits against a 20-credit pool must clamp at
    zero — never double-spend into the negative."""
    store = CreditStore(str(tmp_path / "c.db"), default_plan_total=Decimal("20"))
    try:
        await store.get_or_create()
        results = await asyncio.gather(
            *(store.charge(actor=f"u{i}", input_tokens=1000, output_tokens=500, model="m")
              for i in range(10))
        )
        assert all(r.account.balance >= 0 for r in results)
        final = await store.get_or_create()
        assert final.balance == Decimal("0")
        assert final.is_exhausted
    finally:
        await store.aclose()


async def test_upgrade_moves_to_pro_and_refills(tmp_path) -> None:
    store = CreditStore(str(tmp_path / "c.db"), default_plan_total=Decimal("100"))
    try:
        # Spend enough to exhaust (1M input -> 3000 credits, clamps to 0).
        await store.charge(actor="u1", input_tokens=1_000_000, output_tokens=0, model="m")
        assert (await store.get_or_create()).is_exhausted

        acct = await store.upgrade()
        assert acct.plan_total == Decimal("500")  # 100 * 5
        assert acct.balance == Decimal("500")
        assert not acct.is_exhausted
    finally:
        await store.aclose()


async def test_reset_refills_pool_in_place(tmp_path) -> None:
    store = CreditStore(str(tmp_path / "c.db"), default_plan_total=Decimal("1000"))
    try:
        # Drain the pool (1M/1M -> 12000 credits, clamps to 0).
        await store.charge(actor="u1", input_tokens=1_000_000, output_tokens=1_000_000, model="m")
        assert (await store.get_or_create()).is_exhausted

        acct = await store.reset()
        assert acct.balance == Decimal("1000")
        assert acct.plan_total == Decimal("1000")
        assert (await store.get_or_create()).balance == Decimal("1000")  # persisted
    finally:
        await store.aclose()


async def test_charge_includes_embedding(tmp_path) -> None:
    store = CreditStore(str(tmp_path / "c.db"), default_plan_total=Decimal("1000"))
    try:
        res = await store.charge(
            actor="u1", input_tokens=1000, output_tokens=500, embed_tokens=1000, model="m"
        )
        assert res.breakdown.embed_tokens == 1000
        assert res.breakdown.credits_used == Decimal("7.02")  # 7.0 + embedding, no overhead
    finally:
        await store.aclose()


async def test_usage_log_file_is_written(tmp_path) -> None:
    log_path = tmp_path / "credits_usage.jsonl"
    store = CreditStore(
        str(tmp_path / "c.db"), default_plan_total=Decimal("100"), usage_log_path=str(log_path)
    )
    try:
        await store.charge(
            actor="alice", input_tokens=1000, output_tokens=500, embed_tokens=12, model="fake-llm"
        )
        await store.charge(actor="bob", input_tokens=2000, output_tokens=0, model="fake-llm")
    finally:
        await store.aclose()

    lines = [json.loads(ln) for ln in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert lines[0]["actor"] == "alice"
    assert lines[0]["embed_tokens"] == 12
    assert lines[0]["balance_after"] == 93.0  # 100 - ~7 (incl. embedding, no overhead)
    assert lines[1]["actor"] == "bob"
    assert lines[1]["model"] == "fake-llm"
