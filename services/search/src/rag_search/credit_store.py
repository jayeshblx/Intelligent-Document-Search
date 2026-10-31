"""Authoritative credit ledger (SQLite) — a single *shared* pool.

Every browser/user draws down one balance (like a shared session allowance), so
when anyone runs a prompt the remaining credits drop for everyone. Two tables:

* ``user_credits``       one row: the shared pool's ``balance`` and ``plan_total``
                         (keyed by a fixed account id).
* ``credit_transactions`` an append-only audit row per charged request, tagged
                         with the ``actor`` (the browser/user that spent).

Every charge is also appended as one JSON line to a usage log file
(``credits_usage.jsonl``) for easy external auditing.

Balances are money-like, so amounts are stored as TEXT and manipulated with
``Decimal`` (full precision; round only for display). Concurrency safety — *two
rapid concurrent requests cannot drive the balance below zero* — is enforced two
ways at once: a process-wide ``asyncio.Lock`` (as in ``MetadataStore``) serialises
access to the shared connection, and each deduction runs inside a
``BEGIN IMMEDIATE`` transaction (SQLite's write lock — the row-lock equivalent)
that does the read-modify-write atomically and clamps the new balance at zero.
Persistence is on disk, so a browser refresh (or an app restart) shows the stored
balance, not a fresh one.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from rag_core.logging import get_logger
from rag_core.models.base import utcnow
from rag_search.credit_config import (
    DEFAULT_PLAN_TOTAL,
    PRO_PLAN_MULTIPLIER,
    ChargeBreakdown,
    compute_charge,
    percent_remaining,
)

_log = get_logger("rag_search.credit_store")

# The one pool everyone shares. (A future multi-tenant build could key this per
# team; for now every user of the URL spends from the same balance.)
SHARED_ACCOUNT_ID = "shared"


def _dec(text: str | None) -> Decimal:
    return Decimal(text) if text else Decimal(0)


@dataclass(slots=True)
class Account:
    """The shared pool's current standing (authoritative, from the DB)."""

    account_id: str
    balance: Decimal
    plan_total: Decimal

    @property
    def percent_remaining(self) -> float:
        return percent_remaining(self.balance, self.plan_total)

    @property
    def is_exhausted(self) -> bool:
        return self.balance <= 0


@dataclass(slots=True)
class ChargeResult:
    """The outcome of deducting for one request: what was charged + the new pool."""

    account: Account
    breakdown: ChargeBreakdown


class CreditStore:
    """Persistent, concurrency-safe shared credit pool."""

    def __init__(
        self,
        db_path: str,
        *,
        default_plan_total: Decimal = DEFAULT_PLAN_TOTAL,
        account_id: str = SHARED_ACCOUNT_ID,
        usage_log_path: str | None = None,
    ) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._default_plan_total = default_plan_total
        self._account_id = account_id
        self._usage_log_path = Path(usage_log_path) if usage_log_path else None
        self._conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        # WAL keeps readers (balance polls) from blocking the writer (deductions);
        # busy_timeout lets a concurrent writer (e.g. a live reset) wait for the lock
        # instead of failing with "database is locked".
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._lock = asyncio.Lock()
        self._init_schema()

    # ---- schema / migration -------------------------------------------------
    def _init_schema(self) -> None:
        """Create the credit tables if absent (the idempotent 'migration')."""
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS user_credits (
                user_id     TEXT PRIMARY KEY,
                balance     TEXT NOT NULL,
                plan_total  TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS credit_transactions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                input_tokens  INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                embed_tokens  INTEGER NOT NULL DEFAULT 0,
                base_cost     TEXT NOT NULL,
                charged       TEXT NOT NULL,
                credits_used  TEXT NOT NULL,
                balance_after TEXT NOT NULL,
                model         TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_txn_user ON credit_transactions(user_id, created_at);
            """
        )
        # Migrate older DBs that predate the embed_tokens column.
        existing = {r["name"] for r in self._conn.execute("PRAGMA table_info(credit_transactions)")}
        if "embed_tokens" not in existing:
            self._conn.execute(
                "ALTER TABLE credit_transactions ADD COLUMN embed_tokens INTEGER NOT NULL DEFAULT 0"
            )

    # ---- reads --------------------------------------------------------------
    async def balance(self) -> Account:
        """Return the shared pool, seeding it on first sight."""
        async with self._lock:
            return await asyncio.to_thread(self._get_or_create_sync)

    # Back-compat alias — the pool is shared, so the caller's identity is ignored.
    async def get_or_create(self) -> Account:
        return await self.balance()

    def _get_or_create_sync(self) -> Account:
        row = self._conn.execute(
            "SELECT balance, plan_total FROM user_credits WHERE user_id = ?", (self._account_id,)
        ).fetchone()
        if row is not None:
            return Account(self._account_id, _dec(row["balance"]), _dec(row["plan_total"]))
        now = utcnow().isoformat()
        total = self._default_plan_total
        self._conn.execute(
            "INSERT INTO user_credits (user_id, balance, plan_total, created_at, updated_at) "
            "VALUES (?,?,?,?,?)",
            (self._account_id, str(total), str(total), now, now),
        )
        return Account(self._account_id, total, total)

    # ---- the authoritative deduction ---------------------------------------
    async def charge(
        self,
        *,
        actor: str,
        input_tokens: int,
        output_tokens: int,
        embed_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        model: str,
    ) -> ChargeResult:
        """Deduct the credits for one *successful* request from the shared pool.

        Covers every token the turn spent — input + output + query ``embed_tokens`` +
        prompt-cache read/write — so the meter matches reality after routing, answering,
        and any summarization. ``actor`` is the browser/user that initiated the run —
        recorded in the transaction + usage log, but the balance is shared. Never raises
        for a low balance (the ``balance <= 0`` gate belongs in the endpoint *before* the
        model runs); clamps the new balance at zero so concurrent requests can't go
        negative, while still recording the true ``credits_used``."""
        breakdown = compute_charge(
            input_tokens, output_tokens, embed_tokens, cache_read_tokens, cache_write_tokens
        )
        async with self._lock:
            return await asyncio.to_thread(self._charge_sync, actor, breakdown, model)

    def _charge_sync(self, actor: str, breakdown: ChargeBreakdown, model: str) -> ChargeResult:
        conn = self._conn
        conn.execute("BEGIN IMMEDIATE")  # acquire the write lock up front
        try:
            row = conn.execute(
                "SELECT balance, plan_total FROM user_credits WHERE user_id = ?", (self._account_id,)
            ).fetchone()
            now = utcnow().isoformat()
            if row is None:
                plan_total = self._default_plan_total
                balance = plan_total
                conn.execute(
                    "INSERT INTO user_credits (user_id, balance, plan_total, created_at, updated_at)"
                    " VALUES (?,?,?,?,?)",
                    (self._account_id, str(balance), str(plan_total), now, now),
                )
            else:
                balance = _dec(row["balance"])
                plan_total = _dec(row["plan_total"])

            new_balance = balance - breakdown.credits_used
            if new_balance < 0:
                new_balance = Decimal(0)  # never below zero

            conn.execute(
                "UPDATE user_credits SET balance = ?, updated_at = ? WHERE user_id = ?",
                (str(new_balance), now, self._account_id),
            )
            conn.execute(
                "INSERT INTO credit_transactions (user_id, created_at, input_tokens, output_tokens,"
                " embed_tokens, base_cost, charged, credits_used, balance_after, model)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    actor, now, breakdown.input_tokens, breakdown.output_tokens,
                    breakdown.embed_tokens, str(breakdown.base_cost), str(breakdown.charged),
                    str(breakdown.credits_used), str(new_balance), model,
                ),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        self._append_usage_log(
            {
                "ts": now,
                "actor": actor,
                "input_tokens": breakdown.input_tokens,
                "output_tokens": breakdown.output_tokens,
                "embed_tokens": breakdown.embed_tokens,
                "base_cost_usd": float(round(breakdown.base_cost, 6)),
                "charged_usd": float(round(breakdown.charged, 6)),
                "credits_used": float(round(breakdown.credits_used, 4)),
                "balance_after": float(round(new_balance, 2)),
                "plan_total": float(round(plan_total, 2)),
                "model": model,
            }
        )
        _log.info(
            "credit_charged", actor=actor, credits_used=str(round(breakdown.credits_used, 4)),
            balance_after=str(round(new_balance, 4)), model=model,
        )
        return ChargeResult(Account(self._account_id, new_balance, plan_total), breakdown)

    def _append_usage_log(self, record: dict) -> None:
        """Append one JSON line to the usage log (best-effort; never fails a charge)."""
        if self._usage_log_path is None:
            return
        try:
            self._usage_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._usage_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:  # a log write must not break metering
            _log.warning("credit_usage_log_write_failed", error=str(exc))

    # ---- reset --------------------------------------------------------------
    async def reset(self) -> Account:
        """Refill the shared pool to its configured starting balance, in place.

        Safe to run while the app is serving (an ``UPDATE``, no file delete), so the
        running server picks up the new balance on its next read. Transaction history
        and the JSON usage log are preserved."""
        async with self._lock:
            return await asyncio.to_thread(self._reset_sync)

    def _reset_sync(self) -> Account:
        total = self._default_plan_total
        now = utcnow().isoformat()
        self._conn.execute(
            "INSERT INTO user_credits (user_id, balance, plan_total, created_at, updated_at)"
            " VALUES (?,?,?,?,?)"
            " ON CONFLICT(user_id) DO UPDATE SET balance = excluded.balance,"
            " plan_total = excluded.plan_total, updated_at = excluded.updated_at",
            (self._account_id, str(total), str(total), now, now),
        )
        return Account(self._account_id, total, total)

    # ---- top up -------------------------------------------------------------
    async def top_up(self, amount: Decimal) -> Account:
        """Admin: add credits to the shared pool. If the new balance exceeds the plan,
        the plan is raised to match so the meter still reads out of a sensible total."""
        async with self._lock:
            return await asyncio.to_thread(self._top_up_sync, amount)

    def _top_up_sync(self, amount: Decimal) -> Account:
        conn = self._conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT balance, plan_total FROM user_credits WHERE user_id = ?", (self._account_id,)
            ).fetchone()
            if row is None:
                balance = plan_total = self._default_plan_total
            else:
                balance, plan_total = _dec(row["balance"]), _dec(row["plan_total"])
            balance = balance + amount
            plan_total = max(plan_total, balance)
            now = utcnow().isoformat()
            conn.execute(
                "INSERT INTO user_credits (user_id, balance, plan_total, created_at, updated_at)"
                " VALUES (?,?,?,?,?)"
                " ON CONFLICT(user_id) DO UPDATE SET balance = excluded.balance,"
                " plan_total = excluded.plan_total, updated_at = excluded.updated_at",
                (self._account_id, str(balance), str(plan_total), now, now),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        _log.info("credit_topped_up", amount=str(amount), balance=str(balance))
        return Account(self._account_id, balance, plan_total)

    # ---- plan upgrade -------------------------------------------------------
    async def upgrade(self) -> Account:
        """Move the shared pool to the Pro plan (5x credits) and refill."""
        async with self._lock:
            return await asyncio.to_thread(self._upgrade_sync)

    def _upgrade_sync(self) -> Account:
        pro_total = self._default_plan_total * PRO_PLAN_MULTIPLIER
        now = utcnow().isoformat()
        self._conn.execute(
            "INSERT INTO user_credits (user_id, balance, plan_total, created_at, updated_at)"
            " VALUES (?,?,?,?,?)"
            " ON CONFLICT(user_id) DO UPDATE SET balance = excluded.balance,"
            " plan_total = excluded.plan_total, updated_at = excluded.updated_at",
            (self._account_id, str(pro_total), str(pro_total), now, now),
        )
        return Account(self._account_id, pro_total, pro_total)

    # ---- lifecycle ----------------------------------------------------------
    async def aclose(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._conn.close)
