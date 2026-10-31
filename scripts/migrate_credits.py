"""Provision the credit-metering tables — the credit ledger 'migration'.

This monorepo has no Alembic; SQLite schemas are created idempotently in code
(see ``MetadataStore``). This script does the same for the credit ledger: it opens
(creating if absent) ``<SEARCH_DATA_DIR>/credits.db`` and ensures the
``user_credits`` and ``credit_transactions`` tables exist. Safe to run repeatedly.

    uv run python scripts/migrate_credits.py            # create/verify tables
    uv run python scripts/migrate_credits.py --reset    # refill the pool to full

``--reset`` refills the shared pool to the configured starting balance
(SEARCH_CREDIT_PLAN_TOTAL) **in place** — safe to run while the app is serving (the
running server picks it up on its next read), and it preserves the transaction
history + JSON usage log.
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from pathlib import Path

from rag_search.config import SearchSettings
from rag_search.credit_store import CreditStore


async def main(reset: bool) -> None:
    settings = SearchSettings()
    data_dir = Path(settings.data_dir)
    db_path = data_dir / "credits.db"
    usage_log = data_dir / "credits_usage.jsonl"

    store = CreditStore(str(db_path), usage_log_path=str(usage_log))  # runs the schema migration
    try:
        if reset:
            await store.reset()
            print("[reset] refilled shared pool to the configured starting balance")

        tables = sorted(
            r[0]
            for r in sqlite3.connect(db_path).execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        )
        pool = await store.get_or_create()
        print(f"[ok] Credit ledger ready at {db_path}")
        print(f"     tables: {', '.join(tables)}")
        print(f"     usage log: {usage_log}")
        print(f"     shared pool balance: {float(pool.balance):g} / {float(pool.plan_total):g} credits")
    finally:
        await store.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Provision/reset the credit ledger.")
    parser.add_argument("--reset", action="store_true", help="refill the pool to full (in place)")
    asyncio.run(main(parser.parse_args().reset))
