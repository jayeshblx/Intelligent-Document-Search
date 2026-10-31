"""Give every document under one supplier code the same supplier name.

The indexer extracts a supplier name per document and normally consolidates them at the
end of the run (``IndexingService._consolidate_suppliers``). A run that is interrupted
leaves the per-document names in place but never votes on them, so one supplier shows up
under several spellings — "Northwind Foods and Beverages Pty Ltd", "NWD Milk Pty Ltd" and
"Northwind F&B - Northgate" are all SUPPLIER0005.

This applies the same vote to an index that already exists, and retracts values the
current rules reject (a model's conversational non-answer on a document with no readable
text). It reads and writes only the ``supplier`` facet — no S3, no Bedrock, no
re-embedding — and is idempotent.

Run with the search server STOPPED so the stores aren't locked:

    uv run python scripts/consolidate_suppliers.py            # consolidate
    uv run python scripts/consolidate_suppliers.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rag_search.config import SearchSettings
from rag_search.factory import build_components
from rag_search.metadata_extract import canonical_supplier, is_valid_supplier_name

from rag_core.config import get_settings
from rag_core.logging import configure_logging

Votes = dict[str, Counter[str]]
Docs = dict[str, list[str]]


def _safe(text: str) -> str:
    """Console-safe text. Supplier names carry accents and CJK characters that a Windows
    cp1252 console cannot encode, and a crashed print would abort the whole pass."""
    enc = sys.stdout.encoding or "utf-8"
    return text.encode(enc, errors="replace").decode(enc, errors="replace")


def _collect(db_path: str) -> tuple[Votes, Docs, list[str]]:
    """Per object code: the names its documents produced, and the documents themselves.

    The third value is documents that carry a supplier name but NO object code — there is
    nothing to vote with, so each is judged on its own.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT document_id, facets FROM document_metadata").fetchall()
    finally:
        conn.close()

    votes: Votes = defaultdict(Counter)
    docs: Docs = defaultdict(list)
    codeless: list[str] = []
    for row in rows:
        try:
            facets = json.loads(row["facets"] or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        name = facets.get("supplier")
        code = facets.get("object_code")
        if not isinstance(code, str) or not code.strip():
            if isinstance(name, str) and name and not is_valid_supplier_name(name):
                codeless.append(row["document_id"])
            continue
        docs[code.strip()].append(row["document_id"])
        if isinstance(name, str) and is_valid_supplier_name(name):
            votes[code.strip()][name.strip()] += 1
    return votes, docs, codeless


async def _run(args: argparse.Namespace) -> None:
    search_settings = SearchSettings()
    votes, docs, codeless = _collect(str(Path(search_settings.data_dir) / "metadata.db"))
    if not votes and not codeless:
        print("No extracted supplier names found — nothing to consolidate.")
        return

    components = build_components(get_settings(), search_settings)

    async def write(doc_id: str, name: str | None) -> bool:
        """Set (or retract, when ``name`` is None) one document's supplier everywhere."""
        if not await components.metadata.apply_facets(doc_id, {"supplier": name}):
            return False
        await components.keyword.update_payload(doc_id, {"supplier": name})
        await components.vector.set_document_payload(doc_id, {"supplier": name})
        return True

    try:
        updated = cleared = 0
        for code in sorted(docs):
            winner = canonical_supplier(votes[code]) if code in votes else None
            targets = docs[code]
            if winner is not None:
                print(_safe(f"  {code:16} -> {winner}   ({len(targets)} docs, {len(votes[code])} variant(s))"))
            if args.dry_run:
                continue
            for doc_id in targets:
                if await write(doc_id, winner):
                    if winner is None:
                        cleared += 1
                    else:
                        updated += 1

        if not args.dry_run:
            for doc_id in codeless:
                if await write(doc_id, None):
                    cleared += 1

        unnamed = sorted(set(docs) - set(votes))
        print(f"\nObject codes whose documents named no company: {len(unnamed)}")
        print(f"Documents with no object code and an invalid name: {len(codeless)}")
        if args.dry_run:
            print("\n(dry run — no changes made)")
        else:
            print(
                f"\n=== CONSOLIDATED ===\n"
                f"  object codes named   : {len(votes)}\n"
                f"  documents named      : {updated}\n"
                f"  stale values cleared : {cleared}"
            )
    finally:
        await components.aclose()


def main() -> None:
    configure_logging(level="WARNING", json_logs=False, service_name="consolidate-cli")
    parser = argparse.ArgumentParser(description="One supplier name per supplier code")
    parser.add_argument("--dry-run", action="store_true", help="report only; change nothing")
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
