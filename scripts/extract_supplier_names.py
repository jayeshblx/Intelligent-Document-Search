"""Read supplier names out of the documents already indexed — no S3, no re-embedding.

The supplier NAME is the one facet MongoDB cannot supply (it holds only the supplier
CODE, e.g. SUPPLIER0028), so it has to come from the document: "CONTOSO AUSTRALIA PTY
LTD" printed on the certificate. The indexer does this inline for new documents; this
script does it for an index that already exists, using the chunk text already stored in
the keyword index rather than re-downloading and re-extracting every PDF.

It writes the ``supplier`` facet to all three stores (metadata for display, FTS and
Qdrant payloads for query-time filtering) and leaves embeddings byte-for-byte untouched.
Run ``consolidate_suppliers.py`` afterwards to settle one name per supplier code.

Run with the search server STOPPED so the stores aren't locked:

    uv run python scripts/extract_supplier_names.py
    uv run python scripts/extract_supplier_names.py --limit 20   # try a subset first
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from pathlib import Path

from rag_search.answer import BedrockLLM
from rag_search.config import SearchSettings
from rag_search.factory import build_components
from rag_search.metadata_extract import MetadataExtractor

from rag_core.config import get_settings
from rag_core.di import Container
from rag_core.logging import configure_logging
from rag_core.models.content import ExtractedContent, Page

# Enough of each document to carry the letterhead / certificate front matter where the
# supplier is named, without paying for the whole document.
_CHUNKS_PER_DOC = 14
_CONCURRENCY = 8


def _safe(text: str) -> str:
    """Console-safe text. Supplier names carry accents and CJK characters that a Windows
    cp1252 console cannot encode, and a crashed print would abort the whole pass."""
    enc = sys.stdout.encoding or "utf-8"
    return text.encode(enc, errors="replace").decode(enc, errors="replace")


def _documents(data_dir: Path, limit: int | None) -> list[tuple[str, str]]:
    conn = sqlite3.connect(str(data_dir / "metadata.db"))
    conn.row_factory = sqlite3.Row
    try:
        sql = "SELECT document_id, source_key FROM document_metadata ORDER BY source_key"
        rows = conn.execute(sql + (f" LIMIT {int(limit)}" if limit else "")).fetchall()
    finally:
        conn.close()
    return [(r["document_id"], r["source_key"]) for r in rows]


def _text_for(kw_db: str, document_id: str) -> str:
    conn = sqlite3.connect(kw_db)
    try:
        rows = conn.execute(
            "SELECT text FROM chunks WHERE json_extract(payload, '$.document_id') = ? "
            "ORDER BY rowid LIMIT ?",
            (document_id, _CHUNKS_PER_DOC),
        ).fetchall()
    finally:
        conn.close()
    return "\n".join(r[0] for r in rows if r[0])


async def _run(args: argparse.Namespace) -> None:
    search_settings = SearchSettings()
    data_dir = Path(search_settings.data_dir)
    kw_db = str(data_dir / "keyword.db")
    docs = _documents(data_dir, args.limit)
    print(f"documents to scan: {len(docs)}")

    container = Container(get_settings())
    components = build_components(get_settings(), search_settings)
    extractor = MetadataExtractor(BedrockLLM(container.bedrock()))
    sem = asyncio.Semaphore(_CONCURRENCY)
    named = blank = 0
    lock = asyncio.Lock()

    async def one(document_id: str, source_key: str) -> None:
        nonlocal named, blank
        text = await asyncio.to_thread(_text_for, kw_db, document_id)
        if not text.strip():
            return
        title = source_key.rsplit("/", 1)[-1]
        async with sem:
            try:
                supplier = await extractor.extract_supplier(
                    ExtractedContent(
                        document_id=document_id,
                        pages=[Page(number=1, text=text, blocks=[])],
                    ),
                    title=title,
                )
            except Exception as exc:
                print(_safe(f"  ! {title[:60]}: {exc}"))
                return
        # Write the value (or clear a stale one) across every store that serves results.
        async with lock:
            facets: dict[str, str | None] = {"supplier": supplier}
            if await components.metadata.apply_facets(document_id, facets):
                await components.keyword.update_payload(document_id, facets)
                await components.vector.set_document_payload(document_id, facets)
            if supplier:
                named += 1
                print(_safe(f"  {supplier[:46]:46} <- {title[:52]}"))
            else:
                blank += 1

    try:
        await asyncio.gather(*(one(d, k) for d, k in docs))
        print(f"\n=== EXTRACTION COMPLETE ===\n  named: {named}\n  no company named: {blank}")
        print("\nNow run: python scripts/consolidate_suppliers.py")
    finally:
        await components.aclose()
        await container.aclose()


def main() -> None:
    configure_logging(level="WARNING", json_logs=False, service_name="supplier-cli")
    parser = argparse.ArgumentParser(description="Extract supplier names from indexed text")
    parser.add_argument("--limit", type=int, default=None, help="only the first N documents")
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
