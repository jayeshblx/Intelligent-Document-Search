"""Index and search real documents from the configured S3 bucket/prefix.

Reads credentials + bucket/prefix from .env (see rag_core.config.Settings). Uses
Bedrock Titan embeddings and the local free stores (embedded Qdrant + SQLite FTS5).

Usage
-----
    uv run python scripts/index_s3.py list                 # list objects under the prefix
    uv run python scripts/index_s3.py index                # embed + index everything (reset)
    uv run python scripts/index_s3.py index --mongo-only   # only docs catalogued in MongoDB
    uv run python scripts/index_s3.py index --max 25       # index first 25 (cost control)
    uv run python scripts/index_s3.py search "your query"  # hybrid search + download links
"""

from __future__ import annotations

import argparse
import asyncio

from rag_search.config import SearchSettings
from rag_search.factory import build_components
from rag_search.models import IndexRequest

from rag_core.config import get_settings
from rag_core.di import Container
from rag_core.logging import configure_logging, get_logger

_log = get_logger("index_s3")


async def _list(args: argparse.Namespace) -> None:
    settings = get_settings()
    container = Container(settings)
    s3 = container.s3()
    bucket, prefix = settings.s3.bucket, settings.s3.prefix
    print(f"Listing s3://{bucket}/{prefix} ...")
    count, shown, total_bytes = 0, 0, 0
    try:
        async for obj in s3.list_objects(bucket, prefix):
            count += 1
            total_bytes += obj.size_bytes
            if shown < 30:
                print(f"  {obj.size_bytes:>10,d}  {obj.key}")
                shown += 1
        print(f"\n  {count} object(s), {total_bytes / 1_048_576:.1f} MiB total")
    finally:
        await container.aclose()


async def _index(args: argparse.Namespace) -> None:
    settings = get_settings()
    components = build_components(settings, SearchSettings())
    only = args.only.split(",") if args.only else None
    try:
        summary = await components.indexer.index(
            IndexRequest(
                reset=args.reset, max_docs=args.max, only_types=only,
                mongo_only=args.mongo_only,
            )
        )
        print("\n=== INDEX COMPLETE ===")
        print(f"  documents indexed : {summary.documents_indexed}")
        print(f"  documents failed  : {summary.documents_failed}")
        print(f"  documents deleted : {summary.documents_deleted}  (Mongo DeleteMark=1 -> skipped/purged)")
        print(f"  other tenants     : {summary.documents_other_tenant}  (SEARCH_MONGO_TENANT_CODES -> skipped)")
        print(f"  images skipped    : {summary.documents_non_text}  (no extractable text -> not downloaded)")
        print(f"  chunks indexed    : {summary.chunks_indexed}")
        print(f"  duration          : {summary.duration_s}s")
        for sample in summary.error_samples:
            print(f"  ! {sample}")
    finally:
        await components.aclose()


async def _backfill_metadata(args: argparse.Namespace) -> None:
    """Add attachment-metadata chunks to an existing index (embeds only those)."""
    settings = get_settings()
    components = build_components(settings, SearchSettings())
    try:
        added = await components.indexer.backfill_attachment_metadata()
        print(f"\n=== BACKFILL COMPLETE ===\n  attachment-metadata chunks added : {added}")
    finally:
        await components.aclose()


async def _search(args: argparse.Namespace) -> None:
    settings = get_settings()
    components = build_components(settings, SearchSettings())
    try:
        resp = await components.search.search(args.query, top_k=args.k)
        print(f"\nQUERY: {args.query}   ({resp.took_ms} ms)\n" + "=" * 78)
        for i, r in enumerate(resp.results, 1):
            loc = f"{r.section or ''} p.{r.page_start}".strip()
            print(f"{i}. [{r.matched_by}] score={r.score:.4f}  {r.source_key}  {loc}")
            print(f"   {r.text[:180].strip()}...")
            if r.download_url:
                print(f"   download: {r.download_url[:110]}...")
            print()
    finally:
        await components.aclose()


def main() -> None:
    configure_logging(level="INFO", json_logs=False, service_name="index-cli")
    parser = argparse.ArgumentParser(description="Index/search real S3 documents")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    p_index = sub.add_parser("index")
    p_index.add_argument("--max", type=int, default=None, help="cap number of documents")
    p_index.add_argument("--only", default=None, help="comma types, e.g. docx,txt,html")
    p_index.add_argument("--no-reset", dest="reset", action="store_false", help="append (don't wipe)")
    # Only 5% of the S3 objects are catalogued in MongoDB; the rest have no business
    # facets at all (no status/object code/business object) and show as empty columns.
    # Skipping them is how the working index has always been built.
    p_index.add_argument(
        "--mongo-only", action="store_true",
        help="index only objects present in MongoDB (skips uncatalogued S3 objects)",
    )
    p_index.set_defaults(reset=True)
    sub.add_parser(
        "backfill-metadata",
        help="add MongoDB MetaData field values to an existing index as searchable chunks",
    )
    p_search = sub.add_parser("search")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=8)

    args = parser.parse_args()
    handlers = {
        "list": _list,
        "index": _index,
        "search": _search,
        "backfill-metadata": _backfill_metadata,
    }
    asyncio.run(handlers[args.command](args))


if __name__ == "__main__":
    main()
