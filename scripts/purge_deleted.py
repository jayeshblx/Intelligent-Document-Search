"""Purge soft-deleted documents from the EXISTING index — no re-embedding.

The compliance corpus is mirrored in MongoDB (``AttachmentMetaData``). A record whose
``DeleteMark`` is 1 is soft-deleted upstream and must not appear in search. This script
reconciles the already-built local index against Mongo and removes only those documents'
chunks/vectors/metadata. It does NOT call S3 or Bedrock and does NOT touch any document
that isn't soft-deleted, so the surviving embeddings are byte-for-byte the originals.

Workflow (run with the search server STOPPED so the stores aren't locked):

    uv run python scripts/purge_deleted.py            # archive originals, then purge the copy
    uv run python scripts/purge_deleted.py --dry-run  # report what WOULD be purged, change nothing
    uv run python scripts/purge_deleted.py --no-archive  # skip the archive copy (purge in place)

The archive is a full copy of qdrant/ + keyword.db + metadata.db under
``<data_dir>/archive/index-<stamp>/`` before anything is deleted, so the pre-purge index
is always recoverable.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import time
from pathlib import Path

from rag_search.config import SearchSettings
from rag_search.factory import build_components
from rag_search.mongo_source import basename

from rag_core.config import get_settings
from rag_core.logging import configure_logging, get_logger

_log = get_logger("purge_deleted")

# The index artifacts that make up the embedded search index (everything else in the
# data dir — credits, feedback, usage logs — is left alone).
_INDEX_ARTIFACTS = ["qdrant", "keyword.db", "metadata.db"]


def _archive(data_dir: Path, stamp: str) -> Path:
    """Copy the current index artifacts into an archive folder; return its path."""
    dest = data_dir / "archive" / f"index-{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    for name in _INDEX_ARTIFACTS:
        src = data_dir / name
        if not src.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dest / name)
        else:
            shutil.copy2(src, dest / name)
    return dest


async def _purge(args: argparse.Namespace) -> None:
    settings = get_settings()
    search_settings = SearchSettings()
    data_dir = Path(search_settings.data_dir)

    # 1. Archive the pristine index first (unless disabled), so the pre-purge state is
    #    always recoverable. Done before opening the stores for a clean copy.
    if args.archive and not args.dry_run:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        dest = _archive(data_dir, stamp)
        print(f"Archived current index -> {dest}")

    components = build_components(settings, search_settings)
    try:
        mongo = components.indexer._mongo  # the configured MongoMetadataSource
        if mongo is None:
            print("ERROR: MongoDB is not configured (SEARCH_MONGO_URI unset). Nothing to do.")
            return
        loaded = await mongo.preload()
        print(f"Mongo loaded: {loaded} live records, {mongo.deleted_count} DeleteMark=1 records")

        # 2. Reconcile every indexed document against Mongo's deleted set by filename.
        indexed = await components.metadata.all_documents()
        to_purge = [(doc_id, key) for doc_id, key in indexed if mongo.is_deleted(key)]
        print(f"Indexed documents: {len(indexed)}   soft-deleted & present in index: {len(to_purge)}")

        if args.dry_run:
            for doc_id, key in to_purge:
                print(f"  WOULD PURGE  {basename(key)}   ({doc_id})")
            print("\n(dry run — no changes made)")
            return

        # 3. Delete only those documents from all three stores.
        for doc_id, key in to_purge:
            await components.vector.delete_document(doc_id)
            await components.keyword.delete_document(doc_id)
            await components.metadata.delete_document(doc_id)
            print(f"  purged  {basename(key)}")

        remaining = await components.metadata.count()
        print("\n=== PURGE COMPLETE ===")
        print(f"  documents purged    : {len(to_purge)}")
        print(f"  documents remaining : {remaining}")
    finally:
        await components.aclose()


def main() -> None:
    configure_logging(level="INFO", json_logs=False, service_name="purge-cli")
    parser = argparse.ArgumentParser(description="Purge Mongo-soft-deleted docs from the index")
    parser.add_argument("--dry-run", action="store_true", help="report only; change nothing")
    parser.add_argument(
        "--no-archive", dest="archive", action="store_false",
        help="don't copy the index to archive/ first (purge in place)",
    )
    parser.set_defaults(archive=True)
    args = parser.parse_args()
    asyncio.run(_purge(args))


if __name__ == "__main__":
    main()
