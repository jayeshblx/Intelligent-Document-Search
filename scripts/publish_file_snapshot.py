"""Publish a tenant's index to S3 as FILES the agents open directly.

    s3://{bucket}/{prefix}/{environment}/{tenant}/
        manifest.json    version marker + how to open the snapshot
        qdrant/**        embedded Qdrant collection (the embeddings)
        keyword.db       SQLite FTS5 (keyword search + the chunk text)
        metadata.db      SQLite (document metadata + facets)

This is the format ``rag_agentic.index_store`` mirrors into ``/tmp`` and opens with
no server and no container: everything is pure Python, so the agents can run as an
AgentCore **code** deployment.

The vectors are copied out of the Postgres index as they are -- nothing is
re-embedded, so publishing costs no Bedrock calls and takes minutes, not hours.

Usage
-----
    uv run python scripts/publish_file_snapshot.py --environment acmedev --tenant foxtrot \
        --bucket example-docs-bucket --dsn postgresql://rag:rag@localhost:5432/rag_acmedev_foxtrot

    ... --dry-run     # build the files locally, upload nothing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_search.config import SearchSettings
from rag_search.metadata_store import MetadataStore
from rag_search.stores import KeywordIndex, LocalVectorIndex
from rag_search.stores.postgres import PgDatabase

from rag_core.config import get_settings
from rag_core.di import Container
from rag_core.logging import configure_logging, get_logger
from rag_core.models.metadata import DocumentMetadata

_log = get_logger("publish_file_snapshot")

MANIFEST_KEY = "manifest.json"
MANIFEST_VERSION = 1  # the file-based layout the agents mirror
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
# Rows per round trip. Each row carries a 1024-float vector as text, so a large
# batch is a lot of memory for no gain.
_BATCH = 500


def _parse_vector(raw: str) -> list[float]:
    return [float(x) for x in raw.strip("[]").split(",")] if raw else []


def prefix_disagrees(indexed: str, configured: str) -> bool:
    """True when configuration points somewhere other than where the index came from.

    Compared on the path segments, so ``bravo/`` ``/bravo`` and ``bravo`` all agree. An
    empty ``configured`` means nothing was stated, which is not a disagreement.
    """
    if not configured.strip():
        return False
    return configured.strip().strip("/") != indexed.strip().strip("/")


async def _source_prefix(dsn: str, dim: int) -> str:
    """The S3 prefix these documents were actually indexed from.

    Read from the stored ``source_key`` values rather than from configuration.
    ``source_bucket``/``source_prefix`` in the manifest are what document download
    links are built from, and they used to come from ambient .env -- so publishing
    BRAVO on a shell configured for FOXTROT wrote FOXTROT's bucket into BRAVO's manifest and
    would have pointed BRAVO's links at FOXTROT's bucket.
    """
    db = PgDatabase(dsn, dim=dim)
    try:
        rows = await db.fetch(
            "SELECT DISTINCT split_part(source_key, '/', 1) AS root "
            "FROM document_metadata WHERE source_key <> '' ORDER BY 1"
        )
    finally:
        await db.aclose()
    roots = [str(r["root"]) for r in rows if r["root"]]
    if len(roots) != 1:
        raise SystemExit(
            f"Cannot determine the source prefix: the index spans {roots or 'no'} "
            "prefixes. A snapshot must come from exactly one."
        )
    return f"{roots[0]}/"


async def _tenant_codes(dsn: str, dim: int) -> list[str]:
    """The distinct tenant codes present in the index.

    Read from the chunks themselves: the point is to catch an index that was built
    without a tenant filter, which the CLI arguments would happily misreport.
    """
    db = PgDatabase(dsn, dim=dim)
    try:
        rows = await db.fetch(
            "SELECT DISTINCT payload->>'tenant' AS tenant FROM chunks "
            "WHERE payload->>'tenant' IS NOT NULL ORDER BY 1"
        )
        return [str(r["tenant"]) for r in rows if r["tenant"]]
    finally:
        await db.aclose()


async def _export(dsn: str, root: Path, dim: int, collection: str, namespace: str) -> dict[str, int]:
    """Copy a Postgres index into local Qdrant + SQLite files."""
    db = PgDatabase(dsn, dim=dim)
    vector = LocalVectorIndex(str(root / "qdrant"), collection=collection, dim=dim)
    keyword = KeywordIndex(str(root / "keyword.db"))
    metadata = MetadataStore(str(root / "metadata.db"))
    counts = {"chunks": 0, "documents": 0}
    try:
        await vector.ensure()

        rows = await db.fetch(
            "SELECT document_id, source_key, payload, facets FROM document_metadata"
        )
        for row in rows:
            meta = DocumentMetadata.model_validate(row["payload"] or {})
            await metadata.put(
                meta,
                namespace=namespace,
                source_key=row["source_key"] or "",
                facets=dict(row["facets"] or {}),
            )
        counts["documents"] = len(rows)
        _log.info("exported_documents", documents=counts["documents"])

        last = ""
        while True:
            chunks = await db.fetch(
                "SELECT chunk_id, text, payload, embedding::text AS embedding FROM chunks "
                "WHERE chunk_id > %s ORDER BY chunk_id LIMIT %s",
                (last, _BATCH),
            )
            if not chunks:
                break
            await vector.upsert(
                [(c["chunk_id"], _parse_vector(c["embedding"]), dict(c["payload"] or {}))
                 for c in chunks if c["embedding"]]
            )
            await keyword.add(
                [(c["chunk_id"], c["text"] or "", dict(c["payload"] or {})) for c in chunks]
            )
            counts["chunks"] += len(chunks)
            last = chunks[-1]["chunk_id"]
            if counts["chunks"] % (_BATCH * 10) == 0:
                _log.info("exported_chunks", chunks=counts["chunks"])
    finally:
        for store in (vector, keyword, metadata):
            await store.aclose()
        await db.aclose()
    return counts


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


async def _upload(container: Container, bucket: str, prefix: str, root: Path) -> int:
    """Upload every file, manifest LAST: its ETag is the snapshot version, so a
    consumer must never see a new one pointing at a half-uploaded snapshot."""
    s3 = container.s3()
    paths = [p for p in _files(root) if p.name != MANIFEST_KEY]
    for path in paths:
        key = f"{prefix}/{path.relative_to(root).as_posix()}"
        _log.info("uploading", key=key, bytes=path.stat().st_size)
        await s3.put_bytes(bucket, key, await asyncio.to_thread(path.read_bytes))
    await s3.put_bytes(
        bucket,
        f"{prefix}/{MANIFEST_KEY}",
        await asyncio.to_thread((root / MANIFEST_KEY).read_bytes),
        content_type="application/json",
    )
    return len(paths) + 1


async def _bucket_region(container: Container, bucket: str) -> str:
    """Recorded so the agent signs document links for the right region."""
    try:
        async with container.aws_factory().client("s3") as client:
            location = (await client.get_bucket_location(Bucket=bucket)).get("LocationConstraint")
    except Exception as exc:
        _log.warning("source_region_lookup_failed", bucket=bucket, error=str(exc))
        return ""
    return {None: "us-east-1", "": "us-east-1", "EU": "eu-west-1"}.get(location, location)


async def _publish(args: argparse.Namespace) -> int:
    configure_logging()
    core = get_settings()
    search = SearchSettings()
    for label, value in (("--environment", args.environment), ("--tenant", args.tenant)):
        if not _SEGMENT_RE.match(value):
            print(f"{label} {value!r} is not a valid path segment", file=sys.stderr)
            return 2

    prefix = (args.prefix or f"DocumentSearch/{args.environment}/{args.tenant}").strip("/")
    bucket = args.bucket or core.s3.bucket
    root = Path(args.out_dir) / args.environment / args.tenant
    if root.exists():
        await asyncio.to_thread(shutil.rmtree, root, True)
    root.mkdir(parents=True, exist_ok=True)

    counts = await _export(
        args.dsn or search.postgres_dsn, root, search.embed_dim, search.collection, search.namespace
    )
    if not counts["chunks"]:
        print("Refusing to publish: the index is empty (0 chunks).", file=sys.stderr)
        return 2

    container = Container(core)
    try:
        # What the index actually holds, read back from the data rather than taken
        # on trust from the CLI argument. The agent enforces this on every query, so
        # a snapshot built with a bad tenant filter cannot serve foreign documents.
        dsn = args.dsn or search.postgres_dsn
        tenant_codes = await _tenant_codes(dsn, search.embed_dim)

        # The manifest's source_* fields are what download links are minted from.
        # Take them from the index, and refuse if the ambient/CLI configuration
        # disagrees -- that mismatch is a shell configured for another tenant.
        indexed_prefix = await _source_prefix(dsn, search.embed_dim)
        configured_prefix = (args.source_prefix or core.s3.prefix or "").strip()
        if prefix_disagrees(indexed_prefix, configured_prefix):
            print(
                f"Refusing to publish: these documents were indexed from "
                f"{indexed_prefix!r}, but the configured source prefix is "
                f"{configured_prefix!r}. Document links would point at the wrong "
                f"bucket. Pass --source-bucket/--source-prefix, or run with the "
                f"S3__BUCKET and S3__PREFIX this tenant was indexed from.",
                file=sys.stderr,
            )
            return 2
        source_bucket = args.source_bucket or core.s3.bucket
        # Signed for the bucket's OWN region: a presigned URL made with the
        # wrong region is rejected with HTTP 400.
        source_region = await _bucket_region(container, source_bucket)
        if len(tenant_codes) > 1:
            print(
                f"Refusing to publish: index holds several tenants {tenant_codes}. "
                "Re-index with SEARCH_MONGO_TENANT_CODES set.",
                file=sys.stderr,
            )
            return 2
        manifest: dict[str, Any] = {
            "tenant_codes": tenant_codes,
            "manifest_version": MANIFEST_VERSION,
            "tenant": args.tenant,
            "environment": args.environment,
            "collection": search.collection,
            "embed_dim": search.embed_dim,
            "namespace": search.namespace,
            "built_at": datetime.now(UTC).isoformat(),
            "source_bucket": source_bucket,
            "source_region": source_region,
            "source_prefix": indexed_prefix,
            "counts": counts,
        }
        (root / MANIFEST_KEY).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        total = sum(p.stat().st_size for p in _files(root))
        print(f"\n  files     {len(_files(root))} ({total / 1_048_576:.1f} MiB)")
        for path in _files(root):
            print(f"     {path.relative_to(root).as_posix():48} {path.stat().st_size / 1_048_576:8.1f} MiB")
        print(f"  counts    {counts}")
        # The RESOLVED values, i.e. exactly what the manifest records. Printing
        # ambient config here once hid a manifest pointing at another tenant's
        # bucket -- the summary is what a human checks before a publish.
        print(f"  source    s3://{source_bucket}/{indexed_prefix} ({source_region or 'region unknown'})")
        print(f"  target    s3://{bucket}/{prefix}/")
        if args.dry_run:
            print("\n  --dry-run: nothing uploaded.")
            return 0
        uploaded = await _upload(container, bucket, prefix, root)
        print(f"\n  published {uploaded} object(s) to s3://{bucket}/{prefix}/")
    finally:
        await container.aclose()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", required=True, help="e.g. acmedev")
    parser.add_argument("--tenant", required=True, help="e.g. foxtrot")
    parser.add_argument("--bucket", default="", help="target bucket (default: S3__BUCKET)")
    parser.add_argument("--prefix", default="", help="default DocumentSearch/<environment>/<tenant>")
    parser.add_argument("--dsn", default="", help="source Postgres index (default: SEARCH_POSTGRES_DSN)")
    parser.add_argument("--out-dir", default="./data/snapshots/files")
    parser.add_argument(
        "--source-bucket",
        default=None,
        help="Bucket the documents live in (default: S3__BUCKET). Recorded in the "
        "manifest and used to mint download links.",
    )
    parser.add_argument(
        "--source-prefix",
        default=None,
        help="Prefix the documents were indexed from; checked against the index.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_publish(args)))


if __name__ == "__main__":
    main()
