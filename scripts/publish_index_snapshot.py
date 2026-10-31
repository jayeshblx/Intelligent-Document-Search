"""Publish the Postgres search index to S3 as a restorable snapshot.

The Postgres backend keeps vectors, keyword text and metadata in one database, so
a snapshot is a single ``pg_dump`` rather than the Qdrant/SQLite file set the local
backend publishes. Layout written here:

    s3://{bucket}/{prefix}/
        rag-<timestamp>.dump   -- pg_dump custom format (compressed, restorable)
        manifest.json          -- version marker + how to open the snapshot

``manifest.json`` declares ``"backend": "postgres"``, which tells
``rag_agentic.index_store`` to download this dump, verify its sha256, and restore it
into a PostgreSQL server inside the agent's own container. Its ETag is the snapshot version, so it is uploaded LAST --
a consumer must never see a new version marker pointing at a half-written dump.

Usage
-----
The environment and tenant are required and select the S3 folder the agent reads
for them (``DocumentSearch/<environment>/<tenant>``), so publishing one tenant's
index can never land in another's folder by default:

    uv run python scripts/publish_index_snapshot.py --environment portaltest --tenant corp
    uv run python scripts/publish_index_snapshot.py --environment portaltest --tenant corp --dry-run
    # a legacy flat folder, bound with AGENTIC_INDEX_PREFIX_OVERRIDE_TENANT:
    uv run python scripts/publish_index_snapshot.py --environment foxtrot-Dev --tenant foxtrot \
        --prefix DocumentSearch/foxtrot-Dev

``--dsn`` must be the database holding THIS tenant's index.

Restoring one:
    aws s3 cp s3://<bucket>/<prefix>/rag-<ts>.dump .
    pg_restore --clean --if-exists -d "postgresql://rag:rag@host:5432/rag" rag-<ts>.dump
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from rag_search.config import SearchSettings

from rag_core.config import get_settings
from rag_core.di import Container
from rag_core.logging import configure_logging, get_logger

_log = get_logger("publish_index_snapshot")

MANIFEST_KEY = "manifest.json"
MANIFEST_VERSION = 2  # v1 = local Qdrant/SQLite file set; v2 adds "backend"
_COUNTED_TABLES = ("chunks", "document_metadata", "document_facet")
# Same rule the agent applies to resolved names: they become S3 path segments.
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _dsn_parts(dsn: str) -> tuple[str, str]:
    """(user, database) from a postgresql:// DSN -- what pg_dump needs."""
    parts = urlsplit(dsn)
    return parts.username or "postgres", (parts.path or "/postgres").lstrip("/") or "postgres"


def _run_pg_dump(dsn: str, container: str, destination: Path) -> None:
    """Dump the database to ``destination`` in pg_dump custom format.

    Runs inside the container so the client version always matches the server (a
    mismatched local pg_dump refuses to talk to it), and so no Postgres client
    tooling is required on the host at all.
    """
    user, database = _dsn_parts(dsn)
    command = [
        "docker", "exec", container,
        "pg_dump", "-U", user, "-d", database,
        "--format=custom", "--compress=9", "--no-owner", "--no-privileges",
    ]
    _log.info("pg_dump_start", container=container, database=database, dest=str(destination))
    with destination.open("wb") as sink:
        proc = subprocess.run(command, stdout=sink, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"pg_dump failed (exit {proc.returncode}): {proc.stderr.decode(errors='replace')[:400]}"
        )
    if destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise RuntimeError("pg_dump produced an empty file")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counts(dsn: str) -> dict[str, int]:
    """Row counts recorded in the manifest, so a consumer can sanity-check a restore."""
    import psycopg

    with psycopg.connect(dsn, connect_timeout=15) as conn:
        out: dict[str, int] = {}
        for table in _COUNTED_TABLES:  # a literal tuple, never caller input
            out[table] = int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
        out["embedded_chunks"] = int(
            conn.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
        )
    return out


async def _bucket_region(container: Container, bucket: str) -> str:
    """The source bucket's region, recorded so the agent signs links for it.

    A link signed for the wrong region is rejected by S3 with HTTP 400. Returns ""
    when the lookup is not permitted, so the agent falls back to its setting.
    """
    try:
        async with container.aws_factory().client("s3") as client:
            location = (await client.get_bucket_location(Bucket=bucket)).get(
                "LocationConstraint"
            )
    except Exception as exc:
        _log.warning("source_region_lookup_failed", bucket=bucket, error=str(exc))
        return ""
    # S3 reports us-east-1 as no constraint and eu-west-1 as the legacy "EU".
    return {None: "us-east-1", "": "us-east-1", "EU": "eu-west-1"}.get(location, location)


def _build_snapshot(
    args: argparse.Namespace, core: Any, search: SearchSettings, dsn: str, source_region: str
) -> tuple[Path, bytes, dict[str, Any]] | None:
    """Dump the database and build its manifest.

    Fully synchronous (pg_dump, hashing, file writes); the caller runs it off the
    event loop. Returns None when there is nothing worth publishing.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    dump_name = f"rag-{stamp}.dump"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_path = out_dir / dump_name

    counts = _counts(dsn)
    if counts["chunks"] == 0:
        return None

    _run_pg_dump(dsn, args.container, dump_path)
    size = dump_path.stat().st_size
    checksum = _sha256(dump_path)

    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "backend": "postgres",
        "tenant": args.tenant,
        "environment": args.environment,
        "collection": search.collection,
        "embed_dim": search.embed_dim,
        "namespace": search.namespace,
        "built_at": datetime.now(UTC).isoformat(),
        "source_bucket": core.s3.bucket,
        "source_region": source_region,
        "source_prefix": core.s3.prefix,
        "counts": counts,
        "dump": {
            "key": dump_name,
            "bytes": size,
            "sha256": checksum,
            "format": "pg_dump --format=custom --compress=9",
            "restore": f"pg_restore --clean --if-exists -d <dsn> {dump_name}",
        },
    }
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    (out_dir / MANIFEST_KEY).write_bytes(manifest_bytes)
    return dump_path, manifest_bytes, manifest


async def _publish(args: argparse.Namespace) -> int:
    configure_logging()
    core = get_settings()
    search = SearchSettings()

    for label, value in (("--environment", args.environment), ("--tenant", args.tenant)):
        if not _SEGMENT_RE.match(value):
            print(f"{label} {value!r} is not a valid path segment", file=sys.stderr)
            return 2
    dsn = args.dsn or search.postgres_dsn
    bucket = args.bucket or core.s3.bucket
    # The folder the agent derives for this tenant, unless explicitly overridden.
    prefix = (args.prefix or f"DocumentSearch/{args.environment}/{args.tenant}").strip("/")

    container = Container(core)
    try:
        source_region = await _bucket_region(container, core.s3.bucket)
    finally:
        await container.aclose()
    built = await asyncio.to_thread(_build_snapshot, args, core, search, dsn, source_region)
    if built is None:
        print("Refusing to publish: the index is empty (0 chunks).", file=sys.stderr)
        return 2
    dump_path, manifest_bytes, manifest = built
    dump_name = str(manifest["dump"]["key"])
    size = int(manifest["dump"]["bytes"])

    print(f"\n  dump      {dump_path}  ({size / 1_048_576:.1f} MiB)")
    print(f"  sha256    {manifest['dump']['sha256']}")
    print(f"  counts    {manifest['counts']}")
    print(f"  source    s3://{manifest['source_bucket']}/{manifest['source_prefix']} "
          f"({manifest['source_region'] or 'region unknown'})")
    print(f"  target    s3://{bucket}/{prefix}/")

    if args.dry_run:
        print("\n  --dry-run: nothing uploaded.")
        return 0

    payload = await asyncio.to_thread(dump_path.read_bytes)
    container = Container(core)
    s3 = container.s3()
    try:
        # Payload first, version marker last: a consumer polling manifest.json must
        # never see a new ETag pointing at a dump that is still uploading.
        _log.info("uploading_dump", key=f"{prefix}/{dump_name}", bytes=size)
        await s3.put_bytes(bucket, f"{prefix}/{dump_name}", payload)
        await s3.put_bytes(
            bucket, f"{prefix}/{MANIFEST_KEY}", manifest_bytes, content_type="application/json"
        )
    finally:
        await container.aclose()

    print(f"\n  published s3://{bucket}/{prefix}/{dump_name}")
    print(f"  published s3://{bucket}/{prefix}/{MANIFEST_KEY}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="", help="target bucket (default: S3__BUCKET)")
    parser.add_argument(
        "--prefix",
        default="",
        help="target prefix; default DocumentSearch/<environment>/<tenant>",
    )
    parser.add_argument("--dsn", default="", help="source DSN (default: SEARCH_POSTGRES_DSN)")
    parser.add_argument(
        "--container", default="agentic-rag-postgres-1", help="docker container running Postgres"
    )
    parser.add_argument("--environment", required=True, help="e.g. portaltest")
    parser.add_argument("--tenant", required=True, help="e.g. corp")
    parser.add_argument("--out-dir", default="./data/snapshots", help="where to write the dump")
    parser.add_argument("--dry-run", action="store_true", help="build locally, skip the upload")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_publish(args)))


if __name__ == "__main__":
    main()
