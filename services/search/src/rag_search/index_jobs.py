"""Scheduled index jobs: the Lambda entry point and the wiring it shares with the CLI.

EventBridge Scheduler invokes ``lambda_handler`` with::

    {"job": "index", "target": "FOXTROT"}                    # incremental indexing
    {"job": "purge", "target": "BRAVO", "dry_run": false}  # DeleteMark=1 purge

``target`` names an entry of ``INDEX_TARGETS`` (JSON in the environment)::

    {
      "FOXTROT":  {"collection": "FOXTROT_DocumentSearch_DEV",  "metadata_database": "FOXTROT_DEV"},
      "BRAVO": {"collection": "BRAVO_DocumentSearch_DEV", "metadata_database": "BRAVO_DEV"}
    }

Per target, ``bucket``/``prefix``/``tenant_codes`` default to what the collection's
manifest recorded at build time (``scripts/mongo_index.py manifest``), so the job
always feeds the index from the source it was built from. ``metadata_collection``
defaults to ``SEARCH_MONGO_COLLECTION``.

The Atlas connection string is read from Secrets Manager (``MONGO_URI_SECRET_ARN``)
once per container; AWS access is the function's IAM role -- no keys.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from rag_core.config import S3Settings, Settings, get_settings
from rag_core.logging import configure_logging, get_logger
from rag_search.config import SearchSettings, StoreBackend
from rag_search.factory import Components, build_components, build_mongo_index
from rag_search.incremental import (
    AlreadyRunning,
    DeleteMarkPurger,
    IncrementalIndexer,
    MongoIndexState,
)
from rag_search.stores import MongoVectorIndex

_log = get_logger("rag_search.index_jobs")


@dataclass(slots=True)
class Target:
    name: str
    collection: str
    bucket: str
    prefix: str
    tenant_codes: str
    metadata_database: str
    metadata_collection: str


def load_targets() -> dict[str, dict[str, Any]]:
    raw = os.environ.get("INDEX_TARGETS", "").strip()
    if not raw:
        raise ValueError("INDEX_TARGETS is not set (JSON: {name: {collection, ...}})")
    targets = json.loads(raw)
    if not isinstance(targets, dict) or not targets:
        raise ValueError("INDEX_TARGETS must be a non-empty JSON object")
    return {str(k).upper(): v for k, v in targets.items()}


async def resolve_target(name: str, spec: dict[str, Any], search: SearchSettings) -> Target:
    """Fill gaps in a target from its collection's manifest."""
    if not spec.get("collection"):
        raise ValueError(f"target {name}: 'collection' is required")
    manifest: dict[str, Any] = {}
    if not all(spec.get(k) for k in ("bucket", "prefix", "tenant_codes")):
        db = build_mongo_index(
            search.model_copy(update={"index_mongo_collection": spec["collection"]}),
            dim=search.embed_dim,
        )
        try:
            manifest = await db.read_manifest()
        finally:
            await db.aclose()
    codes = spec.get("tenant_codes") or manifest.get("tenant_codes") or ""
    target = Target(
        name=name,
        collection=spec["collection"],
        bucket=spec.get("bucket") or manifest.get("source_bucket") or "",
        prefix=str(spec["prefix"] if spec.get("prefix") is not None else manifest.get("source_prefix") or ""),
        tenant_codes=",".join(codes) if isinstance(codes, list) else str(codes),
        metadata_database=spec.get("metadata_database") or search.mongo_database,
        metadata_collection=spec.get("metadata_collection") or search.mongo_collection,
    )
    if not target.bucket:
        raise ValueError(f"target {name}: no bucket configured and none in the manifest")
    if not target.tenant_codes:
        # Tenants share S3 folders and catalogue databases: unfiltered, this index
        # would take in every tenant's files.
        raise ValueError(f"target {name}: no tenant_codes configured and none in the manifest")
    return target


def build_for_target(target: Target, core: Settings, search: SearchSettings) -> Components:
    core = core.model_copy(update={"s3": S3Settings(bucket=target.bucket, prefix=target.prefix)})
    search = search.model_copy(
        update={
            "store_backend": StoreBackend.MONGO,
            "index_mongo_collection": target.collection,
            "mongo_database": target.metadata_database,
            "mongo_collection": target.metadata_collection,
            "mongo_tenant_codes": target.tenant_codes,
        }
    )
    if not search.mongo_uri:
        raise ValueError("SEARCH_MONGO_URI (or MONGO_URI_SECRET_ARN) is required")
    return build_components(core, search)


async def run_job(
    job: str,
    target_name: str,
    *,
    deadline: float | None = None,
    dry_run: bool = False,
    core: Settings | None = None,
    search: SearchSettings | None = None,
) -> dict[str, Any]:
    core = core or get_settings()
    search = search or SearchSettings()
    targets = load_targets()
    key = target_name.upper()
    if key not in targets:
        raise ValueError(f"unknown target {target_name!r}; configured: {sorted(targets)}")
    target = await resolve_target(key, targets[key], search)
    components = build_for_target(target, core, search)
    try:
        mongo = components.indexer._mongo
        assert mongo is not None  # build_for_target requires the catalogue
        if job == "index":
            assert isinstance(components.vector, MongoVectorIndex)
            db = components.vector._db
            await db.ensure_schema()
            indexer = IncrementalIndexer(
                indexer=components.indexer,
                metadata_store=components.metadata,
                mongo=mongo,
                state=MongoIndexState(db),
                s3=components.indexer._s3,
                bucket=target.bucket,
                prefix=target.prefix,
                collection=target.collection,
                namespace=search.namespace,
                initial_lookback=timedelta(hours=float(os.environ.get("INDEX_INITIAL_LOOKBACK_H", "24"))),
                pending_max_age=timedelta(days=float(os.environ.get("INDEX_PENDING_MAX_AGE_D", "7"))),
            )
            try:
                result = (await indexer.run(deadline=deadline)).as_dict()
            except AlreadyRunning as exc:
                _log.warning("incremental_index_skipped", reason=str(exc))
                return {"status": "skipped", "reason": str(exc), "target": key}
        elif job == "purge":
            purger = DeleteMarkPurger(
                indexer=components.indexer,
                metadata_store=components.metadata,
                mongo=mongo,
                collection=target.collection,
                namespace=search.namespace,
            )
            result = (await purger.run(dry_run=dry_run)).as_dict()
        else:
            raise ValueError(f"unknown job {job!r}; expected 'index' or 'purge'")
    finally:
        await components.aclose()
    return {"status": "ok", "job": job, "target": key, **result}


# --------------------------------------------------------------------------
# Lambda
# --------------------------------------------------------------------------

_bootstrapped = False


def _bootstrap() -> None:
    """Once per container: logging + the Mongo URI out of Secrets Manager."""
    global _bootstrapped
    if _bootstrapped:
        return
    configure_logging(level=os.environ.get("LOG_LEVEL", "INFO"), json_logs=True, service_name="index-jobs")
    arn = os.environ.get("MONGO_URI_SECRET_ARN", "").strip()
    if arn and not os.environ.get("SEARCH_MONGO_URI"):
        import boto3

        secret = boto3.client("secretsmanager").get_secret_value(SecretId=arn)["SecretString"]
        # A plain string, or a JSON secret with a "uri" key.
        if secret.lstrip().startswith("{"):
            secret = json.loads(secret)["uri"]
        os.environ["SEARCH_MONGO_URI"] = secret
    get_settings.cache_clear()
    _bootstrapped = True


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    _bootstrap()
    job = str(event.get("job") or "index")
    target = str(event.get("target") or "")
    if not target:
        raise ValueError("event needs a 'target' (one of INDEX_TARGETS)")
    deadline = None
    if context is not None and hasattr(context, "get_remaining_time_in_millis"):
        # Leave room for the document in flight (its own timeout) plus state writes.
        margin_s = SearchSettings().doc_timeout_s + 60
        deadline = time.monotonic() + context.get_remaining_time_in_millis() / 1000 - margin_s
    result = asyncio.run(run_job(job, target, deadline=deadline, dry_run=bool(event.get("dry_run"))))
    _log.info("index_job_result", **{k: v for k, v in result.items() if k != "errors"})
    return result
