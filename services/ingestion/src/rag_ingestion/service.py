"""IngestionService — the orchestration core.

Flow per sync:
    list S3 prefix (streaming) -> bounded worker pool -> per object:
        change detection (ETag, then checksum) -> dedup (checksum index)
        -> build Document + version -> emit DISCOVERED to SQS -> update ledger

Design notes:
* A producer/consumer worker pool (not ``gather`` over all keys) bounds memory and
  concurrency so a 1M-object prefix streams through at a fixed footprint.
* Workers keep *local* counters and emit buffers, merged at the end — no shared
  mutable state, no locks on the hot path.
* Everything is idempotent: deterministic ids + checksum index mean re-running a
  sync converges (no duplicate work, no duplicate emits).
"""

from __future__ import annotations

import asyncio
import time

from rag_core.aws import S3Client, S3ObjectSummary, SQSClient
from rag_core.config import Settings
from rag_core.logging import bind_context, get_logger
from rag_core.models import Document, DocumentEvent, DocumentStatus, PipelineMessage
from rag_core.observability import counter
from rag_core.utils import document_id, new_message_id, sha256_astream
from rag_ingestion.config import ChangeDetection, IngestionSettings
from rag_ingestion.dedup import DedupLedger, DocRecord
from rag_ingestion.discovery import build_document
from rag_ingestion.models import Outcome, SyncRequest, SyncSummary

_log = get_logger("rag_ingestion.service")

# Metrics
_OBJECTS = counter("ingestion_objects_total", "Objects processed", ("outcome",))
_EMITTED = counter("ingestion_emitted_total", "Documents emitted to SQS")
_BYTES = counter("ingestion_downloaded_bytes_total", "Bytes downloaded for checksum")


class IngestionService:
    """Stateless (modulo injected ledger) ingestion orchestrator."""

    def __init__(
        self,
        *,
        s3: S3Client,
        sqs: SQSClient,
        ledger: DedupLedger,
        core_settings: Settings,
        settings: IngestionSettings,
    ) -> None:
        self._s3 = s3
        self._sqs = sqs
        self._ledger = ledger
        self._core = core_settings
        self._settings = settings

    async def sync(self, request: SyncRequest) -> SyncSummary:
        """Run an incremental sync and return an aggregate summary."""
        bucket = self._core.s3.bucket
        prefix = request.prefix if request.prefix is not None else self._core.s3.prefix
        concurrency = self._settings.sync_concurrency
        started = time.perf_counter()

        bind_context(sync_bucket=bucket, sync_prefix=prefix, namespace=request.namespace)
        _log.info("sync_started", concurrency=concurrency, dry_run=request.dry_run)

        queue: asyncio.Queue[S3ObjectSummary | None] = asyncio.Queue(maxsize=concurrency * 4)
        partials: list[SyncSummary] = []

        async def worker() -> None:
            local = SyncSummary(
                bucket=bucket, prefix=prefix, namespace=request.namespace, dry_run=request.dry_run
            )
            buf: list[str] = []
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        queue.task_done()
                        break
                    try:
                        outcome, body = await self._process_one(item, request)
                        local.record(outcome)
                        _OBJECTS.labels(outcome=outcome.value).inc()
                        if body and not request.dry_run:
                            buf.append(body)
                            if len(buf) >= self._settings.emit_batch_size:
                                local.emitted += await self._emit(buf)
                                buf.clear()
                    except Exception as exc:
                        local.record(Outcome.ERROR)
                        _OBJECTS.labels(outcome=Outcome.ERROR.value).inc()
                        if len(local.error_samples) < 5:
                            local.error_samples.append(f"{item.key}: {exc}")
                        _log.warning("object_failed", key=item.key, error=str(exc))
                    finally:
                        queue.task_done()
            finally:
                if buf and not request.dry_run:
                    try:
                        local.emitted += await self._emit(buf)
                    except Exception as exc:
                        local.error_samples.append(f"flush: {exc}")
                partials.append(local)

        async def produce() -> None:
            async for summary_obj in self._s3.list_objects(bucket, prefix):
                await queue.put(summary_obj)
            for _ in range(concurrency):
                await queue.put(None)  # one sentinel per worker

        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await produce()
        await asyncio.gather(*workers)

        summary = _merge(partials, bucket=bucket, prefix=prefix, request=request)
        summary.duration_s = round(time.perf_counter() - started, 3)
        _EMITTED.inc(summary.emitted)
        _log.info(
            "sync_complete",
            scanned=summary.scanned,
            new=summary.new,
            changed=summary.changed,
            unchanged=summary.unchanged,
            duplicate=summary.duplicate,
            errors=summary.errors,
            emitted=summary.emitted,
            duration_s=summary.duration_s,
        )
        return summary

    async def _process_one(
        self, obj: S3ObjectSummary, request: SyncRequest
    ) -> tuple[Outcome, str | None]:
        """Classify one object and (unless dry-run) emit + persist its state."""
        # Size guard protects downstream workers from pathological inputs.
        max_bytes = self._settings.max_object_size_mb * 1024 * 1024
        if max_bytes and obj.size_bytes > max_bytes:
            return Outcome.SKIPPED, None

        logical_id = document_id(obj.bucket, obj.key)
        prior = await self._ledger.get(logical_id)

        # Fast path: ETag unchanged and we trust ETags -> nothing to do.
        if (
            prior
            and prior.etag == obj.etag
            and self._settings.change_detection is ChangeDetection.ETAG
            and not self._settings.always_verify_checksum
        ):
            return Outcome.UNCHANGED, None

        # We need the content checksum when: doing checksum-based detection, the
        # object is new/ETag-changed, or paranoid verification is on.
        needs_checksum = (
            self._settings.change_detection is ChangeDetection.CHECKSUM
            or self._settings.always_verify_checksum
            or prior is None
            or prior.etag != obj.etag
        )
        checksum: str | None = None
        if needs_checksum:
            checksum = await self._checksum(obj)

        # ETag changed but bytes identical (e.g. re-upload) -> not a real change.
        if prior and checksum and prior.checksum == checksum:
            if not request.dry_run:
                prior.etag = obj.etag
                await self._ledger.put(prior)
            return Outcome.UNCHANGED, None

        # Deduplication: identical bytes already registered under another doc.
        if checksum:
            owner = await self._ledger.find_by_checksum(checksum)
            if owner and owner != logical_id:
                return Outcome.DUPLICATE, None

        # New document or new version of an existing one.
        if prior is None:
            version, outcome, supersedes = 1, Outcome.NEW, None
        else:
            version, outcome, supersedes = prior.version + 1, Outcome.CHANGED, prior.physical_id

        physical_id = document_id(obj.bucket, obj.key, version_id=checksum or obj.etag)
        document = build_document(
            obj,
            doc_id=physical_id,
            namespace=request.namespace,
            version=version,
            checksum=checksum,
            supersedes_id=supersedes,
        )
        body = self._build_message(document)

        if not request.dry_run:
            await self._ledger.put(
                DocRecord(
                    document_id=logical_id,
                    physical_id=physical_id,
                    etag=obj.etag,
                    checksum=checksum,
                    version=version,
                    status=DocumentStatus.DISCOVERED.value,
                )
            )
            if checksum:
                await self._ledger.register_checksum(checksum, logical_id)

        return outcome, body

    async def _checksum(self, obj: S3ObjectSummary) -> str:
        """Stream the object and compute sha256 without buffering the whole file."""
        digest = await sha256_astream(self._s3.stream(obj.bucket, obj.key))
        _BYTES.inc(obj.size_bytes)
        return digest

    def _build_message(self, document: Document) -> str:
        """Serialise a DISCOVERED pipeline message for SQS."""
        event = DocumentEvent(
            document_id=document.id,
            namespace=document.namespace,
            status=DocumentStatus.DISCOVERED,
            s3_uri=document.source.uri,
            content_checksum=document.content_checksum,
        )
        message: PipelineMessage[DocumentEvent] = PipelineMessage(
            message_id=new_message_id(),
            correlation_id=document.id,
            stage="ingestion",
            idempotency_key=f"{document.id}:processing",
            payload=event,
        )
        return message.model_dump_json()

    async def _emit(self, bodies: list[str]) -> int:
        """Send buffered messages to SQS in batches of 10; return success count."""
        if not self._settings.emit_to_sqs or not self._core.sqs.ingestion_queue_url:
            return 0
        emitted = 0
        for start in range(0, len(bodies), 10):
            chunk = bodies[start : start + 10]
            failed = await self._sqs.send_batch(self._core.sqs.ingestion_queue_url, chunk)
            emitted += len(chunk) - len(failed)
        return emitted


def _merge(
    partials: list[SyncSummary], *, bucket: str, prefix: str, request: SyncRequest
) -> SyncSummary:
    """Sum per-worker partial summaries into one."""
    total = SyncSummary(
        bucket=bucket, prefix=prefix, namespace=request.namespace, dry_run=request.dry_run
    )
    for p in partials:
        total.scanned += p.scanned
        total.new += p.new
        total.changed += p.changed
        total.unchanged += p.unchanged
        total.duplicate += p.duplicate
        total.skipped += p.skipped
        total.errors += p.errors
        total.emitted += p.emitted
        for sample in p.error_samples:
            if len(total.error_samples) < 10:
                total.error_samples.append(sample)
    return total
