"""Tests for IngestionService orchestration: new/unchanged/changed/duplicate/skip."""

from __future__ import annotations

from collections.abc import Callable

from rag_core.aws import S3ObjectSummary
from rag_core.config import Settings
from rag_core.models import DocumentEvent, PipelineMessage
from rag_core.utils import document_id
from rag_ingestion.config import IngestionSettings
from rag_ingestion.dedup import InMemoryDedupLedger
from rag_ingestion.models import SyncRequest

# Type aliases for the factory fixtures (defined in conftest.py).
MakeSummary = Callable[..., S3ObjectSummary]
BuildService = Callable[..., tuple[object, object]]


async def test_sync_new_emits_message(
    make_summary: MakeSummary,
    build_service: BuildService,
    core_settings: Settings,
    ingestion_settings: IngestionSettings,
    ledger: InMemoryDedupLedger,
) -> None:
    objects = [make_summary("Test/a.pdf", etag="e1", size=100)]
    contents = {"Test/a.pdf": b"hello world"}
    service, sqs = build_service(
        objects=objects,
        contents=contents,
        core_settings=core_settings,
        ingestion_settings=ingestion_settings,
        ledger=ledger,
    )

    summary = await service.sync(SyncRequest())

    assert summary.scanned == 1
    assert summary.new == 1
    assert summary.emitted == 1
    assert len(sqs.sent) == 1

    # The emitted body is a valid pipeline message pointing at a DISCOVERED event.
    msg = PipelineMessage[DocumentEvent].model_validate_json(sqs.sent[0])
    assert msg.stage == "ingestion"
    assert msg.payload.s3_uri == "s3://test-bucket/Test/a.pdf"
    assert msg.payload.content_checksum is not None


async def test_sync_second_run_is_unchanged(
    make_summary: MakeSummary,
    build_service: BuildService,
    core_settings: Settings,
    ingestion_settings: IngestionSettings,
    ledger: InMemoryDedupLedger,
) -> None:
    objects = [make_summary("Test/a.pdf", etag="e1", size=100)]
    contents = {"Test/a.pdf": b"hello world"}

    s1, _ = build_service(
        objects=objects, contents=contents, core_settings=core_settings,
        ingestion_settings=ingestion_settings, ledger=ledger,
    )
    first = await s1.sync(SyncRequest())
    assert first.new == 1

    # Same bytes + same ledger -> unchanged, nothing emitted.
    s2, sqs2 = build_service(
        objects=objects, contents=contents, core_settings=core_settings,
        ingestion_settings=ingestion_settings, ledger=ledger,
    )
    second = await s2.sync(SyncRequest())
    assert second.unchanged == 1
    assert second.new == 0
    assert second.emitted == 0
    assert sqs2.sent == []


async def test_sync_changed_creates_new_version(
    make_summary: MakeSummary,
    build_service: BuildService,
    core_settings: Settings,
    ingestion_settings: IngestionSettings,
    ledger: InMemoryDedupLedger,
) -> None:
    s1, _ = build_service(
        objects=[make_summary("Test/a.pdf", etag="e1", size=100)],
        contents={"Test/a.pdf": b"version-1"},
        core_settings=core_settings, ingestion_settings=ingestion_settings, ledger=ledger,
    )
    await s1.sync(SyncRequest())

    # Same key, new etag + new bytes -> changed, version increments.
    s2, _ = build_service(
        objects=[make_summary("Test/a.pdf", etag="e2", size=120)],
        contents={"Test/a.pdf": b"version-2-different"},
        core_settings=core_settings, ingestion_settings=ingestion_settings, ledger=ledger,
    )
    summary = await s2.sync(SyncRequest())
    assert summary.changed == 1
    assert summary.emitted == 1

    record = await ledger.get(document_id("test-bucket", "Test/a.pdf"))
    assert record is not None
    assert record.version == 2
    assert record.physical_id is not None


async def test_sync_detects_duplicate_bytes(
    make_summary: MakeSummary,
    build_service: BuildService,
    core_settings: Settings,
    ingestion_settings: IngestionSettings,
    ledger: InMemoryDedupLedger,
) -> None:
    objects = [
        make_summary("Test/a.pdf", etag="e1", size=10),
        make_summary("Test/copy.pdf", etag="e2", size=10),
    ]
    contents = {"Test/a.pdf": b"identical", "Test/copy.pdf": b"identical"}
    service, _ = build_service(
        objects=objects, contents=contents, core_settings=core_settings,
        ingestion_settings=ingestion_settings, ledger=ledger,
    )

    summary = await service.sync(SyncRequest())
    assert summary.new == 1
    assert summary.duplicate == 1
    assert summary.emitted == 1  # only the original is emitted


async def test_sync_skips_oversize_objects(
    make_summary: MakeSummary,
    build_service: BuildService,
    core_settings: Settings,
    ledger: InMemoryDedupLedger,
) -> None:
    settings = IngestionSettings(max_object_size_mb=1, sync_concurrency=2)
    service, sqs = build_service(
        objects=[make_summary("Test/big.pdf", etag="e1", size=2 * 1024 * 1024)],
        contents={},  # never streamed: skipped before download
        core_settings=core_settings, ingestion_settings=settings, ledger=ledger,
    )
    summary = await service.sync(SyncRequest())
    assert summary.skipped == 1
    assert summary.emitted == 0
    assert sqs.sent == []


async def test_dry_run_emits_nothing_and_leaves_ledger_empty(
    make_summary: MakeSummary,
    build_service: BuildService,
    core_settings: Settings,
    ingestion_settings: IngestionSettings,
    ledger: InMemoryDedupLedger,
) -> None:
    service, sqs = build_service(
        objects=[make_summary("Test/a.pdf", etag="e1", size=10)],
        contents={"Test/a.pdf": b"data"},
        core_settings=core_settings, ingestion_settings=ingestion_settings, ledger=ledger,
    )
    summary = await service.sync(SyncRequest(dry_run=True))
    assert summary.new == 1
    assert summary.emitted == 0
    assert sqs.sent == []
    # Nothing persisted on a dry run.
    assert await ledger.get(document_id("test-bucket", "Test/a.pdf")) is None
