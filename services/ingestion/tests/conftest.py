"""Shared fixtures + in-process fakes for ingestion tests.

The fakes mimic exactly the surface ``IngestionService`` uses (S3 list/stream,
SQS send_batch), so the orchestration logic is tested without any AWS/Redis.
Helpers are exposed as factory fixtures (``make_summary``, ``build_service``) so
tests need no cross-module imports — robust under multi-suite pytest runs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime

import pytest

from rag_core.aws import S3ObjectSummary
from rag_core.config import S3Settings, Settings, SQSSettings
from rag_ingestion.config import ChangeDetection, IngestionSettings
from rag_ingestion.dedup import InMemoryDedupLedger
from rag_ingestion.service import IngestionService


class FakeS3:
    """Minimal async S3 stand-in backed by in-memory objects + bytes."""

    def __init__(self, objects: list[S3ObjectSummary], contents: dict[str, bytes]) -> None:
        self._objects = objects
        self._contents = contents

    async def list_objects(
        self, bucket: str, prefix: str = "", *, page_size: int = 1000
    ) -> AsyncIterator[S3ObjectSummary]:
        for obj in self._objects:
            if obj.key.startswith(prefix):
                yield obj

    async def stream(
        self, bucket: str, key: str, *, chunk_size: int = 1024 * 1024
    ) -> AsyncIterator[bytes]:
        data = self._contents[key]
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]


class FakeSQS:
    """Captures emitted message bodies; never fails."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_batch(self, queue_url: str, bodies: list[str]) -> list[str]:
        self.sent.extend(bodies)
        return []


def _make_summary(
    key: str, *, etag: str, size: int, bucket: str = "test-bucket"
) -> S3ObjectSummary:
    return S3ObjectSummary(
        bucket=bucket,
        key=key,
        size_bytes=size,
        etag=etag,
        last_modified=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _build_service(
    *,
    objects: list[S3ObjectSummary],
    contents: dict[str, bytes],
    core_settings: Settings,
    ingestion_settings: IngestionSettings,
    ledger: InMemoryDedupLedger,
) -> tuple[IngestionService, FakeSQS]:
    sqs = FakeSQS()
    service = IngestionService(
        s3=FakeS3(objects, contents),  # type: ignore[arg-type]
        sqs=sqs,  # type: ignore[arg-type]
        ledger=ledger,
        core_settings=core_settings,
        settings=ingestion_settings,
    )
    return service, sqs


@pytest.fixture
def make_summary() -> Callable[..., S3ObjectSummary]:
    return _make_summary


@pytest.fixture
def build_service() -> Callable[..., tuple[IngestionService, FakeSQS]]:
    return _build_service


@pytest.fixture
def core_settings() -> Settings:
    # init kwargs take precedence over env in pydantic-settings.
    return Settings(
        s3=S3Settings(bucket="test-bucket", prefix=""),
        sqs=SQSSettings(ingestion_queue_url="http://localhost/queue"),
    )


@pytest.fixture
def ingestion_settings() -> IngestionSettings:
    return IngestionSettings(
        sync_concurrency=4,
        emit_batch_size=10,
        change_detection=ChangeDetection.CHECKSUM,
        max_object_size_mb=0,  # no size limit by default
        emit_to_sqs=True,
    )


@pytest.fixture
def ledger() -> InMemoryDedupLedger:
    return InMemoryDedupLedger()
