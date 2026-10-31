"""Fixtures + in-process fakes for processing tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from rag_core.config import S3Settings, Settings, SQSSettings
from rag_core.models.content import ExtractedContent, Page
from rag_processing.config import ProcessingSettings
from rag_processing.extractors.base import ExtractionContext, Extractor
from rag_processing.extractors.registry import ExtractorRegistry
from rag_processing.ocr.base import OCREngine
from rag_processing.service import ProcessingService
from rag_processing.store import InMemoryContentStore


class FakeS3:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self._objects = objects
        self.put_calls: list[tuple[str, str]] = []

    async def get_bytes(self, bucket: str, key: str, *, version_id: str | None = None) -> bytes:
        from rag_core.resilience import NonRetryableError

        try:
            return self._objects[key]
        except KeyError as exc:
            raise NonRetryableError(f"missing key {key}") from exc

    async def put_bytes(
        self, bucket: str, key: str, data: bytes, *, content_type: str = "application/octet-stream"
    ) -> None:
        self._objects[key] = data
        self.put_calls.append((bucket, key))

    async def head_object(self, bucket: str, key: str) -> dict[str, object]:
        return {}


class FakeSQS:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, queue_url: str, body: str, *, attributes: dict | None = None) -> str:
        self.sent.append(body)
        return "msg-1"


class FakeOCREngine(OCREngine):
    name = "fake"

    async def ocr_image(self, data: bytes) -> Page:
        return Page(number=1, text="ocr image text", is_scanned=True)

    async def ocr_pdf(self, *, bucket: str, key: str, max_pages: int = 200) -> list[Page]:
        return [Page(number=1, text="ocr pdf page one", is_scanned=True)]


class StubExtractor(Extractor):
    """Returns whatever ExtractedContent it's constructed with (for service tests)."""

    def __init__(self, supported, content_factory: Callable[[ExtractionContext], ExtractedContent]):
        self.supported = supported
        self._factory = content_factory

    async def extract(self, data: bytes, ctx: ExtractionContext) -> ExtractedContent:
        return self._factory(ctx)


@pytest.fixture
def core_settings() -> Settings:
    return Settings(
        s3=S3Settings(bucket="test-bucket", prefix=""),
        sqs=SQSSettings(ingestion_queue_url="http://localhost/in"),
    )


@pytest.fixture
def processing_settings() -> ProcessingSettings:
    return ProcessingSettings(
        run_worker=False,
        output_queue_url="http://localhost/out",
        detect_language=True,
        detect_pii=True,
        classify=True,
    )


@pytest.fixture
def build_service(
    core_settings: Settings, processing_settings: ProcessingSettings
) -> Callable[..., tuple[ProcessingService, FakeS3, FakeSQS]]:
    def _build(
        *,
        objects: dict[str, bytes],
        registry: ExtractorRegistry,
        ocr_engine: OCREngine | None = None,
        settings: ProcessingSettings | None = None,
    ) -> tuple[ProcessingService, FakeS3, FakeSQS]:
        s3 = FakeS3(objects)
        sqs = FakeSQS()
        store = InMemoryContentStore()
        service = ProcessingService(
            s3=s3,  # type: ignore[arg-type]
            sqs=sqs,  # type: ignore[arg-type]
            registry=registry,
            ocr_engine=ocr_engine or FakeOCREngine(),
            content_store=store,
            core_settings=core_settings,
            settings=settings or processing_settings,
        )
        return service, s3, sqs

    return _build
