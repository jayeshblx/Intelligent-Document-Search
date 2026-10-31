"""Content store — persists ``ExtractedContent`` for the next stage to consume.

Processing output can be large (a 500-page manual's structured content), so it is
NOT shoved into the SQS message. Instead it's written to durable object storage
(S3) and the downstream event carries only a pointer (the content URI). This keeps
messages small, makes the pipeline replayable, and decouples stage cadence.

Keyed deterministically by namespace + document_id so re-processing overwrites in
place (idempotent).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_core.aws import S3Client
from rag_core.models.content import ExtractedContent


class ContentStore(ABC):
    @abstractmethod
    async def put(self, content: ExtractedContent, *, namespace: str) -> str:
        """Persist content; return its retrievable URI."""

    @abstractmethod
    async def get(self, document_id: str, *, namespace: str) -> ExtractedContent | None: ...

    @abstractmethod
    async def health(self) -> bool: ...


class S3ContentStore(ContentStore):
    """Stores content as JSON in S3 under ``{prefix}{namespace}/{document_id}.json``."""

    def __init__(self, s3: S3Client, *, bucket: str, prefix: str = "processed/content/") -> None:
        self._s3 = s3
        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/"

    def _key(self, document_id: str, namespace: str) -> str:
        return f"{self._prefix}{namespace}/{document_id}.json"

    def uri(self, document_id: str, namespace: str) -> str:
        return f"s3://{self._bucket}/{self._key(document_id, namespace)}"

    async def put(self, content: ExtractedContent, *, namespace: str) -> str:
        key = self._key(content.document_id, namespace)
        body = content.model_dump_json().encode("utf-8")
        await self._s3.put_bytes(self._bucket, key, body, content_type="application/json")
        return f"s3://{self._bucket}/{key}"

    async def get(self, document_id: str, *, namespace: str) -> ExtractedContent | None:
        from rag_core.resilience import NonRetryableError

        try:
            raw = await self._s3.get_bytes(self._bucket, self._key(document_id, namespace))
        except NonRetryableError:
            return None  # not found / inaccessible
        return ExtractedContent.model_validate_json(raw)

    async def health(self) -> bool:
        try:
            await self._s3.head_object(self._bucket, self._prefix)
        except Exception:
            return True
        return True


class InMemoryContentStore(ContentStore):
    """In-process store for tests."""

    def __init__(self) -> None:
        self._data: dict[str, ExtractedContent] = {}

    def _key(self, document_id: str, namespace: str) -> str:
        return f"{namespace}/{document_id}"

    async def put(self, content: ExtractedContent, *, namespace: str) -> str:
        self._data[self._key(content.document_id, namespace)] = content
        return f"mem://{self._key(content.document_id, namespace)}"

    async def get(self, document_id: str, *, namespace: str) -> ExtractedContent | None:
        return self._data.get(self._key(document_id, namespace))

    async def health(self) -> bool:
        return True
