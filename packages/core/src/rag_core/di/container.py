"""The shared service container.

Each accessor lazily constructs and memoises its resource, so a service only pays
for what it uses (the ingestion service never opens a Qdrant connection). All
constructed resources that expose ``aclose`` are tracked and closed together.
"""

from __future__ import annotations

from typing import Any

from rag_core.aws import AwsSessionFactory, BedrockClient, S3Client, SQSClient
from rag_core.config import Settings, get_settings
from rag_core.logging import get_logger
from rag_core.stores.vector import QdrantVectorStore

_log = get_logger("rag_core.di")


class Container:
    """Owns shared, long-lived resources for a single service process."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._singletons: dict[str, Any] = {}
        self._closeables: list[Any] = []

    def _memo(self, key: str, factory: Any) -> Any:
        if key not in self._singletons:
            instance = factory()
            self._singletons[key] = instance
            if hasattr(instance, "aclose"):
                self._closeables.append(instance)
        return self._singletons[key]

    # ---- AWS ----
    def aws_factory(self) -> AwsSessionFactory:
        return self._memo("aws_factory", lambda: AwsSessionFactory(self.settings.aws))

    def s3(self) -> S3Client:
        return self._memo("s3", lambda: S3Client(self.aws_factory()))

    def sqs(self) -> SQSClient:
        return self._memo("sqs", lambda: SQSClient(self.aws_factory()))

    def bedrock(self) -> BedrockClient:
        return self._memo("bedrock", lambda: BedrockClient(self.settings.bedrock))

    # ---- Stores ----
    def vector_store(self) -> QdrantVectorStore:
        return self._memo("vector_store", lambda: QdrantVectorStore(self.settings.qdrant))

    # ---- lifecycle ----
    async def aclose(self) -> None:
        """Close every constructed resource, last-created first."""
        for resource in reversed(self._closeables):
            try:
                await resource.aclose()
            except Exception as exc:
                _log.warning("container_close_error", resource=type(resource).__name__, error=str(exc))
        self._closeables.clear()
        self._singletons.clear()

    async def __aenter__(self) -> Container:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
