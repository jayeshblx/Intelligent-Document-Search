"""Fixtures for agentic-service tests (fully offline: no AWS, no real LLM calls).

The interesting fixture is ``tenant_bucket``: it builds a *real* index snapshot
(embedded Qdrant + SQLite FTS5 + metadata) with the same store classes production
uses, then serves it from a fake S3. So the tenancy -> download -> open -> search
path is exercised end to end without touching AWS.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
from rag_agentic.config import AgenticSettings
from rag_agentic.index_store import TenantIndexStore
from rag_agentic.tenancy import TenantResolver
from rag_search.embedder import FakeEmbedder
from rag_search.metadata_store import MetadataStore
from rag_search.stores import KeywordIndex, LocalVectorIndex

from rag_core.aws import S3ObjectSummary

BUCKET = "exampleapp-indexes"
BASE_URL = "https://dev.example.io/acme"
# Two tenants under one environment, so a cross-tenant leak would be visible.
OTHER_BASE_URL = "https://dev.example.io/globex"


class FakeS3:
    """The slice of S3Client TenantIndexStore uses, backed by a dict."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.gets: list[str] = []

    async def list_objects(
        self, bucket: str, prefix: str = "", *, page_size: int = 1000
    ) -> AsyncIterator[S3ObjectSummary]:
        for key, blob in sorted(self.objects.items()):
            if key.startswith(prefix):
                yield S3ObjectSummary(bucket=bucket, key=key, size_bytes=len(blob), etag="e")

    async def get_bytes(self, bucket: str, key: str, *, version_id: str | None = None) -> bytes:
        self.gets.append(key)
        return self.objects[key]

    async def head_object(self, bucket: str, key: str) -> dict[str, Any]:
        if key not in self.objects:
            raise KeyError(key)
        return {"ETag": f'"{len(self.objects[key])}"'}


# Documents each tenant owns. business_object matters: the SUPPLIERDOCUMENTS /
# COMPLAINTS split is what the pre-filter tests assert on.
ACME_DOCS = [
    {
        "document_id": "d1",
        "source_key": "foxtrot/supplier_quality_manual.pdf",
        "title": "Supplier Quality Manual",
        "facets": {"business_object": "SUPPLIERDOCUMENTS", "status": "Active", "object_code": "SQ-1",
                   "created_by": "riley@example.com"},
        "text": "All suppliers must hold a valid ISO 9001 certificate. "
        "Certificates expire after three years and must be renewed before expiry.",
    },
    {
        "document_id": "d2",
        "source_key": "foxtrot/complaint_handling_policy.pdf",
        "title": "Complaint Handling Policy",
        "facets": {"business_object": "COMPLAINTS", "status": "Active", "object_code": "CP-7",
                   "created_by": "storeuser1"},
        "text": "Every customer complaint must be acknowledged within 48 hours "
        "and closed within 30 calendar days of receipt.",
    },
    {
        "document_id": "d3",
        "source_key": "foxtrot/supplier_audit_schedule.pdf",
        "title": "Supplier Audit Schedule",
        "facets": {"business_object": "SUPPLIERDOCUMENTS", "status": "Draft", "object_code": "SQ-2",
                   "created_by": "riley@example.com"},
        "text": "Critical suppliers are audited annually. Non-critical suppliers "
        "are audited every three years using a remote questionnaire.",
    },
]

GLOBEX_DOCS = [
    {
        "document_id": "g1",
        "source_key": "foxtrot/globex_secret_policy.pdf",
        "title": "Globex Confidential Policy",
        "facets": {"business_object": "SUPPLIERDOCUMENTS", "status": "Active", "object_code": "GX-1"},
        "text": "Globex requires ISO 9001 certification from every raw material supplier.",
    },
]


async def _build_snapshot(root: Path, docs: list[dict[str, Any]]) -> None:
    """Write a real index snapshot (the same artifacts the indexer publishes)."""
    from rag_core.models.metadata import DocumentMetadata

    embedder = FakeEmbedder(dim=256)
    vector = LocalVectorIndex(str(root / "qdrant"), collection="compliance", dim=256)
    keyword = KeywordIndex(str(root / "keyword.db"))
    metadata = MetadataStore(str(root / "metadata.db"))
    await vector.ensure()

    points: list[tuple[str, list[float], dict[str, Any]]] = []
    entries: list[tuple[str, str, dict[str, Any]]] = []
    for doc in docs:
        # One id in both stores: RRF fuses on chunk id, and Qdrant point ids must
        # be UUIDs -- so the indexer's uuid-shaped chunk id is what goes in both.
        chunk_id = _uuid_for(f"{doc['document_id']}-c0")
        payload = {
            "document_id": doc["document_id"],
            "source_key": doc["source_key"],
            "text": doc["text"],
            "page_start": 1,
            "page_end": 1,
            **doc["facets"],
        }
        (vec,) = await embedder.embed([doc["text"]])
        points.append((chunk_id, vec, payload))
        entries.append((chunk_id, doc["text"], payload))
        await metadata.put(
            DocumentMetadata(document_id=doc["document_id"], title=doc["title"]),
            namespace="default",
            source_key=doc["source_key"],
            facets=doc["facets"],
        )

    await vector.upsert(points)
    await keyword.add(entries)
    for store in (vector, keyword, metadata):
        await store.aclose()

    (root / "manifest.json").write_text(
        '{"collection": "compliance", "embed_dim": 256, "namespace": "default"}',
        encoding="utf-8",
    )


def _uuid_for(chunk_id: str) -> str:
    """Qdrant point ids must be UUIDs or ints; mirror how the indexer derives them."""
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _upload(root: Path, objects: dict[str, bytes], prefix: str) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            key = f"{prefix}/{path.relative_to(root).as_posix()}"
            objects[key] = path.read_bytes()


@pytest.fixture
def agentic_settings(tmp_path: Path) -> AgenticSettings:
    return AgenticSettings(
        index_bucket=BUCKET,
        index_prefix="indexes",
        cache_dir=str(tmp_path / "cache"),
        cache_max_tenants=2,
        cache_ttl_s=0.0,
        tenant_pattern=r"^(?P<environment>[^.]+)\.example\.io/(?P<tenant>[^/]+)",
        embed_provider="fake",
        embed_dim=256,
        collection="compliance",
        top_k=5,
    )


@pytest.fixture
async def tenant_bucket(tmp_path: Path) -> FakeS3:
    """A fake S3 holding real snapshots for two tenants."""
    objects: dict[str, bytes] = {}
    for tenant, docs in (("acme", ACME_DOCS), ("globex", GLOBEX_DOCS)):
        root = tmp_path / "build" / tenant
        root.mkdir(parents=True, exist_ok=True)
        await _build_snapshot(root, docs)
        _upload(root, objects, f"indexes/dev/{tenant}")
    return FakeS3(objects)


@pytest.fixture
def base_url() -> str:
    return BASE_URL


@pytest.fixture
def other_base_url() -> str:
    return OTHER_BASE_URL


@pytest.fixture
def empty_bucket() -> FakeS3:
    """A bucket with no snapshot in it — the 'tenant never indexed' case."""
    return FakeS3({})


@pytest.fixture
def resolver(agentic_settings: AgenticSettings) -> TenantResolver:
    return TenantResolver(agentic_settings)


@pytest.fixture
async def index_store(
    agentic_settings: AgenticSettings, tenant_bucket: FakeS3
) -> AsyncIterator[TenantIndexStore]:
    store = TenantIndexStore(agentic_settings, tenant_bucket)
    try:
        yield store
    finally:
        await store.aclose()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=256)


@pytest.fixture
async def acme_index(index_store: TenantIndexStore, resolver: TenantResolver) -> Any:
    """The acme tenant's opened snapshot."""
    return await index_store.acquire(resolver.resolve(BASE_URL))


@pytest.fixture
def make_tools(
    acme_index: Any, embedder: FakeEmbedder, agentic_settings: AgenticSettings
) -> Callable[..., Any]:
    """Build a retrieval toolbox over the acme snapshot, optionally pre-filtered."""
    from rag_agentic.tools import DocumentSearchTools
    from rag_search.config import SearchSettings
    from rag_search.retrieval import SearchService

    def _make(
        business_object: str | None = None,
        tenant_codes: list[str] | None = None,
        object_codes: str | None = None,
    ) -> Any:
        search = SearchService(
            embedder=embedder,
            vector=acme_index.vector,
            keyword=acme_index.keyword,
            settings=SearchSettings(
                collection=agentic_settings.collection,
                candidate_k=agentic_settings.candidate_k,
                rrf_k=agentic_settings.rrf_k,
                top_k=agentic_settings.top_k,
                embed_dim=256,
            ),
        )
        return DocumentSearchTools(
            acme_index,
            search,
            business_object=business_object,
            tenant_codes=tenant_codes,
            object_codes=object_codes,
            top_k=agentic_settings.top_k,
        )

    return _make


@pytest.fixture
def make_request() -> Callable[..., Any]:
    from rag_agentic.models import AgentRequest

    def _make(query: str, *, base_url: str = BASE_URL, business_object: str | None = None) -> Any:
        return AgentRequest(base_url=base_url, query=query, business_object=business_object)

    return _make
