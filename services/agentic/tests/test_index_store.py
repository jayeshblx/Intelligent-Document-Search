"""Per-tenant snapshot download, caching and isolation."""

from __future__ import annotations

from typing import Any

import pytest
from rag_agentic.config import AgenticSettings
from rag_agentic.index_store import IndexUnavailableError, TenantIndexStore
from rag_agentic.tenancy import TenantResolver


async def test_downloads_and_opens_a_real_snapshot(
    index_store: TenantIndexStore, resolver: TenantResolver, base_url: str
) -> None:
    index = await index_store.acquire(resolver.resolve(base_url))

    assert index.context.key == "dev/acme"
    assert index.namespace == "default"
    assert (index.root / "keyword.db").exists()
    assert (index.root / "qdrant").is_dir()
    assert await index.keyword.count() == 3


async def test_second_acquire_is_served_from_cache(
    index_store: TenantIndexStore,
    resolver: TenantResolver,
    tenant_bucket: Any,
    base_url: str,
) -> None:
    context = resolver.resolve(base_url)
    first = await index_store.acquire(context)
    downloads = len(tenant_bucket.gets)

    second = await index_store.acquire(context)

    assert second is first
    # cache_ttl_s=0 pins the snapshot, so a warm hit costs no S3 GETs at all.
    assert len(tenant_bucket.gets) == downloads


async def test_tenants_are_isolated(
    index_store: TenantIndexStore,
    resolver: TenantResolver,
    base_url: str,
    other_base_url: str,
) -> None:
    """The same store must not let one tenant's documents appear under another."""
    acme = await index_store.acquire(resolver.resolve(base_url))
    globex = await index_store.acquire(resolver.resolve(other_base_url))

    acme_docs = {doc_id for doc_id, _ in await acme.metadata.all_documents()}
    globex_docs = {doc_id for doc_id, _ in await globex.metadata.all_documents()}

    assert acme_docs == {"d1", "d2", "d3"}
    assert globex_docs == {"g1"}
    assert not acme_docs & globex_docs


async def test_lru_evicts_beyond_capacity(
    agentic_settings: AgenticSettings,
    tenant_bucket: Any,
    base_url: str,
    other_base_url: str,
) -> None:
    settings = agentic_settings.model_copy(update={"cache_max_tenants": 1})
    resolver = TenantResolver(settings)
    store = TenantIndexStore(settings, tenant_bucket)
    try:
        first = await store.acquire(resolver.resolve(base_url))
        root = first.root
        await store.acquire(resolver.resolve(other_base_url))
        # Evicting must reclaim the disk too — /tmp in a runtime container is small.
        assert not root.exists()
    finally:
        await store.aclose()


async def test_missing_snapshot_raises_index_unavailable(
    agentic_settings: AgenticSettings, empty_bucket: Any, base_url: str
) -> None:
    store = TenantIndexStore(agentic_settings, empty_bucket)
    resolver = TenantResolver(agentic_settings)
    try:
        with pytest.raises(IndexUnavailableError):
            await store.acquire(resolver.resolve(base_url))
    finally:
        await store.aclose()


async def test_unset_bucket_is_a_clear_configuration_error(
    agentic_settings: AgenticSettings, tenant_bucket: Any, base_url: str
) -> None:
    settings = agentic_settings.model_copy(update={"index_bucket": ""})
    store = TenantIndexStore(settings, tenant_bucket)
    with pytest.raises(IndexUnavailableError, match="AGENTIC_INDEX_BUCKET"):
        await store.acquire(TenantResolver(settings).resolve(base_url))
