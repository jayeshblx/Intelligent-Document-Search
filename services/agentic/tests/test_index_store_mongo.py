"""Live MongoDB tenants: routing, isolation guards, manifest refresh and warm-up.

The Atlas collection itself is replaced by a fake -- the stores' behaviour against
real Atlas is covered by ``services/search/tests/test_stores_mongo.py``. What is
tested here is the agent's side: which tenants go to Mongo, that nothing touches
S3 for them, and that an empty or unreachable collection is a loud 503.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from pydantic import ValidationError
from rag_agentic import index_store as index_store_module
from rag_agentic.config import AgenticSettings
from rag_agentic.index_store import IndexUnavailableError, TenantIndexStore
from rag_agentic.tenancy import TenantContext
from rag_search.stores import MongoKeywordIndex, MongoMetadataStore, MongoVectorIndex

FOXTROT = TenantContext(environment="foxtrot-Dev", tenant="foxtrot", base_url="https://foxtrot.example")
BRAVO = TenantContext(environment="bravo-Dev", tenant="bravo", base_url="https://bravo.example")
OTHER = TenantContext(environment="dev", tenant="acme", base_url="https://acme.example")


class FakeMongoDB:
    """Stands in for ``MongoIndexDatabase``; one per opened tenant."""

    instances: ClassVar[list[FakeMongoDB]] = []
    manifests: ClassVar[dict[str, dict[str, Any]]] = {}
    chunks: ClassVar[dict[str, int]] = {}
    clients: ClassVar[list[str]] = []

    def __init__(self, uri: str = "", **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.collection = kwargs["collection"]
        self.exact_vector_search = kwargs.get("exact_vector_search", False)
        self.query_timeout_ms = kwargs.get("query_timeout_ms", 10_000)
        self.queries = 0
        self.closed = False
        FakeMongoDB.instances.append(self)

    async def read_manifest(self) -> dict[str, Any]:
        return dict(self.manifests.get(self.collection, {}))

    async def has_chunks(self) -> bool:
        if self.collection not in self.chunks:
            raise ConnectionError("collection unreachable")
        return self.chunks[self.collection] > 0

    async def term_stats(self, tokens: list[str]) -> tuple[int, dict[str, int]]:
        self.queries += 1
        return 100, dict.fromkeys(tokens, 1)

    async def run(self, fn: Any) -> list[Any]:
        self.queries += 1
        return []

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def fake_mongo(monkeypatch: pytest.MonkeyPatch) -> type[FakeMongoDB]:
    FakeMongoDB.instances = []
    FakeMongoDB.manifests = {
        "FOXTROT_DocumentSearch_DEV": {"tenant_codes": ["FOXTROT"], "embed_dim": 4, "built_at": "v1"},
        "BRAVO_DocumentSearch_DEV": {"tenant_codes": ["BRAVO"], "embed_dim": 4, "built_at": "v1"},
    }
    FakeMongoDB.chunks = {"FOXTROT_DocumentSearch_DEV": 10, "BRAVO_DocumentSearch_DEV": 5}
    FakeMongoDB.clients = []
    monkeypatch.setattr(index_store_module, "MongoIndexDatabase", FakeMongoDB)
    monkeypatch.setattr(
        index_store_module,
        "open_client",
        lambda uri, **_kw: FakeMongoDB.clients.append(uri) or FakeClient(),
    )
    return FakeMongoDB


class FakeClient:
    def close(self) -> None:
        pass


class NoS3:
    """Any S3 call fails the test: Mongo tenants must never touch the bucket."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"S3 was called ({name}) for a Mongo-backed tenant")


def _settings(**overrides: Any) -> AgenticSettings:
    values: dict[str, Any] = {
        "index_bucket": "",
        "embed_dim": 4,
        "cache_ttl_s": 0,
        "mongo_uri": "mongodb+srv://user:pw@cluster.example/",
        "mongo_collections": {
            "foxtrot-Dev/foxtrot": "FOXTROT_DocumentSearch_DEV",
            "bravo-Dev/bravo": "BRAVO_DocumentSearch_DEV",
        },
    }
    values.update(overrides)
    return AgenticSettings(_env_file=None, **values)


# ---- configuration ----------------------------------------------------------


def test_collections_need_a_connection_string() -> None:
    with pytest.raises(ValidationError, match="AGENTIC_MONGO_URI"):
        _settings(mongo_uri="")


def test_collection_keys_must_be_environment_slash_tenant() -> None:
    with pytest.raises(ValidationError, match="environment/tenant"):
        _settings(mongo_collections={"foxtrot": "FOXTROT_DocumentSearch_DEV"})


def test_two_tenants_cannot_share_a_collection() -> None:
    with pytest.raises(ValidationError, match="both map to collection"):
        _settings(mongo_collections={"foxtrot-Dev/foxtrot": "Shared", "bravo-Dev/bravo": "Shared"})


# ---- routing ----------------------------------------------------------------


async def test_mapped_tenant_opens_its_collection_without_s3(fake_mongo: Any) -> None:
    store = TenantIndexStore(_settings(), NoS3())
    index = await store.acquire(FOXTROT)

    assert isinstance(index.vector, MongoVectorIndex)
    assert isinstance(index.keyword, MongoKeywordIndex)
    assert isinstance(index.metadata, MongoMetadataStore)
    assert index.manifest["tenant_codes"] == ["FOXTROT"]
    assert fake_mongo.instances[-1].kwargs["collection"] == "FOXTROT_DocumentSearch_DEV"
    assert fake_mongo.instances[-1].kwargs["database"] == "EX_DOCUMENTSEARCH_DEV"
    assert not index.root.exists()  # nothing was downloaded
    await store.aclose()


async def test_tenants_get_their_own_collections_over_one_client(fake_mongo: Any) -> None:
    store = TenantIndexStore(_settings(), NoS3())
    foxtrot, bravo = await store.acquire(FOXTROT), await store.acquire(BRAVO)

    assert foxtrot.manifest["tenant_codes"] == ["FOXTROT"]
    assert bravo.manifest["tenant_codes"] == ["BRAVO"]
    assert {db.kwargs["collection"] for db in fake_mongo.instances} == {
        "FOXTROT_DocumentSearch_DEV",
        "BRAVO_DocumentSearch_DEV",
    }
    assert len(fake_mongo.clients) == 1  # one connection pool for every tenant
    assert await store.acquire(FOXTROT) is foxtrot  # cached
    await store.aclose()


async def test_unmapped_tenant_still_uses_the_snapshot_path(fake_mongo: Any) -> None:
    store = TenantIndexStore(_settings(), NoS3())
    with pytest.raises(IndexUnavailableError, match="AGENTIC_INDEX_BUCKET"):
        await store.acquire(OTHER)


async def test_empty_collection_is_unavailable(fake_mongo: Any) -> None:
    fake_mongo.chunks["FOXTROT_DocumentSearch_DEV"] = 0
    store = TenantIndexStore(_settings(), NoS3())
    with pytest.raises(IndexUnavailableError, match="has not been indexed yet"):
        await store.acquire(FOXTROT)


async def test_unreachable_collection_is_unavailable(fake_mongo: Any) -> None:
    del fake_mongo.chunks["FOXTROT_DocumentSearch_DEV"]
    store = TenantIndexStore(_settings(), NoS3())
    with pytest.raises(IndexUnavailableError, match="no reachable index collection"):
        await store.acquire(FOXTROT)


async def test_dimension_mismatch_is_unavailable(fake_mongo: Any) -> None:
    fake_mongo.manifests["FOXTROT_DocumentSearch_DEV"]["embed_dim"] = 1024
    store = TenantIndexStore(_settings(), NoS3())
    with pytest.raises(IndexUnavailableError, match="1024-dimensional"):
        await store.acquire(FOXTROT)


async def test_manifest_is_refreshed_in_place_not_reloaded(fake_mongo: Any) -> None:
    store = TenantIndexStore(_settings(cache_ttl_s=0.001), NoS3())
    first = await store.acquire(FOXTROT)
    opened = len(fake_mongo.instances)

    fake_mongo.manifests["FOXTROT_DocumentSearch_DEV"] = {
        "tenant_codes": ["FOXTROT", "ACMEFOODS"],
        "embed_dim": 4,
        "built_at": "v2",
    }
    first.checked_at -= 1  # past the TTL
    again = await store.acquire(FOXTROT)

    assert again is first  # same live index: no reload, no cold start
    assert len(fake_mongo.instances) == opened
    assert again.manifest["tenant_codes"] == ["FOXTROT", "ACMEFOODS"]
    assert again.version == "v2"
    await store.aclose()


async def test_warm_opens_and_queries_every_mapped_tenant(fake_mongo: Any) -> None:
    store = TenantIndexStore(_settings(), NoS3())
    await store.warm()

    queried = {db.kwargs["collection"]: db.queries for db in fake_mongo.instances}
    assert set(queried) == {"FOXTROT_DocumentSearch_DEV", "BRAVO_DocumentSearch_DEV"}
    assert all(n >= 2 for n in queried.values())  # one vector + one keyword query each
    await store.aclose()


async def test_warm_never_raises(fake_mongo: Any) -> None:
    del fake_mongo.chunks["BRAVO_DocumentSearch_DEV"]  # BRAVO unreachable
    store = TenantIndexStore(_settings(), NoS3())
    await store.warm()  # logged, not raised
    assert (await store.acquire(FOXTROT)).manifest["tenant_codes"] == ["FOXTROT"]
    await store.aclose()
