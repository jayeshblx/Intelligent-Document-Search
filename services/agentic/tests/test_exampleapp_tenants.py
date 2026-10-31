"""ExampleApp's real base URLs -> (environment, tenant), and nothing else.

The URL shape is ``https://<tenant>.<environment-domain>.com``. The recommended
pattern anchors the whole host and allow-lists the environment domains, so a
lookalike host can never borrow a real tenant's identity. An unanchored or
wildcard pattern would accept ``corp.portaltest.com.attacker.net`` as tenant corp.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from rag_agentic.config import AgenticSettings
from rag_agentic.index_store import TenantIndexStore
from rag_agentic.tenancy import TenancyError, TenantContext, TenantResolver

# The value to deploy as AGENTIC_TENANT_PATTERN.
EXAMPLEAPP_TENANT_PATTERN = (
    r"^(?P<tenant>[a-z0-9][a-z0-9-]*)"
    r"\.(?P<environment>acmedev|acmeqa|portaltest|portalprocess|portalprod)\.com$"
)


@pytest.fixture
def resolver() -> TenantResolver:
    return TenantResolver(AgenticSettings(tenant_pattern=EXAMPLEAPP_TENANT_PATTERN))


@pytest.mark.parametrize(
    ("base_url", "environment", "tenant", "prefix"),
    [
        ("https://foxtrot.acmedev.com/", "acmedev", "foxtrot", "DocumentSearch/acmedev/foxtrot"),
        ("https://globex.acmeqa.com/", "acmeqa", "globex", "DocumentSearch/acmeqa/globex"),
        ("https://corp.portaltest.com/", "portaltest", "corp", "DocumentSearch/portaltest/corp"),
        ("https://corp.portalprocess.com/", "portalprocess", "corp", "DocumentSearch/portalprocess/corp"),
        ("https://corp.portalprod.com/", "portalprod", "corp", "DocumentSearch/portalprod/corp"),
    ],
)
def test_real_exampleapp_urls_resolve(
    resolver: TenantResolver, base_url: str, environment: str, tenant: str, prefix: str
) -> None:
    context = resolver.resolve(base_url)
    assert (context.environment, context.tenant) == (environment, tenant)
    # Each (environment, tenant) reads its own S3 folder -- corp's three environments
    # never share an index.
    assert context.s3_prefix("DocumentSearch") == prefix


@pytest.mark.parametrize(
    "base_url",
    [
        "HTTPS://CORP.PortalTest.com",  # case and a missing trailing slash
        "https://corp.portaltest.com:443/",  # explicit port
        "corp.portaltest.com",  # no scheme
    ],
)
def test_equivalent_spellings_resolve_the_same(resolver: TenantResolver, base_url: str) -> None:
    assert resolver.resolve(base_url).key == "portaltest/corp"


@pytest.mark.parametrize(
    "base_url",
    [
        "https://corp.portaltest.com.attacker.net/",  # real host as a prefix of another
        "https://corp.portaltest.co/",  # different TLD
        "https://foxtrot.acmedevx.com/",  # lookalike of the dev domain
        "https://corp.portaltestx.com/",  # lookalike environment domain
        "https://corp.evil.com/",  # unknown environment
        "https://portaltest.com/",  # no tenant
        "https://a.corp.portaltest.com/",  # extra subdomain level
        "https://corp.portaltest.com/other/app",  # a path: not a base URL
    ],
)
def test_lookalikes_are_rejected_not_guessed(resolver: TenantResolver, base_url: str) -> None:
    with pytest.raises(TenancyError):
        resolver.resolve(base_url)


# --------------------------------------------------------------------------
# The flat-prefix override belongs to exactly one tenant
# --------------------------------------------------------------------------


def test_override_without_a_bound_tenant_fails_at_startup() -> None:
    with pytest.raises(ValidationError, match="AGENTIC_INDEX_PREFIX_OVERRIDE_TENANT"):
        AgenticSettings(index_prefix_override="DocumentSearch/foxtrot-Dev")


async def test_override_refuses_every_other_tenant() -> None:
    """With one flat folder configured, Globex's must not be served FOXTROT's documents."""

    class NoS3:
        async def get_bytes(self, *a: object, **k: object) -> bytes:
            raise AssertionError("must be refused before touching S3")

        async def head_object(self, *a: object, **k: object) -> dict[str, object]:
            raise AssertionError("must be refused before touching S3")

        def list_objects(self, *a: object, **k: object) -> object:
            raise AssertionError("must be refused before touching S3")

    settings = AgenticSettings(
        index_bucket="example-index-bucket",
        index_prefix_override="DocumentSearch/foxtrot-Dev",
        index_prefix_override_tenant="foxtrot-Dev/foxtrot",
        tenant_pattern=EXAMPLEAPP_TENANT_PATTERN,
    )
    store = TenantIndexStore(settings, NoS3())  # type: ignore[arg-type]
    globex = TenantResolver(settings).resolve("https://globex.acmeqa.com/")

    with pytest.raises(TenancyError, match="serves only 'foxtrot-Dev/foxtrot'"):
        await store.acquire(globex)


def test_override_still_serves_its_own_tenant() -> None:
    settings = AgenticSettings(
        index_prefix_override="DocumentSearch/foxtrot-Dev", index_prefix_override_tenant="foxtrot-Dev/foxtrot"
    )
    store = TenantIndexStore(settings, object())  # type: ignore[arg-type]
    own = TenantContext(environment="foxtrot-Dev", tenant="foxtrot", base_url="x")
    assert store._prefix_for(own) == "DocumentSearch/foxtrot-Dev"
