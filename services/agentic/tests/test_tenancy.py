"""Tenancy resolution — the service's security boundary."""

from __future__ import annotations

import pytest
from rag_agentic.config import AgenticSettings
from rag_agentic.tenancy import TenancyError, TenantResolver, normalize_base_url


def _resolver(**overrides: object) -> TenantResolver:
    return TenantResolver(AgenticSettings(**overrides))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("https://Dev.Example.io/acme/", "dev.example.io/acme"),
        ("dev.example.io/acme", "dev.example.io/acme"),
        ("http://dev.example.io:8080/acme", "dev.example.io/acme"),
        ("https://dev.example.io", "dev.example.io"),
    ],
)
def test_normalize_collapses_equivalent_urls(raw: str, expected: str) -> None:
    assert normalize_base_url(raw) == expected


def test_pattern_resolves_environment_and_tenant() -> None:
    resolver = _resolver(
        tenant_pattern=r"^(?P<environment>[^.]+)\.example\.io/(?P<tenant>[^/]+)"
    )
    context = resolver.resolve("https://dev.example.io/acme")
    assert (context.environment, context.tenant) == ("dev", "acme")
    assert context.key == "dev/acme"
    assert context.s3_prefix("indexes") == "indexes/dev/acme"


def test_host_style_pattern_also_works() -> None:
    resolver = _resolver(
        tenant_pattern=r"^(?P<tenant>[^.]+)\.(?P<environment>[^.]+)\.example\.io"
    )
    context = resolver.resolve("https://acme.prod.example.io")
    assert (context.environment, context.tenant) == ("prod", "acme")


def test_explicit_map_wins_over_pattern() -> None:
    resolver = _resolver(
        tenant_map={"https://dev.example.io/acme": "staging/acme-corp"},
        tenant_pattern=r"^(?P<environment>[^.]+)\.example\.io/(?P<tenant>[^/]+)",
    )
    context = resolver.resolve("https://dev.example.io/acme/")
    assert context.key == "staging/acme-corp"


def test_unknown_base_url_raises_rather_than_guessing() -> None:
    """The dangerous failure would be silently serving a default tenant."""
    resolver = _resolver(
        tenant_pattern=r"^(?P<environment>[^.]+)\.example\.io/(?P<tenant>[^/]+)"
    )
    with pytest.raises(TenancyError):
        resolver.resolve("https://evil.example.com/acme")


def test_defaults_apply_only_when_both_are_set() -> None:
    partial = _resolver(default_environment="dev")
    with pytest.raises(TenancyError):
        partial.resolve("https://unmatched.example.com")

    full = _resolver(default_environment="dev", default_tenant="acme")
    assert full.resolve("https://unmatched.example.com").key == "dev/acme"


def test_empty_base_url_is_rejected() -> None:
    with pytest.raises(TenancyError):
        _resolver(default_environment="dev", default_tenant="acme").resolve("")


def test_path_traversal_in_tenant_is_rejected() -> None:
    """A tenant name becomes an S3 path segment, so '..' must never resolve."""
    resolver = _resolver(tenant_map={"https://x.example.com": "dev/../../secrets"})
    with pytest.raises(TenancyError):
        resolver.resolve("https://x.example.com")


def test_map_entry_without_environment_is_rejected() -> None:
    resolver = _resolver(tenant_map={"https://x.example.com": "acme"})
    with pytest.raises(TenancyError):
        resolver.resolve("https://x.example.com")


def test_pattern_missing_named_groups_is_rejected() -> None:
    resolver = _resolver(tenant_pattern=r"^(?P<tenant>[^.]+)\.example\.com")
    with pytest.raises(TenancyError):
        resolver.resolve("https://acme.example.com")
