"""One database per tenant — the isolation property of the DSN backend.

The ``chunks`` table carries no tenant column and the vector/keyword stores apply
no tenant predicate, so isolation here is *physical*: a tenant's documents are in
its own database or they are not isolated at all. These tests pin that, because a
regression would return one customer's compliance documents to another.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from rag_agentic.config import AgenticSettings, PostgresSource
from rag_agentic.index_store import IndexUnavailableError, TenantIndexStore
from rag_agentic.tenancy import TenancyError, TenantContext

TEMPLATE = "postgresql://u:p@db.internal:5432/{environment}_{tenant}"


def _store(**overrides: object) -> TenantIndexStore:
    settings = AgenticSettings(
        index_bucket="b",
        postgres_source=PostgresSource.DSN,
        **overrides,  # type: ignore[arg-type]
    )
    return TenantIndexStore(settings, s3=object())  # type: ignore[arg-type]


def _ctx(environment: str, tenant: str) -> TenantContext:
    return TenantContext(
        environment=environment, tenant=tenant, base_url=f"https://{tenant}.{environment}.com/"
    )


def test_each_tenant_gets_its_own_database() -> None:
    store = _store(postgres_dsn_template=TEMPLATE)

    assert store._dsn_for(_ctx("acmedev", "foxtrot")).endswith("/acmedev_foxtrot")
    assert store._dsn_for(_ctx("acmedev", "bravo")).endswith("/acmedev_bravo")


def test_the_same_tenant_in_two_environments_is_two_databases() -> None:
    """foxtrot on dev and foxtrot on QA are different customers' data."""
    store = _store(postgres_dsn_template=TEMPLATE)

    assert store._dsn_for(_ctx("acmedev", "foxtrot")) != store._dsn_for(_ctx("acmeqa", "foxtrot"))


def test_a_template_without_tenant_is_refused_at_startup() -> None:
    """Every tenant would share one database, which is the leak this prevents."""
    with pytest.raises(ValidationError, match=r"\{tenant\}"):
        AgenticSettings(
            index_bucket="b",
            postgres_source=PostgresSource.DSN,
            postgres_dsn_template="postgresql://u:p@db.internal:5432/shared",
        )


def test_dsn_and_template_together_are_refused() -> None:
    with pytest.raises(ValidationError, match="not both"):
        AgenticSettings(
            index_bucket="b",
            postgres_source=PostgresSource.DSN,
            postgres_dsn="postgresql://u:p@db/one",
            postgres_dsn_template=TEMPLATE,
        )


def test_a_single_dsn_serves_its_first_tenant() -> None:
    store = _store(postgres_dsn="postgresql://u:p@db.internal:5432/only")

    assert store._dsn_for(_ctx("acmedev", "foxtrot")).endswith("/only")
    # Same tenant again is fine - it is the same database.
    assert store._dsn_for(_ctx("acmedev", "foxtrot")).endswith("/only")


def test_a_single_dsn_refuses_a_second_tenant() -> None:
    """The last line of defence: one database, one tenant, or nothing."""
    store = _store(postgres_dsn="postgresql://u:p@db.internal:5432/only")
    store._dsn_for(_ctx("acmedev", "foxtrot"))

    with pytest.raises(TenancyError, match="cannot share it"):
        store._dsn_for(_ctx("acmedev", "bravo"))


def test_missing_configuration_is_reported_clearly() -> None:
    store = _store()

    with pytest.raises(IndexUnavailableError, match="AGENTIC_POSTGRES_DSN"):
        store._dsn_for(_ctx("acmedev", "foxtrot"))
