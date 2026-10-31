"""One user request settles its reservation exactly once.

The orchestrator forwards the request to its specialist verbatim, ReservationId
included. If both runtimes metered, the same reservation would be settled twice
and one user question would be billed as two. Only the entry point meters, and
that is enforced in code -- not by remembering which runtime gets the environment
variable, which is the kind of thing that is correct until someone copies a config.
"""

from __future__ import annotations

from typing import Any

import pytest
from rag_agentic.config import AgenticSettings, AgentRole
from rag_agentic.factory import _build_credit_gate


def _settings(role: AgentRole) -> AgenticSettings:
    return AgenticSettings(
        role=role, licensing_enabled=True, licensing_settle_only=True,
        licensing_base_url="https://tenantadmin.acmedev.com",
    )


def test_the_orchestrator_meters() -> None:
    gate = _build_credit_gate(_settings(AgentRole.ORCHESTRATOR), [])

    assert gate.active, "the entry point must meter"


def test_the_document_search_agent_does_not() -> None:
    """Identical configuration, different role: still inert."""
    gate = _build_credit_gate(_settings(AgentRole.DOCUMENT_SEARCH), [])

    assert not gate.active, "a specialist that settles double-bills the request"


@pytest.mark.parametrize("role", list(AgentRole))
def test_no_role_meters_while_licensing_is_off(role: AgentRole) -> None:
    gate = _build_credit_gate(AgenticSettings(role=role, licensing_enabled=False), [])

    assert not gate.active


def test_an_enabled_specialist_opens_no_connection() -> None:
    """Not merely inert: it must not even build a client, or it holds a pool and a
    credential it has no use for."""
    closeables: list[Any] = []

    _build_credit_gate(_settings(AgentRole.DOCUMENT_SEARCH), closeables)

    assert closeables == []
