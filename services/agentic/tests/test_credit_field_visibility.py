"""The log says whether the credit fields arrived, never what they held.

Integrating two systems across a network, the first question is always "did you
actually send it?". Without this, answering that means either trusting the caller
or logging a live credential. So the log records presence only.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import structlog
from fastapi.testclient import TestClient
from rag_agentic.config import AgenticSettings, AgentRole
from rag_agentic.factory import AgenticComponents
from rag_agentic.models import AgentResponse
from rag_agentic.runtime.app import create_app

TOKEN = "Bearer eyJveryverysecret"


class _Agent:
    async def handle(self, request: Any) -> AgentResponse:
        return AgentResponse(answer="ok", route="document_search",
                             environment="acmedev", tenant="foxtrot")


def _run(body: dict[str, Any], resolver: Any) -> list[dict[str, Any]]:
    """POST once and return the structlog events it emitted."""
    captured: list[dict[str, Any]] = []

    def sink(_logger: Any, _name: str,
             event: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        captured.append(dict(event))
        return event

    settings = AgenticSettings()
    components = AgenticComponents(
        settings=settings, role=AgentRole.ORCHESTRATOR, agent=_Agent(),
        resolver=resolver, model=None,
    )
    structlog.configure(processors=[sink, structlog.processors.JSONRenderer()])
    try:
        with TestClient(create_app(AgentRole.ORCHESTRATOR, components=components)) as client:
            client.post("/invocations", json=body)
    finally:
        structlog.reset_defaults()
    return captured


BASE = {"BaseUrl": "https://foxtrot.acmedev.com/", "QueryText": "how many documents?"}


def _received(events: list[dict[str, Any]]) -> dict[str, Any]:
    return next(e for e in events if e.get("event") == "request_received")


def test_a_token_is_reported_as_present(resolver: Any) -> None:
    received = _received(_run({**BASE, "AccessToken": TOKEN}, resolver))

    assert received["has_access_token"] is True


def test_a_missing_token_is_reported_as_absent(resolver: Any) -> None:
    received = _received(_run(BASE, resolver))

    assert received["has_access_token"] is False


def test_the_reservation_id_is_reported_the_same_way(resolver: Any) -> None:
    with_it = _received(_run({**BASE, "ReservationId": "res-1"}, resolver))
    without = _received(_run(BASE, resolver))

    assert with_it["has_reservation_id"] is True
    assert without["has_reservation_id"] is False


def test_a_blank_reservation_id_counts_as_absent(resolver: Any) -> None:
    """A caller sending "" has not sent one, and should not look like they did."""
    received = _received(_run({**BASE, "ReservationId": "   "}, resolver))

    assert received["has_reservation_id"] is False


def test_the_token_value_never_appears_in_any_log_line(resolver: Any) -> None:
    """The whole point. A token in CloudWatch is a usable credential."""
    events = _run({**BASE, "AccessToken": TOKEN, "ReservationId": "res-1"}, resolver)

    blob = repr(events)
    assert "eyJveryverysecret" not in blob
    assert TOKEN not in blob
