"""The AgentCore Runtime HTTP contract: /ping, /invocations, session header, errors."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from rag_agentic.agents.search_agent import DocumentSearchAgent
from rag_agentic.config import AgenticSettings, AgentRole
from rag_agentic.converse import ScriptedConverseModel, text_turn, tool_turn
from rag_agentic.factory import AgenticComponents
from rag_agentic.index_store import TenantIndexStore
from rag_agentic.runtime.app import SESSION_HEADER, create_app
from rag_agentic.tenancy import TenantResolver


@pytest.fixture
def client(
    agentic_settings: AgenticSettings,
    embedder: Any,
    index_store: TenantIndexStore,
    resolver: TenantResolver,
) -> Any:
    model = ScriptedConverseModel(
        [
            tool_turn("search_documents", {"query": "ISO 9001"}),
            text_turn("Suppliers must hold a valid ISO 9001 certificate."),
        ]
    )
    agent = DocumentSearchAgent(
        settings=agentic_settings,
        model=model,
        embedder=embedder,
        index_store=index_store,
        resolver=resolver,
    )
    components = AgenticComponents(
        settings=agentic_settings,
        role=AgentRole.DOCUMENT_SEARCH,
        agent=agent,
        resolver=resolver,
        model=model,
    )
    with TestClient(create_app(AgentRole.DOCUMENT_SEARCH, components=components)) as test_client:
        yield test_client


def test_ping_reports_healthy(client: Any) -> None:
    """AgentCore marks the runtime unhealthy on anything but this exact shape."""
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "Healthy"}


def test_invocation_answers_and_echoes_the_session(client: Any, base_url: str) -> None:
    session = "s" * 40
    response = client.post(
        "/invocations",
        json={"base_url": base_url, "query": "Do suppliers need ISO 9001?"},
        headers={SESSION_HEADER: session},
    )

    assert response.status_code == 200
    body = response.json()
    assert "ISO 9001" in body["answer"]
    assert body["route"] == "document_search"
    assert body["tenant"] == "acme"
    assert body["session_id"] == session
    assert response.headers[SESSION_HEADER] == session
    assert body["citations"]


def test_missing_session_header_is_generated(client: Any, base_url: str) -> None:
    response = client.post("/invocations", json={"base_url": base_url, "query": "iso?"})

    assert response.status_code == 200
    assert len(response.json()["session_id"]) >= 33


def test_unknown_tenant_is_a_400_not_a_500(client: Any) -> None:
    response = client.post(
        "/invocations", json={"base_url": "https://evil.example.com", "query": "iso?"}
    )

    assert response.status_code == 400
    assert "no tenant configured" in response.json()["error"]


def test_missing_required_field_is_rejected(client: Any) -> None:
    response = client.post("/invocations", json={"query": "iso?"})

    assert response.status_code == 400
    assert response.json()["error"] == "invalid request payload"


def test_non_object_body_is_rejected(client: Any) -> None:
    response = client.post("/invocations", json=["not", "an", "object"])

    assert response.status_code == 400
