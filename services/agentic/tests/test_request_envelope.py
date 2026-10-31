"""The request envelope ExampleApp sends, and what comes back.

Two rules this pins:

* ``UserCode`` is carried end to end and echoed, so an answer can be tied to the
  user who asked -- but it does NOT change what is searched.
* ``top_k`` is deployment configuration (``AGENTIC_TOP_K``), not a per-request
  field: a caller cannot widen or narrow retrieval.
"""

from __future__ import annotations

from typing import Any

import pytest
from rag_agentic.agents.orchestrator import OrchestratorAgent
from rag_agentic.agents.search_agent import DocumentSearchAgent
from rag_agentic.config import AgenticSettings
from rag_agentic.converse import ScriptedConverseModel, text_turn, tool_turn
from rag_agentic.index_store import TenantIndexStore
from rag_agentic.models import AgentRequest
from rag_agentic.search_client import InProcessSearchClient
from rag_agentic.tenancy import TenantResolver


def test_exampleapp_pascal_case_fields_are_accepted() -> None:
    request = AgentRequest.model_validate(
        {
            "base_url": "https://foxtrot.acmedev.com/",
            "Query": "What is complaint RA2865 about?",
            "UserCode": "USER-4471",
            "BusinessObject": "COMPLAINT",
        }
    )
    assert request.user_code == "USER-4471"
    assert request.business_object == "COMPLAINT"
    assert request.query == "What is complaint RA2865 about?"


def test_snake_case_still_works_and_user_code_is_optional() -> None:
    request = AgentRequest.model_validate(
        {"base_url": "https://foxtrot.acmedev.com/", "query": "q", "user_code": "u1"}
    )
    assert request.user_code == "u1"
    assert AgentRequest(base_url="https://foxtrot.acmedev.com/", query="q").user_code is None


def test_top_k_is_not_a_request_field() -> None:
    """Retrieval depth is configuration; a caller sending it changes nothing."""
    request = AgentRequest.model_validate(
        {"base_url": "https://foxtrot.acmedev.com/", "query": "q", "top_k": 50}
    )
    assert not hasattr(request, "top_k")


@pytest.mark.parametrize("configured", [3, 7])
async def test_retrieval_depth_comes_from_settings(
    configured: int,
    agentic_settings: AgenticSettings,
    embedder: Any,
    index_store: TenantIndexStore,
    resolver: TenantResolver,
    base_url: str,
) -> None:
    settings = agentic_settings.model_copy(update={"top_k": configured})
    model = ScriptedConverseModel([text_turn("ok")])
    agent = DocumentSearchAgent(
        settings=settings, model=model, embedder=embedder, index_store=index_store, resolver=resolver
    )
    # The toolbox the agent builds is what carries the depth into every search.
    captured: dict[str, Any] = {}
    import rag_agentic.agents.search_agent as module

    original = module.DocumentSearchTools

    def spy(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return original(*args, **kwargs)

    module.DocumentSearchTools = spy  # type: ignore[misc]
    try:
        await agent.handle(
            AgentRequest.model_validate(
                {"base_url": base_url, "query": "q", "top_k": 50}  # ignored on purpose
            )
        )
    finally:
        module.DocumentSearchTools = original  # type: ignore[misc]

    assert captured["top_k"] == configured


async def test_user_code_is_echoed_and_reaches_the_search_agent(
    agentic_settings: AgenticSettings,
    embedder: Any,
    index_store: TenantIndexStore,
    resolver: TenantResolver,
    base_url: str,
) -> None:
    search_model = ScriptedConverseModel(
        [tool_turn("search_documents", {"query": "iso"}), text_turn("Suppliers need ISO 9001.")]
    )
    search = DocumentSearchAgent(
        settings=agentic_settings,
        model=search_model,
        embedder=embedder,
        index_store=index_store,
        resolver=resolver,
    )
    client = InProcessSearchClient(search)
    router = ScriptedConverseModel([tool_turn("search_documents", {"query": "iso"})])
    orchestrator = OrchestratorAgent(
        settings=agentic_settings,
        model=router,
        search_client=client,
        resolve_tenant=resolver.resolve,
    )

    response = await orchestrator.handle(
        AgentRequest.model_validate(
            {"base_url": base_url, "query": "Do suppliers need ISO 9001?", "UserCode": "USER-4471"}
        )
    )

    assert response.user_code == "USER-4471"
    assert response.route == "document_search"
    # It identifies the caller; it must not silence results.
    assert response.citations


async def test_user_code_does_not_change_the_answer(
    agentic_settings: AgenticSettings,
    embedder: Any,
    index_store: TenantIndexStore,
    resolver: TenantResolver,
    base_url: str,
) -> None:
    """Same question, different users: the documents found are the same. Per-user
    document permissions are not implemented -- tenancy is the only boundary."""

    async def ask(user_code: str | None) -> set[str]:
        model = ScriptedConverseModel(
            [tool_turn("search_documents", {"query": "iso 9001"}), text_turn("answer")]
        )
        agent = DocumentSearchAgent(
            settings=agentic_settings,
            model=model,
            embedder=embedder,
            index_store=index_store,
            resolver=resolver,
        )
        payload = {"base_url": base_url, "query": "Do suppliers need ISO 9001?"}
        if user_code:
            payload["UserCode"] = user_code
        response = await agent.handle(AgentRequest.model_validate(payload))
        return {c.document_id for c in response.citations}

    assert await ask("USER-1") == await ask("USER-2") == await ask(None)
