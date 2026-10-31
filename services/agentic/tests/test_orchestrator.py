"""Orchestrator: routing, faithful delegation, and merged results."""

from __future__ import annotations

from typing import Any

import pytest
from rag_agentic.agents.orchestrator import OrchestratorAgent
from rag_agentic.config import AgenticSettings
from rag_agentic.converse import ScriptedConverseModel, text_turn, tool_turn
from rag_agentic.models import AgentRequest, AgentResponse, Citation, ToolCall, Usage
from rag_agentic.search_client import ensure_session_id
from rag_agentic.tenancy import TenantResolver


class RecordingSearchClient:
    """Stands in for the document-search runtime; records what it was asked."""

    def __init__(self, response: AgentResponse | None = None) -> None:
        self.requests: list[AgentRequest] = []
        self._response = response or AgentResponse(
            answer="Suppliers must hold ISO 9001.",
            citations=[Citation(document_id="d1", document_name="Supplier Quality Manual", score=0.9)],
            route="document_search",
            environment="dev",
            tenant="acme",
            trace=[ToolCall(name="search_documents", summary="1 passage")],
            usage=Usage(input_tokens=500, output_tokens=50, model="search-model"),
        )

    async def invoke(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        return self._response

    async def aclose(self) -> None:
        return None


def _orchestrator(
    settings: AgenticSettings, model: ScriptedConverseModel, client: Any, resolver: TenantResolver
) -> OrchestratorAgent:
    return OrchestratorAgent(
        settings=settings, model=model, search_client=client, resolve_tenant=resolver.resolve
    )


async def test_document_question_is_delegated(
    agentic_settings: AgenticSettings, resolver: TenantResolver, base_url: str
) -> None:
    client = RecordingSearchClient()
    model = ScriptedConverseModel(
        [
            tool_turn("search_documents", {"query": "Do suppliers need ISO 9001?"}),
            text_turn("Suppliers must hold ISO 9001."),
        ]
    )
    agent = _orchestrator(agentic_settings, model, client, resolver)

    response = await agent.handle(
        AgentRequest(base_url=base_url, query="Do suppliers need ISO 9001?")
    )

    assert response.route == "document_search"
    assert len(client.requests) == 1
    assert "d1" in {c.document_id for c in response.citations}
    assert (response.environment, response.tenant) == ("dev", "acme")


async def test_small_talk_is_answered_directly(
    agentic_settings: AgenticSettings, resolver: TenantResolver, base_url: str
) -> None:
    client = RecordingSearchClient()
    model = ScriptedConverseModel([text_turn("Hello! I can search your compliance documents.")])
    agent = _orchestrator(agentic_settings, model, client, resolver)

    response = await agent.handle(AgentRequest(base_url=base_url, query="hi"))

    assert response.route == "direct"
    assert client.requests == []
    assert response.citations == []


async def test_tenancy_and_business_object_come_from_the_envelope(
    agentic_settings: AgenticSettings, resolver: TenantResolver, base_url: str
) -> None:
    """The model rewords the query; it must not be able to change the scope."""
    client = RecordingSearchClient()
    model = ScriptedConverseModel(
        [
            tool_turn(
                "search_documents",
                {"query": "ISO 9001 supplier certification requirement",
                 "base_url": "https://dev.example.io/globex",
                 "business_object": "SUPPLIERDOCUMENTS"},
            ),
            text_turn("done"),
        ]
    )
    agent = _orchestrator(agentic_settings, model, client, resolver)

    await agent.handle(
        AgentRequest(base_url=base_url, query="iso 9001?", business_object="COMPLAINTS")
    )

    delegated = client.requests[0]
    assert delegated.base_url == base_url            # not the model's suggestion
    assert delegated.business_object == "COMPLAINTS"  # not widened
    assert delegated.query == "ISO 9001 supplier certification requirement"  # reworded is fine


async def test_bad_base_url_fails_before_spending_tokens(
    agentic_settings: AgenticSettings, resolver: TenantResolver
) -> None:
    from rag_agentic.tenancy import TenancyError

    client = RecordingSearchClient()
    model = ScriptedConverseModel([text_turn("should never run")])
    agent = _orchestrator(agentic_settings, model, client, resolver)

    with pytest.raises(TenancyError):
        await agent.handle(AgentRequest(base_url="https://evil.example.com", query="hi"))

    assert model.calls == []


async def test_usage_includes_the_specialist_spend(
    agentic_settings: AgenticSettings, resolver: TenantResolver, base_url: str
) -> None:
    client = RecordingSearchClient()
    routing = tool_turn("search_documents", {"query": "iso"})
    routing.usage = Usage(input_tokens=80, output_tokens=10, model="router-model")
    unused = text_turn("a paraphrase nobody asked for")
    unused.usage = Usage(input_tokens=200, output_tokens=30, model="router-model")
    model = ScriptedConverseModel([routing, unused])
    agent = _orchestrator(agentic_settings, model, client, resolver)

    response = await agent.handle(AgentRequest(base_url=base_url, query="iso?"))

    # One routing turn (80/10) plus the search agent's own spend (500/50). The
    # second routing turn is never made, so its 200/30 is never billed.
    assert response.usage.input_tokens == 580
    assert response.usage.output_tokens == 60
    assert len(model.calls) == 1


async def test_delegated_answer_is_returned_verbatim(
    agentic_settings: AgenticSettings, resolver: TenantResolver, base_url: str
) -> None:
    """The specialist read the documents; its wording is what the caller gets."""
    client = RecordingSearchClient()
    model = ScriptedConverseModel(
        [
            tool_turn("search_documents", {"query": "iso"}),
            text_turn("Broadly speaking, some quality standards may apply."),
        ]
    )
    agent = _orchestrator(agentic_settings, model, client, resolver)

    response = await agent.handle(AgentRequest(base_url=base_url, query="iso?"))

    assert response.answer == "Suppliers must hold ISO 9001."
    assert "Broadly speaking" not in response.answer


async def test_empty_specialist_answer_falls_back_to_the_router(
    agentic_settings: AgenticSettings, resolver: TenantResolver, base_url: str
) -> None:
    client = RecordingSearchClient(
        AgentResponse(answer="   ", route="document_search", usage=Usage())
    )
    model = ScriptedConverseModel(
        [tool_turn("search_documents", {"query": "iso"}), text_turn("I could not find anything.")]
    )
    agent = _orchestrator(agentic_settings, model, client, resolver)

    response = await agent.handle(AgentRequest(base_url=base_url, query="iso?"))

    assert response.answer == "I could not find anything."


async def test_trace_includes_the_specialists_tool_calls(
    agentic_settings: AgenticSettings, resolver: TenantResolver, base_url: str
) -> None:
    client = RecordingSearchClient()
    model = ScriptedConverseModel(
        [tool_turn("search_documents", {"query": "iso"}), text_turn("done")]
    )
    agent = _orchestrator(agentic_settings, model, client, resolver)

    response = await agent.handle(AgentRequest(base_url=base_url, query="iso?"))

    assert [t.name for t in response.trace] == ["search_documents", "search_documents"]


@pytest.mark.parametrize(
    "given",
    ["", None, "short-id"],
)
def test_session_ids_are_padded_to_agentcore_minimum(given: str | None) -> None:
    assert len(ensure_session_id(given)) >= 33


def test_long_session_ids_are_preserved_verbatim() -> None:
    original = "a" * 40
    assert ensure_session_id(original) == original


async def test_a_model_routed_delegation_names_the_specialist_too(
    agentic_settings: AgenticSettings, resolver: TenantResolver, base_url: str
) -> None:
    """The router's own spend stays separate from the specialist it chose.

    This is the path where the distinction matters most: two models really did
    run, and only the sub-agent row carries a name to tell them apart.
    """
    client = RecordingSearchClient()
    routing = tool_turn("search_documents", {"query": "iso"})
    routing.usage = Usage(input_tokens=80, output_tokens=10, model="router-model")
    agent = _orchestrator(
        agentic_settings, ScriptedConverseModel([routing, text_turn("done")]), client, resolver
    )

    response = await agent.handle(AgentRequest(base_url=base_url, query="iso?"))

    [sub] = response.sub_usage
    assert sub.agent_id == "document_search"
    assert sub.model == "search-model"
