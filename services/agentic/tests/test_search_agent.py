"""Document Search Agent: the loop, the scope note, and the citations it emits."""

from __future__ import annotations

import json
from typing import Any

from rag_agentic.agents.search_agent import DocumentSearchAgent
from rag_agentic.config import AgenticSettings
from rag_agentic.converse import ModelTurn, ScriptedConverseModel, ToolUse, text_turn, tool_turn
from rag_agentic.index_store import TenantIndexStore
from rag_agentic.models import AgentRequest, Usage
from rag_agentic.tenancy import TenantResolver


def _agent(
    settings: AgenticSettings,
    model: ScriptedConverseModel,
    embedder: Any,
    index_store: TenantIndexStore,
    resolver: TenantResolver,
) -> DocumentSearchAgent:
    return DocumentSearchAgent(
        settings=settings,
        model=model,
        embedder=embedder,
        index_store=index_store,
        resolver=resolver,
    )


def _tool_results(model: ScriptedConverseModel) -> list[dict[str, Any]]:
    """The tool-result payloads that were fed back to the model."""
    payloads = []
    for call in model.calls:
        for message in call["messages"]:
            for block in message.get("content", []):
                if "toolResult" in block:
                    payloads.append(json.loads(block["toolResult"]["content"][0]["text"]))
    return payloads


async def test_searches_then_answers(
    agentic_settings: AgenticSettings,
    embedder: Any,
    index_store: TenantIndexStore,
    resolver: TenantResolver,
    base_url: str,
) -> None:
    model = ScriptedConverseModel(
        [
            tool_turn("search_documents", {"query": "ISO 9001 certificate"}),
            text_turn("Suppliers must hold a valid ISO 9001 certificate, renewed every three years."),
        ]
    )
    agent = _agent(agentic_settings, model, embedder, index_store, resolver)

    response = await agent.handle(
        AgentRequest(base_url=base_url, query="Do suppliers need ISO 9001?")
    )

    assert "ISO 9001" in response.answer
    assert response.route == "document_search"
    assert (response.environment, response.tenant) == ("dev", "acme")
    assert [t.name for t in response.trace] == ["search_documents"]
    assert "d1" in {c.document_id for c in response.citations}


async def test_scope_note_tells_the_model_which_filter_is_active(
    agentic_settings: AgenticSettings,
    embedder: Any,
    index_store: TenantIndexStore,
    resolver: TenantResolver,
    base_url: str,
) -> None:
    model = ScriptedConverseModel([text_turn("ok")])
    agent = _agent(agentic_settings, model, embedder, index_store, resolver)

    await agent.handle(
        AgentRequest(base_url=base_url, query="anything", business_object="COMPLAINTS")
    )

    system = model.calls[0]["system"]
    assert "COMPLAINTS" in system
    assert "acme" in system
    # The model must be offered the retrieval tools, not left to answer unaided.
    assert "search_documents" in model.calls[0]["tools"]


async def test_multi_step_loop_feeds_observations_back(
    agentic_settings: AgenticSettings,
    embedder: Any,
    index_store: TenantIndexStore,
    resolver: TenantResolver,
    base_url: str,
) -> None:
    """Facet lookup, then a filtered listing, then the answer."""
    model = ScriptedConverseModel(
        [
            tool_turn("list_facet_values", {}, use_id="a"),
            tool_turn("list_documents", {"status": "Draft"}, use_id="b"),
            text_turn("One supplier document is in Draft: the Supplier Audit Schedule."),
        ]
    )
    agent = _agent(agentic_settings, model, embedder, index_store, resolver)

    response = await agent.handle(
        AgentRequest(base_url=base_url, query="Which supplier documents are still draft?",
                     business_object="SUPPLIERDOCUMENTS")
    )

    assert [t.name for t in response.trace] == ["list_facet_values", "list_documents"]
    listing = _tool_results(model)[-1]
    assert {d["document_id"] for d in listing["documents"]} == {"d3"}
    assert "Draft" in response.answer


async def test_failing_tool_is_reported_to_the_model_not_raised(
    agentic_settings: AgenticSettings,
    embedder: Any,
    index_store: TenantIndexStore,
    resolver: TenantResolver,
    base_url: str,
) -> None:
    model = ScriptedConverseModel(
        [
            tool_turn("get_document_details", {"document_id": "missing"}),
            text_turn("I could not find that document."),
        ]
    )
    agent = _agent(agentic_settings, model, embedder, index_store, resolver)

    response = await agent.handle(AgentRequest(base_url=base_url, query="tell me about missing"))

    assert response.trace[0].error is not None
    assert _tool_results(model)[0]["error"]
    assert response.answer == "I could not find that document."


async def test_iteration_ceiling_forces_a_final_answer(
    agentic_settings: AgenticSettings,
    embedder: Any,
    index_store: TenantIndexStore,
    resolver: TenantResolver,
    base_url: str,
) -> None:
    """A model that only ever asks for tools must still be made to answer."""
    settings = agentic_settings.model_copy(update={"max_tool_iterations": 2})
    model = ScriptedConverseModel(
        [
            tool_turn("search_documents", {"query": "one"}, use_id="1"),
            tool_turn("search_documents", {"query": "two"}, use_id="2"),
            text_turn("Based on what I found, suppliers need ISO 9001."),
        ]
    )
    agent = _agent(settings, model, embedder, index_store, resolver)

    response = await agent.handle(AgentRequest(base_url=base_url, query="loop forever"))

    assert len(response.trace) == 2
    assert response.answer.startswith("Based on what I found")
    # The forced final turn keeps the tools DEFINED -- Bedrock rejects a request whose
    # history contains toolUse blocks when toolConfig is absent, which showed up in AWS
    # as a 500. What stops the looping is the instruction, not withdrawing the tools.
    assert model.calls[-1]["tools"]
    assert "tool-call limit" in model.calls[-1]["messages"][-1]["content"][0]["text"]


async def test_usage_is_summed_across_turns(
    agentic_settings: AgenticSettings,
    embedder: Any,
    index_store: TenantIndexStore,
    resolver: TenantResolver,
    base_url: str,
) -> None:
    first = tool_turn("search_documents", {"query": "iso"})
    first.usage = Usage(input_tokens=100, output_tokens=20, model="test-model")
    second = text_turn("done")
    second.usage = Usage(input_tokens=300, output_tokens=40, model="test-model")
    model = ScriptedConverseModel([first, second])
    agent = _agent(agentic_settings, model, embedder, index_store, resolver)

    response = await agent.handle(AgentRequest(base_url=base_url, query="iso?"))

    assert response.usage.input_tokens == 400
    assert response.usage.output_tokens == 60
    assert response.usage.model == "test-model"


async def test_parallel_tool_calls_return_in_one_message(
    agentic_settings: AgenticSettings,
    embedder: Any,
    index_store: TenantIndexStore,
    resolver: TenantResolver,
    base_url: str,
) -> None:
    """Splitting parallel results across messages teaches the model to stop batching."""
    parallel = ModelTurn(
        tool_uses=[
            ToolUse(id="p1", name="search_documents", input={"query": "iso"}),
            ToolUse(id="p2", name="list_documents", input={}),
        ],
        stop_reason="tool_use",
    )
    model = ScriptedConverseModel([parallel, text_turn("both done")])
    agent = _agent(agentic_settings, model, embedder, index_store, resolver)

    response = await agent.handle(AgentRequest(base_url=base_url, query="two things"))

    assert len(response.trace) == 2
    result_messages = [
        m
        for m in model.calls[-1]["messages"]
        if any("toolResult" in b for b in m.get("content", []))
    ]
    assert len(result_messages) == 1
    assert len(result_messages[0]["content"]) == 2
