"""A request either yields an answer or an error status -- never an apology with a 200.

Found running the real image: the tenant's snapshot failed to restore, the search
agent raised, the orchestrator handed that to its model as a tool error, and ExampleApp
received HTTP 200 with "I'm experiencing a technical issue" as the answer. These
tests pin the contract at every hop that could swallow a failure.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from rag_agentic.agent_loop import ToolError, run_agent
from rag_agentic.agents.orchestrator import OrchestratorAgent
from rag_agentic.agents.search_agent import DocumentSearchAgent
from rag_agentic.config import AgenticSettings, AgentRole
from rag_agentic.converse import ScriptedConverseModel, ToolSpec, text_turn, tool_turn
from rag_agentic.factory import AgenticComponents
from rag_agentic.index_store import IndexUnavailableError, TenantIndexStore
from rag_agentic.models import AgentRequest
from rag_agentic.runtime.app import create_app
from rag_agentic.search_client import (
    AgentCoreSearchClient,
    DelegateError,
    InProcessSearchClient,
)
from rag_agentic.tenancy import TenantResolver


def _orchestrator_over_empty_bucket(
    settings: AgenticSettings, embedder: Any, empty_bucket: Any, resolver: TenantResolver
) -> tuple[OrchestratorAgent, ScriptedConverseModel]:
    """A real orchestrator delegating to a real search agent whose tenant has no snapshot."""
    search = DocumentSearchAgent(
        settings=settings,
        model=ScriptedConverseModel([text_turn("unreachable")]),
        embedder=embedder,
        index_store=TenantIndexStore(settings, empty_bucket),
        resolver=resolver,
    )
    router = ScriptedConverseModel(
        [
            tool_turn("search_documents", {"query": "ISO 9001?"}),
            # If the failure were swallowed, the router would get to say this:
            text_turn("I'm experiencing a technical issue, please try again."),
        ]
    )
    agent = OrchestratorAgent(
        settings=settings,
        model=router,
        search_client=InProcessSearchClient(search),
        resolve_tenant=resolver.resolve,
    )
    return agent, router


async def test_orchestrator_raises_when_the_index_is_unavailable(
    agentic_settings: AgenticSettings,
    embedder: Any,
    empty_bucket: Any,
    resolver: TenantResolver,
    base_url: str,
) -> None:
    agent, router = _orchestrator_over_empty_bucket(
        agentic_settings, embedder, empty_bucket, resolver
    )
    with pytest.raises(IndexUnavailableError):
        await agent.handle(AgentRequest(base_url=base_url, query="ISO 9001?"))
    # The router's apology turn was never requested.
    assert len(router.calls) == 1


def test_runtime_returns_503_not_an_apology(
    agentic_settings: AgenticSettings,
    embedder: Any,
    empty_bucket: Any,
    resolver: TenantResolver,
    base_url: str,
) -> None:
    agent, router = _orchestrator_over_empty_bucket(
        agentic_settings, embedder, empty_bucket, resolver
    )
    components = AgenticComponents(
        settings=agentic_settings,
        role=AgentRole.ORCHESTRATOR,
        agent=agent,
        resolver=resolver,
        model=router,
    )
    with TestClient(create_app(AgentRole.ORCHESTRATOR, components=components)) as client:
        response = client.post("/invocations", json={"base_url": base_url, "query": "ISO 9001?"})

    assert response.status_code == 503
    body = response.json()
    assert "answer" not in body
    assert body["status"] == 503
    assert "no index snapshot" in body["error"]


async def test_recoverable_tool_errors_still_go_back_to_the_model() -> None:
    """Propagation is opt-in per exception type; a bad argument stays an observation."""

    class Tools:
        def specs(self) -> list[ToolSpec]:
            return [ToolSpec(name="t", description="d", input_schema={"type": "object"})]

        async def invoke(self, name: str, arguments: dict[str, Any]) -> tuple[Any, str]:
            raise ToolError("bad filter value")

    model = ScriptedConverseModel([tool_turn("t", {}), text_turn("fixed it")])
    result = await run_agent(
        model, system="s", user_text="q", toolbox=Tools(), propagate=(IndexUnavailableError,)
    )
    assert result.text == "fixed it"
    assert result.trace[0].error == "bad filter value"


# --------------------------------------------------------------------------
# Across the AgentCore hop: the search runtime's status survives
# --------------------------------------------------------------------------


class _FakeAgentCore:
    def __init__(self, body: dict[str, Any], status_code: int | None = None) -> None:
        self._body = body
        self._status = status_code

    async def invoke_agent_runtime(self, **kwargs: Any) -> dict[str, Any]:
        response: dict[str, Any] = {"response": json.dumps(self._body).encode()}
        if self._status is not None:
            response["statusCode"] = self._status
        return response


class _Factory:
    def __init__(self, client: _FakeAgentCore) -> None:
        self._client = client

    def client(self, service: str, **overrides: Any) -> Any:
        client = self._client

        class _CM:
            async def __aenter__(self) -> _FakeAgentCore:
                return client

            async def __aexit__(self, *exc: object) -> None:
                return None

        return _CM()


def _client(body: dict[str, Any], status_code: int | None = None) -> AgentCoreSearchClient:
    settings = AgenticSettings(search_agent_arn="arn:aws:bedrock-agentcore:us-east-1:1:runtime/x")
    return AgentCoreSearchClient(settings, factory=_Factory(_FakeAgentCore(body, status_code)))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("body", "status_code", "expected"),
    [
        ({"error": "no index snapshot", "status": 503}, None, 503),  # status in the body
        ({"error": "bad base_url"}, 400, 400),  # status only on the envelope
        ({"error": "boom"}, None, 502),  # nothing to go on: bad gateway
        ({"error": "weird", "status": 200}, None, 502),  # a "success" status on an error
    ],
)
async def test_agentcore_client_carries_the_search_runtimes_status(
    body: dict[str, Any], status_code: int | None, expected: int
) -> None:
    client = _client(body, status_code)
    with pytest.raises(DelegateError) as caught:
        await client.invoke(AgentRequest(base_url="https://x/y", query="q"))
    assert caught.value.status == expected


async def test_delegation_uses_the_configured_timeout_not_the_60s_default() -> None:
    """A cold search runtime answers in ~100 s; the AWS default would abandon it."""
    seen: dict[str, object] = {}

    class Factory:
        def client(self, service: str, **overrides: object) -> object:
            seen.update(service=service, **overrides)

            class _CM:
                async def __aenter__(self) -> object:
                    return _FakeAgentCore({"answer": "ok"})

                async def __aexit__(self, *exc: object) -> None:
                    return None

            return _CM()

    settings = AgenticSettings(
        search_agent_arn="arn:aws:bedrock-agentcore:us-east-1:1:runtime/x",
        search_invoke_timeout_s=900,
    )
    client = AgentCoreSearchClient(settings, factory=Factory())  # type: ignore[arg-type]
    await client.invoke(AgentRequest(base_url="https://x/y", query="q"))

    config = seen["config"]
    assert config.read_timeout == 900  # type: ignore[union-attr]
    assert config.retries["max_attempts"] == 1  # type: ignore[union-attr]


class _RaisingFactory:
    """AgentCore raising instead of returning a body, as it does for any non-200."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def client(self, service: str, **overrides: Any) -> Any:
        error = self._error

        class _Client:
            async def invoke_agent_runtime(self, **kwargs: Any) -> dict[str, Any]:
                raise error

        class _CM:
            async def __aenter__(self) -> Any:
                return _Client()

            async def __aexit__(self, *exc: object) -> None:
                return None

        return _CM()


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Received error (503) from runtime. Please check your CloudWatch logs.", 503),
        ("Received error (400) from runtime.", 400),
        ("connection reset by peer", 502),  # nothing to parse: bad gateway
    ],
)
async def test_a_raising_agentcore_call_becomes_a_delegate_error(
    message: str, expected: int
) -> None:
    """Found in AWS: the search runtime answered 503, AgentCore raised it at the
    orchestrator, and the failure became a 200 'I'm experiencing a technical issue'."""
    settings = AgenticSettings(search_agent_arn="arn:aws:bedrock-agentcore:us-east-1:1:runtime/x")
    client = AgentCoreSearchClient(settings, factory=_RaisingFactory(RuntimeError(message)))  # type: ignore[arg-type]

    with pytest.raises(DelegateError) as caught:
        await client.invoke(AgentRequest(base_url="https://x/y", query="q"))
    assert caught.value.status == expected
    assert "document-search runtime failed" in str(caught.value)


async def test_the_final_turn_keeps_toolconfig_defined() -> None:
    """Found in AWS as a 500: Bedrock rejects a request whose history contains
    toolUse blocks unless toolConfig is still defined, so the ceiling turn must keep
    the tools defined and tell the model to stop calling them instead."""

    class Tools:
        def specs(self) -> list[ToolSpec]:
            return [ToolSpec(name="t", description="d", input_schema={"type": "object"})]

        async def invoke(self, name: str, arguments: dict[str, Any]) -> tuple[Any, str]:
            return {"ok": True}, "ok"

    # Never stops calling tools, so the ceiling is always hit.
    model = ScriptedConverseModel([tool_turn("t", {}), tool_turn("t", {}), text_turn("final answer")])
    result = await run_agent(model, system="s", user_text="q", toolbox=Tools(), max_iterations=2)

    assert result.truncated is True
    assert result.text == "final answer"
    final_call = model.calls[-1]
    assert final_call["tools"], "toolConfig must stay defined or Bedrock returns a ValidationException"
    last_message = final_call["messages"][-1]
    assert "tool-call limit" in last_message["content"][0]["text"]
