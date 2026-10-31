"""`FeatureCode` routes straight to a specialist, with no model deciding.

Routing used to be a model call: the orchestrator read the question and chose
whether to delegate. That costs a round trip (~10s measured) and can decide
differently on identical input. When ExampleApp already knows which capability it
is invoking, there is nothing to infer.

A typo must FAIL rather than fall back to model routing — a silent fallback would
restore exactly the non-determinism the field exists to remove.
"""

from __future__ import annotations

from typing import Any

import pytest
from rag_agentic.agents.orchestrator import FEATURE_ROUTES, OrchestratorAgent
from rag_agentic.config import AgenticSettings
from rag_agentic.converse import ScriptedConverseModel, text_turn
from rag_agentic.models import AgentRequest
from rag_agentic.tenancy import TenancyError, TenantResolver

from .test_orchestrator import RecordingSearchClient


@pytest.mark.parametrize("spelling", ["FeatureCode", "featureCode", "feature_code", "feature"])
def test_feature_code_spellings(spelling: str) -> None:
    request = AgentRequest.model_validate(
        {"BaseUrl": "https://bravo.acmedev.com/", "Query": "q", spelling: "DOCUMENT_SEARCH"}
    )
    assert request.feature_code == "DOCUMENT_SEARCH"


async def test_a_named_feature_skips_the_model(
    agentic_settings: AgenticSettings, resolver: TenantResolver, base_url: str
) -> None:
    """The saving: no routing turn at all."""
    client = RecordingSearchClient()
    model = ScriptedConverseModel([text_turn("the router must not be consulted")])
    agent = OrchestratorAgent(
        settings=agentic_settings, model=model, search_client=client,
        resolve_tenant=resolver.resolve,
    )

    response = await agent.handle(
        AgentRequest(base_url=base_url, query="anything at all", feature_code="DOCUMENT_SEARCH")
    )

    assert model.calls == [], "the routing model was called despite a FeatureCode"
    assert len(client.requests) == 1, "the specialist was not reached"
    assert response.route == "document_search"


async def test_the_same_request_always_reaches_the_same_specialist(
    agentic_settings: AgenticSettings, resolver: TenantResolver, base_url: str
) -> None:
    routes = []
    for _ in range(5):
        client = RecordingSearchClient()
        agent = OrchestratorAgent(
            settings=agentic_settings, model=ScriptedConverseModel([]),
            search_client=client, resolve_tenant=resolver.resolve,
        )
        r = await agent.handle(
            AgentRequest(base_url=base_url, query="List of Documents created by Alex",
                         feature_code="DOCUMENT_SEARCH")
        )
        routes.append(r.route)

    assert set(routes) == {"document_search"}


@pytest.mark.parametrize("given", ["document_search", "Document_Search", "DOCUMENT_SEARCH"])
async def test_the_code_is_matched_case_insensitively(
    given: str, agentic_settings: AgenticSettings, resolver: TenantResolver, base_url: str
) -> None:
    client = RecordingSearchClient()
    agent = OrchestratorAgent(
        settings=agentic_settings, model=ScriptedConverseModel([]),
        search_client=client, resolve_tenant=resolver.resolve,
    )

    await agent.handle(AgentRequest(base_url=base_url, query="q", feature_code=given))

    assert len(client.requests) == 1


async def test_an_unknown_feature_code_is_refused(
    agentic_settings: AgenticSettings, resolver: TenantResolver, base_url: str
) -> None:
    """Falling back to model routing on a typo would hide the mistake and give
    back the non-determinism this field removes."""
    agent = OrchestratorAgent(
        settings=agentic_settings, model=ScriptedConverseModel([text_turn("x")]),
        search_client=RecordingSearchClient(), resolve_tenant=resolver.resolve,
    )

    with pytest.raises(TenancyError, match="unknown FeatureCode"):
        await agent.handle(
            AgentRequest(base_url=base_url, query="q", feature_code="DOCUMNET_SEARCH")
        )


async def test_the_error_names_the_valid_codes(
    agentic_settings: AgenticSettings, resolver: TenantResolver, base_url: str
) -> None:
    agent = OrchestratorAgent(
        settings=agentic_settings, model=ScriptedConverseModel([]),
        search_client=RecordingSearchClient(), resolve_tenant=resolver.resolve,
    )

    with pytest.raises(TenancyError, match="DOCUMENT_SEARCH"):
        await agent.handle(AgentRequest(base_url=base_url, query="q", feature_code="NOPE"))


async def test_without_a_feature_code_the_model_still_routes(
    agentic_settings: AgenticSettings, resolver: TenantResolver, base_url: str
) -> None:
    """Existing callers keep working unchanged."""
    from rag_agentic.converse import tool_turn

    client = RecordingSearchClient()
    model = ScriptedConverseModel([
        tool_turn("search_documents", {"query": "ISO 9001?"}),
        text_turn("Suppliers must hold ISO 9001."),
    ])
    agent = OrchestratorAgent(
        settings=agentic_settings, model=model, search_client=client,
        resolve_tenant=resolver.resolve,
    )

    response = await agent.handle(AgentRequest(base_url=base_url, query="Do suppliers need ISO 9001?"))

    assert model.calls, "the router should still decide when no feature is named"
    assert response.route == "document_search"


def test_adding_a_capability_is_a_registry_entry() -> None:
    """A second feature should not mean re-teaching the router's prompt."""
    assert FEATURE_ROUTES["DOCUMENT_SEARCH"] == "document_search"
    assert all(code == code.upper() for code in FEATURE_ROUTES), "codes are compared uppercased"


# ---- a second feature must not land on the first one's specialist -----------------
#
# These register a pretend second capability so the routing is exercised with more
# than one destination. Until this ran, `route` was resolved, logged, and then
# discarded: dispatch always went to the document-search client, so a second
# FEATURE_ROUTES entry would have answered scorecard questions out of the document
# corpus -- indistinguishable, to the caller, from a hallucination.


class _OtherClient:
    """A stand-in for some future specialist."""

    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []

    async def invoke(self, request: AgentRequest) -> Any:
        from rag_agentic.models import AgentResponse

        self.requests.append(request)
        # model_construct: `route` is a Literal of the routes that exist today, and
        # this stub stands in for one that does not. Adding a real feature widens
        # that Literal -- a missing entry is a loud validation error on the
        # response, not a misroute, so it needs no guard of its own.
        future_route: Any = "supplier_scorecard"
        return AgentResponse.model_construct(
            answer="from the other specialist", route=future_route,
            environment="dev", tenant="acme",
        )

    async def aclose(self) -> None:
        return None


@pytest.fixture
def two_features(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(FEATURE_ROUTES, "SUPPLIER_SCORECARD", "supplier_scorecard")


async def test_each_feature_reaches_its_own_specialist(
    two_features: None, agentic_settings: AgenticSettings, resolver: TenantResolver, base_url: str
) -> None:
    search, other = RecordingSearchClient(), _OtherClient()
    agent = OrchestratorAgent(
        settings=agentic_settings, model=ScriptedConverseModel([]), search_client=search,
        specialists={"supplier_scorecard": other}, resolve_tenant=resolver.resolve,
    )

    scorecard = await agent.handle(
        AgentRequest(base_url=base_url, query="q", feature_code="SUPPLIER_SCORECARD")
    )
    documents = await agent.handle(
        AgentRequest(base_url=base_url, query="q", feature_code="DOCUMENT_SEARCH")
    )

    assert scorecard.answer == "from the other specialist"
    assert len(other.requests) == 1
    assert len(search.requests) == 1, "the scorecard request leaked into document search"
    assert documents.route == "document_search"


async def test_a_registered_feature_with_no_client_refuses_to_start(
    two_features: None, agentic_settings: AgenticSettings, resolver: TenantResolver
) -> None:
    """The mistake is caught when the agent is built, not on the first request --
    a half-wired registry must never serve traffic."""
    with pytest.raises(TenancyError, match="supplier_scorecard"):
        OrchestratorAgent(
            settings=agentic_settings, model=ScriptedConverseModel([]),
            search_client=RecordingSearchClient(), resolve_tenant=resolver.resolve,
        )


async def test_routing_is_stable_across_repeats_with_several_features(
    two_features: None, agentic_settings: AgenticSettings, resolver: TenantResolver, base_url: str
) -> None:
    """The guarantee the FeatureCode exists for: same input, same specialist, every
    time, with no model in the path to drift."""
    seen = []
    for _ in range(10):
        search, other = RecordingSearchClient(), _OtherClient()
        agent = OrchestratorAgent(
            settings=agentic_settings, model=ScriptedConverseModel([]), search_client=search,
            specialists={"supplier_scorecard": other}, resolve_tenant=resolver.resolve,
        )
        await agent.handle(
            AgentRequest(base_url=base_url, query="List of Documents created by Alex",
                         feature_code="SUPPLIER_SCORECARD")
        )
        seen.append((len(other.requests), len(search.requests)))

    assert set(seen) == {(1, 0)}


async def test_the_specialists_spend_is_attributed_to_it_by_name(
    agentic_settings: AgenticSettings, resolver: TenantResolver, base_url: str
) -> None:
    """Otherwise a SubAgents row says only that *something* spent 550 tokens.

    Every specialist runs on the same Claude model, so ModelId cannot tell them
    apart. On this path no routing turn happens at all, which makes it the case
    most likely to lose the name: the specialist's spend *is* the whole run.
    """
    client = RecordingSearchClient()
    agent = OrchestratorAgent(
        settings=agentic_settings, model=ScriptedConverseModel([]),
        search_client=client, resolve_tenant=resolver.resolve,
    )

    response = await agent.handle(
        AgentRequest(base_url=base_url, query="q", feature_code="DOCUMENT_SEARCH")
    )

    [sub] = response.sub_usage
    assert sub.agent_id == "document_search"
    assert sub.input_tokens == 500
