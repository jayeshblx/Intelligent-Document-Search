"""Licensing seen from /invocations -- the path a real ExampleApp call takes.

The gate's own tests prove it behaves; these prove it is actually wired in, on
every exit the handler has. A gate that is correct but bypassed on the error path
would under-report exactly the runs worth investigating.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fastapi.testclient import TestClient
from rag_agentic.config import AgenticSettings, AgentRole
from rag_agentic.credit_gate import CreditGate
from rag_agentic.factory import AgenticComponents
from rag_agentic.licensing import LicensingUnavailable
from rag_agentic.models import AgentResponse, Usage
from rag_agentic.runtime.app import create_app
from rag_agentic.tenancy import TenancyError

from .test_credit_gate import FakeLicensing


class _Agent:
    """Stands in for the orchestrator: answers, or fails the way it would."""

    def __init__(self, raises: Exception | None = None) -> None:
        self._raises = raises
        self.calls = 0

    async def handle(self, request: Any) -> AgentResponse:
        self.calls += 1
        if self._raises:
            raise self._raises
        response = AgentResponse(answer="1,359 documents.", route="document_search",
                                 environment="acmedev", tenant="foxtrot")
        response.usage = Usage(input_tokens=3322, output_tokens=51, cache_read_tokens=1955,
                               embedding_tokens=7, model="claude", llm_calls=2)
        return response


@contextmanager
def _client(
    licensing: FakeLicensing | None, resolver: Any, *, agent: _Agent | None = None,
    **settings_over: Any,
) -> Iterator[tuple[TestClient, _Agent]]:
    """The real app with its lifespan run -- that is what attaches components."""
    settings = AgenticSettings(**{
        "licensing_enabled": True,
        "licensing_base_url": "https://tenantadmin.acmedev.com",
        "licensing_tenant_codes": "foxtrot=FOXTROT",
        **settings_over,
    })
    agent = agent or _Agent()
    components = AgenticComponents(
        settings=settings,
        role=AgentRole.ORCHESTRATOR,
        agent=agent,
        resolver=resolver,
        model=None,  # type: ignore[arg-type]  # the agent is a stub; no model is reached
        credit_gate=CreditGate(licensing, settings),
    )
    with TestClient(create_app(AgentRole.ORCHESTRATOR, components=components)) as client:
        yield client, agent


BODY = {
    "BaseUrl": "https://foxtrot.acmedev.com/",
    "QueryText": "how many documents?",
    "FeatureCode": "DOCUMENT_SEARCH",
    "UserCode": "user-1",
}


def test_a_successful_call_reserves_then_settles(resolver: Any) -> None:
    licensing = FakeLicensing()
    with _client(licensing, resolver) as (client, agent):
        response = client.post("/invocations", json=BODY)

    assert response.status_code == 200
    assert agent.calls == 1
    assert len(licensing.reserves) == 1
    [settle] = licensing.settles
    assert settle["result"] == "Success"


def test_settle_reports_the_same_numbers_the_caller_is_shown(resolver: Any) -> None:
    """Two tallies that can disagree is how billing disputes start."""
    licensing = FakeLicensing()
    with _client(licensing, resolver) as (client, _):
        body = client.post("/invocations", json=BODY).json()

    telemetry = licensing.settles[0]["telemetry"]
    assert telemetry.input_tokens == body["input_tokens"]
    assert telemetry.output_tokens == body["output_tokens"]
    assert telemetry.total_tokens == body["total_tokens"]
    assert telemetry.embedding_tokens == body["embedding_tokens"]
    assert telemetry.cache_read_tokens == body["cache_read_tokens"]
    assert telemetry.execution_time_ms == round(body["duration_ms"])


def test_a_decline_never_reaches_the_agent(resolver: Any) -> None:
    licensing = FakeLicensing(allowed=False, reason="insufficient_credits")
    with _client(licensing, resolver) as (client, agent):
        response = client.post("/invocations", json=BODY)

    assert response.status_code == 402
    assert response.json()["reason"] == "insufficient_credits"
    assert agent.calls == 0, "the model must not run on credits the tenant does not have"
    assert licensing.settles == []


def test_an_unreachable_licensing_refuses_the_request(resolver: Any) -> None:
    licensing = FakeLicensing(reserve_raises=LicensingUnavailable("down"))
    with _client(licensing, resolver) as (client, agent):
        response = client.post("/invocations", json=BODY)

    assert response.status_code == 503
    assert agent.calls == 0


def test_fail_open_runs_the_request_unmetered(resolver: Any) -> None:
    licensing = FakeLicensing(reserve_raises=LicensingUnavailable("down"))
    with _client(licensing, resolver, licensing_fail_open=True) as (client, agent):
        response = client.post("/invocations", json=BODY)

    assert response.status_code == 200
    assert agent.calls == 1
    assert licensing.settles == [], "nothing was reserved, so there is nothing to settle"


def test_a_broken_run_settles_as_fail(resolver: Any) -> None:
    licensing = FakeLicensing()
    with _client(licensing, resolver, agent=_Agent(raises=RuntimeError("boom"))) as (client, _):
        response = client.post("/invocations", json=BODY)

    assert response.status_code == 500
    [settle] = licensing.settles
    assert settle["result"] == "Fail"


def test_an_unknown_tenant_settles_as_fail(resolver: Any) -> None:
    """The reservation was taken before the tenant was rejected, so it is closed
    rather than left to expire."""
    licensing = FakeLicensing()
    broken = _Agent(raises=TenancyError("no such tenant"))
    with _client(licensing, resolver, agent=broken) as (client, _):
        response = client.post("/invocations", json=BODY)

    assert response.status_code == 400
    assert licensing.settles[0]["result"] == "Fail"


def test_a_failed_settle_still_returns_the_answer(resolver: Any) -> None:
    licensing = FakeLicensing(settle_raises=LicensingUnavailable("down"))
    with _client(licensing, resolver) as (client, _):
        response = client.post("/invocations", json=BODY)

    assert response.status_code == 200
    assert response.json()["answer"] == "1,359 documents."


def test_the_access_token_is_never_echoed_back(resolver: Any) -> None:
    licensing = FakeLicensing()
    with _client(licensing, resolver) as (client, _):
        body = client.post("/invocations", json={**BODY, "AccessToken": "Bearer eyJsecret"}).json()

    assert "eyJsecret" not in str(body)
    assert licensing.reserves[0]["access_token"] == "Bearer eyJsecret"


def test_licensing_off_leaves_the_request_untouched(resolver: Any) -> None:
    with _client(None, resolver, licensing_enabled=False) as (client, agent):
        response = client.post("/invocations", json=BODY)

    assert response.status_code == 200
    assert agent.calls == 1
