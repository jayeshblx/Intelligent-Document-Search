"""A reservation is always closed, whatever went wrong.

ExampleApp reserves credits before calling. From that moment the reservation is
outstanding, and it is settled or it hangs until it expires -- during which the
tenant's capacity is held against work that already finished. So every way out of
the handler settles, including the ones that never reach the agent:

    bad tenant, broken index, specialist down, unexpected crash, refused payload.

The one case that genuinely cannot settle is a body that is not JSON at all: there
is no reservation id to find in it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient
from rag_agentic.config import AgenticSettings, AgentRole
from rag_agentic.converse import ScriptedConverseModel
from rag_agentic.credit_gate import CreditGate
from rag_agentic.factory import AgenticComponents
from rag_agentic.index_store import IndexUnavailableError
from rag_agentic.models import AgentResponse, Usage
from rag_agentic.runtime.app import create_app
from rag_agentic.search_client import DelegateError
from rag_agentic.tenancy import TenancyError

from .test_credit_gate import FakeLicensing

RESERVATION = "res-from-exampleapp"


class _Agent:
    def __init__(self, raises: Exception | None = None) -> None:
        self._raises = raises

    async def handle(self, request: Any) -> AgentResponse:
        if self._raises:
            raise self._raises
        r = AgentResponse(answer="ok", route="document_search",
                          environment="acmedev", tenant="foxtrot")
        r.usage = Usage(input_tokens=100, output_tokens=10, model="claude", llm_calls=1)
        return r


@contextmanager
def _client(licensing: FakeLicensing, resolver: Any,
            agent: _Agent | None = None) -> Iterator[TestClient]:
    settings = AgenticSettings(
        licensing_enabled=True, licensing_base_url="https://tenantadmin.acmedev.com",
        licensing_settle_only=True, licensing_tenant_codes="foxtrot=FOXTROT",
    )
    components = AgenticComponents(
        settings=settings, role=AgentRole.ORCHESTRATOR, agent=agent or _Agent(),
        resolver=resolver,
        # A real ConverseModel with no script: the agent here is a stub, so
        # nothing ever calls it, and this keeps the component types honest.
        model=ScriptedConverseModel([]),
        credit_gate=CreditGate(licensing, settings),
    )
    with TestClient(create_app(AgentRole.ORCHESTRATOR, components=components)) as c:
        yield c


BODY = {
    "BaseUrl": "https://foxtrot.acmedev.com/",
    "QueryText": "how many documents?",
    "FeatureCode": "DOCUMENT_SEARCH",
    "ReservationId": RESERVATION,
    "UserCode": "user-1",
}


def test_a_successful_run_settles_success(resolver: Any) -> None:
    licensing = FakeLicensing()
    with _client(licensing, resolver) as c:
        assert c.post("/invocations", json=BODY).status_code == 200

    [settle] = licensing.settles
    assert settle["result"] == "Success"
    assert settle["reservation_id"] == RESERVATION


@pytest.mark.parametrize(
    ("failure", "status"),
    [
        (TenancyError("unknown tenant"), 400),
        (IndexUnavailableError("no snapshot"), 503),
        (DelegateError("search runtime failed", status=502), 502),
        (RuntimeError("something unexpected"), 500),
    ],
    ids=["bad-tenant", "index-down", "specialist-failed", "crash"],
)
def test_every_failure_still_settles(failure: Exception, status: int, resolver: Any) -> None:
    licensing = FakeLicensing()
    with _client(licensing, resolver, agent=_Agent(raises=failure)) as c:
        assert c.post("/invocations", json=BODY).status_code == status

    [settle] = licensing.settles
    assert settle["result"] == "Fail", f"{type(failure).__name__} left the reservation open"
    assert settle["reservation_id"] == RESERVATION


def test_a_refused_payload_still_settles(resolver: Any) -> None:
    """Validation fails before the request is parsed, so the reservation has to be
    recovered from the raw body."""
    licensing = FakeLicensing()
    broken = {**BODY}
    broken.pop("QueryText")          # required field
    broken["History"] = "not a list"  # and a type error for good measure

    with _client(licensing, resolver) as c:
        assert c.post("/invocations", json=broken).status_code == 400

    [settle] = licensing.settles
    assert settle["result"] == "Fail"
    assert settle["reservation_id"] == RESERVATION


def test_a_refused_payload_without_a_reservation_settles_nothing(resolver: Any) -> None:
    licensing = FakeLicensing()
    with _client(licensing, resolver) as c:
        assert c.post("/invocations", json={"BaseUrl": "https://foxtrot.acmedev.com/"}).status_code == 400

    assert licensing.settles == []


def test_a_body_that_is_not_json_cannot_settle(resolver: Any) -> None:
    """Documented, not fixed: there is no reservation id to find in it."""
    licensing = FakeLicensing()
    with _client(licensing, resolver) as c:
        r = c.post("/invocations", content=b"<<not json>>",
                   headers={"Content-Type": "application/json"})
        assert r.status_code == 400

    assert licensing.settles == []


def test_the_failure_settle_carries_no_invented_token_counts(resolver: Any) -> None:
    """A run that broke may have spent tokens we cannot account for. Reporting a
    guess would be worse than reporting nothing."""
    licensing = FakeLicensing()
    with _client(licensing, resolver, agent=_Agent(raises=RuntimeError("boom"))) as c:
        c.post("/invocations", json=BODY)

    t = licensing.settles[0]["telemetry"]
    assert (t.input_tokens, t.output_tokens) == (0, 0)
    assert t.execution_time_ms >= 0
