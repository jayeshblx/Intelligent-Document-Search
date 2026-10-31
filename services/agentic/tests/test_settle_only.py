"""Usage reporting without the gate.

ExampleApp reserves the credits itself and sends the ReservationId with the request.
The agent then has one job: settle that reservation with the run's telemetry.

The property that matters most here is the one that is easy to lose: in this mode
metering must never be able to refuse an answer. A reservation that is missing, or
a Licensing service that is down, costs a usage record -- not the user's search.
"""

from __future__ import annotations

from typing import Any

import pytest
from rag_agentic.config import AgenticSettings
from rag_agentic.credit_gate import CreditGate
from rag_agentic.licensing import LicensingUnavailable
from rag_agentic.models import AgentRequest, AgentResponse, Usage

from .test_credit_gate import FakeLicensing


def _settings(**over: Any) -> AgenticSettings:
    return AgenticSettings(**{
        "licensing_enabled": True,
        "licensing_base_url": "https://tenantadmin.acmedev.com",
        "licensing_settle_only": True,
        "licensing_tenant_codes": "foxtrot=FOXTROT,bravo=BRAVO",
        **over,
    })


def _request(**over: Any) -> AgentRequest:
    return AgentRequest.model_validate({
        "BaseUrl": "https://foxtrot.acmedev.com/",
        "QueryText": "how many documents?",
        "FeatureCode": "DOCUMENT_SEARCH",
        **over,
    })


def _response() -> AgentResponse:
    r = AgentResponse(answer="1,359.", route="document_search",
                      environment="acmedev", tenant="foxtrot")
    r.usage = Usage(input_tokens=3322, output_tokens=51, cache_read_tokens=1955,
                    embedding_tokens=7, model="claude", llm_calls=2)
    r.duration_ms = 29760.4
    r.model_id, r.agent_id = "claude", "orchestrator"
    return r


async def test_a_supplied_reservation_is_settled_without_reserving() -> None:
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings())

    hold = await gate.reserve(_request(ReservationId="res-77"), tenant="foxtrot", trace_id="t")
    await gate.settle_success(hold, _response())

    assert licensing.reserves == [], "CheckAndReserve must not be called in this mode"
    [settle] = licensing.settles
    assert settle["reservation_id"] == "res-77"
    assert settle["result"] == "Success"


async def test_the_settled_telemetry_is_the_whole_run() -> None:
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings())

    hold = await gate.reserve(_request(ReservationId="res-1"), tenant="foxtrot", trace_id="t")
    await gate.settle_success(hold, _response())

    t = licensing.settles[0]["telemetry"]
    assert (t.input_tokens, t.output_tokens, t.embedding_tokens) == (3322, 51, 7)
    assert t.execution_time_ms == 29760
    assert t.batch_call_count == 2


async def test_a_failed_run_settles_as_fail() -> None:
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings())

    hold = await gate.reserve(_request(ReservationId="res-2"), tenant="foxtrot", trace_id="t")
    await gate.settle_failure(hold, execution_time_ms=900)

    assert licensing.settles[0]["result"] == "Fail"


# ---- metering must never refuse an answer ----------------------------------------

async def test_a_missing_reservation_does_not_block_the_run() -> None:
    """A caller that forgets the field loses a usage record, not its search."""
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings())

    hold = await gate.reserve(_request(), tenant="foxtrot", trace_id="t")

    assert hold is None
    assert licensing.reserves == [] and licensing.settles == []


async def test_licensing_being_down_does_not_block_the_run() -> None:
    licensing = FakeLicensing(settle_raises=LicensingUnavailable("tenantadmin down"))
    gate = CreditGate(licensing, _settings())

    hold = await gate.reserve(_request(ReservationId="res-3"), tenant="foxtrot", trace_id="t")
    await gate.settle_success(hold, _response())  # must not raise

    assert len(licensing.settles) == 1


async def test_fail_closed_does_not_apply_in_this_mode() -> None:
    """`licensing_fail_open=False` is about the gate. There is no gate here, so it
    must not resurrect one."""
    licensing = FakeLicensing(reserve_raises=LicensingUnavailable("down"))
    gate = CreditGate(licensing, _settings(licensing_fail_open=False))

    hold = await gate.reserve(_request(), tenant="foxtrot", trace_id="t")

    assert hold is None, "a run was refused in a mode that cannot refuse runs"


# ---- the gated mode is unchanged --------------------------------------------------

async def test_without_settle_only_the_gate_still_reserves() -> None:
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings(licensing_settle_only=False))

    await gate.reserve(_request(), tenant="foxtrot", trace_id="t")

    assert len(licensing.reserves) == 1


async def test_a_supplied_reservation_wins_even_in_gated_mode() -> None:
    """If ExampleApp already reserved, reserving again would double-charge."""
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings(licensing_settle_only=False))

    hold = await gate.reserve(_request(ReservationId="res-9"), tenant="foxtrot", trace_id="t")

    assert licensing.reserves == []
    assert hold is not None and hold.reservation_id == "res-9"


@pytest.mark.parametrize("spelling", ["ReservationId", "reservationId", "reservation_id"])
async def test_reservation_id_spellings(spelling: str) -> None:
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings())

    hold = await gate.reserve(_request(**{spelling: "res-5"}), tenant="foxtrot", trace_id="t")

    assert hold is not None and hold.reservation_id == "res-5"
