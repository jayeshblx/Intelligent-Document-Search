"""One settle, one transaction id -- and a trace id you can actually search.

Licensing stores TransactionId as the identity of the settle transaction. Reusing
one across two settles makes them indistinguishable in the ledger, and a request
can legitimately settle more than once over its life. The trace id therefore
cannot serve as the transaction id, however convenient it looks.

TraceId has the opposite problem: AWS hands us a header full of sampling
bookkeeping, of which only the Root value identifies the trace.
"""

from __future__ import annotations

from typing import Any

import pytest
from rag_agentic.config import AgenticSettings
from rag_agentic.credit_gate import CreditGate
from rag_agentic.licensing import new_transaction_id, root_trace_id
from rag_agentic.models import AgentRequest, AgentResponse, Usage

from .test_credit_gate import FakeLicensing

XRAY = ("Root=1-6ad876cb-6fdedc5574a923800e0d2dc7;Parent=c838bb01876a26ba;"
        "Sampled=1;Self=1-6ad876cb-7accf9d731e0b5e776218489")
ROOT = "1-6ad876cb-6fdedc5574a923800e0d2dc7"


def _settings() -> AgenticSettings:
    return AgenticSettings(
        licensing_enabled=True, licensing_settle_only=True,
        licensing_base_url="https://tenantadmin.acmedev.com",
        licensing_tenant_codes="foxtrot=FOXTROT",
    )


def _request() -> AgentRequest:
    return AgentRequest.model_validate({
        "BaseUrl": "https://foxtrot.acmedev.com/", "QueryText": "q",
        "FeatureCode": "DOCUMENT_SEARCH", "ReservationId": "res-1",
    })


def _response() -> AgentResponse:
    r = AgentResponse(answer="ok", route="document_search",
                      environment="acmedev", tenant="foxtrot")
    r.usage = Usage(input_tokens=10, output_tokens=2, model="claude", llm_calls=1)
    return r


# ---- the trace id --------------------------------------------------------------

def test_the_root_trace_id_is_extracted() -> None:
    assert root_trace_id(XRAY) == ROOT


@pytest.mark.parametrize(
    ("header", "expected"),
    [(XRAY, ROOT), (f"Root={ROOT}", ROOT), ("", ""), ("not-a-header", "not-a-header")],
)
def test_trace_id_handles_what_aws_actually_sends(header: str, expected: str) -> None:
    """A plain id, a bare Root, nothing at all -- none of these should throw."""
    assert root_trace_id(header) == expected


def test_the_stored_trace_id_carries_no_sampling_noise() -> None:
    trimmed = root_trace_id(XRAY)

    assert "Parent=" not in trimmed and "Sampled=" not in trimmed and "Self=" not in trimmed
    assert len(trimmed) < 40, "a billing record should not store 115 characters of bookkeeping"


# ---- the transaction id --------------------------------------------------------

def test_transaction_ids_are_unique() -> None:
    assert len({new_transaction_id() for _ in range(500)}) == 500


async def test_two_settles_of_one_request_get_different_transaction_ids() -> None:
    """The case that made this a bug: a run settles Fail, then a correction settles
    again. Sharing a transaction id makes the two indistinguishable in the ledger."""
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings())
    hold = await gate.reserve(_request(), tenant="foxtrot", trace_id=XRAY)

    await gate.settle_failure(hold, execution_time_ms=100)
    await gate.settle_success(hold, _response())

    first, second = licensing.settles
    assert first["transaction_id"] != second["transaction_id"]
    assert first["reservation_id"] == second["reservation_id"] == "res-1"


async def test_the_transaction_id_is_not_the_trace_id() -> None:
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings())
    hold = await gate.reserve(_request(), tenant="foxtrot", trace_id=XRAY)

    await gate.settle_success(hold, _response())

    settle = licensing.settles[0]
    assert settle["transaction_id"] != settle["trace_id"]
    assert settle["trace_id"] == ROOT
    assert "Sampled=" not in settle["transaction_id"]


async def test_the_trace_id_still_correlates_to_cloudwatch(caplog: Any) -> None:
    """Trimming must not break the link back to the logs."""
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings())
    hold = await gate.reserve(_request(), tenant="foxtrot", trace_id=XRAY)

    await gate.settle_success(hold, _response())

    assert licensing.settles[0]["trace_id"] in XRAY, "the stored id must appear in the header"
