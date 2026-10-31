"""A run that fails partway still reports the tokens it burned.

Bedrock bills a half-finished run exactly like a complete one. If the agent
reported zero on failure, that cost would move quietly from the tenant who
incurred it onto whoever operates the agent -- and nothing would ever surface the
discrepancy, because a failed request looks free.

The spend cannot come from the response: there is no response. It comes from a
per-request accumulator that survives the exception unwinding the stack.
"""

from __future__ import annotations

from typing import Any

import pytest
from rag_agentic.config import AgenticSettings
from rag_agentic.converse import reset_spent, spent_so_far
from rag_agentic.credit_gate import CreditGate
from rag_agentic.licensing import RunTelemetry
from rag_agentic.models import AgentRequest, Usage

from .test_credit_gate import FakeLicensing


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


@pytest.fixture(autouse=True)
def _clean_accumulator() -> Any:
    reset_spent()
    yield
    reset_spent()


async def test_a_failed_run_reports_what_it_spent() -> None:
    from rag_agentic.converse import _record_spent

    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings())
    hold = await gate.reserve(_request(), tenant="foxtrot", trace_id="t")

    # Two model calls happened, then the run broke.
    _record_spent(Usage(input_tokens=1200, output_tokens=80, cache_read_tokens=900,
                        model="claude", llm_calls=1))
    _record_spent(Usage(input_tokens=1500, output_tokens=40, model="claude", llm_calls=1))

    await gate.settle_failure(hold, execution_time_ms=4200)

    t = licensing.settles[0]["telemetry"]
    assert licensing.settles[0]["result"] == "Fail"
    assert t.input_tokens == 2700, "the tokens the run burned were not reported"
    assert t.output_tokens == 120
    assert t.cache_read_tokens == 900
    assert t.total_tokens == 2820
    assert t.batch_call_count == 2
    assert t.model_id == "claude"
    assert t.execution_time_ms == 4200


async def test_a_failure_before_any_model_call_reports_zero() -> None:
    """Honest in the other direction too -- an unknown tenant costs nothing."""
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings())
    hold = await gate.reserve(_request(), tenant="foxtrot", trace_id="t")

    await gate.settle_failure(hold, execution_time_ms=90)

    t = licensing.settles[0]["telemetry"]
    assert (t.input_tokens, t.output_tokens, t.total_tokens) == (0, 0, 0)
    assert t.execution_time_ms == 90


def test_the_accumulator_resets_between_requests() -> None:
    """Otherwise one request is billed for the previous one's tokens."""
    from rag_agentic.converse import _record_spent

    _record_spent(Usage(input_tokens=500, output_tokens=10, llm_calls=1))
    assert spent_so_far().input_tokens == 500

    reset_spent()

    assert spent_so_far().input_tokens == 0
    assert spent_so_far().llm_calls == 0


def test_telemetry_from_spend_matches_a_successful_runs_shape() -> None:
    """Fail and Success rows must be comparable; the only difference is Result."""
    spent = Usage(input_tokens=100, output_tokens=20, cache_read_tokens=50,
                  cache_write_tokens=10, model="claude", llm_calls=3)

    t = RunTelemetry.failed(spent=spent, embedding_tokens=7, execution_time_ms=11)

    assert (t.input_tokens, t.output_tokens, t.total_tokens) == (100, 20, 120)
    assert (t.cache_read_tokens, t.cache_write_tokens) == (50, 10)
    assert t.embedding_tokens == 7
    assert t.batch_call_count == 3
