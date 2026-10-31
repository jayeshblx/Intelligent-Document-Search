"""The Settle body Licensing receives.

Licensing bills from these numbers, so the shape is pinned rather than trusted:
PascalCase keys, the display name rather than the code in `Feature`, and none of
the fields the guide reserves for Licensing itself. A field we silently rename or
drop becomes a credit discrepancy nobody notices until a bill is wrong.
"""

from __future__ import annotations

from typing import Any

import pytest
from rag_agentic.licensing import (
    FEATURE_DISPLAY_NAMES,
    RunTelemetry,
    cited_object_codes,
    new_idempotency_key,
    settle_payload,
    strip_bearer,
)
from rag_agentic.models import AgentResponse, Citation, Usage

# Licensing sets these itself; sending them asserts something we have no say over.
FORBIDDEN = ("TokensBurnt", "CacheTokens", "EstimatedCredits", "LicenseCode", "ActualCredits")


def _telemetry(**over: Any) -> RunTelemetry:
    base: dict[str, Any] = {
        "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "input_tokens": 3322, "output_tokens": 51, "cache_read_tokens": 1955,
        "cache_write_tokens": 1955, "embedding_tokens": 7, "total_tokens": 3373,
        "execution_time_ms": 29760, "batch_call_count": 2, "agent_id": "orchestrator",
    }
    base.update(over)
    return RunTelemetry(**base)


def _payload(**over: Any) -> dict:
    args: dict[str, Any] = {
        "tenant_code": "FOXTROT", "reservation_id": "res-1", "transaction_id": "txn-1",
        "result": "Success", "feature_code": "DOCUMENT_SEARCH", "telemetry": _telemetry(),
    }
    args.update(over)
    return settle_payload(**args)


def test_required_success_fields_are_present() -> None:
    """Everything the guide marks "Yes on Success"."""
    payload = _payload()

    for field in ("TenantCode", "ReservationId", "TransactionId", "Result", "ModelId",
                  "InputTokens", "OutputTokens", "CacheReadTokens", "CacheWriteTokens",
                  "EmbeddingTokens"):
        assert field in payload, f"Settle requires {field}"


def test_reserved_fields_are_never_sent() -> None:
    payload = _payload()

    for field in FORBIDDEN:
        assert field not in payload, f"{field} is Licensing's to set, not ours"


def test_keys_are_pascal_case() -> None:
    """Our own models are snake_case; the wire is not."""
    assert all(k[:1].isupper() for k in _payload()), "Licensing reads PascalCase"


def test_token_counts_pass_through_unchanged() -> None:
    payload = _payload()

    assert payload["InputTokens"] == 3322
    assert payload["OutputTokens"] == 51
    assert payload["CacheReadTokens"] == 1955
    assert payload["CacheWriteTokens"] == 1955
    assert payload["EmbeddingTokens"] == 7
    assert payload["TotalTokens"] == 3373


def test_feature_is_the_display_name_not_the_code() -> None:
    assert _payload()["Feature"] == "Document Search"
    assert _payload(feature_code="OBJECT_COMPARE")["Feature"] == "Object Compare"


@pytest.mark.parametrize("code", sorted(FEATURE_DISPLAY_NAMES))
def test_every_locked_feature_code_has_a_display_name(code: str) -> None:
    assert _payload(feature_code=code)["Feature"] == FEATURE_DISPLAY_NAMES[code]


def test_execution_time_is_a_whole_number_of_milliseconds() -> None:
    """duration_ms is a float on our side; the guide types ExecutionTimeMs as int."""
    response = AgentResponse(answer="a", route="direct", environment="acmedev", tenant="foxtrot")
    response.duration_ms = 29760.4

    assert RunTelemetry.from_response(response).execution_time_ms == 29760


def test_sub_agents_carry_the_specialist_breakdown() -> None:
    telemetry = _telemetry(sub_agents=[
        Usage(input_tokens=2000, output_tokens=40, cache_read_tokens=1955,
              embedding_tokens=7, model="search-model"),
    ])

    [sub] = settle_payload(
        tenant_code="FOXTROT", reservation_id="r", transaction_id="t", result="Success",
        feature_code="DOCUMENT_SEARCH", telemetry=telemetry,
    )["SubAgents"]

    assert sub["ModelId"] == "search-model"
    assert sub["InputTokens"] == 2000
    assert sub["TotalTokens"] == 2040


def test_top_level_totals_are_the_run_not_the_router() -> None:
    """The guide: "Top-level token fields remain the totals for the run." The
    orchestrator already folds the specialist in, so the total must exceed any
    single sub-agent rather than double-count or under-report it."""
    response = AgentResponse(answer="a", route="document_search", environment="acmedev", tenant="foxtrot")
    response.usage = Usage(input_tokens=3322, output_tokens=51, model="router-model")
    response.sub_usage = [Usage(input_tokens=2000, output_tokens=40, model="search-model")]

    telemetry = RunTelemetry.from_response(response)

    assert telemetry.input_tokens == 3322
    assert sum(u.input_tokens for u in telemetry.sub_agents) <= telemetry.input_tokens


def test_user_prompt_is_omitted_unless_supplied() -> None:
    """It may carry PII, so it travels only when an operator asks for it."""
    assert "UserPrompt" not in _payload()
    assert _payload(user_prompt="how many complaints")["UserPrompt"] == "how many complaints"


def test_optional_fields_are_dropped_when_empty() -> None:
    """An empty TraceId is worse than no TraceId: it looks like a real value."""
    payload = _payload(telemetry=_telemetry(agent_id="", batch_call_count=0))

    assert "AgentId" not in payload
    assert "BatchCallCount" not in payload
    assert "TraceId" not in payload


def test_a_failed_run_reports_only_what_it_can_account_for() -> None:
    telemetry = RunTelemetry.failed(execution_time_ms=900)
    payload = settle_payload(
        tenant_code="FOXTROT", reservation_id="r", transaction_id="t", result="Fail",
        feature_code="DOCUMENT_SEARCH", telemetry=telemetry,
    )

    assert payload["Result"] == "Fail"
    assert payload["InputTokens"] == 0
    assert payload["ExecutionTimeMs"] == 900


def test_idempotency_keys_differ_per_attempt() -> None:
    """Reusing one would settle a second run against the first reservation."""
    assert len({new_idempotency_key() for _ in range(100)}) == 100


@pytest.mark.parametrize(
    ("given", "expected"),
    [("eyJabc", "eyJabc"), ("Bearer eyJabc", "eyJabc"), ("bearer eyJabc", "eyJabc"),
     ("  Bearer  eyJabc ", "eyJabc"), ("", ""), (None, "")],
)
def test_a_whole_authorization_header_is_accepted(given: str | None, expected: str) -> None:
    assert strip_bearer(given) == expected


# ---- which specialist ran ---------------------------------------------------------
#
# Every specialist runs on the same Claude model, so a SubAgents row carrying only
# a ModelId is unattributable: a credit record shows that a sub-agent spent 4,061
# tokens but not which one. The route name is that missing piece.


def test_each_sub_agent_row_names_the_specialist() -> None:
    telemetry = _telemetry(sub_agents=[
        Usage(input_tokens=2000, output_tokens=40, model="claude",
              agent_id="document_search"),
    ])

    [sub] = _payload(telemetry=telemetry)["SubAgents"]

    assert sub["AgentId"] == "document_search"
    assert sub["ModelId"] == "claude"


def test_parallel_specialists_stay_distinguishable() -> None:
    """Two rows with the same ModelId are only tellable apart by name."""
    telemetry = _telemetry(sub_agents=[
        Usage(input_tokens=2000, model="claude", agent_id="document_search"),
        Usage(input_tokens=900, model="claude", agent_id="template"),
    ])

    names = [s["AgentId"] for s in _payload(telemetry=telemetry)["SubAgents"]]

    assert names == ["document_search", "template"]


def test_an_unnamed_sub_agent_does_not_break_the_row() -> None:
    """Older callers send no name; the spend still has to settle."""
    telemetry = _telemetry(sub_agents=[Usage(input_tokens=2000, model="claude")])

    [sub] = _payload(telemetry=telemetry)["SubAgents"]

    assert sub["AgentId"] == ""
    assert sub["InputTokens"] == 2000


# ---- what the charge was for ------------------------------------------------------
#
# A credit record is read back long after the run, usually to answer "what was
# this spend for?". The codes the answer cited answer that. The request's ACL
# scope does not: it says what the user was permitted to see, and on BRAVO that is
# 4,790 codes -- a result set, not a billing field.


def test_the_cited_object_codes_are_settled() -> None:
    response = AgentResponse(answer="a", route="document_search",
                             environment="acmedev", tenant="bravo")
    response.citations = [
        Citation(document_id="d1", facets={"object_code": "PIF2763"}),
        Citation(document_id="d2", facets={"object_code": "PIF3136"}),
    ]

    telemetry = RunTelemetry.from_response(response)

    assert _payload(telemetry=telemetry)["ObjectCode"] == "PIF2763,PIF3136"


def test_repeated_codes_are_listed_once() -> None:
    """Several passages from one document are one document."""
    response = AgentResponse(answer="a", route="document_search",
                             environment="acmedev", tenant="bravo")
    response.citations = [
        Citation(document_id="d1", page=1, facets={"object_code": "PIF2763"}),
        Citation(document_id="d1", page=7, facets={"object_code": "PIF2763"}),
        Citation(document_id="d2", facets={"object_code": "PIF3136"}),
    ]

    assert cited_object_codes(response) == "PIF2763,PIF3136"


def test_the_list_is_bounded() -> None:
    """A settle body must not grow with the size of the result set."""
    response = AgentResponse(answer="a", route="document_search",
                             environment="acmedev", tenant="bravo")
    response.citations = [
        Citation(document_id=f"d{i}", facets={"object_code": f"PIF{i:04d}"})
        for i in range(500)
    ]

    assert len(cited_object_codes(response).split(",")) == 50


def test_an_answer_citing_nothing_sends_no_object_code() -> None:
    """A greeting cites nothing; an empty string would look like a real value."""
    response = AgentResponse(answer="hello", route="direct",
                             environment="acmedev", tenant="bravo")

    assert "ObjectCode" not in _payload(telemetry=RunTelemetry.from_response(response))


def test_citations_without_the_facet_are_skipped() -> None:
    response = AgentResponse(answer="a", route="document_search",
                             environment="acmedev", tenant="bravo")
    response.citations = [
        Citation(document_id="d1", facets={}),
        Citation(document_id="d2", facets={"object_code": "  PIF3136  "}),
    ]

    assert cited_object_codes(response) == "PIF3136"


# ---- what the caller was told -----------------------------------------------------
#
# The prompt alone says what was asked, not what the spend bought. Reading a record
# back to judge an answer needs both. It rides the same opt-in as the prompt,
# because an answer quotes the documents behind it.


def test_the_answer_is_settled_alongside_the_prompt() -> None:
    payload = _payload(user_prompt="how many are active?",
                       response_text="There are 1,582 documents in Active status.")

    assert payload["UserPrompt"] == "how many are active?"
    assert payload["Response"] == "There are 1,582 documents in Active status."


def test_a_long_answer_is_truncated_not_dropped() -> None:
    """A settle body is not a transcript store, but a blank field diagnoses nothing."""
    payload = _payload(response_text="x" * 10_000)

    assert len(payload["Response"]) == 4000


def test_no_answer_means_no_field() -> None:
    """A run that broke before answering has nothing to report here."""
    assert "Response" not in _payload()
