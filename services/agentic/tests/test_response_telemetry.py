"""Per-request telemetry the caller logs, and history stored as QueryText/Response.

ExampleApp keeps its own request log, so every response carries the numbers that row
needs — tokens, model, which runtime answered, how long it took — flattened to the
top level rather than nested inside `usage`.

ExampleApp also stores a conversation turn as ONE row (`QueryText` + `Response`),
while Converse wants two messages. Both shapes must work.
"""

from __future__ import annotations

from typing import Any

import pytest
from rag_agentic.models import AgentRequest, AgentResponse, Usage, to_converse_history

# ---- telemetry ------------------------------------------------------------------

def test_total_is_input_plus_output() -> None:
    usage = Usage(input_tokens=1426, output_tokens=835, cache_read_tokens=47563)

    assert usage.total_tokens == 2261


def test_cache_reads_are_not_double_counted() -> None:
    """Bedrock already counts a cache read inside inputTokens; adding it again
    would overstate every cached turn, which is most of them."""
    usage = Usage(input_tokens=1000, output_tokens=100, cache_read_tokens=900)

    assert usage.total_tokens == 1100


def test_the_response_carries_the_logging_fields() -> None:
    response = AgentResponse(
        answer="x",
        input_tokens=1426,
        output_tokens=835,
        total_tokens=2261,
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        agent_id="arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/docsearch-abc",
        duration_ms=8312.4,
    )
    payload = response.model_dump(mode="json")

    for field in ("input_tokens", "output_tokens", "total_tokens",
                  "model_id", "agent_id", "duration_ms"):
        assert field in payload, f"{field} missing from the response"
    assert payload["total_tokens"] == 2261


def test_telemetry_defaults_are_harmless() -> None:
    """A response built without telemetry still serialises; the runtime fills it."""
    payload = AgentResponse(answer="x").model_dump(mode="json")

    assert payload["input_tokens"] == 0
    assert payload["duration_ms"] == 0.0
    assert payload["agent_id"] == ""


# ---- history stored as QueryText / Response --------------------------------------

PAIRED = [
    {
        "QueryText": "list all documents in draft status",
        "Response": "There are 15 documents in Draft status. All are INGREDIENTSPEC documents.",
    },
    {
        "QueryText": "can you explain first one form it",
        "Response": "I'd be happy to help explain a form, but I need a bit more information.",
    },
]


def test_a_paired_row_becomes_two_turns() -> None:
    request = AgentRequest.model_validate(
        {"BaseUrl": "https://bravo.acmedev.com/", "Query": "first document in the list",
         "History": PAIRED}
    )

    assert [t.role for t in request.history] == ["user", "assistant", "user", "assistant"]
    assert request.history[0].content.startswith("list all documents")
    assert request.history[1].content.startswith("There are 15 documents")


def test_paired_rows_convert_to_a_valid_transcript() -> None:
    request = AgentRequest.model_validate(
        {"BaseUrl": "b", "Query": "q", "History": PAIRED}
    )
    messages = to_converse_history(request.history)

    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]


def test_the_two_shapes_can_be_mixed() -> None:
    """A caller migrating from one shape to the other must not break mid-way."""
    request = AgentRequest.model_validate(
        {
            "BaseUrl": "b",
            "Query": "q",
            "History": [
                {"QueryText": "asked", "Response": "answered"},
                {"role": "user", "content": "asked again"},
                {"role": "assistant", "content": "answered again"},
            ],
        }
    )

    assert [t.content for t in request.history] == [
        "asked", "answered", "asked again", "answered again",
    ]


def test_a_row_with_only_a_question_is_kept() -> None:
    """An in-flight turn has no Response yet; it must not be dropped or invented."""
    request = AgentRequest.model_validate(
        {"BaseUrl": "b", "Query": "q", "History": [{"QueryText": "asked but unanswered"}]}
    )

    assert [t.role for t in request.history] == ["user"]


@pytest.mark.parametrize("key", ["QueryText", "queryText", "query", "question"])
def test_question_spellings_in_a_paired_row(key: str) -> None:
    request = AgentRequest.model_validate(
        {"BaseUrl": "b", "Query": "q", "History": [{key: "asked", "Response": "answered"}]}
    )

    assert request.history[0].content == "asked"


@pytest.mark.parametrize("key", ["Response", "response", "answer"])
def test_answer_spellings_in_a_paired_row(key: str) -> None:
    request = AgentRequest.model_validate(
        {"BaseUrl": "b", "Query": "q", "History": [{"QueryText": "asked", key: "answered"}]}
    )

    assert request.history[1].content == "answered"


def test_junk_entries_are_ignored_not_fatal() -> None:
    request = AgentRequest.model_validate(
        {"BaseUrl": "b", "Query": "q",
         "History": [None, "a string", {}, {"QueryText": "real", "Response": "turn"}]}
    )

    assert [t.content for t in request.history] == ["real", "turn"]


def test_history_is_still_optional() -> None:
    request: Any = AgentRequest.model_validate({"BaseUrl": "b", "Query": "q"})

    assert request.history == []
