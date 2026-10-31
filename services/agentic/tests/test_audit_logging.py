"""Every step of a request is recorded, so a bad answer can be reconstructed.

When something goes wrong mid-request the question is always "what did the agent
actually do?" — which tools it called, with what arguments, what came back, and
how long each took. Failures were logged before this; SUCCESSES were not, and a
request that returns a confident wrong answer is the harder case to investigate.

Tool arguments are model-authored and can echo document text, so they are masked
with the same rule that masks answers.
"""

from __future__ import annotations

from typing import Any

import pytest
import structlog
from rag_agentic.agent_loop import ToolError, _audit_arguments, _run_tool


@pytest.fixture
def captured() -> Any:
    """Collect structlog events emitted during the test."""
    events: list[dict[str, Any]] = []

    def _sink(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        events.append(dict(event_dict))
        return event_dict

    original = structlog.get_config()["processors"]
    structlog.configure(processors=[_sink, *original[-1:]])
    try:
        yield events
    finally:
        structlog.configure(processors=original)


class _Toolbox:
    def __init__(self, fail: Exception | None = None) -> None:
        self._fail = fail

    async def invoke(self, name: str, arguments: dict[str, Any]) -> tuple[Any, str]:
        if self._fail:
            raise self._fail
        return {"rows": 3}, "3 document(s) matched"


async def test_a_successful_tool_call_is_logged(captured: list[dict[str, Any]]) -> None:
    call, _ = await _run_tool(_Toolbox(), "t1", "list_documents", {"status": "Draft"})

    logged = [e for e in captured if e.get("event") == "agent_tool_call"]
    assert logged, "a successful tool call must be recorded, not only failures"
    entry = logged[0]
    assert entry["tool"] == "list_documents"
    assert entry["status"] == "success"
    assert entry["arguments"] == {"status": "Draft"}
    assert "3 document(s) matched" in entry["summary"]
    assert entry["took_ms"] >= 0
    assert call.error is None


async def test_a_failed_tool_call_records_the_reason(captured: list[dict[str, Any]]) -> None:
    await _run_tool(_Toolbox(ToolError("unknown document 'x'")), "t1", "get_document_details",
                    {"document_id": "x"})

    entry = next(e for e in captured if e.get("event") == "agent_tool_call")
    assert entry["status"] == "error"
    assert "unknown document" in entry["error"]


async def test_the_arguments_are_recorded_for_every_call(captured: list[dict[str, Any]]) -> None:
    """Without the arguments, the log says a search happened but not what for."""
    await _run_tool(_Toolbox(), "t1", "search_documents",
                    {"query": "allergen statement", "top_k": 15, "status": "Active"})

    entry = next(e for e in captured if e.get("event") == "agent_tool_call")
    assert entry["arguments"]["query"] == "allergen statement"
    assert entry["arguments"]["top_k"] == 15
    assert entry["arguments"]["status"] == "Active"


# ---- masking --------------------------------------------------------------------

def test_secrets_in_arguments_are_masked() -> None:
    """A model can echo document text into a query, and a document can hold a key."""
    masked = _audit_arguments({"query": "the key is AKIAIOSFODNN7EXAMPLE"})

    assert "AKIAIOSFODNN7EXAMPLE" not in masked["query"]
    assert "[redacted-aws-key]" in masked["query"]


def test_business_values_survive_masking() -> None:
    """An audit log is useless if it scrubs the identifiers you search it by."""
    args = {"object_code": "INGR00107", "supplier": "SUPPLIER0029",
            "created_by": "riley@example.com"}

    assert _audit_arguments(args) == args


def test_non_string_arguments_pass_through() -> None:
    assert _audit_arguments({"top_k": 20, "offset": 0, "flag": True}) == {
        "top_k": 20, "offset": 0, "flag": True
    }


def test_a_very_long_argument_is_truncated() -> None:
    """A pasted document should not fill the log group."""
    masked = _audit_arguments({"query": "x" * 5000})

    assert len(masked["query"]) <= 400
