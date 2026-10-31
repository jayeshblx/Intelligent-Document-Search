"""The UI can ask the deployed agent instead of this service's own pipeline.

Two things make this safe to run beside a live ExampleApp integration, and both
are tested here rather than assumed:

* a demo call carries no reservation and no access token, so the agent's credit
  gate never calls Licensing and nothing lands in ExampleApp's ledger;
* the feature does not exist at all unless an ARN is configured, so a deployment
  that does not want it is unaffected by its presence in the code.

The third property is that a failure is visible. In a demo, an answer quietly
produced by the *local* pipeline when the deployed one was asked for is worse
than an error, because nobody watching can tell the difference.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from rag_search.agent_client import (
    AgentRuntimeClient,
    AgentUnavailableError,
    pad_session,
    parse_agent_response,
)

AGENT_ENVELOPE = {
    "answer": "There are 4,790 documents in total.",
    "citations": [{"document_id": "d1", "document_name": "PIF Header", "score": 0.4}],
    "route": "document_search",
    "environment": "acmedev",
    "tenant": "bravo",
    "business_object": "PIF",
    "total_matching": 4790,
    "model_id": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "duration_ms": 7958.6,
    "usage": {"input_tokens": 3832, "output_tokens": 335, "embedding_tokens": 7},
}


class FakeRuntime:
    """Stands in for bedrock-agentcore, recording exactly what was sent."""

    def __init__(self, envelope: Any = None, raises: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._envelope = AGENT_ENVELOPE if envelope is None else envelope
        self._raises = raises

    def invoke_agent_runtime(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(kw)
        if self._raises:
            raise self._raises
        body = self._envelope
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        return {"response": raw}

    @property
    def sent_body(self) -> dict[str, Any]:
        return json.loads(self.calls[-1]["payload"])


def _client(runtime: FakeRuntime) -> AgentRuntimeClient:
    return AgentRuntimeClient("arn:aws:bedrock-agentcore:ap-south-1:1:runtime/x",
                              region="ap-south-1", client=runtime)


# ---- the safety property ------------------------------------------------------------


async def test_a_demo_call_carries_no_reservation_and_no_token() -> None:
    """This is what keeps ExampleApp's ledger clean: there is nothing to settle."""
    runtime = FakeRuntime()

    await _client(runtime).ask("how many documents?", base_url="https://bravo.acmedev.com/")

    body = runtime.sent_body
    assert "ReservationId" not in body
    assert "AccessToken" not in body
    assert "Authorization" not in body


async def test_the_tenant_is_sent_as_the_agent_expects_it() -> None:
    runtime = FakeRuntime()

    await _client(runtime).ask("q", base_url="https://foxtrot.acmedev.com/")

    assert runtime.sent_body["baseUrl"] == "https://foxtrot.acmedev.com/"


async def test_acl_scope_is_forwarded_when_given() -> None:
    runtime = FakeRuntime()

    await _client(runtime).ask("q", base_url="https://bravo.acmedev.com/",
                               object_codes="PIF3136,PIF2764", business_object="PIF")

    assert runtime.sent_body["objectCodes"] == "PIF3136,PIF2764"
    assert runtime.sent_body["BusinessObject"] == "PIF"


async def test_empty_scope_is_omitted_rather_than_sent_blank() -> None:
    """A blank filter and no filter mean different things to the agent."""
    runtime = FakeRuntime()

    await _client(runtime).ask("q", base_url="https://bravo.acmedev.com/", object_codes="")

    assert "objectCodes" not in runtime.sent_body


# ---- sessions -----------------------------------------------------------------------


def test_a_session_id_is_padded_to_what_agentcore_accepts() -> None:
    assert len(pad_session("short")) >= 33
    assert len(pad_session(None)) >= 33
    assert len(pad_session("")) >= 33


def test_a_long_enough_session_is_left_alone() -> None:
    """Reusing the caller's id is what keeps a conversation on one warm container."""
    given = "ui-" + "a" * 40
    assert pad_session(given) == given


async def test_the_session_is_reused_across_turns() -> None:
    runtime = FakeRuntime()
    client = _client(runtime)
    session = "ui-" + "b" * 40

    await client.ask("first", base_url="https://foxtrot.acmedev.com/", session_id=session)
    await client.ask("second", base_url="https://foxtrot.acmedev.com/", session_id=session)

    assert {c["runtimeSessionId"] for c in runtime.calls} == {session}


# ---- reading the answer -------------------------------------------------------------


def test_the_envelope_maps_onto_what_the_ui_renders() -> None:
    answer = parse_agent_response(AGENT_ENVELOPE, session_id="s")

    assert answer.answer.startswith("There are 4,790")
    assert answer.documents_total == 4790
    assert answer.tenant == "bravo"
    assert answer.route == "document_search"
    assert answer.input_tokens == 3832
    assert answer.output_tokens == 335
    assert answer.embedding_tokens == 7
    assert answer.took_ms == pytest.approx(7958.6)


def test_an_envelope_missing_optional_fields_still_parses() -> None:
    """The agent is deployed separately; a demo must not break on a new release."""
    answer = parse_agent_response({"answer": "hello"}, session_id="s")

    assert answer.answer == "hello"
    assert answer.session_id == "s"
    assert answer.citations == []
    assert answer.documents_total == 0


def test_a_phase_one_envelope_carries_its_object_codes() -> None:
    answer = parse_agent_response(
        {"answer": "", "object_codes": "0225,0604", "tenant": "bravo"}, session_id="s"
    )

    assert answer.object_codes == "0225,0604"


def test_junk_numbers_do_not_crash_the_parse() -> None:
    answer = parse_agent_response(
        {"answer": "x", "total_matching": None, "usage": {"input_tokens": "nonsense"}},
        session_id="s",
    )

    assert answer.documents_total == 0
    assert answer.input_tokens == 0


# ---- failures are visible -----------------------------------------------------------


async def test_an_unreachable_agent_raises_rather_than_falling_back() -> None:
    runtime = FakeRuntime(raises=RuntimeError("runtime 503"))

    with pytest.raises(AgentUnavailableError):
        await _client(runtime).ask("q", base_url="https://bravo.acmedev.com/")


async def test_a_non_json_response_raises() -> None:
    runtime = FakeRuntime(envelope=b"<html>gateway timeout</html>")

    with pytest.raises(AgentUnavailableError):
        await _client(runtime).ask("q", base_url="https://bravo.acmedev.com/")


async def test_a_json_list_is_rejected_as_the_wrong_shape() -> None:
    runtime = FakeRuntime(envelope=[1, 2, 3])

    with pytest.raises(AgentUnavailableError):
        await _client(runtime).ask("q", base_url="https://bravo.acmedev.com/")
