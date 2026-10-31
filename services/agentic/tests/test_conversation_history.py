"""Follow-up questions: the agent is stateless, so the caller sends the history.

Without it, "can you explain it" arrives as four words with no referent and the
agent asks what the user means -- correct given what it was told, and useless to
the person typing.

Bedrock requires strictly alternating roles starting with `user`, and the current
question is appended as a user turn afterwards. A caller's transcript will not
always satisfy that, and a malformed one is a 400 from Bedrock -- a failed answer.
So most of these pin the normalisation.
"""

from __future__ import annotations

from itertools import pairwise

import pytest
from rag_agentic.models import HISTORY_CHAR_BUDGET, AgentRequest, to_converse_history


def _turns(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"role": r, "content": c} for r, c in pairs]


def _roles(messages: list[dict]) -> list[str]:
    return [m["role"] for m in messages]


def _texts(messages: list[dict]) -> list[str]:
    return [m["content"][0]["text"] for m in messages]


@pytest.mark.parametrize("spelling", ["History", "history", "messages", "conversationHistory"])
def test_history_spellings(spelling: str) -> None:
    request = AgentRequest.model_validate(
        {
            "BaseUrl": "https://bravo.acmedev.com/",
            "Query": "can you explain it",
            spelling: _turns(("user", "which documents are for INGR00107"), ("assistant", "One.")),
        }
    )
    assert len(request.history) == 2


def test_the_real_conversation_round_trips() -> None:
    request = AgentRequest.model_validate(
        {
            "BaseUrl": "https://bravo.acmedev.com/",
            "Query": "can you explain it",
            "History": _turns(
                ("user", "which documents are associated with INGR00107"),
                ("assistant", "One document is associated with INGR00107: ValidationLists."),
            ),
        }
    )
    messages = to_converse_history(request.history)

    assert _roles(messages) == ["user", "assistant"]
    assert "INGR00107" in _texts(messages)[0]


def test_absent_history_is_the_old_behaviour() -> None:
    request = AgentRequest.model_validate(
        {"BaseUrl": "https://bravo.acmedev.com/", "Query": "hello"}
    )
    assert request.history == []
    assert to_converse_history(request.history) == []


# ---- normalisation: a malformed transcript must not become a Bedrock 400 --------

def test_consecutive_same_role_turns_are_merged() -> None:
    messages = to_converse_history(
        AgentRequest.model_validate(
            {"BaseUrl": "b", "Query": "q",
             "History": _turns(("user", "first"), ("user", "second"), ("assistant", "reply"))}
        ).history
    )
    assert _roles(messages) == ["user", "assistant"]
    assert _texts(messages)[0] == "first\n\nsecond"


def test_a_leading_assistant_turn_is_dropped() -> None:
    """Converse must start with a user turn."""
    messages = to_converse_history(
        AgentRequest.model_validate(
            {"BaseUrl": "b", "Query": "q",
             "History": _turns(("assistant", "greeting"), ("user", "ask"), ("assistant", "answer"))}
        ).history
    )
    assert _roles(messages) == ["user", "assistant"]


def test_a_trailing_user_turn_is_dropped() -> None:
    """The current Query is appended as the user turn; two in a row is invalid."""
    messages = to_converse_history(
        AgentRequest.model_validate(
            {"BaseUrl": "b", "Query": "q",
             "History": _turns(("user", "ask"), ("assistant", "answer"), ("user", "stray"))}
        ).history
    )
    assert _roles(messages) == ["user", "assistant"]
    assert "stray" not in " ".join(_texts(messages))


def test_roles_always_alternate() -> None:
    messy = _turns(
        ("assistant", "a1"), ("assistant", "a2"), ("user", "u1"),
        ("user", "u2"), ("assistant", "a3"), ("user", "u3"),
    )
    messages = to_converse_history(
        AgentRequest.model_validate({"BaseUrl": "b", "Query": "q", "History": messy}).history
    )
    roles = _roles(messages)
    assert roles[0] == "user" if roles else True
    assert all(a != b for a, b in pairwise(roles))


def test_empty_turns_are_ignored() -> None:
    messages = to_converse_history(
        AgentRequest.model_validate(
            {"BaseUrl": "b", "Query": "q",
             "History": _turns(("user", "ask"), ("assistant", "   "), ("assistant", "real"))}
        ).history
    )
    assert _texts(messages) == ["ask", "real"]


def test_an_unknown_role_is_treated_as_user() -> None:
    messages = to_converse_history(
        AgentRequest.model_validate(
            {"BaseUrl": "b", "Query": "q",
             "History": _turns(("system", "something"), ("assistant", "answer"))}
        ).history
    )
    assert _roles(messages) == ["user", "assistant"]


# ---- budget ---------------------------------------------------------------------

def test_the_oldest_context_is_dropped_first() -> None:
    """History is re-read on every model round, so it is capped -- and what survives
    must be the RECENT turns, which is what a follow-up refers to."""
    big = "x" * 4000
    history = _turns(
        ("user", "oldest " + big), ("assistant", big),
        ("user", "middle " + big), ("assistant", big),
        ("user", "newest question"), ("assistant", "newest answer"),
    )
    messages = to_converse_history(
        AgentRequest.model_validate({"BaseUrl": "b", "Query": "q", "History": history}).history
    )
    joined = " ".join(_texts(messages))

    assert "newest answer" in joined
    assert "oldest" not in joined
    assert sum(len(t) for t in _texts(messages)) <= HISTORY_CHAR_BUDGET


def test_a_trimmed_transcript_is_still_valid() -> None:
    """Trimming must not leave a leading assistant turn behind."""
    history = _turns(("user", "x" * 11_000), ("assistant", "y" * 11_000), ("user", "u"), ("assistant", "a"))
    messages = to_converse_history(
        AgentRequest.model_validate({"BaseUrl": "b", "Query": "q", "History": history}).history
    )
    assert not messages or _roles(messages)[0] == "user"


@pytest.mark.parametrize("absent", [None, [], "omitted"])
def test_history_is_optional_in_every_form(absent: object) -> None:
    """The FIRST message of a conversation has no history by definition, and a
    .NET caller serialises an empty nullable list as null -- so null must mean
    "none", not a 400 on the one request guaranteed to have no history."""
    payload = {"BaseUrl": "https://bravo.acmedev.com/", "Query": "list all documents"}
    if absent != "omitted":
        payload["History"] = absent  # type: ignore[assignment]

    request = AgentRequest.model_validate(payload)

    assert request.history == []
    assert to_converse_history(request.history) == []

