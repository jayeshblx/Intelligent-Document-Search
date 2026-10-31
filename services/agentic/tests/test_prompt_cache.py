"""Transcript caching: the agent loop re-sends a transcript that grows every round.

Measured on the deployed agent: one document question reached 76,487 input tokens
across five rounds with only 10,482 served from cache, because just the system
prompt was cached. These pin the marker's placement, since a misplaced cache point
either silently does nothing or breaks the request.
"""

from __future__ import annotations

from typing import Any

from rag_agentic.converse import _CACHE_FLOOR_CHARS, _with_cache_point


def _transcript(rounds: int = 1, *, size: int = 40) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": [{"text": "q" * size}]}]
    for _ in range(rounds):
        messages.append({"role": "assistant", "content": [{"text": "a" * size}]})
        messages.append({"role": "user", "content": [{"text": "r" * size}]})
    return messages


def test_marker_goes_at_the_end_of_the_transcript() -> None:
    """Marking the end is what makes the NEXT round read all of it at cache rate."""
    marked = _with_cache_point(_transcript())

    assert marked[-1]["content"][-1] == {"cachePoint": {"type": "default"}}
    # Exactly one, and only on the last message.
    points = [b for m in marked for b in m["content"] if "cachePoint" in b]
    assert len(points) == 1


def test_original_transcript_is_never_mutated() -> None:
    """The loop reuses its list across rounds; a marker belongs to one request."""
    messages = _transcript(rounds=2)
    before = [len(m["content"]) for m in messages]

    _with_cache_point(messages)

    assert [len(m["content"]) for m in messages] == before
    assert not any("cachePoint" in b for m in messages for b in m["content"])


def test_existing_content_is_preserved() -> None:
    messages = [{"role": "user", "content": [{"text": "keep me"}, {"text": "and me"}]}]
    marked = _with_cache_point(messages)

    assert [b for b in marked[-1]["content"] if "text" in b] == [
        {"text": "keep me"},
        {"text": "and me"},
    ]


def test_a_second_marker_is_not_added() -> None:
    once = _with_cache_point(_transcript())
    twice = _with_cache_point(once)

    assert twice == once


def test_empty_transcript_is_left_alone() -> None:
    assert _with_cache_point([]) == []
    empty_content = [{"role": "user", "content": []}]
    assert _with_cache_point(empty_content) == empty_content


async def test_short_transcripts_are_not_marked() -> None:
    """Below the model's minimum cacheable size Bedrock ignores the point anyway;
    not sending one keeps the request identical to what it used to be."""
    from rag_agentic.converse import BedrockConverseModel

    sent: dict[str, Any] = {}

    class _Client:
        async def converse(self, **kwargs: Any) -> dict[str, Any]:
            sent.update(kwargs)
            return {"output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}}}

    model = BedrockConverseModel.__new__(BedrockConverseModel)
    model._client = _Client()  # type: ignore[attr-defined]
    model._cm = None  # type: ignore[attr-defined]
    model._model_id = "test-model"  # type: ignore[attr-defined]
    model._guardrail = None  # type: ignore[attr-defined]

    async def _ensure() -> Any:
        return model._client  # type: ignore[attr-defined]

    model._ensure = _ensure  # type: ignore[attr-defined]

    class _Breaker:
        async def call(self, fn: Any) -> Any:
            return await fn()

    model._breaker = _Breaker()  # type: ignore[attr-defined]

    short = _transcript(size=10)
    assert sum(len(str(m)) for m in short) < _CACHE_FLOOR_CHARS
    await model.converse(short, system="s", cache_messages=True)

    assert not any(
        "cachePoint" in block for m in sent["messages"] for block in m.get("content", [])
    )
