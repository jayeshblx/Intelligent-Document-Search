"""Tests for per-answer feedback: separate likes/dislikes JSONL + /v1/feedback."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from httpx import ASGITransport, AsyncClient
from rag_search.app import create_app
from rag_search.feedback import FeedbackStore


def _lines(path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]


async def test_feedback_store_routes_likes_and_dislikes(tmp_path) -> None:
    store = FeedbackStore(str(tmp_path / "likes.jsonl"), str(tmp_path / "dislikes.jsonl"))
    await store.record({"rating": "up", "question": "q1"})
    await store.record({"rating": "down", "question": "q2", "comment": "wrong"})
    likes, dislikes = _lines(store.positive_path), _lines(store.negative_path)
    assert len(likes) == 1 and likes[0]["rating"] == "up"
    assert len(dislikes) == 1 and dislikes[0]["comment"] == "wrong"


def _make_app(tmp_path) -> tuple[Any, FeedbackStore]:
    app = create_app()
    store = FeedbackStore(str(tmp_path / "likes.jsonl"), str(tmp_path / "dislikes.jsonl"))
    app.state.components = SimpleNamespace(feedback=store)
    return app, store


async def test_feedback_endpoint_separates_and_requires_comment_on_down(tmp_path) -> None:
    app, store = _make_app(tmp_path)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # 👍 — no comment needed, goes to the likes log.
        up = await c.post(
            "/v1/feedback", json={"rating": "up", "question": "q", "answer": "a"},
            headers={"X-User-Id": "u1"},
        )
        assert up.status_code == 200
        # 👎 without a comment — rejected.
        bad = await c.post(
            "/v1/feedback", json={"rating": "down", "question": "q", "answer": "a"},
            headers={"X-User-Id": "u1"},
        )
        assert bad.status_code == 422
        # 👎 with a comment — goes to the dislikes log.
        down = await c.post(
            "/v1/feedback",
            json={"rating": "down", "question": "q", "answer": "a", "comment": "too vague"},
            headers={"X-User-Id": "u1"},
        )
        assert down.status_code == 200
        # Invalid rating rejected.
        meh = await c.post("/v1/feedback", json={"rating": "meh"}, headers={"X-User-Id": "u1"})
        assert meh.status_code == 422

    likes, dislikes = _lines(store.positive_path), _lines(store.negative_path)
    assert len(likes) == 1 and "comment" not in likes[0]  # 👍 carries no comment
    assert len(dislikes) == 1
    assert dislikes[0]["comment"] == "too vague" and dislikes[0]["actor"] == "u1"
