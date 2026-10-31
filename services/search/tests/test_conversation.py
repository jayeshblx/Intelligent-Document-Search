"""Tests for the conversation store: durability window, ttl, reset, token budget,
rolling summary, and semantic recall. The store is async and in-memory here (no Mongo
backend) — persistence is exercised separately."""

from __future__ import annotations

import time

from rag_search.conversation import ConversationStore


async def test_new_session_and_recent() -> None:
    store = ConversationStore(ttl_minutes=30)
    sid = store.new_session()
    assert await store.recent(sid) == []
    await store.append(sid, "user", "hello")
    await store.append(sid, "assistant", "hi there")
    hist = await store.recent(sid)
    assert [m.role for m in hist] == ["user", "assistant"]
    assert hist[0].content == "hello"


async def test_context_refresh_after_ttl() -> None:
    store = ConversationStore(ttl_minutes=0.02 / 60)  # ttl ~0.02s
    sid = store.new_session()
    await store.append(sid, "user", "remember this")
    assert await store.recent(sid)  # still within window
    time.sleep(0.05)
    assert await store.recent(sid) == []  # window elapsed -> context cleared


async def test_no_expiry_keeps_context_until_new_chat() -> None:
    store = ConversationStore(ttl_minutes=0)  # 0 => never auto-expire
    sid = store.new_session()
    await store.append(sid, "user", "remember this")
    # Simulate a long idle gap — context must still be there (no time-based expiry).
    store._cache[sid].last_activity = time.time() - 100_000
    assert [m.content for m in await store.recent(sid)] == ["remember this"]
    assert await store.seconds_remaining(sid) == -1.0  # sentinel: expiry disabled
    # Only an explicit reset ("New chat" mints a new session) clears it.
    await store.reset(sid)
    assert await store.recent(sid) == []


async def test_reset_clears_context_keeps_id() -> None:
    store = ConversationStore(ttl_minutes=30)
    sid = store.new_session()
    await store.append(sid, "user", "x")
    await store.reset(sid)
    assert await store.recent(sid) == []


async def test_recent_window_is_token_budgeted() -> None:
    # A tiny budget keeps only the most recent turns that fit — but always at least one.
    store = ConversationStore(ttl_minutes=30, token_budget=10)
    sid = store.new_session()
    for i in range(10):
        await store.append(sid, "user" if i % 2 == 0 else "assistant", f"message number {i}")
    window = await store.recent(sid)
    assert 0 < len(window) < 10           # trimmed by the budget
    assert window[-1].content.endswith("9")  # newest turn is always kept


async def test_rolling_summary_compresses_oldest_turns() -> None:
    class _FakeLLM:
        calls = 0

        async def complete(self, messages, *, system, max_tokens, temperature, cache_system=False):
            _FakeLLM.calls += 1
            from rag_core.aws.bedrock import GenerationResult

            return GenerationResult(text="SUMMARY", input_tokens=50, output_tokens=10)

    # cap=4, batch=2 → once the 5th message lands, the oldest 2 are summarized away.
    store = ConversationStore(
        ttl_minutes=0, hard_message_cap=4, summarize_batch=2, semantic_recall=False,
        llm=_FakeLLM(),
    )
    sid = store.new_session()
    for i in range(6):
        await store.append(sid, "user" if i % 2 == 0 else "assistant", f"turn {i}")
    conv = store._cache[sid]
    assert conv.summary == "SUMMARY"        # oldest turns rolled into the summary
    assert len(conv.messages) <= 4          # retained set is bounded by the cap
    assert _FakeLLM.calls >= 1


async def test_semantic_recall_pulls_relevant_old_turn() -> None:
    # Fake embedder: deterministic 1-hot-ish vectors so cosine picks the matching turn.
    class _FakeEmbedder:
        dim = 3

        async def embed_query(self, text: str):
            if "halal" in text.lower():
                return [1.0, 0.0, 0.0]
            if "kosher" in text.lower():
                return [0.0, 1.0, 0.0]
            return [0.0, 0.0, 1.0]

    store = ConversationStore(
        ttl_minutes=0, token_budget=1, semantic_recall=True, recall_k=1,
        embedder=_FakeEmbedder(),
    )
    sid = store.new_session()
    await store.append(sid, "user", "tell me about the halal certificate")
    await store.append(sid, "assistant", "the halal cert expires next year")
    await store.append(sid, "user", "tell me about the kosher certificate")
    await store.append(sid, "assistant", "the kosher cert is current")
    # token_budget=1 pushes everything out of the recent window; recall must surface the
    # halal turn for a halal question even though it's old.
    bundle, _usage = await store.context(sid, "what about halal?")
    recalled_text = " ".join(m.content for m in bundle.recalled)
    assert "halal" in recalled_text.lower()
