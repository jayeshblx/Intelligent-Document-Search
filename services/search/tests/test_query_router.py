"""Tests for LLM query routing — offline (the LLM is a stub).

The router replaces eight hand-written intent regexes, so what matters is not that it
parses happy-path JSON but that a *bad* plan cannot hurt: no invented filters, and no
outage when the model misbehaves.
"""

from __future__ import annotations

from typing import Any

from rag_search.query_router import QueryRouter

from rag_core.aws import GenerationResult

FACETS = {
    "created_by": ["storeuser1", "jane.doe@supplierco.example.com"],
    "business_object": ["COMPLAINTS", "SUPPLIERDOCUMENTS", "NFIF"],
    "document_class": ["certificate", "standard", "report"],
    "status": ["Draft", "Active", "Archive"],
}


class StubLLM:
    """Returns a canned reply; records the prompt it was given."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.seen: list[str] = []

    async def complete(self, messages, *, system, max_tokens, temperature, cache_system=False):
        self.seen.append(messages[-1]["content"][0]["text"])
        return GenerationResult(text=self.text, input_tokens=1, output_tokens=1, model="stub")


class BoomLLM:
    async def complete(self, *a: Any, **k: Any):
        raise RuntimeError("model down")


async def _route(text: str, question: str = "q", history: list[Any] | None = None):
    return await QueryRouter(StubLLM(text)).route(
        question, facet_values=FACETS, history=history or []
    )


# ---- the plan is honoured -------------------------------------------------------
async def test_routes_a_listing_with_a_real_filter() -> None:
    plan = await _route(
        '{"intent":"list","filters":{"created_by":"StoreUser1"},"expiry_from":null,'
        '"expiry_to":null,"continues_thread":false,"search_query":"documents by storeuser1"}'
    )
    assert plan is not None
    assert plan["intent"] == "list"
    # Matched case-insensitively but stored in the CORPUS's spelling, so the filter hits.
    assert plan["filters"] == {"created_by": "storeuser1"}
    assert plan["continues_thread"] is False


async def test_routes_content_questions_without_inventing_a_type_filter() -> None:
    """The 'Grade A BRC Global Standard' case: 'standard' is a name here, not a type."""
    plan = await _route(
        '{"intent":"content","filters":{},"expiry_from":null,"expiry_to":null,'
        '"continues_thread":false,"search_query":"Grade A BRC Global Standard"}'
    )
    assert plan is not None
    assert plan["intent"] == "content"
    assert plan["filters"] == {}


async def test_follow_up_is_rewritten_standalone() -> None:
    plan = await _route(
        '{"intent":"content","filters":{},"expiry_from":null,"expiry_to":null,'
        '"continues_thread":true,"search_query":"store for the Orange Soda claim 12345"}',
        question="find store for this complaint",
    )
    assert plan is not None
    assert plan["continues_thread"] is True
    assert "12345" in plan["search_query"]


# ---- a bad plan must not be able to hurt ----------------------------------------
async def test_invented_filter_values_are_dropped() -> None:
    """The safety property: a filter the model made up would show wrong documents as
    though they matched — worse than no filter at all."""
    plan = await _route(
        '{"intent":"list","filters":{"created_by":"nobody@example.com",'
        '"business_object":"COMPLAINTS"},"expiry_from":null,"expiry_to":null,'
        '"continues_thread":false,"search_query":"x"}'
    )
    assert plan is not None
    # The real value survives; the invented one is gone.
    assert plan["filters"] == {"business_object": "COMPLAINTS"}


async def test_unknown_facet_keys_are_dropped() -> None:
    plan = await _route(
        '{"intent":"list","filters":{"grade":"A","colour":"red"},"expiry_from":null,'
        '"expiry_to":null,"continues_thread":false,"search_query":"x"}'
    )
    assert plan is not None and plan["filters"] == {}


async def test_bad_intent_defaults_to_content() -> None:
    """Unknown intent must degrade to the safe mode — content still searches everything."""
    plan = await _route(
        '{"intent":"teleport","filters":{},"expiry_from":null,"expiry_to":null,'
        '"continues_thread":false,"search_query":"x"}'
    )
    assert plan is not None and plan["intent"] == "content"


async def test_malformed_dates_are_dropped() -> None:
    plan = await _route(
        '{"intent":"list","filters":{},"expiry_from":"last tuesday","expiry_to":"2026-07-17",'
        '"continues_thread":false,"search_query":"x"}'
    )
    assert plan is not None
    assert plan["expiry_from"] is None
    assert plan["expiry_to"] == "2026-07-17"


# ---- the router can never take the question down --------------------------------
async def test_model_failure_falls_back() -> None:
    plan = await QueryRouter(BoomLLM()).route("q", facet_values=FACETS, history=[])
    assert plan is None, "caller must fall back to its own logic, not error"


async def test_unparseable_reply_falls_back() -> None:
    assert await _route("I'm afraid I can't do that, Dave.") is None
    assert await _route("") is None


async def test_json_inside_prose_or_fences_is_recovered() -> None:
    """Models add fences and preamble; that must not cost us the plan."""
    plan = await _route(
        'Sure! Here you go:\n```json\n{"intent":"list","filters":{},"expiry_from":null,'
        '"expiry_to":null,"continues_thread":false,"search_query":"x"}\n```'
    )
    assert plan is not None and plan["intent"] == "list"


# ---- prompt construction ---------------------------------------------------------
async def test_prompt_carries_facet_keys_and_history_but_not_every_value() -> None:
    """13k object codes cannot go in a prompt: the model copies values from the question
    and validation makes them true."""
    llm = StubLLM('{"intent":"content","filters":{},"expiry_from":null,"expiry_to":null,'
                  '"continues_thread":false,"search_query":"x"}')
    hist = [type("M", (), {"role": "user", "content": "earlier question about Soda"})()]
    await QueryRouter(llm).route("q", facet_values=FACETS, history=hist)
    prompt = llm.seen[0]
    assert "created_by" in prompt and "business_object" in prompt
    assert "earlier question about Soda" in prompt
    assert "Question: q" in prompt
    # Only a couple of examples per facet, never the whole vocabulary.
    assert prompt.count("jane.doe@supplierco.example.com") <= 1
