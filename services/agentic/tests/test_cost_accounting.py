"""Every billable quantity in one request is reported, separately.

ExampleApp prices requests from these numbers, and the pieces are billed at
different rates: a cached read is discounted, a cache write costs a premium, and
query embeddings are a different model entirely (Titan, not Claude). Reporting
one lumped figure would make a request impossible to cost.

Query embeddings only — the bulk embeddings of an index run are a batch job cost,
not something a caller pays per question.
"""

from __future__ import annotations

import asyncio

import pytest
from rag_agentic.models import AgentResponse, Usage
from rag_search.embedder import (
    _record_query_embedding,
    query_embedding_tokens,
    reset_query_embedding_tokens,
)

# ---- the fields exist and are separate ------------------------------------------

def test_the_response_reports_every_billable_quantity() -> None:
    payload = AgentResponse(
        answer="x",
        input_tokens=76487,
        output_tokens=904,
        total_tokens=77391,
        cache_read_tokens=47563,
        cache_write_tokens=2826,
        embedding_tokens=38,
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    ).model_dump(mode="json")

    for field in ("input_tokens", "output_tokens", "total_tokens",
                  "cache_read_tokens", "cache_write_tokens", "embedding_tokens",
                  "model_id", "agent_id", "duration_ms"):
        assert field in payload, f"{field} missing — a caller cannot price the request"


def test_cache_reads_are_reported_but_not_added_to_the_total() -> None:
    """Bedrock already counts a cache read inside inputTokens. Adding it again
    would overstate every cached turn — and since prompt caching landed that is
    most of them."""
    usage = Usage(input_tokens=1426, output_tokens=835, cache_read_tokens=47563)

    assert usage.total_tokens == 2261
    assert usage.cache_read_tokens == 47563  # reported, for pricing the discount


def test_embedding_tokens_are_kept_apart_from_model_tokens() -> None:
    """Titan and Claude are billed differently, so they must not be merged."""
    usage = Usage(input_tokens=1000, output_tokens=100, embedding_tokens=42)

    assert usage.total_tokens == 1100, "embedding tokens must not inflate the model total"
    assert usage.embedding_tokens == 42


def test_summing_two_agents_keeps_every_field() -> None:
    """The orchestrator reports one row covering its own spend and the specialist's."""
    router = Usage(input_tokens=1200, output_tokens=90, cache_read_tokens=800,
                   cache_write_tokens=0, embedding_tokens=0, model="m")
    specialist = Usage(input_tokens=47000, output_tokens=850, cache_read_tokens=43000,
                       cache_write_tokens=2800, embedding_tokens=57, model="m")

    combined = router.add(specialist)

    assert combined.input_tokens == 48200
    assert combined.output_tokens == 940
    assert combined.cache_read_tokens == 43800
    assert combined.cache_write_tokens == 2800
    assert combined.embedding_tokens == 57
    assert combined.total_tokens == 49140


# ---- the query-embedding counter -------------------------------------------------

def test_the_counter_starts_at_zero_for_each_request() -> None:
    _record_query_embedding(11)
    reset_query_embedding_tokens()

    assert query_embedding_tokens() == 0


def test_each_search_adds_to_the_request_total() -> None:
    """An agent runs several searches per answer; each embeds its own query."""
    reset_query_embedding_tokens()
    for tokens in (12, 9, 14):
        _record_query_embedding(tokens)

    assert query_embedding_tokens() == 35


def test_concurrent_requests_do_not_share_a_count() -> None:
    """One runtime serves several requests at once; a contextvar keeps them apart,
    where a module-level counter would bill one caller for another's searches."""

    async def _request(tokens: int) -> int:
        reset_query_embedding_tokens()
        _record_query_embedding(tokens)
        await asyncio.sleep(0)  # let the other request interleave
        _record_query_embedding(tokens)
        return query_embedding_tokens()

    async def _both() -> tuple[int, int]:
        return await asyncio.gather(_request(10), _request(500))  # type: ignore[return-value]

    first, second = asyncio.run(_both())
    assert (first, second) == (20, 1000)


@pytest.mark.parametrize("bad", [-5, 0])
def test_a_missing_token_count_does_not_corrupt_the_total(bad: int) -> None:
    """Titan reports inputTextTokenCount; a response without it must not subtract."""
    reset_query_embedding_tokens()
    _record_query_embedding(20)
    _record_query_embedding(bad)

    assert query_embedding_tokens() == 20
