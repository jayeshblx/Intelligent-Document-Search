"""Phase 1 must return EVERY object code, not a page of them.

ExampleApp filters the returned list against the user's permissions and sends the
survivors back. So a truncated list is not a display problem -- every document
under a missing code becomes permanently invisible, with no error anywhere.

`facets()` caps each key at 200 values by default, which is right for a UI
dropdown and silently wrong here: FOXTROT has 323 object codes and BRAVO 413, so the
default was already cutting both when this was first deployed.
"""

from __future__ import annotations

from typing import Any

import pytest
from rag_agentic.agents.search_agent import DocumentSearchAgent
from rag_agentic.config import AgenticSettings
from rag_agentic.models import AgentRequest
from rag_agentic.tenancy import TenantContext


class _RecordingMetadata:
    """A metadata store that reports how many values it was asked for."""

    def __init__(self, codes: list[str]) -> None:
        self._codes = codes
        self.asked_for: str | None = None

    async def facets(
        self, namespace: str = "default", *, filters: Any = None, per_facet_limit: int = 200
    ) -> dict[str, list[str]]:
        raise AssertionError("phase 1 must use facet_values(), which does not paginate")

    async def facet_values(
        self, key: str, namespace: str = "default", *, filters: Any = None
    ) -> list[str]:
        self.asked_for = key
        return sorted(self._codes)


class _StubIndex:
    def __init__(self, metadata: _RecordingMetadata) -> None:
        self.metadata = metadata
        self.namespace = "default"
        self.manifest: dict[str, Any] = {"tenant_codes": ["FOXTROT"]}
        self.context = TenantContext(
            environment="acmedev", tenant="foxtrot", base_url="https://foxtrot.acmedev.com/"
        )


class _StubStore:
    def __init__(self, index: _StubIndex) -> None:
        self._index = index
        self.prewarmed: list[str] = []

    async def acquire(self, context: TenantContext) -> _StubIndex:
        return self._index

    def prewarm(self, context: TenantContext) -> None:
        # Phase 1 kicks off a background probe of the search index; recorded
        # rather than run, so these tests stay about the object codes.
        self.prewarmed.append(context.key)


def _agent(index: _StubIndex) -> DocumentSearchAgent:
    return DocumentSearchAgent(
        settings=AgenticSettings(index_bucket="b", embed_provider="fake", embed_dim=256),
        model=object(),  # type: ignore[arg-type]
        embedder=object(),  # type: ignore[arg-type]
        index_store=_StubStore(index),  # type: ignore[arg-type]
        resolver=object(),  # type: ignore[arg-type]
    )


def _request() -> AgentRequest:
    return AgentRequest.model_validate(
        {"BaseUrl": "https://foxtrot.acmedev.com/", "Operation": "objectCodes", "UserCode": "u"}
    )


async def test_more_than_two_hundred_codes_are_all_returned() -> None:
    """The exact bug: 323 codes came back as 200."""
    codes = [f"CMP{i:04d}" for i in range(323)]
    index = _StubIndex(_RecordingMetadata(codes))

    response = await _agent(index)._object_codes(_request(), index.context)

    returned = response.object_codes.split(",")
    assert len(returned) == 323, f"truncated to {len(returned)}"
    assert response.total_matching == 323


async def test_the_paginated_facet_call_is_not_used() -> None:
    """facets() pages at 200; phase 1 must use the unpaginated lookup instead."""
    index = _StubIndex(_RecordingMetadata([f"C{i}" for i in range(10)]))

    await _agent(index)._object_codes(_request(), index.context)

    assert index.metadata.asked_for == "object_code"


async def test_codes_come_back_sorted_and_comma_separated() -> None:
    index = _StubIndex(_RecordingMetadata(["ZZ-9", "AA-1", "MM-5"]))

    response = await _agent(index)._object_codes(_request(), index.context)

    assert response.object_codes == "AA-1,MM-5,ZZ-9"


async def test_a_tenant_with_no_codes_returns_empty_not_an_error() -> None:
    index = _StubIndex(_RecordingMetadata([]))

    response = await _agent(index)._object_codes(_request(), index.context)

    assert response.object_codes == ""
    assert response.total_matching == 0


@pytest.mark.parametrize("count", [199, 200, 201, 1000, 25_000])
async def test_sizes_either_side_of_the_old_cap(count: int) -> None:
    index = _StubIndex(_RecordingMetadata([f"C{i:05d}" for i in range(count)]))

    response = await _agent(index)._object_codes(_request(), index.context)

    assert len([c for c in response.object_codes.split(",") if c]) == count
