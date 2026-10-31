"""The two-phase flow that lets ExampleApp enforce its own document permissions.

ExampleApp owns per-user access, and the agent has no way to evaluate it. So:

1. ExampleApp asks which object codes exist for a BusinessObject (`operation:
   "objectCodes"`). No model call -- a facet lookup.
2. ExampleApp filters that list against the user's ACL.
3. ExampleApp asks the real question, passing the survivors as `ObjectCodes`.

Step 3 is the one that matters here: the agent must not return a document outside
that set by ANY route -- listing, search, citations, facet values, or a directly
supplied document id.
"""

from __future__ import annotations

from typing import Any

import pytest
from rag_agentic.agent_loop import ToolError
from rag_agentic.models import AgentRequest


def test_phase_one_needs_no_question() -> None:
    """A discovery call carries no Query; requiring one would be pointless."""
    request = AgentRequest.model_validate(
        {
            "BaseUrl": "https://foxtrot.acmedev.com/",
            "BusinessObject": "COMPLAINT",
            "Operation": "objectCodes",
            "UserCode": "Taylor",
        }
    )
    assert request.operation == "objectCodes"
    assert request.query == ""
    assert request.business_object == "COMPLAINT"


@pytest.mark.parametrize("spelling", ["ObjectCodes", "objectCodes", "object_codes"])
def test_phase_two_accepts_the_acl_list(spelling: str) -> None:
    request = AgentRequest.model_validate(
        {
            "BaseUrl": "https://foxtrot.acmedev.com/",
            "Query": "which complaints are open?",
            spelling: "SQ-1,CP-7",
        }
    )
    assert request.object_codes == "SQ-1,CP-7"


async def test_listing_is_restricted_to_the_permitted_codes(make_tools: Any) -> None:
    tools = make_tools(object_codes="SQ-1")
    payload, _ = await tools.invoke("list_documents", {})

    assert {d["document_id"] for d in payload["documents"]} == {"d1"}
    assert payload["total_matching"] == 1


async def test_several_permitted_codes_are_combined(make_tools: Any) -> None:
    payload, _ = await make_tools(object_codes="SQ-1,CP-7").invoke("list_documents", {})

    assert {d["document_id"] for d in payload["documents"]} == {"d1", "d2"}


async def test_search_cannot_reach_an_unpermitted_document(make_tools: Any) -> None:
    """The excluded document is the best match for this query, by design."""
    payload, _ = await make_tools(object_codes="CP-7").invoke(
        "search_documents", {"query": "supplier ISO 9001 certificate audit"}
    )

    assert "d1" not in {p["document_id"] for p in payload["passages"]}


async def test_the_model_cannot_widen_the_permitted_set(make_tools: Any) -> None:
    """A model-chosen object_code must not override the user's ACL."""
    payload, _ = await make_tools(object_codes="CP-7").invoke(
        "search_documents", {"query": "supplier audit", "object_code": "SQ-1"}
    )

    assert "d1" not in {p["document_id"] for p in payload["passages"]}


async def test_a_guessed_document_id_outside_the_acl_is_refused(make_tools: Any) -> None:
    """Knowing an id is not permission to read it."""
    with pytest.raises(ToolError, match="this user may see"):
        await make_tools(object_codes="CP-7").invoke(
            "get_document_details", {"document_id": "d1"}
        )


async def test_facet_values_do_not_leak_unpermitted_documents(make_tools: Any) -> None:
    payload, _ = await make_tools(object_codes="CP-7").invoke("list_facet_values", {})

    assert "SQ-1" not in payload["facets"].get("object_code", [])


async def test_citations_stay_inside_the_acl(make_tools: Any) -> None:
    tools = make_tools(object_codes="CP-7")
    await tools.invoke("search_documents", {"query": "complaint acknowledgement"})

    assert "d1" not in {c.document_id for c in await tools.citations_with_urls()}


async def test_no_acl_supplied_means_no_extra_narrowing(make_tools: Any) -> None:
    """ExampleApp does not call at all when a user may see nothing, so an absent list
    means 'not using the two-phase flow', not 'permit nothing'."""
    payload, _ = await make_tools().invoke("list_documents", {})

    assert payload["total_matching"] == 3
