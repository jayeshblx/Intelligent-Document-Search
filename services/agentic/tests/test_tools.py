"""Retrieval tools — above all, that the BusinessObject pre-filter is enforced."""

from __future__ import annotations

from typing import Any

import pytest
from rag_agentic.agent_loop import ToolError


async def test_search_returns_grounded_passages(make_tools: Any) -> None:
    tools = make_tools()
    payload, summary = await tools.invoke("search_documents", {"query": "ISO 9001 certificate"})

    assert payload["count"] > 0
    assert "d1" in {p["document_id"] for p in payload["passages"]}
    top = next(p for p in payload["passages"] if p["document_id"] == "d1")
    assert top["document_name"] == "Supplier Quality Manual"
    assert top["facets"]["business_object"] == "SUPPLIERDOCUMENTS"
    assert "passage" in summary


async def test_business_object_filter_restricts_search(make_tools: Any) -> None:
    """A COMPLAINTS-scoped request must never see SUPPLIERDOCUMENTS passages."""
    tools = make_tools("COMPLAINTS")
    payload, _ = await tools.invoke("search_documents", {"query": "supplier ISO 9001 audit"})

    returned = {p["document_id"] for p in payload["passages"]}
    assert returned <= {"d2"}
    assert payload["business_object_filter"] == "COMPLAINTS"


async def test_model_cannot_widen_the_enforced_filter(make_tools: Any) -> None:
    """The model passing its own business_object must not override the caller's."""
    tools = make_tools("COMPLAINTS")
    payload, _ = await tools.invoke(
        "search_documents",
        {"query": "supplier audit", "business_object": "SUPPLIERDOCUMENTS", "status": "Active"},
    )

    assert {p["document_id"] for p in payload["passages"]} <= {"d2"}


async def test_list_documents_counts_only_in_scope(make_tools: Any) -> None:
    unscoped, _ = await make_tools().invoke("list_documents", {})
    scoped, _ = await make_tools("SUPPLIERDOCUMENTS").invoke("list_documents", {})

    assert unscoped["total_matching"] == 3
    assert scoped["total_matching"] == 2
    assert {d["document_id"] for d in scoped["documents"]} == {"d1", "d3"}


async def test_list_documents_honours_a_model_facet(make_tools: Any) -> None:
    payload, _ = await make_tools("SUPPLIERDOCUMENTS").invoke("list_documents", {"status": "Draft"})

    assert {d["document_id"] for d in payload["documents"]} == {"d3"}


async def test_facet_values_cascade_off_the_enforced_filter(make_tools: Any) -> None:
    payload, _ = await make_tools("SUPPLIERDOCUMENTS").invoke("list_facet_values", {})

    codes = payload["facets"].get("object_code", [])
    assert set(codes) == {"SQ-1", "SQ-2"}  # CP-7 belongs to COMPLAINTS
    assert payload["business_object_filter"] == "SUPPLIERDOCUMENTS"


async def test_document_details_outside_the_filter_are_refused(make_tools: Any) -> None:
    """Even a correctly-guessed document_id must not escape the scope."""
    tools = make_tools("COMPLAINTS")
    with pytest.raises(ToolError, match="outside the"):
        await tools.invoke("get_document_details", {"document_id": "d1"})


async def test_unknown_document_is_a_recoverable_tool_error(make_tools: Any) -> None:
    with pytest.raises(ToolError, match="no document"):
        await make_tools().invoke("get_document_details", {"document_id": "nope"})


async def test_empty_query_is_rejected(make_tools: Any) -> None:
    with pytest.raises(ToolError, match="query is required"):
        await make_tools().invoke("search_documents", {"query": "   "})


async def test_unknown_tool_is_rejected(make_tools: Any) -> None:
    with pytest.raises(ToolError, match="unknown tool"):
        await make_tools().invoke("delete_everything", {})


async def test_citations_track_the_best_passage_per_document(make_tools: Any) -> None:
    tools = make_tools()
    await tools.invoke("search_documents", {"query": "ISO 9001 certificate"})
    await tools.invoke("search_documents", {"query": "supplier audit frequency"})

    citations = tools.citations()
    assert citations
    assert len({c.document_id for c in citations}) == len(citations)  # de-duplicated
    assert citations == sorted(citations, key=lambda c: c.score, reverse=True)


async def test_list_documents_filters_by_creator(make_tools: Any) -> None:
    """'Documents created by X' is answerable: created_by is a real filter."""
    payload, _ = await make_tools().invoke(
        "list_documents", {"created_by": "riley@example.com"}
    )

    assert {d["document_id"] for d in payload["documents"]} == {"d1", "d3"}
    assert payload["total_matching"] == 2


async def test_creator_filter_obeys_the_business_object_scope(make_tools: Any) -> None:
    """The creator filter narrows within the enforced scope, never past it."""
    payload, _ = await make_tools("COMPLAINTS").invoke(
        "list_documents", {"created_by": "riley@example.com"}
    )

    # riley created d1 and d3, but both are SUPPLIERDOCUMENTS.
    assert payload["documents"] == []


async def test_creator_values_are_discoverable(make_tools: Any) -> None:
    """The model resolves a name to a stored value through list_facet_values."""
    payload, _ = await make_tools().invoke("list_facet_values", {})

    assert set(payload["facets"].get("created_by", [])) == {
        "riley@example.com",
        "storeuser1",
    }


# ---- multiple BusinessObjects (ExampleApp sends them comma-separated) -------------

async def test_two_business_objects_are_both_in_scope(make_tools: Any) -> None:
    """"A,B" searches A OR B -- not a literal facet value "A,B", which matches nothing."""
    tools = make_tools("SUPPLIERDOCUMENTS,COMPLAINTS")
    payload, _ = await tools.invoke("list_documents", {})

    assert {d["document_id"] for d in payload["documents"]} == {"d1", "d2", "d3"}
    assert payload["total_matching"] == 3


async def test_multi_scope_still_excludes_everything_else(make_tools: Any) -> None:
    """Widening to two values must not widen to three."""
    single, _ = await make_tools("COMPLAINTS").invoke("list_documents", {})
    both, _ = await make_tools("COMPLAINTS,NOSUCHOBJECT").invoke("list_documents", {})

    # NOSUCHOBJECT contributes nothing; COMPLAINTS is unchanged.
    assert {d["document_id"] for d in both["documents"]} == {"d2"}
    assert both["total_matching"] == single["total_matching"]


async def test_multi_scope_search_spans_both(make_tools: Any) -> None:
    payload, _ = await make_tools("SUPPLIERDOCUMENTS,COMPLAINTS").invoke(
        "search_documents", {"query": "supplier complaint audit certificate"}
    )

    assert {p["document_id"] for p in payload["passages"]} & {"d1", "d2"}
    assert payload["business_object_filter"] == "SUPPLIERDOCUMENTS,COMPLAINTS"


async def test_whitespace_and_blanks_in_the_list_are_tolerated(make_tools: Any) -> None:
    """ExampleApp may send " PIF , , INGREDIENTSPEC " -- parse it, do not fail."""
    payload, _ = await make_tools(" SUPPLIERDOCUMENTS , , COMPLAINTS ").invoke(
        "list_documents", {}
    )

    assert payload["total_matching"] == 3
    assert payload["business_object_filter"] == "SUPPLIERDOCUMENTS,COMPLAINTS"


async def test_detail_guard_accepts_any_value_in_scope(make_tools: Any) -> None:
    """The second-line scope guard must not reject a document from the other value."""
    tools = make_tools("SUPPLIERDOCUMENTS,COMPLAINTS")
    payload, _ = await tools.invoke("get_document_details", {"document_id": "d2"})
    assert payload["document_id"] == "d2"

    with pytest.raises(ToolError, match="outside"):
        await make_tools("COMPLAINTS,FACILITYDOCUMENTS").invoke(
            "get_document_details", {"document_id": "d1"}
        )


async def test_empty_business_object_means_whole_tenant(make_tools: Any) -> None:
    for value in ("", None, "  ", ","):
        payload, _ = await make_tools(value).invoke("list_documents", {})
        assert payload["total_matching"] == 3, value
        assert payload["business_object_filter"] is None


# ---- listings return every row to the caller, not just the model's window --------

async def test_listing_window_is_independent_of_the_full_result(make_tools: Any) -> None:
    """The model sees `limit` rows; the caller's envelope carries all of them."""
    tools = make_tools()
    payload, summary = await tools.invoke("list_documents", {"limit": 1})

    assert payload["returned"] == 1
    assert payload["total_matching"] == 3
    assert len(payload["documents"]) == 1  # what the model was shown

    total, everything = tools.listing
    assert total == 3
    assert {d.document_id for d in everything} == {"d1", "d2", "d3"}
    assert "3 document(s) matched" in summary


async def test_offset_pages_the_model_window(make_tools: Any) -> None:
    tools = make_tools()
    first, _ = await tools.invoke("list_documents", {"limit": 2, "offset": 0})
    second, _ = await tools.invoke("list_documents", {"limit": 2, "offset": 2})

    seen_first = [d["document_id"] for d in first["documents"]]
    seen_second = [d["document_id"] for d in second["documents"]]
    assert len(seen_first) == 2
    assert len(seen_second) == 1
    assert not set(seen_first) & set(seen_second)
    # The total never changes with paging.
    assert first["total_matching"] == second["total_matching"] == 3


async def test_listing_rows_carry_what_the_caller_renders(make_tools: Any) -> None:
    await (tools := make_tools()).invoke("list_documents", {})
    _, rows = tools.listing

    row = next(d for d in rows if d.document_id == "d1")
    assert row.document_name == "Supplier Quality Manual"
    assert row.facets["object_code"] == "SQ-1"
    assert row.facets["business_object"] == "SUPPLIERDOCUMENTS"
    assert row.facets["status"] == "Active"


async def test_listing_respects_the_business_object_scope(make_tools: Any) -> None:
    """The envelope must never carry rows the caller is not scoped to see."""
    tools = make_tools("COMPLAINTS")
    await tools.invoke("list_documents", {})

    total, rows = tools.listing
    assert total == 1
    assert {d.document_id for d in rows} == {"d2"}


async def test_no_listing_when_nothing_was_listed(make_tools: Any) -> None:
    tools = make_tools()
    await tools.invoke("search_documents", {"query": "ISO 9001"})
    assert tools.listing is None


# ---- latency safeguards ----------------------------------------------------------

async def test_a_search_after_a_listing_drops_the_listing(make_tools: Any) -> None:
    """A content question that counted something on the way must not return the list.

    Measured: a "what does the manual say" question called list_documents mid-way,
    which would otherwise ship ~1000 unrelated rows back to the caller.
    """
    tools = make_tools()
    await tools.invoke("list_documents", {})
    assert tools.listing is not None

    await tools.invoke("search_documents", {"query": "ISO 9001 certificate"})
    assert tools.listing is None


async def test_a_listing_after_a_search_is_kept(make_tools: Any) -> None:
    """The normal shape of "how many X" -- search to orient, then list -- still works."""
    tools = make_tools()
    await tools.invoke("search_documents", {"query": "supplier"})
    await tools.invoke("list_documents", {})

    assert tools.listing is not None
    assert tools.listing[0] == 3


async def test_default_passages_per_search_is_generous(make_tools: Any) -> None:
    """One rich search must beat several thin ones: extra passages are near-free,
    extra searches cost a model round trip over a transcript that keeps growing."""
    from rag_agentic.config import AgenticSettings

    assert AgenticSettings().top_k >= 15

