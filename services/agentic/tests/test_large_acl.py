"""A large ObjectCodes ACL is split across queries, never truncated.

ExampleApp sends the object codes a user may see. A broadly-permissioned user on a
large tenant can be thousands. Measured: Atlas handled 5,000 in 0.2s and the file
backends 20,000 — but SQLite has a hard ceiling (SQLITE_MAX_VARIABLE_NUMBER,
32,766) and Lucene caps boolean clauses, so the ACL is batched rather than left
to find that edge in production.

Truncating is the one thing that must never happen: a dropped code makes its
documents invisible, and the agent cannot tell that from "not in your documents".
"""

from __future__ import annotations

from typing import Any

import pytest
from rag_agentic.tools import MAX_FILTER_VALUES, DocumentSearchTools, _batched


def _codes(n: int) -> list[str]:
    return [f"PIF{i:05d}" for i in range(n)]


# ---- batching keeps every value --------------------------------------------------

@pytest.mark.parametrize("size", [0, 1, 999, 1000, 1001, 5000, 32767])
def test_no_value_is_ever_dropped(size: int) -> None:
    codes = _codes(size)

    flattened = [c for batch in _batched(codes) for c in batch]

    assert flattened == codes, "batching must preserve every code, in order"


@pytest.mark.parametrize("size", [1001, 2500, 5000])
def test_no_batch_exceeds_the_limit(size: int) -> None:
    assert all(len(b) <= MAX_FILTER_VALUES for b in _batched(_codes(size)))


def test_a_small_acl_stays_a_single_query() -> None:
    """The common case must not pay for machinery it does not need."""
    assert len(_batched(_codes(324))) == 1   # FOXTROT's whole object-code vocabulary
    assert len(_batched(_codes(413))) == 1   # BRAVO's


def test_the_threshold_is_below_the_measured_ceiling() -> None:
    """Atlas was fine at 5,000 and SQLite allows 32,766 variables; stay well under
    both so one query can also carry its other filters."""
    assert MAX_FILTER_VALUES <= 5000


def test_an_empty_acl_yields_one_empty_batch() -> None:
    """Not zero batches — that would skip the query and silently return nothing."""
    assert _batched([]) == [[]]


# ---- the toolbox splits only the ACL ----------------------------------------------

def _tools() -> DocumentSearchTools:
    """A toolbox with no index behind it.

    ``_acl_batches`` only rewrites a filter dict, so it needs no store, no
    embedder and no tenant — constructing one would only obscure what is
    under test.
    """
    return DocumentSearchTools.__new__(DocumentSearchTools)


def test_a_large_acl_is_split() -> None:
    tools = _tools()
    filters: dict[str, Any] = {"object_code": _codes(2500), "business_object": ["PIF"]}

    batches = tools._acl_batches(filters)

    assert len(batches) == 3
    assert sum(len(b["object_code"]) for b in batches) == 2500


def test_every_batch_keeps_the_other_filters() -> None:
    """Losing business_object or tenant on a batch would widen the scope — the
    opposite of what an ACL is for."""
    tools = _tools()
    filters: dict[str, Any] = {
        "object_code": _codes(2500),
        "business_object": ["PIF", "INGREDIENTSPEC"],
        "tenant": ["BRAVO"],
    }

    for batch in tools._acl_batches(filters):
        assert batch["business_object"] == ["PIF", "INGREDIENTSPEC"]
        assert batch["tenant"] == ["BRAVO"]


def test_a_small_acl_is_not_split() -> None:
    tools = _tools()
    filters: dict[str, Any] = {"object_code": _codes(16)}  # the real ExampleApp payload size

    assert tools._acl_batches(filters) == [filters]


def test_other_large_filters_are_not_split() -> None:
    """Only object_code is sized by permissions; splitting a facet the question
    chose would change what was asked."""
    tools = _tools()
    filters: dict[str, Any] = {"created_by": _codes(2500)}

    assert tools._acl_batches(filters) == [filters]
