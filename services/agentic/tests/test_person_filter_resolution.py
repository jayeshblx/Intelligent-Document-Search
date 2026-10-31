"""A part of a person's name resolves to stored values deterministically.

Observed in production: the identical request "List of Documents created by Alex"
returned **18 documents** on one call and **14** on the next, with temperature at
0.0. The ground truth is 14 (UserAdminAlex 13 + UADMINALEX 1); the 18 came from the
model also counting DEVUSER's four.

The cause was that "Alex" is not a stored value, so the model had to decide what it
meant — and decided differently each run. Resolving it in the tool layer makes the
same question give the same answer, and reports which values matched so the answer
can say so rather than implying an exact match.
"""

from __future__ import annotations

from typing import Any

import pytest

# The real created_by values in BRAVO's index, within the requested object codes.
BRAVO_CREATED_BY = ["UserAdminAlex", "UADMINALEX", "DEVUSER", "USERQA_23", "JORDANQA"]


class _Metadata:
    def __init__(self, values: list[str]) -> None:
        self._values = values

    async def facet_values(self, key: str, namespace: str = "default", **_: Any) -> list[str]:
        return list(self._values) if key == "created_by" else []


class _Index:
    def __init__(self, values: list[str]) -> None:
        self.metadata = _Metadata(values)
        self.namespace = "default"


def _tools(values: list[str] = BRAVO_CREATED_BY) -> Any:
    from rag_agentic.tools import DocumentSearchTools

    tools = DocumentSearchTools.__new__(DocumentSearchTools)
    tools._index = _Index(values)  # type: ignore[attr-defined]
    return tools


async def test_the_production_case_resolves_to_both_om_users() -> None:
    """'Alex' must mean UserAdminAlex AND UADMINALEX — the 14, not the 18."""
    tools = _tools()
    filters: dict[str, Any] = {"created_by": "Alex"}

    resolved = await tools._resolve_person_filters(filters)

    assert set(filters["created_by"]) == {"UserAdminAlex", "UADMINALEX"}
    assert "DEVUSER" not in filters["created_by"], "DEVUSER is not Alex"
    assert resolved["created_by"] == filters["created_by"]


async def test_the_same_input_always_resolves_the_same_way() -> None:
    """The whole point: no run-to-run variation."""
    results = []
    for _ in range(5):
        filters: dict[str, Any] = {"created_by": "Alex"}
        await _tools()._resolve_person_filters(filters)
        results.append(tuple(sorted(filters["created_by"])))

    assert len(set(results)) == 1


@pytest.mark.parametrize("given", ["UserAdminAlex", "useradminalex", "USERADMINALEX"])
async def test_an_exact_value_wins_over_a_substring(given: str) -> None:
    """Naming a user exactly must not also pull in the other Alex user."""
    filters: dict[str, Any] = {"created_by": given}

    await _tools()._resolve_person_filters(filters)

    assert filters["created_by"] == ["UserAdminAlex"]


async def test_matching_is_case_insensitive() -> None:
    filters: dict[str, Any] = {"created_by": "alex"}

    await _tools()._resolve_person_filters(filters)

    assert set(filters["created_by"]) == {"UserAdminAlex", "UADMINALEX"}


async def test_a_name_that_matches_nothing_is_left_alone() -> None:
    """Returning nothing is honest; widening the filter would invent an answer."""
    filters: dict[str, Any] = {"created_by": "Nobody"}

    resolved = await _tools()._resolve_person_filters(filters)

    assert filters["created_by"] == "Nobody"
    assert resolved == {}


async def test_what_matched_is_reported() -> None:
    """So the answer can say '14 created by Alex (UserAdminAlex, UADMINALEX)' rather than
    implying the user asked for exactly those."""
    resolved = await _tools()._resolve_person_filters({"created_by": "Alex"})

    assert sorted(resolved["created_by"]) == ["UADMINALEX", "UserAdminAlex"]


async def test_other_facets_are_untouched() -> None:
    """Only people facets resolve loosely; an object code must stay exact."""
    filters: dict[str, Any] = {"object_code": "PIF39", "status": "Active"}

    resolved = await _tools()._resolve_person_filters(filters)

    assert filters["object_code"] == "PIF39"
    assert filters["status"] == "Active"
    assert resolved == {}


async def test_a_list_of_values_is_not_re_resolved() -> None:
    """An ACL or an already-resolved set must pass through unchanged."""
    filters: dict[str, Any] = {"created_by": ["UserAdminAlex", "DEVUSER"]}

    resolved = await _tools()._resolve_person_filters(filters)

    assert filters["created_by"] == ["UserAdminAlex", "DEVUSER"]
    assert resolved == {}
