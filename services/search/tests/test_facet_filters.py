"""Facet filter values: one value, or any-of several.

The SQL assertions here matter because the Postgres path needs a running server to
test end-to-end; the predicate string and its parameter ORDER are what actually
broke when BusinessObject became a list, and both are checkable offline.
"""

from __future__ import annotations

import pytest
from rag_search.facet_filters import accepted_values, split_multi
from rag_search.stores.postgres import _facet_sql


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("INGREDIENTSPEC,PIF", ["INGREDIENTSPEC", "PIF"]),
        (" INGREDIENTSPEC , PIF ", ["INGREDIENTSPEC", "PIF"]),
        ("PIF", ["PIF"]),
        ("PIF,,PIF", ["PIF"]),  # blanks dropped, duplicates collapsed
        ("", []),
        ("   ", []),
        (",", []),
        (None, []),
        (["PIF", "INGREDIENTSPEC"], ["PIF", "INGREDIENTSPEC"]),
    ],
)
def test_split_multi(raw: object, expected: list[str]) -> None:
    assert split_multi(raw) == expected  # type: ignore[arg-type]


def test_accepted_values_preserves_order() -> None:
    """The echoed-back filter must be stable, so order is not sorted away."""
    assert accepted_values(["PIF", "AUDIT", "PIF"]) == ["PIF", "AUDIT"]


def test_single_value_uses_equality() -> None:
    sql, params = _facet_sql({"business_object": ["PIF"]})

    assert sql == " AND chunks.payload->>%s = %s"
    assert params == ["business_object", "PIF"]


def test_several_values_use_any() -> None:
    sql, params = _facet_sql({"business_object": ["INGREDIENTSPEC", "PIF"]})

    assert sql == " AND chunks.payload->>%s = ANY(%s)"
    assert params == ["business_object", ["INGREDIENTSPEC", "PIF"]]


def test_facets_and_together_in_parameter_order() -> None:
    """Two facets AND; their params must interleave key,value in clause order."""
    sql, params = _facet_sql({"business_object": ["PIF", "AUDIT"], "status": ["Active"]})

    assert sql == " AND chunks.payload->>%s = ANY(%s) AND chunks.payload->>%s = %s"
    assert params == ["business_object", ["PIF", "AUDIT"], "status", "Active"]


def test_empty_value_emits_no_predicate() -> None:
    """A filter with nothing in it must not narrow to zero rows."""
    assert _facet_sql({"business_object": []}) == ("", [])
    assert _facet_sql({}) == ("", [])
