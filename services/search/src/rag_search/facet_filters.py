"""Facet filter values: exactly one value, or any-of several.

A filter value used to be a single string, which made ``BusinessObject`` a
one-or-nothing choice. ExampleApp sends a comma-separated list when a user is
scoped to several business objects at once (``"INGREDIENTSPEC,PIF"``), so a
value is now either a string or a list of accepted values, combined with OR.

Every backend routes its filter values through :func:`accepted_values`, so the
local (Qdrant + SQLite) and Postgres paths agree on what a filter means.
"""

from __future__ import annotations

from collections.abc import Sequence

# One facet -> the value it must equal, or any of the values it may equal.
FacetFilters = dict[str, "str | list[str]"]


def accepted_values(value: str | Sequence[str] | None) -> list[str]:
    """The values a facet may take, empties dropped.

    An empty result means "no constraint": callers skip the predicate entirely
    rather than emitting one that matches nothing.
    """
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    seen: dict[str, None] = {}
    for item in value:
        text = str(item).strip()
        if text:
            seen.setdefault(text, None)
    return list(seen)


def split_multi(raw: str | Sequence[str] | None) -> list[str]:
    """Parse a caller-supplied filter into its values.

    Accepts a list, or the comma-separated string ExampleApp sends. Duplicates are
    collapsed and order is preserved, so the echoed-back value is stable.
    """
    if raw is None:
        return []
    if not isinstance(raw, str):
        return accepted_values(raw)
    return accepted_values(raw.split(","))
