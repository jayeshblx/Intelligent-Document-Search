"""Tests for the MetadataFilter -> Qdrant filter translation (offline, no server).

Only the pure translation logic is exercised here; live Qdrant operations are
covered by integration tests (marked ``integration``) that run against the
docker-compose Qdrant.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from qdrant_client import models

from rag_core.config import QdrantSettings
from rag_core.stores.vector import (
    FieldCondition,
    MatchOp,
    MetadataFilter,
    QdrantVectorStore,
)


@pytest.fixture
async def store() -> QdrantVectorStore:
    # REST client; construction does not open a connection.
    s = QdrantVectorStore(QdrantSettings(prefer_grpc=False, url="http://localhost:6333"))
    yield s
    await s.aclose()


async def test_eq_and_range_in_must(store: QdrantVectorStore) -> None:
    mf = MetadataFilter(
        must=[
            FieldCondition(field="country", op=MatchOp.EQ, value="DE"),
            FieldCondition(field="effective_date", op=MatchOp.GTE, value="2026-01-01"),
        ]
    )
    qf = store._to_qdrant_filter(mf)
    assert qf.must is not None
    assert len(qf.must) == 2

    eq, rng = qf.must
    assert isinstance(eq, models.FieldCondition)
    assert isinstance(eq.match, models.MatchValue)
    assert eq.match.value == "DE"

    # ISO date range -> DatetimeRange (qdrant parses the ISO string to datetime).
    assert isinstance(rng, models.FieldCondition)
    assert isinstance(rng.range, models.DatetimeRange)
    assert rng.range.gte == datetime(2026, 1, 1)


async def test_in_clause_uses_match_any(store: QdrantVectorStore) -> None:
    mf = MetadataFilter(
        should=[FieldCondition(field="category", op=MatchOp.IN, value=["a", "b"])]
    )
    qf = store._to_qdrant_filter(mf)
    assert qf.should is not None and len(qf.should) == 1
    cond = qf.should[0]
    assert isinstance(cond.match, models.MatchAny)
    assert cond.match.any == ["a", "b"]


async def test_numeric_range_uses_plain_range(store: QdrantVectorStore) -> None:
    mf = MetadataFilter(must=[FieldCondition(field="score", op=MatchOp.LT, value=10)])
    qf = store._to_qdrant_filter(mf)
    cond = qf.must[0]
    assert isinstance(cond.range, models.Range)
    assert cond.range.lt == 10


async def test_exists_in_must_becomes_is_empty_in_must_not(store: QdrantVectorStore) -> None:
    mf = MetadataFilter(must=[FieldCondition(field="supplier", op=MatchOp.EXISTS)])
    qf = store._to_qdrant_filter(mf)
    # EXISTS flips: "field is present" == "NOT (field is empty)".
    assert not qf.must
    assert qf.must_not is not None and len(qf.must_not) == 1
    assert isinstance(qf.must_not[0], models.IsEmptyCondition)
