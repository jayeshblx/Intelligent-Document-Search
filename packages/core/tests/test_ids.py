from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from rag_core.utils.ids import document_id, chunk_id, point_id, new_message_id


@pytest.fixture
def mock_factory() -> AwsSessionFactory:
    factory = MagicMock(spec=AwsSessionFactory)
    return factory


async def test_document_id() -> None:
    assert isinstance(document_id("bucket", "key"), str)
    assert len(document_id("bucket", "key")) == 36

    # Test version id is included
    assert document_id("bucket", "key", version_id="123") != document_id("bucket", "key")

    # Test same inputs give the same output
    assert document_id("bucket", "key") == document_id("bucket", "key")


async def test_chunk_id() -> None:
    assert isinstance(chunk_id(document_id("bucket", "key"), "strategy", 1), str)
    assert len(chunk_id(document_id("bucket", "key"), "strategy", 1)) == 36

    # Test same inputs give the same output
    assert chunk_id(document_id("bucket", "key"), "strategy", 1) == chunk_id(document_id("bucket", "key"), "strategy", 1)


async def test_point_id() -> None:
    assert isinstance(point_id(chunk_id(document_id("bucket", "key"), "strategy", 1)), str)
    assert len(point_id(chunk_id(document_id("bucket", "key"), "strategy", 1))) == 36

    # Test same inputs give the same output
    assert point_id(chunk_id(document_id("bucket", "key"), "strategy", 1)) == point_id(chunk_id(document_id("bucket", "key"), "strategy", 1))


async def test_new_message_id() -> None:
    assert isinstance(new_message_id(), str)
    assert len(new_message_id()) == 32

    # Test same inputs give different outputs
    assert new_message_id() != new_message_id()
