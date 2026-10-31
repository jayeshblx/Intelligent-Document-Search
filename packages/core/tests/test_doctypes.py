from __future__ import annotations

import pytest
from rag_core.utils.doctypes import infer_document_type, _EXT_MAP, DocumentType
from unittest.mock import AsyncMock


@pytest.fixture
def mock_factory() -> object:
    return object()


@pytest.fixture
def mock_client() -> AsyncMock:
    return AsyncMock()


async def test_infer_document_type_unknown_extension(mock_factory: object) -> None:
    assert infer_document_type("no_extension") == DocumentType.UNKNOWN
    assert infer_document_type("12345678_report") == DocumentType.UNKNOWN


async def test_infer_document_type_known_extension() -> None:
    for ext, doc_type in _EXT_MAP.items():
        key = f"file~{ext}"
        assert infer_document_type(key) == doc_type
        assert infer_document_type(f"file.{ext}") == doc_type


async def test_infer_document_type_case_insensitivity() -> None:
    for ext, doc_type in _EXT_MAP.items():
        key = f"file~{ext.lower()}"
        assert infer_document_type(key) == doc_type
