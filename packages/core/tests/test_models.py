"""Tests for domain models and their invariants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_core.models import (
    Chunk,
    ChunkEmbedding,
    ChunkStrategy,
    DenseVector,
    DocumentMetadata,
    EmbeddingModel,
    MultiVector,
)


def test_dense_vector_dim_must_match_values() -> None:
    DenseVector(model=EmbeddingModel.TITAN, dim=3, values=[0.1, 0.2, 0.3])
    with pytest.raises(ValidationError):
        DenseVector(model=EmbeddingModel.TITAN, dim=3, values=[0.1, 0.2])


def test_multivector_rows_must_match_dim() -> None:
    MultiVector(model=EmbeddingModel.COLBERT, dim=2, vectors=[[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValidationError):
        MultiVector(model=EmbeddingModel.COLBERT, dim=2, vectors=[[1.0, 2.0], [3.0]])


def test_chunk_embedding_named_vectors_shape() -> None:
    ce = ChunkEmbedding(
        chunk_id="c1",
        document_id="d1",
        dense={EmbeddingModel.TITAN: DenseVector(model=EmbeddingModel.TITAN, dim=2, values=[1.0, 0.0])},
        multi={
            EmbeddingModel.COLBERT: MultiVector(
                model=EmbeddingModel.COLBERT, dim=2, vectors=[[1.0, 0.0], [0.0, 1.0]]
            )
        },
    )
    named = ce.named_vectors()
    assert named["titan"] == [1.0, 0.0]
    assert named["colbert"] == [[1.0, 0.0], [0.0, 1.0]]
    assert set(ce.models) == {EmbeddingModel.TITAN, EmbeddingModel.COLBERT}


def test_chunk_citation_formats_pages_and_section() -> None:
    chunk = Chunk(
        id="c1",
        document_id="doc-42",
        strategy=ChunkStrategy.CHILD,
        text="...",
        page_start=3,
        page_end=4,
        section_path=["3. Safety", "3.2 Handling"],
    )
    citation = chunk.citation
    assert "doc-42" in citation
    assert "3.2 Handling" in citation
    assert "pp.3-4" in citation


def test_metadata_filterable_payload_serialises_dates() -> None:
    from datetime import date

    md = DocumentMetadata(
        document_id="d1",
        country="DE",
        effective_date=date(2026, 1, 1),
        keywords=["gmp", "audit"],
    )
    payload = md.filterable_payload()
    assert payload["country"] == "DE"
    assert payload["effective_date"] == "2026-01-01"
    assert payload["keywords"] == ["gmp", "audit"]
