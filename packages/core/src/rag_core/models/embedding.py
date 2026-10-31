"""Embedding models — one chunk carries multiple vectors.

The platform embeds each chunk with several models because they have
complementary strengths: Titan (Bedrock dense, no GPU), BGE-M3 (multilingual
dense), and ColBERT (late-interaction multi-vector for precise token matching).
Qdrant stores these as *named vectors* on a single point, so retrieval can pick
the best space per query without re-indexing.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from rag_core.models.base import DomainModel
from rag_core.models.enums import EmbeddingModel


class DenseVector(DomainModel):
    """A single dense embedding from one model."""

    model: EmbeddingModel
    dim: int = Field(ge=1)
    values: list[float]

    @model_validator(mode="after")
    def _check_dim(self) -> DenseVector:
        if len(self.values) != self.dim:
            raise ValueError(
                f"vector length {len(self.values)} != declared dim {self.dim} "
                f"for model {self.model}"
            )
        return self


class MultiVector(DomainModel):
    """A late-interaction (ColBERT-style) multi-vector: one vector per token."""

    model: EmbeddingModel = EmbeddingModel.COLBERT
    dim: int = Field(ge=1)
    vectors: list[list[float]]

    @model_validator(mode="after")
    def _check_dims(self) -> MultiVector:
        bad = next((i for i, v in enumerate(self.vectors) if len(v) != self.dim), None)
        if bad is not None:
            raise ValueError(
                f"multi-vector row {bad} length {len(self.vectors[bad])} != dim {self.dim}"
            )
        return self


class ChunkEmbedding(DomainModel):
    """All vectors computed for one chunk, keyed by model.

    Stored together so the indexer writes a single Qdrant point with multiple
    named vectors. Either map may be empty if a provider is disabled."""

    chunk_id: str
    document_id: str
    namespace: str = "default"

    dense: dict[EmbeddingModel, DenseVector] = Field(default_factory=dict)
    multi: dict[EmbeddingModel, MultiVector] = Field(default_factory=dict)

    @property
    def models(self) -> list[EmbeddingModel]:
        return [*self.dense.keys(), *self.multi.keys()]

    def named_vectors(self) -> dict[str, list[float] | list[list[float]]]:
        """Render to Qdrant's ``{vector_name: data}`` shape. Dense models use
        their plain name; ColBERT uses ``colbert`` with a 2-D payload."""
        out: dict[str, list[float] | list[list[float]]] = {}
        for model, vec in self.dense.items():
            out[model.value] = vec.values
        for model, mv in self.multi.items():
            out[model.value] = mv.vectors
        return out
