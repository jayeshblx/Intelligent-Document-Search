"""Extractor registry: DocumentType -> Extractor."""

from __future__ import annotations

from rag_core.logging import get_logger
from rag_core.models.enums import DocumentType
from rag_processing.extractors.base import Extractor

_log = get_logger("rag_processing.registry")


class ExtractorRegistry:
    """Resolves the extractor for a document type."""

    def __init__(self) -> None:
        self._by_type: dict[DocumentType, Extractor] = {}

    def register(self, extractor: Extractor) -> None:
        for doc_type in extractor.supported:
            if doc_type in self._by_type:
                _log.warning("extractor_override", doc_type=doc_type.value)
            self._by_type[doc_type] = extractor

    def get(self, doc_type: DocumentType) -> Extractor | None:
        return self._by_type.get(doc_type)

    @property
    def supported_types(self) -> list[DocumentType]:
        return list(self._by_type)


def build_default_registry() -> ExtractorRegistry:
    """Construct + register every built-in extractor.

    Imports are local so importing this module is cheap; the heavy parser deps load
    only when a registry is actually built (service startup)."""
    from rag_processing.extractors.image import ImageExtractor
    from rag_processing.extractors.office import DocxExtractor, PptxExtractor, XlsxExtractor
    from rag_processing.extractors.pdf import PdfExtractor
    from rag_processing.extractors.text import HtmlExtractor, TextExtractor, XmlExtractor

    registry = ExtractorRegistry()
    for extractor in (
        PdfExtractor(),
        DocxExtractor(),
        PptxExtractor(),
        XlsxExtractor(),
        TextExtractor(),
        HtmlExtractor(),
        XmlExtractor(),
        ImageExtractor(),
    ):
        registry.register(extractor)
    return registry
