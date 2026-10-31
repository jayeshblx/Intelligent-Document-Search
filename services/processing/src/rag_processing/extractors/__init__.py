"""Format-specific extractors and the registry that selects them.

Each extractor implements the same ``Extractor`` interface (bytes -> structured
``ExtractedContent``). The ``ExtractorRegistry`` maps a ``DocumentType`` to an
extractor instance, so the service stays oblivious to format details and adding a
format is a localised change.
"""

from __future__ import annotations

from rag_processing.extractors.base import ExtractionContext, Extractor
from rag_processing.extractors.registry import ExtractorRegistry, build_default_registry

__all__ = [
    "ExtractionContext",
    "Extractor",
    "ExtractorRegistry",
    "build_default_registry",
]
