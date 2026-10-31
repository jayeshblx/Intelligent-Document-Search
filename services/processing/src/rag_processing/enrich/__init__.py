"""Content enrichment: language detection, PII detection, classification.

These run after extraction on the assembled text. Each is independently toggled
(config) and side-effect-free, so they're trivially testable and can be swapped
for managed services (Amazon Comprehend for language/PII, Bedrock for
classification) behind the same function signatures.
"""

from __future__ import annotations

from rag_processing.enrich.classify import classify_document
from rag_processing.enrich.language import detect_language
from rag_processing.enrich.pii import detect_pii

__all__ = ["classify_document", "detect_language", "detect_pii"]
