"""Heuristic document classification into ``DocumentClass``.

A transparent keyword-scoring baseline (title weighted higher than body): fast,
deterministic, no model call. The signature is the seam to upgrade to a Bedrock
zero-shot classifier later without touching callers — return type is unchanged.
"""

from __future__ import annotations

from rag_core.models.enums import DocumentClass

# Signal terms per class. Order matters only for tie-breaking (first wins).
_SIGNALS: list[tuple[DocumentClass, tuple[str, ...]]] = [
    (DocumentClass.CERTIFICATE, ("certificate", "certification", "certified", "accreditation")),
    (DocumentClass.REGULATION, ("regulation", "directive", "regulatory", "statute", "gdpr", "compliance requirement")),
    (DocumentClass.STANDARD, ("iso ", "iec ", "astm", "standard ", "en iso")),
    (DocumentClass.POLICY, ("policy", "policies", "code of conduct")),
    (DocumentClass.WORKFLOW, ("workflow", "standard operating procedure", "sop", "process flow", "procedure")),
    (DocumentClass.TECHNICAL_MANUAL, ("manual", "user guide", "operating instructions", "handbook")),
    (DocumentClass.DESIGN_DOCUMENT, ("design document", "architecture", "design specification")),
    (DocumentClass.SPECIFICATION, ("specification", "requirements", "spec sheet")),
    (DocumentClass.REPORT, ("report", "audit findings", "assessment")),
]

_TITLE_WEIGHT = 5
_BODY_WEIGHT = 1


def classify_document(
    text: str, *, filename: str | None = None, title: str | None = None
) -> DocumentClass:
    """Classify a document from its title/filename (weighted) and body text."""
    head = " ".join(filter(None, (title, filename))).lower()
    body = text[:5000].lower()  # a prefix is sufficient signal and bounds cost

    best_class = DocumentClass.OTHER
    best_score = 0
    for doc_class, terms in _SIGNALS:
        score = sum(_TITLE_WEIGHT * head.count(t) + _BODY_WEIGHT * body.count(t) for t in terms)
        if score > best_score:
            best_class, best_score = doc_class, score
    return best_class
