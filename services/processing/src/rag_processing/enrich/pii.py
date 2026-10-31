"""Regex-based PII detection.

Records *findings* (type + count + a redacted sample) — never the raw value — so
the metadata is safe to store/log while still flagging documents for compliance
review. Credit-card matches are Luhn-validated to cut false positives. Swap for
Amazon Comprehend PII behind ``detect_pii`` when entity-level NLP is needed.
"""

from __future__ import annotations

import re

from rag_core.models.enums import PIIType
from rag_core.models.metadata import PIIFinding

_PATTERNS: dict[PIIType, re.Pattern[str]] = {
    PIIType.EMAIL: re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    PIIType.SSN: re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    PIIType.PHONE: re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?){2,4}\d{2,4}(?!\d)"),
    PIIType.CREDIT_CARD: re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    PIIType.IBAN: re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
}


def _luhn_ok(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _redact(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) <= 2:
        return "*" * len(cleaned)
    return f"{cleaned[0]}{'*' * (len(cleaned) - 2)}{cleaned[-1]}"


def detect_pii(text: str) -> list[PIIFinding]:
    """Return PII findings for ``text``. Empty list if none detected."""
    findings: list[PIIFinding] = []
    for pii_type, pattern in _PATTERNS.items():
        matches = pattern.findall(text)
        if pii_type is PIIType.CREDIT_CARD:
            matches = [m for m in matches if _luhn_ok(m)]
        # Phone pattern is greedy; drop very short spurious hits.
        if pii_type is PIIType.PHONE:
            matches = [m for m in matches if len(re.sub(r"\D", "", m)) >= 7]
        if not matches:
            continue
        findings.append(
            PIIFinding(type=pii_type, count=len(matches), sample_redacted=_redact(str(matches[0])))
        )
    return findings
