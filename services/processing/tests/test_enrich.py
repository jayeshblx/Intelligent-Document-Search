"""Tests for language detection, PII detection, and classification."""

from __future__ import annotations

from rag_core.models.enums import DocumentClass, Language, PIIType
from rag_processing.enrich import classify_document, detect_language, detect_pii


def test_detect_language_english() -> None:
    text = "This is a clearly English compliance policy document about safety procedures."
    assert detect_language(text) == Language.EN


def test_detect_language_too_short_is_unknown() -> None:
    assert detect_language("hi") == Language.UNKNOWN


def test_detect_pii_email_and_ssn() -> None:
    text = "Contact john.doe@example.com or reference SSN 123-45-6789 for the record."
    findings = {f.type: f for f in detect_pii(text)}
    assert PIIType.EMAIL in findings
    assert PIIType.SSN in findings
    # Raw value never leaked; sample is redacted.
    assert "@" not in findings[PIIType.EMAIL].sample_redacted or "*" in findings[PIIType.EMAIL].sample_redacted


def test_detect_pii_credit_card_luhn() -> None:
    # 4111 1111 1111 1111 is a valid Luhn test number.
    valid = detect_pii("card 4111 1111 1111 1111 on file")
    assert any(f.type is PIIType.CREDIT_CARD for f in valid)
    # A Luhn-invalid 16-digit run should not be flagged as a card.
    invalid = detect_pii("ref 1234 5678 9012 3456 only")
    assert not any(f.type is PIIType.CREDIT_CARD for f in invalid)


def test_detect_pii_clean_text() -> None:
    assert detect_pii("A perfectly ordinary sentence with no personal data.") == []


def test_classify_certificate_beats_other() -> None:
    text = "This Certificate of Conformity certifies the product meets the standard."
    assert classify_document(text, filename="cert_of_conformity.pdf") == DocumentClass.CERTIFICATE


def test_classify_policy() -> None:
    text = "Information Security Policy. This policy defines acceptable use."
    assert classify_document(text, filename="security_policy.docx") == DocumentClass.POLICY


def test_classify_unknown_is_other() -> None:
    assert classify_document("lorem ipsum dolor sit amet") == DocumentClass.OTHER
