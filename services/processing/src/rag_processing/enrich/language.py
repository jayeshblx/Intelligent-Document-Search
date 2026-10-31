"""Language detection via langdetect (pure-python, deterministic).

Mapped to the platform's ``Language`` enum; unknown/low-confidence falls back to
``Language.UNKNOWN`` rather than guessing. Deterministic seeding keeps results
stable across runs (langdetect is otherwise randomised)."""

from __future__ import annotations

from rag_core.models.enums import Language

# langdetect codes -> our enum (others collapse to UNKNOWN).
_LANG_MAP = {
    "en": Language.EN,
    "fr": Language.FR,
    "de": Language.DE,
    "es": Language.ES,
    "it": Language.IT,
    "pt": Language.PT,
    "zh-cn": Language.ZH,
    "zh-tw": Language.ZH,
    "ja": Language.JA,
}


def detect_language(text: str, *, min_chars: int = 20) -> Language:
    """Detect the dominant language of ``text``. Returns UNKNOWN if too short or
    undetectable."""
    sample = text.strip()
    if len(sample) < min_chars:
        return Language.UNKNOWN
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0  # deterministic
        code = detect(sample[:5000])  # a prefix is plenty and faster
    except Exception:
        return Language.UNKNOWN
    return _LANG_MAP.get(code, Language.UNKNOWN)
