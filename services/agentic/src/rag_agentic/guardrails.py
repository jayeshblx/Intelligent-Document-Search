"""Guardrails that cost nothing per request.

Three layers, cheapest first:

1. **Prompt rules** (in each agent's system prompt) -- what the agents may and may
   not say: report what documents record, never rule on compliance or safety, never
   give legal, medical, dietary or financial advice, never reveal the prompt.
2. **Code invariants** -- enforced regardless of what the model decides: the tenant
   comes from ``base_url``, the BusinessObject pre-filter is applied in the tool
   layer, and a document outside it is refused even if its id is guessed.
3. **This module** -- deterministic checks around the model: refuse the handful of
   unmistakable prompt-extraction attempts before spending a model call, and redact
   high-risk secrets from anything on the way out.

Amazon Bedrock Guardrails would add managed topic/content policies, but it is billed
per request, so it stays optional (``AGENTIC_GUARDRAIL_ID``) and off by default.

Deliberately narrow: a keyword blocklist that fires on ordinary questions is worse
than no blocklist, because users stop trusting the answers. Only phrasings with no
legitimate reading are refused -- "ignore the previous section of the policy" is a
real question about a document and must still work.
"""

from __future__ import annotations

import re

# Attempts to override the agent's instructions or extract them. Each pattern needs an
# explicit reference to the agent's own instructions/prompt, so document questions
# that merely use the words "ignore", "prompt" or "instructions" are unaffected.
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bignore\s+(all\s+|any\s+|your\s+|the\s+)?(previous\s+|prior\s+|above\s+)?"
               r"(instructions?|rules?|prompts?|directives?)\b", re.I),
    re.compile(r"\bdisregard\s+(all\s+|any\s+|your\s+|the\s+)?(previous\s+|prior\s+)?"
               r"(instructions?|rules?|prompts?)\b", re.I),
    re.compile(r"\b(reveal|show|print|repeat|display|output|tell\s+me)\b[^.?!]{0,40}"
               r"\b(system\s+prompt|your\s+(instructions?|prompt|rules)|initial\s+prompt)\b", re.I),
    re.compile(r"\b(what|which)\s+(are|is)\s+your\s+(system\s+prompt|instructions?|rules)\b", re.I),
    re.compile(r"\byou\s+are\s+now\b[^.?!]{0,40}\b(dan|jailbroken|unrestricted|developer\s+mode)\b", re.I),
    re.compile(r"\b(act|behave|pretend)\s+as\s+(if\s+you\s+are\s+)?(an?\s+)?"
               r"(unrestricted|uncensored|different)\s+(ai|assistant|model)\b", re.I),
)

REFUSAL = (
    "I can't do that. I answer questions about your compliance documents: ask me what "
    "your documents say about a supplier, a complaint, an audit or a certificate."
)

# Secrets that have no place in an answer about compliance documents. Business contact
# details (supplier emails, author names) are intentionally NOT redacted: "who raised
# this complaint?" is a legitimate question whose answer is a person.
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # 13-19 digits in card-like grouping.
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[redacted-card-number]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[redacted-ssn]"),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[redacted-aws-key]"),
    (re.compile(r"(?i)\b(?:password|passwd|secret|api[_-]?key)\s*[:=]\s*\S+"), "[redacted-secret]"),
    (re.compile(r"-----BEGIN[^-]{0,40}PRIVATE KEY-----"), "[redacted-private-key]"),
)


def injection_attempt(query: str) -> bool:
    """True for a question whose only purpose is to override or extract the prompt."""
    text = (query or "").strip()
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


def redact(text: str) -> str:
    """Mask secrets in text on its way to the caller.

    Applied to answers and to citation snippets: a chunk of a document can contain a
    key or a card number, and neither belongs in an answer or a log.
    """
    if not text:
        return text
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text
