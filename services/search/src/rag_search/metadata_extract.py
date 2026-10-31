"""Metadata Extraction Agent (platform service #4).

Extracts the rich compliance facets — title, author, country, region,
department, business_unit, supplier, product, regulation, version/revision,
category/sub_category, effective/expiry dates, keywords, entities — from a
document's extracted text into ``DocumentMetadata``.

Two layers:
* **Heuristics** (free, no LLM): title from headings/filename; regulation, version,
  revision, and dates via regex; keywords from term frequency.
* **LLM** (Bedrock Claude, optional): the semantic facets heuristics can't reliably
  get — country, supplier, department, business_unit, product, category, entities.
  Returns strict JSON; on any failure we fall back to heuristics, so extraction
  never breaks the pipeline.

The **supplier name** gets its own pass (``extract_supplier``). It is the one facet the
system of record cannot supply: an attachment names its owner only by code
(``ObjectCode`` = "SUPPLIER0028"), so the readable name has to come from the document
itself — "CONTOSO AUSTRALIA PTY LTD" printed on the certificate. That pass runs even
for documents whose other facets come from MongoDB, where the big prompt is skipped.

``language``, ``document_class`` and ``pii`` are passed in (already computed by
the processing service) rather than recomputed.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date
from typing import TYPE_CHECKING, Any

from rag_core.logging import get_logger
from rag_core.models.content import ExtractedContent
from rag_core.models.enums import DocumentClass, Language
from rag_core.models.metadata import DocumentMetadata, Entity, PIIFinding

if TYPE_CHECKING:
    from rag_search.answer import LLM

_log = get_logger("rag_search.metadata")

# --- heuristic patterns ------------------------------------------------------
_REGULATIONS = [
    r"ISO/IEC\s?\d{3,5}", r"ISO\s?\d{3,5}", r"IEC\s?\d{3,5}", r"AS/NZS\s?\d{3,5}",
    r"AS\s?\d{3,5}", r"HACCP", r"GDPR", r"BRCGS?", r"SQF", r"GFSI",
    r"FDA\s?21\s?CFR(?:\s?Part\s?\d+)?", r"Food Act\s?\d{4}", r"FSSC\s?\d{5}",
]
_REG_RE = re.compile("|".join(_REGULATIONS), re.IGNORECASE)
_VERSION_RE = re.compile(r"\b(?:version|ver\.?|v)\s?([0-9]+(?:\.[0-9]+)*)\b", re.IGNORECASE)
_REVISION_RE = re.compile(r"\b(?:revision|rev\.?)\s?([0-9]+(?:\.[0-9]+)*)\b", re.IGNORECASE)
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DMY_DATE = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b")
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "will",
    "have", "has", "not", "any", "all", "may", "can", "must", "shall", "which",
    "been", "were", "our", "their", "your", "these", "such", "into", "each",
    "document", "documents", "page", "section", "table", "date", "name",
}


# Certification bodies, accreditors and standards owners. On a certificate these are
# printed larger and earlier than the company being certified, so any "first company
# name in the document" rule picks the auditor instead of the supplier. Never a supplier.
_CERTIFIER_TOKENS = frozenset({
    "bsi", "sgs", "nsf", "intertek", "bureau veritas", "sai global", "dnv", "lloyd",
    "tuv", "tüv", "eurofins", "asurequality", "brcgs", "brc ", "fssc", "sqf", "iso ",
    "ukas", "jas-anz", "jasanz", "aib ", "silliker", "mérieux", "merieux", "kiwa",
    "certification", "certified organizations", "accreditation", "registrar",
})
# A company name ends in a legal-entity suffix — that suffix is what separates an
# organisation from a heading, an address line or a stray phrase. Only true legal
# suffixes qualify: allowing words like "Foods"/"Dairy"/"Group" pulled in text such as
# "No Marking Required 3L Dairy" from the real corpus. The name part stays
# case-SENSITIVE (every word starts uppercase, so "of Registration BSI …" can't match)
# while the suffix is matched case-insensitively via a scoped flag, since documents
# write both "Pty Ltd" and "PTY LTD". No "." inside a word, so a sentence boundary
# ("Australia. CONTOSO …") ends the candidate instead of being swallowed by it.
_COMPANY_RE = re.compile(
    r"(?<![A-Za-z])("
    r"[A-Z][\w&'\-]*(?:\s+[A-Z0-9][\w&'\-]*){0,4}\s+"
    r"(?i:(?:Co\.?,?\s*)?Pty\.?\s*Ltd\.?|(?:Co\.?,?\s*)?Pty\.?\s*Limited|"
    r"Proprietary\s+Limited|Pte\.?\s*Ltd\.?|Co\.?,?\s*Ltd\.?|"
    r"Ltd\.?|Limited|Inc\.?|LLC|GmbH|N\.V\.|B\.V\.)"
    r")(?![A-Za-z])"
)
# Words that mean the capture is document furniture, not a party to it.
_NOT_A_NAME_RE = re.compile(
    r"\b(certificate|certifies|report|reference|marking|required|unless|otherwise|"
    r"stated|rabbinic|administrator|scope|address|issued|valid|expiry|page|annex)\b",
    re.IGNORECASE,
)
# "Supplier: ACME Foods" — a labelled value, when the form actually filled it in. Blank
# labels are common (the AMS form prints "SUPPLIER:" with nothing after it), so the
# value has to look like a name before we trust it.
_SUPPLIER_LABEL_RE = re.compile(
    r"(?:supplier|company)\s*(?:name)?\s*[:\-]\s*([^\n|:]{3,80})", re.IGNORECASE
)
# Label text that means the field was left empty (the next label ran straight into it).
_EMPTY_VALUE_RE = re.compile(
    r"^(?:date|name|n/?a|none|tbc|tbd|audit|standard|address|site|clause)\b", re.IGNORECASE
)


def is_certifier(name: str) -> bool:
    """True when a company name is an auditor/accreditor rather than a supplier."""
    low = f"{name.lower()} "
    return any(tok in low for tok in _CERTIFIER_TOKENS)


def _clean_company(name: str) -> str:
    """Collapse whitespace and drop trailing punctuation from a captured name."""
    return re.sub(r"\s+", " ", name).strip(" .,;:-|\u2013\u2014")


# A model asked about a document with no readable text answers conversationally
# ("I don't see any document attached to analyze…") instead of NONE. That reply is
# displayed to users as a supplier name if it isn't caught, so validate the shape of
# the answer, not just its content: company names are short and are not sentences.
_PROSE_RE = re.compile(
    r"\b(i|i'm|don't|do not|cannot|can't|sorry|please|provide|unable|appears|"
    r"there is no|no document|not (?:see|find|provided)|the document|it seems|"
    r"could you|however|analyze)\b",
    re.IGNORECASE,
)
_MAX_NAME_WORDS = 8


def is_valid_supplier_name(name: str) -> bool:
    """True when a candidate looks like a company name rather than prose or a refusal."""
    name = name.strip()
    if not (3 <= len(name) <= 120) or len(name.split()) > _MAX_NAME_WORDS:
        return False
    if "?" in name or _PROSE_RE.search(name):
        return False
    return bool(re.search(r"[A-Za-z]{2}", name)) and not is_certifier(name)


def supplier_from_text(text: str, *, title: str | None = None) -> str | None:
    """Best-effort supplier name from the document's own words — no LLM, no cost.

    Two high-precision rules only; anything less certain is left to the LLM pass, since
    a wrong supplier name is worse than no supplier name:

    1. A filled-in ``Supplier:``/``Company:`` label.
    2. The most-repeated legal-entity name that isn't a certification body — a
       certificate names the company it certifies more often than anyone else.
    """
    for m in _SUPPLIER_LABEL_RE.finditer(text):
        value = _clean_company(m.group(1))
        if len(value) < 3 or _EMPTY_VALUE_RE.match(value) or is_certifier(value):
            continue
        if _NOT_A_NAME_RE.search(value):
            continue  # the label ran into surrounding prose, not a name
        if re.search(r"[A-Za-z]{3}", value):
            return value[:120]

    counts: Counter[str] = Counter()
    for m in _COMPANY_RE.finditer(f"{title or ''}\n{text}"):
        name = _clean_company(m.group(1))
        if len(name) > 4 and not is_certifier(name) and not _NOT_A_NAME_RE.search(name):
            counts[name] += 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: (kv[1], len(kv[0])))[0][:120]


def canonical_supplier(names: Counter[str]) -> str | None:
    """Pick one display name for a supplier from the names its documents produced.

    Documents under one supplier code spell their owner differently — "Northwind Dairy and
    Drinks Pty Ltd", "NWD Milk Pty Ltd" and "Northwind F&B - Southgate" are all SUPPLIER0005 —
    and an occasional document yields a stray (the individual who signed a certificate).
    Voting across the code's documents settles on the spelling the corpus uses and drops
    the one-offs.

    Two normalisations before the vote:

    * case and punctuation are ignored, so "FABRIKAM Australia Pty Ltd" and "Fabrikam
      Australia Pty. Ltd" are one candidate;
    * a name that is the leading part of a longer one is folded into it, because the
      fuller name is the more useful label — "Contoso" appearing on five documents and
      "CONTOSO AUSTRALIA PTY LTD" on three is one supplier, and the legal name is what
      belongs beside the code.

    The winning group's most common *complete* spelling is what gets displayed.
    """
    if not names:
        return None
    groups: dict[str, Counter[str]] = {}
    for name, count in names.items():
        key = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
        if key:
            groups.setdefault(key, Counter())[name] += count

    # Fold each key into the longest key it prefixes ("contoso" -> "contoso australia
    # pty ltd"). Longest first so a chain collapses onto the fullest name.
    merged: dict[str, Counter[str]] = {}
    for key in sorted(groups, key=len, reverse=True):
        host = next((h for h in merged if h.startswith(f"{key} ")), key)
        merged.setdefault(host, Counter()).update(groups[key])

    best = max(merged.values(), key=lambda variants: (sum(variants.values()),
                                                      max(len(v) for v in variants)))
    # Within the winner, take the fullest spelling that is still well attested. Requiring
    # half the leading variant's votes keeps the legal name ("CONTOSO AUSTRALIA PTY LTD"
    # over a bare "Contoso") while rejecting a one-off that trails an ABN or an address.
    threshold = max(best.values()) / 2
    return max((v for v, c in best.items() if c >= threshold), key=len)


def _first_regulation(text: str) -> str | None:
    m = _REG_RE.search(text)
    return " ".join(m.group(0).split()).upper() if m else None


def _find(pattern: re.Pattern[str], text: str) -> str | None:
    m = pattern.search(text)
    return m.group(1) if m else None


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    value = value.strip()
    m = _ISO_DATE.search(value)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = _DMY_DATE.search(value)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(y, mo, d)
        except ValueError:
            return None
    return None


_MONTHS = {
    m: i for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        start=1,
    )
}
# Keywords that precede an expiry/validity date on certificates and specs.
_EXPIRY_KEY = re.compile(
    r"(?:exp(?:iry|ires?|iration|\.)?|valid\s*(?:until|to|thru|through)|"
    r"use\s*by|best\s*before|renewal(?:\s*date)?|expiration(?:\s*date)?)\b",
    re.IGNORECASE,
)


def _norm_year(y: int) -> int:
    return y + 2000 if y < 100 else y


def _parse_any_date(s: str) -> date | None:
    """Parse the first date at the start of ``s`` in any common format."""
    s = s.strip(" :=\t")
    # ISO: 2025-08-01
    m = re.match(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if m:
        try:
            return date(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            pass
    # DD Mon YYYY: 01 Aug 2025 / 1-Aug-25
    m = re.match(r"(\d{1,2})[ \-/.]*([A-Za-z]{3,9})[ \-/.,]*(\d{2,4})", s)
    if m and m[2][:3].lower() in _MONTHS:
        try:
            return date(_norm_year(int(m[3])), _MONTHS[m[2][:3].lower()], int(m[1]))
        except ValueError:
            pass
    # Mon DD YYYY: Aug 01 2025
    m = re.match(r"([A-Za-z]{3,9})[ \-/.]*(\d{1,2})[ \-/.,]*(\d{2,4})", s)
    if m and m[1][:3].lower() in _MONTHS:
        try:
            return date(_norm_year(int(m[3])), _MONTHS[m[1][:3].lower()], int(m[2]))
        except ValueError:
            pass
    # DD MM YYYY (space or separator, day-first): 01 08 2025 / 31/12/2024 / 31.12.24
    m = re.match(r"(\d{1,2})[ \-/.](\d{1,2})[ \-/.](\d{2,4})", s)
    if m:
        try:
            return date(_norm_year(int(m[3])), int(m[2]), int(m[1]))
        except ValueError:
            pass
    return None


def parse_expiry(text: str) -> date | None:
    """Find an expiry/validity date near an expiry keyword (title or body)."""
    for km in _EXPIRY_KEY.finditer(text):
        found = _parse_any_date(text[km.end() : km.end() + 24])
        if found:
            return found
    return None


def _keywords(text: str, *, top: int = 10) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", text.lower())
    counts = Counter(t for t in tokens if t not in _STOPWORDS)
    return [w for w, _ in counts.most_common(top)]


def _title_from(content: ExtractedContent, source_key: str) -> str | None:
    if content.toc:
        return content.toc[0].title
    for page in content.pages:
        for block in page.blocks:
            if block.role == "heading" and block.text.strip():
                return block.text.strip()[:200]
    # Fall back to a cleaned filename.
    name = source_key.rsplit("/", 1)[-1]
    name = re.sub(r"[~.](docx?|pdf|xlsx?|pptx?|html?|txt|xml)$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^[0-9a-f]{8,}_", "", name)  # strip hash prefix
    return name.replace("_", " ").replace("~", " ").strip() or None


_LLM_SYSTEM = (
    "You extract structured compliance metadata from a document. Respond with ONLY a "
    "single JSON object, no prose, no markdown fences. Keys (use null when unknown): "
    "title, author, country, region, department, business_unit, supplier, product, "
    "regulation, version, revision, category, sub_category, "
    "effective_date (YYYY-MM-DD|null), expiry_date (YYYY-MM-DD|null), "
    "keywords (array of strings), entities (array of {\"text\":..,\"type\":..}). "
    "Base every value only on the provided text; do not guess."
)


# Supplier identification is its own small prompt rather than a field on the big
# extraction prompt above: it has to run for MongoDB-backed documents too (where the
# big prompt is skipped), and it needs these exclusions spelled out. A certificate's
# most prominent name is the certifier's, and a compliance pack's most repeated name is
# often the customer's — neither is the supplier.
_SUPPLIER_SYSTEM = (
    "You identify the SUPPLIER a compliance document belongs to: the company being "
    "certified, audited, or supplying the goods. Ignore the certification body, "
    "accreditation body, auditor or laboratory (e.g. BSI, SGS, NSF, Intertek, Bureau "
    "Veritas, BRCGS, FSSC, UKAS) and ignore the customer/retailer the goods are supplied "
    "to. Never answer with a person's name — certificates are signed by an individual who "
    "is not the supplier. Reply with ONLY the supplier's company name exactly as written "
    "in the document, including its legal suffix. If the document does not name a "
    "supplier company, reply exactly NONE."
)


# The supplier is named in a document's opening matter (letterhead, certificate front
# page, form header), so the supplier prompt reads a shorter window than the full facet
# prompt — same answers, roughly half the input tokens across a corpus-wide pass.
_SUPPLIER_MAX_CHARS = 6000


class MetadataExtractor:
    def __init__(self, llm: LLM | None = None, *, use_llm: bool = True, max_chars: int = 8000) -> None:
        self._llm = llm
        self._use_llm = use_llm and llm is not None
        self._max_chars = max_chars

    async def extract(
        self,
        content: ExtractedContent,
        *,
        document_id: str,
        source_key: str,
        document_class: DocumentClass = DocumentClass.OTHER,
        language: Language = Language.UNKNOWN,
        pii: list[PIIFinding] | None = None,
        use_llm: bool | None = None,
    ) -> DocumentMetadata:
        # ``use_llm`` overrides the instance default per call — the indexer disables
        # the LLM when an authoritative Mongo record already supplies the facets.
        want_llm = self._use_llm if use_llm is None else (use_llm and self._llm is not None)
        text = content.full_text[: self._max_chars]

        meta = DocumentMetadata(
            document_id=document_id,
            title=_title_from(content, source_key),
            document_class=document_class,
            language=language,
            regulation=_first_regulation(text),
            version=_find(_VERSION_RE, text),
            revision=_find(_REVISION_RE, text),
            keywords=_keywords(text),
            pii=pii or [],
            extra={"source_key": source_key},
        )

        # Expiry/validity date via heuristics (runs for every doc, incl. Mongo ones
        # where the LLM is skipped) — certificates carry it in the title/filename/body.
        scan = " ".join(filter(None, [meta.title, source_key.rsplit("/", 1)[-1], text[:6000]]))
        meta.expiry_date = parse_expiry(scan)

        if want_llm and text.strip():
            try:
                await self._enrich_with_llm(meta, text)
                meta.extraction_confidence = 0.85
            except Exception as exc:
                _log.warning("metadata_llm_failed", document_id=document_id, error=str(exc))
                meta.extraction_confidence = 0.4
        else:
            meta.extraction_confidence = 0.4
        return meta

    async def extract_supplier(
        self, content: ExtractedContent, *, title: str | None = None
    ) -> str | None:
        """The supplier named INSIDE the document — heuristics first, LLM if needed.

        Separate from ``extract`` so it can run for documents whose other facets come
        from the system of record: MongoDB knows the supplier's *code* but not its name,
        and the name is only ever written in the document.
        """
        text = content.full_text[: min(self._max_chars, _SUPPLIER_MAX_CHARS)]
        if not text.strip():
            return None
        if self._llm is None:  # no model configured — heuristics are all we have
            return supplier_from_text(text, title=title)
        try:
            result = await self._llm.complete(
                [{"role": "user", "content": [{"text": text}]}],
                system=_SUPPLIER_SYSTEM,
                max_tokens=40,
                temperature=0.0,
            )
        except Exception as exc:
            _log.warning("supplier_llm_failed", error=str(exc))
            return supplier_from_text(text, title=title)
        name = _clean_company(result.text.splitlines()[0] if result.text else "")
        if not name or name.upper().startswith("NONE"):
            return None
        # Rejects both the auditor and a conversational non-answer.
        return name if is_valid_supplier_name(name) else None

    async def _enrich_with_llm(self, meta: DocumentMetadata, text: str) -> None:
        assert self._llm is not None
        result = await self._llm.complete(
            [{"role": "user", "content": [{"text": text}]}],
            system=_LLM_SYSTEM,
            max_tokens=700,
            temperature=0.0,
        )
        data = _parse_json(result.text)
        if not isinstance(data, dict):
            return

        for field in (
            "title", "author", "country", "region", "department", "business_unit",
            "supplier", "product", "regulation", "version", "revision", "category",
            "sub_category",
        ):
            val = data.get(field)
            if val and (getattr(meta, field, None) in (None, "")):
                setattr(meta, field, str(val))

        if (eff := _parse_date(_as_str(data.get("effective_date")))) is not None:
            meta.effective_date = eff
        if (exp := _parse_date(_as_str(data.get("expiry_date")))) is not None:
            meta.expiry_date = exp

        kws = data.get("keywords")
        if isinstance(kws, list) and kws:
            meta.keywords = [str(k) for k in kws][:15]

        ents = data.get("entities")
        if isinstance(ents, list):
            meta.entities = [
                Entity(text=str(e.get("text", "")), type=str(e.get("type", "")))
                for e in ents
                if isinstance(e, dict) and e.get("text")
            ][:30]


def _as_str(v: Any) -> str | None:
    return str(v) if v not in (None, "") else None


def _parse_json(text: str) -> Any:
    """Parse a JSON object from an LLM response, tolerating code fences/prose."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :] if "{" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
