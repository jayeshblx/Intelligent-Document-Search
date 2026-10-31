"""Retrieval-quality eval harness for the ExampleApp search API.

Builds ground-truth questions from the ACTUAL indexed corpus — so every question has a
known-correct document — then asks them through /v1/chat exactly as a user would, and
scores three things that decide whether the product feels trustworthy:

  * source recall   — did the right document come back (as a citation or a listed doc)?
  * answer rate     — did it answer, or wrongly say "I could not find it"?
  * grounding       — when it answered, did it cite a source (vs. ungrounded prose)?

The categories mirror how real users phrase things: exact codes, numeric ids, "who
created", "by supplier", document types, and content questions built from a distinctive
phrase lifted out of a real chunk. No LLM judge — every check is deterministic against
ground truth, so the score is reproducible and cheap.

    python scripts/eval_harness.py                 # ~120 questions
    python scripts/eval_harness.py --per-cat 40    # bigger sample
    python scripts/eval_harness.py --url http://127.0.0.1:8004
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Deterministic sampling without Math.random-style flakiness: a fixed stride over rows.
STRIDE_SEED = 7


# How a category is scored. LISTING categories return a filtered set, so correctness is
# "did the right filter get applied" (the DB query is then exact by construction).
# RETRIEVAL categories must surface a specific document, scored by source_key.
LISTING_CATS = {"object_code", "created_by", "business_object"}


@dataclass
class Case:
    category: str
    question: str
    expect: str            # RETRIEVAL: source_key substring. LISTING: (facet_key, value).
    expect_facet: str = ""  # LISTING only: the facet key the filter should use.
    note: str = ""


@dataclass
class Result:
    case: Case
    answered: bool
    source_hit: bool
    grounded: bool
    sources: list[str] = field(default_factory=list)
    answer_head: str = ""
    error: str = ""


def _rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return conn.execute(sql, params).fetchall()


def _sample(items: list, n: int) -> list:
    if len(items) <= n:
        return items
    stride = max(1, len(items) // n)
    return [items[(i * stride + STRIDE_SEED) % len(items)] for i in range(n)]


def _basename_key(source_key: str) -> str:
    """The distinctive part of an S3 key: drop the foxtrot/<hash>_ prefix."""
    base = source_key.split("/")[-1]
    return re.sub(r"^[0-9a-f]{8,}_", "", base)


def build_cases(data_dir: Path, per_cat: int) -> list[Case]:
    md = sqlite3.connect(str(data_dir / "metadata.db"))
    kw = sqlite3.connect(str(data_dir / "keyword.db"))
    cases: list[Case] = []

    # 1. Exact object_code lookup — a code the user would paste verbatim.
    codes = _rows(
        md,
        "SELECT df.value code, dm.source_key sk FROM document_facet df "
        "JOIN document_metadata dm ON dm.document_id = df.document_id "
        "WHERE df.key='object_code' AND df.value != '' GROUP BY df.value",
    )
    for r in _sample(codes, per_cat):
        cases.append(Case("object_code", f"find the document with object code {r['code']}",
                          r["code"], expect_facet="object_code"))

    # 2. created_by listing — "documents by <user>".
    authors = _rows(
        md,
        "SELECT value v, COUNT(*) c FROM document_facet WHERE key='created_by' AND value!='' "
        "GROUP BY value ORDER BY c DESC",
    )
    for r in _sample(authors, per_cat):
        cases.append(Case("created_by", f"list all documents created by {r['v']}", r["v"],
                          expect_facet="created_by"))

    # 3. business_object listing.
    bos = _rows(
        md,
        "SELECT value v FROM document_facet WHERE key='business_object' AND value!='' GROUP BY value",
    )
    for r in _sample(bos, per_cat):
        cases.append(Case("business_object", f"show all {r['v']} documents", r["v"],
                          expect_facet="business_object"))

    # 4. Exact numeric identifier from real content — but only DISTINCTIVE ones. A number
    #    in 87 documents (a recurring spec value) isn't a "which specific document"
    #    question; a true record id (invoice/delivery no) appears in 1-3 documents.
    idchunks = _rows(
        kw,
        "SELECT text, json_extract(payload,'$.source_key') sk FROM chunks "
        "WHERE text GLOB '*[0-9][0-9][0-9][0-9][0-9][0-9]*' LIMIT 4000",
    )
    id_cases: list[Case] = []
    seen_nums: set[str] = set()
    for r in idchunks:
        m = re.search(r"\b(\d{6,})\b", r["text"] or "")
        if not (m and r["sk"]):
            continue
        num = m.group(1)
        if num in seen_nums:
            continue
        seen_nums.add(num)
        ndocs = kw.execute(
            "SELECT COUNT(DISTINCT json_extract(payload,'$.source_key')) FROM chunks "
            "WHERE text LIKE ?",
            (f"%{num}%",),
        ).fetchone()[0]
        if 1 <= ndocs <= 3:  # distinctive: a genuine record identifier
            id_cases.append(Case("exact_id", f"which document mentions {num}",
                                 _basename_key(r["sk"]), note=num))
    for c in _sample(id_cases, per_cat):
        cases.append(c)

    # 5. Content question from a distinctive phrase in a real chunk — expects that doc.
    content = _rows(
        kw,
        "SELECT text, json_extract(payload,'$.source_key') sk FROM chunks "
        "WHERE length(text) > 200 LIMIT 5000",
    )
    phrase_cases: list[Case] = []
    for r in content:
        # A distinctive 5-word run of alphabetic words. Skip OCR garbage: runs with an
        # immediately-repeated word ("Ultra Weight Ultra Weight") aren't real questions.
        words = re.findall(r"[A-Za-z]{4,}", r["text"] or "")
        if len(words) >= 10 and r["sk"]:
            phrase_words = words[3:8]
            if len(set(phrase_words)) < 5:  # a repeat means garbled text — skip
                continue
            phrase = " ".join(phrase_words)
            phrase_cases.append(Case("content", f"what does the document say about {phrase}",
                                     _basename_key(r["sk"])[:20], note=phrase))
    for c in _sample(phrase_cases, per_cat):
        cases.append(c)

    # 6. Document-by-name — a filename the user might half-remember.
    names = _rows(md, "SELECT source_key sk FROM document_metadata")
    name_cases: list[Case] = []
    for r in names:
        nm = _basename_key(r["sk"])
        nm = re.sub(r"[~.][A-Za-z0-9]+$", "", nm)          # drop extension
        title = " ".join(re.findall(r"[A-Za-z0-9]{3,}", nm)[:6])
        if len(title) > 10:
            name_cases.append(Case("by_name", f"find the document {title}", nm[:20], note=title))
    for c in _sample(name_cases, per_cat):
        cases.append(c)

    md.close()
    kw.close()
    return cases


_NOT_FOUND = re.compile(r"could(?:\s+not|n.?t)\s+find|do(?:n.?t| not) have enough", re.I)


def _source_discusses_phrase(kw: sqlite3.Connection, source_keys: list[str], phrase: str) -> bool:
    """Does any cited document's content actually contain most of the phrase's words?

    Strict content check: rather than accept any grounded answer, confirm the source the
    answer cited genuinely discusses what was asked — a phrase repeats across documents,
    so 'the exact doc' is unfair, but 'a document that really contains these words' is not.
    """
    words = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", phrase)]
    if not words or not source_keys:
        return False
    need = max(2, int(len(words) * 0.6))  # a clear majority of the phrase's words
    for sk in source_keys:
        if not sk:
            continue
        rows = kw.execute(
            "SELECT lower(group_concat(text, ' ')) t FROM chunks "
            "WHERE json_extract(payload,'$.source_key') = ?",
            (sk,),
        ).fetchone()
        blob = (rows[0] if rows else "") or ""
        if sum(1 for w in words if w in blob) >= need:
            return True
    return False


def _source_contains_id(kw: sqlite3.Connection, source_keys: list[str], identifier: str) -> bool:
    """Does any returned document's content literally contain the identifier?"""
    if not identifier or not source_keys:
        return False
    for sk in source_keys:
        if sk and kw.execute(
            "SELECT 1 FROM chunks WHERE json_extract(payload,'$.source_key') = ? "
            "AND text LIKE ? LIMIT 1",
            (sk, f"%{identifier}%"),
        ).fetchone():
            return True
    return False


def run_case(url: str, case: Case, uid: str, kw: sqlite3.Connection | None = None) -> Result:
    body = json.dumps({"message": case.question}).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat", data=body,
        headers={"content-type": "application/json", "X-User-Id": uid},
    )
    try:
        with urllib.request.urlopen(req, timeout=150) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        return Result(case, False, False, False, error=str(exc)[:80])

    answer = data.get("answer", "") or ""
    cites = data.get("citations", []) or []
    docs = data.get("documents", []) or []
    dquery = data.get("documents_query", {}) or {}
    sources = [c.get("source_key", "") for c in cites] + [d.get("source_key", "") for d in docs]

    answered = not _NOT_FOUND.search(answer)
    if case.category in LISTING_CATS:
        # Correct = the right filter was applied and returned a non-empty set. The DB
        # query is then exact by construction, so the returned set is correct.
        applied = str(dquery.get(case.expect_facet, "")).lower()
        source_hit = applied == case.expect.lower() and (data.get("documents_total", 0) > 0)
    elif case.category == "content":
        # Strict: the answer must be grounded in a document that actually contains the
        # phrase's words — not merely "answered and cited something".
        source_hit = bool(answered and kw and _source_discusses_phrase(kw, sources, case.note))
    elif case.category == "exact_id":
        # Strict-but-fair: a returned document must literally contain the identifier. A
        # number can recur across near-identical documents, so requiring the one it was
        # lifted from is wrong — "a document that genuinely contains it" is the real bar.
        source_hit = bool(answered and kw and _source_contains_id(kw, sources, case.note))
    else:
        needle = case.expect.lower()
        source_hit = any(needle in (s or "").lower() for s in sources)
    grounded = bool(cites or docs)
    return Result(case, answered, source_hit, grounded, sources[:3], answer[:70])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8004")
    ap.add_argument("--per-cat", type=int, default=20)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services/search/src"))
        from rag_search.config import SearchSettings

        data_dir = Path(SearchSettings().data_dir)

    cases = build_cases(data_dir, args.per_cat)
    print(f"Built {len(cases)} ground-truth cases from {data_dir}\n")

    kw = sqlite3.connect(str(data_dir / "keyword.db"))  # content-verification lookups
    results: list[Result] = []
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        r = run_case(args.url, case, f"eval-{case.category}-{i}", kw)
        results.append(r)
        flag = "OK " if r.source_hit else ("NF " if not r.answered else "MISS")
        print(f"  [{flag}] {case.category:<15} {case.question[:58]}")
    kw.close()

    # ---- scorecard ----
    print(f"\n{'='*66}\nSCORECARD  ({len(results)} cases, {time.time()-t0:.0f}s)\n{'='*66}")
    by_cat: dict[str, list[Result]] = {}
    for r in results:
        by_cat.setdefault(r.case.category, []).append(r)

    def pct(num: int, den: int) -> str:
        return f"{100*num/den:5.1f}%" if den else "   n/a"

    print(f"{'category':<16} {'n':>3}  {'src recall':>10} {'answer rate':>12} {'grounded':>9}")
    print("-" * 66)
    tot_hit = tot_ans = tot_grnd = tot = 0
    for cat, rs in by_cat.items():
        n = len(rs)
        hit = sum(r.source_hit for r in rs)
        ans = sum(r.answered for r in rs)
        grnd = sum(r.grounded for r in rs)
        tot_hit += hit
        tot_ans += ans
        tot_grnd += grnd
        tot += n
        print(f"{cat:<16} {n:>3}  {pct(hit,n):>10} {pct(ans,n):>12} {pct(grnd,n):>9}")
    print("-" * 66)
    print(f"{'OVERALL':<16} {tot:>3}  {pct(tot_hit,tot):>10} {pct(tot_ans,tot):>12} {pct(tot_grnd,tot):>9}")

    # False negatives: doc IS indexed but the app said "could not find".
    fn = [r for r in results if not r.answered]
    print(f"\nFalse 'could not find' (doc IS indexed): {len(fn)}/{tot} ({pct(len(fn),tot).strip()})")
    worst = Counter(r.case.category for r in results if not r.source_hit)
    print("Missed-source by category:", dict(worst.most_common()))

    # Persist for diffing across runs.
    out = data_dir / "eval_results.json"
    out.write_text(json.dumps(
        [{"cat": r.case.category, "q": r.case.question, "answered": r.answered,
          "hit": r.source_hit, "grounded": r.grounded, "sources": r.sources} for r in results],
        indent=2,
    ))
    print(f"\nDetails written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
