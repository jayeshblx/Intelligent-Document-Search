"""End-to-end conversational RAG test (offline: fake embedder + fake LLM)."""

from __future__ import annotations

from collections.abc import Callable

from rag_search.answer import AnswerService, FakeLLM
from rag_search.conversation import ConversationStore
from rag_search.indexer import IndexingService
from rag_search.models import IndexRequest
from rag_search.retrieval import SearchService

Stack = Callable[[dict[str, bytes]], tuple[IndexingService, SearchService, object, list]]


async def test_chat_answers_with_citations_and_cost(
    build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    indexer, search, _meta, closeables = build_stack(sample_objects)
    try:
        await indexer.index(IndexRequest(reset=True))
        conv = ConversationStore(ttl_minutes=30)
        answerer = AnswerService(search=search, llm=FakeLLM(), conversations=conv, top_k=3)

        sid = conv.new_session()
        resp = await answerer.answer(
            "how must confidential data be encrypted at rest", session_id=sid
        )
        assert resp.session_id == sid
        assert resp.answer  # produced an answer
        assert resp.citations, "answer should be grounded on sources"
        assert resp.citations[0].source_key == "foxtrot/security_policy.docx"
        assert resp.citations[0].index == 1
        # cost accounting populated
        assert resp.usage.output_tokens > 0
        assert resp.usage.total_cost_usd >= 0
        assert resp.context_expires_in_s > 0

        # follow-up maintains conversation context
        await answerer.answer("what about access control", session_id=sid)
        assert len(await conv.recent(sid)) == 4  # 2 user + 2 assistant turns
    finally:
        for c in closeables:
            await c.aclose()


def test_citations_are_relevant_and_deduped() -> None:
    """Only cited passages are returned, one card per document."""
    from rag_search.answer import _select_citations
    from rag_search.models import Passage

    def p(idx: int, key: str) -> Passage:
        return Passage(
            chunk_id=f"c{idx}", document_id=key, source_key=key, score=1.0, text=f"text {idx}"
        )

    passages = [p(1, "foxtrot/a.pdf"), p(2, "foxtrot/a.pdf"), p(3, "foxtrot/b.pdf"), p(4, "foxtrot/c.pdf")]
    # Model cited [1] and [3]; [2] is a duplicate of doc a, [4] uncited.
    cites = _select_citations(passages, "The policy says X [1] and also Y [3].")
    keys = [c.source_key for c in cites]
    assert keys == ["foxtrot/a.pdf", "foxtrot/b.pdf"]  # cited + deduped; c.pdf excluded
    # No inline citations (e.g. "could not find") -> no sources, not all passages.
    assert _select_citations(passages, "I could not find that in the documents.") == []


def test_uncited_summary_answer_still_shows_its_sources() -> None:
    """A grounded answer that writes no [n] must not lose its provenance.

    The system prompt asks for count+characterisation summaries, and that style cites
    nothing — which silently stripped every source from real answers.
    """
    from rag_search.answer import _select_citations
    from rag_search.models import Passage

    def p(idx: int, key: str) -> Passage:
        return Passage(
            chunk_id=f"c{idx}", document_id=key, source_key=key, score=1.0, text=f"text {idx}"
        )

    passages = [p(1, "foxtrot/a.pdf"), p(2, "foxtrot/b.pdf"), p(3, "foxtrot/c.pdf")]
    summary = "There are 173 halal certificates, issued between 2022 and 2028."

    # Without a fallback the answer is left with no sources at all — the reported bug.
    assert _select_citations(passages, summary) == []

    # With one, the context that produced the answer is shown, capped and deduped.
    cites = _select_citations(passages, summary, fallback_limit=2)
    assert [c.source_key for c in cites] == ["foxtrot/a.pdf", "foxtrot/b.pdf"]

    # The fallback must never rescue a "could not find" reply: nothing grounded it.
    assert _select_citations(passages, "I could not find that.", fallback_limit=3) == []

    # An explicit citation still wins over the fallback (precise beats approximate).
    exact = _select_citations(passages, "Yes, per the policy [3].", fallback_limit=3)
    assert [c.source_key for c in exact] == ["foxtrot/c.pdf"]


def test_fallback_sources_exclude_weak_passages() -> None:
    """The fallback must not surface the irrelevant tail of a follow-up's retrieval.

    Retrieval always returns top_k, and a follow-up blends earlier turns into the query,
    so the low-scoring tail is often about the PREVIOUS question. Showing it as a source
    for this answer is worse than showing nothing.
    """
    from rag_search.answer import _select_citations
    from rag_search.models import Passage

    def p(idx: int, key: str, score: float) -> Passage:
        return Passage(
            chunk_id=f"c{idx}", document_id=key, source_key=key, score=score, text=f"t{idx}"
        )

    # Mirrors a real run: two strong hits, then noise carried over from the last question.
    passages = [p(1, "foxtrot/a.pdf", 1.0), p(2, "foxtrot/b.pdf", 0.9), p(3, "foxtrot/stale.pdf", 0.031)]
    cites = _select_citations(passages, "A summary with no bracket markers.", fallback_limit=6)
    keys = [c.source_key for c in cites]
    assert keys == ["foxtrot/a.pdf", "foxtrot/b.pdf"], "weak tail must be dropped"
    assert "foxtrot/stale.pdf" not in keys

    # But if the model explicitly cites the weak one, trust the model over the score.
    exact = _select_citations(passages, "See [3].", fallback_limit=6)
    assert [c.source_key for c in exact] == ["foxtrot/stale.pdf"]

    # All-equal scores: nothing is relatively weak, so nothing is dropped.
    flat = [p(1, "foxtrot/x.pdf", 0.4), p(2, "foxtrot/y.pdf", 0.4)]
    assert len(_select_citations(flat, "No markers here.", fallback_limit=6)) == 2


def test_listing_subject_ignores_type_words_buried_in_names() -> None:
    """A document type is only the listing subject when the listing introduces it.

    Regression: "Find all Documents who has Grade A BRC Global Standard for Food Safety
    Issue 9: August 2022" matched the stray "Standard" and listed all 132 documents of
    type `standard`. The subject there is "all Documents" — generic, so no type filter.
    """
    from rag_search.answer import _listing_subject, _match_facet_value

    # The corpus's real document_class vocabulary.
    values = ["other", "certificate", "standard", "technical_manual", "report",
              "workflow", "specification", "regulation"]

    def resolve(q: str) -> str | None:
        """What _resolve_document_class does, minus the metadata-store round trip."""
        subject = _listing_subject(q)
        return _match_facet_value(subject, values) if subject else None

    # The reported bug: the type word sits inside a proper noun, far from the subject.
    assert resolve("Find all Documents who has Grade A BRC Global Standard "
                   "for Food Safety Issue 9: August 2022") is None
    assert resolve("show me documents about the BRC Global Standard for Food Safety") is None

    # Genuine type listings must still resolve — plural/inflection tolerant, no hardcoding.
    assert resolve("list all certificates") == "certificate"
    assert resolve("list all certifications") == "certificate"
    assert resolve("show me all halal certificates") == "certificate"
    assert resolve("how many specifications are there") == "specification"
    assert resolve("show all reports for Northwind") == "report"
    assert resolve("all standards") == "standard"

    # No listing opener at all -> nothing to resolve (dclass is only ever a list filter).
    assert resolve("what does the certificate say about storage") is None


async def test_new_topic_does_not_inherit_the_previous_conversation(
    build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    """A question on a new subject must not be answered against the old thread.

    Real report: after a chain about BRC Grade A certificates, "WE MISSING 1 BOX ORANGE
    SODA..." was answered with the BRC findings, and the next BRC question was refused
    as a delivery complaint. Retrieval knew both were new topics; the model was still
    being handed the whole conversation.
    """
    from typing import Any

    from rag_search.answer import LLM, AnswerService
    from rag_search.conversation import ConversationStore
    from rag_search.models import IndexRequest

    from rag_core.aws import GenerationResult

    seen: list[list[dict[str, Any]]] = []

    class RecordingLLM(LLM):
        async def complete(self, messages, *, system, max_tokens, temperature, cache_system=False):
            seen.append(messages)
            return GenerationResult(text="ok [1]", input_tokens=5, output_tokens=2, model="f")

    indexer, search, meta, closeables = build_stack(sample_objects)
    try:
        await indexer.index(IndexRequest(reset=True))
        conv = ConversationStore(ttl_minutes=30)
        svc = AnswerService(
            search=search, llm=RecordingLLM(), conversations=conv, metadata=meta, top_k=3
        )
        sid = conv.new_session()

        await svc.answer("how must confidential data be encrypted at rest", session_id=sid)
        assert len(seen[0]) == 1, "first turn has no history to send"

        # A referential follow-up SHOULD see the thread, including the prior answer.
        await svc.answer("what about it in transit", session_id=sid)
        assert any(m["role"] == "assistant" for m in seen[1]), "a follow-up keeps its context"

        # An unrelated new subject must NOT — this is the reported bug. Asserting on roles
        # rather than words: the retrieved *context* may legitimately mention anything.
        await svc.answer(
            "WE MISSING 1 BOX ORANGE SODA FROM OUR TODAY DELIVERY PLEASE GIVE US CREDIT",
            session_id=sid,
        )
        assert len(seen[2]) == 1, "a new topic must not inherit the previous conversation"
        assert not any(m["role"] == "assistant" for m in seen[2]), "no prior answers leak in"

        # The session itself is untouched: history persists until New chat.
        assert len(await conv.recent(sid)) == 6  # 3 user + 3 assistant turns
    finally:
        for c in closeables:
            await c.aclose()


async def test_listing_is_suppressed_when_the_answer_disowns_it(
    build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    """A "could not find" answer must not ship a document list contradicting it.

    Real report: "Find all Documents who has Grade A BRC Global Standard..." matched the
    word "Standard" to document type `standard`, listed 132 unrelated documents, and the
    model then said it could not find anything — so the UI asserted both at once.
    """
    from rag_search.answer import LLM, AnswerService
    from rag_search.conversation import ConversationStore
    from rag_search.models import IndexRequest

    from rag_core.aws import GenerationResult

    class NotFoundLLM(LLM):
        async def complete(self, messages, *, system, max_tokens, temperature, cache_system=False):
            return GenerationResult(
                text="I could not find any documents with that in the indexed documents.",
                input_tokens=10,
                output_tokens=5,
                model="fake",
            )

    indexer, search, meta, closeables = build_stack(sample_objects)
    try:
        await indexer.index(IndexRequest(reset=True))
        conv = ConversationStore(ttl_minutes=30)
        answerer = AnswerService(
            search=search, llm=NotFoundLLM(), conversations=conv, metadata=meta, top_k=3
        )
        resp = await answerer.answer("list all documents", session_id=conv.new_session())
        assert resp.documents == [], "a disowned listing must not be shown"
        assert resp.documents_total == 0
        assert resp.documents_query == {}
        assert resp.citations == []  # "could not find" is ungrounded either way
    finally:
        for c in closeables:
            await c.aclose()


async def test_facet_detection_routes_to_filter(
    build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    indexer, search, meta, closeables = build_stack(sample_objects)
    try:
        await indexer.index(IndexRequest(reset=True))
        answerer = AnswerService(
            search=search, llm=FakeLLM(), conversations=ConversationStore(ttl_minutes=30),
            metadata=meta,
        )
        # The supplier register mentions ISO 9001 -> a detectable 'regulation' facet.
        detected = await answerer._detect_filters("which suppliers hold ISO 9001", "default")
        assert detected.get("regulation") == "ISO 9001"
        # An unrelated question detects no facet value.
        assert await answerer._detect_filters("how is data encrypted at rest", "default") == {}
    finally:
        for c in closeables:
            await c.aclose()


def test_detect_expiry_range() -> None:
    from datetime import date

    from rag_search.answer import AnswerService

    today = date(2026, 7, 3)  # a Friday
    f = AnswerService._detect_expiry_range
    assert f("which documents are expiring next month", today) == ("2026-08-01", "2026-08-31")
    assert f("what certificates expired yesterday", today) == ("2026-07-02", "2026-07-02")
    assert f("certificates that expire today", today) == ("2026-07-03", "2026-07-03")
    assert f("show expired certificates", today) == ("1900-01-01", "2026-07-02")
    assert f("documents expiring in 14 days", today) == ("2026-07-03", "2026-07-17")
    # "last N months/days" -> [today - N, today] (the screenshot bug).
    assert f("how many expired in the last 2 months", today) == ("2026-05-03", "2026-07-03")
    assert f("certificates expired in the past 30 days", today) == ("2026-06-03", "2026-07-03")
    # Absolute dates.
    assert f("which certificates expired in 2024", today) == ("2024-01-01", "2024-12-31")
    assert f("expired before 2025", today) == ("1900-01-01", "2024-12-31")
    assert f("which expire after 2026", today) == ("2027-01-01", "9999-12-31")
    assert f("expire between January and December 2027", today) == ("2027-01-01", "2027-12-31")
    assert f("expiring in Q1 2027", today) == ("2027-01-01", "2027-03-31")
    assert f("which certificates are valid today", today) == ("2026-07-03", "9999-12-31")
    # Future / still-valid -> today .. far future (the bug from the screenshot).
    assert f("which certificates will expire in future", today) == ("2026-07-03", "9999-12-31")
    assert f("which certificates are still valid", today) == ("2026-07-03", "9999-12-31")
    # Bare expiry query -> every dated document.
    assert f("list certificates with an expiry date", today) == ("1900-01-01", "9999-12-31")
    assert f("how is confidential data encrypted", today) is None  # not a date query


def test_listing_vs_content_intent() -> None:
    from rag_search.answer import _CONTENT_INTENT, _LISTING_INTENT

    # Listing questions.
    assert _LISTING_INTENT.search("which certificates expired in 2024")
    assert _LISTING_INTENT.search("how many documents are in draft")
    # A single-item lookup that mentions a date is NOT a listing.
    assert not _LISTING_INTENT.search("what is the expiry date of the Southgate certificate")
    assert _CONTENT_INTENT.search("what is the expiry date of the Southgate FSSC 2024 certificate")
    assert _CONTENT_INTENT.search("when does the certificate expire")
    assert not _CONTENT_INTENT.search("certificates expiring next month")


async def test_facet_detection_guards(tmp_path) -> None:
    from rag_search.answer import AnswerService, FakeLLM
    from rag_search.conversation import ConversationStore
    from rag_search.metadata_store import MetadataStore

    from rag_core.models.enums import Language
    from rag_core.models.metadata import DocumentMetadata

    store = MetadataStore(str(tmp_path / "m.db"))
    try:
        await store.put(
            DocumentMetadata(document_id="a", language=Language.EN),
            namespace="default", source_key="foxtrot/a.pdf",
            facets={"status": "Active", "business_object": "SUPPLIERDOCUMENTS",
                    "object_name": "SUPPLIERDOCUMENTS", "object_code": "SUPPLIER0005", "tenant": "FOXTROT"},
        )
        svc = AnswerService(
            search=None, llm=FakeLLM(), conversations=ConversationStore(ttl_minutes=30), metadata=store,
        )  # type: ignore[arg-type]
        # Code-like values are always detected.
        got = await svc._detect_filters("tell me about SUPPLIER0005", "default")
        assert got.get("object_code") == "SUPPLIER0005"
        # A common word ("active") only becomes a filter with a listing cue.
        assert "status" not in await svc._detect_filters("give me an active summary", "default")
        assert (await svc._detect_filters("which documents are active", "default")).get("status") == "Active"
        # Negation drops the facet (so we don't filter to the opposite).
        assert "status" not in await svc._detect_filters("which documents are not active", "default")
        # A value shared by two facets maps to ONE (no over-constrained AND filter).
        d = await svc._detect_filters("how many documents have business object SUPPLIERDOCUMENTS", "default")
        assert d == {"business_object": "SUPPLIERDOCUMENTS"}
        # The named facet wins when a value is shared ("object name X" -> object_name).
        d2 = await svc._detect_filters("how many documents have object name SUPPLIERDOCUMENTS", "default")
        assert d2 == {"object_name": "SUPPLIERDOCUMENTS"}
        # Short codes (2 chars) are detectable with a cue.
        assert (await svc._detect_filters("how many documents have tenant FOXTROT", "default")).get("tenant") == "FOXTROT"
    finally:
        await store.aclose()


async def test_object_code_question_tells_model_passages_are_the_document() -> None:
    """A content question that names a metadata value (e.g. an object code) must tell
    the model the retrieved passages ARE that document, so it explains them instead of
    saying it 'could not find' the code in the body text."""
    from typing import Any

    from rag_search.answer import LLM, AnswerService
    from rag_search.conversation import ConversationStore
    from rag_search.models import Passage, SearchResponse

    from rag_core.aws import GenerationResult

    class _FakeSearch:
        async def search(
            self, q: str, *, namespace: str = "default", top_k: int = 6, filters: Any = None
        ) -> SearchResponse:
            p = Passage(
                chunk_id="c1", document_id="d1", source_key="foxtrot/vandelay_haccp.pdf",
                score=1.0, text="Vandelay Supplies HACCP certificate — food hygiene principles.",
            )
            return SearchResponse(query=q, took_ms=1.0, results=[p])

        async def expand_document(self, document_id: str, text: str, *, cap: int) -> str:
            return text

    class _CapturingLLM(LLM):
        def __init__(self) -> None:
            self.prompt = ""

        async def complete(
            self, messages: list[dict[str, Any]], *, system: str, max_tokens: int, temperature: float, cache_system: bool = False
        ) -> GenerationResult:
            self.prompt = messages[-1]["content"][0]["text"]
            return GenerationResult(text="This document is a HACCP certificate [1]",
                                    input_tokens=10, output_tokens=5, model="fake")

    llm = _CapturingLLM()
    answerer = AnswerService(
        search=_FakeSearch(), llm=llm, conversations=ConversationStore(ttl_minutes=30),  # type: ignore[arg-type]
    )
    resp = await answerer.answer(
        "explain the document", session_id="s1", filters={"object_code": "SUPPLIER0009"}
    )
    assert "object_code=SUPPLIER0009" in llm.prompt
    assert "metadata tag" in llm.prompt.lower()
    assert "could not find" in llm.prompt.lower()  # the explicit "don't say" instruction
    assert resp.answer  # produced a grounded explanation, not a not-found reply


async def test_named_document_grounds_from_title_and_metadata(tmp_path) -> None:
    """A question that names a document/type is answered from the TITLE + metadata
    (MongoDB) even when body-text search returns nothing — the fix for 'AMS form' /
    'EHO audit' / object-code questions returning 'could not find'."""
    from typing import Any

    from rag_search.answer import LLM, AnswerService
    from rag_search.conversation import ConversationStore
    from rag_search.metadata_store import MetadataStore
    from rag_search.models import SearchResponse

    from rag_core.aws import GenerationResult
    from rag_core.models.enums import Language
    from rag_core.models.metadata import DocumentMetadata

    class _EmptySearch:  # body-text search finds nothing (generic-boilerplate forms)
        async def search(
            self, q: str, *, namespace: str = "default", top_k: int = 6, filters: Any = None
        ) -> SearchResponse:
            return SearchResponse(query=q, took_ms=1.0, results=[])

        async def expand_document(self, document_id: str, seed_text: str, *, cap: int) -> str:
            return seed_text

    class _CapturingLLM(LLM):
        def __init__(self) -> None:
            self.prompt = ""

        async def complete(
            self, messages: list[dict[str, Any]], *, system: str, max_tokens: int, temperature: float, cache_system: bool = False
        ) -> GenerationResult:
            self.prompt = messages[-1]["content"][0]["text"]
            return GenerationResult(text="The AMS form is an audit form [1]",
                                    input_tokens=8, output_tokens=4, model="fake")

    store = MetadataStore(str(tmp_path / "m.db"))
    llm = _CapturingLLM()
    try:
        await store.put(
            DocumentMetadata(document_id="ams1", language=Language.EN),
            namespace="default", source_key="foxtrot/ams_form.pdf",
            facets={"business_object": "SUPPLIERDOCUMENTS", "object_name": "AMS FORM",
                    "object_code": "SUPPLIER0047", "status": "Draft"},
        )
        async def _presign(key: str) -> str:
            return f"https://signed/{key}"

        answerer = AnswerService(
            search=_EmptySearch(), llm=llm,  # type: ignore[arg-type]
            conversations=ConversationStore(ttl_minutes=30), metadata=store, presign=_presign,
        )
        resp = await answerer.answer(
            "what is the AMS form and when can a supplier use it", session_id="s1"
        )
        # The AMS document was surfaced by title/metadata and put into the prompt...
        assert "object_code=SUPPLIER0047" in llm.prompt
        assert "SUPPLIERDOCUMENTS" in llm.prompt
        # ...and the answer is grounded on it (not a "could not find" reply)...
        assert resp.citations and resp.citations[0].source_key == "foxtrot/ams_form.pdf"
        # ...and the source is downloadable (presigned link attached).
        assert resp.citations[0].download_url == "https://signed/foxtrot/ams_form.pdf"
    finally:
        await store.aclose()


async def test_followup_answers_from_conversation_history() -> None:
    """A short, referential follow-up ("how long is it valid?") that retrieves nothing
    fresh still reaches the model with the prior turns — not a canned "couldn't find" —
    and its retrieval query is blended with the previous question."""
    from typing import Any

    from rag_search.answer import LLM, AnswerService
    from rag_search.conversation import ConversationStore
    from rag_search.models import SearchResponse

    from rag_core.aws import GenerationResult

    class _EmptySearch:
        def __init__(self) -> None:
            self.last_query = ""

        async def search(
            self, q: str, *, namespace: str = "default", top_k: int = 6, filters: Any = None
        ) -> SearchResponse:
            self.last_query = q
            return SearchResponse(query=q, took_ms=1.0, results=[])

        async def expand_document(self, document_id: str, seed_text: str, *, cap: int) -> str:
            return seed_text

    class _CapturingLLM(LLM):
        def __init__(self) -> None:
            self.messages: list[dict[str, Any]] = []

        async def complete(
            self, messages: list[dict[str, Any]], *, system: str, max_tokens: int, temperature: float, cache_system: bool = False
        ) -> GenerationResult:
            self.messages = messages
            return GenerationResult(text="It is valid for one year.",
                                    input_tokens=5, output_tokens=3, model="fake")

    search = _EmptySearch()
    llm = _CapturingLLM()
    conv = ConversationStore(ttl_minutes=30)
    answerer = AnswerService(search=search, llm=llm, conversations=conv)  # type: ignore[arg-type]
    sid = conv.new_session()
    await conv.append(sid, "user", "explain the AMS form")
    await conv.append(sid, "assistant", "The AMS form lets a supplier skip a full audit.")

    resp = await answerer.answer("how long is it valid?", session_id=sid)
    # Answered by the model from context, not the canned bail-out.
    assert resp.answer == "It is valid for one year."
    assert "couldn't find anything relevant" not in resp.answer
    # Prior turns were sent to the model, and retrieval was contextualised.
    all_text = " ".join(m["content"][0]["text"] for m in llm.messages)
    assert "AMS form" in all_text
    assert "AMS form" in search.last_query and "valid" in search.last_query

    # Chained follow-up (2 hops from the AMS mention) must STILL carry the topic — the
    # previous turn ("how long is it valid?") has no "AMS", so blending only that would
    # drop it. Blending the last few user turns keeps "AMS form" in the retrieval query.
    await answerer.answer("who used it and why?", session_id=sid)
    assert "AMS form" in search.last_query


async def test_type_listing_lists_documents_with_links(tmp_path) -> None:
    """"list all certificates" resolves the type to a document_class filter and returns
    the matching documents as a clickable list (with download links), not just a count."""
    from rag_search.answer import AnswerService, FakeLLM, _match_facet_value
    from rag_search.conversation import ConversationStore
    from rag_search.metadata_store import MetadataStore
    from rag_search.models import SearchResponse

    from rag_core.models.enums import Language
    from rag_core.models.metadata import DocumentMetadata

    # Data-driven + inflection-tolerant: certifications / certs -> the real 'certificate'.
    assert _match_facet_value("list all certifications", ["certificate", "report"]) == "certificate"
    assert _match_facet_value("how many certs are there", ["certificate"]) == "certificate"
    assert _match_facet_value("how does encryption work", ["certificate"]) is None

    class _EmptySearch:
        async def search(self, q, *, namespace="default", top_k=6, filters=None):
            return SearchResponse(query=q, took_ms=1.0, results=[])

        async def expand_document(self, document_id, seed_text, *, cap):
            return seed_text

    async def _presign(key: str) -> str:
        return f"https://signed/{key}"

    store = MetadataStore(str(tmp_path / "m.db"))
    try:
        await store.put(
            DocumentMetadata(document_id="c1", language=Language.EN),
            namespace="default", source_key="foxtrot/cert1.pdf",
            facets={"document_class": "certificate", "regulation": "FSSC 22000"},
        )
        answerer = AnswerService(
            search=_EmptySearch(), llm=FakeLLM(),  # type: ignore[arg-type]
            conversations=ConversationStore(ttl_minutes=30), metadata=store, presign=_presign,
        )
        # "certifications" (inflected, never hardcoded) still resolves to the certificate type.
        resp = await answerer.answer("list all certifications", session_id="s1")
        assert resp.documents_total == 1
        assert resp.documents and resp.documents[0].source_key == "foxtrot/cert1.pdf"
        assert resp.documents[0].download_url == "https://signed/foxtrot/cert1.pdf"  # clickable link
    finally:
        await store.aclose()


async def test_type_listing_after_unrelated_turn_stays_on_topic(tmp_path) -> None:
    """After an AMS turn, "show all specifications" must NOT inherit that context (it's a
    fresh type-listing) — it lists specifications and its retrieval isn't polluted."""
    from rag_search.answer import AnswerService, FakeLLM, _is_followup, _match_facet_value
    from rag_search.conversation import ConversationStore
    from rag_search.metadata_store import MetadataStore
    from rag_search.models import SearchResponse

    from rag_core.models.enums import Language
    from rag_core.models.metadata import DocumentMetadata

    # A type-listing is its own topic (not a follow-up); a referential one still is.
    assert _is_followup("show all specifications") is False
    assert _is_followup("how long is it valid?") is True
    assert _match_facet_value("show all specifications", ["specification"]) == "specification"

    class _CapturingSearch:
        def __init__(self) -> None:
            self.last_query = ""

        async def search(self, q, *, namespace="default", top_k=6, filters=None):
            self.last_query = q
            return SearchResponse(query=q, took_ms=1.0, results=[])

        async def expand_document(self, document_id, seed_text, *, cap):
            return seed_text

    async def _presign(key: str) -> str:
        return f"https://signed/{key}"

    store = MetadataStore(str(tmp_path / "m.db"))
    search = _CapturingSearch()
    try:
        await store.put(
            DocumentMetadata(document_id="s1", language=Language.EN),
            namespace="default", source_key="foxtrot/spec1.pdf",
            facets={"document_class": "specification"},
        )
        conv = ConversationStore(ttl_minutes=30)
        answerer = AnswerService(
            search=search, llm=FakeLLM(), conversations=conv,  # type: ignore[arg-type]
            metadata=store, presign=_presign,
        )
        sid = conv.new_session()
        await conv.append(sid, "user", "explain the AMS form")
        await conv.append(sid, "assistant", "The AMS form lets a supplier skip a full audit.")

        resp = await answerer.answer("show all specifications", session_id=sid)
        # Lists the specification (not AMS), and retrieval was NOT contaminated with "AMS".
        assert resp.documents_total == 1 and resp.documents[0].source_key == "foxtrot/spec1.pdf"
        assert "AMS" not in search.last_query
    finally:
        await store.aclose()


async def test_chat_handles_no_results(build_stack: Stack) -> None:
    indexer, search, _meta, closeables = build_stack({})
    try:
        await indexer.index(IndexRequest(reset=True))
        conv = ConversationStore(ttl_minutes=30)
        answerer = AnswerService(search=search, llm=FakeLLM(), conversations=conv)
        resp = await answerer.answer("anything", session_id=conv.new_session())
        assert resp.citations == []
        assert "could" in resp.answer.lower() or "couldn't" in resp.answer.lower()
    finally:
        for c in closeables:
            await c.aclose()


async def test_expiry_listing_with_no_matches_shows_no_sources(tmp_path) -> None:
    """A listing that matches ZERO documents must not fall back to the retrieved context
    passages as sources. "which documents expire in the next 30 days" with a facet filter
    still runs body-text retrieval (which is NOT date-filtered), so passages come back for
    documents that are *not* in the expiry window — showing them as sources contradicts the
    "none match" answer. The empty document list IS the provenance."""
    from datetime import date

    from rag_search.answer import AnswerService, FakeLLM
    from rag_search.conversation import ConversationStore
    from rag_search.metadata_store import MetadataStore
    from rag_search.models import Passage, SearchResponse

    from rag_core.models.enums import Language
    from rag_core.models.metadata import DocumentMetadata

    # Retrieval returns a passage (the facet filter matched a document by content), but its
    # expiry is far outside the 30-day window, so the metadata listing matches nothing.
    class _OneHitSearch:
        async def search(self, q, *, namespace="default", top_k=6, filters=None):
            return SearchResponse(
                query=q, took_ms=1.0,
                results=[Passage(
                    chunk_id="c1::0", document_id="c1", source_key="foxtrot/cert1.pdf",
                    score=1.0, text="Certificate body text.",
                )],
            )

        async def expand_document(self, document_id, seed_text, *, cap):
            return seed_text

    async def _presign(key: str) -> str:
        return f"https://signed/{key}"

    store = MetadataStore(str(tmp_path / "m.db"))
    try:
        await store.put(
            DocumentMetadata(
                document_id="c1", language=Language.EN, expiry_date=date(2030, 1, 1)
            ),
            namespace="default", source_key="foxtrot/cert1.pdf",
            facets={"status": "Active"},
        )
        answerer = AnswerService(
            search=_OneHitSearch(), llm=FakeLLM(),  # type: ignore[arg-type]
            conversations=ConversationStore(ttl_minutes=30), metadata=store, presign=_presign,
        )
        # Explicit status filter forces a metadata listing (not a pure date-only listing),
        # so retrieval runs and returns the out-of-window passage above.
        resp = await answerer.answer(
            "which documents expire in the next 30 days",
            session_id="s1", filters={"status": "Active"},
        )
        assert resp.documents_total == 0            # nothing expires in the window
        assert resp.citations == []                 # ...so no context passages leak as sources
        assert resp.answer                          # the model still answered (from "0 match")
    finally:
        await store.aclose()


async def test_followup_listing_refines_previous_filter(tmp_path) -> None:
    """A follow-up that narrows the previous listing ("out of this, which are expired")
    must stay scoped to that result set — the prior filter is carried forward, so it lists
    the expired ACTIVE documents, not every expired document in the corpus."""
    from datetime import date

    from rag_search.answer import AnswerService, FakeLLM
    from rag_search.conversation import ConversationStore
    from rag_search.metadata_store import MetadataStore
    from rag_search.models import SearchResponse

    from rag_core.models.enums import Language
    from rag_core.models.metadata import DocumentMetadata

    class _EmptySearch:
        async def search(self, q, *, namespace="default", top_k=6, filters=None):
            return SearchResponse(query=q, took_ms=1.0, results=[])

        async def expand_document(self, document_id, seed_text, *, cap):
            return seed_text

    # The router reads the follow-up as a listing that continues the thread, but — as the
    # real model does — it does NOT restate the status=Active value the user only said last
    # turn. The carry-forward, not the router, must supply it.
    class _StubRouter:
        def __init__(self) -> None:
            self.last_input_tokens = 0
            self.last_output_tokens = 0
            self._plans = [
                {"intent": "list", "filters": {"status": "Active"}, "expiry_from": None,
                 "expiry_to": None, "continues_thread": False,
                 "search_query": "documents in Active status"},
                {"intent": "list", "filters": {}, "expiry_from": None,
                 "expiry_to": date.today().isoformat(), "continues_thread": True,
                 "search_query": "expired documents"},
            ]

        async def route(self, question, *, facet_values, history=None):
            return self._plans.pop(0)

    async def _presign(key: str) -> str:
        return f"https://signed/{key}"

    store = MetadataStore(str(tmp_path / "m.db"))
    try:
        # A,B are Active; B and C are expired. So "Active AND expired" is B alone.
        await store.put(
            DocumentMetadata(document_id="A", language=Language.EN, expiry_date=date(2030, 1, 1)),
            namespace="default", source_key="foxtrot/active_valid.pdf", facets={"status": "Active"})
        await store.put(
            DocumentMetadata(document_id="B", language=Language.EN, expiry_date=date(2024, 1, 1)),
            namespace="default", source_key="foxtrot/active_expired.pdf", facets={"status": "Active"})
        await store.put(
            DocumentMetadata(document_id="C", language=Language.EN, expiry_date=date(2024, 1, 1)),
            namespace="default", source_key="foxtrot/inactive_expired.pdf", facets={"status": "Inactive"})

        conv = ConversationStore(ttl_minutes=30)
        answerer = AnswerService(
            search=_EmptySearch(), llm=FakeLLM(),  # type: ignore[arg-type]
            conversations=conv, metadata=store, presign=_presign, router=_StubRouter(),
        )
        sid = conv.new_session()

        first = await answerer.answer("list the documents in Active status", session_id=sid)
        assert first.documents_total == 2  # A and B

        second = await answerer.answer("out of this which documents are expired", session_id=sid)
        # Carried the status=Active filter forward, so only the expired ACTIVE doc (B) — the
        # inactive-but-expired C is NOT in the previous result set and must not appear.
        assert second.documents_total == 1
        assert [d.source_key for d in second.documents] == ["foxtrot/active_expired.pdf"]
    finally:
        await store.aclose()


async def test_halal_listing_does_not_dump_every_certificate(tmp_path) -> None:
    """"list all halal certificates" must NOT collapse to a facet-only list of every
    certificate — "halal" is a content qualifier no facet can express. It is demoted to a
    content question: the matching-documents list is suppressed and the halal documents are
    surfaced as cited sources instead."""
    from rag_search.answer import AnswerService, FakeLLM
    from rag_search.conversation import ConversationStore
    from rag_search.metadata_store import MetadataStore
    from rag_search.models import Passage, SearchResponse

    from rag_core.models.enums import Language
    from rag_core.models.metadata import DocumentMetadata

    # Content/keyword retrieval is what actually understands "halal"; the stub returns the
    # halal certificates for the query (as the real vector+keyword search would).
    class _HalalSearch:
        async def search(self, q, *, namespace="default", top_k=6, filters=None):
            hits = [
                Passage(chunk_id="h1::0", document_id="H1", source_key="foxtrot/halal_cert_a.pdf",
                        score=0.9, text="Halal certificate for meat products.", matched_by="both"),
                Passage(chunk_id="h2::0", document_id="H2", source_key="foxtrot/halal_cert_b.pdf",
                        score=0.8, text="Halal certification, ICQ.", matched_by="both"),
            ]
            return SearchResponse(query=q, took_ms=1.0, results=hits)

        async def expand_document(self, document_id, seed_text, *, cap):
            return seed_text

    class _StubRouter:
        def __init__(self) -> None:
            self.last_input_tokens = 0
            self.last_output_tokens = 0

        async def route(self, question, *, facet_values, history=None):
            # The router (mis)reads it as a certificate listing — "halal" is not a facet, so
            # it drops out. Left unchecked this lists every certificate.
            return {"intent": "list", "filters": {"document_class": "certificate"},
                    "expiry_from": None, "expiry_to": None, "continues_thread": False,
                    "search_query": "halal certificates"}

    async def _presign(key: str) -> str:
        return f"https://signed/{key}"

    store = MetadataStore(str(tmp_path / "m.db"))
    try:
        # Three certificates; only two are halal (by title). A facet-only listing would
        # return all three.
        for did, key, title in [
            ("H1", "foxtrot/halal_cert_a.pdf", "Halal Certificate - Meat"),
            ("H2", "foxtrot/halal_cert_b.pdf", "ICQ Halal Certification"),
            ("P1", "foxtrot/fssc_cert.pdf", "FSSC 22000 Food Safety Certificate"),
        ]:
            await store.put(
                DocumentMetadata(document_id=did, language=Language.EN, title=title),
                namespace="default", source_key=key,
                facets={"document_class": "certificate"},
            )
        answerer = AnswerService(
            search=_HalalSearch(), llm=FakeLLM(),  # type: ignore[arg-type]
            conversations=ConversationStore(ttl_minutes=30), metadata=store,
            presign=_presign, router=_StubRouter(),
        )
        resp = await answerer.answer("list all halal certificates", session_id="s1")
        assert resp.documents_total == 0                       # NOT a 3-certificate dump
        assert resp.citations                                  # halal docs surfaced as sources
        cited = {c.source_key for c in resp.citations}
        assert cited <= {"foxtrot/halal_cert_a.pdf", "foxtrot/halal_cert_b.pdf"}  # only the halal ones
        assert "foxtrot/fssc_cert.pdf" not in cited
    finally:
        await store.aclose()
