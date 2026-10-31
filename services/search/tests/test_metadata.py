"""Tests for the Metadata Extraction Agent (#4): extractor, store, filtered search."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from datetime import date
from typing import Any

from rag_search.metadata_extract import (
    MetadataExtractor,
    canonical_supplier,
    is_certifier,
    is_valid_supplier_name,
    parse_expiry,
    supplier_from_text,
)
from rag_search.metadata_store import MetadataStore
from rag_search.models import IndexRequest
from rag_search.mongo_source import (
    MongoMetadataSource,
    apply_mongo_to_metadata,
    facets_from_mongo,
)

from rag_core.aws import GenerationResult
from rag_core.models.content import BlockType, ExtractedContent, Page, TextBlock
from rag_core.models.enums import DocumentClass, Language
from rag_core.models.metadata import DocumentMetadata


def _content() -> ExtractedContent:
    return ExtractedContent(
        document_id="doc-1",
        pages=[
            Page(
                number=1,
                text="Quality Manual. This manual complies with ISO 9001 version 2.1 revision 3. "
                "Supplier audits and corrective actions are described herein.",
                blocks=[TextBlock(text="Quality Manual", role=BlockType.HEADING, page=1)],
            )
        ],
    )


async def test_heuristic_extraction() -> None:
    meta = await MetadataExtractor(None, use_llm=False).extract(
        _content(), document_id="doc-1", source_key="foxtrot/abc123_Quality Manual~docx",
        document_class=DocumentClass.TECHNICAL_MANUAL, language=Language.EN,
    )
    assert meta.title == "Quality Manual"
    assert meta.regulation == "ISO 9001"
    assert meta.version == "2.1"
    assert meta.revision == "3"
    assert meta.language == Language.EN
    assert meta.keywords  # frequency-based
    assert meta.country is None  # heuristics don't infer this


class _JsonLLM:
    async def complete(self, messages: list[dict[str, Any]], **kw: Any) -> GenerationResult:
        return GenerationResult(
            text='{"title":"Quality Manual","country":"Australia","supplier":"ACME",'
            '"department":"Quality","business_unit":"Operations","category":"Manual",'
            '"effective_date":"2024-01-15","keywords":["audit","supplier"],'
            '"entities":[{"text":"ACME","type":"ORG"}]}',
            input_tokens=100, output_tokens=40, model="fake",
        )


async def test_llm_extraction_fills_semantic_facets() -> None:
    meta = await MetadataExtractor(_JsonLLM(), use_llm=True).extract(  # type: ignore[arg-type]
        _content(), document_id="doc-1", source_key="foxtrot/x~docx",
        document_class=DocumentClass.TECHNICAL_MANUAL, language=Language.EN,
    )
    assert meta.country == "Australia"
    assert meta.supplier == "ACME"
    assert meta.department == "Quality"
    assert meta.effective_date is not None and meta.effective_date.year == 2024
    assert any(e.text == "ACME" for e in meta.entities)
    assert meta.extraction_confidence >= 0.8


async def test_metadata_store_roundtrip_and_facets(tmp_path) -> None:
    store = MetadataStore(str(tmp_path / "meta.db"))
    try:
        await store.put(
            DocumentMetadata(document_id="d1", country="Australia", supplier="ACME",
                             document_class=DocumentClass.POLICY, language=Language.EN),
            namespace="default", source_key="foxtrot/a.docx",
        )
        got = await store.get("d1")
        assert got is not None and got.supplier == "ACME"
        facets = await store.facets("default")
        assert facets["country"] == ["Australia"]
        assert "ACME" in facets["supplier"]
        assert await store.count() == 1
    finally:
        await store.aclose()


async def test_facets_cascade_on_active_filters(tmp_path) -> None:
    """Selecting one facet narrows the others to values that co-occur with it."""
    store = MetadataStore(str(tmp_path / "meta.db"))
    try:
        # Two Draft docs (codes C1, C2) and one Active doc (code C3).
        await store.put(
            DocumentMetadata(document_id="d1"), namespace="default", source_key="foxtrot/1.pdf",
            facets={"status": "Draft", "object_code": "C1", "business_object": "BO_A"},
        )
        await store.put(
            DocumentMetadata(document_id="d2"), namespace="default", source_key="foxtrot/2.pdf",
            facets={"status": "Draft", "object_code": "C2", "business_object": "BO_A"},
        )
        await store.put(
            DocumentMetadata(document_id="d3"), namespace="default", source_key="foxtrot/3.pdf",
            facets={"status": "Active", "object_code": "C3", "business_object": "BO_B"},
        )

        # Unfiltered: every value is offered.
        allf = await store.facets("default")
        assert set(allf["object_code"]) == {"C1", "C2", "C3"}
        assert set(allf["business_object"]) == {"BO_A", "BO_B"}

        # status=Draft -> only Draft codes/objects; status itself stays fully switchable.
        draft = await store.facets("default", filters={"status": "Draft"})
        assert set(draft["object_code"]) == {"C1", "C2"}
        assert set(draft["business_object"]) == {"BO_A"}
        assert set(draft["status"]) == {"Draft", "Active"}   # not narrowed by itself

        # Two filters compound.
        both = await store.facets("default", filters={"status": "Draft", "business_object": "BO_A"})
        assert set(both["object_code"]) == {"C1", "C2"}

        # A selection with no co-occurring values yields an empty (not stale) list.
        active = await store.facets("default", filters={"status": "Active"})
        assert set(active["object_code"]) == {"C3"}
    finally:
        await store.aclose()


async def test_match_facet_values_resolves_rare_codes_past_the_dropdown_cap(tmp_path) -> None:
    """A rare code beyond the 200-value facets() cap must still resolve for filtering.

    facets() returns the 200 most common values per key for dropdowns; a one-off object
    code isn't in that list. match_facet_values queries the full vocabulary, so pasting
    the code verbatim still routes to a filter.
    """
    store = MetadataStore(str(tmp_path / "meta.db"))
    try:
        # 250 common codes to push a rare one past the 200 cap, each on many docs.
        for i in range(250):
            for j in range(3):  # 3 docs each -> high count, fills the dropdown
                await store.put(
                    DocumentMetadata(document_id=f"common{i}-{j}"),
                    namespace="default", source_key=f"foxtrot/c{i}_{j}.pdf",
                    facets={"object_code": f"COMMON{i:04d}"},
                )
        # One rare code on a single document.
        await store.put(
            DocumentMetadata(document_id="rare"),
            namespace="default", source_key="foxtrot/rare.pdf",
            facets={"object_code": "STD3936", "created_by": "af3133@acmefoods.example.com"},
        )

        # It is NOT in the capped dropdown list...
        assert "STD3936" not in (await store.facets("default")).get("object_code", [])

        # ...but resolves exactly against the full vocabulary, case-insensitively.
        assert await store.match_facet_values(["std3936"]) == {"object_code": "STD3936"}
        got = await store.match_facet_values(["AF3133@acmefoods.example.com"])
        assert got == {"created_by": "af3133@acmefoods.example.com"}

        # A token that is not any facet value resolves to nothing (no bogus filter).
        assert await store.match_facet_values(["not-a-real-code"]) == {}
    finally:
        await store.aclose()


def test_parse_expiry_various_formats() -> None:
    assert parse_expiry("Northwind Classic Kosher exp 01 08 2025").isoformat() == "2025-08-01"
    assert parse_expiry("Halal Certificate valid until 31/12/2024").isoformat() == "2024-12-31"
    assert parse_expiry("Certificate expires 31 Dec 2024").isoformat() == "2024-12-31"
    assert parse_expiry("expiry 2026-03-15").isoformat() == "2026-03-15"
    assert parse_expiry("no relevant date in this text") is None


async def test_list_documents_expiry_range(tmp_path) -> None:
    store = MetadataStore(str(tmp_path / "m.db"))
    try:
        await store.put(
            DocumentMetadata(document_id="a", expiry_date=date(2025, 8, 15), language=Language.EN),
            namespace="default", source_key="foxtrot/a.pdf",
        )
        await store.put(
            DocumentMetadata(document_id="b", expiry_date=date(2024, 12, 31), language=Language.EN),
            namespace="default", source_key="foxtrot/b.pdf",
        )
        total, docs = await store.list_documents(
            "default", expiry_from="2025-08-01", expiry_to="2025-08-31"
        )
        assert total == 1
        assert docs[0]["document_id"] == "a"
        assert docs[0]["expiry_date"] == "2025-08-15"
    finally:
        await store.aclose()


async def test_list_documents_pagination_and_filter(tmp_path) -> None:
    store = MetadataStore(str(tmp_path / "meta.db"))
    try:
        for i in range(5):
            status = "Draft" if i < 3 else "Active"
            await store.put(
                DocumentMetadata(document_id=f"d{i}", language=Language.EN),
                namespace="default",
                source_key=f"foxtrot/doc{i}.pdf",
                facets={"status": status, "business_object": "NFIF"},
            )
        # Filter by status=Draft -> 3 total; page size 2 -> 2 then 1.
        total, page1 = await store.list_documents("default", filters={"status": "Draft"}, limit=2)
        assert total == 3 and len(page1) == 2
        assert all(d["facets"]["status"] == "Draft" for d in page1)
        total2, page2 = await store.list_documents(
            "default", filters={"status": "Draft"}, limit=2, offset=2
        )
        assert total2 == 3 and len(page2) == 1
        # No filter -> all 5.
        total_all, _ = await store.list_documents("default", limit=100)
        assert total_all == 5
    finally:
        await store.aclose()


Stack = Callable[..., tuple[Any, Any, MetadataStore, list]]


async def test_filtered_search_by_facet(build_stack: Stack, sample_objects: dict[str, bytes]) -> None:
    indexer, search, meta_store, closeables = build_stack(sample_objects)
    try:
        await indexer.index(IndexRequest(reset=True))

        # Facets persisted + queryable.
        facets = await meta_store.facets("default")
        assert "en" in facets.get("language", [])

        # Filter matches -> results; non-matching filter -> empty.
        hit = await search.search("encryption", filters={"language": "en"})
        assert hit.results
        miss = await search.search("encryption", filters={"language": "de"})
        assert miss.results == []
    finally:
        for c in closeables:
            await c.aclose()


# --- MongoDB metadata source (system of record) ------------------------------

def _mongo_doc() -> dict[str, Any]:
    return {
        "OriginalFileName": "security_policy.docx",
        "BusinessObject": "SUPPLIERAUDIT",
        "ObjectCode": "DISTAUDIT0135",
        "Status": "Draft",
        "DocumentCategory": "Audit",
        "DocumentSubCategory": "Supplier",
        "TenantCode": "FOXTROT",
        "CreatedBy": "audit",
        "Tags": ["urgent", "q1"],
    }


def test_facets_from_mongo_maps_business_fields() -> None:
    facets = facets_from_mongo(_mongo_doc())
    assert facets["business_object"] == "SUPPLIERAUDIT"
    assert facets["object_code"] == "DISTAUDIT0135"
    assert facets["status"] == "Draft"
    assert facets["category"] == "Audit"
    assert facets["tags"] == ["urgent", "q1"]


def test_apply_mongo_to_metadata_is_authoritative() -> None:
    meta = DocumentMetadata(document_id="d1", category="Guessed", language=Language.EN)
    apply_mongo_to_metadata(_mongo_doc(), meta)
    assert meta.category == "Audit"           # Mongo overrides the LLM guess
    assert meta.sub_category == "Supplier"
    assert meta.author == "audit"
    assert meta.business_unit == "FOXTROT"
    assert meta.extra["business_object"] == "SUPPLIERAUDIT"


async def test_indexing_uses_mongo_facets(
    build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    # OriginalFileName joins to the S3 basename ("security_policy.docx").
    mongo = MongoMetadataSource("mongodb://x", database="d", collection="c")
    mongo.seed({"security_policy.docx": _mongo_doc()})

    indexer, search, meta_store, closeables = build_stack(sample_objects, mongo=mongo)
    try:
        await indexer.index(IndexRequest(reset=True))

        facets = await meta_store.facets("default")
        assert "SUPPLIERAUDIT" in facets.get("business_object", [])
        assert "Draft" in facets.get("status", [])

        # The Mongo facet is denormalised onto chunks -> filterable retrieval.
        hit = await search.search("encryption", filters={"business_object": "SUPPLIERAUDIT"})
        assert hit.results
        miss = await search.search("encryption", filters={"business_object": "NONEXISTENT"})
        assert miss.results == []
    finally:
        for c in closeables:
            await c.aclose()


async def test_deletemark_excludes_document_from_index(
    build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    """A Mongo record with DeleteMark=1 must never be indexed or returned in search."""
    doc = {**_mongo_doc(), "DeleteMark": 1}
    mongo = MongoMetadataSource("mongodb://x", database="d", collection="c")
    mongo.seed({"security_policy.docx": doc})
    assert mongo.is_deleted("foxtrot/security_policy.docx") is True
    assert mongo.lookup("foxtrot/security_policy.docx") is None  # kept out of the live cache

    indexer, search, meta_store, closeables = build_stack(sample_objects, mongo=mongo)
    try:
        summary = await indexer.index(IndexRequest(reset=True))
        assert summary.documents_deleted == 1
        # The deleted doc never enters any index; the other sample object indexes fine.
        keys = {p.source_key for p in (await search.search("encryption")).results}
        assert "foxtrot/security_policy.docx" not in keys
        assert any(
            p.source_key == "foxtrot/supplier_register.html"
            for p in (await search.search("ACME supplier")).results
        )
        # Its authoritative Mongo facet never reached the metadata store either.
        facets = await meta_store.facets("default")
        assert "SUPPLIERAUDIT" not in facets.get("business_object", [])
    finally:
        for c in closeables:
            await c.aclose()


async def test_deletemark_purges_previously_indexed_document(
    build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    """A doc indexed while live is purged on the next re-index once DeleteMark is set."""
    live = MongoMetadataSource("mongodb://x", database="d", collection="c")
    live.seed({"security_policy.docx": _mongo_doc()})
    indexer, search, meta_store, closeables = build_stack(sample_objects, mongo=live)
    try:
        await indexer.index(IndexRequest(reset=True))
        indexed = {p.source_key for p in (await search.search("encryption")).results}
        assert "foxtrot/security_policy.docx" in indexed  # indexed while live

        # Now the record is soft-deleted upstream; an incremental re-index purges it.
        indexer._mongo.seed({"security_policy.docx": {**_mongo_doc(), "DeleteMark": 1}})
        summary = await indexer.index(IndexRequest(reset=False))
        assert summary.documents_deleted == 1
        after = {p.source_key for p in (await search.search("encryption")).results}
        assert "foxtrot/security_policy.docx" not in after
        facets = await meta_store.facets("default")
        assert "SUPPLIERAUDIT" not in facets.get("business_object", [])
    finally:
        for c in closeables:
            await c.aclose()


# --- Supplier NAME: read from the document, since Mongo holds only the code ---------

_CERT_TEXT = (
    "bsi. Certificate of Registration BSI Assurance UK Limited certifies that: "
    "CONTOSO AUSTRALIA PTY LTD Site Code: 1234567 65 Exemplar Road, Sampletown, "
    "QLD, 4000, Australia. CONTOSO AUSTRALIA PTY LTD has achieved Grade A."
)


def test_certification_bodies_are_never_the_supplier() -> None:
    """A certificate prints its auditor larger than the company it certifies."""
    assert is_certifier("BSI Assurance UK Limited") is True
    assert is_certifier("SGS Australia Pty Ltd") is True
    assert is_certifier("Contoso Australia Pty Ltd") is False


def test_supplier_from_text_prefers_the_certified_company() -> None:
    assert supplier_from_text(_CERT_TEXT) == "CONTOSO AUSTRALIA PTY LTD"


def test_supplier_from_text_reads_a_filled_in_label() -> None:
    assert supplier_from_text("SUPPLIER: Fabrikam Australia Pty Ltd\nDATE: 16/06/2025") == (
        "Fabrikam Australia Pty Ltd"
    )


def test_supplier_from_text_ignores_an_empty_label_and_prose() -> None:
    """The real AMS form prints "SUPPLIER:" with nothing after it."""
    assert supplier_from_text("SUPPLIER: DATE: Note: valid for 1 year after approval.") is None
    # Document furniture that merely ends in a company-ish word is not a name.
    assert supplier_from_text("No Marking Required 3L Dairy. Report reference 12/4.") is None


def test_canonical_supplier_votes_across_a_code_s_documents() -> None:
    """One supplier's documents spell it several ways; the corpus's own spelling wins."""
    votes = Counter({
        "NWD Milk Pty Ltd": 4,
        "nwd milk pty ltd": 1,      # same name, different case — one group
        "Northwind F&B - Southgate": 2,
        "John Smith": 1,       # a signatory picked up from one certificate
    })
    assert canonical_supplier(votes) == "NWD Milk Pty Ltd"
    assert canonical_supplier(Counter()) is None


async def test_extract_supplier_falls_back_to_heuristics_without_an_llm() -> None:
    content = ExtractedContent(
        document_id="d1", pages=[Page(number=1, text=_CERT_TEXT, blocks=[])]
    )
    extractor = MetadataExtractor(None, use_llm=False)
    assert await extractor.extract_supplier(content) == "CONTOSO AUSTRALIA PTY LTD"


async def test_indexing_extracts_and_consolidates_supplier_names(
    build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    """End to end: the name comes from the document, not from Mongo, and is filterable."""
    objects = {
        **sample_objects,
        "foxtrot/cert_one.txt": _CERT_TEXT.encode(),
        # Same supplier code, a spelling the vote must discard in favour of the majority.
        "foxtrot/cert_two.txt": _CERT_TEXT.encode(),
    }
    mongo = MongoMetadataSource("mongodb://x", database="d", collection="c")
    mongo.seed({
        "cert_one.txt": {"OriginalFileName": "cert_one.txt", "ObjectCode": "SUPPLIER0028",
                         "BusinessObject": "SUPPLIERDOCUMENTS", "ObjectName": "SUPPLIERDOCUMENTS"},
        "cert_two.txt": {"OriginalFileName": "cert_two.txt", "ObjectCode": "SUPPLIER0028",
                         "BusinessObject": "SUPPLIERDOCUMENTS", "ObjectName": "SUPPLIERDOCUMENTS"},
    })

    indexer, search, meta_store, closeables = build_stack(objects, mongo=mongo)
    try:
        await indexer.index(IndexRequest(reset=True))

        # Mongo still owns the code; the NAME came out of the document text.
        _, docs = await meta_store.list_documents(
            "default", filters={"supplier": "CONTOSO AUSTRALIA PTY LTD"}
        )
        assert len(docs) == 2
        assert {d["facets"]["object_code"] for d in docs} == {"SUPPLIER0028"}

        hit = await search.search("Exemplar", filters={"supplier": "CONTOSO AUSTRALIA PTY LTD"})
        assert hit.results
    finally:
        for c in closeables:
            await c.aclose()


def test_supplier_from_text_rejects_prose_ending_in_a_legal_suffix() -> None:
    """A certificate's signature block reads "…Rabbinic Administrator Northwind F&B Pty Ltd"."""
    text = "Kosher certified. Rabbinic Administrator Northwind F&B Pty Ltd. Valid to 2025."
    assert supplier_from_text(text) is None


def test_conversational_non_answers_are_not_supplier_names() -> None:
    """A model asked about an empty document replies in prose, not with NONE."""
    assert is_valid_supplier_name("Contoso Australia Pty Ltd") is True
    assert is_valid_supplier_name("UMBRELLA GROUP") is True
    assert is_valid_supplier_name(
        "I don't see any document attached to analyze. Could you please provide it?"
    ) is False
    assert is_valid_supplier_name("BSI Assurance UK Limited") is False
    assert is_valid_supplier_name("no") is False


async def test_apply_facets_retracts_a_value(tmp_path: Any) -> None:
    """A None value removes the facet — how a bad extraction gets withdrawn."""
    store = MetadataStore(str(tmp_path / "m.db"))
    try:
        meta = DocumentMetadata(document_id="d1", language=Language.EN)
        await store.put(meta, namespace="default", source_key="foxtrot/a.pdf",
                        facets={"object_code": "PIF0171", "supplier": "I don't see a document"})
        assert await store.apply_facets("d1", {"supplier": None}) is True

        _, docs = await store.list_documents("default")
        assert docs[0]["facets"] == {"object_code": "PIF0171"}
        assert "supplier" not in (await store.facets("default"))
        stored = await store.get("d1")
        assert stored is not None and stored.supplier is None
    finally:
        await store.aclose()


def test_canonical_supplier_prefers_the_full_legal_name() -> None:
    """A bare "Contoso" on more documents must not beat the full name it prefixes."""
    votes = Counter({"Contoso": 5, "CONTOSO AUSTRALIA PTY LTD": 3})
    assert canonical_supplier(votes) == "CONTOSO AUSTRALIA PTY LTD"
    # Distinct suppliers that merely share a word are NOT merged.
    assert canonical_supplier(Counter({"Northwind F&B - Southgate": 3, "Initech Foods": 1})) == (
        "Northwind F&B - Southgate"
    )


def test_canonical_supplier_ignores_a_one_off_trailing_address() -> None:
    """"NWD Milk Pty Ltd (A.B.N. …) Addr" appears once; the clean name appears often."""
    votes = Counter({"NWD Milk Pty Ltd": 4, "NWD Milk Pty Ltd (A.B.N. 12 345 678 901) Addr": 1})
    assert canonical_supplier(votes) == "NWD Milk Pty Ltd"
