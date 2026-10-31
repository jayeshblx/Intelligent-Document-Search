"""Document links on citations, and the optional BusinessObject scope.

The two things ExampleApp renders from a response beyond the answer text: a working
link to every document the answer used, and results confined to the BusinessObject
it asked for -- or the whole tenant when it asked for none.
"""

from __future__ import annotations

from typing import Any

import pytest
from rag_agentic.agents.orchestrator import OrchestratorAgent
from rag_agentic.agents.search_agent import DocumentSearchAgent
from rag_agentic.config import AgenticSettings, DocumentUrlMode
from rag_agentic.converse import ScriptedConverseModel, text_turn, tool_turn
from rag_agentic.document_urls import DocumentUrlBuilder, validate_template
from rag_agentic.index_store import TenantIndexStore
from rag_agentic.models import AgentRequest, AgentResponse, Citation
from rag_agentic.search_client import InProcessSearchClient
from rag_agentic.tenancy import TenantContext, TenantResolver

CONTEXT = TenantContext(environment="dev", tenant="acme", base_url="https://dev.example.io/acme")
MANIFEST = {"source_bucket": "tenant-source-bucket", "source_prefix": "foxtrot/"}


class FakePresigner:
    """Signs deterministically so tests can assert exactly what was signed."""

    def __init__(self, *, fail_on: str = "") -> None:
        self.calls: list[tuple[str, str, int]] = []
        self._fail_on = fail_on

    async def presigned_get(self, bucket: str, key: str, *, expires_s: int = 3600) -> str:
        self.calls.append((bucket, key, expires_s))
        if self._fail_on and key == self._fail_on:
            raise RuntimeError("signing failed")
        return f"https://s3.example/{bucket}/{key}?X-Amz-Expires={expires_s}"


def _settings(base: AgenticSettings, **overrides: Any) -> AgenticSettings:
    return base.model_copy(update=overrides)


# --------------------------------------------------------------------------
# DocumentUrlBuilder
# --------------------------------------------------------------------------


async def test_presigned_url_uses_the_tenants_own_source_bucket(
    agentic_settings: AgenticSettings,
) -> None:
    signer = FakePresigner()
    settings = _settings(agentic_settings, source_bucket="fallback-bucket", document_url_ttl_s=900)
    url_for = DocumentUrlBuilder(settings, signer).for_tenant(CONTEXT, MANIFEST, CONTEXT.base_url)

    assert url_for is not None
    url = await url_for("d1", "foxtrot/abc_manual~pdf")
    # The manifest's bucket wins over the service-wide fallback.
    assert signer.calls == [("tenant-source-bucket", "foxtrot/abc_manual~pdf", 900)]
    assert url == "https://s3.example/tenant-source-bucket/foxtrot/abc_manual~pdf?X-Amz-Expires=900"


async def test_presigned_falls_back_to_setting_for_old_manifests(
    agentic_settings: AgenticSettings,
) -> None:
    signer = FakePresigner()
    settings = _settings(agentic_settings, source_bucket="fallback-bucket")
    url_for = DocumentUrlBuilder(settings, signer).for_tenant(CONTEXT, {}, CONTEXT.base_url)

    assert url_for is not None
    await url_for("d1", "foxtrot/x.pdf")
    assert signer.calls[0][0] == "fallback-bucket"


async def test_no_source_bucket_means_no_links_rather_than_a_guess(
    agentic_settings: AgenticSettings,
) -> None:
    builder = DocumentUrlBuilder(_settings(agentic_settings, source_bucket=""), FakePresigner())
    assert builder.for_tenant(CONTEXT, {}, CONTEXT.base_url) is None


async def test_keys_outside_the_tenants_prefix_are_never_signed(
    agentic_settings: AgenticSettings,
) -> None:
    signer = FakePresigner()
    url_for = DocumentUrlBuilder(agentic_settings, signer).for_tenant(
        CONTEXT, MANIFEST, CONTEXT.base_url
    )
    assert url_for is not None
    assert await url_for("d9", "other-tenant/secret.pdf") is None
    assert signer.calls == []


async def test_a_failed_signature_drops_one_link_not_the_answer(
    agentic_settings: AgenticSettings,
) -> None:
    url_for = DocumentUrlBuilder(agentic_settings, FakePresigner(fail_on="foxtrot/bad.pdf")).for_tenant(
        CONTEXT, MANIFEST, CONTEXT.base_url
    )
    assert url_for is not None
    assert await url_for("d1", "foxtrot/bad.pdf") is None
    assert await url_for("d2", "foxtrot/good.pdf")


async def test_template_mode_builds_exampleapp_links_with_safe_quoting(
    agentic_settings: AgenticSettings,
) -> None:
    settings = _settings(
        agentic_settings,
        document_url_mode=DocumentUrlMode.TEMPLATE,
        document_url_template="{base_url}/{environment}/{tenant}/documents/{document_id}?name={filename}",
    )
    url_for = DocumentUrlBuilder(settings).for_tenant(
        CONTEXT, MANIFEST, "https://dev.example.io/acme/"
    )
    assert url_for is not None
    url = await url_for("doc 1/2", "foxtrot/0a1b2c3d4e5f_Supplier Manual~pdf")
    # Trailing slash on base_url does not double up; the id cannot inject a path.
    assert url == (
        "https://dev.example.io/acme/dev/acme/documents/doc%201%2F2?name=Supplier%20Manual.pdf"
    )


def test_bad_templates_fail_at_startup() -> None:
    with pytest.raises(ValueError, match="needs AGENTIC_DOCUMENT_URL_TEMPLATE"):
        validate_template("")
    with pytest.raises(ValueError, match="unknown placeholder"):
        validate_template("{base_url}/{bucket}/{document_id}")
    with pytest.raises(ValueError, match="must identify the document"):
        validate_template("{base_url}/documents")


async def test_none_mode_produces_no_links(agentic_settings: AgenticSettings) -> None:
    builder = DocumentUrlBuilder(_settings(agentic_settings, document_url_mode=DocumentUrlMode.NONE))
    assert builder.for_tenant(CONTEXT, MANIFEST, CONTEXT.base_url) is None


# --------------------------------------------------------------------------
# Toolbox citations
# --------------------------------------------------------------------------


async def test_search_citations_carry_links_to_their_own_documents(make_tools: Any) -> None:
    tools = make_tools()
    signer = FakePresigner()
    tools._url_for = DocumentUrlBuilder(
        AgenticSettings(source_bucket="src"), signer
    ).for_tenant(CONTEXT, {}, CONTEXT.base_url)

    await tools.invoke("search_documents", {"query": "ISO 9001 certificate"})
    citations = await tools.citations_with_urls()

    by_id = {c.document_id: c for c in citations}
    assert by_id["d1"].download_url == "https://s3.example/src/foxtrot/supplier_quality_manual.pdf?X-Amz-Expires=3600"
    assert all(c.download_url for c in citations)
    # Only returned citations are signed -- one signature per document.
    assert len(signer.calls) == len(citations)


async def test_listing_questions_cite_every_document_they_name(make_tools: Any) -> None:
    tools = make_tools()
    tools._url_for = DocumentUrlBuilder(
        AgenticSettings(source_bucket="src"), FakePresigner()
    ).for_tenant(CONTEXT, {}, CONTEXT.base_url)

    await tools.invoke("list_documents", {})
    citations = await tools.citations_with_urls()

    assert {c.document_id for c in citations} == {"d1", "d2", "d3"}
    assert all(c.download_url for c in citations)


async def test_search_hits_rank_ahead_of_listed_documents(make_tools: Any) -> None:
    tools = make_tools()
    await tools.invoke("list_documents", {})
    await tools.invoke("search_documents", {"query": "complaint acknowledged within 48 hours"})

    citations = tools.citations()
    assert citations[0].score > 0  # a search hit, not a score-less listing
    assert len({c.document_id for c in citations}) == len(citations)


async def test_the_model_never_sees_a_url(make_tools: Any) -> None:
    tools = make_tools()
    tools._url_for = DocumentUrlBuilder(
        AgenticSettings(source_bucket="src"), FakePresigner()
    ).for_tenant(CONTEXT, {}, CONTEXT.base_url)

    search, _ = await tools.invoke("search_documents", {"query": "ISO 9001"})
    listing, _ = await tools.invoke("list_documents", {})

    assert "http" not in str(search) and "http" not in str(listing)


# --------------------------------------------------------------------------
# Search agent end to end: links + optional BusinessObject
# --------------------------------------------------------------------------


def _agent(
    settings: AgenticSettings,
    model: ScriptedConverseModel,
    embedder: Any,
    index_store: TenantIndexStore,
    resolver: TenantResolver,
    signer: FakePresigner | None = None,
) -> DocumentSearchAgent:
    settings = _settings(settings, source_bucket="exampleapp-source")
    return DocumentSearchAgent(
        settings=settings,
        model=model,
        embedder=embedder,
        index_store=index_store,
        resolver=resolver,
        document_urls=DocumentUrlBuilder(settings, signer or FakePresigner()),
    )


async def test_response_citations_have_download_urls(
    agentic_settings: AgenticSettings,
    embedder: Any,
    index_store: TenantIndexStore,
    resolver: TenantResolver,
    base_url: str,
) -> None:
    model = ScriptedConverseModel(
        [tool_turn("search_documents", {"query": "ISO 9001"}), text_turn("Yes, ISO 9001.")]
    )
    agent = _agent(agentic_settings, model, embedder, index_store, resolver)

    response = await agent.handle(AgentRequest(base_url=base_url, query="Need ISO 9001?"))

    assert response.citations
    assert all(
        c.download_url and c.download_url.startswith("https://s3.example/exampleapp-source/foxtrot/")
        for c in response.citations
    )


async def test_without_business_object_every_document_is_searchable(
    agentic_settings: AgenticSettings,
    embedder: Any,
    index_store: TenantIndexStore,
    resolver: TenantResolver,
    base_url: str,
) -> None:
    model = ScriptedConverseModel([tool_turn("list_documents", {}), text_turn("Three.")])
    agent = _agent(agentic_settings, model, embedder, index_store, resolver)

    response = await agent.handle(AgentRequest(base_url=base_url, query="List all documents"))

    assert response.business_object is None
    assert {c.document_id for c in response.citations} == {"d1", "d2", "d3"}


@pytest.mark.parametrize("supplied", ["COMPLAINTS", "complaints", "  Complaints "])
async def test_business_object_scopes_the_search_whatever_its_case(
    supplied: str,
    agentic_settings: AgenticSettings,
    embedder: Any,
    index_store: TenantIndexStore,
    resolver: TenantResolver,
    base_url: str,
) -> None:
    model = ScriptedConverseModel(
        [tool_turn("search_documents", {"query": "supplier ISO 9001 complaint"}), text_turn("ok")]
    )
    agent = _agent(agentic_settings, model, embedder, index_store, resolver)

    response = await agent.handle(
        AgentRequest(base_url=base_url, query="anything", business_object=supplied)
    )

    # Canonical spelling applied and echoed back; nothing outside COMPLAINTS cited.
    assert response.business_object == "COMPLAINTS"
    # Exactly d2: an empty result would also be "inside" the scope, and is the
    # symptom of a case mismatch silently matching nothing.
    assert {c.document_id for c in response.citations} == {"d2"}
    assert "COMPLAINTS" in model.calls[0]["system"]


async def test_unknown_business_object_is_enforced_not_ignored(
    agentic_settings: AgenticSettings,
    embedder: Any,
    index_store: TenantIndexStore,
    resolver: TenantResolver,
    base_url: str,
) -> None:
    """A value the tenant lacks must not silently fall back to searching everything."""
    model = ScriptedConverseModel([tool_turn("list_documents", {}), text_turn("none")])
    agent = _agent(agentic_settings, model, embedder, index_store, resolver)

    response = await agent.handle(
        AgentRequest(base_url=base_url, query="list", business_object="NOSUCHOBJECT")
    )

    assert response.business_object == "NOSUCHOBJECT"
    assert response.citations == []


# --------------------------------------------------------------------------
# Through the orchestrator
# --------------------------------------------------------------------------


async def test_orchestrator_returns_links_and_passes_business_object_through(
    agentic_settings: AgenticSettings,
    embedder: Any,
    index_store: TenantIndexStore,
    resolver: TenantResolver,
    base_url: str,
) -> None:
    search_model = ScriptedConverseModel(
        [tool_turn("search_documents", {"query": "supplier certificate"}), text_turn("ISO 9001.")]
    )
    search_agent = _agent(agentic_settings, search_model, embedder, index_store, resolver)
    router = ScriptedConverseModel([tool_turn("search_documents", {"query": "supplier certificate"})])
    orchestrator = OrchestratorAgent(
        settings=agentic_settings,
        model=router,
        search_client=InProcessSearchClient(search_agent),
        resolve_tenant=resolver.resolve,
    )

    response = await orchestrator.handle(
        AgentRequest(
            base_url=base_url, query="What certificate do suppliers need?",
            business_object="supplierdocuments",
        )
    )

    assert response.route == "document_search"
    assert (response.environment, response.tenant) == ("dev", "acme")
    # Echoes the filter actually applied, not the caller's lower-case spelling.
    assert response.business_object == "SUPPLIERDOCUMENTS"
    assert response.citations and all(c.download_url for c in response.citations)
    assert {c.document_id for c in response.citations} <= {"d1", "d3"}  # SUPPLIERDOCUMENTS only
    # The router's single model call never needed, and never got, a URL.
    assert len(router.calls) == 1


async def test_router_is_not_shown_download_urls() -> None:
    from rag_agentic.agents.orchestrator import _DelegationToolbox

    class Client:
        async def invoke(self, request: AgentRequest) -> AgentResponse:
            return AgentResponse(
                answer="a",
                citations=[Citation(document_id="d1", download_url="https://signed/secret")],
            )

        async def aclose(self) -> None:
            return None

    toolbox = _DelegationToolbox(Client(), AgentRequest(base_url="https://x/y", query="q"))
    payload, _ = await toolbox.invoke("search_documents", {"query": "q"})

    assert "download_url" not in payload["citations"][0]
    assert toolbox.citations[0].download_url == "https://signed/secret"  # still returned


# --------------------------------------------------------------------------
# Links are signed for the source bucket's own region
# --------------------------------------------------------------------------


class _RegionRecorder:
    """Stands in for RegionalPresigners: one fake signer per region, recorded."""

    def __init__(self) -> None:
        self.signers: dict[str, FakePresigner] = {}

    def __call__(self, region: str) -> FakePresigner:
        return self.signers.setdefault(region, FakePresigner())


async def test_links_are_signed_for_the_manifests_source_region(
    agentic_settings: AgenticSettings,
) -> None:
    """Signed for us-east-1, a link to ap-south-1's example-docs-bucket is an HTTP 400."""
    default, regional = FakePresigner(), _RegionRecorder()
    builder = DocumentUrlBuilder(agentic_settings, default, presigner_for_region=regional)
    manifest = {**MANIFEST, "source_region": "ap-south-1"}

    url_for = builder.for_tenant(CONTEXT, manifest, CONTEXT.base_url)
    assert url_for is not None
    await url_for("d1", "foxtrot/policy~pdf")

    assert default.calls == []
    assert list(regional.signers) == ["ap-south-1"]
    assert regional.signers["ap-south-1"].calls[0][:2] == ("tenant-source-bucket", "foxtrot/policy~pdf")


async def test_region_setting_covers_older_manifests(agentic_settings: AgenticSettings) -> None:
    regional = _RegionRecorder()
    settings = _settings(agentic_settings, source_region="ap-south-1")
    url_for = DocumentUrlBuilder(
        settings, FakePresigner(), presigner_for_region=regional
    ).for_tenant(CONTEXT, MANIFEST, CONTEXT.base_url)
    assert url_for is not None
    await url_for("d1", "foxtrot/policy~pdf")
    assert list(regional.signers) == ["ap-south-1"]


async def test_unknown_region_falls_back_to_the_default_signer(
    agentic_settings: AgenticSettings,
) -> None:
    default, regional = FakePresigner(), _RegionRecorder()
    url_for = DocumentUrlBuilder(
        agentic_settings, default, presigner_for_region=regional
    ).for_tenant(CONTEXT, MANIFEST, CONTEXT.base_url)
    assert url_for is not None
    await url_for("d1", "foxtrot/policy~pdf")
    assert len(default.calls) == 1 and regional.signers == {}
