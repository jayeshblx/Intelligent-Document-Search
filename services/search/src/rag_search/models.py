"""API + internal DTOs for the search service."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from rag_core.models.base import DomainModel


class IndexRequest(DomainModel):
    """Index documents under an S3 prefix into the local hybrid index."""

    prefix: str | None = Field(default=None, description="S3 prefix; defaults to configured.")
    namespace: str = "default"
    reset: bool = Field(default=False, description="Drop existing index before indexing.")
    max_docs: int | None = Field(default=None, description="Cap documents indexed (testing).")
    only_types: list[str] | None = Field(
        default=None,
        description="Restrict to these DocumentType values (e.g. ['docx','txt']).",
    )
    mongo_only: bool = Field(
        default=False,
        description="Only index S3 objects catalogued in the MongoDB metadata source.",
    )


class IndexSummary(DomainModel):
    bucket: str
    prefix: str
    namespace: str
    documents_indexed: int = 0
    documents_failed: int = 0
    documents_deleted: int = 0  # soft-deleted upstream (Mongo DeleteMark) -> skipped/purged
    documents_other_tenant: int = 0  # catalogued only by another TenantCode -> skipped
    documents_non_text: int = 0  # images and other bodies with no extractable text
    documents_skipped_existing: int = 0  # already in the index (resumed run)
    chunks_indexed: int = 0
    duration_s: float = 0.0
    error_samples: list[str] = Field(default_factory=list)


class Passage(DomainModel):
    """A single retrieved passage shown in results."""

    chunk_id: str
    document_id: str
    source_key: str          # S3 key of the source document (which document)
    score: float
    text: str
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    doc_class: str | None = None
    language: str | None = None
    # Provenance of the match, for UI badges: "vector", "keyword", or "both".
    matched_by: str = "vector"
    # Credential-free, app-relative URL (ending in the original filename) that streams the
    # exact source document through the app — never a presigned S3 URL. See downloads.py.
    download_url: str | None = None


class SearchRequest(DomainModel):
    query: str
    namespace: str = "default"
    top_k: int | None = None


class SearchResponse(DomainModel):
    query: str
    took_ms: float
    results: list[Passage] = Field(default_factory=list)


# ---- Document browse (list + paginate by facet) ----------------------------
class DocumentListItem(DomainModel):
    document_id: str
    source_key: str
    title: str | None = None
    expiry_date: str | None = None  # ISO date, if known
    download_url: str | None = None  # app-streamed link to the original file (see downloads.py)
    facets: dict[str, str] = Field(default_factory=dict)


class DocumentListResponse(DomainModel):
    total: int          # total documents matching the filter (drives pagination)
    limit: int
    offset: int
    documents: list[DocumentListItem] = Field(default_factory=list)


# ---- Conversational RAG (chat) ---------------------------------------------
class Citation(DomainModel):
    """A source the answer was grounded on (rendered under the answer)."""

    index: int  # matches the [n] used inline in the answer
    document_id: str
    source_key: str
    snippet: str
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    download_url: str | None = None
    matched_by: str = "vector"
    score: float = 0.0
    # Metadata facets (status, business_object, object_code, document_class, …) so the
    # source card can show the same chips as the Matching-documents list.
    facets: dict[str, str] = Field(default_factory=dict)


class UsageInfo(DomainModel):
    """Token usage + cost for one answered question."""

    input_tokens: int = 0
    output_tokens: int = 0
    embed_tokens: int = 0
    # Prompt-cache tokens (0 unless caching is active): read = discounted reuse of the
    # cached prefix, write = the first call that populated it.
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    embed_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    model: str = ""


# ---- Credit metering (usage-based billing) ---------------------------------
class CreditInfo(DomainModel):
    """Per-request credit accounting + the resulting balance (server-authoritative).

    ``base_cost``/``charged`` are USD; the rest are credits. Values are rounded for
    display; the ledger keeps full precision. Attached to a chat answer so the UI
    can render the "This run" readout and refresh the meter from the server."""

    credits_used: float = 0.0
    base_cost_usd: float = 0.0
    charged_usd: float = 0.0
    balance: float = 0.0
    plan_total: float = 0.0
    percent_remaining: float = 100.0


class CreditBalance(DomainModel):
    """A user's current standing — drives the meter on page load / refresh."""

    balance: float = 0.0
    plan_total: float = 0.0
    percent_remaining: float = 100.0


class TopUpRequest(DomainModel):
    """Admin: add credits to the shared pool."""

    amount: float = Field(..., gt=0, description="Credits to add to the pool.")


class CreditRunRequest(DomainModel):
    """Run a raw prompt against Claude with credit metering."""

    prompt: str = Field(..., description="The user's prompt sent to the model.")


class CreditRunResponse(DomainModel):
    """Spec-shaped result of a metered model run."""

    reply: str
    usage: UsageInfo = Field(default_factory=UsageInfo)
    credits_used: float = 0.0
    base_cost_usd: float = 0.0
    charged_usd: float = 0.0
    balance: float = 0.0
    plan_total: float = 0.0
    percent_remaining: float = 100.0


class FeedbackRequest(DomainModel):
    """User feedback on one answer — thumbs up/down plus an optional comment."""

    rating: str = Field(..., description="'up' or 'down'.")
    question: str = ""
    answer: str = ""
    session_id: str | None = None
    comment: str = ""


class ChatRequest(DomainModel):
    message: str
    session_id: str | None = Field(default=None, description="Omit to start a new session.")
    namespace: str = "default"
    filters: dict[str, str] | None = Field(
        default=None, description="Metadata facet filters, e.g. {'supplier':'ACME','country':'AU'}."
    )


class ChatResponse(DomainModel):
    session_id: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    # For listing questions ("which documents are expired / have code X"): the full
    # matching set as clickable rows (first page). ``documents_total`` + ``documents_query``
    # let the UI page through the rest via GET /v1/documents.
    documents: list[DocumentListItem] = Field(default_factory=list)
    documents_total: int = 0
    documents_query: dict[str, str] = Field(default_factory=dict)
    usage: UsageInfo = Field(default_factory=UsageInfo)
    # Credit accounting for this turn + the resulting balance. None when nothing was
    # charged (e.g. the model wasn't called, so no deduction happened).
    credit: CreditInfo | None = None
    took_ms: float = 0.0
    context_expires_in_s: int = 0


class NewSessionResponse(DomainModel):
    session_id: str
    ttl_minutes: float


class AgentChatRequest(DomainModel):
    """A question for the DEPLOYED agent rather than this service's own pipeline."""

    message: str
    # Which ExampleApp deployment to answer as. This is the agent's tenancy key, so
    # it decides which tenant's index is searched -- not a display preference.
    base_url: str
    session_id: str | None = Field(default=None, description="Omit to start a new session.")
    business_object: str | None = None
    object_codes: str | None = Field(
        default=None, description="Comma-separated object codes, as ExampleApp's ACL would send."
    )


class AgentTenantChoice(DomainModel):
    label: str
    base_url: str


class AgentStatus(DomainModel):
    """Whether the deployed agent is reachable from this service, and as what."""

    enabled: bool = False
    runtime_arn: str = ""
    region: str = ""
    tenants: list[AgentTenantChoice] = Field(default_factory=list)
    # Always true while this route exists: demo traffic carries no reservation, so
    # it never reaches Licensing. Surfaced so the UI can say so on screen.
    metered: bool = False


class AgentCitation(DomainModel):
    """A source as the DEPLOYED agent reports it.

    Deliberately not this service's own ``Citation``: the agent is a separate
    deployment with its own envelope (no ``index``, no ``source_key``), and
    coercing one into the other would break the demo every time either side
    added a field. Only ``source_key`` is synthesised, because the UI's document
    grid keys off it and a UUID is not a source anyone can recognise.
    """

    document_id: str = ""
    document_name: str = ""
    source_key: str = ""
    page: int | None = None
    snippet: str = ""
    score: float = 0.0
    facets: dict[str, str] = Field(default_factory=dict)
    download_url: str | None = None

    @classmethod
    def from_agent(cls, raw: dict[str, Any]) -> AgentCitation:
        name = str(raw.get("document_name") or "")
        document_id = str(raw.get("document_id") or "")
        return cls(
            document_id=document_id,
            document_name=name,
            source_key=str(raw.get("source_key") or name or document_id),
            page=raw.get("page"),
            snippet=str(raw.get("snippet") or ""),
            score=float(raw.get("score") or 0.0),
            facets={str(k): str(v) for k, v in (raw.get("facets") or {}).items() if v is not None},
            download_url=raw.get("download_url"),
        )


class AgentChatResponse(DomainModel):
    session_id: str
    answer: str
    citations: list[AgentCitation] = Field(default_factory=list)
    documents: list[AgentCitation] = Field(default_factory=list)
    documents_total: int = 0
    route: str = ""
    tenant: str = ""
    environment: str = ""
    business_object: str = ""
    object_codes: str = ""
    usage: UsageInfo = Field(default_factory=UsageInfo)
    took_ms: float = 0.0
    # No credits are charged for a demo call, locally or in ExampleApp's ledger.
    metered: bool = False
