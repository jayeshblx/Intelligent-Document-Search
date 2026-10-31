"""Search-service settings (env prefix ``SEARCH_``)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbedProvider(StrEnum):
    TITAN = "titan"   # Amazon Bedrock Titan v2 (default)
    FAKE = "fake"     # deterministic hashing embedder (tests / offline demo)


class StoreBackend(StrEnum):
    """Where vectors, keyword text and metadata live."""

    # Embedded Qdrant + SQLite files under ``data_dir``. Zero infrastructure, but
    # the Qdrant directory is locked to ONE process, so indexing and serving
    # cannot run at the same time.
    LOCAL = "local"
    # One Postgres (pgvector) server for all three. Needs a server, but allows
    # concurrent readers/writers and indexes a document in one transaction.
    POSTGRES = "postgres"
    # One MongoDB Atlas collection per tenant for all three (Atlas Vector Search +
    # Atlas Search). Managed, always on, and read live by the agents -- no snapshot.
    MONGO = "mongo"


class SearchSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SEARCH_", env_file=".env", extra="ignore", case_sensitive=False
    )

    # Local storage root (Qdrant dir + SQLite file live here).
    data_dir: str = "./data/search"
    collection: str = "compliance"
    namespace: str = "default"

    # Storage backend for vectors + keyword + metadata (see ``StoreBackend``).
    # SEARCH_STORE_BACKEND=postgres switches all three onto ``postgres_dsn``.
    store_backend: StoreBackend = StoreBackend.LOCAL
    postgres_dsn: str = "postgresql://rag:rag@localhost:5432/rag"
    postgres_pool_max: int = Field(default=10, ge=1)
    # SEARCH_STORE_BACKEND=mongo: the tenant's collection. Naming: database
    # EX_<FEATURE>_<ENV>, collection <TenantCode>_DocumentSearch_<ENV>, e.g.
    # EX_DOCUMENTSEARCH_DEV.FOXTROT_DocumentSearch_DEV.
    # A blank URI reuses ``mongo_uri`` (same cluster as the system of record).
    index_mongo_uri: str = ""
    index_mongo_database: str = "EX_DOCUMENTSEARCH_DEV"
    index_mongo_collection: str = ""
    index_mongo_tls_ca_file: str | None = None
    # Exact (brute-force) vector search reads EVERY vector per query: 30 s on 45k
    # chunks and 60 s+ on 300k in Atlas -- a timeout. Approximate (HNSW, rescored on
    # full-precision vectors) matched exact/Qdrant quality in the FOXTROT/BRAVO benchmark,
    # so it is the default; exact is for offline comparisons only.
    index_mongo_exact_vectors: bool = False
    # Server-side cap per search query. A query that hits it fails fast and the
    # other retriever (keyword or vector) still answers -- see HybridSearch.
    index_mongo_query_timeout_ms: int = Field(default=8_000, ge=100)
    # Re-run a cheap probe query this often so Atlas keeps the search indexes in
    # memory and the first user after a quiet spell is not the one who pays the
    # cold read. 0 disables (warm-up at startup still happens).
    index_mongo_keep_warm_s: float = Field(default=240.0, ge=0.0)

    # Embeddings
    embed_provider: EmbedProvider = EmbedProvider.TITAN
    embed_dim: int = Field(default=1024, ge=1)
    embed_batch: int = Field(default=64, ge=1)

    # Chunking (approx word counts; child = retrieved unit, parent = context)
    parent_max_words: int = Field(default=350, ge=50)
    child_words: int = Field(default=180, ge=20)
    child_overlap: int = Field(default=40, ge=0)

    # Retrieval
    candidate_k: int = Field(default=30, ge=1)   # per-retriever candidates before fusion
    top_k: int = Field(default=8, ge=1)          # results returned to the UI
    rrf_k: int = Field(default=60, ge=1)         # RRF damping constant

    # Indexing
    index_max_docs: int = Field(default=0, ge=0)  # cap docs indexed (0 = no cap)
    # Per-document extraction+embedding timeout — protects the run from a single
    # pathological file (e.g. pdfplumber stalling on a complex PDF).
    doc_timeout_s: float = Field(default=45.0, ge=1.0)

    # Credit metering (usage-based billing shown in the UI). The rates/overhead/
    # credit value live in ``credit_config.py`` (single source of truth); only the
    # shared-pool starting balance is deployment-tunable here.
    credit_plan_total: float = Field(default=1000.0, gt=0)  # starting credits (1000 = $1.00)
    # Refill the shared pool to ``credit_plan_total`` on each app start. False (default)
    # carries the balance across restarts, so usage resumes where it left off instead of
    # jumping back to 100%. Set True only if you want every restart to top up to full.
    credit_reset_on_start: bool = False

    # Conversational RAG (chat). Context persists until the user clicks "New chat";
    # set > 0 to also auto-expire the context after that many idle minutes.
    session_ttl_minutes: float = Field(default=0.0, ge=0.0)  # 0 = never expire
    max_history_turns: int = Field(default=12, ge=1)          # (legacy) still read by tests
    answer_top_k: int = Field(default=6, ge=1, le=20)         # passages grounded on
    answer_max_tokens: int = Field(default=1024, ge=64)
    answer_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    # Read question intent with the model instead of hand-written regexes. Costs one
    # small extra LLM call per question; set false to fall back to the pattern rules.
    query_router: bool = True

    # ---- Conversation memory (durable, multi-user; see conversation.py) ---------
    # Recent context is chosen by a TOKEN budget (not a fixed turn count): short turns
    # keep many, long turns keep fewer — so a follow-up to question #1 after 20 prompts
    # still has it, as long as the thread fits the budget.
    history_token_budget: int = Field(default=30000, ge=1000)
    # Retained messages per session (available for semantic recall). Beyond this, the
    # oldest are compressed into a rolling summary so nothing is truly forgotten.
    history_hard_message_cap: int = Field(default=200, ge=20)
    history_summarize_batch: int = Field(default=40, ge=2)   # turns compressed per summary pass
    # Tier-3: embed each user turn and pull back the ones most relevant to the current
    # question (beyond the recent window). Off = recent window + summary only.
    history_semantic_recall: bool = True
    history_recall_k: int = Field(default=4, ge=1, le=20)
    # In-process LRU of active sessions; Mongo is the source of truth so eviction only
    # frees RAM. Caps memory no matter how many concurrent users.
    conversation_cache_capacity: int = Field(default=1000, ge=16)
    # Where conversations persist. Empty uri falls back to the query-log Mongo (same app
    # database); if that is also empty the store runs in-memory only (non-durable).
    conversation_uri: str = ""
    conversation_database: str = "BRAVO_AI_DEV"
    conversation_collection: str = "ExampleApp_AI_Conversations"
    conversation_tls_ca_file: str | None = None

    # Bedrock prompt caching: cache the stable prompt prefix (long system prompt) so
    # follow-up turns pay the ~90%-discounted cache-read rate on it instead of full price.
    answer_prompt_cache: bool = True

    # Download links. The UI never receives a presigned S3 URL (those leak the bucket, the
    # storage key and an AWS access-key id into the address bar); instead it gets an
    # app-relative URL ending in the original filename, and the app streams the file. That
    # URL carries a short HMAC token so the endpoint can't be turned into an open proxy to
    # the bucket. Leave the secret blank to sign with a per-process random key (fine for a
    # single worker); set it to keep links valid across restarts / multiple workers.
    download_url_secret: str = ""
    download_url_ttl_s: int = Field(default=86_400, ge=60)  # link lifetime (default 24h)

    # Metadata Extraction Agent (#4)
    extract_metadata: bool = True             # extract + store rich facets at index time
    metadata_use_llm: bool = True             # LLM facets (country/supplier/…); off = heuristics only (free)

    # External MongoDB metadata (system of record for document facets).
    # When a doc is found here its Mongo fields (BusinessObject/ObjectCode/Status/
    # DocumentCategory/…) become authoritative facets and the LLM step is skipped
    # (free + accurate). Empty ``mongo_uri`` disables the integration entirely.
    mongo_uri: str = ""
    mongo_database: str = "FOXTROT_DEV"
    mongo_collection: str = "AttachmentMetaData"
    # S3 basename == GeneratedFileName (verified) — index by both for a robust join.
    mongo_join_field: str = "GeneratedFileName,OriginalFileName"
    mongo_tls_ca_file: str | None = None         # corporate-proxy CA bundle, if TLS is intercepted
    # Comma-separated TenantCode(s) this index is for, e.g. "FOXTROT". Neither the S3 folder
    # nor the Mongo database separates tenants (FOXTROT_DEV is mostly ACMEFOODS), so without
    # this an index silently mixes tenants. Blank = no filter.
    mongo_tenant_codes: str = ""

    # Query log (audit trail of every answered question) — a SEPARATE database and
    # credential from the metadata source above, and write-only rather than read-only,
    # so it gets its own settings. Empty ``query_log_uri`` disables logging entirely.
    query_log_uri: str = ""
    query_log_database: str = "BRAVO_AI_DEV"
    query_log_collection: str = "ExampleApp_AI_Document_Search_Log"
    query_log_tls_ca_file: str | None = None
    # Truncation caps: an answer is normally ~2 KB, but a runaway one must not push the
    # document near Mongo's 16 MB per-document ceiling.
    query_log_max_answer_chars: int = Field(default=20000, ge=500)

    # ---- deployed AgentCore runtime (optional, for local demos) -------------------
    # When set, the UI can send a question to the DEPLOYED orchestrator instead of
    # this service's own pipeline, so a demo shows what ExampleApp actually receives.
    # Blank disables the feature entirely: no route, no client, no behaviour change.
    #
    # Calls made this way are deliberately unmetered -- no ReservationId and no
    # AccessToken -- so they cannot write into ExampleApp's credit ledger.
    agent_runtime_arn: str = ""
    agent_region: str = "ap-south-1"
    # Tenants offered in the UI's selector, as "label=baseUrl" pairs.
    agent_tenants: str = "FOXTROT=https://foxtrot.acmedev.com/,BRAVO=https://bravo.acmedev.com/"
    # A cold tenant can wait ~30 s for its index, and a cold container minutes more.
    agent_read_timeout_s: float = Field(default=600.0, ge=1.0)
    agent_tls_ca_file: str | None = None

    def agent_tenant_choices(self) -> list[tuple[str, str]]:
        """The (label, base_url) pairs the UI should offer."""
        choices: list[tuple[str, str]] = []
        for part in (self.agent_tenants or "").split(","):
            label, _, url = part.partition("=")
            if label.strip() and url.strip():
                choices.append((label.strip(), url.strip()))
        return choices
