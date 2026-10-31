"""Builds and owns the search-service component graph.

One place wires embedder -> indexes -> processing -> indexer/search, so the app
lifespan and tests share identical construction. ``aclose`` tears everything down.
"""

from __future__ import annotations

import contextlib
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from rag_core.config import Settings
from rag_core.di import Container
from rag_processing.config import ProcessingSettings
from rag_processing.extractors import build_default_registry
from rag_processing.ocr import build_ocr_engine
from rag_processing.service import ProcessingService
from rag_processing.store import InMemoryContentStore
from rag_search.answer import LLM, AnswerService, BedrockLLM, FakeLLM
from rag_search.chunking import Chunker
from rag_search.config import EmbedProvider, SearchSettings, StoreBackend
from rag_search.conversation import ConversationStore, _MongoBackend
from rag_search.credit_store import CreditStore
from rag_search.downloads import DownloadService
from rag_search.embedder import Embedder, FakeEmbedder, TitanEmbedder
from rag_search.feedback import FeedbackStore
from rag_search.indexer import IndexingService
from rag_search.metadata_extract import MetadataExtractor
from rag_search.metadata_store import MetadataStore
from rag_search.retrieval import SearchService
from rag_search.stores import (
    DocumentMetadataStore,
    KeywordIndex,
    KeywordSearchIndex,
    LocalVectorIndex,
    MongoIndexDatabase,
    MongoKeywordIndex,
    MongoMetadataStore,
    MongoVectorIndex,
    PgDatabase,
    PgKeywordIndex,
    PgMetadataStore,
    PgVectorIndex,
    VectorIndex,
)


@dataclass(slots=True)
class Components:
    embedder: Embedder
    vector: VectorIndex
    keyword: KeywordSearchIndex
    indexer: IndexingService
    search: SearchService
    conversations: ConversationStore
    answer: AnswerService
    metadata: DocumentMetadataStore
    credits: CreditStore
    feedback: FeedbackStore
    llm: LLM
    presign: Callable[[str], Awaitable[str | None]]
    # Streams source documents to the browser behind credential-free, filename-ending URLs.
    downloads: DownloadService
    # None when SEARCH_QUERY_LOG_URI is unset — logging is opt-in.
    query_log: Any | None
    _closeables: list[Any]

    async def aclose(self) -> None:
        for resource in reversed(self._closeables):
            with contextlib.suppress(Exception):  # best-effort shutdown
                await resource.aclose()


def build_mongo_index(search_settings: SearchSettings, *, dim: int) -> MongoIndexDatabase:
    """The tenant collection named by ``SEARCH_INDEX_MONGO_*`` (URI defaults to ``mongo_uri``)."""
    uri = search_settings.index_mongo_uri or search_settings.mongo_uri
    if not uri or not search_settings.index_mongo_collection:
        raise ValueError(
            "SEARCH_STORE_BACKEND=mongo needs SEARCH_INDEX_MONGO_COLLECTION (e.g. "
            "FOXTROT_DocumentSearch_DEV) and a connection string in SEARCH_INDEX_MONGO_URI "
            "or SEARCH_MONGO_URI"
        )
    return MongoIndexDatabase(
        uri,
        database=search_settings.index_mongo_database,
        collection=search_settings.index_mongo_collection,
        dim=dim,
        tls_ca_file=search_settings.index_mongo_tls_ca_file or search_settings.mongo_tls_ca_file,
        exact_vector_search=search_settings.index_mongo_exact_vectors,
        query_timeout_ms=search_settings.index_mongo_query_timeout_ms,
    )


def build_components(
    core_settings: Settings,
    search_settings: SearchSettings,
    *,
    container: Container | None = None,
    embedder: Embedder | None = None,
    llm: BedrockLLM | FakeLLM | None = None,
) -> Components:
    """Construct the component graph. Pass ``embedder``/``llm`` to inject fakes in tests."""
    container = container or Container(core_settings)
    data_dir = Path(search_settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if embedder is None:
        if search_settings.embed_provider is EmbedProvider.FAKE:
            embedder = FakeEmbedder(dim=256)
        else:
            embedder = TitanEmbedder(
                container.bedrock(),
                dim=search_settings.embed_dim,
                batch=search_settings.embed_batch,
            )

    # Vectors, keyword text and document metadata. Every backend exposes the same
    # surfaces, so nothing downstream (indexer, search, answer) changes.
    vector: VectorIndex
    keyword: KeywordSearchIndex
    metadata_store: DocumentMetadataStore
    if search_settings.store_backend is StoreBackend.POSTGRES:
        # One server, one connection pool, one transaction per document. Also lifts
        # the embedded-Qdrant restriction of a single process per index directory.
        pg = PgDatabase(
            search_settings.postgres_dsn,
            dim=embedder.dim,
            max_size=search_settings.postgres_pool_max,
        )
        vector = PgVectorIndex(pg)
        keyword = PgKeywordIndex(pg)
        metadata_store = PgMetadataStore(pg, namespace=search_settings.namespace)
    elif search_settings.store_backend is StoreBackend.MONGO:
        # One Atlas collection holds this tenant's chunks, vectors and metadata.
        mongo = build_mongo_index(search_settings, dim=embedder.dim)
        vector = MongoVectorIndex(mongo)
        keyword = MongoKeywordIndex(mongo)
        metadata_store = MongoMetadataStore(mongo, namespace=search_settings.namespace)
    else:
        vector = LocalVectorIndex(
            str(data_dir / "qdrant"), collection=search_settings.collection, dim=embedder.dim
        )
        keyword = KeywordIndex(str(data_dir / "keyword.db"))
        metadata_store = MetadataStore(str(data_dir / "metadata.db"))

    # LLM (shared by chat + metadata extraction).
    if llm is None:
        llm = (
            FakeLLM()
            if search_settings.embed_provider is EmbedProvider.FAKE
            else BedrockLLM(container.bedrock())
        )

    # Credit metering ledger — the authoritative shared balance (persists across
    # browser refreshes and restarts; every concurrent user draws down the same
    # pool). Each charge is also appended to a JSON usage log for auditing.
    credit_store = CreditStore(
        str(data_dir / "credits.db"),
        default_plan_total=Decimal(str(search_settings.credit_plan_total)),
        usage_log_path=str(data_dir / "credits_usage.jsonl"),
    )

    # Per-answer user feedback -> separate append-only JSONL logs for likes vs dislikes.
    feedback_store = FeedbackStore(
        str(data_dir / "feedback_likes.jsonl"),
        str(data_dir / "feedback_dislikes.jsonl"),
    )

    # Metadata Extraction Agent (#4): extractor over the store built above.
    metadata_extractor = MetadataExtractor(
        llm if search_settings.metadata_use_llm else None,
        use_llm=search_settings.extract_metadata and search_settings.metadata_use_llm,
    )

    # External MongoDB metadata (system of record) — optional, enabled by mongo_uri.
    mongo_source = None
    if search_settings.mongo_uri:
        from rag_search.mongo_source import MongoMetadataSource

        mongo_source = MongoMetadataSource(
            search_settings.mongo_uri,
            database=search_settings.mongo_database,
            collection=search_settings.mongo_collection,
            join_field=search_settings.mongo_join_field,
            tls_ca_file=search_settings.mongo_tls_ca_file,
            tenant_codes=search_settings.mongo_tenant_codes,
        )

    # Query audit log -> MongoDB (optional, enabled by query_log_uri). A different
    # database/credential to the metadata source above, and write-only.
    query_log = None
    if search_settings.query_log_uri:
        from rag_search.query_log import MongoQueryLog

        query_log = MongoQueryLog(
            search_settings.query_log_uri,
            database=search_settings.query_log_database,
            collection=search_settings.query_log_collection,
            tls_ca_file=search_settings.query_log_tls_ca_file,
            max_answer_chars=search_settings.query_log_max_answer_chars,
        )

    # Extraction/enrichment is reused from the processing service (no SQS emit,
    # ephemeral in-memory content store).
    content_store = InMemoryContentStore()
    ps = ProcessingSettings(run_worker=False, output_queue_url="")
    ocr = build_ocr_engine(ps, container.aws_factory())
    processing = ProcessingService(
        s3=container.s3(),
        sqs=container.sqs(),
        registry=build_default_registry(),
        ocr_engine=ocr,
        content_store=content_store,
        core_settings=core_settings,
        settings=ps,
    )

    chunker = Chunker(
        parent_max_words=search_settings.parent_max_words,
        child_words=search_settings.child_words,
        child_overlap=search_settings.child_overlap,
    )
    indexer = IndexingService(
        s3=container.s3(),
        processing=processing,
        content_store=content_store,
        chunker=chunker,
        embedder=embedder,
        vector=vector,
        keyword=keyword,
        metadata_extractor=metadata_extractor,
        metadata_store=metadata_store,
        core_settings=core_settings,
        settings=search_settings,
        mongo=mongo_source,
    )

    # Source-document downloads. The UI is handed an app-relative URL ending in the
    # original filename (streamed through the app), NOT a presigned S3 URL — so the bucket,
    # the storage key and the AWS access-key id never reach the browser. The link carries a
    # short HMAC token so the streaming endpoint can't be turned into an open proxy.
    s3_client = container.s3()
    bucket = core_settings.s3.bucket
    download_secret = (
        search_settings.download_url_secret.encode()
        if search_settings.download_url_secret
        else secrets.token_bytes(32)  # per-process key: links stay valid until restart
    )
    downloads = DownloadService(
        s3_client, bucket, secret=download_secret, ttl_s=search_settings.download_url_ttl_s
    )

    async def _presign(key: str) -> str | None:
        return downloads.url_for(key)

    search = SearchService(
        embedder=embedder,
        vector=vector,
        keyword=keyword,
        settings=search_settings,
        presign=_presign,
    )

    # Conversational RAG (chat over the search results). Durable, multi-user memory:
    # persisted to MongoDB, bounded in-process LRU, token-budgeted window, rolling
    # summary + semantic recall for long sessions. The conversation Mongo uri defaults to
    # the query-log database (same app DB); empty everywhere → in-memory only (non-durable).
    conv_uri = search_settings.conversation_uri or search_settings.query_log_uri
    conv_backend = None
    if conv_uri:
        conv_backend = _MongoBackend(
            conv_uri,
            database=search_settings.conversation_database,
            collection=search_settings.conversation_collection,
            tls_ca_file=search_settings.conversation_tls_ca_file
            or search_settings.query_log_tls_ca_file,
        )
    conversations = ConversationStore(
        ttl_minutes=search_settings.session_ttl_minutes,
        token_budget=search_settings.history_token_budget,
        hard_message_cap=search_settings.history_hard_message_cap,
        summarize_batch=search_settings.history_summarize_batch,
        semantic_recall=search_settings.history_semantic_recall,
        recall_k=search_settings.history_recall_k,
        cache_capacity=search_settings.conversation_cache_capacity,
        embedder=embedder,
        llm=llm,
        backend=conv_backend,
    )
    # Reads each question's intent with the model rather than hand-written regexes.
    # Shares the LLM: routing is one small extra call per question.
    from rag_search.query_router import QueryRouter

    answer = AnswerService(
        search=search,
        llm=llm,
        router=QueryRouter(llm) if search_settings.query_router else None,
        conversations=conversations,
        metadata=metadata_store,  # enables facet-aware questions (e.g. "object code SUPPLIER0005")
        presign=_presign,         # download links on listed documents
        top_k=search_settings.answer_top_k,
        max_tokens=search_settings.answer_max_tokens,
        temperature=search_settings.answer_temperature,
        prompt_cache=search_settings.answer_prompt_cache,
    )

    closeables: list[Any] = [embedder, ocr, vector, keyword, metadata_store, credit_store, container]
    if conv_backend is not None:
        closeables.append(conversations)
    if mongo_source is not None:
        closeables.append(mongo_source)
    if query_log is not None:
        closeables.append(query_log)
    return Components(
        embedder=embedder,
        vector=vector,
        keyword=keyword,
        indexer=indexer,
        search=search,
        conversations=conversations,
        answer=answer,
        metadata=metadata_store,
        credits=credit_store,
        feedback=feedback_store,
        llm=llm,
        presign=_presign,
        downloads=downloads,
        query_log=query_log,
        _closeables=closeables,
    )


def build_offline_components(search_settings: SearchSettings) -> Components:
    """Fully offline components (fake embedder + fake LLM, no AWS) — demo/tests."""
    from rag_core.config import Settings as CoreSettings

    return build_components(
        CoreSettings(), search_settings, embedder=FakeEmbedder(dim=256), llm=FakeLLM()
    )
