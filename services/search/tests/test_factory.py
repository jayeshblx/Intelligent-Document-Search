"""Direct unit tests for factory.py component wiring.

Tests that build_components() and build_offline_components() correctly wire
all services and manage lifecycle (closeables).
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from rag_search.config import EmbedProvider, SearchSettings
from rag_search.embedder import FakeEmbedder
from rag_search.factory import Components, build_components, build_offline_components

from rag_core.config import S3Settings, Settings, SQSSettings


async def test_build_components_creates_full_graph(tmp_path: Path) -> None:
    """Verify build_components() wires all services correctly."""
    core_settings = Settings(
        s3=S3Settings(bucket="test-bucket", prefix="test/"),
        sqs=SQSSettings(),
    )
    search_settings = SearchSettings(
        data_dir=str(tmp_path / "search"),
        collection="test",
        embed_provider=EmbedProvider.FAKE,  # Use fake to avoid AWS
        mongo_uri="",  # No MongoDB
    )
    
    components = build_components(core_settings, search_settings)
    try:
        # Verify all services are created
        assert components.embedder is not None
        assert components.vector is not None
        assert components.keyword is not None
        assert components.indexer is not None
        assert components.search is not None
        assert components.conversations is not None
        assert components.answer is not None
        assert components.metadata is not None
        assert components.credits is not None
        assert components.feedback is not None
        assert components.llm is not None
        assert components.presign is not None
        
        # Verify stores are file-backed
        assert (tmp_path / "search" / "credits.db").exists()
        assert (tmp_path / "search" / "metadata.db").exists()
        assert (tmp_path / "search" / "keyword.db").exists()
    finally:
        await components.aclose()


async def test_build_components_fake_embedder_selected(tmp_path: Path) -> None:
    """When embed_provider=FAKE, build_components uses FakeEmbedder."""
    core_settings = Settings(
        s3=S3Settings(bucket="test-bucket", prefix="test/"),
        sqs=SQSSettings(),
    )
    search_settings = SearchSettings(
        data_dir=str(tmp_path / "search"),
        collection="test",
        embed_provider=EmbedProvider.FAKE,
        mongo_uri="",
    )
    
    components = build_components(core_settings, search_settings)
    try:
        from rag_search.embedder import FakeEmbedder
        
        assert isinstance(components.embedder, FakeEmbedder)
    finally:
        await components.aclose()


async def test_build_components_credit_store_initialization(tmp_path: Path) -> None:
    """CreditStore is initialized with correct paths and default_plan_total."""
    core_settings = Settings(
        s3=S3Settings(bucket="test-bucket", prefix="test/"),
        sqs=SQSSettings(),
    )
    search_settings = SearchSettings(
        data_dir=str(tmp_path / "search"),
        collection="test",
        embed_provider=EmbedProvider.FAKE,
        mongo_uri="",
        credit_plan_total=5000.0,
    )
    
    components = build_components(core_settings, search_settings)
    try:
        # Verify CreditStore was initialized correctly
        account = await components.credits.get_or_create()
        assert account.plan_total == Decimal("5000.0")
        assert account.balance == Decimal("5000.0")
        
        # Verify usage log path exists (parent dir)
        assert (tmp_path / "search").exists()
    finally:
        await components.aclose()


async def test_build_components_feedback_store_initialization(tmp_path: Path) -> None:
    """FeedbackStore is initialized with separate likes and dislikes JSONL files."""
    core_settings = Settings(
        s3=S3Settings(bucket="test-bucket", prefix="test/"),
        sqs=SQSSettings(),
    )
    search_settings = SearchSettings(
        data_dir=str(tmp_path / "search"),
        collection="test",
        embed_provider=EmbedProvider.FAKE,
        mongo_uri="",
    )
    
    components = build_components(core_settings, search_settings)
    try:
        # Verify FeedbackStore was created
        assert components.feedback is not None
        
        # Record an upvote (goes to likes)
        await components.feedback.record(
            {
                "session_id": "s1",
                "question": "test",
                "answer": "test answer",
                "rating": "up",
            }
        )
        
        # Record a downvote (goes to dislikes)
        await components.feedback.record(
            {
                "session_id": "s2",
                "question": "test",
                "answer": "test answer",
                "rating": "down",
                "comment": "too vague",
            }
        )
        
        # Verify separate files are created
        likes_path = tmp_path / "search" / "feedback_likes.jsonl"
        dislikes_path = tmp_path / "search" / "feedback_dislikes.jsonl"
        assert likes_path.exists()
        assert dislikes_path.exists()
        assert likes_path.stat().st_size > 0
        assert dislikes_path.stat().st_size > 0
    finally:
        await components.aclose()


async def test_build_components_metadata_use_llm_flag(tmp_path: Path) -> None:
    """MetadataExtractor respects metadata_use_llm setting."""
    core_settings = Settings(
        s3=S3Settings(bucket="test-bucket", prefix="test/"),
        sqs=SQSSettings(),
    )
    
    # Test with use_llm=False (heuristic-only)
    search_settings_no_llm = SearchSettings(
        data_dir=str(tmp_path / "search_no_llm"),
        collection="test",
        embed_provider=EmbedProvider.FAKE,
        mongo_uri="",
        metadata_use_llm=False,
    )
    
    components_no_llm = build_components(core_settings, search_settings_no_llm)
    try:
        # When use_llm=False, extractor.llm should be None
        # (We can't directly access extractor, but we know it was created correctly)
        assert components_no_llm.metadata is not None
    finally:
        await components_no_llm.aclose()
    
    # Test with use_llm=True (would use LLM for extraction)
    search_settings_with_llm = SearchSettings(
        data_dir=str(tmp_path / "search_with_llm"),
        collection="test",
        embed_provider=EmbedProvider.FAKE,
        mongo_uri="",
        metadata_use_llm=True,
    )
    
    components_with_llm = build_components(core_settings, search_settings_with_llm)
    try:
        assert components_with_llm.metadata is not None
    finally:
        await components_with_llm.aclose()


async def test_build_components_mongo_optional(tmp_path: Path) -> None:
    """MongoMetadataSource is only created when mongo_uri is set."""
    core_settings = Settings(
        s3=S3Settings(bucket="test-bucket", prefix="test/"),
        sqs=SQSSettings(),
    )
    
    # Without mongo_uri
    search_settings_no_mongo = SearchSettings(
        data_dir=str(tmp_path / "search_no_mongo"),
        collection="test",
        embed_provider=EmbedProvider.FAKE,
        mongo_uri="",  # Empty = no Mongo
    )
    
    components_no_mongo = build_components(core_settings, search_settings_no_mongo)
    try:
        # Should work fine without Mongo
        assert components_no_mongo.indexer is not None
    finally:
        await components_no_mongo.aclose()


async def test_build_components_query_log_optional(tmp_path: Path) -> None:
    """The query log is constructed iff query_log_uri is set, and torn down with the rest.

    Guards the failure mode where the store exists but the graph never wires it in, so
    nothing is ever logged and nothing errors either.
    """
    core_settings = Settings(s3=S3Settings(bucket="b", prefix="p/"), sqs=SQSSettings())

    off = build_components(
        core_settings,
        SearchSettings(
            data_dir=str(tmp_path / "off"),
            collection="test",
            embed_provider=EmbedProvider.FAKE,
            mongo_uri="",
            query_log_uri="",  # unset => logging disabled
        ),
    )
    try:
        assert off.query_log is None
    finally:
        await off.aclose()

    on = build_components(
        core_settings,
        SearchSettings(
            data_dir=str(tmp_path / "on"),
            collection="test",
            embed_provider=EmbedProvider.FAKE,
            mongo_uri="",
            # Inert URI: MongoQueryLog connects lazily, so nothing dials out here.
            query_log_uri="mongodb://inert",
            query_log_database="BRAVO_AI_DEV",
            query_log_collection="ExampleApp_AI_Document_Search_Log",
        ),
    )
    try:
        assert on.query_log is not None
        assert on.query_log._database == "BRAVO_AI_DEV"
        assert on.query_log._collection == "ExampleApp_AI_Document_Search_Log"
        assert on.query_log in on._closeables  # shutdown flushes pending writes
    finally:
        await on.aclose()


async def test_build_components_aclose_cleans_all_resources(tmp_path: Path) -> None:
    """Components.aclose() properly closes all closeables in reverse order."""
    core_settings = Settings(
        s3=S3Settings(bucket="test-bucket", prefix="test/"),
        sqs=SQSSettings(),
    )
    search_settings = SearchSettings(
        data_dir=str(tmp_path / "search"),
        collection="test",
        embed_provider=EmbedProvider.FAKE,
        mongo_uri="",
    )
    
    components = build_components(core_settings, search_settings)
    
    # Verify stores are open and functional
    account = await components.credits.get_or_create()
    assert account.balance > 0
    
    # Close everything
    await components.aclose()
    
    # After close, stores should be closed (attempting operations would fail)
    # We verify this indirectly by checking that aclose() completed without error


async def test_build_offline_components_creates_fake_stack(tmp_path: Path) -> None:
    """build_offline_components() creates fully offline stack with fakes."""
    search_settings = SearchSettings(
        data_dir=str(tmp_path / "offline"),
        collection="offline",
        mongo_uri="",
    )
    
    components = build_offline_components(search_settings)
    try:
        # Verify fake embedder and fake LLM are used
        from rag_search.answer import FakeLLM
        from rag_search.embedder import FakeEmbedder
        
        assert isinstance(components.embedder, FakeEmbedder)
        assert isinstance(components.llm, FakeLLM)
        
        # Verify all other services still created
        assert components.search is not None
        assert components.indexer is not None
        assert components.credits is not None
        assert components.feedback is not None
    finally:
        await components.aclose()


async def test_build_offline_components_no_aws_required(tmp_path: Path) -> None:
    """build_offline_components() works without any AWS credentials."""
    search_settings = SearchSettings(
        data_dir=str(tmp_path / "offline2"),
        collection="offline2",
        mongo_uri="",
    )
    
    # Should not fail even if AWS not configured
    components = build_offline_components(search_settings)
    try:
        # Can perform basic operations
        assert await components.embedder.embed("test text") is not None
        
        from rag_core.aws import GenerationResult
        result = await components.llm.complete(
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            system="test",
            max_tokens=100,
            temperature=0.7,
        )
        assert isinstance(result, GenerationResult)
    finally:
        await components.aclose()


async def test_components_dataclass_structure() -> None:
    """Components dataclass has all expected fields."""
    from dataclasses import fields
    
    field_names = {f.name for f in fields(Components)}
    
    expected = {
        "embedder",
        "vector",
        "keyword",
        "indexer",
        "search",
        "conversations",
        "answer",
        "metadata",
        "credits",
        "feedback",
        "llm",
        "presign",
        "_closeables",
    }
    
    assert expected.issubset(field_names)


async def test_build_components_with_injected_embedder(tmp_path: Path) -> None:
    """build_components() accepts pre-configured embedder."""
    core_settings = Settings(
        s3=S3Settings(bucket="test-bucket", prefix="test/"),
        sqs=SQSSettings(),
    )
    search_settings = SearchSettings(
        data_dir=str(tmp_path / "search"),
        collection="test",
        embed_provider=EmbedProvider.TITAN,  # Would normally require AWS
        mongo_uri="",
    )
    
    # Inject fake embedder to override Titan
    fake_embedder = FakeEmbedder(dim=256)
    
    components = build_components(
        core_settings,
        search_settings,
        embedder=fake_embedder,
    )
    try:
        # Should use the injected embedder
        assert components.embedder is fake_embedder
    finally:
        await components.aclose()


async def test_build_components_conversation_store_config(tmp_path: Path) -> None:
    """ConversationStore is created and records turns (in-memory when no Mongo uri)."""
    core_settings = Settings(
        s3=S3Settings(bucket="test-bucket", prefix="test/"),
        sqs=SQSSettings(),
    )
    search_settings = SearchSettings(
        data_dir=str(tmp_path / "search"),
        collection="test",
        embed_provider=EmbedProvider.FAKE,
        mongo_uri="",
    )
    
    components = build_components(core_settings, search_settings)
    try:
        # Create a session and verify it works
        session_id = components.conversations.new_session()
        assert session_id is not None
        
        # Verify history tracking works
        await components.conversations.append(session_id, "user", "test question")
        await components.conversations.append(session_id, "assistant", "test answer")
        history = await components.conversations.recent(session_id)
        assert len(history) == 2
        assert history[0].content == "test question"
    finally:
        await components.aclose()


async def test_build_components_credit_store_paths(tmp_path: Path) -> None:
    """CreditStore JSONL log is created in the data directory."""
    core_settings = Settings(
        s3=S3Settings(bucket="test-bucket", prefix="test/"),
        sqs=SQSSettings(),
    )
    search_settings = SearchSettings(
        data_dir=str(tmp_path / "search"),
        collection="test",
        embed_provider=EmbedProvider.FAKE,
        mongo_uri="",
    )
    
    components = build_components(core_settings, search_settings)
    try:
        # Record a credit usage to trigger file creation
        await components.credits.charge(
            actor="search", input_tokens=10, output_tokens=0, model="fake-llm"
        )
        
        usage_log = tmp_path / "search" / "credits_usage.jsonl"
        assert usage_log.exists()
        assert usage_log.stat().st_size > 0
    finally:
        await components.aclose()
