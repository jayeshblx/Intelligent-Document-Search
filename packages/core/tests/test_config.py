"""Tests for typed configuration."""

from __future__ import annotations

import pytest

from rag_core.config import Settings, get_settings


def test_defaults_construct_without_env() -> None:
    s = Settings()
    assert s.s3.bucket  # has a sensible default
    assert s.qdrant.collection == "compliance_chunks"
    assert s.bedrock.llm_model.startswith("us.anthropic.claude")


def test_nested_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT__URL", "http://qdrant.internal:6333")
    monkeypatch.setenv("QDRANT__COLLECTION", "my_chunks")
    s = Settings()
    assert s.qdrant.url == "http://qdrant.internal:6333"
    assert s.qdrant.collection == "my_chunks"


def test_log_level_is_validated_and_uppercased() -> None:
    assert Settings(log_level="debug").log_level == "DEBUG"
    with pytest.raises(ValueError):
        Settings(log_level="verbose")


def test_is_production_flag() -> None:
    assert Settings(app_env="prod").is_production is True
    assert Settings(app_env="local").is_production is False


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()
    get_settings.cache_clear()
