"""Typed, 12-factor configuration for the platform.

Why this module exists
----------------------
Every service needs the same backing-store coordinates (Qdrant, OpenSearch,
Neo4j, Bedrock, S3, SQS, ...) but must never hardcode them. Centralising config
in one validated ``Settings`` object means:

* a single source of truth that fails fast at startup if misconfigured,
* type-safe access (`settings.qdrant.url`) with IDE/mypy support,
* trivial overrides per environment via env vars / secrets managers.

Design choice: nested ``pydantic`` models with the ``__`` env delimiter
(`QDRANT__URL`). This keeps related knobs grouped while remaining flat in the
environment, which is what Docker/K8s/Secrets-Manager all speak natively.
"""

from __future__ import annotations

from rag_core.config.settings import (
    AWSSettings,
    BedrockSettings,
    Environment,
    Neo4jSettings,
    OpenSearchSettings,
    OtelSettings,
    PrometheusSettings,
    QdrantSettings,
    RedisSettings,
    S3Settings,
    Settings,
    SQSSettings,
    TemporalSettings,
    get_settings,
)

__all__ = [
    "AWSSettings",
    "BedrockSettings",
    "Environment",
    "Neo4jSettings",
    "OpenSearchSettings",
    "OtelSettings",
    "PrometheusSettings",
    "QdrantSettings",
    "RedisSettings",
    "S3Settings",
    "SQSSettings",
    "Settings",
    "TemporalSettings",
    "get_settings",
]
