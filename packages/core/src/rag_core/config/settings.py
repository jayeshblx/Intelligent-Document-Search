"""Application settings, validated at startup via pydantic-settings.

All values are environment-driven. Nested groups use the ``__`` delimiter, e.g.::

    AWS__REGION=us-east-1
    QDRANT__URL=http://qdrant:6333
    BEDROCK__LLM_MODEL=us.anthropic.claude-sonnet-4-6-20250929-v1:0

``get_settings()`` is memoised so the (potentially I/O-touching) env/.env parse
happens once per process. Tests can clear the cache or construct ``Settings``
directly with overrides — nothing here reads global state at import time.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """Deployment environment. Drives log format, sampling, and safety toggles."""

    LOCAL = "local"
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class AWSSettings(BaseModel):
    """Base AWS credentials/region. In real AWS, leave keys blank to use the
    instance/task IAM role; ``endpoint_url`` targets LocalStack for dev only."""

    region: str = "us-east-1"
    endpoint_url: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    max_retries: int = 8
    # TLS trust: path to a CA bundle (e.g. exported corporate root CAs for a MITM
    # proxy). If unset and verify_ssl is False, verification is disabled (dev only).
    ca_bundle: str | None = None
    verify_ssl: bool = True


class S3Settings(BaseModel):
    bucket: str = "example-index-bucket"
    prefix: str = "foxtrot/"
    multipart_threshold_mb: int = Field(default=64, ge=5)


class SQSSettings(BaseModel):
    ingestion_queue_url: str = ""
    ingestion_dlq_url: str = ""
    visibility_timeout_s: int = Field(default=300, ge=0)
    max_messages: int = Field(default=10, ge=1, le=10)
    wait_time_s: int = Field(default=20, ge=0, le=20)  # long-poll


class BedrockSettings(BaseModel):
    region: str = "us-east-1"
    titan_embed_model: str = "amazon.titan-embed-text-v2:0"
    # Bedrock inference-profile id (verified available in the target account).
    llm_model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    rerank_model: str = "amazon.rerank-v1:0"
    max_retries: int = 6
    timeout_s: int = 60


class QdrantSettings(BaseModel):
    url: str = "http://localhost:6333"
    api_key: str | None = None
    collection: str = "compliance_chunks"
    prefer_grpc: bool = True
    timeout_s: int = 30


class OpenSearchSettings(BaseModel):
    url: str = "http://localhost:9200"
    index: str = "compliance_fulltext"
    user: str | None = None
    password: str | None = None


class Neo4jSettings(BaseModel):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"


class TemporalSettings(BaseModel):
    host: str = "localhost:7233"
    namespace: str = "default"
    task_queue: str = "rag-ingestion"


class RedisSettings(BaseModel):
    url: str = "redis://localhost:6379/0"


class OtelSettings(BaseModel):
    exporter_otlp_endpoint: str = "http://localhost:4317"
    service_namespace: str = "rag"
    enabled: bool = True


class PrometheusSettings(BaseModel):
    port: int = 9100


class Settings(BaseSettings):
    """Root configuration object. One instance per process via ``get_settings()``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application-level ---
    app_env: Environment = Environment.LOCAL
    app_name: str = "agentic-rag"
    log_level: str = "INFO"
    log_json: bool = True
    debug: bool = False

    # --- Backing services (nested) ---
    aws: AWSSettings = Field(default_factory=AWSSettings)
    s3: S3Settings = Field(default_factory=S3Settings)
    sqs: SQSSettings = Field(default_factory=SQSSettings)
    bedrock: BedrockSettings = Field(default_factory=BedrockSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    temporal: TemporalSettings = Field(default_factory=TemporalSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    otel: OtelSettings = Field(default_factory=OtelSettings)
    prometheus: PrometheusSettings = Field(default_factory=PrometheusSettings)

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {v!r}")
        return upper

    @property
    def is_production(self) -> bool:
        return self.app_env in (Environment.STAGING, Environment.PROD)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Memoised so env/.env parsing happens once. Call ``get_settings.cache_clear()``
    in tests that need to re-read the environment.
    """
    return Settings()
