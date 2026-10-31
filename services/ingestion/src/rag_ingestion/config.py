"""Ingestion-service-specific settings (env prefix ``INGESTION_``).

Kept separate from ``rag_core.config.Settings`` so each service only declares the
knobs it owns. Core (S3/SQS/AWS coordinates) still comes from the shared Settings.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChangeDetection(StrEnum):
    """How to decide an object changed.

    ETAG  — cheap; trusts S3 ETag (note: multipart ETags are not plain MD5, so
            an ETag change reliably means *changed*, but equal ETags across
            different upload methods are rare-but-possible false "unchanged").
    CHECKSUM — authoritative; downloads + sha256. Use when correctness > cost.
    """

    ETAG = "etag"
    CHECKSUM = "checksum"


class IngestionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INGESTION_", env_file=".env", extra="ignore", case_sensitive=False
    )

    # Parallelism for per-object download+checksum+emit.
    sync_concurrency: int = Field(default=32, ge=1, le=512)
    # SQS send batching (max 10 per AWS API call).
    emit_batch_size: int = Field(default=10, ge=1, le=10)
    emit_to_sqs: bool = True

    change_detection: ChangeDetection = ChangeDetection.CHECKSUM
    # Skip objects larger than this to protect workers (0 = no limit).
    max_object_size_mb: int = Field(default=512, ge=0)
    # Verify content checksum even when ETag is unchanged (paranoid mode).
    always_verify_checksum: bool = False
