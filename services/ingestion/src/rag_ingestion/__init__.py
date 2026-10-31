"""Document Ingestion Service (platform service #1).

Responsibility: be the *source of truth for what exists*. It performs incremental
S3 sync, detects new/changed/duplicate objects via ETag + content checksum,
assigns deterministic document ids and monotonic versions, and emits
``DISCOVERED`` events onto SQS for the downstream processing pipeline.

It is deliberately a thin, idempotent producer: it owns discovery and dedup, not
extraction. That separation lets ingestion scan millions of keys cheaply while a
separate worker fleet does the heavy processing at its own pace.
"""

from __future__ import annotations

__version__ = "0.1.0"
