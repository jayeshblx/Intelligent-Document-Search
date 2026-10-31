"""Structured logging built on structlog.

Why structlog: compliance workloads need machine-parseable, queryable logs with
stable keys (document_id, chunk_id, trace_id) rather than free-form strings.
structlog gives us key-value events that render as JSON in prod and as colorised
console output in dev, with the *same* call sites.

Logs are automatically correlated with OpenTelemetry traces: every event emitted
within an active span carries ``trace_id``/``span_id`` so a log line can be
pivoted straight to its distributed trace.
"""

from __future__ import annotations

from rag_core.logging.setup import bind_context, configure_logging, get_logger

__all__ = ["bind_context", "configure_logging", "get_logger"]
