"""Observability: distributed tracing (OpenTelemetry) + metrics (Prometheus).

Three pillars, one module: traces (tracing.py) and metrics (metrics.py) live
here; logs (correlated via trace ids) live in ``rag_core.logging``. Services call
``setup_tracing`` + ``start_metrics_server`` once at startup, then use
``get_tracer`` / the metric registry throughout.
"""

from __future__ import annotations

from rag_core.observability.metrics import (
    REGISTRY,
    counter,
    gauge,
    histogram,
    start_metrics_server,
    track_latency,
)
from rag_core.observability.tracing import get_tracer, setup_tracing, traced

__all__ = [
    "REGISTRY",
    "counter",
    "gauge",
    "get_tracer",
    "histogram",
    "setup_tracing",
    "start_metrics_server",
    "traced",
    "track_latency",
]
