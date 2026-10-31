"""Prometheus metrics helpers.

Why a thin wrapper over ``prometheus_client``: metric objects must be created
exactly once per process (re-registering the same name raises). These factories
memoise by name so call sites can be naive — ``counter("docs_ingested_total")``
from anywhere returns the same instrument. This matters in a codebase where the
same metric is touched from multiple modules.

A dedicated ``REGISTRY`` (not the global default) keeps tests isolated and lets a
service expose only its own metrics.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

REGISTRY = CollectorRegistry()

# Caches keyed by metric name so repeated factory calls return one instrument.
_counters: dict[str, Counter] = {}
_gauges: dict[str, Gauge] = {}
_histograms: dict[str, Histogram] = {}

# Latency buckets tuned for sub-ms cache hits up to multi-second LLM calls.
DEFAULT_LATENCY_BUCKETS: tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
)


def counter(name: str, documentation: str = "", labelnames: Sequence[str] = ()) -> Counter:
    if name not in _counters:
        _counters[name] = Counter(
            name, documentation or name, list(labelnames), registry=REGISTRY
        )
    return _counters[name]


def gauge(name: str, documentation: str = "", labelnames: Sequence[str] = ()) -> Gauge:
    if name not in _gauges:
        _gauges[name] = Gauge(name, documentation or name, list(labelnames), registry=REGISTRY)
    return _gauges[name]


def histogram(
    name: str,
    documentation: str = "",
    labelnames: Sequence[str] = (),
    buckets: Sequence[float] = DEFAULT_LATENCY_BUCKETS,
) -> Histogram:
    if name not in _histograms:
        _histograms[name] = Histogram(
            name,
            documentation or name,
            list(labelnames),
            buckets=tuple(buckets),
            registry=REGISTRY,
        )
    return _histograms[name]


@contextmanager
def track_latency(metric_name: str, **labels: str) -> Iterator[None]:
    """Time a block and record it into a latency histogram.

    Example::

        with track_latency("embedding_seconds", provider="titan"):
            await embedder.embed(batch)
    """
    hist = histogram(metric_name, f"Latency of {metric_name}", labelnames=tuple(labels.keys()))
    start = time.perf_counter()
    try:
        yield
    finally:
        observed = time.perf_counter() - start
        (hist.labels(**labels) if labels else hist).observe(observed)


def start_metrics_server(port: int) -> None:
    """Expose ``/metrics`` over HTTP for Prometheus to scrape."""
    start_http_server(port, registry=REGISTRY)
