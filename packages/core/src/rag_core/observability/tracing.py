"""OpenTelemetry tracing setup.

Why: a single query fans out across 9+ agents and 3 stores; a single document
ingest crosses 6 services. Without distributed tracing, debugging a slow or
failed request means grepping logs across services. OTel gives one trace id that
stitches the whole causal chain together (and our logs carry that id — see
``rag_core.logging``).

The exporter is OTLP/gRPC to the collector (``OTEL__EXPORTER_OTLP_ENDPOINT``),
which fans out to Tempo/Jaeger/X-Ray in prod.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Tracer

from rag_core.config import OtelSettings

_INITIALISED = False

P = ParamSpec("P")
R = TypeVar("R")


def setup_tracing(settings: OtelSettings, *, service_name: str) -> None:
    """Initialise the global tracer provider. Idempotent; no-op if disabled.

    Args:
        settings: OTel config (endpoint, namespace, enabled flag).
        service_name: e.g. "ingestion" — becomes ``service.name`` on every span.
    """
    global _INITIALISED
    if _INITIALISED or not settings.enabled:
        return

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.namespace": settings.service_namespace,
        }
    )
    provider = TracerProvider(resource=resource)

    # Imported here so the OTLP/gRPC exporter dependency is only needed when
    # tracing is actually enabled.
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    exporter = OTLPSpanExporter(endpoint=settings.exporter_otlp_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    _INITIALISED = True


def get_tracer(name: str) -> Tracer:
    """Return a tracer. Works even if ``setup_tracing`` was never called
    (returns a no-op tracer), so library code can trace unconditionally."""
    return trace.get_tracer(name)


def traced(
    name: str | None = None,
    *,
    tracer_name: str = "rag_core",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that wraps a sync or async function in a span.

    Records exceptions and sets the span status to ERROR on failure so traces
    surface the actual fault, not just a missing child span.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        span_name = name or f"{func.__module__}.{func.__qualname__}"
        tracer = get_tracer(tracer_name)

        if _is_coroutine(func):

            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                with tracer.start_as_current_span(span_name) as span:
                    try:
                        return await func(*args, **kwargs)  # type: ignore[misc]
                    except Exception as exc:
                        span.record_exception(exc)
                        span.set_status(trace.StatusCode.ERROR, str(exc))
                        raise

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with tracer.start_as_current_span(span_name) as span:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(trace.StatusCode.ERROR, str(exc))
                    raise

        return sync_wrapper

    return decorator


def _is_coroutine(func: Callable[..., Any]) -> bool:
    import asyncio

    return asyncio.iscoroutinefunction(func)


# Re-exported for callers that want to add attributes to the current span.
def current_span() -> trace.Span:
    return trace.get_current_span()
