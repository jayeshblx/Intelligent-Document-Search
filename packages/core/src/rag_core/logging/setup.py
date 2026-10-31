"""structlog + stdlib logging configuration.

``configure_logging`` is idempotent and must be called once per process during
startup (see each service's app factory). It routes stdlib logging (boto3,
uvicorn, qdrant-client, ...) through the same structlog pipeline so every line —
ours or a dependency's — comes out in one consistent format.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

_CONFIGURED = False


def _add_otel_trace_context(_: Any, __: str, event_dict: EventDict) -> EventDict:
    """Inject the active OTel trace/span ids so logs pivot to traces.

    Imported lazily and guarded: logging must work even when OTel isn't
    installed/initialised (e.g. unit tests, CLI tools).
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx and ctx.is_valid:
            event_dict["trace_id"] = format(ctx.trace_id, "032x")
            event_dict["span_id"] = format(ctx.span_id, "016x")
    except Exception:
        pass
    return event_dict


def _shared_processors() -> list[Processor]:
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        _add_otel_trace_context,
    ]


def configure_logging(
    *,
    level: str = "INFO",
    json_logs: bool = True,
    service_name: str | None = None,
) -> None:
    """Configure process-wide structured logging. Idempotent.

    Args:
        level: Minimum level (DEBUG/INFO/WARNING/ERROR/CRITICAL).
        json_logs: JSON renderer (prod) vs. colorised console (dev).
        service_name: Bound to every event as ``service`` for multi-service queries.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    shared = _shared_processors()

    renderer: Processor = (
        structlog.processors.JSONRenderer(serializer=_orjson_dumps)
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    # structlog -> stdlib bridge so library logs share the pipeline.
    structlog.configure(
        processors=[
            *shared,
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Tame noisy dependencies; let our own loggers flow at the configured level.
    for noisy in ("botocore", "boto3", "urllib3", "s3transfer", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if service_name:
        structlog.contextvars.bind_contextvars(service=service_name)

    _CONFIGURED = True


def _orjson_dumps(obj: Any, *, default: Any) -> str:
    import orjson

    return orjson.dumps(obj, default=default).decode("utf-8")


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger. Safe to call before ``configure_logging``."""
    return structlog.get_logger(name)


def bind_context(**kwargs: Any) -> None:
    """Bind key-values to the current async/thread context.

    Everything logged downstream in this context (e.g. one document's processing)
    carries these keys without threading them through every call.
    """
    structlog.contextvars.bind_contextvars(**kwargs)
