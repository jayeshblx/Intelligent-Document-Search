"""Resilience primitives: typed exceptions, retry policies, and a circuit breaker.

These are the building blocks every network-touching call in the platform uses.
Centralising them keeps retry/backoff behaviour consistent (and tunable in one
place) instead of each service inventing its own ad-hoc loops.
"""

from __future__ import annotations

from rag_core.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerOpen, CircuitState
from rag_core.resilience.errors import (
    NonRetryableError,
    RateLimitedError,
    RetryableError,
    TransientError,
)
from rag_core.resilience.retry import async_retry, retry, retryable_status

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerOpen",
    "CircuitState",
    "NonRetryableError",
    "RateLimitedError",
    "RetryableError",
    "TransientError",
    "async_retry",
    "retry",
    "retryable_status",
]
