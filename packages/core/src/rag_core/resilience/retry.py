"""Retry decorators built on tenacity.

Policy: exponential backoff with full jitter (avoids thundering-herd retries when
many workers fail simultaneously), capped attempts, and retry only on
``RetryableError``/``RateLimitedError``. ``NonRetryableError`` propagates
immediately so bad messages reach the DLQ fast instead of being hammered.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from tenacity import retry as _tenacity_retry
from tenacity import (
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from rag_core.logging import get_logger
from rag_core.resilience.errors import RateLimitedError, RetryableError

_log = get_logger("rag_core.resilience")

P = ParamSpec("P")
R = TypeVar("R")

# HTTP/AWS status codes worth retrying.
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


def retryable_status(status_code: int) -> bool:
    """True if an HTTP/AWS status code represents a transient condition."""
    return status_code in _RETRYABLE_STATUS


def _log_before_sleep(retry_state: Any) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    _log.warning(
        "retrying_after_error",
        attempt=retry_state.attempt_number,
        sleep_s=round(getattr(retry_state.next_action, "sleep", 0.0), 3),
        error=str(exc) if exc else None,
        error_type=type(exc).__name__ if exc else None,
    )


def _make(
    *,
    max_attempts: int,
    initial_wait: float,
    max_wait: float,
    exc_types: tuple[type[BaseException], ...],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Build a tenacity decorator. Works for both sync and async callables —
    tenacity dispatches on whether the wrapped function is a coroutine."""
    return _tenacity_retry(
        retry=retry_if_exception_type(exc_types),
        wait=wait_random_exponential(multiplier=initial_wait, max=max_wait),
        stop=stop_after_attempt(max_attempts),
        before_sleep=_log_before_sleep,
        reraise=True,
    )


def retry(
    *,
    max_attempts: int = 5,
    initial_wait: float = 0.5,
    max_wait: float = 30.0,
    exc_types: tuple[type[BaseException], ...] = (RetryableError, RateLimitedError),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry decorator for synchronous functions."""
    return _make(
        max_attempts=max_attempts,
        initial_wait=initial_wait,
        max_wait=max_wait,
        exc_types=exc_types,
    )


def async_retry(
    *,
    max_attempts: int = 5,
    initial_wait: float = 0.5,
    max_wait: float = 30.0,
    exc_types: tuple[type[BaseException], ...] = (RetryableError, RateLimitedError),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry decorator for async functions (same policy; explicit name for intent)."""
    return _make(
        max_attempts=max_attempts,
        initial_wait=initial_wait,
        max_wait=max_wait,
        exc_types=exc_types,
    )
