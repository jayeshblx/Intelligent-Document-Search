"""Async circuit breaker.

Why, on top of retries: retries handle *transient* faults, but when a dependency
(Bedrock, Qdrant, OpenSearch) is genuinely down, continuing to send traffic just
piles up latency and burns quota. The breaker trips OPEN after repeated failures,
fails fast for a cooldown, then probes with a single HALF_OPEN trial before fully
closing — protecting both us and the struggling dependency.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import TypeVar

from rag_core.logging import get_logger

_log = get_logger("rag_core.resilience.circuit")

T = TypeVar("T")


class CircuitState(StrEnum):
    CLOSED = "closed"        # healthy; calls pass through
    OPEN = "open"            # tripped; calls fail fast
    HALF_OPEN = "half_open"  # probing recovery with a single trial


class CircuitBreakerOpen(Exception):
    """Raised when a call is rejected because the breaker is OPEN."""

    def __init__(self, name: str, retry_after_s: float) -> None:
        super().__init__(f"circuit '{name}' is OPEN; retry in {retry_after_s:.1f}s")
        self.retry_after_s = retry_after_s


class CircuitBreaker:
    """A single named breaker. Thread/async-safe via an internal lock.

    Args:
        name: Identifier for logs/metrics (e.g. "bedrock").
        failure_threshold: Consecutive failures before tripping OPEN.
        recovery_timeout_s: Cooldown before a HALF_OPEN probe is allowed.
        success_threshold: Successes in HALF_OPEN before fully CLOSED.
    """

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 30.0,
        success_threshold: int = 2,
    ) -> None:
        self.name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout_s = recovery_timeout_s
        self._success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
        self._opened_at = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        """Execute ``fn`` under breaker protection.

        Raises ``CircuitBreakerOpen`` immediately if tripped and still cooling
        down. On the underlying call, success/failure updates breaker state.
        """
        await self._before_call()
        try:
            result = await fn()
        except Exception:
            await self._on_failure()
            raise
        else:
            await self._on_success()
            return result

    async def _before_call(self) -> None:
        async with self._lock:
            if self._state is CircuitState.OPEN:
                elapsed = time.monotonic() - self._opened_at
                if elapsed < self._recovery_timeout_s:
                    raise CircuitBreakerOpen(self.name, self._recovery_timeout_s - elapsed)
                # Cooldown elapsed — allow a single probe.
                self._state = CircuitState.HALF_OPEN
                self._successes = 0
                _log.info("circuit_half_open", circuit=self.name)

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                self._successes += 1
                if self._successes >= self._success_threshold:
                    self._reset()
                    _log.info("circuit_closed", circuit=self.name)
            else:
                self._failures = 0

    async def _on_failure(self) -> None:
        async with self._lock:
            self._failures += 1
            if self._state is CircuitState.HALF_OPEN or self._failures >= self._failure_threshold:
                self._trip()

    def _trip(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()
        _log.warning("circuit_open", circuit=self.name, failures=self._failures)

    def _reset(self) -> None:
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
