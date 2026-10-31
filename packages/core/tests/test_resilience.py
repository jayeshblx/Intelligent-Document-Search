"""Tests for retry policy and circuit breaker."""

from __future__ import annotations

import pytest

from rag_core.resilience import (
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitState,
    NonRetryableError,
    RetryableError,
    async_retry,
    retry,
    retryable_status,
)


def test_retryable_status() -> None:
    assert retryable_status(503) is True
    assert retryable_status(429) is True
    assert retryable_status(404) is False


def test_retry_retries_then_succeeds() -> None:
    calls = {"n": 0}

    @retry(max_attempts=4, initial_wait=0.0, max_wait=0.0)
    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RetryableError("blip")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3


def test_retry_gives_up_after_max_attempts() -> None:
    calls = {"n": 0}

    @retry(max_attempts=2, initial_wait=0.0, max_wait=0.0)
    def always_fail() -> None:
        calls["n"] += 1
        raise RetryableError("nope")

    with pytest.raises(RetryableError):
        always_fail()
    assert calls["n"] == 2


def test_non_retryable_propagates_immediately() -> None:
    calls = {"n": 0}

    @retry(max_attempts=5, initial_wait=0.0, max_wait=0.0)
    def bad_input() -> None:
        calls["n"] += 1
        raise NonRetryableError("permanent")

    with pytest.raises(NonRetryableError):
        bad_input()
    assert calls["n"] == 1


async def test_async_retry_works() -> None:
    calls = {"n": 0}

    @async_retry(max_attempts=3, initial_wait=0.0, max_wait=0.0)
    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise RetryableError("blip")
        return "ok"

    assert await flaky() == "ok"
    assert calls["n"] == 2


async def test_circuit_breaker_opens_and_fails_fast() -> None:
    breaker = CircuitBreaker("test", failure_threshold=2, recovery_timeout_s=60.0)

    async def boom() -> None:
        raise RetryableError("down")

    for _ in range(2):
        with pytest.raises(RetryableError):
            await breaker.call(boom)

    assert breaker.state is CircuitState.OPEN
    # Now it fails fast without invoking the callable.
    with pytest.raises(CircuitBreakerOpen):
        await breaker.call(boom)


async def test_circuit_breaker_recovers_after_cooldown() -> None:
    breaker = CircuitBreaker(
        "test", failure_threshold=1, recovery_timeout_s=0.0, success_threshold=1
    )

    async def boom() -> None:
        raise RetryableError("down")

    async def ok() -> str:
        return "ok"

    with pytest.raises(RetryableError):
        await breaker.call(boom)
    assert breaker.state is CircuitState.OPEN

    # recovery_timeout_s=0 -> immediately allowed to probe (HALF_OPEN), success closes it.
    assert await breaker.call(ok) == "ok"
    assert breaker.state is CircuitState.CLOSED
