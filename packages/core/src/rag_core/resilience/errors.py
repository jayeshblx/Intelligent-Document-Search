"""Error taxonomy that drives retry decisions.

The single most important property of a resilient pipeline is *correctly
classifying* failures: retry transient faults, fail fast on permanent ones, and
back off harder on rate limits. We model that as an exception hierarchy so the
retry policy can decide purely on type.
"""

from __future__ import annotations


class PlatformError(Exception):
    """Root of all platform-raised errors."""


class RetryableError(PlatformError):
    """A failure that may succeed if retried (network blip, 5xx, timeout)."""


class TransientError(RetryableError):
    """Alias for a short-lived, retryable condition (kept for call-site clarity)."""


class RateLimitedError(RetryableError):
    """Throttling (HTTP 429 / Bedrock ThrottlingException). Retry with longer
    backoff; carries an optional server-suggested delay."""

    def __init__(self, message: str, *, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class NonRetryableError(PlatformError):
    """A permanent failure (validation, 4xx auth, malformed document). Retrying
    wastes capacity and delays DLQ routing — fail fast."""
