"""Async Bedrock client: Titan embeddings + Claude generation.

Bedrock is a metered, rate-limited dependency, so this client layers three
protections: per-call retry with backoff (transient 5xx/timeout), a circuit
breaker (stop hammering a degraded endpoint), and explicit throttling handling
(ThrottlingException -> RateLimitedError -> longer backoff). Embeddings are the
hot path during a 1M-doc backfill, so ``embed_batch`` bounds concurrency to stay
under account TPS while keeping workers busy.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

from rag_core.aws.session import AwsSessionFactory
from rag_core.config import BedrockSettings
from rag_core.logging import get_logger
from rag_core.models.base import DomainModel
from rag_core.resilience import (
    CircuitBreaker,
    NonRetryableError,
    RateLimitedError,
    RetryableError,
    async_retry,
)

_log = get_logger("rag_core.aws.bedrock")

_THROTTLE_CODES = {"ThrottlingException", "TooManyRequestsException"}
_TRANSIENT_CODES = {"ModelTimeoutException", "ServiceUnavailableException", "InternalServerException"}
_PERMANENT_CODES = {"ValidationException", "AccessDeniedException", "ResourceNotFoundException"}


class EmbeddingResult(DomainModel):
    """One embedding plus its token accounting (for cost/metrics)."""

    vector: list[float]
    dim: int
    input_tokens: int = 0


class GenerationResult(DomainModel):
    """An LLM completion plus token usage, for cost accounting.

    ``input_tokens``/``output_tokens`` are the ordinary (uncached) counts. When prompt
    caching is used the provider bills the cached prefix separately, so those tokens are
    reported apart: ``cache_write_tokens`` (first call, writing the cache) and
    ``cache_read_tokens`` (later calls, reading it at a steep discount)."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    model: str = ""


def _classify(exc: ClientError) -> Exception:
    code = exc.response.get("Error", {}).get("Code", "")
    if code in _THROTTLE_CODES:
        return RateLimitedError(f"Bedrock throttled: {code}")
    if code in _TRANSIENT_CODES:
        return RetryableError(f"Bedrock transient error: {code}")
    if code in _PERMANENT_CODES:
        return NonRetryableError(f"Bedrock permanent error {code}: {exc}")
    return RetryableError(f"Bedrock error {code}: {exc}")


class BedrockClient:
    """Persistent async Bedrock Runtime client with resilience built in."""

    def __init__(
        self,
        settings: BedrockSettings,
        *,
        factory: AwsSessionFactory | None = None,
        max_embed_concurrency: int = 16,
    ) -> None:
        self._settings = settings
        # Bedrock may live in a different region than the rest of the stack.
        self._factory = factory or AwsSessionFactory(
            _bedrock_aws_settings(settings), region_override=settings.region
        )
        self._client: Any | None = None
        self._cm: Any | None = None
        self._breaker = CircuitBreaker("bedrock", failure_threshold=8, recovery_timeout_s=20.0)
        self._embed_sema = asyncio.Semaphore(max_embed_concurrency)

    async def __aenter__(self) -> BedrockClient:
        await self._ensure()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _ensure(self) -> Any:
        if self._client is None:
            self._cm = self._factory.client("bedrock-runtime")
            self._client = await self._cm.__aenter__()
        return self._client

    async def aclose(self) -> None:
        if self._cm is not None:
            await self._cm.__aexit__(None, None, None)
            self._cm = None
            self._client = None

    # ---- Embeddings -------------------------------------------------------
    @async_retry(max_attempts=6, max_wait=45.0)
    async def embed_text(self, text: str, *, dimensions: int = 1024) -> EmbeddingResult:
        """Embed a single string with Titan v2. Normalised vectors (cosine-ready)."""
        client = await self._ensure()
        body = json.dumps({"inputText": text, "dimensions": dimensions, "normalize": True})

        async def _call() -> EmbeddingResult:
            try:
                resp = await client.invoke_model(
                    modelId=self._settings.titan_embed_model,
                    body=body,
                    accept="application/json",
                    contentType="application/json",
                )
                payload = json.loads(await resp["body"].read())
            except ClientError as exc:
                raise _classify(exc) from exc
            except BotoCoreError as exc:
                raise RetryableError(f"Bedrock transport error: {exc}") from exc
            vec = payload["embedding"]
            return EmbeddingResult(
                vector=vec,
                dim=len(vec),
                input_tokens=int(payload.get("inputTextTokenCount", 0)),
            )

        return await self._breaker.call(_call)

    async def embed_batch(
        self, texts: list[str], *, dimensions: int = 1024
    ) -> list[EmbeddingResult]:
        """Embed many strings with bounded concurrency.

        Titan has no native batch endpoint; we fan out under a semaphore so a
        backfill saturates throughput without breaching account TPS limits.
        Order is preserved to align results with input chunks.
        """

        async def _one(t: str) -> EmbeddingResult:
            async with self._embed_sema:
                return await self.embed_text(t, dimensions=dimensions)

        return await asyncio.gather(*(_one(t) for t in texts))

    # ---- Generation -------------------------------------------------------
    @async_retry(max_attempts=5, max_wait=30.0)
    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        cache_system: bool = False,
    ) -> GenerationResult:
        """Generate a completion via the Bedrock Converse API (Claude).

        ``messages`` follows the Converse shape:
        ``[{"role": "user", "content": [{"text": "..."}]}]``. Returns the text plus token
        usage (from the Converse ``usage`` block) for cost accounting.

        ``cache_system`` appends a Converse ``cachePoint`` after the system prompt, so the
        (large, stable) system prefix is cached and re-read at a discount on later calls.
        The cached-token counts come back in the ``usage`` block and are reported
        separately on the result.
        """
        client = await self._ensure()

        async def _call() -> GenerationResult:
            kwargs: dict[str, Any] = {
                "modelId": self._settings.llm_model,
                "messages": messages,
                "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
            }
            if system:
                sys_blocks: list[dict[str, Any]] = [{"text": system}]
                if cache_system:
                    sys_blocks.append({"cachePoint": {"type": "default"}})
                kwargs["system"] = sys_blocks
            try:
                resp = await client.converse(**kwargs)
            except ClientError as exc:
                raise _classify(exc) from exc
            except BotoCoreError as exc:
                raise RetryableError(f"Bedrock transport error: {exc}") from exc
            text = resp["output"]["message"]["content"][0]["text"]
            usage = resp.get("usage", {})
            return GenerationResult(
                text=text,
                input_tokens=int(usage.get("inputTokens", 0)),
                output_tokens=int(usage.get("outputTokens", 0)),
                # Present only when caching is active; absent keys default to 0.
                cache_read_tokens=int(usage.get("cacheReadInputTokens", 0) or 0),
                cache_write_tokens=int(usage.get("cacheWriteInputTokens", 0) or 0),
                model=self._settings.llm_model,
            )

        return await self._breaker.call(_call)


def _bedrock_aws_settings(_: BedrockSettings) -> Any:
    """Build an AWSSettings for Bedrock.

    Carries the same credentials as the rest of the platform (static keys when
    provided, otherwise the default IAM/credential chain), but never the
    ``endpoint_url`` — Bedrock is real AWS, never LocalStacked.
    """
    from rag_core.config import AWSSettings, get_settings

    base = get_settings().aws
    return AWSSettings(
        region=base.region,
        max_retries=base.max_retries,
        access_key_id=base.access_key_id,
        secret_access_key=base.secret_access_key,
        ca_bundle=base.ca_bundle,
        verify_ssl=base.verify_ssl,
        # Intentionally omit endpoint_url: Bedrock isn't LocalStacked.
    )
