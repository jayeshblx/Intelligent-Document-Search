"""Tool-capable Bedrock Converse binding.

``rag_core.aws.BedrockClient.generate`` covers plain completions; an agent also
needs ``toolConfig`` and the ``toolUse`` blocks that come back with it, so this
module adds that one capability while keeping the same resilience posture as the
core client (adaptive botocore retries + explicit retry policy + circuit breaker
+ the platform's retryable/non-retryable taxonomy).

Everything the agents call is behind ``ConverseModel``, so the whole loop runs
offline against ``ScriptedConverseModel`` in tests -- no AWS, no credentials.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol

from botocore.exceptions import BotoCoreError, ClientError

from rag_agentic.models import Usage
from rag_core.aws.session import AwsSessionFactory
from rag_core.config import BedrockSettings
from rag_core.logging import get_logger
from rag_core.resilience import (
    CircuitBreaker,
    NonRetryableError,
    RateLimitedError,
    RetryableError,
    async_retry,
)

_log = get_logger("rag_agentic.converse")

_THROTTLE_CODES = {"ThrottlingException", "TooManyRequestsException"}
_TRANSIENT_CODES = {"ModelTimeoutException", "ServiceUnavailableException", "InternalServerException"}
_PERMANENT_CODES = {"ValidationException", "AccessDeniedException", "ResourceNotFoundException"}


def _classify(exc: ClientError) -> Exception:
    """Map a botocore error onto the platform's retry taxonomy."""
    code = exc.response.get("Error", {}).get("Code", "")
    if code in _THROTTLE_CODES:
        return RateLimitedError(f"Bedrock throttled: {code}")
    if code in _TRANSIENT_CODES:
        return RetryableError(f"Bedrock transient error: {code}")
    if code in _PERMANENT_CODES:
        return NonRetryableError(f"Bedrock rejected the request: {code} - {exc}")
    return RetryableError(f"Bedrock error: {code or exc}")


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One tool offered to the model, in Converse's ``toolSpec`` shape."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def to_converse(self) -> dict[str, Any]:
        return {
            "toolSpec": {
                "name": self.name,
                "description": self.description,
                "inputSchema": {"json": self.input_schema},
            }
        }


@dataclass(frozen=True, slots=True)
class ToolUse:
    """A tool call the model asked for."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass(slots=True)
class ModelTurn:
    """One assistant turn: its text, any tool calls, and what it cost."""

    text: str = ""
    tool_uses: list[ToolUse] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)
    # The raw assistant message, appended verbatim to the transcript. Converse
    # requires the exact block list back -- reconstructing it drops toolUse ids.
    message: dict[str, Any] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return self.stop_reason == "tool_use" and bool(self.tool_uses)


class ConverseModel(Protocol):
    """Minimal model interface the agent loop depends on."""

    async def converse(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        cache_system: bool = False,
        cache_messages: bool = False,
    ) -> ModelTurn: ...


# Below roughly this much transcript, Bedrock ignores a cache point because the
# prefix is under the model's minimum cacheable size. Measured in characters because
# the exact token count is not knowable locally; ~8k chars is comfortably past a
# 1k-token floor, so a marked request is always worth marking.
_CACHE_FLOOR_CHARS = 8000


def _with_cache_point(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A copy of the transcript with one cache point at its end.

    The agent loop re-sends a transcript that grows with every tool result, so
    without this each round pays full price to re-read everything gathered so far.
    Marking the END of the current transcript makes the NEXT round read all of it
    at the cache rate -- the point rolls forward each turn, extending the cached
    prefix. The caller's list is never mutated: the marker belongs to one request.
    """
    if not messages:
        return messages
    last = messages[-1]
    content = list(last.get("content") or [])
    if not content or any("cachePoint" in block for block in content):
        return messages
    content.append({"cachePoint": {"type": "default"}})
    return [*messages[:-1], {**last, "content": content}]


def _parse(response: dict[str, Any], model_id: str) -> ModelTurn:
    message = response.get("output", {}).get("message", {}) or {}
    blocks = message.get("content", []) or []

    texts: list[str] = []
    tool_uses: list[ToolUse] = []
    for block in blocks:
        if "text" in block:
            texts.append(str(block["text"]))
        elif "toolUse" in block:
            use = block["toolUse"]
            raw = use.get("input", {})
            # Converse normally hands back a decoded object, but a model can emit
            # a JSON *string* for a single-argument call -- accept both.
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    raw = {}
            tool_uses.append(
                ToolUse(
                    id=str(use.get("toolUseId", "")),
                    name=str(use.get("name", "")),
                    input=dict(raw) if isinstance(raw, dict) else {},
                )
            )

    raw_usage = response.get("usage", {}) or {}
    turn_usage = Usage(
        input_tokens=int(raw_usage.get("inputTokens", 0) or 0),
        output_tokens=int(raw_usage.get("outputTokens", 0) or 0),
        cache_read_tokens=int(raw_usage.get("cacheReadInputTokens", 0) or 0),
        cache_write_tokens=int(raw_usage.get("cacheWriteInputTokens", 0) or 0),
        model=model_id,
        llm_calls=1,
    )
    _record_spent(turn_usage)
    return ModelTurn(
        text="\n".join(t for t in texts if t).strip(),
        tool_uses=tool_uses,
        stop_reason=str(response.get("stopReason", "end_turn")),
        usage=turn_usage,
        message=message,
    )



# Model tokens spent while serving ONE request, accumulated as each Converse call
# returns. A contextvar for two reasons: concurrent requests on the same runtime
# cannot bleed into each other's count, and the total survives an exception
# unwinding the stack.
#
# That second reason is the point. A run that fails partway has still spent every
# token it used up to the failure, and Bedrock bills them. Reporting zero would
# quietly move that cost onto whoever runs the agent instead of the tenant who
# incurred it.
# Default is None rather than a Usage(): a mutable default is shared by every
# context that never sets one.
_spent: ContextVar[Usage | None] = ContextVar("converse_spent", default=None)


def reset_spent() -> None:
    """Start counting afresh for a new request."""
    _spent.set(Usage())


def spent_so_far() -> Usage:
    """Model tokens spent since the last reset, however the request ended."""
    return _spent.get() or Usage()


def _record_spent(usage: Usage) -> None:
    _spent.set(spent_so_far().add(usage))


class BedrockConverseModel:
    """Persistent async Bedrock Runtime binding with tool support."""

    def __init__(
        self,
        settings: BedrockSettings,
        *,
        model_id: str = "",
        factory: AwsSessionFactory | None = None,
        guardrail_id: str = "",
        guardrail_version: str = "DRAFT",
        guardrail_trace: str = "enabled",
    ) -> None:
        self._settings = settings
        # Applied to every call when configured -- input AND output, so a prompt
        # attack hidden in a retrieved document is assessed too.
        self._guardrail = (
            {
                "guardrailIdentifier": guardrail_id,
                "guardrailVersion": guardrail_version,
                "trace": guardrail_trace,
            }
            if guardrail_id
            else None
        )
        # Blank falls back to the account-verified inference profile in BedrockSettings.
        self._model_id = model_id or settings.llm_model
        self._factory = factory or AwsSessionFactory(
            _aws_settings(settings), region_override=settings.region
        )
        self._client: Any | None = None
        self._cm: Any | None = None
        self._breaker = CircuitBreaker("bedrock-converse", failure_threshold=8, recovery_timeout_s=20.0)

    @property
    def model_id(self) -> str:
        return self._model_id

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

    async def __aenter__(self) -> BedrockConverseModel:
        await self._ensure()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    @async_retry(max_attempts=5, max_wait=30.0)
    async def converse(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        cache_system: bool = False,
        cache_messages: bool = False,
    ) -> ModelTurn:
        client = await self._ensure()
        outbound = messages
        if cache_messages and sum(len(str(m)) for m in messages) > _CACHE_FLOOR_CHARS:
            outbound = _with_cache_point(messages)

        async def _call() -> ModelTurn:
            kwargs: dict[str, Any] = {
                "modelId": self._model_id,
                "messages": outbound,
                "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
            }
            if system:
                blocks: list[dict[str, Any]] = [{"text": system}]
                if cache_system:
                    # The system prompt is the largest stable prefix in an agent
                    # turn; caching it makes every tool round-trip re-read it at
                    # the discounted rate instead of paying full price each time.
                    blocks.append({"cachePoint": {"type": "default"}})
                kwargs["system"] = blocks
            if tools:
                kwargs["toolConfig"] = {"tools": [t.to_converse() for t in tools]}
            if self._guardrail is not None:
                kwargs["guardrailConfig"] = self._guardrail

            try:
                response = await client.converse(**kwargs)
            except ClientError as exc:
                raise _classify(exc) from exc
            except BotoCoreError as exc:
                raise RetryableError(f"Bedrock transport error: {exc}") from exc
            return _parse(response, self._model_id)

        return await self._breaker.call(_call)


def _aws_settings(settings: BedrockSettings) -> Any:
    """AWSSettings for Bedrock: platform credentials, Bedrock's own region.

    ``endpoint_url`` is deliberately dropped -- Bedrock is always real AWS, never
    LocalStacked, so inheriting a dev endpoint would silently break the agent.
    """
    from rag_core.config import AWSSettings, get_settings

    base = get_settings().aws
    return AWSSettings(
        region=settings.region or base.region,
        max_retries=base.max_retries,
        access_key_id=base.access_key_id,
        secret_access_key=base.secret_access_key,
        ca_bundle=base.ca_bundle,
        verify_ssl=base.verify_ssl,
    )


class ScriptedConverseModel:
    """Deterministic offline model: replays a fixed list of turns.

    Each ``converse`` call pops the next scripted turn, so a test can drive the
    exact tool sequence it wants to assert on. Records every request for
    inspection (system prompt, transcript, tools offered).
    """

    def __init__(self, turns: list[ModelTurn], *, model_id: str = "scripted") -> None:
        self._turns = list(turns)
        self._model_id = model_id
        self.calls: list[dict[str, Any]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    async def converse(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        cache_system: bool = False,
        cache_messages: bool = False,
    ) -> ModelTurn:
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "system": system,
                "tools": [t.name for t in tools or []],
            }
        )
        if not self._turns:
            return ModelTurn(text="", stop_reason="end_turn", message={"role": "assistant", "content": []})
        turn = self._turns.pop(0)
        if not turn.message:
            content: list[dict[str, Any]] = []
            if turn.text:
                content.append({"text": turn.text})
            for use in turn.tool_uses:
                content.append(
                    {"toolUse": {"toolUseId": use.id, "name": use.name, "input": use.input}}
                )
            turn.message = {"role": "assistant", "content": content}
        turn.usage.model = turn.usage.model or self._model_id
        return turn

    async def aclose(self) -> None:
        return None


def text_turn(text: str) -> ModelTurn:
    """Scripted helper: a final answer with no tool calls."""
    return ModelTurn(text=text, stop_reason="end_turn")


def tool_turn(name: str, tool_input: dict[str, Any], *, use_id: str = "t1", text: str = "") -> ModelTurn:
    """Scripted helper: a turn that asks for one tool call."""
    return ModelTurn(
        text=text,
        tool_uses=[ToolUse(id=use_id, name=name, input=tool_input)],
        stop_reason="tool_use",
    )
