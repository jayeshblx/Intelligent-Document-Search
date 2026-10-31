"""How the orchestrator reaches the document-search agent.

Two interchangeable implementations behind one protocol:

* ``AgentCoreSearchClient`` -- production. Calls the document-search agent's own
  AgentCore Runtime with ``InvokeAgentRuntime``, so the two agents scale, version
  and fail independently.
* ``InProcessSearchClient`` -- local dev, tests, and single-container demos. Calls
  the agent object directly.

Both take an ``AgentRequest`` and return an ``AgentResponse``, so switching between
them is a config change (``AGENTIC_SEARCH_TRANSPORT``) and nothing else.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import TYPE_CHECKING, Any, Protocol

import structlog

from rag_agentic.config import AgenticSettings
from rag_agentic.models import AgentRequest, AgentResponse
from rag_core.aws.session import AwsSessionFactory
from rag_core.logging import get_logger
from rag_core.resilience import NonRetryableError, RetryableError

if TYPE_CHECKING:  # import-time only: rag_agentic.agents imports this module back
    from rag_agentic.agents.search_agent import DocumentSearchAgent

_log = get_logger("rag_agentic.search_client")

# AgentCore rejects a runtimeSessionId shorter than this.
_MIN_SESSION_ID_LEN = 33


def ensure_session_id(session_id: str | None) -> str:
    """A runtimeSessionId AgentCore will accept (>= 33 chars), stable if supplied."""
    candidate = (session_id or "").strip()
    if len(candidate) >= _MIN_SESSION_ID_LEN:
        return candidate
    # Pad rather than replace, so a caller's short id still maps 1:1 to a session.
    filler = uuid.uuid4().hex
    return f"{candidate}-{filler}" if candidate else filler + uuid.uuid4().hex[:1]


class DelegateError(RuntimeError):
    """The document-search runtime answered with an error instead of an answer.

    Carries that runtime's HTTP status so the orchestrator can return the same one:
    a tenant with no snapshot is a 503 whichever runtime noticed it.
    """

    def __init__(self, message: str, *, status: int = 502) -> None:
        super().__init__(message)
        self.status = status if 400 <= status <= 599 else 502


class SearchAgentClient(Protocol):
    async def invoke(self, request: AgentRequest) -> AgentResponse: ...

    async def aclose(self) -> None: ...


class InProcessSearchClient:
    """Calls the document-search agent in this process."""

    def __init__(self, agent: DocumentSearchAgent) -> None:
        self._agent = agent

    async def invoke(self, request: AgentRequest) -> AgentResponse:
        return await self._agent.handle(request)

    async def aclose(self) -> None:
        return None


class AgentCoreSearchClient:
    """Calls the document-search agent's AgentCore Runtime."""

    def __init__(
        self,
        settings: AgenticSettings,
        *,
        factory: AwsSessionFactory | None = None,
    ) -> None:
        if not settings.search_agent_arn:
            raise ValueError(
                "AGENTIC_SEARCH_AGENT_ARN must be set when AGENTIC_SEARCH_TRANSPORT=agentcore"
            )
        self._settings = settings
        self._factory = factory or _default_factory(settings)
        self._client: Any | None = None
        self._cm: Any | None = None

    def _client_config(self) -> Any:
        """Timeouts for one delegation call.

        The default 60 s read timeout is shorter than a cold document-search runtime
        takes to answer (its first request downloads and restores the tenant snapshot),
        so the orchestrator would abandon a call that was going to succeed. Retries are
        off: re-sending a request that may already be running costs a second full
        answer, and the caller gets a clean 502 instead.
        """
        from botocore.config import Config

        return Config(
            connect_timeout=10,
            read_timeout=int(self._settings.search_invoke_timeout_s),
            retries={"max_attempts": 1, "mode": "standard"},
        )

    async def _ensure(self) -> Any:
        if self._client is None:
            try:
                self._cm = self._factory.client("bedrock-agentcore", config=self._client_config())
            except Exception as exc:  # botocore too old to know the service
                raise NonRetryableError(
                    "boto3/botocore does not know the 'bedrock-agentcore' service - "
                    f"upgrade the AWS SDK in this image: {exc}"
                ) from exc
            self._client = await self._cm.__aenter__()
        return self._client

    async def invoke(self, request: AgentRequest) -> AgentResponse:
        client = await self._ensure()
        session_id = ensure_session_id(request.session_id)
        # Forward the envelope untouched: the search agent resolves tenancy from
        # base_url itself, so the two runtimes cannot drift on what "tenant" means.
        payload = request.model_dump(mode="json")
        payload["session_id"] = session_id

        # Carry this request's id across the runtime boundary so the search
        # agent's tool calls appear under the SAME id in CloudWatch as the
        # routing decision that caused them.
        trace_id = str(structlog.contextvars.get_contextvars().get("request_id") or "")
        started = time.perf_counter()
        _log.info("delegating_to_search_agent", trace_id=trace_id, session_id=session_id)

        try:
            kwargs: dict[str, Any] = {
                "agentRuntimeArn": self._settings.search_agent_arn,
                "runtimeSessionId": session_id,
                "qualifier": self._settings.search_agent_qualifier,
                "payload": json.dumps(payload).encode("utf-8"),
            }
            if trace_id:
                kwargs["traceId"] = trace_id
            response = await client.invoke_agent_runtime(**kwargs)
        except Exception as exc:
            # AgentCore raises rather than returning a body when the runtime answers
            # with a non-200 ("Received error (503) from runtime"). Without this the
            # failure would reach the orchestrator's model as a recoverable tool
            # error, and the caller would get a polite apology with HTTP 200.
            _log.warning(
                "delegation_failed",
                trace_id=trace_id,
                took_ms=round((time.perf_counter() - started) * 1000, 1),
                error=str(exc)[:300],
            )
            raise DelegateError(
                f"document-search runtime failed: {exc}", status=_status_in_message(str(exc))
            ) from exc
        _log.info(
            "delegation_returned",
            trace_id=trace_id,
            took_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        body = await _read(response.get("response"))
        if not body:
            raise RetryableError("document-search runtime returned an empty response")

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RetryableError(
                f"document-search runtime returned non-JSON: {body[:200]!r}"
            ) from exc
        if isinstance(data, dict) and data.get("error") and "answer" not in data:
            raise DelegateError(
                f"document-search agent failed: {data['error']}",
                status=_status_of(data, response),
            )
        return AgentResponse.model_validate(data)

    async def aclose(self) -> None:
        if self._cm is not None:
            await self._cm.__aexit__(None, None, None)
            self._cm = None
            self._client = None


def _status_in_message(message: str) -> int:
    """The runtime's status out of an AgentCore error string.

    It reports failures as ``Received error (503) from runtime``; the body (our own
    explanation) is not carried, only logged by that runtime. 502 when absent.
    """
    match = re.search(r"\((\d{3})\)", message)
    if match and 400 <= int(match.group(1)) <= 599:
        return int(match.group(1))
    return 502


def _status_of(data: dict[str, Any], response: dict[str, Any]) -> int:
    """The failing runtime's HTTP status: from its error body, else AgentCore's envelope."""
    for candidate in (data.get("status"), response.get("statusCode")):
        try:
            status = int(candidate)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if 400 <= status <= 599:
            return status
    return 502


async def _read(stream: Any) -> str:
    """Read an InvokeAgentRuntime response body (async or sync StreamingBody)."""
    if stream is None:
        return ""
    if isinstance(stream, bytes | bytearray):
        return bytes(stream).decode("utf-8", errors="replace")
    if isinstance(stream, str):
        return stream
    data = stream.read()
    if hasattr(data, "__await__"):
        data = await data
    if isinstance(data, bytes | bytearray):
        return bytes(data).decode("utf-8", errors="replace")
    return str(data)


def _default_factory(settings: AgenticSettings) -> AwsSessionFactory:
    from rag_core.config import get_settings

    return AwsSessionFactory(get_settings().aws, region_override=settings.region)
