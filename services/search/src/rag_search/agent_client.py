"""Call the deployed AgentCore orchestrator from this service.

The local UI normally answers from this service's own pipeline. That is useful,
but it is NOT what ExampleApp receives: different prompt, different routing, its
own credit pool. To demonstrate the deployed system, the question has to go to
the orchestrator -- through here, because ``InvokeAgentRuntime`` is SigV4-signed
and AWS credentials must never reach a browser.

Deliberately unmetered. No ``ReservationId`` and no ``AccessToken`` are sent, so
the agent's credit gate records ``credit_reservation_missing`` and never calls
Licensing. A demo therefore cannot write anything into ExampleApp's ledger, which
is the property that makes this safe to run beside a live integration.

Disabled unless ``SEARCH_AGENT_RUNTIME_ARN`` names a runtime, so a deployment
that does not want it is unaffected by its existence.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from rag_core.logging import get_logger

_log = get_logger("rag_search.agent_client")

# AgentCore requires a session id of at least 33 characters.
_MIN_SESSION_CHARS = 33


class AgentUnavailableError(RuntimeError):
    """The agent could not be reached, or refused the request.

    Raised rather than returning a fallback answer: a fabricated answer is
    indistinguishable from a grounded one to whoever is watching the demo.
    """


def new_agent_session() -> str:
    return f"ui-{uuid.uuid4().hex}"


def pad_session(session_id: str | None) -> str:
    """A session id AgentCore will accept, preserving the caller's where possible."""
    value = (session_id or "").strip() or new_agent_session()
    if len(value) < _MIN_SESSION_CHARS:
        value = f"{value}-{uuid.uuid4().hex}"
    return value


@dataclass(frozen=True)
class AgentAnswer:
    """One answer from the deployed agent, in this service's own terms."""

    answer: str
    session_id: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    documents: list[dict[str, Any]] = field(default_factory=list)
    documents_total: int = 0
    object_codes: str = ""
    route: str = ""
    tenant: str = ""
    environment: str = ""
    business_object: str = ""
    model_id: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    took_ms: float = 0.0


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def parse_agent_response(payload: dict[str, Any], *, session_id: str) -> AgentAnswer:
    """Map the agent's envelope onto what this service's UI already renders.

    Tolerant by design: the agent is deployed separately and may add fields or
    change spellings between releases, and a demo must not break because a key
    it never used appeared.
    """
    usage = payload.get("usage") or {}
    return AgentAnswer(
        answer=str(payload.get("answer") or ""),
        session_id=str(payload.get("session_id") or session_id),
        citations=list(payload.get("citations") or []),
        documents=list(payload.get("documents") or []),
        documents_total=_int(payload.get("total_matching") or payload.get("documents_total")),
        object_codes=str(payload.get("object_codes") or ""),
        route=str(payload.get("route") or ""),
        tenant=str(payload.get("tenant") or ""),
        environment=str(payload.get("environment") or ""),
        business_object=str(payload.get("business_object") or ""),
        model_id=str(payload.get("model_id") or usage.get("model") or ""),
        input_tokens=_int(usage.get("input_tokens")),
        output_tokens=_int(usage.get("output_tokens")),
        embedding_tokens=_int(usage.get("embedding_tokens")),
        took_ms=float(payload.get("duration_ms") or 0.0),
    )


class AgentRuntimeClient:
    """Invokes one deployed AgentCore runtime."""

    def __init__(
        self,
        runtime_arn: str,
        *,
        region: str,
        client: Any = None,
        read_timeout_s: float = 600.0,
        ca_bundle: str | None = None,
    ) -> None:
        self._arn = runtime_arn
        self._region = region
        self._read_timeout_s = read_timeout_s
        self._ca_bundle = ca_bundle
        self._client = client

    @property
    def runtime_arn(self) -> str:
        return self._arn

    def _boto(self) -> Any:
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client(
                "bedrock-agentcore",
                region_name=self._region,
                verify=self._ca_bundle or True,
                config=Config(
                    read_timeout=self._read_timeout_s,
                    connect_timeout=15,
                    # One attempt: the agent is not idempotent per call, and a
                    # silent retry would double the Bedrock spend on a slow run.
                    retries={"max_attempts": 1},
                ),
            )
        return self._client

    async def ask(
        self,
        question: str,
        *,
        base_url: str,
        session_id: str | None = None,
        business_object: str | None = None,
        object_codes: str | None = None,
        operation: str | None = None,
        user_code: str = "local-ui",
    ) -> AgentAnswer:
        """Ask the deployed agent, as an unmetered caller.

        No reservation and no access token: see the module docstring.
        """
        session = pad_session(session_id)
        body: dict[str, Any] = {
            "baseUrl": base_url,
            "query": question,
            "UserCode": user_code,
        }
        if business_object:
            body["BusinessObject"] = business_object
        if object_codes:
            body["objectCodes"] = object_codes
        if operation:
            body["Operation"] = operation

        try:
            raw = await asyncio.to_thread(self._invoke, session, body)
        except AgentUnavailableError:
            raise
        except Exception as exc:
            _log.warning(
                "agent_invoke_failed",
                arn=self._arn, base_url=base_url, error=str(exc)[:300],
            )
            raise AgentUnavailableError(str(exc)) from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AgentUnavailableError("the agent returned a non-JSON response") from exc
        if not isinstance(payload, dict):
            raise AgentUnavailableError("the agent returned an unexpected response shape")

        answer = parse_agent_response(payload, session_id=session)
        _log.info(
            "agent_answered",
            base_url=base_url, route=answer.route, tenant=answer.tenant,
            citations=len(answer.citations), took_ms=round(answer.took_ms, 1),
        )
        return answer

    def _invoke(self, session: str, body: dict[str, Any]) -> bytes:
        response = self._boto().invoke_agent_runtime(
            agentRuntimeArn=self._arn,
            runtimeSessionId=session,
            qualifier="DEFAULT",
            contentType="application/json",
            accept="application/json",
            payload=json.dumps(body).encode(),
        )
        stream = response.get("response")
        return stream.read() if hasattr(stream, "read") else bytes(stream or b"")
