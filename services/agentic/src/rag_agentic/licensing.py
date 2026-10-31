r"""ExampleApp Licensing: reserve credits before a run, settle them after.

Licensing owns every credit decision and all the arithmetic. This module only
drives the lifecycle it publishes:

    CheckAndReserve -> run the agent -> Settle(Success | Fail)
                                     \-> Release (nothing ran)

Three rules shape the implementation, each a judgement call worth stating:

* **A reservation that cannot be evaluated blocks the run.** A gate that fails
  open is not a gate -- an unreachable Licensing service would make credits
  optional for anyone able to make it unreachable. ``licensing_fail_open`` can
  invert that for an operator who would rather serve answers than meter them.
* **A failed Settle never loses a good answer.** The run already happened and the
  caller is owed its result. An unsettled reservation reconciles itself (it
  expires), whereas a discarded answer is work the tenant paid for and did not
  get. Settle failures are logged at error level for exactly that reconciliation.
* **Bearer tokens are never logged** -- not in a trace line, not in an error
  message, not in a repr.

Wire format is PascalCase, per the Licensing integration guide. The guide also
forbids sending ``EstimatedCredits``, ``LicenseCode``, ``ActualCredits``,
``TokensBurnt`` and ``CacheTokens``: Licensing sets those itself, so a caller
that sends them is asserting something it has no authority over.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from rag_agentic.models import AgentResponse, Usage
from rag_core.logging import get_logger

_log = get_logger("rag_agentic.licensing")

# The codes Licensing accepts. Sending anything else is a validation error there,
# so an unknown code is caught here rather than spending a round trip to learn it.
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "COMPLIANCE_AI_ASSISTANT": "Compliance AI Assistant",
    "OBJECT_COMPARE": "Object Compare",
    "DOCUMENT_SEARCH": "Document Search",
}

# Decline reasons from section 9. Listed so a decline can be reported precisely
# instead of as a generic refusal -- "no active license" and "out of credits" are
# different problems for whoever has to fix them.
DECLINE_MESSAGES: dict[str, str] = {
    "insufficient_credits": "This tenant has no AI credits left.",
    "no_active_license": "This tenant's AI licence is not active.",
    "no_credit_config": "Document Search is not configured for this tenant.",
    "invalid_license_unit": "This tenant's licence cannot be used for AI credits.",
    "validation_error": "Licensing rejected the credit request.",
    "declined": "Licensing declined this request.",
}


class LicensingDeclined(RuntimeError):
    """Licensing answered Allowed=false. The AI run must not start."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message or DECLINE_MESSAGES.get(reason, "Licensing declined this request."))
        self.reason = reason or "declined"


class LicensingUnavailable(RuntimeError):
    """Licensing could not be reached, or answered in a way we cannot act on."""


@dataclass(frozen=True)
class Reservation:
    id: str
    status: str = "Reserved"
    estimated_credits: int = 0
    is_internal_user: bool = False
    expires_at: str = ""

    @property
    def expired(self) -> bool:
        return self.status.lower() == "expired"


@dataclass(frozen=True)
class ReserveOutcome:
    allowed: bool
    reason: str = ""
    message: str = ""
    estimated_credits: int = 0
    is_internal_user: bool = False
    reservation: Reservation | None = None


# A credit record is read later to answer "what was this charge for?". The codes
# the answer actually cited are what answer it; the request's ACL scope does not,
# since it says only what the user was permitted to see and runs to thousands of
# codes on a large tenant. Bounded because a settle body is not a result set.
_MAX_OBJECT_CODES = 50

# An answer can quote several passages, so unlike a prompt it has no natural
# length. Truncated rather than dropped: the opening of an answer is enough to
# recognise what was returned, and a settle body is not a transcript store.
_MAX_RESPONSE_CHARS = 4000


def cited_object_codes(response: AgentResponse, limit: int = _MAX_OBJECT_CODES) -> str:
    """The object codes behind the answer, de-duplicated and in citation order."""
    seen: dict[str, None] = {}
    for citation in response.citations:
        code = str(citation.facets.get("object_code") or "").strip()
        if code and code not in seen:
            seen[code] = None
            if len(seen) >= limit:
                break
    return ",".join(seen)


@dataclass
class RunTelemetry:
    """What a finished run cost, in the shape Settle wants.

    Built from an ``AgentResponse`` so there is one definition of "what this run
    used" rather than a second tally that can drift from the one the caller sees.
    """

    model_id: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    embedding_tokens: int = 0
    total_tokens: int = 0
    execution_time_ms: int = 0
    batch_call_count: int = 0
    agent_id: str = ""
    # The documents the answer was built from, so a charge can be traced to a result.
    object_code: str = ""
    sub_agents: list[Usage] = field(default_factory=list)

    @classmethod
    def from_response(cls, response: AgentResponse) -> RunTelemetry:
        usage = response.usage
        return cls(
            model_id=response.model_id or usage.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            embedding_tokens=usage.embedding_tokens,
            total_tokens=usage.total_tokens,
            execution_time_ms=round(response.duration_ms or 0),
            batch_call_count=usage.llm_calls,
            agent_id=response.agent_id or "",
            object_code=cited_object_codes(response),
            sub_agents=list(response.sub_usage),
        )

    @classmethod
    def failed(
        cls, *, spent: Usage | None = None, embedding_tokens: int = 0,
        model_id: str = "", execution_time_ms: int = 0,
    ) -> RunTelemetry:
        """A run that produced no answer, reporting what it spent before breaking.

        The tokens a half-finished run consumed are billed by Bedrock exactly like
        a successful one's. Reporting zero would move that cost quietly onto
        whoever operates the agent instead of the tenant who incurred it, so the
        accumulated spend is reported even though there is no answer to show for it.
        """
        spent = spent or Usage()
        return cls(
            model_id=model_id or spent.model,
            input_tokens=spent.input_tokens,
            output_tokens=spent.output_tokens,
            cache_read_tokens=spent.cache_read_tokens,
            cache_write_tokens=spent.cache_write_tokens,
            embedding_tokens=embedding_tokens,
            total_tokens=spent.total_tokens,
            batch_call_count=spent.llm_calls,
            execution_time_ms=execution_time_ms,
        )


def settle_payload(
    *,
    tenant_code: str,
    reservation_id: str,
    transaction_id: str,
    result: str,
    feature_code: str,
    telemetry: RunTelemetry,
    region: str = "",
    trace_id: str = "",
    user_prompt: str = "",
    response_text: str = "",
    created_by: str = "",
) -> dict[str, Any]:
    """The Settle body. Pure, so the contract can be tested without a server."""
    payload: dict[str, Any] = {
        "TenantCode": tenant_code,
        "ReservationId": reservation_id,
        "TransactionId": transaction_id,
        "Result": result,
        # Required on Success; harmless and more useful than silence on Fail.
        "ModelId": telemetry.model_id,
        "InputTokens": telemetry.input_tokens,
        "OutputTokens": telemetry.output_tokens,
        "CacheReadTokens": telemetry.cache_read_tokens,
        "CacheWriteTokens": telemetry.cache_write_tokens,
        "EmbeddingTokens": telemetry.embedding_tokens,
        "TotalTokens": telemetry.total_tokens,
        "ExecutionTimeMs": telemetry.execution_time_ms,
        "Feature": FEATURE_DISPLAY_NAMES.get(feature_code.upper(), ""),
        "Host": "AWS",
        "SubAgents": [
            {
                # Which specialist ran. Every one of them uses the same model, so
                # without this a SubAgents row cannot be attributed.
                "AgentId": u.agent_id,
                "ModelId": u.model,
                "InputTokens": u.input_tokens,
                "OutputTokens": u.output_tokens,
                "CacheReadTokens": u.cache_read_tokens,
                "CacheWriteTokens": u.cache_write_tokens,
                "EmbeddingTokens": u.embedding_tokens,
                "TotalTokens": u.total_tokens,
            }
            for u in telemetry.sub_agents
        ],
    }
    if telemetry.batch_call_count:
        payload["BatchCallCount"] = telemetry.batch_call_count
    if telemetry.agent_id:
        payload["AgentId"] = telemetry.agent_id
    if telemetry.object_code:
        payload["ObjectCode"] = telemetry.object_code
    if region:
        payload["Region"] = region
    if trace_id:
        payload["TraceId"] = trace_id
    # The guide flags UserPrompt as possibly carrying PII, so it travels only when
    # an operator has asked for it.
    if user_prompt:
        payload["UserPrompt"] = user_prompt
    # What the caller was actually told, which is the other half of reading a
    # record back. Travels under the same operator opt-in as the prompt.
    if response_text:
        payload["Response"] = response_text[:_MAX_RESPONSE_CHARS]
    if created_by:
        payload["CreatedBy"] = created_by
    return payload


# ---- authentication ---------------------------------------------------------------


class TokenProvider(Protocol):
    """Supplies the ExampleApp IAM bearer token for one request."""

    async def token(self, forwarded: str | None) -> str: ...


class ForwardedToken:
    """Use the token ExampleApp sent with the request.

    Preferred: ExampleApp already holds the signed-in user's IAM token, so forwarding
    it keeps the Licensing record attributed to the real user, and keeps per-tenant
    OAuth client secrets out of this runtime's environment entirely. Tokens live
    about five minutes, which comfortably covers one run.
    """

    async def token(self, forwarded: str | None) -> str:
        token = strip_bearer(forwarded)
        if not token:
            raise LicensingUnavailable(
                "no access token on the request; ExampleApp must send AccessToken, "
                "or client credentials must be configured"
            )
        return token


class ClientCredentialsToken:
    """Fetch a token from the tenant ExampleApp IAM with a client credential.

    For callers that cannot forward a user token. The token is cached until shortly
    before expiry: a five-minute lifetime against a 30-90s run would otherwise mean
    a token fetch on nearly every request.
    """

    # Refresh this far ahead of expiry so a token cannot die mid-run.
    _EARLY_REFRESH_S = 60.0

    def __init__(
        self,
        *,
        token_endpoint: str,
        client_id: str,
        client_secret: str,
        scope: str = "",
        timeout_s: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._endpoint = token_endpoint
        self._client_id = client_id
        self._secret = client_secret
        self._scope = scope
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._cached = ""
        self._expires_at = 0.0

    async def token(self, forwarded: str | None) -> str:
        # A forwarded token still wins: it identifies the actual user.
        if token := strip_bearer(forwarded):
            return token
        if self._cached and time.monotonic() < self._expires_at:
            return self._cached

        form = {
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._secret,
        }
        if self._scope:
            form["scope"] = self._scope
        try:
            response = await self._client.post(self._endpoint, data=form)
            response.raise_for_status()
            body = response.json()
        except Exception as exc:  # network, 4xx/5xx, or a non-JSON body
            raise LicensingUnavailable(
                f"could not obtain an IAM token: {type(exc).__name__}"
            ) from exc

        token = str(body.get("access_token") or "")
        if not token:
            raise LicensingUnavailable("IAM token response carried no access_token")
        lifetime = float(body.get("expires_in") or 300)
        self._cached = token
        self._expires_at = time.monotonic() + max(0.0, lifetime - self._EARLY_REFRESH_S)
        return token

    async def aclose(self) -> None:
        await self._client.aclose()


# ---- the client -------------------------------------------------------------------


class LicensingClient(Protocol):
    async def check_and_reserve(
        self, *, tenant_code: str, feature_code: str, idempotency_key: str,
        user_code: str = "", access_token: str | None = None,
    ) -> ReserveOutcome: ...

    async def settle(
        self, *, tenant_code: str, reservation_id: str, transaction_id: str, result: str,
        feature_code: str, telemetry: RunTelemetry, trace_id: str = "",
        user_prompt: str = "", response_text: str = "", created_by: str = "",
        access_token: str | None = None,
    ) -> dict[str, Any]: ...

    async def release(
        self, *, tenant_code: str, reservation_id: str, reason: str = "",
        access_token: str | None = None,
    ) -> dict[str, Any]: ...

    async def aclose(self) -> None: ...


class HttpLicensingClient:
    """Talks to the TenantAdmin Credits API."""

    def __init__(
        self,
        *,
        base_url: str,
        tokens: TokenProvider,
        region: str = "",
        timeout_s: float = 10.0,
        send_user_prompt: bool = False,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._tokens = tokens
        self._region = region
        self._send_user_prompt = send_user_prompt
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def _headers(self, access_token: str | None) -> dict[str, str]:
        token = await self._tokens.token(access_token)
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def check_and_reserve(
        self, *, tenant_code: str, feature_code: str, idempotency_key: str,
        user_code: str = "", access_token: str | None = None,
    ) -> ReserveOutcome:
        code = feature_code.upper()
        if code not in FEATURE_DISPLAY_NAMES:
            # Licensing would answer validation_error; saying so here names the bug.
            raise LicensingUnavailable(
                f"unknown FeatureCode {feature_code!r}; Licensing accepts "
                f"{', '.join(sorted(FEATURE_DISPLAY_NAMES))}"
            )
        body: dict[str, Any] = {
            "TenantCode": tenant_code,
            "FeatureCode": code,
            "IdempotencyKey": idempotency_key,
        }
        if user_code:
            body["UserCode"] = user_code

        data = await self._post("/Credits/CheckAndReserve", body, access_token)
        raw = data.get("Reservation")
        reservation = None
        if isinstance(raw, dict):
            reservation = Reservation(
                id=str(raw.get("_id") or ""),
                status=str(raw.get("Status") or "Reserved"),
                estimated_credits=int(raw.get("EstimatedCredits") or 0),
                is_internal_user=bool(raw.get("IsInternalUser")),
                expires_at=str(raw.get("ExpiresAt") or ""),
            )
        return ReserveOutcome(
            allowed=bool(data.get("Allowed")),
            reason=str(data.get("Reason") or ""),
            message=str(data.get("Message") or ""),
            estimated_credits=int(data.get("EstimatedCredits") or 0),
            is_internal_user=bool(data.get("IsInternalUser")),
            reservation=reservation,
        )

    async def settle(
        self, *, tenant_code: str, reservation_id: str, transaction_id: str, result: str,
        feature_code: str, telemetry: RunTelemetry, trace_id: str = "",
        user_prompt: str = "", response_text: str = "", created_by: str = "",
        access_token: str | None = None,
    ) -> dict[str, Any]:
        body = settle_payload(
            tenant_code=tenant_code,
            reservation_id=reservation_id,
            transaction_id=transaction_id,
            result=result,
            feature_code=feature_code,
            telemetry=telemetry,
            region=self._region,
            trace_id=trace_id,
            user_prompt=user_prompt if self._send_user_prompt else "",
            response_text=response_text if self._send_user_prompt else "",
            created_by=created_by,
        )
        return await self._post("/Credits/Settle", body, access_token)

    async def release(
        self, *, tenant_code: str, reservation_id: str, reason: str = "",
        access_token: str | None = None,
    ) -> dict[str, Any]:
        params = {"tenantCode": tenant_code}
        if reason:
            params["reason"] = reason
        return await self._post(
            f"/Credits/Release/{reservation_id}", None, access_token, params=params
        )

    async def _post(
        self, path: str, body: dict[str, Any] | None, access_token: str | None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            response = await self._client.post(
                url, json=body, params=params, headers=await self._headers(access_token)
            )
        except LicensingUnavailable:
            raise
        except Exception as exc:
            raise LicensingUnavailable(f"{path} unreachable: {type(exc).__name__}") from exc

        if response.status_code == 401:
            raise LicensingUnavailable(f"{path}: ExampleApp IAM token rejected (401)")
        if response.status_code >= 400:
            # 400 bodies are plain text per the guide, so do not assume JSON.
            raise LicensingUnavailable(
                f"{path}: HTTP {response.status_code} {response.text[:200]}"
            )
        try:
            data = response.json()
        except Exception as exc:
            raise LicensingUnavailable(f"{path}: response was not JSON") from exc
        return data if isinstance(data, dict) else {}

    async def aclose(self) -> None:
        await self._client.aclose()
        closer = getattr(self._tokens, "aclose", None)
        if closer is not None:
            await closer()


def strip_bearer(value: str | None) -> str:
    """Accept either a bare token or a whole `Authorization` header value."""
    token = (value or "").strip()
    if token.lower().startswith("bearer "):
        token = token[len("bearer "):].strip()
    return token


def new_transaction_id() -> str:
    """One settle, one transaction id.

    Licensing stores this as the identifier of the settle transaction, so two
    settles must never share one. The trace id cannot serve here: a request can
    settle more than once over its life -- a Fail recorded when a run breaks, then
    a later correction -- and both would carry the same trace.
    """
    return str(uuid.uuid4())


def root_trace_id(header: str) -> str:
    """The trace id out of an X-Ray header.

    AWS sends `Root=1-abc-def;Parent=...;Sampled=1;Self=...`. Only the Root value
    identifies the trace; the rest is sampling bookkeeping that is noise in a
    stored billing record, and it is what a CloudWatch search matches on.
    """
    for part in (header or "").split(";"):
        key, _, value = part.partition("=")
        if key.strip().lower() == "root" and value.strip():
            return value.strip()
    return (header or "").strip()


def new_idempotency_key() -> str:
    """Unique per attempt.

    A retry of the *same* attempt reuses the key and so reuses the reservation; a
    genuine second attempt must not, or it would run on a reservation already
    settled.
    """
    return str(uuid.uuid4())
