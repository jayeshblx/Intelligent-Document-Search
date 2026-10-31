"""The credit lifecycle around one agent run.

`app.py` already owns the only place that sees a whole request -- its timing, its
failures and its token totals -- so the Licensing lifecycle hangs there. This
module keeps the policy out of the handler:

    gate = CreditGate(client, settings)
    hold = await gate.reserve(request, tenant, trace_id)   # may raise LicensingDeclined
    ...run the agent...
    await gate.settle_success(hold, response)              # or settle_failure(hold, ...)

A `Hold` is `None` whenever licensing is off or not applicable, and every method
tolerates that, so the handler reads the same whether credits are metered or not.

Settling is deliberately best-effort. The run has already happened by then: if
Licensing is unreachable at that moment, discarding the answer would punish the
user for an outage they did not cause and would not recover the credits either.
An unsettled reservation expires on its own and shows up in Licensing's own
reconciliation; a dropped answer is simply lost. Every such failure is logged at
error level with the reservation id, so the gap is traceable rather than silent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_search.embedder import query_embedding_tokens

from rag_agentic.config import AgenticSettings
from rag_agentic.converse import spent_so_far
from rag_agentic.licensing import (
    LicensingClient,
    LicensingDeclined,
    LicensingUnavailable,
    RunTelemetry,
    new_idempotency_key,
    new_transaction_id,
    root_trace_id,
)
from rag_agentic.models import AgentRequest, AgentResponse
from rag_core.logging import get_logger

_log = get_logger("rag_agentic.credit_gate")


@dataclass(frozen=True)
class Hold:
    """A reservation waiting to be settled."""

    reservation_id: str
    tenant_code: str
    feature_code: str
    trace_id: str
    access_token: str | None = None
    user_code: str = ""
    query: str = ""
    estimated_credits: int = 0
    is_internal_user: bool = False


class CreditGate:
    def __init__(self, client: LicensingClient | None, settings: AgenticSettings) -> None:
        self._client = client
        self._settings = settings

    @property
    def active(self) -> bool:
        return self._client is not None and self._settings.licensing_enabled

    async def reserve(
        self, request: AgentRequest, *, tenant: str, trace_id: str
    ) -> Hold | None:
        """Hold credits for this run, or raise rather than let it start.

        Raises `LicensingDeclined` when Licensing says no -- the caller turns that
        into a refusal naming the reason. Raises `LicensingUnavailable` when the
        gate could not be evaluated and `licensing_fail_open` is off.
        """
        if not self.active or self._client is None:
            return None

        feature_code = (request.feature_code or self._settings.licensing_feature_code).upper()
        tenant_code = self._settings.licensing_tenant_code(tenant)
        token = request.access_token.get_secret_value() if request.access_token else None

        # Whatever arrives here is what the credit record will be attributed to,
        # and a wrong value is only discoverable once the ledger is full of it.
        if reason := suspicious_user_code(request.user_code or ""):
            _log.warning(
                "credit_user_code_suspect",
                user_code=request.user_code or "", reason=reason,
                tenant_code=tenant_code, feature_code=feature_code,
            )

        # ExampleApp reserved the credits already and sent the reservation with the
        # request. There is nothing to decide here: settle it when the run ends.
        if supplied := (request.reservation_id or "").strip():
            _log.info(
                "credit_reservation_supplied",
                tenant_code=tenant_code, feature_code=feature_code, reservation_id=supplied,
            )
            return Hold(
                reservation_id=supplied, tenant_code=tenant_code, feature_code=feature_code,
                trace_id=trace_id, access_token=token, user_code=request.user_code or "",
                query=request.query,
            )

        if self._settings.licensing_settle_only:
            # Usage reporting only: no reservation means nothing to report against.
            # Never refuse the run for it -- metering must not gate the answer in
            # this mode, or a caller that forgot the field loses its search.
            _log.warning(
                "credit_reservation_missing",
                tenant_code=tenant_code, feature_code=feature_code,
                user_code=request.user_code or "",
            )
            return None

        try:
            outcome = await self._client.check_and_reserve(
                tenant_code=tenant_code,
                feature_code=feature_code,
                idempotency_key=new_idempotency_key(),
                user_code=request.user_code or "",
                access_token=token,
            )
        except LicensingUnavailable as exc:
            if self._settings.licensing_fail_open:
                # Loud on purpose: this ran without being metered, and somebody
                # has to be able to find out how often that happens.
                _log.error(
                    "credit_gate_bypassed",
                    tenant_code=tenant_code, feature_code=feature_code, error=str(exc),
                )
                return None
            _log.error(
                "credit_gate_unavailable",
                tenant_code=tenant_code, feature_code=feature_code, error=str(exc),
            )
            raise

        if not outcome.allowed:
            _log.warning(
                "credit_reservation_declined",
                tenant_code=tenant_code, feature_code=feature_code,
                reason=outcome.reason, user_code=request.user_code or "",
            )
            raise LicensingDeclined(outcome.reason, outcome.message)

        if outcome.reservation is None or not outcome.reservation.id:
            # Allowed with nothing to settle later: treat as a failure of the gate
            # rather than invent an id and fail at Settle instead.
            raise LicensingUnavailable("Licensing allowed the run but returned no reservation")

        _log.info(
            "credit_reserved",
            tenant_code=tenant_code, feature_code=feature_code,
            reservation_id=outcome.reservation.id,
            estimated_credits=outcome.estimated_credits,
            is_internal_user=outcome.is_internal_user,
        )
        return Hold(
            reservation_id=outcome.reservation.id,
            tenant_code=tenant_code,
            feature_code=feature_code,
            trace_id=trace_id,
            access_token=token,
            user_code=request.user_code or "",
            query=request.query,
            estimated_credits=outcome.estimated_credits,
            is_internal_user=outcome.is_internal_user,
        )

    async def settle_success(self, hold: Hold | None, response: AgentResponse) -> None:
        if hold is None:
            return
        await self._settle(hold, "Success", RunTelemetry.from_response(response),
                           answer=response.answer)

    async def settle_failure(
        self, hold: Hold | None, *, model_id: str = "", execution_time_ms: int = 0
    ) -> None:
        """The run broke. Record the attempt, and what it cost before it broke.

        The spend is read from the per-request accumulator rather than from a
        response, because there is no response -- the exception unwound the stack
        and took the partial result with it.
        """
        if hold is None:
            return
        await self._settle(
            hold, "Fail",
            RunTelemetry.failed(
                spent=spent_so_far(),
                embedding_tokens=query_embedding_tokens(),
                model_id=model_id,
                execution_time_ms=execution_time_ms,
            ),
        )

    async def _settle(
        self, hold: Hold, result: str, telemetry: RunTelemetry, *, answer: str = ""
    ) -> None:
        if self._client is None:
            return
        try:
            settled = await self._client.settle(
                tenant_code=hold.tenant_code,
                reservation_id=hold.reservation_id,
                transaction_id=new_transaction_id(),
                result=result,
                feature_code=hold.feature_code,
                telemetry=telemetry,
                trace_id=root_trace_id(hold.trace_id),
                user_prompt=hold.query,
                response_text=answer,
                created_by=hold.user_code,
                access_token=hold.access_token,
            )
        except LicensingUnavailable as exc:
            # Never re-raised: see the module docstring. The answer still goes back.
            _log.error(
                "credit_settle_failed",
                reservation_id=hold.reservation_id, tenant_code=hold.tenant_code,
                result=result, error=str(exc),
                input_tokens=telemetry.input_tokens, output_tokens=telemetry.output_tokens,
            )
            return
        _log.info(
            "credit_settled",
            reservation_id=hold.reservation_id, tenant_code=hold.tenant_code,
            result=result, status=str(settled.get("Status") or ""),
            actual_credits=settled.get("ActualCredits"),
            input_tokens=telemetry.input_tokens, output_tokens=telemetry.output_tokens,
            total_tokens=telemetry.total_tokens,
            embedding_tokens=telemetry.embedding_tokens,
            execution_time_ms=telemetry.execution_time_ms,
            # Named here too: a credit record is queried after the fact, but a
            # CloudWatch line is what someone reads while the run is fresh.
            sub_agents=",".join(u.agent_id for u in telemetry.sub_agents if u.agent_id),
            object_code=telemetry.object_code,
        )

    async def settle_rejected(
        self, body: dict[str, Any], *, tenant: str, trace_id: str, reason: str
    ) -> None:
        """Settle a reservation whose request never became a valid one.

        A payload the agent refuses still cost the caller a reservation, and that
        reservation is only discoverable from the raw body -- validation failed, so
        there is no parsed request to read it from. Without this the reservation
        hangs until it expires and the failure is invisible to Licensing.
        """
        if not self.active or self._client is None:
            return
        reservation = _raw(body, "ReservationId")
        if not reservation:
            return  # nothing was reserved, so there is nothing to settle
        feature = (_raw(body, "FeatureCode") or self._settings.licensing_feature_code).upper()
        hold = Hold(
            reservation_id=reservation,
            tenant_code=self._settings.licensing_tenant_code(tenant),
            feature_code=feature,
            trace_id=trace_id,
            access_token=_raw(body, "AccessToken", "Authorization") or None,
            user_code=_raw(body, "UserCode"),
            # A rejected request is exactly the one an operator most wants to read
            # back, so the prompt is recovered from the raw body too.
            query=_raw(body, "query", "queryText", "question"),
        )
        _log.warning(
            "credit_settling_rejected_request",
            reservation_id=reservation, reason=reason, tenant_code=hold.tenant_code,
        )
        await self._settle(hold, "Fail", RunTelemetry.failed())

    async def release(self, hold: Hold | None, reason: str = "Cancelled") -> None:
        """Give the hold back when the run never started."""
        if hold is None or self._client is None:
            return
        try:
            await self._client.release(
                tenant_code=hold.tenant_code,
                reservation_id=hold.reservation_id,
                reason=reason,
                access_token=hold.access_token,
            )
            _log.info("credit_released", reservation_id=hold.reservation_id, reason=reason)
        except LicensingUnavailable as exc:
            _log.error(
                "credit_release_failed",
                reservation_id=hold.reservation_id, reason=reason, error=str(exc),
            )


_SECRET_KEYS = {"accesstoken", "authorization", "token", "bearertoken", "password"}
_IDENTITY_KEYS = {
    "usercode", "createdby", "username", "userid", "user", "code", "loginname",
    "userloginname", "useremail", "email", "createdbyuser",
}


def caller_identity_fields(body: dict[str, Any]) -> dict[str, Any]:
    """Every field in the raw payload that could be carrying the caller's identity.

    Used only for diagnosis. ExampleApp has been sending ``UserCode: "N"`` where a
    code like ``UserAdminAlex`` belongs, and the agent's own logs cannot settle
    whether the value arrives that way or is lost here -- they record what was
    parsed, not what was sent. This records the identity-ish fields exactly as
    received, so the question is answerable from one request.

    Secrets are never included: a bearer token in a log line is a usable
    credential for its lifetime, and no amount of debugging value justifies it.
    """
    found: dict[str, Any] = {}
    for key, value in body.items():
        canonical = "".join(ch for ch in str(key).lower() if ch.isalnum())
        if canonical in _SECRET_KEYS:
            continue
        if canonical in _IDENTITY_KEYS and not isinstance(value, dict | list):
            found[str(key)] = value
    return found


def suspicious_user_code(user_code: str) -> str:
    """Why this UserCode cannot be a real one, or "" if it looks fine.

    ExampleApp has been sending a single character ("N") where a code like
    "UserAdminAlex" belongs, which lands in the credit record's UserCode and makes
    the ledger unattributable. The value cannot be repaired here -- the rest of
    it never arrives -- so the next best thing is to refuse to let it pass
    silently. Deliberately narrow: only shapes that cannot be a real code, so a
    short-but-genuine code never trips it.
    """
    value = (user_code or "").strip()
    if not value:
        return ""  # absent is a different problem, already visible as user_code=
    if len(value) == 1:
        return "single character"
    if value.upper() in {"Y", "N", "YES", "NO", "TRUE", "FALSE", "NULL", "NONE", "0", "1"}:
        return "looks like a flag, not a user code"
    return ""


def _raw(body: dict[str, Any], *names: str) -> str:
    """Read a field from an unvalidated body, tolerating the caller's spelling.

    Used only on the rejection path: the payload failed validation, so the normal
    alias handling never ran, but the reservation still has to be found.
    """
    flat = {str(k).replace("_", "").lower(): v for k, v in body.items()}
    for name in names:
        value = flat.get(name.replace("_", "").lower())
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""
