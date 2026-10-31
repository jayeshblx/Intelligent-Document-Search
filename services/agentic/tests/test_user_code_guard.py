"""A UserCode that cannot be real must not reach the ledger unnoticed.

ExampleApp has been sending "N" where a code like "UserAdminAlex" belongs. The
value cannot be repaired on this side -- the rest of it never arrives -- so the
agent records the run and says loudly that the attribution is wrong, rather
than filling the credit ledger with entries nobody can trace to a person.

The guard is deliberately narrow. A short but genuine code must never trip it:
a false positive here trains people to ignore the warning, which is worse than
not having one.
"""

from __future__ import annotations

from typing import Any

import pytest
from rag_agentic.config import AgenticSettings
from rag_agentic.credit_gate import CreditGate, suspicious_user_code
from rag_agentic.models import AgentRequest


@pytest.mark.parametrize(
    "value",
    ["N", "Y", "n", "0", "1", "NO", "Yes", "true", "FALSE", "null", "None"],
)
def test_flag_shaped_values_are_rejected(value: str) -> None:
    assert suspicious_user_code(value), f"{value!r} cannot be a user code"


@pytest.mark.parametrize(
    "value",
    ["UserAdminAlex", "UADMINALEX", "CASEY", "USERQA_23", "C_PifEdit", "ab", "u12", "DEVUSER"],
)
def test_real_codes_pass(value: str) -> None:
    assert suspicious_user_code(value) == "", f"{value!r} is a plausible code"


def test_absent_is_not_flagged() -> None:
    """Missing is a different problem, and already visible as an empty field."""
    assert suspicious_user_code("") == ""
    assert suspicious_user_code("   ") == ""


def test_whitespace_around_a_single_character_still_trips() -> None:
    assert suspicious_user_code("  N  ")


# ---- the gate logs it ---------------------------------------------------------------


class _Licensing:
    def __init__(self) -> None:
        self.settles: list[dict[str, Any]] = []

    async def settle(self, **kw: Any) -> dict[str, Any]:
        self.settles.append(kw)
        return {"Status": "Committed"}

    async def check_and_reserve(self, **kw: Any) -> Any:  # pragma: no cover - unused
        raise AssertionError("settle-only")

    async def release(self, **kw: Any) -> dict[str, Any]:  # pragma: no cover - unused
        return {}

    async def aclose(self) -> None:
        return None


def _gate(licensing: _Licensing) -> CreditGate:
    return CreditGate(
        licensing,  # type: ignore[arg-type]
        AgenticSettings(
            licensing_enabled=True,
            licensing_base_url="https://tenantadmin.acmedev.com",
            licensing_tenant_codes="foxtrot=FOXTROT,bravo=BRAVO",
            licensing_settle_only=True,
        ),
    )


def _request(user_code: str) -> AgentRequest:
    return AgentRequest(
        base_url="https://bravo.acmedev.com/", query="how many documents?",
        user_code=user_code, reservation_id="res-1",
    )


async def test_a_suspect_code_is_logged(capsys: pytest.CaptureFixture[str]) -> None:
    gate = _gate(_Licensing())

    await gate.reserve(_request("N"), tenant="bravo", trace_id="t")

    out = capsys.readouterr().out
    assert "credit_user_code_suspect" in out
    assert "single character" in out


async def test_a_real_code_is_not_logged(capsys: pytest.CaptureFixture[str]) -> None:
    gate = _gate(_Licensing())

    await gate.reserve(_request("UserAdminAlex"), tenant="bravo", trace_id="t")

    assert "credit_user_code_suspect" not in capsys.readouterr().out


async def test_the_run_still_settles_despite_a_suspect_code() -> None:
    """The work happened and Licensing must still hear about it.

    Refusing to settle would turn a labelling problem into lost revenue and a
    reservation that hangs until it expires.
    """
    licensing = _Licensing()
    gate = _gate(licensing)

    hold = await gate.reserve(_request("N"), tenant="bravo", trace_id="t")
    assert hold is not None
    assert hold.user_code == "N", "reported as received, not silently blanked"
