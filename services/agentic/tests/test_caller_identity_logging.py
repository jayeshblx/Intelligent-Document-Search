"""What the caller actually sent, recorded without recording their credentials.

The agent's logs record parsed values, which cannot answer "did ExampleApp send
this, or did we lose it?". That question has already cost a round trip, so the
identity fields are captured as received.

The hard rule is that no secret may ride along. A bearer token in a log line is
a usable credential for its lifetime, and these logs go to CloudWatch where far
more people can read them than can read the request.
"""

from __future__ import annotations

import pytest
from rag_agentic.credit_gate import caller_identity_fields


def test_the_user_code_is_captured_as_sent() -> None:
    found = caller_identity_fields({"BaseUrl": "https://bravo.acmedev.com/", "UserCode": "N"})

    assert found == {"UserCode": "N"}


def test_every_spelling_is_captured_under_its_own_name() -> None:
    """Which spelling the caller used is itself the evidence."""
    found = caller_identity_fields(
        {"userCode": "N", "CreatedBy": "N", "UserName": "UserAdminAlex"}
    )

    assert found == {"userCode": "N", "CreatedBy": "N", "UserName": "UserAdminAlex"}


@pytest.mark.parametrize(
    "key", ["AccessToken", "accessToken", "access_token", "Authorization",
            "authorization", "Token", "BearerToken", "password"],
)
def test_secrets_are_never_captured(key: str) -> None:
    found = caller_identity_fields({key: "Bearer eyJsecret", "UserCode": "N"})

    assert found == {"UserCode": "N"}
    assert "eyJsecret" not in str(found)


def test_unrelated_fields_are_left_out() -> None:
    """This is a targeted diagnostic, not a dump of the request."""
    found = caller_identity_fields(
        {"query": "how many documents?", "objectCodes": "a,b,c",
         "ReservationId": "res-1", "UserCode": "N"}
    )

    assert found == {"UserCode": "N"}


def test_structured_values_are_skipped() -> None:
    """A nested object is not an identity, and could hide a secret inside it."""
    found = caller_identity_fields({"user": {"code": "UserAdminAlex", "token": "secret"}})

    assert found == {}


def test_an_empty_or_missing_value_is_still_reported() -> None:
    """An empty string and an absent field are different diagnoses."""
    assert caller_identity_fields({"UserCode": ""}) == {"UserCode": ""}
    assert caller_identity_fields({"UserCode": None}) == {"UserCode": None}
    assert caller_identity_fields({}) == {}


def test_a_payload_with_nothing_identifying_is_empty() -> None:
    assert caller_identity_fields({"baseUrl": "x", "query": "y"}) == {}
