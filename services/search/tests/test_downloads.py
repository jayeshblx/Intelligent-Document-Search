"""Credential-free download URLs: signed tokens + streaming, no secrets in the URL."""

from __future__ import annotations

import pytest
from rag_search.downloads import (
    DownloadNotFound,
    DownloadService,
    DownloadTokenError,
    _sign,
    _verify,
)

_KEY = "foxtrot/04807ffba4dd_Southgate FSSC 2024~pdf"
_SECRET = b"unit-test-secret"


def test_token_round_trips_and_hides_the_key() -> None:
    token = _sign(_KEY, _SECRET, ttl_s=3600)
    assert _verify(token, _SECRET) == _KEY
    # The token is opaque: the raw key text is not sitting in it.
    assert "Southgate" not in token
    assert "foxtrot/" not in token


def test_tampered_or_wrong_secret_is_rejected() -> None:
    token = _sign(_KEY, _SECRET, ttl_s=3600)
    assert _verify(token, b"different-secret") is None
    assert _verify(token[:-2] + "AA", _SECRET) is None   # flipped signature/payload bytes
    assert _verify("not-base64-@@", _SECRET) is None
    assert _verify("", _SECRET) is None


def test_expired_token_is_rejected() -> None:
    assert _verify(_sign(_KEY, _SECRET, ttl_s=-1), _SECRET) is None


def test_url_contains_only_the_filename_no_credentials_or_key() -> None:
    svc = DownloadService(_FakeS3(), "my-bucket", secret=_SECRET)
    url = svc.url_for(_KEY)
    assert url is not None
    # Ends in the clean, original filename (percent-encoded) — the visible part.
    assert url.endswith("/Southgate%20FSSC%202024.pdf")
    assert url.startswith("/v1/files/")
    # None of the things a presigned S3 URL leaked are present.
    for leaked in ("AWSAccessKeyId", "AKIA", "Signature", "Expires", "amazonaws", "04807ffba4dd", "~pdf"):
        assert leaked not in url
    # ...and the round trip still resolves to the real key server-side.
    token = url.split("/")[3]
    assert svc.resolve(token) == _KEY


def test_url_for_empty_key_is_none() -> None:
    assert DownloadService(_FakeS3(), "b", secret=_SECRET).url_for("") is None


async def test_open_returns_bytes_with_clean_filename_and_type() -> None:
    svc = DownloadService(_FakeS3(data=b"%PDF-1.4 body"), "b", secret=_SECRET)
    dl = await svc.open(svc.url_for(_KEY).split("/")[3])
    assert dl.filename == "Southgate FSSC 2024.pdf"
    assert dl.content_type == "application/pdf"
    assert dl.disposition == "inline"          # PDFs render in-tab
    assert dl.data == b"%PDF-1.4 body"
    assert dl.content_length == len(b"%PDF-1.4 body")
    assert "filename*=UTF-8" in dl.content_disposition


async def test_open_rejects_bad_token_and_missing_object() -> None:
    svc = DownloadService(_FakeS3(), "b", secret=_SECRET)
    with pytest.raises(DownloadTokenError):
        await svc.open("garbage")
    missing = DownloadService(_FakeS3(fail=True), "b", secret=_SECRET)
    with pytest.raises(DownloadNotFound):
        await missing.open(missing.url_for(_KEY).split("/")[3])


class _FakeS3:
    def __init__(self, *, data: bytes = b"%PDF-1.4 abc", fail: bool = False) -> None:
        self._data = data
        self._fail = fail

    async def get_bytes(self, bucket: str, key: str) -> bytes:
        if self._fail:
            raise RuntimeError("NoSuchKey")
        return self._data
