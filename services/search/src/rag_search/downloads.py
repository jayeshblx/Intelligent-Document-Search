"""Credential-free document downloads.

A presigned S3 URL leaks the bucket, the storage key, an ``AWSAccessKeyId`` and a
signature into the browser's address bar — everything a user sees when they save a
file. Instead, the UI is handed an app-relative URL that ends in the *original
filename* and carries only a short opaque token; the app streams the object from S3
server-side (using its own credentials, which never leave the process).

The token is an HMAC-signed, time-limited reference to the S3 key. It is signed, not
just encoded, for one reason: without a signature the endpoint would be an open proxy
to *any* key in the bucket (change the path, read someone else's document). The
signature means the only keys anyone can fetch are the ones the app itself minted a
link for. The key is never revealed to the client — only the filename is.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

# Reuse the exact filename / content-type derivation the presigned path used, so the
# streamed download is byte-for-byte the same experience minus the credentials.
from rag_core.aws.s3 import (
    _CONTENT_TYPES,
    _INLINE_EXTENSIONS,
    _content_disposition,
    _download_name,
    _object_extension,
)

_SIG_LEN = 16  # 128-bit truncated HMAC tag — ample against forgery, keeps the URL short


class DownloadTokenError(Exception):
    """The token is missing, tampered with, or past its expiry."""


class DownloadNotFound(Exception):
    """The token was valid but the object could not be fetched (gone / no access)."""


def _sign(key: str, secret: bytes, ttl_s: int) -> str:
    exp = int(time.time()) + ttl_s
    payload = f"{exp}:{key}".encode()
    tag = hmac.new(secret, payload, hashlib.sha256).digest()[:_SIG_LEN]
    return base64.urlsafe_b64encode(tag + payload).decode().rstrip("=")


def _verify(token: str, secret: bytes) -> str | None:
    try:
        raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
    except (ValueError, TypeError):
        return None
    if len(raw) <= _SIG_LEN:
        return None
    tag, payload = raw[:_SIG_LEN], raw[_SIG_LEN:]
    expected = hmac.new(secret, payload, hashlib.sha256).digest()[:_SIG_LEN]
    if not hmac.compare_digest(tag, expected):  # constant-time: no signature oracle
        return None
    try:
        exp_str, key = payload.decode().split(":", 1)
        expires_at = int(exp_str)
    except (ValueError, UnicodeDecodeError):
        return None
    if time.time() > expires_at:
        return None
    return key


@dataclass(slots=True)
class Download:
    filename: str
    content_type: str
    disposition: str  # "inline" | "attachment"
    data: bytes

    @property
    def content_length(self) -> int:
        return len(self.data)

    @property
    def content_disposition(self) -> str:
        return _content_disposition(self.disposition, self.filename)


class DownloadService:
    """Mints and resolves credential-free download URLs backed by S3 streaming."""

    def __init__(
        self,
        s3: Any,
        bucket: str,
        *,
        secret: bytes,
        url_prefix: str = "/v1/files",
        ttl_s: int = 86_400,
    ) -> None:
        self._s3 = s3
        self._bucket = bucket
        self._secret = secret
        self._prefix = url_prefix.rstrip("/")
        self._ttl = ttl_s

    def url_for(self, key: str) -> str | None:
        """The public, credential-free URL for an S3 key: ``/v1/files/<token>/<name>``.

        The token authorises the fetch; the trailing segment is the original filename,
        so it is what the browser shows and what "Save as" defaults to. Returns None for
        a falsy key (nothing to link)."""
        if not key:
            return None
        token = _sign(key, self._secret, self._ttl)
        # The filename is cosmetic (the token is authoritative), but percent-encode it so
        # spaces/punctuation in real document names stay a single valid path segment.
        return f"{self._prefix}/{token}/{quote(_download_name(key), safe='')}"

    def resolve(self, token: str) -> str:
        key = _verify(token, self._secret)
        if key is None:
            raise DownloadTokenError("invalid or expired download token")
        return key

    async def open(self, token: str) -> Download:
        """Resolve a token and fetch the object. Raises DownloadTokenError (→ 404) for a bad
        token, DownloadNotFound (→ 404) if the object is gone.

        The object is buffered in full rather than streamed: these are compliance documents
        (PDFs of a few hundred KB), buffering lets a fetch error surface as a clean 404
        *before* any bytes are sent, and gives the browser an exact Content-Length — a
        truncated stream is what made the viewer report a 0-page file."""
        key = self.resolve(token)
        try:
            data = await self._s3.get_bytes(self._bucket, key)
        except Exception as exc:  # NoSuchKey / AccessDenied / transport — all 404 to the user
            raise DownloadNotFound(str(exc)) from exc
        ext = _object_extension(key)
        return Download(
            filename=_download_name(key),
            content_type=_CONTENT_TYPES.get(ext, "application/octet-stream"),
            disposition="inline" if ext in _INLINE_EXTENSIONS else "attachment",
            data=data,
        )
