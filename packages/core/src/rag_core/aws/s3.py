"""Async S3 client tailored to ingestion.

Responsibilities: enumerate the source prefix (paginated, memory-bounded), fetch
object metadata cheaply (HEAD for etag/size/version), and stream object bodies for
checksumming/processing without loading whole files into memory. botocore errors
are mapped to the platform retry taxonomy.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from datetime import datetime
from typing import Any
from urllib.parse import quote

from botocore.exceptions import BotoCoreError, ClientError

from rag_core.aws.session import AwsSessionFactory
from rag_core.logging import get_logger
from rag_core.models.base import DomainModel
from rag_core.resilience import NonRetryableError, RetryableError, async_retry, retryable_status

_log = get_logger("rag_core.aws.s3")

# 4xx codes that are permanent (no point retrying).
_PERMANENT_S3_CODES = {"NoSuchKey", "NoSuchBucket", "AccessDenied", "InvalidObjectState"}

# Objects are stored with names like "<hash>_<name>~pdf" (no real extension), so S3
# serves them as text/plain. These maps let presigned URLs override the response
# content-type + filename so browsers render/download them correctly.
_CONTENT_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "doc": "application/msword",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "xls": "application/vnd.ms-excel",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "ppt": "application/vnd.ms-powerpoint",
    "html": "text/html", "htm": "text/html", "txt": "text/plain", "csv": "text/csv",
    "xml": "application/xml", "json": "application/json",
    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif",
    "webp": "image/webp", "tiff": "image/tiff", "tif": "image/tiff", "bmp": "image/bmp",
}
# Types the browser can display in-tab; others download as attachments.
_INLINE_EXTENSIONS = {"pdf", "html", "htm", "txt", "csv", "xml", "json",
                      "png", "jpg", "jpeg", "gif", "webp", "tiff", "tif", "bmp"}
_EXT_RE = re.compile(r"[.~]([A-Za-z0-9]+)$")


def _object_extension(key: str) -> str:
    m = _EXT_RE.search(key.rsplit("/", 1)[-1])
    return m.group(1).lower() if m else ""


def _download_name(key: str) -> str:
    """A clean filename: drop the hash prefix, turn ``~pdf`` into ``.pdf``."""
    name = key.rsplit("/", 1)[-1]
    name = re.sub(r"^[0-9a-f]{8,}_", "", name)          # strip the storage hash prefix
    name = _EXT_RE.sub(lambda m: "." + m.group(1).lower(), name)  # ~pdf -> .pdf
    name = name.replace('"', "").replace("\r", " ").replace("\n", " ").strip()
    return name or "document"


# Typographic characters real document names are full of, mapped to ASCII.
# Transliterating beats dropping: an en-dash should become "-", not vanish. Written as
# escapes because the literal glyphs are visually identical to their ASCII twins.
_ASCII_PUNCT = str.maketrans(
    {
        "\u2013": "-",    # en dash
        "\u2014": "-",    # em dash
        "\u2012": "-",    # figure dash
        "\u2212": "-",    # minus sign
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote / apostrophe
        "\u201a": "'",    # single low quote
        "\u201b": "'",    # single high-reversed quote
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u201e": '"',    # double low quote
        "\u2026": "...",  # ellipsis
        "\u00a0": " ",    # non-breaking space
        "\u2022": "-",    # bullet
        "\u00ad": "",     # soft hyphen
    }
)


def _ascii_fallback(name: str) -> str:
    """An ASCII-only rendering of ``name`` for the legacy ``filename`` parameter."""
    name = name.translate(_ASCII_PUNCT)
    # NFKD splits "é" into "e" + combining accent; dropping non-ASCII then leaves "e".
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r'[\\"]', "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name or "document"


def _content_disposition(disposition: str, name: str) -> str:
    """Build a Content-Disposition value S3 will accept (RFC 6266).

    S3 rejects the entire presigned URL if this header value isn't ISO-8859-1
    encodable, and real filenames are full of en-dashes and curly quotes. So emit
    both parameters: an ASCII-only ``filename`` for old clients, and ``filename*``
    carrying the true UTF-8 name percent-encoded (therefore pure ASCII on the wire).
    Modern browsers prefer ``filename*``, so the user still gets the exact name.
    """
    star = quote(name, safe="")
    return f"{disposition}; filename=\"{_ascii_fallback(name)}\"; filename*=UTF-8''{star}"


class S3ObjectSummary(DomainModel):
    """Lightweight listing entry — enough to decide change/dedup before download."""

    bucket: str
    key: str
    size_bytes: int
    etag: str
    last_modified: datetime | None = None
    storage_class: str | None = None


def _map_client_error(exc: ClientError) -> Exception:
    code = exc.response.get("Error", {}).get("Code", "")
    status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0)
    if code in _PERMANENT_S3_CODES:
        return NonRetryableError(f"S3 permanent error {code}: {exc}")
    if retryable_status(status) or code in {"SlowDown", "RequestTimeout", "Throttling"}:
        return RetryableError(f"S3 transient error {code} (status {status}): {exc}")
    return NonRetryableError(f"S3 error {code}: {exc}")


class S3Client:
    """Persistent async S3 client. Use as an async context manager or call
    ``aclose()`` from your service lifecycle."""

    def __init__(self, factory: AwsSessionFactory) -> None:
        self._factory = factory
        self._stack = AsyncExitStack()
        self._client: Any | None = None

    async def __aenter__(self) -> S3Client:
        await self._ensure()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _ensure(self) -> Any:
        if self._client is None:
            self._client = await self._stack.enter_async_context(self._factory.client("s3"))
        return self._client

    async def aclose(self) -> None:
        await self._stack.aclose()
        self._client = None

    async def list_objects(
        self, bucket: str, prefix: str = "", *, page_size: int = 1000
    ) -> AsyncIterator[S3ObjectSummary]:
        """Async-generate object summaries under ``prefix`` (handles pagination).

        Memory-bounded: yields page by page rather than materialising the whole
        listing — essential when a prefix holds millions of keys. Transient
        per-request faults are absorbed by botocore's adaptive retries; a caller
        that needs whole-listing retry should wrap its own consumption loop.
        """
        client = await self._ensure()
        paginator = client.get_paginator("list_objects_v2")
        try:
            async for page in paginator.paginate(
                Bucket=bucket, Prefix=prefix, PaginationConfig={"PageSize": page_size}
            ):
                for obj in page.get("Contents", []):
                    # Folder placeholder keys (size 0, trailing slash) are skipped.
                    if obj["Key"].endswith("/") and obj["Size"] == 0:
                        continue
                    yield S3ObjectSummary(
                        bucket=bucket,
                        key=obj["Key"],
                        size_bytes=obj["Size"],
                        etag=obj["ETag"].strip('"'),
                        last_modified=obj.get("LastModified"),
                        storage_class=obj.get("StorageClass"),
                    )
        except ClientError as exc:
            raise _map_client_error(exc) from exc
        except BotoCoreError as exc:
            raise RetryableError(f"S3 list transport error: {exc}") from exc

    @async_retry(max_attempts=5)
    async def head_object(self, bucket: str, key: str) -> dict[str, Any]:
        """HEAD an object for etag/size/version/content-type without downloading."""
        client = await self._ensure()
        try:
            return await client.head_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            raise _map_client_error(exc) from exc

    @async_retry(max_attempts=5)
    async def put_bytes(
        self,
        bucket: str,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> None:
        """Upload an object. Used to persist derived artifacts (extracted content,
        snapshots) back to S3. Idempotent on (bucket, key)."""
        client = await self._ensure()
        try:
            await client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
        except ClientError as exc:
            raise _map_client_error(exc) from exc

    @async_retry(max_attempts=5)
    async def get_bytes(self, bucket: str, key: str, *, version_id: str | None = None) -> bytes:
        """Download a full object into memory. Prefer ``stream`` for large files."""
        client = await self._ensure()
        kwargs: dict[str, Any] = {"Bucket": bucket, "Key": key}
        if version_id:
            kwargs["VersionId"] = version_id
        try:
            resp = await client.get_object(**kwargs)
            async with resp["Body"] as body:
                return await body.read()
        except ClientError as exc:
            raise _map_client_error(exc) from exc

    @async_retry(max_attempts=5)
    async def get_range(self, bucket: str, key: str, start: int, end: int) -> bytes:
        """Bytes ``[start, end]`` of an object (inclusive, as HTTP ranges are).

        One GET saturates a single connection, which is the whole cost of restoring
        a large index snapshot: it is one multi-hundred-megabyte object, so object-
        level concurrency buys nothing. Fetching ranges in parallel does.
        """
        client = await self._ensure()
        try:
            resp = await client.get_object(
                Bucket=bucket, Key=key, Range=f"bytes={start}-{end}"
            )
            async with resp["Body"] as body:
                data: bytes = await body.read()
                return data
        except ClientError as exc:
            raise _map_client_error(exc) from exc

    async def presigned_get(
        self, bucket: str, key: str, *, expires_s: int = 3600
    ) -> str:
        """Generate a time-limited presigned GET URL for downloading an object.

        Signing is a local crypto operation (no network round-trip). Used to give
        the UI a downloadable link to the exact source document an answer came from,
        without making the bucket public.

        Objects are stored without a real file extension (``<hash>_name~pdf``), so S3
        serves them as ``text/plain`` and browsers can't open them. We override the
        response content-type (e.g. ``application/pdf``) and set a clean filename via
        ``ResponseContentType`` / ``ResponseContentDisposition`` so PDFs render inline
        and other formats download with a sensible name.
        """
        client = await self._ensure()
        params: dict[str, Any] = {"Bucket": bucket, "Key": key}
        ext = _object_extension(key)
        if content_type := _CONTENT_TYPES.get(ext):
            params["ResponseContentType"] = content_type
        disposition = "inline" if ext in _INLINE_EXTENSIONS else "attachment"
        params["ResponseContentDisposition"] = _content_disposition(
            disposition, _download_name(key)
        )
        return await client.generate_presigned_url(
            "get_object", Params=params, ExpiresIn=expires_s
        )

    async def stream(
        self, bucket: str, key: str, *, chunk_size: int = 1024 * 1024
    ) -> AsyncIterator[bytes]:
        """Stream an object body in ``chunk_size`` blocks (memory-bounded).

        Uses aiobotocore's ``iter_chunks`` rather than ``body.read(size)``: the streaming
        body in this stack wraps an aiohttp response whose ``read()`` takes no size
        argument, so ``read(chunk_size)`` raises ``TypeError`` against real S3 (moto's body
        happened to accept it, hiding the bug in tests).
        """
        client = await self._ensure()
        try:
            resp = await client.get_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            raise _map_client_error(exc) from exc
        async with resp["Body"] as body:
            yielded = False
            try:
                while True:
                    block = await body.read(chunk_size)
                    if not block:
                        break
                    yielded = True
                    yield block
            except TypeError:
                # Real aiobotocore's streaming body is an aiohttp response whose read()
                # takes no size argument, so read(chunk_size) raises immediately. Fall back
                # to a chunk iterator: aiohttp exposes ``.content.iter_chunked``; a plain
                # aiobotocore StreamingBody exposes ``.iter_chunks``. (moto's body accepts a
                # size, so the read() path above serves the unit tests.)
                if yielded:
                    raise
                content = getattr(body, "content", None)
                if content is not None and hasattr(content, "iter_chunked"):
                    async for block in content.iter_chunked(chunk_size):
                        if block:
                            yield block
                else:
                    async for block in body.iter_chunks(chunk_size):
                        if block:
                            yield block
