"""Tests for S3Client error handling, streaming, and presigned URLs."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from botocore.exceptions import BotoCoreError, ClientError

from rag_core.aws.s3 import (
    S3Client,
    _ascii_fallback,
    _content_disposition,
    _download_name,
    _object_extension,
)
from rag_core.aws.session import AwsSessionFactory
from rag_core.resilience import NonRetryableError, RetryableError


@pytest.fixture
def mock_factory() -> AwsSessionFactory:
    factory = MagicMock(spec=AwsSessionFactory)
    return factory


@pytest.fixture
def mock_client() -> AsyncMock:
    return AsyncMock()


def test_object_extension_parsing() -> None:
    assert _object_extension("doc~pdf") == "pdf"
    assert _object_extension("report.docx") == "docx"
    assert _object_extension("a/b/file~xlsx") == "xlsx"
    assert _object_extension("no_extension") == ""
    assert _object_extension("mixed.PDF") == "pdf"


def test_download_name_cleanup() -> None:
    assert _download_name("12345678_report~pdf") == "report.pdf"
    # "abc" is not an 8+ char hex hash, so the prefix is kept.
    assert _download_name("abc_file~docx") == "abc_file.docx"
    assert _download_name('doc"with"quotes~txt') == "docwithquotes.txt"
    assert _download_name("file\rwith\nnewlines~csv") == "file with newlines.csv"
    assert _download_name("normal.pdf") == "normal.pdf"
    # Known edge case: a nameless key becomes the dotfile ".pdf", which is truthy,
    # so the `or "document"` fallback does not catch it.
    assert _download_name("~pdf") == ".pdf"


async def test_s3_list_objects_success(mock_factory: AwsSessionFactory, mock_client: AsyncMock) -> None:
    mock_factory.client = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_client)))
    
    paginator = AsyncMock()
    async def mock_paginate(*args, **kwargs):
        yield {
            "Contents": [
                {"Key": "docs/a.pdf", "Size": 100, "ETag": '"abc123"', "LastModified": None},
                {"Key": "docs/b.txt", "Size": 50, "ETag": '"def456"', "LastModified": None},
            ]
        }
    paginator.paginate = mock_paginate
    mock_client.get_paginator = MagicMock(return_value=paginator)
    
    s3 = S3Client(mock_factory)
    results = []
    async with s3:
        async for obj in s3.list_objects("bucket", "docs/"):
            results.append(obj)
    
    assert len(results) == 2
    assert results[0].key == "docs/a.pdf"
    assert results[0].etag == "abc123"
    assert results[1].key == "docs/b.txt"


async def test_s3_list_objects_skips_folder_placeholders(mock_factory: AwsSessionFactory, mock_client: AsyncMock) -> None:
    mock_factory.client = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_client)))
    
    paginator = AsyncMock()
    async def mock_paginate(*args, **kwargs):
        yield {
            "Contents": [
                {"Key": "folder/", "Size": 0, "ETag": '""', "LastModified": None},
                {"Key": "folder/file.txt", "Size": 20, "ETag": '"xyz"', "LastModified": None},
            ]
        }
    paginator.paginate = mock_paginate
    mock_client.get_paginator = MagicMock(return_value=paginator)
    
    s3 = S3Client(mock_factory)
    results = []
    async with s3:
        async for obj in s3.list_objects("bucket", "folder/"):
            results.append(obj)
    
    assert len(results) == 1
    assert results[0].key == "folder/file.txt"


async def test_s3_list_objects_client_error_mapped(mock_factory: AwsSessionFactory, mock_client: AsyncMock) -> None:
    mock_factory.client = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_client)))
    
    paginator = AsyncMock()
    async def mock_paginate(*args, **kwargs):
        raise ClientError(
            {"Error": {"Code": "NoSuchBucket"}},
            "ListObjectsV2"
        )
        yield  # unreachable: makes this a true async generator, not a coroutine
    paginator.paginate = mock_paginate
    mock_client.get_paginator = MagicMock(return_value=paginator)
    
    s3 = S3Client(mock_factory)
    with pytest.raises(NonRetryableError, match="NoSuchBucket"):
        async with s3:
            async for _ in s3.list_objects("bad-bucket", ""):
                pass


async def test_s3_list_objects_retryable_error(mock_factory: AwsSessionFactory, mock_client: AsyncMock) -> None:
    mock_factory.client = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_client)))
    
    paginator = AsyncMock()
    async def mock_paginate(*args, **kwargs):
        raise BotoCoreError()
        yield  # unreachable: makes this a true async generator, not a coroutine
    paginator.paginate = mock_paginate
    mock_client.get_paginator = MagicMock(return_value=paginator)
    
    s3 = S3Client(mock_factory)
    with pytest.raises(RetryableError, match="transport error"):
        async with s3:
            async for _ in s3.list_objects("bucket", ""):
                pass


async def test_s3_stream_yields_chunks(mock_factory: AwsSessionFactory, mock_client: AsyncMock) -> None:
    mock_factory.client = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_client)))
    
    body = AsyncMock()
    async def mock_read(n):
        if not hasattr(mock_read, 'called'):
            mock_read.called = True
            return b"chunk1"
        return b""
    body.read = mock_read
    body.__aenter__ = AsyncMock(return_value=body)
    body.__aexit__ = AsyncMock()
    
    mock_client.get_object.return_value = {"Body": body}
    
    s3 = S3Client(mock_factory)
    chunks = []
    async with s3:
        async for chunk in s3.stream("bucket", "key", chunk_size=10):
            chunks.append(chunk)
    
    assert chunks == [b"chunk1"]


async def test_s3_stream_error_handling(mock_factory: AwsSessionFactory, mock_client: AsyncMock) -> None:
    mock_factory.client = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_client)))
    
    mock_client.get_object.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey"}},
        "GetObject"
    )
    
    s3 = S3Client(mock_factory)
    with pytest.raises(NonRetryableError, match="NoSuchKey"):
        async with s3:
            async for _ in s3.stream("bucket", "missing"):
                pass


async def test_s3_presigned_get_with_content_type(mock_factory: AwsSessionFactory, mock_client: AsyncMock) -> None:
    mock_factory.client = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_client)))
    mock_client.generate_presigned_url.return_value = "https://presigned.url"
    
    s3 = S3Client(mock_factory)
    async with s3:
        url = await s3.presigned_get("bucket", "doc~pdf", expires_s=600)
    
    assert url == "https://presigned.url"
    call_args = mock_client.generate_presigned_url.call_args
    assert call_args[0][0] == "get_object"
    params = call_args[1]["Params"]
    assert params["ResponseContentType"] == "application/pdf"
    assert "inline" in params["ResponseContentDisposition"]


async def test_s3_presigned_get_attachment_for_office(mock_factory: AwsSessionFactory, mock_client: AsyncMock) -> None:
    mock_factory.client = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_client)))
    mock_client.generate_presigned_url.return_value = "https://presigned.url"
    
    s3 = S3Client(mock_factory)
    async with s3:
        await s3.presigned_get("bucket", "file~docx", expires_s=600)
    
    params = mock_client.generate_presigned_url.call_args[1]["Params"]
    # docx is not in _INLINE_EXTENSIONS, so should be attachment
    assert "attachment" in params["ResponseContentDisposition"]
    assert "application/vnd.openxmlformats" in params["ResponseContentType"]


async def test_s3_presigned_get_handles_non_latin1_filenames(
    mock_factory: AwsSessionFactory, mock_client: AsyncMock
) -> None:
    """S3 rejects a presigned URL whose Content-Disposition isn't ISO-8859-1 encodable.

    Real document names carry en-dashes and curly quotes, which broke 4.3% of the
    corpus with 'InvalidArgument: Header value cannot be represented using ISO-8859-1'.
    """
    mock_factory.client = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_client))
    )
    mock_client.generate_presigned_url.return_value = "https://presigned.url"

    s3 = S3Client(mock_factory)
    async with s3:
        # en-dash + curly apostrophe: neither is encodable in Latin-1. Escapes, not
        # literals, so the test says exactly which codepoints it exercises.
        name = "abc123def456_Supplierco \u2013 Acme Food\u2019s~pdf"
        await s3.presigned_get("bucket", name)

    disp = mock_client.generate_presigned_url.call_args[1]["Params"]["ResponseContentDisposition"]
    disp.encode("iso-8859-1")  # the exact rule S3 enforces; raises => the URL 400s
    # Transliterated, not dropped: the dash survives as "-" and the quote as "'".
    assert 'filename="Supplierco - Acme Food\'s.pdf"' in disp
    # The true UTF-8 name is preserved for modern browsers via RFC 6266 filename*.
    assert "filename*=UTF-8''" in disp
    assert "%E2%80%93" in disp  # the en-dash, percent-encoded


def test_content_disposition_ascii_names_are_untouched() -> None:
    """The common case must not regress into percent-encoding noise."""
    disp = _content_disposition("inline", "Quarterly Report.pdf")
    assert disp.startswith('inline; filename="Quarterly Report.pdf"')
    disp.encode("iso-8859-1")


def test_ascii_fallback_never_returns_empty() -> None:
    """A name of only non-ASCII characters must still yield a usable filename."""
    assert _ascii_fallback("日本語") == "document"
    assert _ascii_fallback("") == "document"


async def test_s3_put_bytes_success(mock_factory: AwsSessionFactory, mock_client: AsyncMock) -> None:
    mock_factory.client = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_client)))
    mock_client.put_object.return_value = {}
    
    s3 = S3Client(mock_factory)
    async with s3:
        await s3.put_bytes("bucket", "key", b"data", content_type="text/plain")
    
    mock_client.put_object.assert_awaited_once()
    call_args = mock_client.put_object.call_args
    assert call_args[1]["Body"] == b"data"
    assert call_args[1]["ContentType"] == "text/plain"


async def test_s3_get_bytes_with_version(mock_factory: AwsSessionFactory, mock_client: AsyncMock) -> None:
    mock_factory.client = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_client)))
    
    body = AsyncMock()
    body.read.return_value = b"versioned content"
    body.__aenter__ = AsyncMock(return_value=body)
    body.__aexit__ = AsyncMock()
    mock_client.get_object.return_value = {"Body": body}
    
    s3 = S3Client(mock_factory)
    async with s3:
        result = await s3.get_bytes("bucket", "key", version_id="v123")
    
    assert result == b"versioned content"
    call_args = mock_client.get_object.call_args
    assert call_args[1]["VersionId"] == "v123"
