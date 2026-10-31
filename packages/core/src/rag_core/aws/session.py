"""Shared aioboto3 session + botocore configuration.

Centralising client construction means one place defines credentials resolution
(IAM role in prod, static keys + LocalStack endpoint in dev) and the retry/pool
tuning every client inherits. ``adaptive`` retry mode lets botocore client-side
throttle in response to 429s, complementing our application-level backoff.
"""

from __future__ import annotations

from typing import Any

import aioboto3
from botocore.config import Config as BotoConfig

from rag_core.config import AWSSettings


class AwsSessionFactory:
    """Builds aioboto3 client context managers with consistent config.

    Holds the credential/region settings and a tuned botocore ``Config`` reused
    by every service-specific client (S3/SQS/Bedrock).
    """

    def __init__(self, settings: AWSSettings, *, region_override: str | None = None) -> None:
        self._settings = settings
        self._region = region_override or settings.region
        self._session = aioboto3.Session()
        self._config = BotoConfig(
            region_name=self._region,
            retries={"max_attempts": settings.max_retries, "mode": "adaptive"},
            max_pool_connections=64,
            connect_timeout=10,
            read_timeout=60,
            tcp_keepalive=True,
        )

    def _common_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"config": self._config, "region_name": self._region}
        # Endpoint + static creds only when explicitly provided (LocalStack/dev).
        if self._settings.endpoint_url:
            kwargs["endpoint_url"] = self._settings.endpoint_url
        if self._settings.access_key_id and self._settings.secret_access_key:
            kwargs["aws_access_key_id"] = self._settings.access_key_id
            kwargs["aws_secret_access_key"] = self._settings.secret_access_key
        # TLS verification: a custom CA bundle (corporate proxy) or disable it.
        if self._settings.ca_bundle:
            kwargs["verify"] = self._settings.ca_bundle
        elif not self._settings.verify_ssl:
            kwargs["verify"] = False
        return kwargs

    def client(self, service: str, **overrides: Any) -> Any:
        """Return an aioboto3 client *context manager* for ``service``.

        Usage::

            async with factory.client("s3") as s3:
                ...
        """
        return self._session.client(service, **{**self._common_kwargs(), **overrides})

    @property
    def region(self) -> str:
        return self._region
