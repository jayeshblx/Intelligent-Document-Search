"""The ``download_url`` on each citation -- the link ExampleApp renders for a source.

A URL is minted per request for exactly the tenant that request resolved to, from
that tenant's own snapshot manifest. Nothing in the request body or the model's
output can influence which bucket a link points into.

Two modes (``AGENTIC_DOCUMENT_URL_MODE``):

* ``presigned`` -- a time-limited S3 GET URL. The document opens straight from the
  browser with no extra infrastructure, but the URL carries the bucket, the key and
  an access-key id, and a URL signed with temporary credentials dies with them.
* ``template`` -- a ExampleApp-served URL such as ``{base_url}/documents/{document_id}``.
  ExampleApp authenticates the user and streams the file, so storage details never
  reach the browser. Needs a matching route on the ExampleApp side.

The search web app streams through ``/v1/files/<token>/...`` instead, but an
AgentCore runtime has no browsable HTTP surface, so a link it minted there would
point nowhere -- hence these two.

Failure to mint one link never fails the answer: that citation just has no URL.

A presigned URL must be signed for the bucket's own region -- signed for another,
S3 answers 400 ``AuthorizationQueryParametersError`` (verified against dev's
``example-docs-bucket`` in ap-south-1 while the service runs in us-east-1). The region
comes from the manifest's ``source_region``, and links are signed with a client
for that region.
"""

from __future__ import annotations

import string
from collections.abc import Awaitable, Callable
from typing import Any, Protocol
from urllib.parse import quote

from rag_agentic.config import AgenticSettings, DocumentUrlMode
from rag_agentic.tenancy import TenantContext
from rag_core.aws.s3 import S3Client, _download_name
from rag_core.aws.session import AwsSessionFactory
from rag_core.config import AWSSettings
from rag_core.logging import get_logger

_log = get_logger("rag_agentic.document_urls")

# (document_id, source_key) -> URL, or None when no link can be made.
UrlFor = Callable[[str, str], Awaitable[str | None]]

TEMPLATE_FIELDS = frozenset(
    {"base_url", "environment", "tenant", "document_id", "source_key", "filename"}
)


class Presigner(Protocol):
    """The slice of ``rag_core.aws.S3Client`` this module needs."""

    async def presigned_get(self, bucket: str, key: str, *, expires_s: int = 3600) -> str: ...


PresignerForRegion = Callable[[str], Presigner]


class RegionalPresigners:
    """One S3 client per region, created on first use and closed together."""

    def __init__(self, aws: AWSSettings) -> None:
        self._aws = aws
        self._clients: dict[str, S3Client] = {}

    def __call__(self, region: str) -> Presigner:
        if region not in self._clients:
            self._clients[region] = S3Client(AwsSessionFactory(self._aws, region_override=region))
        return self._clients[region]

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()


def validate_template(template: str) -> None:
    """Fail at startup, not on the first request, if the template is unusable."""
    if not template.strip():
        raise ValueError(
            "AGENTIC_DOCUMENT_URL_MODE=template needs AGENTIC_DOCUMENT_URL_TEMPLATE, "
            "e.g. '{base_url}/documents/{document_id}'"
        )
    used = {name for _, name, _, _ in string.Formatter().parse(template) if name is not None}
    if unknown := used - TEMPLATE_FIELDS:
        raise ValueError(
            f"AGENTIC_DOCUMENT_URL_TEMPLATE uses unknown placeholder(s) {sorted(unknown)}; "
            f"allowed: {sorted(TEMPLATE_FIELDS)}"
        )
    if not used & {"document_id", "source_key", "filename"}:
        raise ValueError(
            "AGENTIC_DOCUMENT_URL_TEMPLATE must identify the document -- include "
            "{document_id}, {source_key} or {filename}"
        )


class DocumentUrlBuilder:
    """Hands out a per-tenant ``UrlFor``."""

    def __init__(
        self,
        settings: AgenticSettings,
        s3: Presigner | None = None,
        *,
        presigner_for_region: PresignerForRegion | None = None,
    ) -> None:
        self._settings = settings
        # Used when the source region is unknown (the service's own region).
        self._s3 = s3
        self._presigner_for_region = presigner_for_region
        if settings.document_url_mode is DocumentUrlMode.TEMPLATE:
            validate_template(settings.document_url_template)
        elif settings.document_url_mode is DocumentUrlMode.PRESIGNED and s3 is None:
            raise ValueError("document_url_mode=presigned needs an S3 client to sign with")

    def for_tenant(
        self, context: TenantContext, manifest: dict[str, Any], base_url: str
    ) -> UrlFor | None:
        mode = self._settings.document_url_mode
        if mode is DocumentUrlMode.NONE:
            return None
        if mode is DocumentUrlMode.TEMPLATE:
            return self._template(context, base_url)
        return self._presigned(context, manifest)

    # ---- template -----------------------------------------------------------

    def _template(self, context: TenantContext, base_url: str) -> UrlFor:
        template = self._settings.document_url_template
        root = base_url.strip().rstrip("/")

        async def url_for(document_id: str, source_key: str) -> str | None:
            if not document_id:
                return None
            return template.format(
                base_url=root,
                environment=quote(context.environment, safe=""),
                tenant=quote(context.tenant, safe=""),
                document_id=quote(document_id, safe=""),
                source_key=quote(source_key or "", safe="/"),
                filename=quote(_download_name(source_key) if source_key else "", safe=""),
            )

        return url_for

    # ---- presigned ----------------------------------------------------------

    def _presigned(self, context: TenantContext, manifest: dict[str, Any]) -> UrlFor | None:
        # The manifest names the tenant's own source bucket; the setting is only a
        # fallback for snapshots published before manifests carried it.
        bucket = str(manifest.get("source_bucket") or self._settings.source_bucket or "")
        if not bucket:
            _log.warning(
                "document_urls_disabled_no_source_bucket",
                tenant=context.key,
                hint="publish a manifest with source_bucket or set AGENTIC_SOURCE_BUCKET",
            )
            return None
        prefix = str(manifest.get("source_prefix") or "")
        ttl = self._settings.document_url_ttl_s
        region = str(manifest.get("source_region") or self._settings.source_region or "")
        s3: Presigner | None = self._s3
        if region and self._presigner_for_region is not None:
            s3 = self._presigner_for_region(region)
        elif not region:
            _log.warning(
                "document_url_region_unknown",
                tenant=context.key,
                bucket=bucket,
                hint="links break if the bucket is in another region; publish a manifest "
                "with source_region or set AGENTIC_SOURCE_REGION",
            )
        assert s3 is not None  # enforced in __init__

        async def url_for(document_id: str, source_key: str) -> str | None:
            if not source_key:
                return None
            # Defence in depth: only sign keys under the prefix this tenant was
            # indexed from, so a corrupted row cannot become a link elsewhere.
            if prefix and not source_key.startswith(prefix):
                _log.warning(
                    "document_url_refused_outside_prefix",
                    tenant=context.key,
                    document_id=document_id,
                    prefix=prefix,
                )
                return None
            try:
                return await s3.presigned_get(bucket, source_key, expires_s=ttl)
            except Exception as exc:  # never fail the answer over one link
                _log.warning(
                    "document_url_presign_failed",
                    tenant=context.key,
                    document_id=document_id,
                    error=str(exc),
                )
                return None

        return url_for
