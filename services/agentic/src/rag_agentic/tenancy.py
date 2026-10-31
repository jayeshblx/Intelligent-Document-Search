"""Resolve a caller's base URL to the (environment, tenant) that owns its index.

This is the security boundary of the whole service: every retrieval call is scoped
to the prefix this module returns, so a wrong answer here reads another customer's
documents. It therefore resolves from *configuration only* — never from anything
the model produced, and never from a hint inside the user's question.

Resolution order (first match wins):
  1. ``tenant_map``     exact, normalised base URL -> "environment/tenant"
  2. ``tenant_pattern`` regex with named groups ``environment`` and ``tenant``
  3. ``default_*``      only if both are configured

An unresolvable base URL raises rather than falling back to a shared index — a
loud 4xx is always preferable to silently serving the wrong tenant's data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urlsplit

from rag_agentic.config import AgenticSettings

# Tenant/environment names become S3 path segments, so keep them boring: no dots,
# no slashes, no traversal. Anything else is a misconfiguration, not a request.
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class TenancyError(ValueError):
    """The base URL could not be mapped to exactly one tenant."""


@dataclass(frozen=True, slots=True)
class TenantContext:
    environment: str
    tenant: str
    base_url: str

    @property
    def key(self) -> str:
        """Cache key / log label."""
        return f"{self.environment}/{self.tenant}"

    def s3_prefix(self, root: str) -> str:
        """Prefix under which this tenant's index artifacts live."""
        root = root.strip("/")
        return f"{root}/{self.environment}/{self.tenant}" if root else self.key


def normalize_base_url(base_url: str) -> str:
    """Canonical form used for matching: no scheme, no port, no trailing slash, lowercase.

    So ``HTTPS://Dev.Example.io/acme/`` and ``dev.example.io/acme`` are one key.
    """
    raw = (base_url or "").strip()
    if not raw:
        raise TenancyError("base_url is required")
    # urlsplit only populates netloc when a scheme is present; add one if missing.
    parts = urlsplit(raw if "://" in raw else f"//{raw}", scheme="https")
    host = (parts.hostname or "").lower()
    if not host:
        raise TenancyError(f"base_url has no host: {base_url!r}")
    path = parts.path.rstrip("/")
    return f"{host}{path}"


def _validate(environment: str, tenant: str, base_url: str) -> TenantContext:
    for label, value in (("environment", environment), ("tenant", tenant)):
        if not _SEGMENT_RE.match(value or ""):
            raise TenancyError(
                f"resolved {label} {value!r} is not a valid path segment (from base_url {base_url!r})"
            )
    return TenantContext(environment=environment, tenant=tenant, base_url=base_url)


@lru_cache(maxsize=8)
def _compiled(pattern: str) -> re.Pattern[str]:
    compiled = re.compile(pattern, re.IGNORECASE)
    missing = {"environment", "tenant"} - set(compiled.groupindex)
    if missing:
        raise TenancyError(
            f"AGENTIC_TENANT_PATTERN must define named groups 'environment' and 'tenant'; "
            f"missing {sorted(missing)}"
        )
    return compiled


class TenantResolver:
    """Config-driven base_url -> TenantContext."""

    def __init__(self, settings: AgenticSettings) -> None:
        self._settings = settings
        # Normalise the map's keys once so lookups match however they were written.
        self._map = {normalize_base_url(k): v for k, v in settings.tenant_map.items()}

    def resolve(self, base_url: str) -> TenantContext:
        normalized = normalize_base_url(base_url)

        if mapped := self._map.get(normalized):
            environment, _, tenant = mapped.partition("/")
            if not tenant:
                raise TenancyError(
                    f"tenant_map entry for {normalized!r} must be 'environment/tenant', got {mapped!r}"
                )
            return _validate(environment, tenant, base_url)

        pattern = self._settings.tenant_pattern
        if pattern and (match := _compiled(pattern).search(normalized)):
            return _validate(
                match.group("environment") or "", match.group("tenant") or "", base_url
            )

        if self._settings.default_environment and self._settings.default_tenant:
            return _validate(
                self._settings.default_environment, self._settings.default_tenant, base_url
            )

        raise TenancyError(
            f"no tenant configured for base_url {base_url!r} (normalised {normalized!r}); "
            "set AGENTIC_TENANT_MAP or AGENTIC_TENANT_PATTERN"
        )
