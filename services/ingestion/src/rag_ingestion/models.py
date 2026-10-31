"""API DTOs for the ingestion service."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from rag_core.models.base import DomainModel


class Outcome(StrEnum):
    """Per-object decision taken during a sync."""

    NEW = "new"               # first time seen -> emitted
    CHANGED = "changed"       # content changed -> new version emitted
    UNCHANGED = "unchanged"   # ETag/checksum match -> skipped
    DUPLICATE = "duplicate"   # same bytes as an existing doc -> linked, not re-emitted
    SKIPPED = "skipped"       # filtered (too large / unsupported)
    ERROR = "error"           # processing failed for this object


class SyncRequest(DomainModel):
    """Trigger an incremental sync over an S3 prefix."""

    prefix: str | None = Field(
        default=None, description="S3 prefix to scan; defaults to configured S3 prefix."
    )
    namespace: str = Field(default="default", description="Target namespace/tenant.")
    dry_run: bool = Field(
        default=False, description="Detect changes but emit nothing (planning mode)."
    )


class SyncSummary(DomainModel):
    """Aggregate result of a sync run."""

    bucket: str
    prefix: str
    namespace: str
    dry_run: bool
    scanned: int = 0
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    duplicate: int = 0
    skipped: int = 0
    errors: int = 0
    emitted: int = 0
    duration_s: float = 0.0
    error_samples: list[str] = Field(default_factory=list)

    def record(self, outcome: Outcome) -> None:
        self.scanned += 1
        setattr(self, outcome.value, getattr(self, outcome.value) + 1)
