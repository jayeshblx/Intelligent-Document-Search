"""Base model conventions for all domain objects.

These objects cross process boundaries (SQS, Temporal, store payloads), so the
config below optimises for safe, forward-compatible (de)serialisation:

* ``extra="ignore"`` — a newer producer may add fields; older consumers must not
  crash. (We deliberately do NOT use ``forbid`` for wire contracts.)
* ``populate_by_name=True`` — accept both field name and alias on input.
* ``ser_json_timedelta="iso8601"`` and ISO datetimes — stable, portable JSON.
* ``validate_assignment=True`` — mutations during pipeline enrichment stay valid.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    """Timezone-aware UTC now. Centralised so every timestamp is consistent."""
    return datetime.now(UTC)


class DomainModel(BaseModel):
    """Common base for all domain models."""

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        validate_assignment=True,
        ser_json_timedelta="iso8601",
        use_enum_values=False,
    )


class TimestampedModel(DomainModel):
    """Domain model carrying creation/update audit timestamps."""

    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    def touch(self) -> None:
        """Update ``updated_at`` to now (call on mutation)."""
        self.updated_at = utcnow()
