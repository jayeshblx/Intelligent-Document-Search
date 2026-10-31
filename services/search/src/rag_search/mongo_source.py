"""External MongoDB metadata source — the customer's system of record.

The compliance corpus in S3 (``foxtrot/<hash>_<name>~<ext>``) is mirrored in a MongoDB
``AttachmentMetaData`` collection whose ``OriginalFileName`` equals the S3 object's
basename. Each Mongo document carries the *authoritative* business facets
(BusinessObject, ObjectCode, Status, DocumentCategory/SubCategory, TenantCode,
CreatedBy, Tags, …) — far better than LLM inference, and free.

Tenancy: neither the S3 folder nor the Mongo database separates tenants -- in dev,
``FOXTROT_DEV.AttachmentMetaData`` is mostly ``TenantCode=ACMEFOODS`` and ``foxtrot/`` holds
their files too. ``TenantCode`` is the only reliable marker, so with ``tenant_codes``
set, other tenants' records never enrich anything and their files are reported as
foreign so the indexer skips them.

At index time we preload the collection once into a filename→doc map, then look up
each S3 object. On a hit, the Mongo fields become filterable facets (and the LLM
extraction step is skipped). ``pymongo`` is imported lazily so the dependency is
only required when the integration is actually configured.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from rag_core.logging import get_logger

if TYPE_CHECKING:
    from rag_core.models.metadata import DocumentMetadata

_log = get_logger("rag_search.mongo")

# Mongo field -> facet key (snake_case, filterable). category/sub_category overlap
# the LLM-extracted facets on purpose: when present, Mongo wins (it's authoritative).
MONGO_FACET_MAP: dict[str, str] = {
    "BusinessObject": "business_object",
    "ObjectName": "object_name",
    "ObjectCode": "object_code",
    "Status": "status",
    "DocumentCategory": "category",
    "DocumentSubCategory": "sub_category",
    "TenantCode": "tenant",
    "CreatedBy": "created_by",
    "MetaDataName": "meta_data_name",
}
# Fields projected out of Mongo (the mapped ones + join key + tags/dates for enrichment).
# ``DeleteMark`` is the system-of-record soft-delete flag (1 = deleted): such documents
# must be excluded from the index and from search.
_PROJECTION_FIELDS = [
    *MONGO_FACET_MAP.keys(),
    "Tags", "DateCreated", "LastModifiedDate", "OriginalFileName", "GeneratedFileName",
    "DeleteMark", "MetaData",
]

# Record-level fields repeated at the top of the metadata text, so a match on a
# MetaData value arrives with what the record IS. (Label, Mongo field.)
_RECORD_TEXT_FIELDS = (
    ("Business object", "BusinessObject"),
    ("Object name", "ObjectName"),
    ("Object code", "ObjectCode"),
    ("Status", "Status"),
    ("Category", "DocumentCategory"),
    ("Sub-category", "DocumentSubCategory"),
    ("Metadata form", "MetaDataName"),
)
_MAX_METADATA_VALUE_CHARS = 2000
_MAX_METADATA_TEXT_CHARS = 8000


def basename(s3_key: str) -> str:
    """The S3 object name used to join against Mongo ``OriginalFileName``."""
    return s3_key.rsplit("/", 1)[-1]


def is_deleted(doc: dict[str, Any]) -> bool:
    """True when the Mongo record is soft-deleted (``DeleteMark`` set).

    The flag arrives in mixed shapes across drivers/exports (int 1, bool True, the
    string "1"/"true"), so normalise rather than compare to a single literal.
    """
    val = doc.get("DeleteMark")
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return int(val) == 1
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes"}
    return False


def parse_tenant_codes(value: str | Iterable[str] | None) -> frozenset[str]:
    """``"FOXTROT, ACMEFOODS"`` or an iterable -> normalised codes. Empty = no filter."""
    if value is None:
        return frozenset()
    parts = value.split(",") if isinstance(value, str) else list(value)
    return frozenset(p.strip().upper() for p in parts if p and p.strip())


def facets_from_mongo(doc: dict[str, Any]) -> dict[str, Any]:
    """Extract filterable facets from a Mongo attachment document."""
    out: dict[str, Any] = {}
    for field, facet_key in MONGO_FACET_MAP.items():
        val = doc.get(field)
        if isinstance(val, str) and val.strip():
            out[facet_key] = val.strip()
    tags = doc.get("Tags")
    if isinstance(tags, list):
        clean = [str(t).strip() for t in tags if str(t).strip()]
        if clean:
            out["tags"] = clean
    return out


def attachment_metadata_text(doc: dict[str, Any]) -> str:
    """The record's ``MetaData`` field values as searchable text; "" when it has none.

    ``MetaData`` is a list of ``{FieldName, DisplayName, Value}`` -- e.g. a complaint's
    ``ComplaintDescription`` or ``Complaint Number`` -- that exists only in the
    catalogue, not in the file, so without this it is never searchable. Both names
    are kept when they differ (``Complaint Number (ObjectCode): STD1647``), so a
    question can use either.
    """
    fields = doc.get("MetaData")
    lines: list[str] = []
    for item in fields if isinstance(fields, list) else []:
        if not isinstance(item, dict):
            continue
        value = item.get("Value")
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value if v not in (None, ""))
        text = "" if value is None else str(value).strip()
        if not text:
            continue
        display = str(item.get("DisplayName") or "").strip()
        name = str(item.get("FieldName") or "").strip()
        label = display or name
        if display and name and display.replace(" ", "").lower() != name.lower():
            label = f"{display} ({name})"
        if label:
            lines.append(f"{label}: {text[:_MAX_METADATA_VALUE_CHARS]}")
    if not lines:
        return ""

    header = ["Attachment metadata"]
    if isinstance(file_name := doc.get("OriginalFileName"), str) and file_name.strip():
        header[0] += f" for {file_name.strip()}"
    for label, field in _RECORD_TEXT_FIELDS:
        if isinstance(val := doc.get(field), str) and val.strip():
            header.append(f"{label}: {val.strip()}")
    return "\n".join([*header, *lines])[:_MAX_METADATA_TEXT_CHARS]


def apply_mongo_to_metadata(doc: dict[str, Any], meta: DocumentMetadata) -> None:
    """Fold Mongo fields into the canonical ``DocumentMetadata`` (authoritative)."""
    if (cat := doc.get("DocumentCategory")) and isinstance(cat, str) and cat.strip():
        meta.category = cat.strip()
    if (sub := doc.get("DocumentSubCategory")) and isinstance(sub, str) and sub.strip():
        meta.sub_category = sub.strip()
    if (author := doc.get("CreatedBy")) and isinstance(author, str) and author.strip():
        meta.author = author.strip()
    if (tenant := doc.get("TenantCode")) and isinstance(tenant, str) and tenant.strip():
        meta.business_unit = meta.business_unit or tenant.strip()
    tags = doc.get("Tags")
    if isinstance(tags, list) and tags and not meta.keywords:
        meta.keywords = [str(t).strip() for t in tags if str(t).strip()][:15]
    # Keep every mapped Mongo value on the canonical record for /v1/metadata.
    for field, facet_key in MONGO_FACET_MAP.items():
        val = doc.get(field)
        if isinstance(val, str) and val.strip():
            meta.extra[facet_key] = val.strip()


class MongoMetadataSource:
    """Loads attachment metadata from MongoDB and joins it to S3 objects by filename."""

    def __init__(
        self,
        uri: str,
        *,
        database: str,
        collection: str,
        join_field: str = "GeneratedFileName,OriginalFileName",
        tls_ca_file: str | None = None,
        tenant_codes: str | Iterable[str] | None = None,
    ) -> None:
        self._uri = uri
        # Case-insensitive: AF_DEV holds both "AF" and "af".
        self._tenant_codes = parse_tenant_codes(tenant_codes)
        self._database = database
        self._collection = collection
        # S3 basenames match GeneratedFileName (sometimes OriginalFileName); index by
        # every configured field so the join is robust to the mixed naming.
        self._join_fields = [f.strip() for f in join_field.split(",") if f.strip()]
        self._tls_ca_file = tls_ca_file
        self._client: Any = None
        self._cache: dict[str, dict[str, Any]] = {}
        # Join-key values (GeneratedFileName/OriginalFileName) of soft-deleted records.
        # Kept out of ``_cache`` so they never enrich, and consulted to skip/purge.
        self._deleted: set[str] = set()
        # Join-key values whose ONLY records belong to other tenants.
        self._foreign: set[str] = set()
        self._loaded = False

    def _connect(self) -> Any:
        from pymongo import MongoClient  # lazy: only needed when Mongo is configured

        kwargs: dict[str, Any] = {"serverSelectionTimeoutMS": 8000}
        if self._tls_ca_file:
            kwargs["tlsCAFile"] = self._tls_ca_file
        if self._client is None:
            self._client = MongoClient(self._uri, **kwargs)
        return self._client

    def seed(self, mapping: dict[str, dict[str, Any]]) -> None:
        """Inject a filename→doc map directly (tests) — skips the network preload.

        Classified exactly as the real preload: soft-deleted records go to the deleted
        set, other tenants' records to the foreign set."""
        self._classify((doc, [key]) for key, doc in mapping.items())
        self._loaded = True

    async def preload(self) -> int:
        """Load the whole collection into a filename→doc map (once per run)."""
        if self._loaded:
            return len(self._cache)
        count = await asyncio.to_thread(self._preload_sync)
        self._loaded = True
        _log.info(
            "mongo_metadata_loaded",
            documents=count,
            deleted=len(self._deleted),
            other_tenants=len(self._foreign),
            tenant_codes=sorted(self._tenant_codes),
            database=self._database,
            collection=self._collection,
        )
        return count

    def _preload_sync(self) -> int:
        self._classify(self._records_sync({}))
        return len(self._cache)

    def _find_sync(self, query: dict[str, Any]) -> Iterable[dict[str, Any]]:
        """Raw records matching ``query`` (the projected fields only)."""
        coll = self._connect()[self._database][self._collection]
        projection = dict.fromkeys([*_PROJECTION_FIELDS, *self._join_fields], 1)
        return coll.find(query, projection)

    def _records_sync(
        self, query: dict[str, Any]
    ) -> Iterable[tuple[dict[str, Any], list[str]]]:
        for doc in self._find_sync(query):
            keys = [doc.get(f) for f in self._join_fields]
            yield doc, [k for k in keys if isinstance(k, str) and k]

    def _by_names_query(self, names: list[str]) -> dict[str, Any]:
        return {"$or": [{f: {"$in": names}} for f in self._join_fields]}

    async def load_keys(self, s3_keys: Iterable[str], *, batch: int = 500) -> int:
        """Load only the records for these S3 objects -- what an incremental run needs.

        Replaces the whole-collection :meth:`preload` (FOXTROT_DEV alone is 35k records) with
        indexed ``$in`` lookups on the join fields. Classified exactly like a preload, so
        ``lookup``/``is_deleted``/``is_foreign`` answer the same for these keys.
        """
        names = sorted({basename(k) for k in s3_keys if k})

        def _load() -> list[tuple[dict[str, Any], list[str]]]:
            rows: list[tuple[dict[str, Any], list[str]]] = []
            for start in range(0, len(names), batch):
                rows.extend(self._records_sync(self._by_names_query(names[start : start + batch])))
            return rows

        self._classify(await asyncio.to_thread(_load) if names else [])
        self._loaded = False  # a partial load: a later preload() must still read everything
        return len(self._cache)

    async def deleted_names(self) -> set[str]:
        """Join-key values (S3 basenames) this tenant has soft-deleted (``DeleteMark``).

        Reads only the soft-deleted records, then re-classifies every record naming
        those files, so a deletion is judged exactly as at index time (another tenant's
        DeleteMark never counts as ours).
        """
        marks: list[Any] = [1, True, "1", "true", "True", "TRUE", "yes"]

        def _names() -> set[str]:
            found: set[str] = set()
            for doc, keys in self._records_sync({"DeleteMark": {"$in": marks}}):
                if self.belongs(doc):
                    found.update(keys)
            return found

        names = await asyncio.to_thread(_names)
        await self.load_keys(names)
        return {n for n in names if n in self._deleted}

    def belongs(self, doc: dict[str, Any]) -> bool:
        """True when the record is this index's tenant (always, with no filter)."""
        if not self._tenant_codes:
            return True
        code = doc.get("TenantCode")
        return isinstance(code, str) and code.strip().upper() in self._tenant_codes

    def _classify(self, records: Iterable[tuple[dict[str, Any], list[str]]]) -> None:
        cache: dict[str, dict[str, Any]] = {}
        deleted: set[str] = set()
        foreign: set[str] = set()
        for doc, keys in records:
            if not self.belongs(doc):
                # Another tenant's record: never enriches, and its deletion is not ours.
                foreign.update(keys)
                continue
            if is_deleted(doc):
                # Soft-deleted: never enrich, and mark for skip/purge at index time.
                deleted.update(keys)
                for k in keys:
                    cache.pop(k, None)
                continue
            for key in keys:  # index under every join field
                cache[key] = doc  # last write wins on duplicate filenames
        self._cache = cache
        self._deleted = deleted
        # A file this tenant also catalogues is this tenant's, whatever else names it.
        self._foreign = foreign - cache.keys() - deleted

    def lookup(self, s3_key: str) -> dict[str, Any] | None:
        """Return the Mongo document for an S3 object, or None if unmatched."""
        return self._cache.get(basename(s3_key))

    def is_deleted(self, s3_key: str) -> bool:
        """True when the S3 object is soft-deleted in the system of record."""
        return basename(s3_key) in self._deleted

    def is_foreign(self, s3_key: str) -> bool:
        """True when the S3 object is catalogued only by OTHER tenants."""
        return basename(s3_key) in self._foreign

    @property
    def filters_tenants(self) -> bool:
        return bool(self._tenant_codes)

    @property
    def loaded_count(self) -> int:
        return len(self._cache)

    @property
    def deleted_count(self) -> int:
        return len(self._deleted)


    async def aclose(self) -> None:
        if self._client is not None:
            await asyncio.to_thread(self._client.close)
            self._client = None
