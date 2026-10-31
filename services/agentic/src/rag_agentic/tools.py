"""Retrieval tools for the document-search agent, scoped to one tenant.

Two invariants hold for every tool in here, and both are enforced in code rather
than asked of the model:

* **Tenant scope** -- the stores come from an already-resolved ``TenantIndex``.
  There is no tool parameter that can reach another tenant's data.
* **BusinessObject pre-filter** -- when the caller supplied one, it is merged into
  the filter set of every query *before* ranking, and a model-supplied value can
  never widen or replace it. The model is told the filter is active (see the
  system prompt) so it can explain the scope, not so it can choose it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from rag_search.facet_filters import FacetFilters, split_multi
from rag_search.retrieval import SearchService

from rag_agentic.agent_loop import ToolError
from rag_agentic.converse import ToolSpec
from rag_agentic.document_urls import UrlFor
from rag_agentic.index_store import TenantIndex
from rag_agentic.models import Citation, DocumentSummary
from rag_core.logging import get_logger

_log = get_logger("rag_agentic.tools")

# Facets the model may filter on beyond the enforced business_object. Restricting
# the set keeps it from inventing column names that silently match nothing.
FILTERABLE_FACETS = (
    "status", "object_code", "object_name", "category", "sub_category", "supplier",
    # Who uploaded the record. Values are usernames or email addresses and matching is
    # exact, so "documents created by Riley" only works once the model has resolved the
    # name to the stored value via list_facet_values.
    "created_by",
)


# Facets whose values are people or organisations, where a user names a part of
# the value ("Alex", "ExampleCorp") rather than the whole of it.
PERSON_FACETS = ("created_by", "supplier")

# Largest ACL applied in a single query. Measured safe well above this -- Atlas did
# 5,000 codes in 0.2s and the file backends 20,000 -- but SQLite has a hard ceiling
# (SQLITE_MAX_VARIABLE_NUMBER, 32,766) and a Lucene query has a clause limit, so a
# big ACL is split rather than left to find the edge in production.
#
# Splitting is safe because every batch runs the SAME query: vector scores are
# cosine against one embedding and keyword scores come from one index, so results
# merge by score exactly as if the filter had been applied once.
MAX_FILTER_VALUES = 1000


def _batched(values: list[str], size: int = MAX_FILTER_VALUES) -> list[list[str]]:
    """Split a filter's values into query-sized batches (one batch if it fits)."""
    return [values[i : i + size] for i in range(0, len(values), size)] or [[]]


def _display_name(title: str | None, source_key: str) -> str:
    """What to call a document in a citation: its title, else its filename."""
    return (title or "").strip() or (source_key or "").rsplit("/", 1)[-1]

_SEARCH = ToolSpec(
    name="search_documents",
    description=(
        "Hybrid semantic + keyword search over this tenant's documents. Use this for any "
        "question about what documents SAY. Returns ranked passages with the document name, "
        "page number and metadata facets. ONE well-chosen query normally suffices -- it "
        "already blends semantic and keyword matching, so rewording the same question "
        "returns substantially the same passages at the cost of a slow extra round trip. "
        "Search again only to cover a genuinely DIFFERENT aspect of the question."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language search query. Use the user's own wording plus "
                "any synonyms that would appear in the document text.",
            },
            "top_k": {
                "type": "integer",
                "description": "How many passages to return (1-25). Default 15, which is "
                "usually enough -- prefer raising this once over searching twice.",
                "minimum": 1,
                "maximum": 25,
            },
            "status": {"type": "string", "description": "Optional status facet, e.g. Active or Draft."},
            "object_code": {"type": "string", "description": "Optional object-code facet."},
            "supplier": {"type": "string", "description": "Optional supplier facet."},
            "created_by": {
                "type": "string",
                "description": "Optional creator facet. Values are usernames or email "
                "addresses and must match exactly, so call list_facet_values first to turn "
                "a person's name into the stored value.",
            },
        },
        "required": ["query"],
    },
)

_LIST = ToolSpec(
    name="list_documents",
    description=(
        "List/count documents by their metadata, without searching their text. Use this for "
        "questions like 'how many...', 'which documents are...', 'list all...', or 'documents "
        "created by <person>'. Returns document names with their facets, plus the total "
        "number of matches."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "status": {"type": "string", "description": "Filter by status, e.g. Active."},
            "object_code": {"type": "string", "description": "Filter by object code."},
            "category": {"type": "string", "description": "Filter by document category."},
            "supplier": {"type": "string", "description": "Filter by supplier."},
            "created_by": {
                "type": "string",
                "description": "Filter by who created the document. Values are usernames or "
                "email addresses and must match exactly, so call list_facet_values first to "
                "turn a person's name into the stored value.",
            },
            "limit": {
                "type": "integer", "minimum": 1, "maximum": 100,
                "description": "How many rows to return to YOU. Default 25. The full "
                "result set is returned to the caller automatically, so do not raise this "
                "just to enumerate everything -- report total_matching instead.",
            },
            "offset": {
                "type": "integer", "minimum": 0,
                "description": "Skip this many rows. Only needed to inspect later rows of "
                "a large result set; the caller already receives all of them.",
            },
        },
    },
)

_FACETS = ToolSpec(
    name="list_facet_values",
    description=(
        "The valid values available for each metadata facet in this tenant (status, "
        "object_code, supplier, category, ...). Call this FIRST when you are unsure what a "
        "filter value should be, so you filter on a value that actually exists."
    ),
    input_schema={"type": "object", "properties": {}},
)

_DETAIL = ToolSpec(
    name="get_document_details",
    description=(
        "Full metadata and a longer extract for one document, by its document_id from an "
        "earlier search result. Use it when a passage looks relevant but is too short to "
        "answer the question."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "document_id": {"type": "string", "description": "document_id from a search result."}
        },
        "required": ["document_id"],
    },
)


class DocumentSearchTools:
    """Tenant-scoped retrieval toolbox.

    Collects the documents its tools return so the agent can emit real citations
    without re-parsing the model's prose.

    Document URLs are never shown to the model: they are long, the model would
    only mangle or truncate them in its answer, and a presigned URL does not belong
    in a conversation log. They are attached to the final citations instead.
    """

    def __init__(
        self,
        index: TenantIndex,
        search: SearchService,
        *,
        business_object: str | Sequence[str] | None = None,
        tenant_codes: Sequence[str] | None = None,
        object_codes: str | Sequence[str] | None = None,
        top_k: int = 15,
        url_for: UrlFor | None = None,
        max_citations: int = 25,
        max_list_documents: int = 1000,
    ) -> None:
        self._index = index
        self._search = search
        # A caller can be scoped to several business objects at once; they are
        # combined with OR, and an empty list means the whole tenant is in scope.
        self._business_objects = split_multi(business_object)
        # Second, independent barrier. Snapshots are already physically separate --
        # one S3 prefix per tenant -- so this only matters if an index was built
        # with a broken tenant filter. That has happened once (ACMEFOODS documents
        # in FOXTROT's first index), which is why it is enforced rather than assumed.
        self._tenant_codes = [str(c).strip() for c in (tenant_codes or []) if str(c).strip()]
        # The documents ExampleApp's own permissions allow this user to see. Empty
        # means no per-user restriction was supplied.
        self._object_codes = split_multi(object_codes)
        self._top_k = top_k
        self._url_for = url_for
        self._max_citations = max_citations
        # How many rows of a listing travel back in the envelope. The model still
        # sees only its own `limit`; this is what the caller renders.
        self._max_list_documents = max_list_documents
        # The last listing the agent ran: (total_matching, every row up to the cap).
        self._listing: tuple[int, list[DocumentSummary]] | None = None
        # document_id -> citation from a search passage, best-scoring occurrence wins.
        self._seen: dict[str, Citation] = {}
        # document_id -> citation from a listing, in the order it was listed.
        self._listed: dict[str, Citation] = {}
        # document_id -> storage key, which is what a link is minted from.
        self._source_keys: dict[str, str] = {}

    @property
    def business_object(self) -> str | None:
        """The enforced scope, as it goes back on the wire (comma-separated)."""
        return ",".join(self._business_objects) or None

    @property
    def business_objects(self) -> list[str]:
        return list(self._business_objects)

    @property
    def listing(self) -> tuple[int, list[DocumentSummary]] | None:
        """The last listing's (total, rows), or None if nothing was listed."""
        return self._listing

    @property
    def namespace(self) -> str:
        return self._index.namespace

    def specs(self) -> list[ToolSpec]:
        return [_SEARCH, _LIST, _FACETS, _DETAIL]

    def citations(self, limit: int = 8) -> list[Citation]:
        """Sources touched during the turn, without URLs.

        Search-grounded sources first, best score first, capped at ``limit``; then any
        documents a listing returned, so a "list all ..." answer cites what it named.
        The whole list is capped at ``max_citations``.
        """
        ranked = sorted(self._seen.values(), key=lambda c: c.score, reverse=True)[:limit]
        cited = {c.document_id for c in ranked}
        listed = [c for doc_id, c in self._listed.items() if doc_id not in cited]
        return (ranked + listed)[: self._max_citations]

    async def citations_with_urls(self, limit: int = 8) -> list[Citation]:
        """``citations()`` with each ``download_url`` filled in.

        Only the citations actually returned are signed -- not every passage every
        tool call touched.
        """
        chosen = self.citations(limit)
        if self._url_for is None or not chosen:
            return chosen
        url_for = self._url_for
        urls = await asyncio.gather(
            *(url_for(c.document_id, self._source_keys.get(c.document_id, "")) for c in chosen)
        )
        return [
            c.model_copy(update={"download_url": url}) if url else c
            for c, url in zip(chosen, urls, strict=True)
        ]

    # ---- filter enforcement -------------------------------------------------

    async def _resolve_person_filters(self, filters: FacetFilters) -> dict[str, list[str]]:
        """Turn a part of a name into the stored values it matches.

        People are not stored the way users name them: asking for documents
        "created by Alex" has to become ``created_by IN (UserAdminAlex, UADMINALEX)``.
        Left to the model this is guesswork, and it guessed differently on
        identical requests -- one run answered 18 documents, the next 14, with the
        correct answer being 14. Resolving it here makes the same question give
        the same answer, and reports WHICH values matched so the answer can say so.
        """
        resolved: dict[str, list[str]] = {}
        for key in PERSON_FACETS:
            wanted = filters.get(key)
            if not isinstance(wanted, str) or not wanted.strip():
                continue
            needle = wanted.strip().lower()
            known = await self._index.metadata.facet_values(key, self.namespace)
            exact = [v for v in known if v.lower() == needle]
            matches = exact or [v for v in known if needle in v.lower()]
            if matches:
                filters[key] = matches
                resolved[key] = matches
            # No match: leave the value as given. The query then returns nothing,
            # which is the honest answer -- better than silently widening it.
        return resolved

    def _acl_batches(self, filters: FacetFilters) -> list[FacetFilters]:
        """``filters`` split so no single query carries too many ACL values.

        Only ``object_code`` is split: it is the one filter sized by a user's
        permissions rather than by the question, so it is the only one that can
        reach thousands of values.
        """
        codes = filters.get("object_code")
        if not isinstance(codes, list) or len(codes) <= MAX_FILTER_VALUES:
            return [filters]
        return [{**filters, "object_code": batch} for batch in _batched(codes)]

    async def _search_batched(
        self, query: str, filters: FacetFilters, top_k: int
    ) -> Any:
        """Search once per ACL batch and keep the globally best passages.

        Every batch embeds the same query, so the scores are directly comparable
        and merging by score gives what a single unsplit query would have.
        """
        batches = self._acl_batches(filters)
        if len(batches) == 1:
            return await self._search.search(
                query, namespace=self.namespace, top_k=top_k, filters=batches[0]
            )

        _log.info("acl_split_across_queries", batches=len(batches),
                  values=len(filters.get("object_code") or []))
        results = await asyncio.gather(*(
            self._search.search(query, namespace=self.namespace, top_k=top_k, filters=f)
            for f in batches
        ))
        merged: dict[str, Any] = {}
        for response in results:
            for passage in response.results:
                best = merged.get(passage.chunk_id)
                if best is None or passage.score > best.score:
                    merged[passage.chunk_id] = passage
        combined = sorted(merged.values(), key=lambda p: p.score, reverse=True)[:top_k]
        first = results[0]
        return first.model_copy(update={"results": combined})

    async def _list_batched(self, filters: FacetFilters) -> tuple[int, list[dict[str, Any]]]:
        """List once per ACL batch and union the documents.

        A document carries one object_code, so batches do not overlap -- but the
        rows are de-duplicated by id anyway rather than relying on that.
        """
        batches = self._acl_batches(filters)
        if len(batches) == 1:
            return await self._index.metadata.list_documents(
                self.namespace, filters=batches[0], limit=self._max_list_documents
            )

        _log.info("acl_split_across_queries", batches=len(batches),
                  values=len(filters.get("object_code") or []))
        pages = await asyncio.gather(*(
            self._index.metadata.list_documents(
                self.namespace, filters=f, limit=self._max_list_documents
            )
            for f in batches
        ))
        seen: dict[str, dict[str, Any]] = {}
        for _, rows in pages:
            for row in rows:
                seen.setdefault(str(row.get("document_id") or ""), row)
        seen.pop("", None)
        return len(seen), list(seen.values())

    def _filters(self, arguments: dict[str, Any]) -> FacetFilters:
        """Model-chosen facets, with the caller's BusinessObject forced on top.

        Assembled in this order on purpose: the enforced value is written last, so
        a model that passes ``business_object`` itself cannot override the scope.
        """
        filters: FacetFilters = {
            facet: str(arguments[facet]).strip()
            for facet in FILTERABLE_FACETS
            if arguments.get(facet) not in (None, "")
        }
        if self._business_objects:
            filters["business_object"] = list(self._business_objects)
        # Written last, like business_object: the model cannot widen or replace it.
        if self._tenant_codes:
            filters["tenant"] = list(self._tenant_codes)
        # The user's ACL, decided by ExampleApp. Overwrites any object_code the model
        # chose, so it can only ever narrow within what the user may see.
        if self._object_codes:
            filters["object_code"] = list(self._object_codes)
        return filters

    # ---- dispatch -----------------------------------------------------------

    async def invoke(self, name: str, arguments: dict[str, Any]) -> tuple[Any, str]:
        match name:
            case _SEARCH.name:
                return await self._do_search(arguments)
            case _LIST.name:
                return await self._do_list(arguments)
            case _FACETS.name:
                return await self._do_facets()
            case _DETAIL.name:
                return await self._do_detail(arguments)
            case _:
                raise ToolError(f"unknown tool {name!r}")

    async def _do_search(self, arguments: dict[str, Any]) -> tuple[Any, str]:
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ToolError("query is required and must not be empty")

        # A listing followed by a search is a content question that happened to
        # count something on the way past -- the answer is not that list, so it
        # must not ride back with the response. (Without this, asking what a
        # manual says returns a few hundred KB of unrelated document rows.)
        self._listing = None

        top_k = int(arguments.get("top_k") or self._top_k)
        filters = self._filters(arguments)
        resolved = await self._resolve_person_filters(filters)
        wanted = max(1, min(top_k, 25))
        response = await self._search_batched(query, filters, wanted)

        # Label passages with what each document IS, so the model can distinguish
        # two similarly-worded passages from different documents.
        document_ids = [p.document_id for p in response.results if p.document_id]
        described = await self._index.metadata.describe(document_ids, self.namespace)

        rows: list[dict[str, Any]] = []
        for passage in response.results:
            info = described.get(passage.document_id) or {}
            facets = dict(info.get("facets") or {})
            name = _display_name(info.get("title"), passage.source_key)
            rows.append(
                {
                    "document_id": passage.document_id,
                    "document_name": name,
                    "page": passage.page_start,
                    "score": round(passage.score, 4),
                    "matched_by": passage.matched_by,
                    "text": passage.text,
                    "facets": facets,
                }
            )
            self._remember(passage, name, facets)

        payload = {
            "query": query,
            "resolved_filters": resolved,
            "business_object_filter": self.business_object,
            "passages": rows,
            "count": len(rows),
        }
        return payload, f"{len(rows)} passage(s) for {query!r}"

    async def _do_list(self, arguments: dict[str, Any]) -> tuple[Any, str]:
        """List documents by metadata.

        Fetches the whole result set once (capped), then hands the model only the
        window it asked for. The model reasons over a readable slice while the
        caller gets every row in the response envelope -- so "594 documents" can be
        rendered as 594 rows without the model writing 594 names into its answer.
        """
        limit = max(1, min(int(arguments.get("limit") or 25), 100))
        offset = max(0, int(arguments.get("offset") or 0))
        filters = self._filters(arguments)
        resolved = await self._resolve_person_filters(filters)
        total, rows = await self._list_batched(filters)

        wanted = (*FILTERABLE_FACETS, "business_object")
        everything: list[DocumentSummary] = []
        for row in rows:
            facets = dict(row.get("facets") or {})
            document_id = row.get("document_id", "")
            name = _display_name(row.get("title"), row.get("source_key", ""))
            everything.append(
                DocumentSummary(
                    document_id=document_id,
                    document_name=name,
                    expiry_date=row.get("expiry_date"),
                    facets={f: facets[f] for f in wanted if facets.get(f)},
                )
            )
            if document_id and document_id not in self._source_keys and row.get("source_key"):
                self._source_keys[document_id] = row["source_key"]
        self._listing = (total, everything)

        window = everything[offset : offset + limit]
        # Only rows the model actually saw become citations, so a citation always
        # corresponds to something the answer could have been written from.
        for doc in window:
            if doc.document_id and doc.document_id not in self._listed:
                self._listed[doc.document_id] = Citation(
                    document_id=doc.document_id,
                    document_name=doc.document_name,
                    facets=doc.facets,
                )

        payload = {
            "total_matching": total,
            "returned": len(window),
            "offset": offset,
            "business_object_filter": self.business_object,
            # What a partial name was taken to mean, so the answer can say it.
            "resolved_filters": resolved,
            # The model does not need to enumerate these; they reach the caller anyway.
            "all_rows_returned_to_caller": len(everything),
            "documents": [
                {
                    "document_id": d.document_id,
                    "document_name": d.document_name,
                    "expiry_date": d.expiry_date,
                    **d.facets,
                }
                for d in window
            ],
        }
        return payload, f"{total} document(s) matched, {len(window)} shown"

    async def _do_facets(self) -> tuple[Any, str]:
        # Cascade off the enforced filter so the model only ever sees values that
        # actually co-occur with the caller's BusinessObject.
        filters: FacetFilters = {}
        if self._business_objects:
            filters["business_object"] = list(self._business_objects)
        if self._tenant_codes:
            filters["tenant"] = list(self._tenant_codes)
        if self._object_codes:
            filters["object_code"] = list(self._object_codes)
        facets = await self._index.metadata.facets(self.namespace, filters=filters or None)
        # Only what the model may actually filter on. `tenant` in particular must
        # never appear: the cascade deliberately does not narrow a facet by its own
        # selection (so a UI dropdown stays switchable), which would otherwise
        # advertise every other tenant's code in this environment.
        offered = (*FILTERABLE_FACETS, "business_object")
        trimmed = {k: v[:40] for k, v in facets.items() if v and k in offered}
        # The cascade never narrows a facet by its OWN selection (so a UI dropdown
        # stays switchable), which would otherwise list every object code in the
        # tenant -- including the ones this user's ACL excluded. Intersect it.
        if self._object_codes:
            permitted = set(self._object_codes)
            trimmed["object_code"] = [c for c in trimmed.get("object_code", []) if c in permitted]
        return (
            {"facets": trimmed, "business_object_filter": self.business_object},
            f"{len(trimmed)} facet(s)",
        )

    async def _do_detail(self, arguments: dict[str, Any]) -> tuple[Any, str]:
        document_id = str(arguments.get("document_id", "")).strip()
        if not document_id:
            raise ToolError("document_id is required")

        described = await self._index.metadata.describe([document_id], self.namespace)
        info = described.get(document_id)
        if info is None:
            raise ToolError(
                f"no document {document_id!r} in this tenant -- use a document_id from a "
                "search_documents result"
            )
        facets = dict(info.get("facets") or {})
        # Guard the tenant boundary a second time: a document that does not carry
        # the enforced BusinessObject is out of scope even if its id was guessed.
        if self._tenant_codes and facets.get("tenant", "") not in self._tenant_codes:
            # Should be unreachable: a tenant's snapshot holds only its own documents.
            # If it fires, the index is wrong -- refuse rather than serve it.
            _log.error(
                "cross_tenant_document_refused",
                document_id=document_id,
                found=facets.get("tenant", ""),
                expected=",".join(self._tenant_codes),
            )
            raise ToolError(f"document {document_id!r} does not belong to this tenant")
        if self._object_codes and facets.get("object_code", "") not in self._object_codes:
            raise ToolError(
                f"document {document_id!r} is outside the documents this user may see"
            )
        if self._business_objects and facets.get("business_object", "") not in self._business_objects:
            raise ToolError(
                f"document {document_id!r} is outside the {self.business_object!r} business object"
            )

        text = await self._search.expand_document(document_id, "", cap=6000)
        name = _display_name(info.get("title"), "")
        return (
            {"document_id": document_id, "document_name": name, "facets": facets, "text": text},
            f"details for {name or document_id}",
        )

    def _remember(self, passage: Any, name: str, facets: dict[str, str]) -> None:
        if passage.source_key:
            self._source_keys[passage.document_id] = passage.source_key
        existing = self._seen.get(passage.document_id)
        if existing is not None and existing.score >= passage.score:
            return
        self._seen[passage.document_id] = Citation(
            document_id=passage.document_id,
            document_name=name,
            page=passage.page_start,
            snippet=(passage.text or "")[:400],
            score=round(float(passage.score), 4),
            facets=facets,
            download_url=passage.download_url,
        )
