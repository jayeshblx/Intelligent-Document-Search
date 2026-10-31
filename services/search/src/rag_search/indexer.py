"""IndexingService — builds the local hybrid index from S3.

Pipeline per document:
    S3 object -> ProcessingService.process_object (extract + enrich) -> Chunker
    -> embed child chunks (Titan) -> write children to Qdrant(embedded) + FTS5.

We reuse the processing service for extraction/enrichment so the write path here
is identical to the async pipeline — no duplicated extraction logic. Only *child*
chunks are indexed (the searchable unit); parents remain available for context.

Supplier names are the exception to "MongoDB is authoritative": the system of record
holds only the supplier CODE, so the name is read out of the document itself and then
consolidated per code at the end of the run (see ``_consolidate_suppliers``).
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from typing import Any

from rag_core.aws import S3Client
from rag_core.config import Settings
from rag_core.logging import bind_context, get_logger
from rag_core.models.enums import DocumentType
from rag_core.utils import chunk_id, document_id, infer_document_type
from rag_processing.service import ProcessingService
from rag_processing.store import ContentStore
from rag_search.chunking import Chunker
from rag_search.config import SearchSettings
from rag_search.embedder import Embedder
from rag_search.metadata_extract import MetadataExtractor, canonical_supplier
from rag_search.models import IndexRequest, IndexSummary
from rag_search.mongo_source import (
    MongoMetadataSource,
    apply_mongo_to_metadata,
    attachment_metadata_text,
    facets_from_mongo,
)
from rag_search.stores import DocumentMetadataStore, KeywordSearchIndex, VectorIndex

_log = get_logger("rag_search.indexer")


# Chunk-id stream for the catalogue record's MetaData text (one chunk per document).
# Bodies with no extractable text: skipped at listing so they cost neither a
# download nor an extraction attempt, and never inflate a document count.
NON_TEXT_TYPES = frozenset({DocumentType.IMAGE})

METADATA_CHUNK_STRATEGY = "attachment_metadata"
METADATA_CHUNK_SECTION = "Attachment metadata"
# Payload keys that describe one chunk rather than its document.
_CHUNK_ONLY_KEYS = frozenset({"page_start", "page_end", "section", "text", "_chunk_id"})


def _metadata_chunk(
    doc_id: str, base_payload: dict[str, Any], mongo_doc: dict[str, Any]
) -> tuple[str, str, dict[str, Any]] | None:
    """(chunk id, text, payload) for a record's MetaData, or None when it has none.

    The id is deterministic, so re-indexing replaces the chunk instead of adding one.
    """
    text = attachment_metadata_text(mongo_doc)
    if not text:
        return None
    payload = {
        **base_payload,
        "page_start": None,
        "page_end": None,
        "section": METADATA_CHUNK_SECTION,
    }
    return chunk_id(doc_id, METADATA_CHUNK_STRATEGY, 0), text, payload


class IndexingService:
    def __init__(
        self,
        *,
        s3: S3Client,
        processing: ProcessingService,
        content_store: ContentStore,
        chunker: Chunker,
        embedder: Embedder,
        vector: VectorIndex,
        keyword: KeywordSearchIndex,
        metadata_extractor: MetadataExtractor,
        metadata_store: DocumentMetadataStore,
        core_settings: Settings,
        settings: SearchSettings,
        mongo: MongoMetadataSource | None = None,
    ) -> None:
        self._s3 = s3
        self._processing = processing
        self._content = content_store
        self._chunker = chunker
        self._embedder = embedder
        self._vector = vector
        self._keyword = keyword
        self._metadata_extractor = metadata_extractor
        self._metadata_store = metadata_store
        self._mongo = mongo
        self._core = core_settings
        self._settings = settings
        # Per-run supplier votes (reset at the start of every index run).
        self._supplier_votes: dict[str, Counter[str]] = {}
        self._supplier_docs: dict[str, list[str]] = {}

    async def index(self, request: IndexRequest) -> IndexSummary:
        bucket = self._core.s3.bucket
        prefix = request.prefix if request.prefix is not None else self._core.s3.prefix
        max_docs = request.max_docs if request.max_docs is not None else self._settings.index_max_docs
        started = time.perf_counter()

        # Supplier names extracted this run, per object code, plus the documents to
        # write the winning name back to once every vote is in.
        self._supplier_votes = {}
        self._supplier_docs = {}

        summary = IndexSummary(bucket=bucket, prefix=prefix, namespace=request.namespace)
        bind_context(index_bucket=bucket, index_prefix=prefix)
        _log.info("index_started", reset=request.reset, max_docs=max_docs)

        already_indexed: set[str] = set()
        if request.reset:
            await self._vector.reset()
            await self._keyword.reset()
            await self._metadata_store.reset()
        else:
            # Resume: a run that was interrupted must not pay to re-embed what it
            # already did. Every document is otherwise re-downloaded, re-extracted
            # and re-embedded, which made a stopped run cost as much as a fresh one.
            already_indexed = await self._metadata_store.indexed_ids(request.namespace)
            _log.info("resuming_index", already_indexed=len(already_indexed))
        await self._vector.ensure()

        # Preload the external MongoDB metadata once (filename -> authoritative facets).
        if self._mongo is not None:
            try:
                loaded = await self._mongo.preload()
                _log.info("mongo_preloaded", documents=loaded)
            except Exception as exc:
                if self._mongo.filters_tenants:
                    # Without the catalogue there is no way to tell tenants apart:
                    # carrying on would index every tenant's files into this index.
                    _log.error("mongo_preload_failed_tenant_filter", error=str(exc))
                    raise
                # No tenant filter: indexing without enrichment is still correct.
                _log.warning("mongo_preload_failed", error=str(exc))
                self._mongo = None

        only = set(request.only_types) if request.only_types else None
        mongo_only = request.mongo_only and self._mongo is not None
        seen = 0
        async for obj in self._s3.list_objects(bucket, prefix):
            doc_type = infer_document_type(obj.key)
            if only and doc_type.value not in only:
                continue  # cheap skip -- no download
            # Images hold no extractable text. Indexed anyway they cost a download
            # and an extraction attempt, yield no chunks, and still get counted as
            # documents -- 1,167 of BRAVO's first 2,147 (54%) were unsearchable jpgs.
            # Skipped here, before the download. `--only image` still forces them in.
            if not only and doc_type in NON_TEXT_TYPES:
                summary.documents_non_text += 1
                continue
            # Soft-deleted in the system of record (Mongo DeleteMark=1): never index, and
            # purge any copy left over from a previous run so it drops out of search.
            if self._mongo is not None and self._mongo.is_deleted(obj.key):
                await self._purge_document(obj.key, request.namespace)
                summary.documents_deleted += 1
                continue
            # Another tenant's file (they share S3 folders): skip it whether or not
            # mongo_only is set -- indexed without metadata it would still leak into
            # this tenant's search. Purge leftovers from an unfiltered earlier run.
            if self._mongo is not None and self._mongo.is_foreign(obj.key):
                if not request.reset:
                    await self._purge_document(obj.key, request.namespace)
                summary.documents_other_tenant += 1
                continue
            if already_indexed and document_id(bucket, obj.key) in already_indexed:
                summary.documents_skipped_existing += 1
                continue  # already in this index -- no download, no embedding
            if mongo_only and self._mongo is not None and self._mongo.lookup(obj.key) is None:
                continue  # not catalogued in Mongo — skip when mongo_only is set
            if max_docs and seen >= max_docs:
                break
            seen += 1
            try:
                chunk_count = await asyncio.wait_for(
                    self._index_one(obj.key, request.namespace),
                    timeout=self._settings.doc_timeout_s,
                )
                if chunk_count:
                    summary.documents_indexed += 1
                    summary.chunks_indexed += chunk_count
            except TimeoutError:
                summary.documents_failed += 1
                if len(summary.error_samples) < 5:
                    summary.error_samples.append(f"{obj.key}: extraction timed out")
                _log.warning("index_document_timeout", key=obj.key)
            except Exception as exc:
                summary.documents_failed += 1
                if len(summary.error_samples) < 5:
                    summary.error_samples.append(f"{obj.key}: {exc}")
                _log.warning("index_document_failed", key=obj.key, error=str(exc))

        # One name per supplier code, applied to every document under it.
        consolidated = await self._consolidate_suppliers(request.namespace)
        if summary.chunks_indexed:
            await self._rebuild_vector_index()

        summary.duration_s = round(time.perf_counter() - started, 3)
        _log.info(
            "index_complete",
            documents=summary.documents_indexed,
            failed=summary.documents_failed,
            deleted=summary.documents_deleted,
            other_tenant=summary.documents_other_tenant,
            suppliers_named=consolidated,
            chunks=summary.chunks_indexed,
            duration_s=summary.duration_s,
        )
        return summary

    async def index_keys(
        self,
        keys: list[str],
        namespace: str = "default",
        *,
        deadline: float | None = None,
    ) -> tuple[IndexSummary, list[str]]:
        """Index exactly these S3 objects -- the incremental path.

        The caller has already decided they belong here (tenant, not deleted, has
        metadata, not yet indexed); this embeds and writes them. ``deadline`` is a
        ``time.monotonic()`` instant: keys not started by then are returned unattempted
        so a time-boxed run (Lambda) can hand them to the next one.
        """
        started = time.perf_counter()
        summary = IndexSummary(bucket=self._core.s3.bucket, prefix="", namespace=namespace)
        self._supplier_votes = {}
        self._supplier_docs = {}
        await self._vector.ensure()
        unattempted: list[str] = []
        for pos, key in enumerate(keys):
            if deadline is not None and time.monotonic() >= deadline:
                unattempted = keys[pos:]
                break
            try:
                chunks = await asyncio.wait_for(
                    self._index_one(key, namespace), timeout=self._settings.doc_timeout_s
                )
                if chunks:
                    summary.documents_indexed += 1
                    summary.chunks_indexed += chunks
            except Exception as exc:  # TimeoutError included: one bad file never stops a run
                summary.documents_failed += 1
                if len(summary.error_samples) < 5:
                    summary.error_samples.append(f"{key}: {exc or 'extraction timed out'}")
                _log.warning("index_document_failed", key=key, error=str(exc))
        # Consistent within the batch; documents indexed earlier keep their own name.
        await self._consolidate_suppliers(namespace)
        if summary.chunks_indexed:
            await self._rebuild_vector_index()
        summary.duration_s = round(time.perf_counter() - started, 3)
        return summary, unattempted

    async def purge_document(self, key: str, namespace: str = "default") -> None:
        """Public form of :meth:`_purge_document` for the scheduled delete job."""
        await self._purge_document(key, namespace)

    async def _index_one(self, key: str, namespace: str) -> int:
        doc_id = document_id(self._core.s3.bucket, key)
        result = await self._processing.process_object(
            bucket=self._core.s3.bucket,
            key=key,
            document_id=doc_id,
            namespace=namespace,
            doc_type=infer_document_type(key),
        )
        content = await self._content.get(doc_id, namespace=namespace)
        if content is None:
            return 0

        # Metadata Extraction Agent (#4): rich facets -> canonical store + chunk payload.
        # If MongoDB (system of record) has this document, its business fields are
        # authoritative — use them and skip the (costly) LLM extraction entirely.
        mongo_doc = self._mongo.lookup(key) if self._mongo is not None else None
        metadata = await self._metadata_extractor.extract(
            content,
            document_id=doc_id,
            source_key=key,
            document_class=result.document_class,
            language=result.language,
            use_llm=False if mongo_doc is not None else None,
        )

        # Merge facets: standard (LLM/heuristic) first, then Mongo overrides.
        facets: dict[str, Any] = {
            k: v for k, v in metadata.filterable_payload().items() if v not in (None, [], "")
        }
        if mongo_doc is not None:
            apply_mongo_to_metadata(mongo_doc, metadata)
            facets.update(facets_from_mongo(mongo_doc))  # authoritative

        # The supplier NAME is never in Mongo (it holds the code only) and the big LLM
        # extraction is skipped for Mongo-backed documents — so ask the document itself.
        if not facets.get("supplier"):
            supplier = await self._metadata_extractor.extract_supplier(
                content, title=metadata.title
            )
            if supplier:
                metadata.supplier = supplier
                facets["supplier"] = supplier
        # Record the vote so every document under this code ends up on one name.
        if (code := facets.get("object_code")) and isinstance(code, str):
            self._supplier_docs.setdefault(code, []).append(doc_id)
            if facets.get("supplier"):
                self._supplier_votes.setdefault(code, Counter())[str(facets["supplier"])] += 1

        await self._metadata_store.put(
            metadata, namespace=namespace, source_key=key, facets=facets
        )

        # Denormalise only *scalar* facets onto every chunk for query-time filtering
        # (list facets like keywords/tags aren't reliable equality filters across stores).
        base_payload: dict[str, Any] = {
            "document_id": doc_id,
            "source_key": key,
            "doc_class": result.document_class.value,
            **{k: v for k, v in facets.items() if isinstance(v, str)},
        }
        chunks = self._chunker.chunk(content, namespace=namespace, base_payload=base_payload)
        children = [c for c in chunks if not c.is_parent and c.text.strip()]
        # The catalogue record's own field values, as one extra chunk. A file with no
        # extractable text is still findable through its metadata.
        extra = [
            c for c in [_metadata_chunk(doc_id, base_payload, mongo_doc)] if c is not None
        ] if mongo_doc is not None else []
        if not children and not extra:
            return 0

        vectors = await self._embedder.embed(
            [c.text for c in children] + [text for _, text, _ in extra]
        )
        extra_vectors = vectors[len(children):]
        vectors = vectors[: len(children)]

        vector_points: list[tuple[str, list[float], dict[str, Any]]] = []
        keyword_entries: list[tuple[str, str, dict[str, Any]]] = []
        for chunk, vector in zip(children, vectors, strict=True):
            # The text deliberately does NOT go in the payload: the keyword store
            # holds it in its own column (and is the primary source at read time),
            # so a copy here would be the same string stored twice in the vector
            # index -- which is most of a published snapshot's size.
            payload = {
                **base_payload,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section": chunk.section_path[-1] if chunk.section_path else None,
            }
            vector_points.append((chunk.id, vector, payload))
            keyword_entries.append((chunk.id, chunk.text, payload))
        for (extra_id, text, payload), vector in zip(extra, extra_vectors, strict=True):
            vector_points.append((extra_id, vector, payload))
            keyword_entries.append((extra_id, text, payload))

        await self._vector.upsert(vector_points)
        await self._keyword.add(keyword_entries)
        return len(children) + len(extra)

    async def backfill_attachment_metadata(
        self, namespace: str = "default", *, batch: int = 64
    ) -> int:
        """Add the attachment-metadata chunk to documents indexed before it existed.

        Embeds only those chunks -- no download, no re-extraction, no re-embedding of
        file text. Documents that already have the chunk are skipped, so it is safe to
        re-run (a full re-index refreshes values that changed upstream). Honours the
        tenant filter: only this index's own catalogue records are read.
        """
        if self._mongo is None:
            raise RuntimeError("backfilling attachment metadata needs the MongoDB source")
        await self._mongo.preload()

        pending: list[tuple[str, str, dict[str, Any]]] = []
        for doc_id, key in await self._metadata_store.all_documents():
            mongo_doc = self._mongo.lookup(key)
            if mongo_doc is None:
                continue
            cid = chunk_id(doc_id, METADATA_CHUNK_STRATEGY, 0)
            if await self._keyword.get([cid]):
                continue
            existing = await self._vector.fetch_document(doc_id, namespace=namespace, limit=1)
            if existing:
                base = {k: v for k, v in existing[0].items() if k not in _CHUNK_ONLY_KEYS}
            else:
                described = (await self._metadata_store.describe([doc_id], namespace)).get(
                    doc_id
                ) or {}
                base = {
                    "document_id": doc_id,
                    "source_key": key,
                    **dict(described.get("facets") or {}),
                }
            if (chunk := _metadata_chunk(doc_id, base, mongo_doc)) is not None:
                pending.append(chunk)

        for start in range(0, len(pending), batch):
            part = pending[start : start + batch]
            vectors = await self._embedder.embed([text for _, text, _ in part])
            await self._vector.upsert(
                [(cid, vec, payload) for (cid, _, payload), vec in zip(part, vectors, strict=True)]
            )
            await self._keyword.add(part)
        if pending:
            await self._rebuild_vector_index()
        _log.info("attachment_metadata_backfilled", chunks=len(pending))
        return len(pending)

    async def _consolidate_suppliers(self, namespace: str) -> int:
        """Give every document under one supplier code the same supplier name.

        Extraction runs per document, so one supplier's documents come back with several
        spellings of its name (and the odd stray from a signature block). The vote picks
        the corpus's own spelling; this pass then writes it to all three stores — the
        metadata record that Browse and the answer context read, and the chunk payloads
        that query-time filtering reads.
        """
        updated = 0
        for code, names in self._supplier_votes.items():
            winner = canonical_supplier(names)
            if winner is None:
                continue
            for doc_id in self._supplier_docs.get(code, []):
                facets = {"supplier": winner}
                if not await self._metadata_store.apply_facets(doc_id, facets):
                    continue
                await self._keyword.update_payload(doc_id, facets)
                await self._vector.set_document_payload(doc_id, facets)
                updated += 1
            _log.info(
                "supplier_resolved", object_code=code, supplier=winner,
                documents=len(self._supplier_docs.get(code, [])), variants=len(names),
            )
        return updated

    async def _rebuild_vector_index(self) -> None:
        """Rebuild the approximate vector index after adding rows, where the store
        has one (Postgres HNSW). Late inserts into a large graph measurably lose
        recall -- see ``PgVectorIndex.rebuild_index``."""
        rebuild = getattr(self._vector, "rebuild_index", None)
        if rebuild is not None:
            await rebuild()

    async def _purge_document(self, key: str, namespace: str) -> None:
        """Remove a document from every index (vectors, keyword, metadata). Used for
        objects soft-deleted in the system of record so they vanish from search."""
        doc_id = document_id(self._core.s3.bucket, key)
        await self._vector.delete_document(doc_id)
        await self._keyword.delete_document(doc_id)
        await self._metadata_store.delete_document(doc_id)
