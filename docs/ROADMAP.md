# Build Roadmap & Status Matrix

This platform is delivered as a **deep vertical slice first, then expanded** —
rather than 15 shallow skeletons. The shared `rag_core` foundation and the
ingestion write-path entrypoint are production-grade; everything else is wired in
order of dependency (you cannot query what isn't indexed).

Legend: ✅ done · 🟡 in progress · ⬜ planned

## Shared foundation — `packages/core`

| Component | Status | Notes |
|---|---|---|
| Config (pydantic-settings) | ✅ | Nested, env-driven, validated. |
| Structured logging (structlog) | ✅ | JSON + trace correlation. |
| Observability (OTel + Prometheus) | ✅ | Tracing setup, metric factories. |
| Domain models (Pydantic) | ✅ | Document/Chunk/Metadata/Embedding/Content/Messages. |
| Resilience (retry + circuit breaker) | ✅ | Error taxonomy drives retry decisions. |
| AWS clients (S3/SQS/Bedrock) | ✅ | Async, resilient, error-classified. |
| Vector store interface + Qdrant | ✅ | Named vectors, filter DSL, snapshots. |
| Full-text interface + OpenSearch | ⬜ | Lands with the read path. |
| Graph interface + Neo4j | ⬜ | Lands with the graph builder. |
| DI container | ✅ | Lifecycle-managed shared resources. |

## Services

| # | Service | Status | Depends on |
|---|---|---|---|
| 1 | Ingestion | ✅ (sync API) · 🟡 (Temporal backfill) | core, S3, SQS, Redis |
| 2 | Processing (text/OCR/tables/layout) | ✅ **PDF→Markdown via PyMuPDF (default)** | core, S3, SQS, Textract |
| 3 | Chunking (parent-child) | ✅ (MVP, in `search`) | processing output |
| 4 | Metadata extraction | ✅ **MongoDB system-of-record facets (authoritative) + Sonnet/heuristic fallback; dynamic facet store; filterable** | processing output, Bedrock, MongoDB |
| 5 | Embedding (Titan) | ✅ (MVP, in `search`) | chunks, Bedrock |
| 6 | Indexer — vectors | ✅ **Qdrant embedded (local/free)** | embeddings, Qdrant |
| 7 | Full-text indexer (BM25) | ✅ **SQLite FTS5 (local/free)** | chunks |
| 8 | Graph builder | ⬜ | metadata/entities, Neo4j |
| 9 | Hybrid retrieval | ✅ (MVP, in `search`) | stores |
| 10 | RRF fusion | ✅ | retrieval |
| — | **Search UI** | ✅ `services/search` (`/`) — search + **Browse-by-facet with pagination** | retrieval |
| 11 | Reranker (Bedrock/BGE/cross-encoder) | ⬜ | fusion |
| 12 | Agent orchestrator (LangGraph) | 🟡 (single-call RAG in `search`) | retrieval/rerank/graph |
| 13 | Response generator (Claude) | ✅ **Bedrock Sonnet 4.5 chat** (`/v1/chat`) | orchestrator |
| — | **Chat UI + sessions + token/cost** | ✅ `services/search` (`/`) | response |
| 14 | Evaluation framework | ⬜ | end-to-end |
| 15 | Observability dashboards | 🟡 | Prometheus/Grafana provisioned |

## Next steps (recommended order)

1. ✅ **Processing service (#2)** — done. SQS consumer turning `DISCOVERED` events
   into `ExtractedContent` (extractor registry per `DocumentType`, Textract OCR for
   scanned PDFs, language/PII/classification enrichment).
2. **Chunking (#3)** + **Metadata (#4)** — consume extracted content (read the
   content URI from the `PROCESSED` event).
3. **Embedding (#5)** + **Indexer (#6/#7/#8)** — complete the write path to all
   three stores; defines the Qdrant `CollectionConfig` (named vectors + payload
   indexes) and seeds queryable data.
4. **Temporal ingestion workflow** — wrap the existing `IngestionService` for
   durable, resumable 1M-document backfills.
5. **Read path (#9–#13)** — hybrid retrieve → RRF → rerank → LangGraph agents →
   Claude response with citations.
6. **Evaluation (#14)** — groundedness/faithfulness/precision/recall/MRR harness.

## Cross-cutting, still to wire

- Helm charts + HPA under `deploy/helm` (Compose exists for dev).
- LocalStack bootstrap script (create bucket + queues + DLQ) for one-command dev.
- Integration test suite (marked `integration`) against docker-compose backends.
- CI pipeline (ruff + mypy + pytest + image build).
