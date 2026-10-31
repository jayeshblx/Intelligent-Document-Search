# Agentic RAG Platform

An enterprise, multi-agent Retrieval-Augmented Generation platform engineered to
index and query **1M+ compliance documents** (≈100M chunks) across PDF, DOCX,
PPTX, XLSX, XML, images, scanned/OCR documents, tables, flowcharts, certificates,
policies, technical manuals, and regulatory texts.

> Built on AWS Bedrock (Titan embeddings + Claude Sonnet), Qdrant (vectors),
> OpenSearch (BM25), Neo4j (knowledge graph), Temporal (durable ingestion),
> and LangGraph (query-time agent orchestration).

---

## Architecture at a glance

```
                         ┌─────────────────────── WRITE PATH (Temporal-orchestrated) ───────────────────────┐
  S3 (compliance docs)   │                                                                                   │
        │                ▼                                                                                   │
        └──►  Ingestion ──► Processing ──► Chunking ──► Metadata ──► Embedding ──► Indexer ──► Qdrant        │
             (S3 sync,      (text/OCR,     (parent/    (entities,    (Titan/      (atomic     OpenSearch     │
              dedup,         tables,        child,      regulation,   BGE-M3/      writes)     Neo4j          │
              versioning)    layout)        semantic)   dates...)     ColBERT)                                │
                         └───────────────────────────────────────────────────────────────────────────────────┘

                         ┌─────────────────────── READ PATH (LangGraph agent graph) ────────────────────────┐
  User query  ─►  Intent ─► Metadata-Filter ─► [Vector ‖ BM25 ‖ Graph] ─► RRF Fusion ─► Rerank ─►            │
                                                                                                              │
                  Compliance-Reasoning ─► Verification ─► Citation ─► Response (Claude) ─► answer+citations   │
                         └───────────────────────────────────────────────────────────────────────────────────┘
```

Each numbered service from the design spec maps to a directory under `services/`.
All services share `packages/core` (`rag_core`): config, logging, observability,
domain models, AWS clients, resilience primitives, and store adapters.

## Technology decisions (and why)

| Concern | Choice | Rationale |
|---|---|---|
| Vector store | **Qdrant** | Native *named vectors* (one point → Titan + BGE-M3 + ColBERT vectors), payload-indexed metadata filtering, collections-as-namespaces, snapshots for versioning, scales embedded → sharded cluster. No Pinecone. |
| Full-text / BM25 | **OpenSearch** | Production BM25 with fuzzy + phrase + analyzer control; managed equivalent on AWS. |
| Knowledge graph | **Neo4j** | Cypher traversals for `regulates`/`supplied_by`/`derived_from` relationships. |
| Ingestion orchestration | **Temporal** | Durable execution → checkpoint recovery + automatic retries for 1M-doc backfills. |
| Query orchestration | **LangGraph** | Stateful agent DAG with conditional edges + token streaming. |
| Embeddings | **Bedrock Titan v2** (default) + BGE-M3 + ColBERT (optional) | Titan needs no GPU; BGE-M3/ColBERT pluggable behind an interface. |
| LLM | **Bedrock Claude Sonnet** | Reasoning + grounded synthesis. |
| Config | **pydantic-settings** | Typed, validated, 12-factor config. |
| Logging | **structlog** | Structured JSON logs with trace correlation. |
| Telemetry | **OpenTelemetry + Prometheus + Grafana** | Traces + metrics + dashboards. |

## Repository layout

```
.
├── pyproject.toml              # uv workspace (shared lockfile)
├── docker-compose.yml          # local infra: qdrant, opensearch, neo4j, temporal, redis, otel, prom, grafana
├── packages/
│   └── core/                   # rag_core — shared library (config, models, AWS, stores, resilience, telemetry)
├── services/
│   ├── ingestion/              # (1) S3 sync, dedup, versioning, SQS/Temporal emit
│   ├── processing/             # (2) text/OCR/tables/layout extraction
│   ├── chunking/               # (3) parent-child / semantic / section chunking
│   ├── metadata/               # (4) metadata + entity extraction
│   ├── embedding/              # (5) multi-vector embeddings
│   ├── indexer/                # (6/7/8) writes to Qdrant + OpenSearch + Neo4j
│   ├── retrieval/              # (9/10/11) hybrid retrieve + RRF + rerank
│   ├── orchestrator/           # (12) LangGraph multi-agent workflow
│   └── api/                    # (13) query API + streaming responses
└── deploy/
    ├── observability/          # otel-collector, prometheus, grafana provisioning
    └── helm/                   # K8s charts (prod)
```

## Run & test locally

Install once: `uv sync --all-packages` (add `--native-tls` behind a TLS-proxy).
There are three tiers, by how much infra they need.

### Tier 1 — Tests (no infra)

```bash
uv run pytest -q                 # whole workspace (79 tests)
uv run pytest packages/core -q   # just the shared library
uv run ruff check . && uv run mypy packages services/*/src
```

### Tier 2 — Extract a real document (no infra)

The closest thing to "retrieve the correct document" available today: fetch a
document and show its faithfully-extracted content (pages, headings, tables,
language, classification, PII).

```bash
# Zero infra: generates a sample DOCX in memory and extracts it.
uv run python scripts/demo_pipeline.py process --source fake

# A real S3 object (needs AWS creds + s3:GetObject):
uv run python scripts/demo_pipeline.py process --source s3 \
    --bucket example-index-bucket --key Test/your-file.pdf
```

> Semantic search ("which document answers X?") is **not built yet** — it needs
> chunking (#3) → embedding (#5) → Qdrant indexer (#6) → retrieval (#9). See
> `docs/ROADMAP.md`. The current build covers the *write path* only.

### Tier 3 — Full pipeline (Docker or real AWS)

```bash
# 1. Backing services (Qdrant, OpenSearch, Neo4j, Temporal, Redis, LocalStack...)
docker compose up -d

# 2. Create bucket + queues in LocalStack (optionally upload sample docs)
uv run python scripts/bootstrap_localstack.py --upload ./samples

# 3. Run the services (each on its own port)
uv run uvicorn rag_ingestion.app:app  --reload --port 8001
uv run uvicorn rag_processing.app:app --reload --port 8002

# 4. Trigger an incremental sync; processing consumes from SQS automatically
curl -X POST localhost:8001/v1/sync \
  -H 'content-type: application/json' \
  -d '{"prefix": "Test/", "dry_run": false}'

# Inspect extracted content written to S3 under processed/content/<ns>/<id>.json
```

Health/OpenAPI for any service: `GET /healthz`, `GET /readyz`, `GET /metrics`,
and Swagger UI at `/docs`.

## Build status

This platform is delivered incrementally as a deep vertical slice, then expanded.
See `docs/ROADMAP.md` for the per-service status matrix. Current focus: the
**ingestion → index write path** plus the shared `rag_core` foundation.

## Configuration

All config is environment-driven and typed via `rag_core.config.Settings`.
Copy `.env.example` → `.env` for local dev. See the env file for every key.

## License

Proprietary — internal enterprise platform.
