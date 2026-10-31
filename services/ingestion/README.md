# Ingestion Service

Platform service **#1**. The source-of-truth producer for the write path.

## Responsibilities

- **Incremental S3 sync** over the configured prefix (streaming, memory-bounded).
- **Change detection** via S3 ETag (cheap) escalating to content **checksum**
  (sha256, authoritative).
- **Deduplication** — identical bytes under a different key are linked, not
  re-processed (checksum index).
- **Version detection** — a changed object becomes a new monotonic version with a
  distinct physical document id that records what it supersedes.
- **SQS emit** — `DISCOVERED` pipeline messages for the processing fleet.
- **Document lifecycle** — assigns deterministic ids and initial status.

It does **not** extract content — that's the processing service. Keeping ingestion
thin lets it scan millions of keys cheaply.

## Why this design

| Decision | Rationale |
|---|---|
| Deterministic ids (`document_id`) | Re-running a sync converges; no dupes. |
| ETag-first, checksum-fallback | Avoid downloading unchanged objects (cost) while staying correct on change. |
| Logical vs. physical id | Stable identity across versions + per-version indexable docs. |
| Bounded worker pool (not `gather`) | Fixed memory/concurrency at 1M-object scale. |
| Redis ledger | Shared change-detection state across the ingestion fleet, survives restarts. |
| Produce to SQS | Decouples discovery from processing; natural back-pressure + horizontal scale. |

## Trade-offs / limits

- The synchronous `POST /v1/sync` is right for incremental scans and tests; full
  **1M-document backfills** run via a Temporal workflow calling the same
  `IngestionService` (durable, resumable). See `docs/ROADMAP.md`.
- ETag-based "unchanged" can theoretically miss a same-ETag content change across
  different upload methods; set `INGESTION_CHANGE_DETECTION=checksum` (default) or
  `INGESTION_ALWAYS_VERIFY_CHECKSUM=true` for authoritative detection.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | Liveness (process up). |
| GET | `/readyz` | Readiness (ledger reachable) → 503 if not. |
| GET | `/metrics` | Prometheus exposition. |
| POST | `/v1/sync` | Trigger an incremental sync. |
| GET | `/docs` | OpenAPI / Swagger UI. |

### Example

```bash
curl -X POST localhost:8001/v1/sync \
  -H 'content-type: application/json' \
  -d '{"prefix": "Test/", "namespace": "default", "dry_run": true}'
```

## Configuration

Core coordinates (S3/SQS/Redis) come from `rag_core` settings; service knobs use
the `INGESTION_` prefix (see `config.py`):

| Var | Default | Meaning |
|---|---|---|
| `INGESTION_SYNC_CONCURRENCY` | 32 | Parallel per-object workers. |
| `INGESTION_CHANGE_DETECTION` | checksum | `etag` or `checksum`. |
| `INGESTION_MAX_OBJECT_SIZE_MB` | 512 | Skip larger objects (0 = no limit). |
| `INGESTION_EMIT_TO_SQS` | true | Disable to dry-run emits. |

## Run locally

```bash
uv run uvicorn rag_ingestion.app:app --reload --port 8001
# or containerised (context = repo root):
docker build -f services/ingestion/Dockerfile -t rag-ingestion:dev .
```

## Tests

```bash
uv run pytest services/ingestion -q
```
