# rag_core

Shared foundation library for the Agentic RAG platform. Every microservice
depends on this package; it deliberately contains **no business logic** — only
cross-cutting concerns and the typed contracts services exchange.

## Modules

| Module | Purpose |
|---|---|
| `rag_core.config` | Typed, env-driven settings (`Settings`, `get_settings`). |
| `rag_core.logging` | structlog setup + trace-correlated structured logs. |
| `rag_core.observability` | OpenTelemetry tracing + Prometheus metrics helpers. |
| `rag_core.models` | Pydantic domain models — the inter-service contracts. |
| `rag_core.resilience` | Error taxonomy, retry decorators, circuit breaker. |
| `rag_core.aws` | Async S3 / SQS / Bedrock clients with resilience baked in. |
| `rag_core.stores` | Vendor-neutral store interfaces + Qdrant adapter. |
| `rag_core.utils` | Deterministic IDs and checksums (idempotency primitives). |
| `rag_core.di` | Lightweight DI container managing shared-resource lifecycle. |

## Design principles

- **Side-effect-free import.** Importing `rag_core` never opens a socket or reads
  a file. Wiring is explicit via `Container`.
- **Idempotency first.** Deterministic IDs (`utils.ids`) + checksums
  (`utils.checksum`) make the whole pipeline safe to re-run.
- **Correct failure classification.** `resilience.errors` distinguishes
  retryable vs. permanent faults so retries/DLQ behave correctly.
- **Vendor-neutral storage.** Services depend on interfaces, not Qdrant/OpenSearch
  types — enforcing the "no Pinecone" constraint and keeping backends swappable.

## Tests

```bash
uv run pytest packages/core -q
```
