# Agentic Service

The **agentic AI approach** to the ExampleApp document corpus: two agents deployed as
two Amazon Bedrock AgentCore Runtimes. ExampleApp calls the **Orchestrator**, which
routes each request to the **Document Search Agent**.

```
ExampleApp ──POST /invocations──▶ Orchestrator Agent  (AgentCore Runtime #1)
                                      │
                                      │ routes document questions
                                      ▼
                                Document Search Agent (AgentCore Runtime #2)
                                      │  InvokeAgentRuntime
                                      ▼
                    tenant snapshot from S3 ── vectors + keyword + metadata
```

Both agents run a plain **Bedrock Converse tool loop** (`toolConfig` → `toolUse` →
`toolResult`) over `rag_core`'s resilience layer. No agent framework.

## The request envelope

One shape crosses every boundary, so a payload can be replayed against either
runtime unchanged:

```jsonc
{
  "base_url": "https://foxtrot.acmedev.com/",     // required: selects environment + tenant
  "query": "Do suppliers need ISO 9001?",   // required: the user's question
  "business_object": "COMPLAINT",           // optional: hard pre-filter
  "user_code": "USER-4471",                 // optional: who asked (UserCode)
  "session_id": "…"                         // optional: AgentCore pads to >= 33 chars
}
```

ExampleApp's own spellings are accepted too — `UserCode`, `BusinessObject`, `Query`,
`BaseUrl` — so the payload can be sent as the platform already names these fields.

**`user_code`** identifies the caller: it is carried through both agents, written to
both runtimes' logs, and echoed on the response, so an answer can be tied back to the
person who asked. It does **not** change what is searched — `base_url` is the only
scope boundary, and per-user document permissions are not implemented.

**Retrieval depth is not a request field.** How many passages a search returns is
deployment configuration (`AGENTIC_TOP_K`), so a caller cannot widen it per request;
sending `top_k` is ignored.

The response is always an `AgentResponse`: `answer`, `citations[]`, `route`
(`document_search` / `direct`), `environment`, `tenant`, `trace[]` (every tool call)
and `usage` (tokens, summed across both agents).

## How ExampleApp invokes it and reads the response

The answer comes back **on the same call** — one synchronous request/response, no
polling and no callback. AgentCore returns the orchestrator's JSON body in
`response["response"]`, a streaming body you read once:

```python
import json, uuid, boto3

client = boto3.client("bedrock-agentcore", region_name="us-east-1")

result = client.invoke_agent_runtime(
    agentRuntimeArn=ORCHESTRATOR_ARN,
    runtimeSessionId=session_id,          # >= 33 chars; reuse it to keep a conversation
    qualifier="DEFAULT",
    payload=json.dumps({
        "base_url": "https://dev.example.io/acme",
        "query": "Do suppliers need ISO 9001?",
        "business_object": "SUPPLIERDOCUMENTS",   # optional
    }).encode(),
)

body = json.loads(result["response"].read())
body["answer"]     # the grounded answer, ready to render
body["citations"]  # [{document_id, document_name, page, snippet, score, facets, download_url}]
body["route"]      # "document_search" | "direct"
body["usage"]      # tokens across BOTH agents, for metering
```

A real payload:

```jsonc
{
  "answer": "Yes. According to the Supplier Quality Manual, all suppliers must hold a valid ISO 9001 certificate, renewed every three years before expiry.",
  "citations": [
    { "document_id": "d1", "document_name": "Supplier Quality Manual", "page": 1,
      "snippet": "All suppliers must hold a valid ISO 9001 certificate…", "score": 0.0328,
      "facets": { "business_object": "SUPPLIERDOCUMENTS", "status": "Active", "object_code": "SQ-1" },
      "download_url": "https://example-index-bucket.s3.amazonaws.com/foxtrot/…~pdf?X-Amz-…" }
  ],
  "route": "document_search",
  "environment": "acmedev", "tenant": "foxtrot", "business_object": "COMPLAINT",
  "user_code": "USER-4471",
  "trace": [
    { "name": "search_documents", "input": { "query": "ISO 9001 certificate requirement" },
      "summary": "2 passage(s)", "took_ms": 1.78, "error": null },
    { "name": "search_documents", "input": { "query": "Do suppliers need ISO 9001?" },
      "summary": "document search: 2 citation(s)", "took_ms": 14.94, "error": null }
  ],
  "usage": { "input_tokens": 5350, "output_tokens": 260, "model": "…" },
  "session_id": "exampleapp-session-…"
}
```

**Errors** come back as `{"error": "…", "status": 503, "session_id": "…"}` with a
non-200 status: `400` for an unresolvable `base_url` or a malformed payload, `503`
when the tenant's snapshot is missing or cannot be restored, the search runtime's own
status (else `502`) when it fails behind the orchestrator, `500` otherwise. There is
no partial-success case — a request either yields an `answer` or an `error`.

That holds **through the orchestrator** too. Its model normally sees tool errors so
it can correct itself, but a platform failure is not something wording can fix:
fed to the model it becomes "I'm experiencing a technical issue" with HTTP 200,
which a caller cannot tell from an answer. Those failures (`IndexUnavailableError`,
`TenancyError`, `DelegateError`) end the turn instead — see `propagate` in
`agent_loop.py` and `tests/test_failure_contract.py`.

### The `answer` is the search agent's own words

When the orchestrator delegates, the specialist's answer is returned **verbatim**
and the orchestrator makes no further model call — its routing turn is the only
one billed on that side. That keeps the caller's answer identical to what actually
read the documents (no paraphrase risk), and saves a round trip. Only an
undelegated turn — greeting, capability question, clarification — returns the
orchestrator's own text. See `_answer_of` and the `stop_after_tools` hook.

### Document links (`citations[].download_url`)

Every citation carries a link to its source document, for every document the answer
used — search hits first (best score first), then any documents a listing question
named, so "list all Draft supplier documents" links each one. Links are attached to
citations only: the model never sees them and is told not to write URLs, so they are
never mangled into the answer text or leaked into logs.

| `AGENTIC_DOCUMENT_URL_MODE` | Link | Trade-off |
|---|---|---|
| `presigned` (default) | Time-limited S3 GET URL; PDFs open inline with their real filename | No extra infrastructure. The URL exposes bucket, key and an access-key id, and **dies with the credentials that signed it** — under an AgentCore execution role that is the role session, even if `AGENTIC_DOCUMENT_URL_TTL_S` is longer |
| `template` | A ExampleApp URL, e.g. `{base_url}/documents/{document_id}` | Nothing about storage reaches the browser and ExampleApp can authenticate the user — needs a matching ExampleApp route |
| `none` | No links | |

A presigned link is signed against the **tenant's own** source bucket, read from its
snapshot manifest (`source_bucket`), and only for keys under that tenant's
`source_prefix` — never from anything in the request. Template placeholders:
`{base_url} {environment} {tenant} {document_id} {source_key} {filename}`; an
invalid template fails at startup, not on the first request.

### `base_url` → tenant

`base_url` is the tenancy key, resolved from **configuration only** — never from
the model, never from a hint inside the question. Resolution order, first match wins:

| Order | Setting | Example |
|---|---|---|
| 1 | `AGENTIC_TENANT_MAP` | `{"https://corp.portaltest.com": "portaltest/corp"}` |
| 2 | `AGENTIC_TENANT_PATTERN` | see below |
| 3 | `AGENTIC_DEFAULT_ENVIRONMENT` + `AGENTIC_DEFAULT_TENANT` | only if both are set |

ExampleApp's URLs are `https://<tenant>.<environment-domain>.com`. Deploy this pattern:

```
AGENTIC_TENANT_PATTERN=^(?P<tenant>[a-z0-9][a-z0-9-]*)\.(?P<environment>acmedev|acmeqa|portaltest|portalprocess|portalprod)\.com$
```

| base_url | environment | tenant | S3 folder (`AGENTIC_INDEX_PREFIX=DocumentSearch`) |
|---|---|---|---|
| `https://foxtrot.acmedev.com/` | `acmedev` | `foxtrot` | `DocumentSearch/acmedev/foxtrot/` (in `example-docs-bucket`) |
| `https://globex.acmeqa.com/` | `acmeqa` | `globex` | `DocumentSearch/acmeqa/globex/` |
| `https://corp.portaltest.com/` | `portaltest` | `corp` | `DocumentSearch/portaltest/corp/` |
| `https://corp.portalprocess.com/` | `portalprocess` | `corp` | `DocumentSearch/portalprocess/corp/` |
| `https://corp.portalprod.com/` | `portalprod` | `corp` | `DocumentSearch/portalprod/corp/` |

**Anchor it (`^…$`) and list the environment domains.** A pattern like
`^(?P<tenant>[^.]+)\.(?P<environment>[^.]+)\.com` also accepts
`corp.portaltest.com.attacker.net` as tenant `corp`, and invents an environment
for any `*.com` host — both verified in `tests/test_exampleapp_tenants.py`. A new
environment domain is a deliberate one-line change here.

An unresolvable `base_url` is a **400**, never a fallback to a shared index —
serving the wrong tenant silently is the one failure worth crashing over. Resolved
names are validated as safe path segments, so `..` can never escape the prefix.

> **Set one of these before deploying.** The pattern above is an example; swap in
> your real ExampleApp URL shape (a host-style `acme.prod.example.io` works too —
> see `tests/test_tenancy.py` for both).

### `business_object` (optional pre-filter)

**Optional.** Omit it (or send an empty string) and the whole tenant is searched.
Supply it and results are confined to documents carrying that BusinessObject.

It is matched **case-insensitively** against the tenant's own values, so
`"complaints"` applies the stored `COMPLAINTS` — facet matching in the stores is
exact, and a case mismatch would otherwise silently match nothing. The response's
`business_object` echoes the value actually applied. A value the tenant does not
have is still enforced as given (zero results), never dropped in favour of searching
everything.

When supplied it is merged into **every** retrieval call before ranking, in the
tool layer. The model is *told* the filter is active so it can explain the scope,
but it cannot widen, replace or bypass it — a model-supplied `business_object`
argument is overwritten, and `get_document_details` re-checks the facet before
returning a document. `tests/test_tools.py` pins all three behaviours.

## Per-tenant index snapshots in S3

The agent containers are ephemeral, so the index lives in S3 and is pulled into
`/tmp` on first use:

```
s3://{AGENTIC_INDEX_BUCKET}/{AGENTIC_INDEX_PREFIX}/{environment}/{tenant}/
    manifest.json    {"collection": "compliance", "embed_dim": 1024, "namespace": "default"}
    qdrant/**        embedded Qdrant collection (vectors)
    keyword.db       SQLite FTS5 (keyword)
    metadata.db      SQLite (facets: business_object, status, object_code, …)
```

### Postgres snapshots — S3 is the only storage

The Postgres-backed index is published by `scripts/publish_index_snapshot.py` as a
`pg_dump` beside a manifest:

```
s3://{AGENTIC_INDEX_BUCKET}/DocumentSearch/foxtrot-Dev/
    rag-<timestamp>.dump   pg_dump custom format
    manifest.json          {"backend": "postgres", "embed_dim": 1024, "namespace": "default",
                            "source_bucket": "...", "source_prefix": "foxtrot/",
                            "counts": {"chunks": 13545, ...},
                            "dump": {"key": "rag-<timestamp>.dump", "sha256": "...", ...}}
```

A dump is not queryable on its own, so the search agent runs a **PostgreSQL 16 server
private to its container** (`embedded_pg.py`; installed in the image, Unix socket
only, no TCP listener). On a tenant's first request it:

1. downloads the dump (streamed to `/tmp`, never held in memory),
2. checks it against the manifest's `sha256` — a mismatch is a `503`, never a partial index,
3. restores it into a database for that tenant (`pg_restore --exit-on-error`),
4. checks the restored chunk count against the manifest's `counts.chunks`,
5. deletes the dump and serves queries from the restored database.

**One index per `TenantCode`.** Tenants share storage — in dev, `FOXTROT_DEV` is mostly
`ACMEFOODS` records and `example-docs-bucket/foxtrot/` holds their files too — so build each
index with `SEARCH_MONGO_TENANT_CODES` (e.g. `FOXTROT`) into its own database, then publish
it to that tenant's folder:

```powershell
$env:S3__BUCKET = "example-docs-bucket"; $env:S3__PREFIX = "foxtrot/"
$env:SEARCH_STORE_BACKEND = "postgres"
$env:SEARCH_POSTGRES_DSN = "postgresql://rag:rag@localhost:5432/rag_acmedev_foxtrot"
$env:SEARCH_MONGO_DATABASE = "FOXTROT_DEV"; $env:SEARCH_MONGO_TENANT_CODES = "FOXTROT"
uv run python scripts/index_s3.py index --mongo-only --only pdf,docx,xlsx
uv run python scripts/publish_index_snapshot.py --environment acmedev --tenant foxtrot `
    --bucket example-docs-bucket --dsn postgresql://rag:rag@localhost:5432/rag_acmedev_foxtrot
```

`dump.key` must be a bare file name beside the manifest, so a manifest cannot point
the agent at another tenant's objects. Evicting a tenant, or a changed manifest ETag,
drops its database; the next request restores the new snapshot.

**Cold start**, measured in the agent image with the real 69.5 MiB, 13,545-chunk
snapshot (amd64 build on a laptop, default 64 MB `/dev/shm`):

| Phase | Time |
|---|---|
| S3 download | 38 s over a laptop link through a corporate proxy — in-region is far faster |
| server start | ~1 s |
| `pg_restore` incl. HNSW / GIN / trigram index builds | 4.6 s |

Every *new* container pays this on its first request per tenant; warm requests pay
nothing (a warm document question answered in ~7 s, all of it model time). A small
runtime rebuilds the index more slowly than the laptop did, so measure on the target.

The server uses `dynamic_shared_memory_type = mmap`: a parallel index build
otherwise asks `/dev/shm` for ~250 MB, and containers cap it — the first real
restore in the image failed with *"No space left on device"* until this was set.

`AGENTIC_POSTGRES_SOURCE=dsn` + `AGENTIC_POSTGRES_DSN` instead connects to an
already-running database (RDS, or `docker compose up -d postgres` for local dev on
a machine without PostgreSQL binaries) — no restore, no cold start.

Normally each tenant reads `DocumentSearch/<environment>/<tenant>/`. A legacy flat
folder such as `DocumentSearch/foxtrot-Dev/` needs `AGENTIC_INDEX_PREFIX_OVERRIDE`, and
because that bypasses the per-tenant path it **must** be bound with
`AGENTIC_INDEX_PREFIX_OVERRIDE_TENANT=foxtrot-Dev/foxtrot`: startup fails without it, and
every other tenant gets a `400` instead of that tenant's documents.

### Versioning and caching

`manifest.json`'s **ETag is the snapshot version**. A warm tenant costs one HEAD
per `AGENTIC_CACHE_TTL_S`; a changed ETag triggers a re-pull. Snapshots are held
in an LRU (`AGENTIC_CACHE_MAX_TENANTS`) and evicting one reclaims its disk.

> `scripts/publish_index_snapshot.py` publishes the Postgres snapshot described
> above. The file-based layout is still read for manifests without `"backend"`.

## Tools the Document Search Agent has

| Tool | Purpose |
|---|---|
| `search_documents` | Hybrid semantic + keyword search (RRF), returns ranked passages |
| `list_documents` | Metadata-only listing with a true total — for "how many" / "list all" |
| `list_facet_values` | Valid facet values, cascaded off the enforced filter |
| `get_document_details` | Full metadata + a longer extract for one document |

The loop stops at `AGENTIC_MAX_TOOL_ITERATIONS`; on hitting it the model is asked
once more **with no tools**, so it must answer from what it gathered rather than
loop on your bill.

## Configuration

All settings use the `AGENTIC_` prefix (see `config.py`).

| Variable | Default | Notes |
|---|---|---|
| `AGENTIC_ROLE` | `orchestrator` | `orchestrator` \| `document_search` |
| `AGENTIC_REGION` | `us-east-1` | |
| `AGENTIC_INDEX_BUCKET` | — | **required** for the search agent |
| `AGENTIC_INDEX_PREFIX` | `indexes` | |
| `AGENTIC_INDEX_PREFIX_OVERRIDE` | — | one literal prefix, for exactly one tenant |
| `AGENTIC_INDEX_PREFIX_OVERRIDE_TENANT` | — | `environment/tenant` it belongs to; required with the override |
| `AGENTIC_POSTGRES_SOURCE` | `s3` | `s3` restores the dump in-container \| `dsn` uses a live database |
| `AGENTIC_POSTGRES_DSN` | — | only for `dsn` |
| `AGENTIC_POSTGRES_BIN_DIR` | image: `/usr/lib/postgresql/16/bin` | only for `s3` |
| `AGENTIC_POSTGRES_RESTORE_JOBS` | `2` | parallel `pg_restore` workers |
| `AGENTIC_DOCUMENT_URL_MODE` | `presigned` | `presigned` \| `template` \| `none` |
| `AGENTIC_DOCUMENT_URL_TTL_S` | `3600` | presigned lifetime (capped by the signing credentials) |
| `AGENTIC_DOCUMENT_URL_TEMPLATE` | — | required for `template` |
| `AGENTIC_SOURCE_BUCKET` | — | fallback when a manifest has no `source_bucket` |
| `AGENTIC_SOURCE_REGION` | — | fallback when a manifest has no `source_region`; links signed for the wrong region get HTTP 400 |
| `AGENTIC_MAX_CITATIONS` | `25` | |
| `AGENTIC_TENANT_PATTERN` / `AGENTIC_TENANT_MAP` | — | **set one** |
| `AGENTIC_SEARCH_TRANSPORT` | `agentcore` | `in_process` runs both agents in one container |
| `AGENTIC_SEARCH_AGENT_ARN` | — | required when transport is `agentcore` |
| `AGENTIC_MODEL` | — | blank inherits `BEDROCK__LLM_MODEL` (account-verified profile) |
| `AGENTIC_MAX_TOOL_ITERATIONS` | `6` | |
| `AGENTIC_CACHE_MAX_TENANTS` | `4` | snapshots held open at once |
| `AGENTIC_CACHE_TTL_S` | `300` | `0` pins the snapshot for the container's life |
| `AGENTIC_EMBED_PROVIDER` | `titan` | `fake` = fully offline |

## Run it locally

Both agents in one process (no second runtime, no ARNs):

```bash
$env:AGENTIC_SEARCH_TRANSPORT = "in_process"
$env:AGENTIC_INDEX_BUCKET     = "exampleapp-indexes"
$env:AGENTIC_TENANT_PATTERN   = '^(?P<environment>[^.]+)\.example\.io/(?P<tenant>[^/]+)'

uv run uvicorn rag_agentic.runtime.orchestrator_app:app --port 8080
```

```bash
curl localhost:8080/ping
curl -X POST localhost:8080/invocations -H 'Content-Type: application/json' \
  -d '{"base_url":"https://dev.example.io/acme","query":"Do suppliers need ISO 9001?"}'
```

That is byte-identical to what AgentCore sends.

## Deploy to AgentCore

The runtime contract is `POST /invocations`, `GET /ping` → `{"status":"Healthy"}`,
port 8080, `linux/arm64`. Build **from the repo root**:

```bash
# Document Search Agent first — the orchestrator needs its ARN.
docker build -f services/agentic/Dockerfile --platform linux/arm64 \
  --build-arg AGENT_MODULE=rag_agentic.runtime.search_app \
  -t $ECR/exampleapp-doc-search:latest .
docker push $ECR/exampleapp-doc-search:latest

docker build -f services/agentic/Dockerfile --platform linux/arm64 \
  --build-arg AGENT_MODULE=rag_agentic.runtime.orchestrator_app \
  -t $ECR/exampleapp-orchestrator:latest .
docker push $ECR/exampleapp-orchestrator:latest
```

The image installs **PostgreSQL 16 + pgvector** (matching the dumps' major version —
`pg_restore` cannot read a newer archive) and runs as the non-root `agent` user,
because PostgreSQL refuses to start as root. Behind a TLS-intercepting proxy, pass
its CA as a build **secret**, which is never written to an image layer:

```bash
docker build ... --secret id=extra_ca,src=certs/ca-bundle.pem .
```

To test the image on an amd64 machine, add `--build-arg PLATFORM=linux/amd64` — and
never push that build.

Create both runtimes (console, `agentcore` CLI, or `bedrock-agentcore-control`),
set `AGENTIC_ROLE` per runtime, then put the search runtime's ARN into the
orchestrator's `AGENTIC_SEARCH_AGENT_ARN`.

**IAM.** The search agent's execution role needs `s3:GetObject` + `s3:ListBucket`
on its index prefix, `bedrock:InvokeModel` (Converse + Titan embeddings), and
`s3:GetObject` on the **source documents** prefix — a presigned URL only works if
the role that signed it could read the object itself, otherwise the link returns 403. The
orchestrator's role needs `bedrock:InvokeModel` and
`bedrock-agentcore:InvokeAgentRuntime` on the search runtime's ARN — and no S3 at
all, since it never retrieves.

## Tests

```bash
uv run python -m pytest services/agentic/tests -q     # fully offline
```

No AWS and no model calls: `ScriptedConverseModel` replays turns, `FakeEmbedder`
gives deterministic vectors, and a fake S3 serves **real** snapshots built with
the production store classes — so tenancy → download → open → search → answer is
exercised end to end.

Postgres tests need `docker compose up -d postgres` (they use a separate `rag_test`
database and skip without it). The real `initdb` → `pg_restore` round trip needs
PostgreSQL binaries and a non-root user, so `test_embedded_pg.py` runs inside the
agent image and skips elsewhere:

```bash
docker run --rm -v "$PWD/services/agentic/tests:/app/services/agentic/tests" \
  exampleapp-agent:local-amd64 sh -c 'uv pip install -q --python /app/.venv/bin/python \
    --target /tmp/pydeps pytest pytest-asyncio && \
  PYTHONPATH=/tmp/pydeps python -m pytest services/agentic/tests/test_embedded_pg.py -v'
```
