# Search Service

Local, **free** hybrid search over compliance documents — the read path
(platform capabilities #3, #5, #6, #7, #9, #10) as a single runnable app with a UI.

```
S3 (foxtrot/) → extract → chunk (parent/child) → embed (Titan)
        → Qdrant (embedded, on-disk) + SQLite FTS5 (keyword)
query   → [vector ‖ keyword] → Reciprocal Rank Fusion → ranked passages → UI
```

## Free & local by design

| Concern | Backend | Why free/local |
|---|---|---|
| Vectors | **Qdrant embedded** (`QdrantClient(path=…)`) | Runs in-process, persists to a folder. No server, no Docker, no cost. |
| Keyword / BM25 | **SQLite FTS5** | Standard library. One file on disk. |
| Fusion | **RRF** | In-process. |
| Embeddings | **Bedrock Titan** (default) or a local fake | Titan = your existing Bedrock access; `SEARCH_EMBED_PROVIDER=fake` = fully offline. |

No Pinecone, no OpenSearch Serverless anywhere.

## Run it

```bash
# 1. (optional) fully offline mode — no AWS, deterministic fake embeddings:
$env:SEARCH_EMBED_PROVIDER = "fake"

# 2. provision the credit-metering tables (idempotent; safe to re-run)
uv run python scripts/migrate_credits.py

# 3. start the UI + API
uv run uvicorn rag_search.app:app --port 8004
# open http://localhost:8004
```

In the UI: set the S3 prefix (`foxtrot/`) → **Index documents** → then search. With real
Bedrock Titan embeddings, first configure AWS creds in `.env`; indexing pulls from
`s3://example-index-bucket/foxtrot/`.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Search UI |
| POST | `/v1/index` | Build the index from S3 (`{prefix, reset, max_docs}`) |
| GET | `/v1/search?q=…&k=…` | Hybrid search → ranked passages |
| POST | `/v1/chat` | Conversational RAG answer — **credit-metered** (402 when out of credits) |
| GET | `/v1/stats` | Index sizes |
| GET | `/api/credits/balance` | `{balance, plan_total, percent_remaining}` for the current user |
| POST | `/api/credits/run` | Run a prompt against Claude (Sonnet) and deduct real-usage credits |
| POST | `/api/credits/reset` | **Admin:** reset the shared pool to full (`SEARCH_CREDIT_PLAN_TOTAL`, default 1000) |
| POST | `/api/credits/upgrade` | Move the pool to the Pro plan (5× credits) |
| GET | `/healthz` `/readyz` `/metrics` | probes + Prometheus |

## Credit metering

Every prompt against Claude deducts **credits** from a per-user balance, based on
the real token usage returned by Bedrock plus a **50% overhead**:

```
base_cost    = i/1e6 * INPUT_RATE + o/1e6 * OUTPUT_RATE     # USD
charged      = base_cost * (1 + OVERHEAD)                   # +50%
credits_used = charged / CREDIT_VALUE                       # 1000 credits = $1
```

All rates live in one file — [`credit_config.py`](src/rag_search/credit_config.py)
(`INPUT_RATE=2.00`, `OUTPUT_RATE=10.00`, `OVERHEAD=0.50`, `CREDIT_VALUE=0.001`).

**One shared pool.** Every browser/user draws down a single balance (default
**1000 credits**, `SEARCH_CREDIT_PLAN_TOTAL`), so when anyone runs a prompt the
remaining credits drop for everyone — each open page polls `GET /api/credits/balance`
every ~5s and converges on the shared number. The balance is **server-authoritative**
(SQLite, `data/search/credits.db`): the client renders it only from the server
response, so a browser refresh shows the stored balance, never a client value. By
default the pool **refills to its starting balance on each app restart**
(`SEARCH_CREDIT_RESET_ON_START=true`) — credits survive refreshes but reset on
restart; set it `false` to carry the balance across restarts (the JSON audit log is
kept either way).
Deduction happens inside a row-locked (`BEGIN IMMEDIATE`) transaction, so concurrent
requests can't double-spend or go negative. A per-browser cookie (or `X-User-Id` /
`X-User-Email` header) tags **who** spent — recorded in `credit_transactions` and in
a JSON usage log (`data/search/credits_usage.jsonl`, one object per charge) — but the
balance itself is shared. At **≥90% used** the UI shows a "running low" banner; at
**0** it shows an out-of-credits dialog offering the Pro plan (5× credits, 5× search).

Each result tells you **which document** (S3 key), **which page/section**, the
fused score, and **why it matched** (`vector` / `keyword` / `both`).

## Config (`SEARCH_` prefix)

| Var | Default | Meaning |
|---|---|---|
| `SEARCH_DATA_DIR` | `./data/search` | Where Qdrant + SQLite live. |
| `SEARCH_EMBED_PROVIDER` | `titan` | `titan` or `fake` (offline). |
| `SEARCH_CHILD_WORDS` | 180 | Child chunk size (retrieved unit). |
| `SEARCH_TOP_K` | 8 | Results returned. |
| `SEARCH_RRF_K` | 60 | Fusion damping. |
| `SEARCH_CREDIT_PLAN_TOTAL` | 1000 | Starting size of the shared credit pool (1000 = $1.00). |

## Tests

```bash
uv run pytest services/search -q
```
Includes an end-to-end test: index two real documents (offline) and assert a
natural-language query returns the correct one.
