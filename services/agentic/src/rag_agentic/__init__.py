"""Agentic AI approach over the ExampleApp compliance corpus.

Two agents, deployed as two Amazon Bedrock AgentCore Runtimes:

* ``OrchestratorAgent``   -- the entry point ExampleApp calls. Owns routing, not
  retrieval: it decides whether a request needs the document corpus and delegates,
  or answers directly (greetings, capability questions, clarifications).
* ``DocumentSearchAgent`` -- a leaf agent scoped to exactly one tenant. Plans a
  query, calls retrieval tools, reads what comes back, and loops until it can
  answer with citations.

Both run a plain Bedrock Converse tool loop over ``rag_core``'s resilience layer
(retry + circuit breaker + the platform's error taxonomy) rather than an agent
framework, and both reuse ``rag_search``'s stores for retrieval.

Two invariants are enforced in code, never asked of the model: the tenant is
resolved from the caller's ``base_url`` by configuration alone (see ``tenancy``),
and an optional ``BusinessObject`` is applied as a hard pre-filter to every query
before ranking (see ``tools``).

Layout:
    config.py       AGENTIC_* settings
    tenancy.py      base_url -> (environment, tenant)   [security boundary]
    index_store.py  per-tenant S3 snapshots, cached locally
    converse.py     tool-capable Bedrock Converse binding (+ offline fake)
    agent_loop.py   model <-> tool loop
    tools.py        retrieval tools, tenant- and BusinessObject-scoped
    agents/         the orchestrator and the search agent
    runtime/        the AgentCore HTTP contract (/invocations, /ping)
"""

from __future__ import annotations

__version__ = "0.1.0"
