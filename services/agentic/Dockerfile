# syntax=docker/dockerfile:1
#
# AgentCore Runtime image for both agents.
#
# AgentCore requires linux/arm64, so that is the default platform rather than
# whatever the build host is -- an amd64 image is accepted at push time and then
# fails to start, which is a slow way to find out. Override PLATFORM only to test
# the image locally on an amd64 machine; never push such a build.
#
# One image serves both roles; AGENT_MODULE (build arg) or AGENTIC_ROLE (env)
# selects which. Build it from the REPO ROOT, not this directory:
#
#   docker build -f services/agentic/Dockerfile --platform linux/arm64 \
#     --build-arg AGENT_MODULE=rag_agentic.runtime.orchestrator_app \
#     -t exampleapp-orchestrator:latest .
#
#   docker build -f services/agentic/Dockerfile --platform linux/arm64 \
#     --build-arg AGENT_MODULE=rag_agentic.runtime.search_app \
#     -t exampleapp-doc-search:latest .
#
# Behind a TLS-intercepting proxy, pass its CA bundle as a build SECRET:
#
#   docker build ... --secret id=extra_ca,src=certs/ca-bundle.pem .
#
# A secret is mounted only for the steps that download, and is never written to an
# image layer -- unlike COPYing a certificate in. Builds without the proxy (CI)
# simply omit it.

ARG PLATFORM=linux/arm64
FROM --platform=${PLATFORM} python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# ---- PostgreSQL 16 + pgvector ------------------------------------------------
# The document index lives in S3 as a pg_dump. The search agent restores it into a
# PostgreSQL server private to this container (rag_agentic.embedded_pg), so S3 is
# the only storage. Major version 16 to match the dumps: pg_restore cannot read an
# archive written by a newer pg_dump. From the PGDG repo, which ships arm64 builds
# of both PostgreSQL and pgvector.
RUN --mount=type=secret,id=extra_ca,required=false \
    # apt downloads as the unprivileged _apt user, which cannot read the root-only
    # secret mount; a readable copy is made for this step and removed in it.
    CA=/tmp/extra_ca.pem; APT_CA=""; CURL_CA=""; \
    if [ -s /run/secrets/extra_ca ]; then \
      cp /run/secrets/extra_ca "$CA" && chmod 644 "$CA"; \
      APT_CA="-o Acquire::https::CAInfo=$CA"; CURL_CA="--cacert $CA"; \
    fi; \
    apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl gnupg postgresql-common \
 # Only the binaries are wanted; the server is initialised at runtime under /tmp.
 && sed -i 's/^#\?\s*create_main_cluster.*/create_main_cluster = false/' /etc/postgresql-common/createcluster.conf \
 && install -d /usr/share/postgresql-common/pgdg \
 && curl -fsSL $CURL_CA -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc \
      https://www.postgresql.org/media/keys/ACCC4CF8.asc \
 && . /etc/os-release \
 && echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt ${VERSION_CODENAME}-pgdg main" \
      > /etc/apt/sources.list.d/pgdg.list \
 && apt-get $APT_CA update \
 && apt-get $APT_CA install -y --no-install-recommends postgresql-16 postgresql-16-pgvector \
 && apt-get purge -y curl gnupg \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/* "$CA"

ENV AGENTIC_POSTGRES_BIN_DIR=/usr/lib/postgresql/16/bin

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Workspace metadata first, so a code-only change does not re-resolve the graph.
COPY pyproject.toml uv.lock ./
COPY packages/core/pyproject.toml packages/core/
COPY services/processing/pyproject.toml services/processing/
COPY services/search/pyproject.toml services/search/
COPY services/agentic/pyproject.toml services/agentic/

# Only the members this agent actually imports: rag_agentic -> rag_search ->
# rag_processing + rag_core. The ingestion service is deliberately excluded.
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -s /run/secrets/extra_ca ]; then export SSL_CERT_FILE=/run/secrets/extra_ca; fi; \
    uv sync --locked --no-dev \
      --package rag-agentic \
      --no-install-workspace || \
    uv sync --no-dev --package rag-agentic --no-install-workspace

COPY packages/core packages/core
COPY services/processing services/processing
COPY services/search services/search
COPY services/agentic services/agentic

RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -s /run/secrets/extra_ca ]; then export SSL_CERT_FILE=/run/secrets/extra_ca; fi; \
    uv sync --locked --no-dev --package rag-agentic || uv sync --no-dev --package rag-agentic

ENV PATH="/app/.venv/bin:${PATH}"

# Snapshots are materialised here; must be writable and is wiped between runs. The
# embedded PostgreSQL data directory lives under it too (…/tenants/_postgres).
ENV AGENTIC_CACHE_DIR=/tmp/rag-agentic/tenants

# PostgreSQL refuses to start as root, and nothing here needs root at runtime.
RUN useradd --create-home --uid 10001 agent
USER agent

ARG AGENT_MODULE=rag_agentic.runtime.orchestrator_app
ENV AGENT_MODULE=${AGENT_MODULE}

# AgentCore always invokes port 8080 and probes GET /ping.
EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn ${AGENT_MODULE}:app --host 0.0.0.0 --port 8080"]
