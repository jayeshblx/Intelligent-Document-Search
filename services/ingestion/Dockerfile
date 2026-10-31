# Multi-stage build for the ingestion service.
#
# IMPORTANT: build with the **repository root** as context so the workspace
# dependency (packages/core) is reachable:
#
#   docker build -f services/ingestion/Dockerfile -t rag-ingestion:dev .
#
# Stage 1 resolves+installs deps into an isolated venv; stage 2 is a slim runtime
# carrying only the venv + source (no uv, no build tools) and runs as non-root.

# ---- Stage 1: builder ------------------------------------------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.4 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Copy manifests first for better layer caching of the dependency install.
COPY pyproject.toml ./
COPY packages/core/pyproject.toml packages/core/
COPY services/ingestion/pyproject.toml services/ingestion/

# Copy sources needed to build the workspace members.
COPY packages/core packages/core
COPY services/ingestion services/ingestion

# Install only the ingestion service + its deps (incl. rag-core) into /app/.venv.
RUN uv sync --package rag-ingestion --no-dev

# ---- Stage 2: runtime ------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Non-root runtime user.
RUN groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app
COPY --from=builder --chown=app:app /app /app

USER app
EXPOSE 8001

# Liveness via the app's own probe (no curl in slim image).
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8001/healthz').status==200 else 1)"]

CMD ["uvicorn", "rag_ingestion.app:app", "--host", "0.0.0.0", "--port", "8001"]
