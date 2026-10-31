# syntax=docker/dockerfile:1
#
# Lambda container image for the scheduled index jobs (rag_search.index_jobs):
# incremental indexing of new S3 documents and the DeleteMark purge, for every
# tenant collection in INDEX_TARGETS.
#
# A container image rather than a zip: the processing stack (pdfplumber, docx,
# openpyxl, qdrant-client, pymongo, ...) is well past the 250 MB zip limit.
#
# Build from the REPO ROOT, for arm64 (the template's Architectures):
#
#   docker build -f deploy/index-jobs/Dockerfile --platform linux/arm64 \
#     --provenance=false -t exampleapp-index-jobs:latest .
#
# --provenance=false: Lambda rejects the multi-manifest index that BuildKit's
# default attestations produce. Behind the TLS-intercepting proxy, add
#   --secret id=extra_ca,src=certs/ca-bundle.pem
# (mounted only while downloading; never written to a layer).

ARG PLATFORM=linux/arm64
FROM --platform=${PLATFORM} public.ecr.aws/lambda/python:3.12

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /build

# Workspace metadata first, so a code-only change does not re-resolve dependencies.
# Every member's pyproject is needed for the lockfile to resolve.
COPY pyproject.toml uv.lock ./
COPY packages/core/pyproject.toml packages/core/README.md packages/core/
COPY services/processing/pyproject.toml services/processing/README.md services/processing/
COPY services/search/pyproject.toml services/search/README.md services/search/
COPY services/ingestion/pyproject.toml services/ingestion/README.md services/ingestion/
COPY services/agentic/pyproject.toml services/agentic/README.md services/agentic/

RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -s /run/secrets/extra_ca ]; then export SSL_CERT_FILE=/run/secrets/extra_ca; fi; \
    uv export --frozen --no-dev --package rag-search --no-emit-workspace --no-hashes \
      -o /tmp/requirements.txt \
 && uv pip install --python /var/lang/bin/python --target "${LAMBDA_TASK_ROOT}" \
      -r /tmp/requirements.txt

# Only the members the jobs import: rag_search -> rag_processing + rag_core.
COPY packages/core packages/core
COPY services/processing services/processing
COPY services/search services/search
RUN uv pip install --python /var/lang/bin/python --target "${LAMBDA_TASK_ROOT}" --no-deps \
      ./packages/core ./services/processing ./services/search \
 && rm -rf /build

WORKDIR ${LAMBDA_TASK_ROOT}

# The only writable path in Lambda; the factory creates its scratch files here.
ENV SEARCH_DATA_DIR=/tmp/search

CMD ["rag_search.index_jobs.lambda_handler"]
