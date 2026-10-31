# Multi-stage build for the processing service.
# Build with the repository root as context:
#   docker build -f services/processing/Dockerfile -t rag-processing:dev .

# ---- Stage 1: builder ------------------------------------------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.4 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml ./
COPY packages/core/pyproject.toml packages/core/
COPY services/processing/pyproject.toml services/processing/

COPY packages/core packages/core
COPY services/processing services/processing

RUN uv sync --package rag-processing --no-dev

# ---- Stage 2: runtime ------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# libstdc++ etc. are present in slim; the parser deps (lxml/pdfplumber/Pillow)
# ship manylinux wheels, so no extra system packages are required. (If you switch
# OCR to the local Tesseract provider, add: tesseract-ocr poppler-utils here.)
RUN groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app
COPY --from=builder --chown=app:app /app /app

USER app
EXPOSE 8002

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8002/healthz').status==200 else 1)"]

CMD ["uvicorn", "rag_processing.app:app", "--host", "0.0.0.0", "--port", "8002"]
