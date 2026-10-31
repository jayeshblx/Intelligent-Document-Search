"""Lightweight dependency-injection container.

Rather than a heavyweight DI framework, we use a small typed container that owns
the lifecycle of shared resources (AWS clients, vector store). Benefits: explicit
wiring, trivial to mock in tests (construct with fakes), and a single ``aclose``
that tears everything down — which is exactly what FastAPI lifespan / Temporal
worker shutdown need.
"""

from __future__ import annotations

from rag_core.di.container import Container

__all__ = ["Container"]
