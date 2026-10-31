"""rag_core — shared foundation for the Agentic RAG platform.

This package is dependency-light and side-effect-free on import: importing
``rag_core`` must never open a network connection, read a file, or configure
global logging. Wiring is explicit (see ``rag_core.di``) so services control
their own lifecycle and tests stay hermetic.
"""

from __future__ import annotations

__version__ = "0.1.0"
