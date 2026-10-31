"""Entrypoint for the Document Search runtime.

    uvicorn rag_agentic.runtime.search_app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

from rag_agentic.config import AgentRole
from rag_agentic.runtime.app import create_app

app = create_app(AgentRole.DOCUMENT_SEARCH)
