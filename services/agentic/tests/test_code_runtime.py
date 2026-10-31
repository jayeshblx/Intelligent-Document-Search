"""The AgentCore code-deployment entrypoint (deploy/runtime.py).

It replaces the container image, not the app: AgentCore's contract is
``POST /invocations`` + ``GET /ping`` on 8080, which ``rag_agentic.runtime.app``
already serves, so the deployed agents keep the same envelope AND the same error
status codes (a 503 must never arrive as a 200 with an apology in ``answer``).

These tests pin what the packaging depends on: the file exposes an ASGI ``app``,
that app is this project's runtime app, and importing it starts no server.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from rag_agentic.config import AgentRole

RUNTIME_PY = Path(__file__).resolve().parents[1] / "deploy" / "runtime.py"


def _load(monkeypatch: Any, role: str) -> Any:
    monkeypatch.setenv("AGENTIC_ROLE", role)
    spec = importlib.util.spec_from_file_location(f"exampleapp_code_runtime_{role}", RUNTIME_PY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exposes_an_asgi_app_without_starting_a_server(monkeypatch: Any) -> None:
    """Importing must not block: a host that imports the file would never finish."""
    started: list[str] = []
    module = _load(monkeypatch, "orchestrator")
    monkeypatch.setattr(module, "main", lambda: started.append("server"))

    assert module.app is not None
    assert hasattr(module.app, "router")  # an ASGI/Starlette app
    assert started == []


@pytest.mark.parametrize("role", ["orchestrator", "document_search"])
def test_serves_agentcores_health_probe(monkeypatch: Any, role: str) -> None:
    """AgentCore marks a runtime unhealthy on anything but this exact shape."""
    module = _load(monkeypatch, role)
    # No context manager: that would run the lifespan, which builds the real
    # component graph and needs AWS. /ping must answer without any of that.
    response = TestClient(module.app).get("/ping")

    assert response.status_code == 200
    assert response.json() == {"status": "Healthy"}


def test_the_role_env_var_selects_the_agent(monkeypatch: Any) -> None:
    """One package, two runtimes: AGENTIC_ROLE is what differs between them."""
    from rag_agentic.config import AgenticSettings

    monkeypatch.setenv("AGENTIC_ROLE", "document_search")
    assert AgenticSettings().role is AgentRole.DOCUMENT_SEARCH
    monkeypatch.setenv("AGENTIC_ROLE", "orchestrator")
    assert AgenticSettings().role is AgentRole.ORCHESTRATOR


def test_entrypoint_file_is_what_the_packager_ships() -> None:
    """scripts/package_agent_code.py copies this file to the zip root as runtime.py."""
    packager = (Path(__file__).resolve().parents[3] / "scripts" / "package_agent_code.py").read_text(
        encoding="utf-8"
    )
    assert 'ENTRYPOINT = ROOT / "services" / "agentic" / "deploy" / "runtime.py"' in packager
    assert 'package / "runtime.py"' in packager
