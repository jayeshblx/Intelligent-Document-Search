"""AgentCore **code deployment** entrypoint — the same two agents, no container.

AgentCore runs this file (``entryPoint: ["runtime.py"]``) on its managed Python
runtime, from a zip holding this file plus the project packages and their Linux
dependencies. ``AGENTIC_ROLE`` decides which agent a runtime is, so both runtimes
deploy from one package.

It serves AgentCore's contract with the app this project already has
(``rag_agentic.runtime.app``): ``POST /invocations``, ``GET /ping`` ->
``{"status": "Healthy"}``, port 8080. Identical behaviour to the container image --
same envelope, same citations, and the same error STATUS codes.

Why not the ``bedrock-agentcore`` SDK: its releases that can serve a handler's own
``Response`` require ``botocore>=1.43``, while ``aioboto3`` (the async AWS client
this project's core is built on) pins ``botocore<1.40.62``. The only version of the
SDK that resolves alongside it is 1.3.2, which *serialises whatever the handler
returns into a 200* -- so a "no snapshot for this tenant" failure would reach the
caller as a successful-looking answer. Serving the FastAPI app directly keeps 400
and 503 intact and drops a dependency conflict from the package.

Both shapes an ASGI host might expect are provided: module-level ``app`` for a host
that imports this file, and a uvicorn server when it is executed as a script.
"""

from __future__ import annotations

import os

from rag_agentic.runtime.app import create_app

# AgentCore sets no ASGI entry of its own: this is the application it talks to.
app = create_app()


def main() -> None:
    import uvicorn

    # AgentCore always invokes port 8080 and probes GET /ping.
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), log_level="info")


if __name__ == "__main__":
    main()
