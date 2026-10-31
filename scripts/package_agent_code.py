"""Build the AgentCore code-deployment zip for the ExampleApp agents.

AgentCore's managed Python runtime executes ``runtime.py`` from a zip that must
carry everything it imports -- the runtime installs nothing -- so this vendors the
dependencies as **linux aarch64** wheels (AgentCore is Graviton) next to the
project's own packages:

    deployment.zip
        runtime.py            entrypoint (services/agentic/deploy/runtime.py)
        requirements.txt      what was vendored, for the record
        rag_agentic/ rag_search/ rag_core/     this project
        <third-party packages>

One zip serves both agents; AGENTIC_ROLE picks which one a runtime runs.

``rag_processing`` is deliberately NOT included: it is the indexing/extraction side
(pdfplumber, python-docx, OCR) and the agents never import it -- it would add
hundreds of MB to a package that has to be downloaded into every runtime.

Usage
-----
    uv run python scripts/package_agent_code.py
    uv run python scripts/package_agent_code.py --upload --bucket example-index-bucket

AgentCore fetches the code package from the same region it runs in, so the bucket must
be us-east-1. Behind a TLS-inspecting proxy, set ``UV_NATIVE_TLS=1`` so the vendoring
step trusts the system certificate store.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "services" / "agentic" / "deploy" / "runtime.py"
SOURCE_PACKAGES = (
    ROOT / "services" / "agentic" / "src" / "rag_agentic",
    ROOT / "services" / "search" / "src" / "rag_search",
    ROOT / "packages" / "core" / "src" / "rag_core",
)
# Exactly what the two agents import at runtime. Kept explicit rather than resolved
# from the workspace, so the package cannot quietly grow the indexing stack.
# The app serves AgentCore's /invocations and /ping itself (FastAPI + uvicorn), so
# the bedrock-agentcore SDK is deliberately absent: the only release that resolves
# alongside aioboto3 (1.3.2) serialises a handler's error Response into a 200, which
# would turn "no snapshot for this tenant" into a successful-looking answer.
# Versions are unpinned so the resolver can agree on one botocore.
REQUIREMENTS = """\
fastapi
uvicorn
aioboto3
boto3
pydantic
pydantic-settings
structlog
tenacity
qdrant-client
prometheus-client
# rag_search.stores imports the Mongo backend eagerly, so the agent needs the
# driver even though it reads snapshots rather than Mongo. Without it the runtime
# dies on import and AgentCore reports it as an initialisation timeout.
pymongo
# The Licensing credit calls (reserve/settle) are plain HTTPS. httpx already
# arrives via FastAPI, but an agent that cannot settle credits is broken, so it is
# named here rather than left to a transitive resolution that could drop it.
httpx
"""
# AgentCore runs arm64; wheels must match the target, not this machine.
TARGET_PLATFORM = "aarch64-manylinux2014"
TARGET_PYTHON = "3.12"
# Trimmed from vendored wheels: never imported at runtime, only bulk.
PRUNE = ("*.dist-info/RECORD", "**/__pycache__", "**/*.pyc", "**/tests", "**/test")


def _run(*command: str) -> None:
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.exit(f"command failed: {' '.join(command)}\n{(proc.stderr or proc.stdout)[-2500:]}")


def build(build_dir: Path) -> Path:
    package = build_dir / "package"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    package.mkdir(parents=True)

    requirements = build_dir / "requirements.txt"
    requirements.write_text(REQUIREMENTS, encoding="utf-8")
    print(f"vendoring dependencies for {TARGET_PLATFORM} / python {TARGET_PYTHON} ...")
    _run(
        "uv", "pip", "install",
        "--target", str(package),
        "--python-platform", TARGET_PLATFORM,
        "--python-version", TARGET_PYTHON,
        # A wheel built for this machine would not run on AgentCore, so refuse to
        # fall back to building from source.
        "--only-binary", ":all:",
        "-r", str(requirements),
    )
    for source in SOURCE_PACKAGES:
        shutil.copytree(source, package / source.name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(ENTRYPOINT, package / "runtime.py")
    shutil.copy2(requirements, package / "requirements.txt")

    for pattern in PRUNE:
        for path in package.glob(pattern):
            shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(missing_ok=True)

    archive = build_dir / "deployment.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(package).as_posix())
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default="./data/agent-package")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--bucket", default="example-docs-bucket")
    parser.add_argument("--key", default="agent_core/exampleapp_docsearch/deployment.zip")
    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    archive = build(build_dir)
    unpacked = sum(f.stat().st_size for f in (build_dir / "package").rglob("*") if f.is_file())
    print(f"\n  {archive}")
    print(f"  zipped   {archive.stat().st_size / 1_048_576:7.1f} MiB")
    print(f"  unpacked {unpacked / 1_048_576:7.1f} MiB")
    biggest = sorted(
        ((sum(f.stat().st_size for f in d.rglob('*') if f.is_file()), d.name)
         for d in (build_dir / "package").iterdir() if d.is_dir()),
        reverse=True,
    )[:8]
    for size, name in biggest:
        print(f"     {name:36} {size / 1_048_576:7.1f} MiB")

    if not args.upload:
        print("\n  (not uploaded; pass --upload)")
        return

    import boto3

    from rag_core.config import get_settings

    aws = get_settings().aws
    secret = getattr(aws.secret_access_key, "get_secret_value", lambda: aws.secret_access_key)()
    s3 = boto3.client(
        "s3",
        region_name=aws.region,
        aws_access_key_id=aws.access_key_id or None,
        aws_secret_access_key=secret or None,
        verify=aws.ca_bundle or True,
    )
    s3.upload_file(str(archive), args.bucket, args.key)
    print(f"\n  uploaded s3://{args.bucket}/{args.key}")


if __name__ == "__main__":
    main()
