"""Dynamic test-automation updater — runs in CI on every push.

WHAT IT DOES
------------
1. Computes the set of source files that changed in the push (git diff).
2. For each changed source module, asks an LLM to:
     - summarise what changed, and
     - write or update the corresponding pytest test file.
3. Writes the generated test files to disk (a later CI step opens a PR).
4. Runs the test suite and captures the result.
5. Records token usage + cost for every model call.

Everything is emitted to ``ci_report.json`` for the e-mail step to consume.

PROVIDERS
---------
``LLM_PROVIDER=ollama`` (default) — a local Llama model over Ollama's HTTP API.
Runs on a self-hosted GitHub Actions runner, so inference happens on this
machine: no per-token billing, no data leaving the host. Cost is reported as
$0.00 (tokens are still counted, for visibility into prompt size).

``LLM_PROVIDER=bedrock`` — Claude on Amazon Bedrock via the ``converse`` API,
mirroring the call shape already used in ``rag_core.aws.bedrock``. Kept as a
drop-in alternative for when test quality matters more than cost.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

# --- Configuration ----------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

PROVIDER = (os.environ.get("LLM_PROVIDER") or "ollama").lower()

# --- Ollama (local, free) ---
OLLAMA_HOST = os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL") or "llama3.1:8b"
# Llama needs an explicit context window; Ollama's default is far too small to
# hold a source file + its diff + the existing test file.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))
OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "4096"))
OLLAMA_TIMEOUT_S = int(os.environ.get("OLLAMA_TIMEOUT_S", "600"))

# --- Bedrock (cloud, paid) ---
BEDROCK_MODEL = os.environ.get("BEDROCK_LLM_MODEL") or "us.anthropic.claude-haiku-4-5-20251001-v1:0"
BEDROCK_REGION = os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION") or "us-east-1"

# USD per 1,000,000 tokens for the Bedrock path, keyed by a substring of the id.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "haiku-4-5": (1.0, 5.0),
    "sonnet": (3.0, 15.0),
    "opus": (5.0, 25.0),
}

# Bound the blast radius (and runtime) of a huge push.
MAX_FILES = int(os.environ.get("MAX_FILES", "12"))
# Local 8B models are slower and have less context than Claude, so keep the
# per-file payload tighter than a cloud model would need.
MAX_SOURCE_CHARS = int(os.environ.get("MAX_SOURCE_CHARS", "8000"))
MAX_TEST_CHARS = int(os.environ.get("MAX_TEST_CHARS", "5000"))
# How many generate -> run -> repair rounds before giving up on a file.
MAX_CODE_ATTEMPTS = int(os.environ.get("MAX_CODE_ATTEMPTS", "3"))
PYTEST_TIMEOUT_S = int(os.environ.get("PYTEST_TIMEOUT_S", "180"))

SOURCE_PREFIXES = ("packages/", "services/")
SKIP_BASENAMES = {"__init__.py", "py.typed", "conftest.py"}

# Which test layers to generate. A QA engineer covers more than unit tests, but
# each layer has a different shape, so the model is prompted differently per kind.
#   unit -> plain pytest against a module
#   api  -> httpx ASGITransport against FastAPI routes (in-process, no server)
#   ui   -> Playwright against the web UI (opt-in: needs browsers installed)
ENABLE_API_TESTS = os.environ.get("ENABLE_API_TESTS", "true").lower() == "true"
ENABLE_UI_TESTS = os.environ.get("ENABLE_UI_TESTS", "false").lower() == "true"


@dataclass
class Analysis:
    """The model's verdict for one changed source file."""

    source: str
    summary: str
    test_path: str
    action: str  # "created" | "updated" | "skipped"
    new_test_cases: list[str] = field(default_factory=list)
    notes: str = ""
    kind: str = "unit"  # "unit" | "api" | "ui"


@dataclass
class Usage:
    """Running token accumulator across all model calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, in_tok: int, out_tok: int) -> None:
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.calls += 1


# --- Providers --------------------------------------------------------------

# The work is split across two model calls, and this schema covers only the
# first (the decision). WHY: asking a small local model to embed a whole test
# module inside a JSON string field reliably fails — it either returns an empty
# string or mangles the newline/quote escaping. Generating prose-free code as
# plain text is a much easier job for an 8B model, so the code comes back on a
# second, unconstrained call.
ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "action": {"type": "string", "enum": ["created", "updated", "skipped"]},
        "new_test_cases": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "required": ["summary", "action", "new_test_cases", "notes"],
}


class Provider(Protocol):
    name: str
    model: str

    def complete(
        self, system: str, prompt: str, schema: dict | None = None
    ) -> tuple[str, int, int]:
        """Return (text, input_tokens, output_tokens). Constrain to ``schema`` if given."""

    def cost_usd(self, usage: Usage) -> float: ...


class OllamaProvider:
    """Local Llama via Ollama's /api/chat. Uses stdlib HTTP — no extra deps."""

    name = "ollama"

    def __init__(self) -> None:
        self.model = OLLAMA_MODEL

    def complete(
        self, system: str, prompt: str, schema: dict | None = None
    ) -> tuple[str, int, int]:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0,
                "num_ctx": OLLAMA_NUM_CTX,
                # Without an explicit cap Ollama stops early on long code output.
                "num_predict": OLLAMA_NUM_PREDICT,
            },
        }
        if schema is not None:
            # Ollama structured outputs: constrain generation to the schema.
            payload["format"] = schema
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_S) as resp:
            data = json.loads(resp.read())
        text = data.get("message", {}).get("content", "")
        return text, int(data.get("prompt_eval_count", 0)), int(data.get("eval_count", 0))

    def cost_usd(self, usage: Usage) -> float:
        return 0.0  # local inference — electricity only


class BedrockProvider:
    """Claude on Bedrock via converse — same shape as rag_core.aws.bedrock."""

    name = "bedrock"

    def __init__(self) -> None:
        import boto3  # imported lazily so the Ollama path needs no AWS deps
        from botocore.config import Config as BotoConfig

        self.model = BEDROCK_MODEL
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=BEDROCK_REGION,
            config=BotoConfig(retries={"max_attempts": 6, "mode": "adaptive"}),
        )

    def complete(
        self, system: str, prompt: str, schema: dict | None = None
    ) -> tuple[str, int, int]:
        # Bedrock's converse has no schema-constrain knob here; the prompt
        # already states the required shape and Claude follows it reliably.
        resp = self._client.converse(
            modelId=self.model,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 4096, "temperature": 0.0},
        )
        tok = resp.get("usage", {})
        text = resp["output"]["message"]["content"][0]["text"]
        return text, int(tok.get("inputTokens", 0)), int(tok.get("outputTokens", 0))

    def cost_usd(self, usage: Usage) -> float:
        in_rate, out_rate = next(
            (r for k, r in PRICING_PER_MTOK.items() if k in self.model),
            PRICING_PER_MTOK["haiku-4-5"],
        )
        return (usage.input_tokens / 1_000_000) * in_rate + (
            usage.output_tokens / 1_000_000
        ) * out_rate


def build_provider() -> Provider:
    if PROVIDER == "bedrock":
        return BedrockProvider()
    if PROVIDER == "ollama":
        return OllamaProvider()
    raise SystemExit(f"unknown LLM_PROVIDER={PROVIDER!r} (expected 'ollama' or 'bedrock')")


# --- Git --------------------------------------------------------------------

def _run(cmd: list[str]) -> str:
    return subprocess.run(
        cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout


def _resolve_base(before: str, head: str) -> str:
    """Pick a valid base commit to diff against.

    On a brand-new branch ``github.event.before`` is all-zeros (no prior
    commit), so fall back to the push's first parent, then the root commit.
    """
    if before and not re.fullmatch(r"0+", before):
        try:
            _run(["git", "cat-file", "-e", f"{before}^{{commit}}"])
            return before
        except subprocess.CalledProcessError:
            pass
    try:
        return _run(["git", "rev-parse", f"{head}~1"]).strip()
    except subprocess.CalledProcessError:
        return _run(["git", "rev-list", "--max-parents=0", head]).strip().splitlines()[0]


def changed_source_files(base: str, head: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Return (analysable source files, all changed name/status pairs)."""
    out = _run(["git", "diff", "--name-status", base, head])
    all_changes: list[tuple[str, str]] = []
    sources: list[str] = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status, path = parts[0], parts[-1]
        all_changes.append((status, path))
        if status.startswith("D") or not path.endswith(".py"):
            continue
        if not path.startswith(SOURCE_PREFIXES) or "/src/" not in path:
            continue
        if "/tests/" in path or Path(path).name in SKIP_BASENAMES:
            continue
        sources.append(path)
    return sources, all_changes


def _diff_for(base: str, head: str, path: str) -> str:
    try:
        return _run(["git", "diff", base, head, "--", path])
    except subprocess.CalledProcessError:
        return ""


def _test_path_for(source: str) -> Path:
    """``<component>/src/<pkg>/.../name.py`` -> ``<component>/tests/test_name.py``."""
    p = Path(source)
    parts = p.parts
    src_idx = parts.index("src")
    return REPO_ROOT.joinpath(*parts[:src_idx]) / "tests" / f"test_{p.stem}.py"


def _module_path_for(source: str) -> str:
    """``packages/core/src/rag_core/utils/checksum.py`` -> ``rag_core.utils.checksum``.

    This repo uses a src layout: packages are installed and imported by their
    distribution name, never by filesystem path. A small local model cannot
    infer that, and left to itself will emit
    ``from packages.core.src.rag_core...`` — which does not import. So we
    compute the real dotted path and hand it to the model.
    """
    p = Path(source)
    parts = p.parts
    src_idx = parts.index("src")
    return ".".join([*parts[src_idx + 1 : -1], p.stem])


_ROUTE_RE = re.compile(r"@(?:app|v1|ops|router|api)\.(?:get|post|put|patch|delete)\(")


def _test_kind(source: str, source_code: str) -> str:
    """Which test layer this changed file calls for.

    A QA engineer wouldn't write a unit test for an HTTP route or a browser
    test for a pure function — the layer follows from what the file is.
    """
    if "/web/" in source and source.endswith((".html", ".js")):
        return "ui"
    if _ROUTE_RE.search(source_code):
        return "api"
    return "unit"


# Layer-specific rules appended to the code prompt. Each names the exact harness
# this repo already uses, so the model extends the house pattern instead of
# inventing one.
_KIND_RULES = {
    "unit": (
        "LAYER: unit test. Test the module's functions/classes directly. No HTTP, "
        "no browser."
    ),
    "api": (
        "LAYER: API/integration test. This module defines FastAPI routes, so test "
        "them over HTTP in-process — do NOT call the handler functions directly.\n"
        "- Use the harness this repo already uses:\n"
        "    from httpx import ASGITransport, AsyncClient\n"
        "    client = AsyncClient(transport=ASGITransport(app=app), "
        'base_url="http://test")\n'
        "  Then `resp = await client.post('/v1/...', json={...})`.\n"
        "- Assert on resp.status_code and resp.json() — the real HTTP contract.\n"
        "- Cover the failure paths too (invalid body -> 422, missing resource -> "
        "404), not just the happy path.\n"
        "- Build the app via the same fixture/helper the sibling tests use; do not "
        "hit real AWS/Qdrant/Mongo — stub the components."
    ),
    "ui": (
        "LAYER: UI/end-to-end test with Playwright (sync API).\n"
        "- `from playwright.sync_api import Page, expect`, and take `page: Page` as "
        "a fixture argument.\n"
        "- Mark the module `pytestmark = pytest.mark.integration` — it needs a live "
        "server.\n"
        "- Prefer role/text selectors (`page.get_by_role`, `get_by_text`) over "
        "brittle CSS/XPath.\n"
        "- Use `expect(...)` auto-waiting assertions; never `time.sleep`.\n"
        "- Only assert on elements you can actually see in the provided HTML."
    ),
}


def _module_symbols(source_code: str) -> list[str]:
    """Top-level names a test could import from the module under test."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names.extend(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
    return names


def _unimported_symbols(code: str, module: str, symbols: list[str]) -> list[str]:
    """Symbols the test calls but never imported — the #1 generated-test bug.

    A small model reliably writes ``test_is_pdf`` while leaving ``is_pdf`` out of
    the import line, producing a NameError at runtime. Catching it statically is
    both cheaper than a pytest round-trip and gives far more precise feedback.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            imported.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Import):
            imported.update((a.asname or a.name).split(".")[0] for a in node.names)

    # Names bound inside the test file itself are not missing imports.
    local: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            local.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            local.add(node.id)
        elif isinstance(node, ast.arg):
            local.add(node.arg)

    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return sorted(used & set(symbols) - imported - local)


def _style_reference(test_abs: Path) -> str:
    """An existing sibling test, to anchor the model on house conventions."""
    tests_dir = test_abs.parent
    if not tests_dir.is_dir():
        return "(no sibling tests available)"
    for cand in sorted(tests_dir.glob("test_*.py")):
        if cand != test_abs:
            return cand.read_text(encoding="utf-8", errors="replace")[:1800]
    return "(no sibling tests available)"


# --- Prompting --------------------------------------------------------------

_SYSTEM = (
    "You are a senior Python test engineer. You write correct, runnable pytest "
    "tests for a Python 3.12 monorepo using pytest, pytest-asyncio "
    "(asyncio_mode=auto), pytest-mock, and moto. You never invent APIs: you only "
    "test behaviour visible in the provided source. You reply with JSON only."
)

# A separate persona for the code call: the JSON-only instruction above would
# actively fight the "emit raw Python" requirement.
_SYSTEM_CODE = (
    "You are a senior Python test engineer writing pytest tests for a Python "
    "3.12 monorepo (pytest, pytest-asyncio with asyncio_mode=auto, pytest-mock, "
    "moto). You never invent APIs: you only test behaviour visible in the "
    "provided source. You output raw Python source code only — never markdown, "
    "never prose, never code fences."
)

# Shared context block, reused by both calls so the second one needs no recap.
_CONTEXT_TEMPLATE = """SOURCE FILE:   {source}
IMPORT IT AS:  from {module} import <name>

SYMBOLS DEFINED IN THIS MODULE — these are the ONLY names you may import from
`{module}`, and every one you use MUST appear in your import line:
  {symbols}

--- CURRENT SOURCE (may be truncated) ---
{source_code}

--- DIFF OF THIS PUSH ---
{diff}

--- EXISTING TEST FILE ({test_path}) ---
{existing_test}

--- STYLE REFERENCE: a real test file from this same package ---
{style_reference}
"""

_ANALYSIS_PROMPT = """{context}
A source file changed in this push. Decide what should happen to its pytest \
test file. Fill in these fields:
- summary: one or two sentences on what changed in the source.
- action: "created" if there is no existing test file, "updated" if one exists \
and should be revised, "skipped" ONLY if the change is trivial (comments, \
formatting, config) or untestable.
- new_test_cases: the names of the test functions that should exist.
- notes: anything the reviewer should know (edge cases not covered, mocks needed).
"""

_CODE_PROMPT = """{context}
Write the COMPLETE pytest test file for the source above.

It must define these test functions: {cases}

Output ONLY raw Python code. No markdown fences, no ``` , no explanation, no \
prose before or after. Your entire reply must be the contents of the .py file.

{kind_rules}

IMPORT RULES (critical — getting these wrong makes the test unrunnable):
- Import ONLY as `from {module} import <name>`. This project uses a src layout; \
the package is installed under that name.
- NEVER prefix imports with `packages.`, `services.`, or `src.`. \
`from packages.core.src.rag_core...` is WRONG and will not import.
- NEVER import from conftest; pytest fixtures are injected automatically as \
test-function arguments.

CODE RULES:
- Any function declared `async def` in the source MUST be awaited: \
`result = await thing(...)`, and its test must itself be `async def`. This \
project sets asyncio_mode=auto, so no @pytest.mark.asyncio decorator is needed.
- Do NOT invent expected hash/UUID/constant literals you have not computed. \
Assert on properties instead: length, type, determinism (same input twice gives \
the same output), or that different inputs differ.
- Only use imports and symbols that exist in the source; do not fabricate APIs.
- IMPORT EVERY NAME YOU USE, including typing helpers \
(`from collections.abc import AsyncIterator`) and mocks \
(`from unittest.mock import AsyncMock`). An un-imported name is a NameError.
- NEVER hardcode filesystem paths like "/path/to/file.txt" — they do not exist \
and the test will fail. Use pytest's `tmp_path` fixture: \
`def test_x(tmp_path): p = tmp_path / "f.txt"; p.write_bytes(b"data")`.
- To build an async iterator for a test, write a tiny async generator rather \
than mocking one: `async def _gen():\\n    yield b"a"\\n    yield b"b"`.
- Preserve existing tests that still apply.
- Keep tests deterministic and fast; mark anything needing live services with \
@pytest.mark.integration.
"""


_APPEND_PROMPT = """{context}
The test file above ALREADY EXISTS and its tests already pass. Do NOT rewrite it.

Write ONLY the NEW test code to append to the end of that file, covering what \
the diff added: {cases}

Output ONLY raw Python. No markdown fences, no ``` , no explanation. Your reply \
is appended verbatim to the existing file, so:
- Do NOT repeat any test function that already exists in the file above.
- Do NOT repeat imports that the file above already has.
- DO include an import line for any name you use that the file above does not \
already import (e.g. `from {module} import <the new function>`). An import \
placed here is fine — it runs at module level.

{kind_rules}

IMPORT RULES (critical — getting these wrong makes the test unrunnable):
- Import ONLY as `from {module} import <name>`. This project uses a src layout.
- NEVER prefix imports with `packages.`, `services.`, or `src.`.
- NEVER import from conftest; fixtures are injected as test arguments.

CODE RULES:
- Any `async def` function in the source MUST be awaited, and its test must be \
`async def`. asyncio_mode=auto, so no decorator is needed.
- Do NOT invent expected hash/UUID/constant literals. Assert on properties: \
type, length, determinism, or that different inputs differ.
- NEVER hardcode filesystem paths; use the `tmp_path` fixture.
- IMPORT EVERY NAME YOU USE, including typing helpers and mocks.
"""


def _strip_code_fences(text: str) -> str:
    """Small models add ```python fences despite being told not to."""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip() + "\n"
    return text.strip() + "\n"


def _extract_json(text: str) -> dict:
    """Parse the model's reply, tolerating code fences and stray prose.

    Ollama's structured output returns clean JSON, but Bedrock (and a model
    ignoring the schema) may wrap it — so stay permissive.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    raw = fenced.group(1) if fenced else text
    start = raw.find("{")
    if start == -1:
        raise ValueError("no JSON object in model response")
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(raw[start : i + 1])
    raise ValueError("unbalanced JSON in model response")


_BAD_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+(packages|services|src)\.", re.MULTILINE)
_CONFTEST_IMPORT_RE = re.compile(r"^\s*from\s+[\w.]*conftest\s+import", re.MULTILINE)


def validate_test_code(
    content: str, module: str = "", symbols: list[str] | None = None
) -> str | None:
    """Return a human-readable reason the generated test is unusable, else None.

    A local 8B model produces plausible-looking but unrunnable tests often
    enough that shipping its output unchecked would fill the PR with red. These
    are the failure modes observed in practice, all cheaply detectable.
    """
    try:
        ast.parse(content)
    except SyntaxError as exc:
        return f"generated test is not valid Python: {exc}"
    if _BAD_IMPORT_RE.search(content):
        return (
            "generated test imports via a filesystem path "
            "(packages./services./src.) instead of the installed package name"
        )
    if _CONFTEST_IMPORT_RE.search(content):
        return "generated test imports from conftest (fixtures are injected, not imported)"
    if "def test_" not in content:
        return "generated test contains no test functions"
    if module and symbols:
        missing = _unimported_symbols(content, module, symbols)
        if missing:
            return (
                f"these symbols are used but never imported (NameError at runtime): "
                f"{', '.join(missing)}. Add them to the "
                f"`from {module} import ...` line."
            )
    return None


def _run_single_test(test_abs: Path) -> tuple[bool, str]:
    """Execute one generated test file. Returns (passed, failure output)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_abs), "-q", "--no-header", "-x"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=PYTEST_TIMEOUT_S,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    # Only the tail is useful as repair feedback, and it keeps the prompt small.
    return proc.returncode == 0, "\n".join(output.strip().splitlines()[-30:])


def analyse_file(provider: Provider, source: str, base: str, head: str, usage: Usage) -> Analysis:
    src_abs = REPO_ROOT / source
    source_code = src_abs.read_text(encoding="utf-8", errors="replace")[:MAX_SOURCE_CHARS]
    diff = _diff_for(base, head, source)[:MAX_SOURCE_CHARS]
    test_abs = _test_path_for(source)
    rel_test = test_abs.relative_to(REPO_ROOT).as_posix()
    existing = (
        test_abs.read_text(encoding="utf-8", errors="replace")[:MAX_TEST_CHARS]
        if test_abs.exists()
        else "(no existing test file)"
    )

    full_source = src_abs.read_text(encoding="utf-8", errors="replace")
    kind = _test_kind(source, full_source)
    module = _module_path_for(source)
    symbols = _module_symbols(full_source)
    context = _CONTEXT_TEMPLATE.format(
        source=source,
        module=module,
        symbols=", ".join(symbols) or "(none detected)",
        source_code=source_code,
        diff=diff or "(diff unavailable)",
        test_path=rel_test,
        existing_test=existing,
        style_reference=_style_reference(test_abs),
    )

    # --- Call 1: the decision (schema-constrained JSON) ---
    reply, in_tok, out_tok = provider.complete(
        _SYSTEM, _ANALYSIS_PROMPT.format(context=context), schema=ANALYSIS_SCHEMA
    )
    usage.add(in_tok, out_tok)
    try:
        data = _extract_json(reply)
    except (ValueError, json.JSONDecodeError) as exc:
        return Analysis(
            source, f"(model returned unparseable output: {exc})", rel_test,
            "skipped", kind=kind,
        )

    summary = str(data.get("summary", "")).strip()
    notes = str(data.get("notes", "")).strip()
    cases = [str(x) for x in data.get("new_test_cases", []) if x]
    action = str(data.get("action", "skipped")).lower()

    if action not in {"created", "updated"}:
        return Analysis(source, summary, rel_test, "skipped", cases, notes, kind)

    # --- Call 2: the code, with a generate -> run -> repair loop ---
    #
    # Static checks cannot catch a NameError or a hardcoded path; only running
    # the test can. So each candidate is executed against the real source, and
    # its own failure output is fed back for repair. A test that never passes is
    # never committed — the original file is restored — because a red PR full of
    # broken generated tests is worse than no tests at all.
    original: str | None = (
        test_abs.read_text(encoding="utf-8") if test_abs.exists() else None
    )
    # APPEND mode when a test file already exists. Asking an 8B model to
    # reproduce a whole existing module perfectly just to add one test fails
    # almost every time (measured); asking only for the new tests, and merging
    # them ourselves, keeps the existing passing tests safe by construction.
    append_mode = original is not None and original.strip() != ""
    prompt_tpl = _APPEND_PROMPT if append_mode else _CODE_PROMPT
    code_prompt = prompt_tpl.format(
        context=context,
        module=module,
        cases=", ".join(cases) or "(you decide)",
        kind_rules=_KIND_RULES[kind],
    )
    feedback: str | None = None
    last_reason = "model did not produce a passing test"

    for attempt in range(1, MAX_CODE_ATTEMPTS + 1):
        attempt_prompt = code_prompt
        if feedback:
            attempt_prompt = (
                f"{code_prompt}\n\nYour previous attempt FAILED:\n{feedback}\n\n"
                "How to fix it:\n"
                "- The SOURCE CODE IS CORRECT BY DEFINITION. You are testing it, "
                "not reviewing it. An AssertionError means YOUR EXPECTED VALUE was "
                "wrong, never that the source has a bug.\n"
                "- Re-read the source above and correct that expectation, or "
                "simply DELETE the failing assertion. Deleting a wrong assertion "
                "is always acceptable.\n"
                "- Do not assert behaviour you cannot point to in the source. Do "
                "not copy assumptions from the sibling test file — it tests a "
                "DIFFERENT function with different rules.\n"
                "- Keep the assertions that passed; change only what failed.\n"
                "Output only the corrected Python."
            )
        reply, in_tok, out_tok = provider.complete(_SYSTEM_CODE, attempt_prompt)
        usage.add(in_tok, out_tok)
        snippet = _strip_code_fences(reply)
        content = f"{original.rstrip()}\n\n\n{snippet}" if append_mode else snippet

        static_reason = validate_test_code(content, module, symbols)
        if static_reason is not None:
            last_reason, feedback = static_reason, static_reason
            print(f"  [attempt {attempt}] rejected: {static_reason}", flush=True)
            continue

        test_abs.parent.mkdir(parents=True, exist_ok=True)
        test_abs.write_text(content, encoding="utf-8")
        ok, output = _run_single_test(test_abs)
        if ok:
            print(f"  [attempt {attempt}] tests pass", flush=True)
            return Analysis(source, summary, rel_test, action, cases, notes, kind)

        last_reason = "generated tests did not pass"
        feedback = output
        print(f"  [attempt {attempt}] generated tests failed; repairing", flush=True)

    # Never leave a failing generated test behind.
    if original is None:
        test_abs.unlink(missing_ok=True)
    else:
        test_abs.write_text(original, encoding="utf-8")
    return Analysis(
        source, summary, rel_test, "skipped", cases,
        f"NOT WRITTEN — {last_reason} after {MAX_CODE_ATTEMPTS} attempts. {notes}".strip(),
        kind,
    )


# --- Test run ---------------------------------------------------------------

def _counts(summary: str) -> dict[str, int]:
    """Pull passed/failed/error counts out of pytest's summary line."""
    out = {"passed": 0, "failed": 0, "errors": 0}
    for key, pat in (
        ("passed", r"(\d+) passed"),
        ("failed", r"(\d+) failed"),
        ("errors", r"(\d+) error"),
    ):
        m = re.search(pat, summary)
        if m:
            out[key] = int(m.group(1))
    return out


def run_pytest() -> dict:
    """Run the fast test subset; capture the result without failing the job."""
    cmd = [sys.executable, "-m", "pytest", "-m", "not integration and not slow", "-q", "--no-header"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")
    summary_line = ""
    for line in reversed(output.strip().splitlines()):
        if re.search(r"(passed|failed|error|no tests ran)", line):
            summary_line = line.strip()
            break
    return {
        "ran": True,
        "exit_code": proc.returncode,
        "passed": proc.returncode == 0,
        "summary": summary_line or "(no pytest summary captured)",
        "counts": _counts(summary_line),
        "output_tail": "\n".join(output.strip().splitlines()[-40:]),
    }


# --- Entry point ------------------------------------------------------------

def main() -> int:
    before = os.environ.get("EVENT_BEFORE", "")
    head = os.environ.get("HEAD_SHA") or _run(["git", "rev-parse", "HEAD"]).strip()
    base = _resolve_base(before, head)

    sources, all_changes = changed_source_files(base, head)
    to_analyse = sources[:MAX_FILES]
    skipped_over_cap = sources[MAX_FILES:]

    # Baseline BEFORE touching anything. Without it, a suite that already has
    # failures reports "FAILED" on every run and the report becomes noise —
    # what matters is whether *this run* introduced a regression.
    baseline = run_pytest()
    print(f"[baseline] {baseline['summary']}", flush=True)

    usage = Usage()
    analyses: list[Analysis] = []
    provider: Provider | None = None

    if to_analyse:
        provider = build_provider()
        print(f"[provider] {provider.name} model={provider.model}", flush=True)
        for src in to_analyse:
            print(f"[analyse] {src}", flush=True)
            try:
                analyses.append(analyse_file(provider, src, base, head, usage))
            except (urllib.error.URLError, TimeoutError) as exc:
                print(f"[warn] model unreachable for {src}: {exc}", flush=True)
                analyses.append(Analysis(src, f"(model unreachable: {exc})", "", "skipped"))
            except Exception as exc:  # one bad file must not abort the whole run
                print(f"[warn] failed on {src}: {exc}", flush=True)
                analyses.append(Analysis(src, f"(analysis failed: {exc})", "", "skipped"))

    tests = run_pytest()

    # A regression is a failure count that grew relative to the baseline. The
    # generated tests are individually verified, so this should stay at zero;
    # if it doesn't, that is the signal worth e-mailing about.
    new_failures = max(
        0, tests["counts"]["failed"] - baseline["counts"]["failed"]
    ) + max(0, tests["counts"]["errors"] - baseline["counts"]["errors"])
    tests["pre_existing_failures"] = baseline["counts"]["failed"]
    tests["new_failures"] = new_failures
    tests["tests_added"] = max(0, tests["counts"]["passed"] - baseline["counts"]["passed"])
    tests["no_regression"] = new_failures == 0

    wrote_any = any(a.action in {"created", "updated"} for a in analyses)
    cost = provider.cost_usd(usage) if provider else 0.0
    report = {
        "commit": {
            "base": base,
            "head": head,
            "ref": os.environ.get("GITHUB_REF_NAME", ""),
            "repo": os.environ.get("GITHUB_REPOSITORY", ""),
            "message": os.environ.get("HEAD_COMMIT_MESSAGE", ""),
            "author": os.environ.get("HEAD_COMMIT_AUTHOR", ""),
        },
        "provider": provider.name if provider else PROVIDER,
        "model": provider.model if provider else "(no model call needed)",
        "changed_files": [{"status": s, "path": p} for s, p in all_changes],
        "analyzed": [a.__dict__ for a in analyses],
        "skipped_over_cap": skipped_over_cap,
        "wrote_tests": wrote_any,
        "tokens": {"input": usage.input_tokens, "output": usage.output_tokens, "calls": usage.calls},
        "cost_usd": round(cost, 6),
        "tests": tests,
        "baseline_tests": {"summary": baseline["summary"], "counts": baseline["counts"]},
    }
    (REPO_ROOT / "ci_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as fh:
            fh.write(f"wrote_tests={'true' if wrote_any else 'false'}\n")

    print(
        f"[done] analysed={len(analyses)} wrote_tests={wrote_any} "
        f"tokens_in={usage.input_tokens} tokens_out={usage.output_tokens} "
        f"cost=${cost:.4f} pytest={tests['summary']}",
        flush=True,
    )
    return 0  # never fail the job — the report/e-mail carry the outcome


if __name__ == "__main__":
    raise SystemExit(main())
