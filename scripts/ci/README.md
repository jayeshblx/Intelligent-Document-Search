# Auto Test Update — CI workflow

On **every push to any branch**, [`.github/workflows/auto-test-update.yml`](../../.github/workflows/auto-test-update.yml):

1. Diffs the push to find changed source files (`packages/**`, `services/**`).
2. Baselines the test suite (so pre-existing failures aren't blamed on this run).
3. Asks a **local Llama model via Ollama** to summarise each change and write/update
   the matching pytest test file.
4. **Verifies each generated test by running it**, repairing up to 3 times.
   A test that never passes is **not committed** — the original file is restored.
5. Opens a **pull request** with the tests that pass.
6. E-mails a report to `reports@example.com` — what changed, which tests
   were written, regression status, token usage, and cost.

Inference is local, so **there is no API cost**.

## Prerequisite: a self-hosted runner

GitHub's cloud runners cannot reach Ollama on your machine, so this workflow
**must** run on a self-hosted runner on the machine that has Ollama.

Register one at:
**Settings → Actions → Runners → New self-hosted runner** (or
`https://github.com/jayeshblx/Intelligent-Document-Search/settings/actions/runners`)

The workflow targets `runs-on: [self-hosted, windows]`, so give the runner the
`windows` label (the default Windows runner config adds it automatically).

The runner must have: `git`, `uv`, Python 3.12, and `ollama` with the model pulled.

> Self-hosted runners are safe here because this repo is **private**. Never enable
> a self-hosted runner on a public repo — fork PRs could execute code on your machine.

## Configuration

These are **variables**, not secrets (Settings → Secrets and variables → Actions → *Variables*).
Both are optional; the defaults match a standard local Ollama install.

| Variable | Default | Notes |
|----------|---------|-------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint. |
| `OLLAMA_MODEL` | `llama3.1:8b` | Pulled automatically if missing. |

### E-mail secrets (SMTP — provider-agnostic)

| Secret | Required | Notes |
|--------|----------|-------|
| `SMTP_HOST` | ✅ | e.g. `smtp.office365.com`, `smtp.gmail.com`, `smtp.sendgrid.net`. |
| `SMTP_PORT` | optional | Defaults to `587` (STARTTLS). |
| `SMTP_USERNAME` | ✅ | For SendGrid this is the literal `apikey`. |
| `SMTP_PASSWORD` | ✅ | Mailbox app password / API key. |
| `EMAIL_FROM` | ✅ | Sender address. |
| `EMAIL_TO` | optional | Defaults to `reports@example.com`. |

Without these the workflow still runs; it just logs the report instead of sending it.

## Expectations: what an 8B model can and cannot do

`llama3.1:8b` is free but limited. Measured on this repo:

- **Good** at fresh tests for small, self-contained modules (pure functions).
- **Weak** at *updating* an existing test file while preserving its tests, and at
  anything involving async iterators or elaborate mocking.
- Roughly **half** of attempted files end up with a committed test; the rest are
  reported as `skipped` with the reason.

This is safe by design — the verify-and-repair loop means the failure mode is
*"no test written"*, never *"a broken test committed"*. Every test that lands in
the PR was executed and passed first.

To raise the hit rate, either:
- pull a larger local model and set the `OLLAMA_MODEL` variable
  (e.g. `ollama pull qwen2.5-coder:14b`), or
- set `LLM_PROVIDER=bedrock` in the workflow to use Claude Haiku 4.5
  (~$0.005/push, materially better tests). The Bedrock path is still implemented
  in `generate_tests.py`; it needs `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
  and `BEDROCK_REGION` secrets.

## Tuning

Environment knobs read by `generate_tests.py`:

| Env | Default | Purpose |
|-----|---------|---------|
| `LLM_PROVIDER` | `ollama` | `ollama` or `bedrock`. |
| `MAX_FILES` | `12` | Max source files analysed per push. |
| `MAX_CODE_ATTEMPTS` | `3` | Generate → run → repair rounds per file. |
| `OLLAMA_NUM_CTX` | `16384` | Context window. |
| `OLLAMA_NUM_PREDICT` | `4096` | Max output tokens. |
| `PYTEST_TIMEOUT_S` | `180` | Per-file test-run timeout. |

Skip a push entirely with `[skip-auto-test]` in the commit message.

## Note on the current suite

The suite currently has **33 pre-existing failures** (in `services/search/tests/`)
unrelated to this workflow. The report accounts for this: it baselines before
generating, so the e-mail headline reports whether *this run* introduced a
regression, not the absolute suite state.
