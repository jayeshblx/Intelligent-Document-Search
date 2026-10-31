"""E-mail the CI test-automation report.

Reads ``ci_report.json`` (produced by generate_tests.py), renders an HTML
summary — what changed, which tests were written/updated, the pytest result,
and the token usage + estimated cost — and sends it via SMTP.

The transport is deliberately provider-agnostic (plain SMTP + STARTTLS): point
the ``SMTP_*`` secrets at Office 365, Gmail, SendGrid, Resend, or anything else
without touching this code.
"""

from __future__ import annotations

import html
import json
import os
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "ci_report.json"

DEFAULT_TO = "reports@example.com"


def _env(name: str, default: str = "") -> str:
    """Read an env var, treating empty as absent.

    GitHub Actions injects unset secrets as EMPTY STRINGS rather than omitting
    them, so `os.environ.get(name, default)` returns "" and never the default —
    which made int(SMTP_PORT) raise ValueError on every run.
    """
    val = os.environ.get(name)
    return val.strip() if val and val.strip() else default


def _esc(x: object) -> str:
    return html.escape(str(x))


def _cost_label(report: dict) -> str:
    """Local inference has no per-token price; say so rather than print $0.0000."""
    cost = report.get("cost_usd", 0) or 0
    if report.get("provider") == "ollama":
        return "$0.00 - local inference (Ollama), no API billing"
    return f"${cost:.4f} USD"


def _rows(items: list[dict]) -> str:
    if not items:
        return "<tr><td colspan='4'><em>No source files required test changes.</em></td></tr>"
    out = []
    for a in items:
        cases = ", ".join(a.get("new_test_cases", [])) or "&mdash;"
        badge = {
            "created": "#1a7f37",
            "updated": "#9a6700",
            "skipped": "#57606a",
        }.get(a.get("action", "skipped"), "#57606a")
        out.append(
            "<tr>"
            f"<td><code>{_esc(a.get('source'))}</code></td>"
            f"<td><span style='color:{badge};font-weight:600'>{_esc(a.get('action'))}</span>"
            f"<br><code>{_esc(a.get('test_path'))}</code></td>"
            f"<td>{_esc(a.get('summary'))}"
            + (f"<br><small style='color:#57606a'>{_esc(a.get('notes'))}</small>" if a.get("notes") else "")
            + "</td>"
            f"<td>{_esc(cases)}</td>"
            "</tr>"
        )
    return "\n".join(out)


def render_html(report: dict, pr_url: str) -> str:
    c = report.get("commit", {})
    t = report.get("tests", {})
    tok = report.get("tokens", {})
    analyzed = report.get("analyzed", [])
    # The headline is "did THIS run break anything", not "is the whole suite
    # green" — the suite may carry pre-existing failures that predate the bot.
    no_regression = t.get("no_regression", t.get("passed"))
    status_color = "#1a7f37" if no_regression else "#cf222e"
    status_text = "NO REGRESSION" if no_regression else "NEW FAILURES INTRODUCED"

    changed = report.get("changed_files", [])
    changed_list = "".join(
        f"<li><code>{_esc(f.get('status'))}</code> {_esc(f.get('path'))}</li>" for f in changed
    ) or "<li><em>none</em></li>"

    pr_block = (
        f"<p><strong>Pull request:</strong> <a href='{_esc(pr_url)}'>{_esc(pr_url)}</a></p>"
        if pr_url
        else "<p><strong>Pull request:</strong> none opened (no test changes).</p>"
    )

    over_cap = report.get("skipped_over_cap", [])
    cap_note = (
        f"<p style='color:#9a6700'>Note: {len(over_cap)} additional changed source "
        "file(s) were not analysed this run (per-run cap).</p>"
        if over_cap
        else ""
    )

    return f"""\
<html><body style="font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;color:#1f2328;max-width:840px">
  <h2 style="margin-bottom:4px">Automated Test Update</h2>
  <p style="color:#57606a;margin-top:0">
    {_esc(c.get('repo'))} &middot; branch <code>{_esc(c.get('ref'))}</code>
    &middot; commit <code>{_esc((c.get('head') or '')[:8])}</code>
  </p>
  <p><strong>Commit:</strong> {_esc(c.get('message'))}<br>
     <small style="color:#57606a">by {_esc(c.get('author'))}</small></p>

  <table style="border-collapse:collapse;margin:12px 0">
    <tr>
      <td style="padding:6px 14px;background:#f6f8fa;border:1px solid #d0d7de">This run</td>
      <td style="padding:6px 14px;border:1px solid #d0d7de">
        <span style="color:{status_color};font-weight:700">{status_text}</span>
        &nbsp;&mdash; {_esc(t.get('new_failures', 0))} new failure(s),
        {_esc(t.get('tests_added', 0))} test(s) added
      </td>
    </tr>
    <tr>
      <td style="padding:6px 14px;background:#f6f8fa;border:1px solid #d0d7de">Suite now</td>
      <td style="padding:6px 14px;border:1px solid #d0d7de"><code>{_esc(t.get('summary'))}</code></td>
    </tr>
    <tr>
      <td style="padding:6px 14px;background:#f6f8fa;border:1px solid #d0d7de">Suite before</td>
      <td style="padding:6px 14px;border:1px solid #d0d7de">
        <code>{_esc(report.get('baseline_tests', {}).get('summary'))}</code>
        <br><small style="color:#57606a">{_esc(t.get('pre_existing_failures', 0))} failure(s)
        already existed before this run and are not caused by it.</small>
      </td>
    </tr>
    <tr>
      <td style="padding:6px 14px;background:#f6f8fa;border:1px solid #d0d7de">Model</td>
      <td style="padding:6px 14px;border:1px solid #d0d7de">
        <code>{_esc(report.get('model'))}</code>
        &nbsp;<small style="color:#57606a">via {_esc(report.get('provider'))}</small>
      </td>
    </tr>
    <tr>
      <td style="padding:6px 14px;background:#f6f8fa;border:1px solid #d0d7de">Token usage</td>
      <td style="padding:6px 14px;border:1px solid #d0d7de">
        {_esc(tok.get('input', 0))} in / {_esc(tok.get('output', 0))} out
        &nbsp;({_esc(tok.get('calls', 0))} call(s))
      </td>
    </tr>
    <tr>
      <td style="padding:6px 14px;background:#f6f8fa;border:1px solid #d0d7de">Cost</td>
      <td style="padding:6px 14px;border:1px solid #d0d7de">{_esc(_cost_label(report))}</td>
    </tr>
  </table>

  <p style="color:#57606a;font-size:13px;margin-top:0">
    Every test listed as <em>created</em> or <em>updated</em> below was executed and
    passed before being committed. Anything the model could not get passing is
    listed as <em>skipped</em> with the reason, and was not written.
  </p>

  {pr_block}
  {cap_note}

  <h3>What was updated</h3>
  <table style="border-collapse:collapse;width:100%;font-size:14px">
    <tr style="background:#f6f8fa">
      <th style="text-align:left;padding:6px 10px;border:1px solid #d0d7de">Source</th>
      <th style="text-align:left;padding:6px 10px;border:1px solid #d0d7de">Action / Test file</th>
      <th style="text-align:left;padding:6px 10px;border:1px solid #d0d7de">What changed</th>
      <th style="text-align:left;padding:6px 10px;border:1px solid #d0d7de">New test cases</th>
    </tr>
    {_rows(analyzed)}
  </table>

  <h3>All changed files in this push</h3>
  <ul>{changed_list}</ul>

  <h3>pytest output (tail)</h3>
  <pre style="background:#f6f8fa;border:1px solid #d0d7de;padding:10px;overflow:auto;font-size:12px">{_esc(t.get('output_tail'))}</pre>

  <p style="color:#57606a;font-size:12px">Sent automatically by the auto-test-update GitHub Actions workflow.</p>
</body></html>"""


def render_text(report: dict, pr_url: str) -> str:
    c = report.get("commit", {})
    t = report.get("tests", {})
    tok = report.get("tokens", {})
    lines = [
        "Automated Test Update",
        f"Repo:   {c.get('repo')}  branch {c.get('ref')}  commit {(c.get('head') or '')[:8]}",
        f"Commit: {c.get('message')} (by {c.get('author')})",
        "",
        f"This run:    {'NO REGRESSION' if t.get('no_regression', t.get('passed')) else 'NEW FAILURES INTRODUCED'}"
        f" - {t.get('new_failures', 0)} new failure(s), {t.get('tests_added', 0)} test(s) added",
        f"Suite now:   {t.get('summary')}",
        f"Suite before: {report.get('baseline_tests', {}).get('summary')}",
        f"  ({t.get('pre_existing_failures', 0)} failure(s) pre-date this run)",
        f"Model:       {report.get('model')} (via {report.get('provider')})",
        f"Tokens:      {tok.get('input',0)} in / {tok.get('output',0)} out ({tok.get('calls',0)} calls)",
        f"Cost:        {_cost_label(report)}",
        f"Pull request: {pr_url or 'none opened'}",
        "",
        "Updated tests:",
    ]
    for a in report.get("analyzed", []):
        lines.append(f"  - [{a.get('action')}] {a.get('source')} -> {a.get('test_path')}")
        if a.get("summary"):
            lines.append(f"      {a.get('summary')}")
        if a.get("new_test_cases"):
            lines.append(f"      new: {', '.join(a['new_test_cases'])}")
    return "\n".join(lines)


def main() -> int:
    if not REPORT_PATH.exists():
        print("[send_report] ci_report.json missing; nothing to send", file=sys.stderr)
        return 0

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    pr_url = _env("PR_URL")

    host = _env("SMTP_HOST")
    port = int(_env("SMTP_PORT", "587"))
    user = _env("SMTP_USERNAME")
    password = _env("SMTP_PASSWORD")
    mail_from = _env("EMAIL_FROM", user)
    mail_to = _env("EMAIL_TO", DEFAULT_TO)

    if not (host and user and password and mail_from):
        print(
            "[send_report] SMTP secrets not fully configured "
            "(need SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, EMAIL_FROM). Skipping send.",
            file=sys.stderr,
        )
        print("---- report that would have been sent ----")
        print(render_text(report, pr_url))
        return 0

    t = report.get("tests", {})
    branch = report.get("commit", {}).get("ref", "")
    ok = t.get("no_regression", t.get("passed"))
    added = t.get("tests_added", 0)
    subject = (
        f"[Auto-Tests] {'OK' if ok else 'REGRESSION'} on {branch} "
        f"- {added} test(s) added, {t.get('new_failures', 0)} new failure(s)"
    )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = mail_from
    msg["To"] = mail_to
    msg.set_content(render_text(report, pr_url))
    msg.add_alternative(render_html(report, pr_url), subtype="html")

    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(user, password)
        smtp.send_message(msg)

    print(f"[send_report] sent to {mail_to} via {host}:{port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
