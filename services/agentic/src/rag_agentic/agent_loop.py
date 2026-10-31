"""The agent loop: model -> tool calls -> observations -> model, until it answers.

Deliberately small and explicit. It owns three things the agents should not each
reinvent: appending the assistant message *verbatim* (Converse rejects a rebuilt
block list -- toolUse ids must survive), returning every parallel tool result in
a single user message (splitting them teaches the model to stop calling tools in
parallel), and stopping at a hard iteration ceiling so a confused model cannot
bill an unbounded number of round trips.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from rag_agentic.converse import ConverseModel, ToolSpec
from rag_agentic.guardrails import redact
from rag_agentic.models import ToolCall, Usage
from rag_core.logging import get_logger

_log = get_logger("rag_agentic.loop")

# Tool payloads are fed back into the prompt, so an unbounded result would blow
# the context (and the bill) on one bad call. Truncate defensively.
_MAX_RESULT_CHARS = 24_000

# Sent when the tool-call ceiling is reached, in place of withdrawing the tools.
_ANSWER_NOW = (
    "You have reached the tool-call limit for this turn. Answer the question now, "
    "using only what you have already retrieved, and do not call any more tools. "
    "If what you gathered is not enough, say so and say what you searched for."
)


class ToolBox(Protocol):
    """A set of tools an agent can call."""

    def specs(self) -> list[ToolSpec]:
        """Tool definitions offered to the model."""
        ...

    async def invoke(self, name: str, arguments: dict[str, Any]) -> tuple[Any, str]:
        """Run one tool. Returns ``(json_payload, short_summary_for_the_trace)``."""
        ...


class ToolError(RuntimeError):
    """A tool failed in a way the model should see and can recover from."""


@dataclass(slots=True)
class LoopResult:
    text: str = ""
    trace: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    messages: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    # True when the ceiling cut the loop short -- the answer may be partial.
    truncated: bool = False


async def run_agent(
    model: ConverseModel,
    *,
    system: str,
    user_text: str,
    toolbox: ToolBox | None = None,
    max_iterations: int = 6,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    cache_system: bool = True,
    history: list[dict[str, Any]] | None = None,
    stop_after_tools: Callable[[LoopResult], bool] | None = None,
    propagate: tuple[type[BaseException], ...] = (),
) -> LoopResult:
    """Drive one agent turn to completion.

    ``stop_after_tools`` is checked once per round, after the tool results are in.
    Returning True ends the turn *without* another model call -- used when a tool
    has already produced the final answer, so paying for the model to restate it
    would add cost and latency and risk paraphrasing it.

    ``propagate`` lists exception types that end the turn instead of being shown
    to the model. A tool error is normally an observation the model can recover
    from; a platform failure (the tenant's index cannot be opened) is not, and
    feeding it to the model only produces a polite apology that the caller cannot
    tell apart from a real answer.
    """
    messages: list[dict[str, Any]] = list(history or [])
    messages.append({"role": "user", "content": [{"text": user_text}]})

    specs = toolbox.specs() if toolbox else []
    result = LoopResult(messages=messages)

    for iteration in range(1, max_iterations + 1):
        result.iterations = iteration
        turn = await model.converse(
            messages,
            system=system,
            tools=specs or None,
            max_tokens=max_tokens,
            temperature=temperature,
            cache_system=cache_system,
            # Tool results accumulate fast (a search adds 15 passages), and every
            # later round re-reads them. Cache the transcript as well as the prompt.
            cache_messages=cache_system,
        )
        result.usage = result.usage.add(turn.usage)
        _log_call_usage(iteration, turn.usage, result.usage)
        # Verbatim: Converse validates toolUse ids against what it sent us.
        messages.append(turn.message or {"role": "assistant", "content": [{"text": turn.text}]})

        if turn.text:
            result.text = turn.text
        if not turn.wants_tools or toolbox is None:
            return result

        blocks: list[dict[str, Any]] = []
        for use in turn.tool_uses:
            call, block = await _run_tool(toolbox, use.id, use.name, use.input, propagate)
            result.trace.append(call)
            blocks.append(block)
        # All results in ONE user message -- see module docstring.
        messages.append({"role": "user", "content": blocks})

        if stop_after_tools is not None and stop_after_tools(result):
            return result

    # Ceiling hit: the model still wanted tools. Ask once more for a final answer
    # from what it already gathered, rather than returning whatever half-thought it
    # happened to have said last.
    #
    # The tools stay DEFINED on this call even though we want none used: Bedrock
    # rejects a request whose history contains toolUse blocks when toolConfig is
    # absent ("The toolConfig field must be defined when using toolUse"), which in
    # AWS turned a question needing many searches into a 500. The instruction below
    # is what actually stops the tool calling.
    result.truncated = True
    messages.append({"role": "user", "content": [{"text": _ANSWER_NOW}]})
    final = await model.converse(
        messages,
        system=system,
        tools=specs or None,
        max_tokens=max_tokens,
        temperature=temperature,
        cache_system=cache_system,
        # The longest transcript of the turn by definition -- the one most worth
        # reading at the cache rate.
        cache_messages=cache_system,
    )
    result.usage = result.usage.add(final.usage)
    _log_call_usage(max_iterations + 1, final.usage, result.usage)
    messages.append(final.message or {"role": "assistant", "content": [{"text": final.text}]})
    if final.text:
        result.text = final.text
    _log.warning("agent_loop_truncated", iterations=max_iterations, tools=len(result.trace))
    return result


def _log_call_usage(call: int, this_call: Usage, running: Usage) -> None:
    """Log one model call's usage next to the loop's running total.

    Bedrock reports per call: ``cache_write`` is only the NEW prefix written by this
    call (the delta since the previous cache point) and ``cache_read`` is the whole
    prefix read back. Summing reads across calls therefore counts the same prefix
    once per call -- correct for billing, but only the per-call lines show what was
    actually added to the cache each round.
    """
    _log.info(
        "agent_llm_call",
        call=call,
        input_tokens=this_call.input_tokens,
        output_tokens=this_call.output_tokens,
        cache_read_tokens=this_call.cache_read_tokens,
        cache_write_tokens=this_call.cache_write_tokens,
        prompt_tokens=this_call.input_tokens + this_call.cache_read_tokens + this_call.cache_write_tokens,
        total_input_tokens=running.input_tokens,
        total_output_tokens=running.output_tokens,
        total_cache_read_tokens=running.cache_read_tokens,
        total_cache_write_tokens=running.cache_write_tokens,
    )


async def _run_tool(
    toolbox: ToolBox,
    use_id: str,
    name: str,
    arguments: dict[str, Any],
    propagate: tuple[type[BaseException], ...] = (),
) -> tuple[ToolCall, dict[str, Any]]:
    """Execute one tool call, returning its trace entry and its Converse block."""
    started = time.perf_counter()
    try:
        payload, summary = await toolbox.invoke(name, arguments)
        status = "success"
        error: str | None = None
    except propagate:
        _log.warning("tool_aborted_turn", tool=name)
        raise
    except ToolError as exc:
        # Expected, actionable failures (bad filter, unknown document) go back to
        # the model as an error result so it can correct itself.
        payload, summary, status, error = {"error": str(exc)}, str(exc), "error", str(exc)
    except Exception as exc:  # unexpected: still surfaced, never crashes the turn
        _log.exception("tool_failed", tool=name)
        message = f"{type(exc).__name__}: {exc}"
        payload, summary, status, error = {"error": message}, message, "error", message

    took_ms = round((time.perf_counter() - started) * 1000, 2)
    # Every call, not just the failures: an audit needs to show what the agent did
    # on a request that "worked" but answered wrongly, which is the harder case.
    # Arguments are small and model-authored (a query, a facet value), so they are
    # logged whole -- they are the only record of WHY a tool returned what it did.
    _log.info(
        "agent_tool_call",
        tool=name,
        status=status,
        took_ms=took_ms,
        arguments=_audit_arguments(arguments),
        summary=summary[:300],
        error=error,
    )
    block = {
        "toolResult": {
            "toolUseId": use_id,
            "content": [{"text": _serialize(payload)}],
            "status": status,
        }
    }
    call = ToolCall(name=name, input=arguments, summary=summary, took_ms=took_ms, error=error)
    return call, block


def _audit_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Tool arguments for the audit log, with any secret in them masked.

    A model-authored argument is normally a query or a facet value, but it can
    echo text out of a document -- which is exactly where a credential could be.
    """
    return {
        key: redact(value)[:400] if isinstance(value, str) else value
        for key, value in arguments.items()
    }


def _serialize(payload: Any) -> str:
    """JSON-encode a tool result, bounded so one huge result cannot blow context."""
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(payload)
    if len(text) > _MAX_RESULT_CHARS:
        return text[:_MAX_RESULT_CHARS] + " ... [truncated]"
    return text
