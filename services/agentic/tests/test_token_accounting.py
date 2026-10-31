"""Reported tokens must cover the WHOLE request, tool calling included.

ExampleApp logs these numbers per request, so an undercount is a billing and
capacity figure that quietly disagrees with the AWS bill.

Bedrock bills tool work as ordinary tokens: the tool definitions and every
`toolResult` land in a later round's `inputTokens`, and the model's `toolUse`
blocks land in `outputTokens`. So the requirement is that EVERY round is summed —
not just the one that produced the final text — and that a delegated agent's
spend is folded into the orchestrator's total.
"""

from __future__ import annotations

from typing import Any

import pytest
from rag_agentic.agent_loop import run_agent
from rag_agentic.converse import ModelTurn, ScriptedConverseModel, ToolSpec, ToolUse
from rag_agentic.models import Usage


class _Toolbox:
    """Two tools' worth of surface; records what it was asked."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="search_documents",
                description="search",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ]

    async def invoke(self, name: str, arguments: dict[str, Any]) -> tuple[Any, str]:
        self.calls.append(name)
        return {"passages": []}, "0 passages"


def _tool_turn(tokens: tuple[int, int]) -> ModelTurn:
    return ModelTurn(
        text="",
        tool_uses=[ToolUse(id="t1", name="search_documents", input={"query": "x"})],
        stop_reason="tool_use",
        usage=Usage(input_tokens=tokens[0], output_tokens=tokens[1], model="m"),
        message={
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "t1", "name": "search_documents",
                                     "input": {"query": "x"}}}],
        },
    )


def _text_turn(text: str, tokens: tuple[int, int]) -> ModelTurn:
    return ModelTurn(
        text=text,
        stop_reason="end_turn",
        usage=Usage(input_tokens=tokens[0], output_tokens=tokens[1], model="m"),
        message={"role": "assistant", "content": [{"text": text}]},
    )


async def test_every_round_is_counted_not_just_the_last() -> None:
    """Three rounds: two tool calls then the answer. Undercounting would report
    only the final round, which is the cheapest one."""
    model = ScriptedConverseModel([
        _tool_turn((1000, 50)),   # round 1: asks for a tool
        _tool_turn((3000, 60)),   # round 2: tool results are in THIS input
        _text_turn("done", (5000, 200)),
    ])

    result = await run_agent(model, system="s", user_text="q", toolbox=_Toolbox())

    assert result.usage.input_tokens == 9000
    assert result.usage.output_tokens == 310
    assert result.usage.total_tokens == 9310


async def test_tool_rounds_dominate_the_total() -> None:
    """The point of the above: the tool rounds are most of the spend, so dropping
    them would understate a search request several-fold."""
    with_tools = ScriptedConverseModel([
        _tool_turn((1000, 50)), _tool_turn((3000, 60)), _text_turn("done", (5000, 200)),
    ])
    without = ScriptedConverseModel([_text_turn("done", (1000, 100))])

    busy = await run_agent(with_tools, system="s", user_text="q", toolbox=_Toolbox())
    plain = await run_agent(without, system="s", user_text="q", toolbox=_Toolbox())

    assert busy.usage.total_tokens > plain.usage.total_tokens * 5


async def test_a_single_round_still_adds_up() -> None:
    model = ScriptedConverseModel([_text_turn("hello", (900, 40))])

    result = await run_agent(model, system="s", user_text="hi", toolbox=_Toolbox())

    assert (result.usage.input_tokens, result.usage.output_tokens) == (900, 40)
    assert result.usage.total_tokens == 940


def test_a_delegated_agent_s_spend_is_added() -> None:
    """The orchestrator reports ONE number for the whole request, so the search
    agent's rounds must be folded into its own."""
    router = Usage(input_tokens=1200, output_tokens=90, model="m")
    specialist = Usage(input_tokens=47000, output_tokens=850, model="m")

    combined = router.add(specialist)

    assert combined.input_tokens == 48200
    assert combined.output_tokens == 940
    assert combined.total_tokens == 49140


def test_cache_reads_are_not_added_on_top() -> None:
    """Bedrock already counts a cache read inside inputTokens. Adding it again
    would overstate every cached turn — which, since prompt caching landed, is
    most of them."""
    usage = Usage(input_tokens=1426, output_tokens=835, cache_read_tokens=47563)

    assert usage.total_tokens == 2261


@pytest.mark.parametrize(
    ("rounds", "expected"),
    [(1, 1000), (2, 2000), (5, 5000)],
)
async def test_totals_scale_with_the_number_of_rounds(rounds: int, expected: int) -> None:
    turns = [_tool_turn((1000, 0)) for _ in range(rounds - 1)]
    turns.append(_text_turn("done", (1000, 0)))

    result = await run_agent(
        ScriptedConverseModel(turns), system="s", user_text="q", toolbox=_Toolbox()
    )

    assert result.usage.input_tokens == expected
