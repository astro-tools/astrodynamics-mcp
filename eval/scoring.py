"""Hybrid trace + functional-answer scorer.

For each sample, the scorer reconstructs the LLM's tool-call trace from
``state.messages`` and parses the final tool response. A prompt scores 1
iff:

1. **Trace check.** At least one of the prompt's ``permitted_traces``
   appears as an in-order subsequence of the actual tool calls, with
   every named tool's ``arg_constraints`` satisfied by the recorded
   argument dict. Extra interleaved tool calls (model exploration,
   retries) are tolerated.
2. **Functional check.** Every entry in ``functional_answer`` evaluates
   to true against the final ``ChatMessageTool``'s parsed JSON.

The two checks catch genuinely different failure modes — see
``eval/README.md`` for the rationale. Sub-scores and per-step failure
reasons are surfaced via the :class:`Score` ``metadata`` and
``explanation`` so the PR-comment workflow can show why a prompt failed
rather than just the boolean.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolCall

from eval._constraints import match_args
from eval._functional import evaluate_checks


def extract_trace(messages: list[Any]) -> list[ToolCall]:
    """Flatten the assistant messages' ``tool_calls`` into a single in-order list."""
    trace: list[ToolCall] = []
    for msg in messages:
        if isinstance(msg, ChatMessageAssistant) and msg.tool_calls:
            trace.extend(msg.tool_calls)
    return trace


def extract_final_tool_response(messages: list[Any]) -> tuple[Any | None, str | None]:
    """Return ``(parsed_json, error_reason)`` for the last ``ChatMessageTool``.

    ``parsed_json`` is the deserialised body of the last tool message;
    ``error_reason`` is non-``None`` when there is no tool message at all
    or the message text fails to parse as JSON. Exactly one of the two is
    populated.
    """
    last_tool_msg: ChatMessageTool | None = None
    for msg in messages:
        if isinstance(msg, ChatMessageTool):
            last_tool_msg = msg
    if last_tool_msg is None:
        return None, "no ChatMessageTool message found in trace"
    text = last_tool_msg.text
    if not text:
        return None, "final tool message has empty text content"
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, f"final tool message is not valid JSON: {exc}"


def _match_trace(
    actual: list[ToolCall], permitted: list[Mapping[str, Any]]
) -> tuple[bool, list[str]]:
    """Match *permitted* (a single trace's steps) against *actual* as a subsequence.

    Greedy left-to-right: for each permitted step, advance the cursor
    through *actual* until a matching call is found. Returns
    ``(passed, failure_reasons)``; the reasons describe the first step
    that could not be matched (subsequent steps are not evaluated since
    they have no anchor).
    """
    cursor = 0
    for step_index, step in enumerate(permitted):
        expected_tool = step["tool"]
        constraints = step.get("arg_constraints") or {}
        first_arg_failure: list[str] | None = None
        match_index: int | None = None
        for i in range(cursor, len(actual)):
            call = actual[i]
            if call.function != expected_tool:
                continue
            passed, reasons = match_args(call.arguments, constraints)
            if passed:
                match_index = i
                break
            if first_arg_failure is None:
                first_arg_failure = reasons
        if match_index is None:
            if first_arg_failure is not None:
                return False, [
                    f"step {step_index} ({expected_tool!r}): all candidate calls failed "
                    f"arg_constraints (first candidate's failures: {first_arg_failure})"
                ]
            return False, [
                f"step {step_index} ({expected_tool!r}): no matching call in trace "
                f"after position {cursor}; actual tools seen = "
                f"{[c.function for c in actual]}"
            ]
        cursor = match_index + 1
    return True, []


def _trace_check(
    actual: list[ToolCall], permitted_traces: list[list[Mapping[str, Any]]]
) -> tuple[bool, list[str]]:
    """At least one permitted trace must match; return overall pass + per-trace reasons."""
    if not permitted_traces:
        return False, ["no permitted_traces declared for this prompt"]
    all_reasons: list[str] = []
    for i, trace in enumerate(permitted_traces):
        passed, reasons = _match_trace(actual, trace)
        if passed:
            return True, []
        all_reasons.append(f"trace[{i}] failed: {'; '.join(reasons)}")
    return False, all_reasons


@scorer(metrics=[accuracy(), stderr()])
def hybrid_scorer() -> Scorer:
    """Inspect AI Scorer combining permitted-trace and functional-answer checks.

    The prompt's spec (permitted_traces + functional_answer) is read from
    ``state.metadata`` — :mod:`eval.tasks` writes it there when building the
    dataset. The scorer is pure: same inputs always yield the same Score.
    """

    async def score(state: TaskState, target: Target) -> Score:
        del target  # the prompt spec lives in state.metadata, not target

        permitted_traces = state.metadata.get("permitted_traces") or []
        functional_answer = state.metadata.get("functional_answer") or []

        trace = extract_trace(state.messages)
        trace_passed, trace_reasons = _trace_check(trace, permitted_traces)

        response, parse_error = extract_final_tool_response(state.messages)
        if parse_error is not None:
            functional_passed = not functional_answer  # no tool, no checks → vacuous pass
            functional_reasons = [] if functional_passed else [parse_error]
        else:
            functional_passed, functional_reasons = evaluate_checks(response, functional_answer)

        overall = trace_passed and functional_passed
        explanation_lines = [
            f"trace_check: {'PASS' if trace_passed else 'FAIL'}",
            *(f"  - {r}" for r in trace_reasons),
            f"functional_check: {'PASS' if functional_passed else 'FAIL'}",
            *(f"  - {r}" for r in functional_reasons),
        ]
        return Score(
            value=1.0 if overall else 0.0,
            answer=state.output.completion if state.output else None,
            explanation="\n".join(explanation_lines),
            metadata={
                "trace_passed": trace_passed,
                "functional_passed": functional_passed,
                "trace_failure_reasons": trace_reasons,
                "functional_failure_reasons": functional_reasons,
                "actual_trace": [{"tool": c.function, "arguments": c.arguments} for c in trace],
            },
        )

    return score
