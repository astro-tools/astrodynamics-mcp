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


def _error_code_from_message(message: str | None) -> str | None:
    """Pull the typed ``code`` out of one of our JSON error envelopes.

    Tool failures originate as ``{"code", "message", "data"}`` JSON (see
    ``astrodynamics_mcp.server``), but FastMCP prepends
    ``"Error executing tool <name>: "`` to the message before it crosses the
    wire, and Inspect AI carries that whole string through to
    ``ChatMessageTool.error.message``. So the envelope is usually *embedded*
    in the message rather than being the entire message — we recover the
    first JSON object regardless of any prefix. Returns the ``code`` string,
    or ``None`` when no envelope is present (so a non-typed failure never
    accidentally matches an ``expect_error`` assertion).
    """
    if not message:
        return None
    envelope = _extract_json_object(message)
    if isinstance(envelope, Mapping):
        code = envelope.get("code")
        if isinstance(code, str):
            return code
    return None


def _extract_json_object(message: str) -> Any:
    """Parse the message as JSON, or the first embedded JSON object within it."""
    try:
        return json.loads(message)
    except (json.JSONDecodeError, TypeError):
        pass
    brace = message.find("{")
    if brace == -1:
        return None
    try:
        obj, _end = json.JSONDecoder().raw_decode(message[brace:])
    except (json.JSONDecodeError, TypeError):
        return None
    return obj


def extract_tool_errors(messages: list[Any]) -> dict[str, str]:
    """Map ``tool_call_id -> typed error code`` for every errored tool response.

    Inspect AI surfaces an MCP ``ToolError`` on the ``ChatMessageTool`` as a
    ``ToolCallError`` whose ``message`` is our JSON error envelope; the parsed
    ``code`` is the stable string the prompts assert against via
    ``expect_error``. Responses without an error, or whose error message isn't
    our envelope, are omitted.
    """
    errors: dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, ChatMessageTool):
            continue
        if msg.error is None or msg.tool_call_id is None:
            continue
        code = _error_code_from_message(msg.error.message)
        if code is not None:
            errors[msg.tool_call_id] = code
    return errors


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
    actual: list[ToolCall],
    permitted: list[Mapping[str, Any]],
    errors_by_id: Mapping[str, str],
) -> tuple[bool, list[str]]:
    """Match *permitted* (a single trace's steps) against *actual* as a subsequence.

    Greedy left-to-right: for each permitted step, advance the cursor
    through *actual* until a matching call is found. A call matches a step
    when the tool name and ``arg_constraints`` agree **and** its error
    status lines up with the step's ``expect_error``:

    - ``expect_error`` unset → the call must *not* have errored (a step
      expecting a usable response never matches an errored call, so a
      tool that failed silently before a retry is skipped over).
    - ``expect_error`` set → the call must have produced exactly that
      typed error code. This is what makes a silent empty success fail
      an error-path prompt: no typed error ⇒ no match.

    Returns ``(passed, failure_reasons)``; the reasons describe the first
    step that could not be matched.
    """
    cursor = 0
    for step_index, step in enumerate(permitted):
        expected_tool = step["tool"]
        constraints = step.get("arg_constraints") or {}
        expect_error = step.get("expect_error")
        first_arg_failure: list[str] | None = None
        first_error_failure: str | None = None
        match_index: int | None = None
        for i in range(cursor, len(actual)):
            call = actual[i]
            if call.function != expected_tool:
                continue
            passed, reasons = match_args(call.arguments, constraints)
            if not passed:
                if first_arg_failure is None:
                    first_arg_failure = reasons
                continue
            actual_error = errors_by_id.get(call.id)
            if expect_error is None:
                if actual_error is None:
                    match_index = i
                    break
                if first_error_failure is None:
                    first_error_failure = (
                        f"call matched args but errored with {actual_error!r}; "
                        f"step expects a successful (non-error) call"
                    )
                continue
            if actual_error == expect_error:
                match_index = i
                break
            if first_error_failure is None:
                first_error_failure = (
                    f"call matched args but error code was {actual_error!r}; "
                    f"step expects expect_error={expect_error!r}"
                )
        if match_index is None:
            if first_error_failure is not None:
                return False, [f"step {step_index} ({expected_tool!r}): {first_error_failure}"]
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
    actual: list[ToolCall],
    permitted_traces: list[list[Mapping[str, Any]]],
    errors_by_id: Mapping[str, str],
) -> tuple[bool, list[str]]:
    """At least one permitted trace must match; return overall pass + per-trace reasons."""
    if not permitted_traces:
        return False, ["no permitted_traces declared for this prompt"]
    all_reasons: list[str] = []
    for i, trace in enumerate(permitted_traces):
        passed, reasons = _match_trace(actual, trace, errors_by_id)
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
        errors_by_id = extract_tool_errors(state.messages)
        trace_passed, trace_reasons = _trace_check(trace, permitted_traces, errors_by_id)

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
