"""Hybrid-scorer regression tests with synthetic Inspect AI ``TaskState`` fixtures.

These cover the two negative controls called out in the eval-suite
acceptance criteria — wrong-tool-called and right-tool-wrong-arg — plus
the positive-control and the cross-fire cases (trace-pass+functional-fail
and trace-fail+functional-pass) needed to prove neither check alone
suffices.

We construct ``TaskState`` directly (no LLM, no MCP server) so the tests
are deterministic, fast, and re-runnable without network access.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from eval.scoring import (
    extract_final_tool_response,
    extract_tool_errors,
    extract_trace,
    hybrid_scorer,
)
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    ModelOutput,
)
from inspect_ai.model._model import ModelName
from inspect_ai.scorer import Score, Target
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolCall
from inspect_ai.tool._tool_call import ToolCallError


def _tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, function=name, arguments=args)


def _error_envelope(code: str) -> str:
    """A JSON error envelope shaped like astrodynamics_mcp.server emits."""
    return json.dumps({"code": code, "message": f"{code} raised", "data": {}})


def _build_state(
    *,
    user_prompt: str,
    assistant_calls: list[ToolCall],
    tool_responses: list[tuple[str, dict[str, Any]]],
    metadata: dict[str, Any],
    tool_errors: dict[str, str] | None = None,
) -> TaskState:
    """Assemble a ``TaskState`` carrying a synthetic LLM conversation.

    ``tool_responses`` is ``[(tool_call_id, response_payload), ...]`` —
    each gets JSON-encoded onto a :class:`ChatMessageTool`. ``tool_errors``
    maps a ``tool_call_id`` to a typed error code; the matching tool message
    gets a :class:`ToolCallError` carrying the JSON error envelope, mirroring
    how Inspect AI surfaces an MCP ``ToolError``.
    """
    errors = tool_errors or {}
    sample = Sample(input=user_prompt, metadata=metadata)
    messages: list[Any] = [
        ChatMessageUser(content=user_prompt),
        ChatMessageAssistant(content="", tool_calls=assistant_calls),
    ]
    for call_id, payload in tool_responses:
        code = errors.get(call_id)
        if code is not None:
            envelope = _error_envelope(code)
            messages.append(
                ChatMessageTool(
                    content=envelope,
                    tool_call_id=call_id,
                    error=ToolCallError("unknown", envelope),
                )
            )
        else:
            messages.append(
                ChatMessageTool(
                    content=json.dumps(payload),
                    tool_call_id=call_id,
                )
            )
    state = TaskState(
        model=ModelName("test/model"),
        sample_id=sample.id or 0,
        epoch=0,
        input=user_prompt,
        messages=messages,
        output=ModelOutput.from_content(model="test/model", content=""),
        metadata=metadata,
    )
    return state


async def _score(state: TaskState) -> tuple[Score, dict[str, Any]]:
    """Invoke the hybrid scorer; assert non-None and return ``(score, narrowed_metadata)``.

    Wrapping the return narrows ``score.metadata`` for downstream call sites
    that index into it (the scorer always populates metadata, but the type
    annotation declares it optional).
    """
    result = await hybrid_scorer()(state, Target(""))
    assert result is not None
    assert result.metadata is not None
    return result, result.metadata


def _good_prompt_metadata() -> dict[str, Any]:
    return {
        "tier": "single_tool",
        "tools_required": ["tle_lookup"],
        "permitted_traces": [
            [
                {
                    "tool": "tle_lookup",
                    "arg_constraints": {"query": {"equals": "25544"}},
                }
            ]
        ],
        "functional_answer": [
            {"path": "$.results[0].norad_id", "equals": "25544"},
            {"path": "$.results", "length": {"min": 1}},
        ],
        "notes": None,
    }


def _good_response() -> dict[str, Any]:
    return {
        "results": [
            {
                "name": "ISS (ZARYA)",
                "norad_id": "25544",
                "tle_line1": "1 " + "x" * 67,
                "tle_line2": "2 " + "x" * 67,
                "omm": {},
                "fetched_at": "2026-05-25T00:00:00Z",
                "stale": False,
            }
        ]
    }


class TestExtractHelpers:
    def test_extract_trace_flattens_assistant_tool_calls(self) -> None:
        msgs: list[Any] = [
            ChatMessageUser(content="hi"),
            ChatMessageAssistant(
                content="",
                tool_calls=[_tool_call("tle_lookup", {"query": "25544"}, "c1")],
            ),
            ChatMessageTool(content="{}", tool_call_id="c1"),
            ChatMessageAssistant(
                content="",
                tool_calls=[_tool_call("sgp4_propagate", {"frame": "TEME"}, "c2")],
            ),
        ]
        trace = extract_trace(msgs)
        assert [c.function for c in trace] == ["tle_lookup", "sgp4_propagate"]

    def test_extract_final_tool_response_parses_last_tool_message(self) -> None:
        msgs: list[Any] = [
            ChatMessageTool(content='{"a": 1}', tool_call_id="c1"),
            ChatMessageTool(content='{"b": 2}', tool_call_id="c2"),
        ]
        response, err = extract_final_tool_response(msgs)
        assert err is None
        assert response == {"b": 2}

    def test_extract_final_tool_response_handles_missing_tool(self) -> None:
        msgs: list[Any] = [ChatMessageUser(content="hi")]
        response, err = extract_final_tool_response(msgs)
        assert response is None
        assert err is not None
        assert "no ChatMessageTool" in err

    def test_extract_final_tool_response_handles_invalid_json(self) -> None:
        msgs: list[Any] = [ChatMessageTool(content="not-json", tool_call_id="c1")]
        response, err = extract_final_tool_response(msgs)
        assert response is None
        assert err is not None
        assert "not valid JSON" in err

    def test_extract_tool_errors_reads_typed_code(self) -> None:
        envelope = _error_envelope("credential_required.spacetrack")
        msgs: list[Any] = [
            ChatMessageTool(
                content=envelope,
                tool_call_id="c1",
                error=ToolCallError("unknown", envelope),
            )
        ]
        assert extract_tool_errors(msgs) == {"c1": "credential_required.spacetrack"}

    def test_extract_tool_errors_skips_non_envelope_errors(self) -> None:
        msgs: list[Any] = [
            ChatMessageTool(
                content="boom",
                tool_call_id="c1",
                error=ToolCallError("unknown", "plain non-JSON failure"),
            )
        ]
        assert extract_tool_errors(msgs) == {}

    def test_extract_tool_errors_omits_successful_responses(self) -> None:
        msgs: list[Any] = [ChatMessageTool(content='{"ok": true}', tool_call_id="c1")]
        assert extract_tool_errors(msgs) == {}


@pytest.mark.asyncio
class TestHybridScorer:
    """The load-bearing scorer behaviour per the eval-suite acceptance criteria."""

    async def test_positive_control_passes(self) -> None:
        metadata = _good_prompt_metadata()
        state = _build_state(
            user_prompt="Fetch the ISS TLE.",
            assistant_calls=[_tool_call("tle_lookup", {"query": "25544"})],
            tool_responses=[("c1", _good_response())],
            metadata=metadata,
        )
        score, meta = await _score(state)
        assert score.value == 1.0
        assert meta["trace_passed"] is True
        assert meta["functional_passed"] is True

    async def test_wrong_tool_called_fails(self) -> None:
        """Negative control 1: model picks the wrong tool entirely."""
        metadata = _good_prompt_metadata()
        state = _build_state(
            user_prompt="Fetch the ISS TLE.",
            assistant_calls=[_tool_call("sgp4_propagate", {"frame": "TEME"})],
            tool_responses=[("c1", {"states": []})],
            metadata=metadata,
        )
        score, meta = await _score(state)
        assert score.value == 0.0
        assert meta["trace_passed"] is False
        assert any(
            "tle_lookup" in r and "no matching call" in r for r in meta["trace_failure_reasons"]
        )

    async def test_right_tool_wrong_arg_fails(self) -> None:
        """Negative control 2: right tool, wrong-arg binding."""
        metadata = _good_prompt_metadata()
        state = _build_state(
            user_prompt="Fetch the ISS TLE.",
            assistant_calls=[_tool_call("tle_lookup", {"query": "WRONG-ID"})],
            tool_responses=[("c1", _good_response())],
            metadata=metadata,
        )
        score, meta = await _score(state)
        assert score.value == 0.0
        assert meta["trace_passed"] is False
        assert any("arg_constraints" in r and "query" in r for r in meta["trace_failure_reasons"])

    async def test_trace_pass_functional_fail(self) -> None:
        """Trace matches but the response shape doesn't — answer-by-coincidence guard."""
        metadata = _good_prompt_metadata()
        bad_response = {"results": [{"norad_id": "00000"}]}
        state = _build_state(
            user_prompt="Fetch the ISS TLE.",
            assistant_calls=[_tool_call("tle_lookup", {"query": "25544"})],
            tool_responses=[("c1", bad_response)],
            metadata=metadata,
        )
        score, meta = await _score(state)
        assert score.value == 0.0
        assert meta["trace_passed"] is True
        assert meta["functional_passed"] is False

    async def test_alternative_trace_branch_passes(self) -> None:
        """At least one permitted trace matches — alternative branches are honoured."""
        metadata = _good_prompt_metadata()
        # Add a second permitted trace branch (e.g. name lookup).
        metadata["permitted_traces"].append(
            [
                {
                    "tool": "tle_lookup",
                    "arg_constraints": {"query": {"case_insensitive_contains": "ISS"}},
                }
            ]
        )
        state = _build_state(
            user_prompt="Fetch the ISS TLE.",
            assistant_calls=[_tool_call("tle_lookup", {"query": "iss (zarya)"})],
            tool_responses=[("c1", _good_response())],
            metadata=metadata,
        )
        score, _meta = await _score(state)
        assert score.value == 1.0

    async def test_sequential_trace_with_extra_calls_tolerated(self) -> None:
        """An extra interleaved call doesn't break a subsequence match."""
        metadata = {
            "tier": "sequential",
            "tools_required": ["tle_lookup", "sgp4_propagate"],
            "permitted_traces": [
                [
                    {"tool": "tle_lookup", "arg_constraints": {"query": {"equals": "25544"}}},
                    {
                        "tool": "sgp4_propagate",
                        "arg_constraints": {"frame": {"one_of": ["TEME", None]}},
                    },
                ]
            ],
            "functional_answer": [],
            "notes": None,
        }
        # Model retried tle_lookup with the name first, then by id, then propagated.
        state = _build_state(
            user_prompt="Fetch and propagate.",
            assistant_calls=[
                _tool_call("tle_lookup", {"query": "ISS"}, "c1"),
                _tool_call("tle_lookup", {"query": "25544"}, "c2"),
                _tool_call("sgp4_propagate", {}, "c3"),
            ],
            tool_responses=[("c1", {}), ("c2", _good_response()), ("c3", {})],
            metadata=metadata,
        )
        score, _meta = await _score(state)
        assert score.value == 1.0

    async def test_explanation_lists_both_checks(self) -> None:
        metadata = _good_prompt_metadata()
        state = _build_state(
            user_prompt="Fetch the ISS TLE.",
            assistant_calls=[_tool_call("tle_lookup", {"query": "25544"})],
            tool_responses=[("c1", _good_response())],
            metadata=metadata,
        )
        score, _meta = await _score(state)
        assert score.explanation is not None
        assert "trace_check: PASS" in score.explanation
        assert "functional_check: PASS" in score.explanation


def _error_path_metadata(expect_error: str, *, source: str = "space-track") -> dict[str, Any]:
    """An error-path prompt: the call must raise ``expect_error``."""
    return {
        "tier": "single_tool",
        "tools_required": ["tle_lookup"],
        "permitted_traces": [
            [
                {
                    "tool": "tle_lookup",
                    "arg_constraints": {"source": {"equals": source}},
                    "expect_error": expect_error,
                }
            ]
        ],
        "functional_answer": [],
        "notes": None,
    }


@pytest.mark.asyncio
class TestErrorPathScoring:
    """expect_error verification — the credentialed-error-path contract."""

    async def test_typed_error_matches_expect_error(self) -> None:
        metadata = _error_path_metadata("credential_required.spacetrack")
        state = _build_state(
            user_prompt="Fetch NORAD 25544 from Space-Track.",
            assistant_calls=[_tool_call("tle_lookup", {"query": "25544", "source": "space-track"})],
            tool_responses=[("c1", {})],
            tool_errors={"c1": "credential_required.spacetrack"},
            metadata=metadata,
        )
        score, meta = await _score(state)
        assert score.value == 1.0
        assert meta["trace_passed"] is True

    async def test_silent_empty_success_fails_expect_error(self) -> None:
        """The negative control: a non-error empty response must NOT satisfy expect_error."""
        metadata = _error_path_metadata("credential_required.spacetrack")
        state = _build_state(
            user_prompt="Fetch NORAD 25544 from Space-Track.",
            assistant_calls=[_tool_call("tle_lookup", {"query": "25544", "source": "space-track"})],
            tool_responses=[("c1", {"results": []})],  # silent empty success, no error
            metadata=metadata,
        )
        score, meta = await _score(state)
        assert score.value == 0.0
        assert meta["trace_passed"] is False
        assert any("expect_error" in r for r in meta["trace_failure_reasons"])

    async def test_wrong_error_code_fails(self) -> None:
        metadata = _error_path_metadata("credential_required.spacetrack")
        state = _build_state(
            user_prompt="Fetch NORAD 25544 from Space-Track.",
            assistant_calls=[_tool_call("tle_lookup", {"query": "25544", "source": "space-track"})],
            tool_responses=[("c1", {})],
            tool_errors={"c1": "upstream.spacetrack_unreachable"},
            metadata=metadata,
        )
        score, meta = await _score(state)
        assert score.value == 0.0
        assert meta["trace_passed"] is False

    async def test_happy_path_step_does_not_match_errored_call(self) -> None:
        """A step with no expect_error must not match a call that errored."""
        metadata = _good_prompt_metadata()
        metadata["functional_answer"] = []
        state = _build_state(
            user_prompt="Fetch the ISS TLE.",
            assistant_calls=[_tool_call("tle_lookup", {"query": "25544"})],
            tool_responses=[("c1", {})],
            tool_errors={"c1": "upstream.celestrak_unreachable"},
            metadata=metadata,
        )
        score, meta = await _score(state)
        assert score.value == 0.0
        assert meta["trace_passed"] is False

    async def test_errored_call_then_successful_retry_matches_happy_path(self) -> None:
        """Greedy matcher skips the errored call and anchors on the later success."""
        metadata = _good_prompt_metadata()
        metadata["functional_answer"] = []
        state = _build_state(
            user_prompt="Fetch the ISS TLE.",
            assistant_calls=[
                _tool_call("tle_lookup", {"query": "25544"}, "c1"),
                _tool_call("tle_lookup", {"query": "25544"}, "c2"),
            ],
            tool_responses=[("c1", {}), ("c2", _good_response())],
            tool_errors={"c1": "upstream.celestrak_unreachable"},
            metadata=metadata,
        )
        score, meta = await _score(state)
        assert score.value == 1.0
        assert meta["trace_passed"] is True
