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
    extract_attachment_kinds,
    extract_errored_call_ids,
    extract_final_tool_response,
    extract_trace,
    hybrid_scorer,
)
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    ContentImage,
    ContentText,
    ModelOutput,
)
from inspect_ai.model._model import ModelName
from inspect_ai.scorer import Score, Target
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolCall
from inspect_ai.tool._tool_call import ToolCallError


def _tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, function=name, arguments=args)


def _build_state(
    *,
    user_prompt: str,
    assistant_calls: list[ToolCall],
    tool_responses: list[tuple[str, dict[str, Any]]],
    metadata: dict[str, Any],
    errored_ids: set[str] | None = None,
) -> TaskState:
    """Assemble a ``TaskState`` carrying a synthetic LLM conversation.

    ``tool_responses`` is ``[(tool_call_id, response_payload), ...]`` — each
    gets JSON-encoded onto a :class:`ChatMessageTool`. Any id in
    ``errored_ids`` gets a :class:`ToolCallError` instead, mirroring how
    Inspect AI surfaces a failed tool call.
    """
    errored = errored_ids or set()
    sample = Sample(input=user_prompt, metadata=metadata)
    messages: list[Any] = [
        ChatMessageUser(content=user_prompt),
        ChatMessageAssistant(content="", tool_calls=assistant_calls),
    ]
    for call_id, payload in tool_responses:
        if call_id in errored:
            messages.append(
                ChatMessageTool(
                    content="Error executing tool: the tool failed",
                    tool_call_id=call_id,
                    error=ToolCallError("unknown", "the tool failed"),
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

    def test_extract_errored_call_ids_collects_failed_calls(self) -> None:
        msgs: list[Any] = [
            ChatMessageTool(
                content="boom",
                tool_call_id="c1",
                error=ToolCallError("unknown", "the tool failed"),
            ),
            ChatMessageTool(content='{"ok": true}', tool_call_id="c2"),
        ]
        assert extract_errored_call_ids(msgs) == {"c1"}

    def test_extract_errored_call_ids_empty_when_all_succeed(self) -> None:
        msgs: list[Any] = [ChatMessageTool(content='{"ok": true}', tool_call_id="c1")]
        assert extract_errored_call_ids(msgs) == set()


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


@pytest.mark.asyncio
class TestErroredCallGuard:
    """A trace step must not match a tool call that errored."""

    async def test_step_does_not_match_errored_call(self) -> None:
        metadata = _good_prompt_metadata()
        metadata["functional_answer"] = []
        state = _build_state(
            user_prompt="Fetch the ISS TLE.",
            assistant_calls=[_tool_call("tle_lookup", {"query": "25544"})],
            tool_responses=[("c1", {})],
            errored_ids={"c1"},
            metadata=metadata,
        )
        score, meta = await _score(state)
        assert score.value == 0.0
        assert meta["trace_passed"] is False
        assert any("errored" in r for r in meta["trace_failure_reasons"])

    async def test_errored_call_then_successful_retry_matches(self) -> None:
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
            errored_ids={"c1"},
            metadata=metadata,
        )
        score, meta = await _score(state)
        assert score.value == 1.0
        assert meta["trace_passed"] is True


# A short fake PNG data URI — the scorer only checks the block *type*, not the
# bytes, so a placeholder is enough to stand in for a rendered ImageContent.
_FAKE_IMAGE_URI = "data:image/png;base64,iVBORw0KGgo="
_FAKE_CZML = '[{"id": "document", "version": "1.0", "name": "trajectory"}]'


def _viz_metadata(tool: str, expected_attachment: str) -> dict[str, Any]:
    return {
        "tier": "single_tool",
        "tools_required": [tool],
        "permitted_traces": [[{"tool": tool, "arg_constraints": {}}]],
        "functional_answer": [],
        "expected_attachment": expected_attachment,
        "notes": None,
    }


def _build_viz_state(
    *,
    tool: str,
    tool_content: Any,
    metadata: dict[str, Any],
) -> TaskState:
    """Assemble a ``TaskState`` whose single viz tool message carries *tool_content*.

    ``tool_content`` is whatever a viz tool's :class:`ChatMessageTool` would carry
    — a ``list[Content]`` leading with the ASCII summary text block followed by an
    attachment (an :class:`ContentImage` for a PNG, a second :class:`ContentText`
    for a CZML embedded resource), mirroring Inspect AI's MCP bridge.
    """
    call = _tool_call(tool, {}, "v1")
    sample = Sample(input="please visualise", metadata=metadata)
    messages: list[Any] = [
        ChatMessageUser(content="please visualise"),
        ChatMessageAssistant(content="", tool_calls=[call]),
        ChatMessageTool(content=tool_content, tool_call_id="v1"),
    ]
    return TaskState(
        model=ModelName("test/model"),
        sample_id=sample.id or 0,
        epoch=0,
        input="please visualise",
        messages=messages,
        output=ModelOutput.from_content(model="test/model", content=""),
        metadata=metadata,
    )


class TestAttachmentExtraction:
    """``extract_attachment_kinds`` reads the structural shape Inspect's bridge produces."""

    def test_image_block_is_an_image_attachment(self) -> None:
        msgs: list[Any] = [
            ChatMessageTool(
                content=[
                    ContentText(text="Ground track summary. PNG attached."),
                    ContentImage(image=_FAKE_IMAGE_URI),
                ],
                tool_call_id="v1",
            )
        ]
        assert extract_attachment_kinds(msgs) == {"image"}

    def test_trailing_text_block_is_a_resource_attachment(self) -> None:
        msgs: list[Any] = [
            ChatMessageTool(
                content=[
                    ContentText(text="CZML summary. document attached."),
                    ContentText(text=_FAKE_CZML),
                ],
                tool_call_id="v1",
            )
        ]
        assert extract_attachment_kinds(msgs) == {"resource"}

    def test_single_text_block_has_no_attachment(self) -> None:
        msgs: list[Any] = [
            ChatMessageTool(content=[ContentText(text='{"states": []}')], tool_call_id="c1")
        ]
        assert extract_attachment_kinds(msgs) == set()

    def test_string_content_has_no_attachment(self) -> None:
        msgs: list[Any] = [ChatMessageTool(content='{"states": []}', tool_call_id="c1")]
        assert extract_attachment_kinds(msgs) == set()


@pytest.mark.asyncio
class TestAttachmentScorer:
    """The attachment check gates the viz prompts on attachment presence + type."""

    async def test_image_attachment_present_passes(self) -> None:
        metadata = _viz_metadata("plot_ground_track", "image")
        state = _build_viz_state(
            tool="plot_ground_track",
            tool_content=[
                ContentText(text="Ground track summary. PNG attached."),
                ContentImage(image=_FAKE_IMAGE_URI),
            ],
            metadata=metadata,
        )
        score, meta = await _score(state)
        assert score.value == 1.0
        assert meta["attachment_passed"] is True

    async def test_resource_attachment_present_passes(self) -> None:
        metadata = _viz_metadata("czml_trajectory", "resource")
        state = _build_viz_state(
            tool="czml_trajectory",
            tool_content=[
                ContentText(text="CZML summary. document attached."),
                ContentText(text=_FAKE_CZML),
            ],
            metadata=metadata,
        )
        score, meta = await _score(state)
        assert score.value == 1.0
        assert meta["attachment_passed"] is True

    async def test_missing_attachment_fails(self) -> None:
        """Trace matches but no attachment came back — the PNG never rendered."""
        metadata = _viz_metadata("plot_ground_track", "image")
        state = _build_viz_state(
            tool="plot_ground_track",
            tool_content=[ContentText(text="Ground track summary (no image).")],
            metadata=metadata,
        )
        score, meta = await _score(state)
        assert score.value == 0.0
        assert meta["trace_passed"] is True
        assert meta["attachment_passed"] is False
        assert any("image" in r for r in meta["attachment_failure_reasons"])

    async def test_wrong_attachment_type_fails(self) -> None:
        """A CZML resource does not satisfy a golden expecting a PNG image."""
        metadata = _viz_metadata("plot_ground_track", "image")
        state = _build_viz_state(
            tool="plot_ground_track",
            tool_content=[
                ContentText(text="Summary."),
                ContentText(text=_FAKE_CZML),
            ],
            metadata=metadata,
        )
        score, meta = await _score(state)
        assert score.value == 0.0
        assert meta["attachment_passed"] is False

    async def test_non_viz_prompt_attachment_vacuously_passes(self) -> None:
        """A prompt with no expected_attachment never fails the attachment check."""
        metadata = _good_prompt_metadata()
        assert "expected_attachment" not in metadata  # the default (None) path
        state = _build_state(
            user_prompt="Fetch the ISS TLE.",
            assistant_calls=[_tool_call("tle_lookup", {"query": "25544"})],
            tool_responses=[("c1", _good_response())],
            metadata=metadata,
        )
        score, meta = await _score(state)
        assert score.value == 1.0
        assert meta["attachment_passed"] is True

    async def test_explanation_lists_attachment_check(self) -> None:
        metadata = _viz_metadata("plot_trajectory", "image")
        state = _build_viz_state(
            tool="plot_trajectory",
            tool_content=[
                ContentText(text="Trajectory summary. PNG attached."),
                ContentImage(image=_FAKE_IMAGE_URI),
            ],
            metadata=metadata,
        )
        score, _meta = await _score(state)
        assert score.explanation is not None
        assert "attachment_check: PASS" in score.explanation
