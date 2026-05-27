"""Tests for `astrodynamics_mcp.server_lint`.

Each test registers a fixture tool against a *per-test* FastMCP instance
to keep the real module-level singleton clean. The lint then runs against
``await per_test_mcp.list_tools()``.
"""

from __future__ import annotations

import pytest
from mcp.server.fastmcp import FastMCP

from astrodynamics_mcp.server import mcp as real_mcp
from astrodynamics_mcp.server_lint import (
    COMMON_MISTAKE_HINTS,
    DescriptionViolation,
    check_tool_descriptions,
)


class TestRealServerSurfaceIsClean:
    """Every tool registered against the real module-level instance must pass lint."""

    async def test_v01_surface_passes_lint(self) -> None:
        tools = await real_mcp.list_tools()
        assert check_tool_descriptions(tools) == []


class TestRealServerCapabilityMetadata:
    """MCP-client ergonomics guard: every tool needs annotations, outputSchema, and
    a non-empty description on every input parameter so that any LLM/client
    consuming `tools/list` over the wire gets the full Tool shape."""

    async def test_every_tool_has_annotations(self) -> None:
        tools = await real_mcp.list_tools()
        for t in tools:
            assert t.annotations is not None, f"{t.name}: missing ToolAnnotations"
            assert t.annotations.readOnlyHint is True, (
                f"{t.name}: readOnlyHint must be True (none of the v0.1 tools mutate state)"
            )

    async def test_every_tool_has_output_schema(self) -> None:
        tools = await real_mcp.list_tools()
        for t in tools:
            assert t.outputSchema is not None, (
                f"{t.name}: outputSchema is None. Ensure the tool returns a typed "
                "pydantic BaseModel so FastMCP can derive a schema."
            )

    async def test_every_parameter_has_a_description(self) -> None:
        tools = await real_mcp.list_tools()
        for t in tools:
            props = (t.inputSchema or {}).get("properties", {})
            for param_name, schema in props.items():
                assert schema.get("description"), (
                    f"{t.name}.{param_name}: parameter is missing a Field(description=…). "
                    "Declare the parameter as "
                    "Annotated[type, Field(description='…')] so the description lands "
                    "in the generated JSON Schema."
                )


class TestGoodToolPassesLint:
    @pytest.fixture
    def lint_mcp(self) -> FastMCP:
        m = FastMCP("good-fixture")

        @m.tool(
            name="good_tool",
            description=(
                "Resolve a satellite TLE by name or NORAD ID, e.g. "
                "tle_lookup('25544') for the ISS. Returns parsed OMM JSON plus the raw "
                "two-line element strings."
            ),
        )
        async def good_tool(query: str) -> dict[str, str]:
            return {"name": query}

        return m

    async def test_no_violations(self, lint_mcp: FastMCP) -> None:
        tools = await lint_mcp.list_tools()
        assert check_tool_descriptions(tools) == []


class TestMissingExampleFlagged:
    @pytest.fixture
    def lint_mcp(self) -> FastMCP:
        m = FastMCP("missing-example-fixture")

        @m.tool(
            name="no_example",
            # Long enough to pass the length rule, but no example marker.
            description=(
                "Compute something useful. The result is in canonical units and "
                "should be interpreted carefully by the consuming agent."
            ),
        )
        async def no_example(x: int) -> int:
            return x

        return m

    async def test_violation_emitted(self, lint_mcp: FastMCP) -> None:
        tools = await lint_mcp.list_tools()
        violations = check_tool_descriptions(tools)
        assert len(violations) == 1
        assert violations[0].rule == "missing_example"
        assert violations[0].tool_name == "no_example"


class TestTooShortFlagged:
    @pytest.fixture
    def lint_mcp(self) -> FastMCP:
        m = FastMCP("too-short-fixture")

        @m.tool(name="terse", description="Brief.")
        async def terse(x: int) -> int:
            return x

        return m

    async def test_both_too_short_and_missing_example_flagged(self, lint_mcp: FastMCP) -> None:
        tools = await lint_mcp.list_tools()
        violations = check_tool_descriptions(tools)
        # "Brief." is both under length and lacks an example marker.
        rules = {v.rule for v in violations}
        assert "too_short" in rules
        assert "missing_example" in rules


class TestCommonMistakeWarningFlagged:
    @pytest.fixture
    def lint_mcp(self) -> FastMCP:
        m = FastMCP("missing-mistake-fixture")

        @m.tool(
            name="time_thing",
            # Has example + length. Missing the canonical ISO 8601 warning
            # for the `epoch` input arg.
            description=(
                "Convert a timestamp to a different scale, for instance "
                "from UTC to TAI. Returns the converted timestamp."
            ),
        )
        async def time_thing(epoch: str, target_scale: str) -> str:
            return epoch

        return m

    async def test_missing_warning_flagged(self, lint_mcp: FastMCP) -> None:
        tools = await lint_mcp.list_tools()
        violations = check_tool_descriptions(tools)
        rules = {v.rule for v in violations}
        assert "missing_common_mistake_warning" in rules
        warning_violation = next(
            v for v in violations if v.rule == "missing_common_mistake_warning"
        )
        assert "epoch" in warning_violation.detail


class TestFrameArgWarning:
    @pytest.fixture
    def lint_mcp(self) -> FastMCP:
        m = FastMCP("frame-fixture")

        @m.tool(
            name="frame_thing",
            # Mentions a frame name ("TEME") so the warning rule is satisfied.
            description=(
                "Transform a state vector between frames, e.g. TEME → ICRF for "
                "a state at the given epoch (ISO 8601). Returns the rotated state."
            ),
        )
        async def frame_thing(epoch: str, from_frame: str, to_frame: str) -> str:
            return epoch

        return m

    async def test_frame_warning_satisfied(self, lint_mcp: FastMCP) -> None:
        tools = await lint_mcp.list_tools()
        assert check_tool_descriptions(tools) == []


class TestRegistry:
    def test_epoch_pattern_present(self) -> None:
        assert "epoch" in COMMON_MISTAKE_HINTS
        assert "ISO 8601" in COMMON_MISTAKE_HINTS["epoch"]

    def test_frame_pattern_lists_alternatives(self) -> None:
        # The frame warning is satisfied by any of TEME / ICRF / GCRS.
        assert "frame" in COMMON_MISTAKE_HINTS
        assert len(COMMON_MISTAKE_HINTS["frame"]) >= 2


class TestMalformedInputSchemaDefensivePaths:
    """The walker tolerates schemas FastMCP would never emit but JSON-Schema allows."""

    def test_non_dict_schema_treated_as_no_args(self) -> None:
        # JSON-Schema permits `True` (any) or `False` (none) as a schema body.
        # FastMCP always emits a dict, but the defensive branch shouldn't crash.
        from mcp.types import Tool

        from astrodynamics_mcp.server_lint import _input_arg_names

        tool = Tool.model_construct(
            name="weird",
            description="x" * 100 + " e.g. foo",
            inputSchema=True,
        )
        assert _input_arg_names(tool) == set()

    def test_non_dict_properties_treated_as_no_args(self) -> None:
        from mcp.types import Tool

        from astrodynamics_mcp.server_lint import _input_arg_names

        tool = Tool.model_construct(
            name="weird",
            description="x" * 100 + " e.g. foo",
            inputSchema={"type": "object", "properties": "not-a-dict"},
        )
        assert _input_arg_names(tool) == set()


class TestViolationModel:
    def test_violation_is_frozen(self) -> None:
        v = DescriptionViolation(tool_name="foo", rule="too_short", detail="...")
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            v.tool_name = "bar"
