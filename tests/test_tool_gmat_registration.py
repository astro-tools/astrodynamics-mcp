"""Tests for the GMAT tool-slot conditional-registration mechanism.

The contract from issue #70 is: every ``gmat_*`` tool slot registers on the
module-level ``mcp`` singleton iff ``gmat-run`` is importable. The test env
does not install ``gmat-run`` (it ships only with the ``[gmat]`` extra), so
the negative case is verified against the real singleton and the positive
case is verified against a fresh ``FastMCP`` driven by the registration
helper directly.
"""

from __future__ import annotations

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from astrodynamics_mcp import server as server_module
from astrodynamics_mcp.server import mcp as real_mcp
from astrodynamics_mcp.tools import gmat as gmat_tools

_EXPECTED_TOOL_NAMES = frozenset(
    {
        "gmat_run_mission",
        "gmat_sweep",
        "gmat_execute_script",
        "gmat_validate_script",
    }
)


class TestNegativeCase:
    """Without ``gmat-run`` installed, the real surface has no GMAT slots."""

    async def test_module_guard_is_false_in_test_env(self) -> None:
        # The test environment is the bare dev install — `gmat-run` lives in
        # the `[gmat]` extra and isn't pulled by `uv sync --all-groups`. If
        # this assertion ever flips, the rest of this file is testing the
        # wrong case and needs updating alongside CI.
        assert gmat_tools._GMAT_RUN_AVAILABLE is False

    async def test_real_surface_has_no_gmat_tools(self) -> None:
        tools = await real_mcp.list_tools()
        names = {t.name for t in tools}
        leaked = names & _EXPECTED_TOOL_NAMES
        assert not leaked, f"GMAT slots leaked into the bare surface: {leaked}"


class TestPositiveCase:
    """When the guard is satisfied, all four slots register identically."""

    @pytest.fixture(autouse=True)
    def _isolated_mcp(self, monkeypatch: pytest.MonkeyPatch) -> FastMCP:
        fresh = FastMCP("gmat-registration-test")
        monkeypatch.setattr("astrodynamics_mcp.server.mcp", fresh)
        return fresh

    async def test_helper_registers_all_four_slots(self, _isolated_mcp: FastMCP) -> None:
        gmat_tools._register_gmat_tools()
        tools = await _isolated_mcp.list_tools()
        names = {t.name for t in tools}
        assert _EXPECTED_TOOL_NAMES.issubset(names), (
            f"missing slots: {_EXPECTED_TOOL_NAMES - names}"
        )

    async def test_every_slot_carries_lint_metadata(self, _isolated_mcp: FastMCP) -> None:
        """Mirror the real-surface lint guarantees on the placeholder slots so a
        future ``[gmat]``-installed CI run gets a clean ``list_tools`` payload."""
        from astrodynamics_mcp.server_lint import check_tool_descriptions

        gmat_tools._register_gmat_tools()
        tools = [t for t in await _isolated_mcp.list_tools() if t.name in _EXPECTED_TOOL_NAMES]
        assert len(tools) == len(_EXPECTED_TOOL_NAMES)
        for t in tools:
            assert t.annotations is not None and t.annotations.readOnlyHint is True
            assert t.outputSchema is not None
            props = (t.inputSchema or {}).get("properties", {})
            for param_name, schema in props.items():
                assert schema.get("description"), f"{t.name}.{param_name} missing description"
        assert check_tool_descriptions(tools) == []

    @pytest.mark.parametrize("tool_name", sorted(_EXPECTED_TOOL_NAMES))
    async def test_placeholder_body_raises_not_implemented(
        self, _isolated_mcp: FastMCP, tool_name: str
    ) -> None:
        """Placeholder bodies must error loudly — never silently return None or
        an empty payload. ``register_tool`` wraps the ``NotImplementedError`` in
        an :class:`UpstreamError` with the canonical
        ``upstream.unexpected_exception`` code (see
        ``tests/test_server.py::TestRegisterToolAsync``)."""
        gmat_tools._register_gmat_tools()
        with pytest.raises(ToolError) as excinfo:
            await server_module.mcp.call_tool(tool_name, {"script": "noop"})
        # The wrapped envelope is JSON inside "Error executing tool <name>: …".
        raw = str(excinfo.value)
        assert "upstream.unexpected_exception" in raw
        assert "NotImplementedError" in raw
