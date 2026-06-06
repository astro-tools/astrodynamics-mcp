"""Tests for the visualisation tool-slot conditional-registration mechanism.

The contract mirrors the GMAT and SPICE gates (see
test_tool_gmat_registration.py / test_tool_spice_registration.py): every
``plot_* / czml_*`` slot registers on the module-level ``mcp`` singleton iff the
``[viz]`` extra is importable (matplotlib *and* gmat-czml). The test env
installs neither (they ship only with the extra), so the negative case is
verified against the real singleton and the positive case is verified against a
fresh ``FastMCP`` driven by the registration helper directly.
"""

from __future__ import annotations

import importlib.util

import pytest
from mcp.server.fastmcp import FastMCP

from astrodynamics_mcp.server import mcp as real_mcp
from astrodynamics_mcp.tools import viz as viz_tools

_EXPECTED_TOOL_NAMES = frozenset(
    {
        "plot_ground_track",
        "plot_trajectory",
        "plot_porkchop",
        "czml_trajectory",
    }
)

_VIZ_FULLY_INSTALLED = (
    importlib.util.find_spec("matplotlib") is not None
    and importlib.util.find_spec("gmat_czml") is not None
)

# The slots are NotImplementedError placeholders for now: the static-plot
# follow-up replaces plot_ground_track / plot_trajectory / plot_porkchop with
# real bodies, and the CZML follow-up replaces czml_trajectory. The generic
# register_tool exception-wrapping contract is covered by
# tests/test_server.py::TestRegisterToolAsync, so this file does not re-assert
# the NotImplementedError path.


@pytest.mark.skipif(
    _VIZ_FULLY_INSTALLED,
    reason="negative case requires a bare environment (no [viz] extra installed)",
)
class TestNegativeCase:
    """Without the [viz] extra, the real surface has no visualisation slots."""

    async def test_module_guard_is_false_in_test_env(self) -> None:
        assert viz_tools._VIZ_AVAILABLE is False

    async def test_real_surface_has_no_viz_tools(self) -> None:
        tools = await real_mcp.list_tools()
        names = {t.name for t in tools}
        leaked = names & _EXPECTED_TOOL_NAMES
        assert not leaked, f"viz slots leaked into the bare surface: {leaked}"


class TestPositiveCase:
    """When the guard is satisfied, all four slots register identically."""

    @pytest.fixture(autouse=True)
    def _isolated_mcp(self, monkeypatch: pytest.MonkeyPatch) -> FastMCP:
        fresh = FastMCP("viz-registration-test")
        monkeypatch.setattr("astrodynamics_mcp.server.mcp", fresh)
        return fresh

    async def test_helper_registers_all_four_slots(self, _isolated_mcp: FastMCP) -> None:
        viz_tools._register_viz_tools()
        tools = await _isolated_mcp.list_tools()
        names = {t.name for t in tools}
        assert _EXPECTED_TOOL_NAMES.issubset(names), (
            f"missing slots: {_EXPECTED_TOOL_NAMES - names}"
        )

    async def test_every_slot_carries_lint_metadata(self, _isolated_mcp: FastMCP) -> None:
        """Mirror the real-surface lint guarantees on every slot so a future
        ``[viz]``-installed CI run gets a clean ``list_tools`` payload."""
        from astrodynamics_mcp.server_lint import check_tool_descriptions

        viz_tools._register_viz_tools()
        tools = [t for t in await _isolated_mcp.list_tools() if t.name in _EXPECTED_TOOL_NAMES]
        assert len(tools) == len(_EXPECTED_TOOL_NAMES)
        for t in tools:
            assert t.annotations is not None
            assert t.outputSchema is not None
            props = (t.inputSchema or {}).get("properties", {})
            for param_name, schema in props.items():
                assert schema.get("description"), f"{t.name}.{param_name} missing description"
        assert check_tool_descriptions(tools) == []

    async def test_all_slots_are_read_only(self, _isolated_mcp: FastMCP) -> None:
        """A plot consumes the series it is handed and renders locally — every
        slot is read-only and reaches nothing outside the process."""
        viz_tools._register_viz_tools()
        by_name = {t.name: t for t in await _isolated_mcp.list_tools()}
        for name in _EXPECTED_TOOL_NAMES:
            ann = by_name[name].annotations
            assert ann is not None
            assert ann.readOnlyHint is True
            assert ann.openWorldHint is False
