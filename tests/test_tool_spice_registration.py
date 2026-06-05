"""Tests for the SPICE tool-slot conditional-registration mechanism.

The contract mirrors the GMAT gate (see test_tool_gmat_registration.py): every
``spice_*`` tool slot registers on the module-level ``mcp`` singleton iff
``spiceypy`` is importable. The test env does not install ``spiceypy`` (it ships
only with the ``[spice]`` extra), so the negative case is verified against the
real singleton and the positive case is verified against a fresh ``FastMCP``
driven by the registration helper directly.
"""

from __future__ import annotations

import importlib.util

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from astrodynamics_mcp import server as server_module
from astrodynamics_mcp.server import mcp as real_mcp
from astrodynamics_mcp.tools import spice as spice_tools

_EXPECTED_TOOL_NAMES = frozenset(
    {
        "spice_load_kernel",
        "spice_list_kernels",
        "spice_unload_kernel",
        "spice_state",
        "spice_frame_transform",
        "spice_body_parameters",
        "spice_time_convert",
    }
)

# Slots still landing in follow-up work. Tools with real bodies are excluded from
# the placeholder-body parametrize so this file stays accurate as each slot
# graduates — the kernel-management trio (load / list / unload) now has real
# bodies and is covered by tests/test_tool_spice_kernels.py instead.
_PLACEHOLDER_TOOL_NAMES = frozenset(
    {
        "spice_state",
        "spice_frame_transform",
        "spice_body_parameters",
        "spice_time_convert",
    }
)


@pytest.mark.skipif(
    importlib.util.find_spec("spiceypy") is not None,
    reason="negative case requires a bare environment (no [spice] extra installed)",
)
class TestNegativeCase:
    """Without ``spiceypy`` installed, the real surface has no SPICE slots."""

    async def test_module_guard_is_false_in_test_env(self) -> None:
        assert spice_tools._SPICEYPY_AVAILABLE is False

    async def test_real_surface_has_no_spice_tools(self) -> None:
        tools = await real_mcp.list_tools()
        names = {t.name for t in tools}
        leaked = names & _EXPECTED_TOOL_NAMES
        assert not leaked, f"SPICE slots leaked into the bare surface: {leaked}"


class TestPositiveCase:
    """When the guard is satisfied, all seven slots register identically."""

    @pytest.fixture(autouse=True)
    def _isolated_mcp(self, monkeypatch: pytest.MonkeyPatch) -> FastMCP:
        fresh = FastMCP("spice-registration-test")
        monkeypatch.setattr("astrodynamics_mcp.server.mcp", fresh)
        return fresh

    async def test_helper_registers_all_seven_slots(self, _isolated_mcp: FastMCP) -> None:
        spice_tools._register_spice_tools()
        tools = await _isolated_mcp.list_tools()
        names = {t.name for t in tools}
        assert _EXPECTED_TOOL_NAMES.issubset(names), (
            f"missing slots: {_EXPECTED_TOOL_NAMES - names}"
        )

    async def test_every_slot_carries_lint_metadata(self, _isolated_mcp: FastMCP) -> None:
        """Mirror the real-surface lint guarantees on the placeholder slots so a
        future ``[spice]``-installed CI run gets a clean ``list_tools`` payload."""
        from astrodynamics_mcp.server_lint import check_tool_descriptions

        spice_tools._register_spice_tools()
        tools = [t for t in await _isolated_mcp.list_tools() if t.name in _EXPECTED_TOOL_NAMES]
        assert len(tools) == len(_EXPECTED_TOOL_NAMES)
        for t in tools:
            assert t.annotations is not None
            assert t.outputSchema is not None
            props = (t.inputSchema or {}).get("properties", {})
            for param_name, schema in props.items():
                assert schema.get("description"), f"{t.name}.{param_name} missing description"
        assert check_tool_descriptions(tools) == []

    async def test_pool_mutating_slots_are_not_read_only(self, _isolated_mcp: FastMCP) -> None:
        """load / unload mutate the process-global kernel pool, so their
        annotations must not advertise read-only; the query slots are read-only."""
        spice_tools._register_spice_tools()
        by_name = {t.name: t for t in await _isolated_mcp.list_tools()}
        for mutating in ("spice_load_kernel", "spice_unload_kernel"):
            ann = by_name[mutating].annotations
            assert ann is not None and ann.readOnlyHint is False
        for query in ("spice_state", "spice_frame_transform", "spice_time_convert"):
            ann = by_name[query].annotations
            assert ann is not None and ann.readOnlyHint is True

    @pytest.mark.parametrize("tool_name", sorted(_PLACEHOLDER_TOOL_NAMES))
    async def test_placeholder_body_raises_not_implemented(
        self, _isolated_mcp: FastMCP, tool_name: str
    ) -> None:
        """Placeholder bodies must error loudly — never silently return None or an
        empty payload. ``register_tool`` wraps the ``NotImplementedError`` in an
        :class:`UpstreamError` with the canonical ``upstream.unexpected_exception``
        code (see ``tests/test_server.py::TestRegisterToolAsync``). The slots take
        no arguments, so an empty payload reaches the body."""
        spice_tools._register_spice_tools()
        with pytest.raises(ToolError) as excinfo:
            await server_module.mcp.call_tool(tool_name, {})
        raw = str(excinfo.value)
        assert "upstream.unexpected_exception" in raw
        assert "NotImplementedError" in raw
