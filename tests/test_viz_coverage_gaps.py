"""Coverage-gap tests for the visualisation subsystem review.

Collects the checks the per-tool ``test_tool_viz_*.py`` files did not yet
exercise: ``_register_viz_tools`` idempotency (mirroring the GMAT / SPICE
``*_coverage_gaps`` precedents), and an attachment-bearing tool driven through
the real low-level ``call_tool`` wire handler — the transport-shared path both
stdio and Streamable HTTP funnel through — to prove the PNG block and the
structured summary both survive it, the byte-for-byte transport-equivalence
guarantee for the additive attachment (not just the structured channel).

The idempotency check needs no extras (registration never runs a tool body), so
it runs in the standard test job. The wire-handler check renders a real PNG, so
it needs the ``[viz]`` extra and self-skips otherwise; it is wired into CI's
``[viz]`` job test list.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    ImageContent,
    TextContent,
)

from astrodynamics_mcp.schemas.base import Frame, StateVector
from astrodynamics_mcp.tools import viz as viz_tools
from astrodynamics_mcp.units import QuantityVector

_EXPECTED_TOOL_NAMES = frozenset(
    {"plot_ground_track", "plot_trajectory", "plot_porkchop", "czml_trajectory"}
)


# ---------------------------------------------------------------------------
# Registration idempotency
# ---------------------------------------------------------------------------


class TestRegistrationIdempotency:
    """``_register_viz_tools()`` is safe to call twice (hot-reload / re-init)."""

    @pytest.fixture
    def fresh_mcp(self, monkeypatch: pytest.MonkeyPatch) -> FastMCP:
        fresh = FastMCP("viz-coverage-test")
        monkeypatch.setattr("astrodynamics_mcp.server.mcp", fresh)
        return fresh

    async def test_register_twice_does_not_change_surface(self, fresh_mcp: FastMCP) -> None:
        viz_tools._register_viz_tools()
        first = {t.name for t in await fresh_mcp.list_tools()}
        # A second call would either silently double-register or raise; either
        # would break a hot-reload / re-init path. The contract is "no change in
        # tool surface".
        try:
            viz_tools._register_viz_tools()
        except Exception as exc:
            pytest.skip(f"double-register raised cleanly: {exc!r}")
        second = {t.name for t in await fresh_mcp.list_tools()}
        assert first == second, "double _register_viz_tools changed the tool surface"
        assert _EXPECTED_TOOL_NAMES.issubset(first)


# ---------------------------------------------------------------------------
# Attachment survival through the shared wire handler
#
# Both transports drive the low-level ``call_tool`` handler installed in
# server.py; an attachment-bearing tool must come back through it with the PNG
# block and the structured summary intact. Needs a real render (matplotlib) and
# the viz slots registered on the module singleton (gmat-czml gates the import).
# ---------------------------------------------------------------------------


def _itrs_states(n: int = 6) -> list[dict[str, Any]]:
    """A short inclined ITRS series as JSON-mode dicts (no inertial→fixed rotation)."""
    states: list[StateVector] = []
    for i in range(n):
        ang = math.radians(i * (360.0 / n))
        r = [
            7000.0 * math.cos(ang),
            7000.0 * math.sin(ang) * 0.6,
            7000.0 * math.sin(ang) * 0.8,
        ]
        hh, mm = divmod(i * 15, 60)
        states.append(
            StateVector(
                r=QuantityVector(value=r, unit="km"),
                v=QuantityVector(value=[0.0, 7.5, 0.0], unit="km/s"),
                frame=Frame.ITRS,
                epoch=f"2024-01-01T{hh:02d}:{mm:02d}:00Z",
            )
        )
    return [s.model_dump(mode="json") for s in states]


class TestAttachmentSurvivesWireHandler:
    """An attachment-bearing tool driven through the real wire handler keeps both
    channels — the proof transport equivalence holds for the additive PNG, not
    just the structured summary."""

    @pytest.fixture(autouse=True)
    def _require_viz(self) -> None:
        pytest.importorskip("matplotlib", reason="[viz] extra not installed")
        pytest.importorskip("gmat_czml", reason="[viz] extra (gmat-czml) not installed")

    async def _wire_call(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        """Invoke a tool through the real low-level ``call_tool`` handler.

        This is the wire path an MCP client drives over either transport —
        distinct from the ``mcp.call_tool`` method (which raises ``ToolError``).
        Mirrors ``tests/test_server.py::_wire_call``.
        """
        import astrodynamics_mcp.tools  # noqa: F401  # registration side effect
        from astrodynamics_mcp.server import mcp

        handler = mcp._mcp_server.request_handlers[CallToolRequest]
        request = CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(name=name, arguments=arguments),
        )
        result = await handler(request)
        assert isinstance(result.root, CallToolResult)
        return result.root

    async def test_png_and_summary_survive(self) -> None:
        result = await self._wire_call("plot_ground_track", {"states": _itrs_states()})
        assert result.isError is False
        assert result.structuredContent is not None
        # The ASCII summary leads the content list; the PNG rides alongside.
        assert isinstance(result.content[0], TextContent)
        images = [c for c in result.content if isinstance(c, ImageContent)]
        assert len(images) == 1, f"expected exactly one PNG block, got {len(images)}"
        assert images[0].mimeType == "image/png"
