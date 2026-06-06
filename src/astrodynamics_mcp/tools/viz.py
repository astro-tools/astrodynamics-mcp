"""Visualisation tool slots — registered only when the ``[viz]`` extra is present.

The visualisation surface ships behind the optional ``[viz]`` extra (matplotlib
for the static raster plots, gmat-czml for the CZML 3D view). When both are
importable the four ``plot_* / czml_*`` tool slots register; on a bare install
they are absent and the rest of the tool surface is unaffected — the same gate
the ``[gmat]`` and ``[spice]`` tools use.

This module owns the conditional-registration mechanism and the reserved slot
names. The bodies are ``NotImplementedError`` placeholders for now: the
static-plot follow-up replaces ``plot_ground_track`` / ``plot_trajectory`` /
``plot_porkchop`` with their real signatures, response models, and PNG output,
and the CZML follow-up replaces ``czml_trajectory`` — the way the GMAT and SPICE
slots graduated one at a time.

The attachment plumbing the bodies build on lands alongside this gate:
:mod:`astrodynamics_mcp.attachments` assembles the structured-summary-plus-PNG /
CZML result, and :mod:`astrodynamics_mcp.viz_render` renders matplotlib figures
to deterministic PNG bytes so the byte-for-byte transport-equivalence contract
holds across stdio and Streamable HTTP.
"""

from __future__ import annotations

from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from astrodynamics_mcp.server import register_tool

try:
    import gmat_czml  # noqa: F401  # availability probe; the symbol itself isn't used yet
    import matplotlib  # noqa: F401  # availability probe; the symbol itself isn't used yet

    _VIZ_AVAILABLE = True
except ImportError:
    _VIZ_AVAILABLE = False


class VizPlaceholderResult(BaseModel):
    """Reserved output shape for the not-yet-implemented visualisation slots.

    Each slot declares this return type only so the server can derive an output
    schema for it; the bodies raise ``NotImplementedError`` and never construct
    one. The static-plot and CZML follow-up work replaces this with each tool's
    real response model — a structured summary carried alongside the PNG / CZML
    attachment via :mod:`astrodynamics_mcp.attachments`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    detail: str = Field(
        ...,
        description="Placeholder marker; never populated at runtime.",
    )


_PLOT_GROUND_TRACK_DESCRIPTION = (
    "Render a satellite's sub-satellite ground track as a PNG over a world map, "
    "e.g. a 24-hour ISS track from an SGP4 propagation series, with an inline "
    "summary (revolutions, lat / lon extent) carried alongside the image. "
    "Reserved slot — not yet implemented; lands in follow-up work."
)

_PLOT_TRAJECTORY_DESCRIPTION = (
    "Render an orbit or transfer trajectory as a 2D or 3D PNG about a central "
    "body, e.g. a Hohmann transfer arc from a state series, with an inline "
    "summary (arc length, apsides) carried alongside the image. Reserved slot "
    "— not yet implemented; lands in follow-up work."
)

_PLOT_PORKCHOP_DESCRIPTION = (
    "Render a porkchop C3 / delta-v contour plot as a PNG from an existing "
    "porkchop grid result, e.g. an Earth-Mars 2026 launch window, reusing the "
    "computed grid with no recompute and keeping the inline grid summary. "
    "Reserved slot — not yet implemented; lands in follow-up work."
)

_CZML_TRAJECTORY_DESCRIPTION = (
    "Emit a trajectory as a CZML document for a Cesium 3D client, e.g. an SGP4 "
    "propagation series rendered as an animated orbit, returned as an embedded "
    "resource with an inline summary (packet count, time span). Reserved slot "
    "— not yet implemented; lands in follow-up work."
)


def _register_viz_tools() -> None:
    """Attach the four visualisation tool slots to ``astrodynamics_mcp.server.mcp``.

    Factored out of module top-level — like :func:`_register_spice_tools` and
    :func:`_register_gmat_tools` — so unit tests can drive registration against
    a fresh :class:`~mcp.server.fastmcp.FastMCP` instance without relying on the
    import-time guard being satisfied.

    Every slot is read-only and touches nothing outside the process: a plot
    consumes the state / grid series it is handed and renders locally, so none
    of them reach the network.
    """

    @register_tool(
        name="plot_ground_track",
        description=_PLOT_GROUND_TRACK_DESCRIPTION,
        annotations=ToolAnnotations(
            title="Plot Ground Track", readOnlyHint=True, openWorldHint=False
        ),
    )
    async def plot_ground_track() -> VizPlaceholderResult:
        raise NotImplementedError("plot_ground_track lands in follow-up work")

    @register_tool(
        name="plot_trajectory",
        description=_PLOT_TRAJECTORY_DESCRIPTION,
        annotations=ToolAnnotations(
            title="Plot Trajectory", readOnlyHint=True, openWorldHint=False
        ),
    )
    async def plot_trajectory() -> VizPlaceholderResult:
        raise NotImplementedError("plot_trajectory lands in follow-up work")

    @register_tool(
        name="plot_porkchop",
        description=_PLOT_PORKCHOP_DESCRIPTION,
        annotations=ToolAnnotations(title="Plot Porkchop", readOnlyHint=True, openWorldHint=False),
    )
    async def plot_porkchop() -> VizPlaceholderResult:
        raise NotImplementedError("plot_porkchop lands in follow-up work")

    @register_tool(
        name="czml_trajectory",
        description=_CZML_TRAJECTORY_DESCRIPTION,
        annotations=ToolAnnotations(
            title="CZML Trajectory", readOnlyHint=True, openWorldHint=False
        ),
    )
    async def czml_trajectory() -> VizPlaceholderResult:
        raise NotImplementedError("czml_trajectory lands in follow-up work")


if _VIZ_AVAILABLE:
    _register_viz_tools()
