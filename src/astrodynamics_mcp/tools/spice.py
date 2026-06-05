"""SPICE tool slots — registered only when ``spiceypy`` is importable.

The SPICE surface ships behind the optional ``[spice]`` extra. When
``spiceypy`` (the Python binding to NASA NAIF's CSPICE) is installed the seven
``spice_*`` tool slots register; on a bare install they are absent and the rest
of the tool surface is unaffected — the same gate the ``[gmat]`` tools use.

This module owns the conditional-registration mechanism and the reserved slot
names. The bodies are ``NotImplementedError`` placeholders for now: each
per-tool follow-up replaces one slot with its real signature, response model,
and body, the way the GMAT slots graduated one at a time.

Per the locked SPICE integration contract (``docs/spice-integration.md``) the
slots register identically on stdio and Streamable HTTP — there is no
transport-specific gating; the kernel pool is process-global and the trust
boundary of an HTTP deployment is the operator's. The concurrency contract
(every CSPICE call serialised onto one dedicated worker thread) and the
furnish-from-URL policy (:mod:`astrodynamics_mcp.spice_kernels`) are the wiring
the tool bodies build on as they land.
"""

from __future__ import annotations

from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from astrodynamics_mcp.server import register_tool

try:
    import spiceypy  # noqa: F401  # availability probe; the symbol itself isn't used yet

    _SPICEYPY_AVAILABLE = True
except ImportError:
    _SPICEYPY_AVAILABLE = False


class SpicePlaceholderResult(BaseModel):
    """Reserved output shape for the not-yet-implemented SPICE tool slots.

    Each slot declares this return type only so the server can derive an output
    schema for it; the bodies raise ``NotImplementedError`` and never construct
    one. Per-tool follow-up work replaces this with the tool's real response
    model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    detail: str = Field(
        ...,
        description="Placeholder marker; never populated at runtime.",
    )


_LOAD_KERNEL_DESCRIPTION = (
    "Furnish a SPICE kernel into the process kernel pool from a local path or a "
    "NAIF https URL, e.g. a planetary SPK ephemeris or a leap-second kernel, so "
    "later spice_* queries can read it. Reserved slot — not yet implemented; the "
    "kernel-management tool lands in follow-up work."
)

_LIST_KERNELS_DESCRIPTION = (
    "List the SPICE kernels currently furnished in the process kernel pool, e.g. "
    "to confirm an SPK and a leap-second kernel are loaded before a state query. "
    "Reserved slot — not yet implemented; lands in follow-up work."
)

_UNLOAD_KERNEL_DESCRIPTION = (
    "Unload a previously furnished SPICE kernel from the process kernel pool, "
    "e.g. to drop a stale ephemeris before furnishing a newer one. Reserved slot "
    "— not yet implemented; lands in follow-up work."
)

_STATE_DESCRIPTION = (
    "Query the state — position and velocity — of one body relative to another "
    "at an epoch from furnished SPK kernels, e.g. Mars relative to the Solar "
    "System barycentre. Reserved slot — not yet implemented; lands in follow-up "
    "work."
)

_FRAME_TRANSFORM_DESCRIPTION = (
    "Rotate a vector between SPICE reference frames defined by furnished FK / PCK "
    "kernels at an epoch, e.g. into a non-Earth body-fixed frame the astropy "
    "frame_transform tool cannot provide. Reserved slot — not yet implemented; "
    "lands in follow-up work."
)

_BODY_PARAMETERS_DESCRIPTION = (
    "Read physical and orientation constants for a body from furnished PCK "
    "kernels — radii, GM, orientation models — e.g. the triaxial radii of Mars. "
    "Reserved slot — not yet implemented; lands in follow-up work."
)

_TIME_CONVERT_DESCRIPTION = (
    "Convert between SPICE kernel-defined time systems — ephemeris time (ET / "
    "TDB), UTC, and spacecraft clock (SCLK) — using a furnished leap-second "
    "kernel, e.g. an ET seconds-past-J2000 value to a UTC calendar string. "
    "Reserved slot — not yet implemented; lands in follow-up work."
)


def _register_spice_tools() -> None:
    """Attach the seven SPICE tool slots to ``astrodynamics_mcp.server.mcp``.

    Factored out of module top-level — like :func:`_register_gmat_tools` — so
    unit tests can drive registration against a fresh
    :class:`~mcp.server.fastmcp.FastMCP` instance without relying on the
    import-time guard being satisfied.

    Annotations are honest about each slot's eventual semantics: the two
    pool-mutating tools (load / unload) are not read-only, and only
    ``spice_load_kernel`` reaches the network (the NAIF furnish-from-URL path);
    the five query tools read the in-process pool and touch nothing outside it.
    """

    @register_tool(
        name="spice_load_kernel",
        description=_LOAD_KERNEL_DESCRIPTION,
        annotations=ToolAnnotations(
            title="SPICE Load Kernel", readOnlyHint=False, openWorldHint=True
        ),
    )
    async def spice_load_kernel() -> SpicePlaceholderResult:
        raise NotImplementedError("spice_load_kernel lands in follow-up work")

    @register_tool(
        name="spice_list_kernels",
        description=_LIST_KERNELS_DESCRIPTION,
        annotations=ToolAnnotations(
            title="SPICE List Kernels", readOnlyHint=True, openWorldHint=False
        ),
    )
    async def spice_list_kernels() -> SpicePlaceholderResult:
        raise NotImplementedError("spice_list_kernels lands in follow-up work")

    @register_tool(
        name="spice_unload_kernel",
        description=_UNLOAD_KERNEL_DESCRIPTION,
        annotations=ToolAnnotations(
            title="SPICE Unload Kernel", readOnlyHint=False, openWorldHint=False
        ),
    )
    async def spice_unload_kernel() -> SpicePlaceholderResult:
        raise NotImplementedError("spice_unload_kernel lands in follow-up work")

    @register_tool(
        name="spice_state",
        description=_STATE_DESCRIPTION,
        annotations=ToolAnnotations(title="SPICE State", readOnlyHint=True, openWorldHint=False),
    )
    async def spice_state() -> SpicePlaceholderResult:
        raise NotImplementedError("spice_state lands in follow-up work")

    @register_tool(
        name="spice_frame_transform",
        description=_FRAME_TRANSFORM_DESCRIPTION,
        annotations=ToolAnnotations(
            title="SPICE Frame Transform", readOnlyHint=True, openWorldHint=False
        ),
    )
    async def spice_frame_transform() -> SpicePlaceholderResult:
        raise NotImplementedError("spice_frame_transform lands in follow-up work")

    @register_tool(
        name="spice_body_parameters",
        description=_BODY_PARAMETERS_DESCRIPTION,
        annotations=ToolAnnotations(
            title="SPICE Body Parameters", readOnlyHint=True, openWorldHint=False
        ),
    )
    async def spice_body_parameters() -> SpicePlaceholderResult:
        raise NotImplementedError("spice_body_parameters lands in follow-up work")

    @register_tool(
        name="spice_time_convert",
        description=_TIME_CONVERT_DESCRIPTION,
        annotations=ToolAnnotations(
            title="SPICE Time Convert", readOnlyHint=True, openWorldHint=False
        ),
    )
    async def spice_time_convert() -> SpicePlaceholderResult:
        raise NotImplementedError("spice_time_convert lands in follow-up work")


if _SPICEYPY_AVAILABLE:
    _register_spice_tools()
