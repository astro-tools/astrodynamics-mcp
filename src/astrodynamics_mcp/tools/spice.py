"""SPICE tool slots — registered only when ``spiceypy`` is importable.

The SPICE surface ships behind the optional ``[spice]`` extra. When
``spiceypy`` (the Python binding to NASA NAIF's CSPICE) is installed the seven
``spice_*`` tool slots register; on a bare install they are absent and the rest
of the tool surface is unaffected — the same gate the ``[gmat]`` tools use.

Three slots are implemented — the kernel-management trio
(``spice_load_kernel`` / ``spice_list_kernels`` / ``spice_unload_kernel``) that
furnishes, enumerates, and unloads kernels in the process-global pool. The
other four are ``NotImplementedError`` placeholders; each per-tool follow-up
replaces one slot the way these three and the GMAT slots graduated.

Per the locked SPICE integration contract (``docs/spice-integration.md``) the
slots register identically on stdio and Streamable HTTP — there is no
transport-specific gating; the kernel pool is process-global and the trust
boundary of an HTTP deployment is the operator's. Every CSPICE call is
serialised onto one dedicated worker thread (:mod:`astrodynamics_mcp.spice_runtime`)
and URL loads route through the NAIF allowlist + XDG cache
(:mod:`astrodynamics_mcp.spice_kernels`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlparse

from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from astrodynamics_mcp.errors import InvalidInputError
from astrodynamics_mcp.server import register_tool
from astrodynamics_mcp.spice_kernels import (
    KernelCache,
    default_kernel_cache,
    validate_kernel_url,
)
from astrodynamics_mcp.spice_runtime import (
    SPICE_KERNEL_CATEGORIES,
    furnish_and_describe,
    list_pool,
    normalize_kind_filter,
    run_on_spice_thread,
    unload_kernel,
)

try:
    import spiceypy  # noqa: F401  # availability probe; the symbol itself isn't used here

    _SPICEYPY_AVAILABLE = True
except ImportError:
    _SPICEYPY_AVAILABLE = False


# URL schemes that route a load through the NAIF allowlist + cache. Anything
# else (no scheme, a drive letter, a bare path) is taken as a local filesystem
# path and furnished directly.
_URL_SCHEMES = frozenset({"http", "https"})

# The category literals exposed on the wire — the CSPICE pool keywords, kept in
# lockstep with the runtime's authoritative tuple.
SpiceKernelCategory = Literal["SPK", "CK", "PCK", "EK", "DSK", "META", "TEXT"]


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class SpiceKernelInfo(BaseModel):
    """One kernel-pool entry, exactly as CSPICE reports it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(
        ...,
        description=(
            "The local path CSPICE knows this kernel by — the furnished filesystem "
            "path, or the on-disk cache path for a kernel loaded from a URL. This is "
            "the unload key: pass this exact string to spice_unload_kernel, never the "
            "original URL."
        ),
    )
    type: str = Field(
        ...,
        description=(
            "CSPICE kernel category: one of SPK, CK, PCK, EK, DSK, META (a "
            "meta-kernel), or TEXT. Leap-second (LSK), frame (FK), and "
            "spacecraft-clock (SCLK) kernels all report as TEXT — CSPICE does not "
            "distinguish them at this layer."
        ),
    )
    source: str = Field(
        ...,
        description=(
            "Provenance within the pool: the meta-kernel that furnished this kernel, "
            "or an empty string when it was furnished directly. CSPICE does not "
            "retain the original URL for a URL load, so this is not the download "
            "source."
        ),
    )
    handle: int = Field(
        ...,
        description=(
            "CSPICE file handle for binary kernels (SPK / CK / binary PCK / EK / "
            "DSK); 0 for text kernels, which load into the kernel pool rather than "
            "as DAF/DAS files. An opaque identifier, not a physical quantity — "
            "unitless."
        ),
    )


class SpiceLoadKernelResponse(BaseModel):
    """Result of furnishing a kernel source into the process pool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    loaded: list[SpiceKernelInfo] = Field(
        ...,
        description=(
            "Every kernel this call added to the pool. A plain kernel yields one "
            "entry; a meta-kernel yields the META entry plus every kernel it "
            "references, each with its own resolved type. Empty only if the source "
            "was already fully loaded."
        ),
    )
    from_cache: bool = Field(
        ...,
        description=(
            "Whether the source was served from the on-disk kernel cache with no "
            "network download. Always false for a local-path load; true for a URL "
            "whose bytes were already cached and fresh."
        ),
    )


class SpiceListKernelsResponse(BaseModel):
    """The kernels currently furnished in the process pool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kernels: list[SpiceKernelInfo] = Field(
        ...,
        description=(
            "One entry per kernel currently furnished in the process pool, after "
            "any `kind` filter. Shared by every client of an HTTP deployment — the "
            "pool is process-global."
        ),
    )


class SpiceUnloadKernelResponse(BaseModel):
    """Confirmation that a kernel was unloaded, plus the remaining pool size."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unloaded: str = Field(
        ...,
        description="The name of the kernel that was unloaded; echoes the `name` argument.",
    )
    remaining_count: int = Field(
        ...,
        description=(
            "Number of kernels still furnished in the pool after the unload. A "
            "cardinality, not a physical quantity — unitless."
        ),
    )


# ---------------------------------------------------------------------------
# Tool-body implementations (module-level for direct testability; the
# registered slots below are thin wrappers, mirroring the GMAT layout).
# ---------------------------------------------------------------------------


def _looks_like_url(source: str) -> bool:
    """Whether *source* should route through the NAIF allowlist + cache."""
    return urlparse(source).scheme in _URL_SCHEMES


def _resolve_local_kernel(source: str) -> str:
    """Confirm *source* is a readable local kernel file; return it unchanged.

    CSPICE furnishes the path as given, so an existence/readability check here
    turns a missing or non-file path into a typed input error instead of a
    CSPICE abort.
    """
    if not Path(source).is_file():
        raise InvalidInputError(
            f"no readable kernel file at {source!r}; pass a local filesystem path to a "
            "kernel, or an https NAIF URL",
            code="invalid_input.spice_kernel_not_found",
            data={"source": source},
        )
    return source


async def _do_load_kernel(
    source: str, *, cache: KernelCache | None = None
) -> SpiceLoadKernelResponse:
    """Resolve *source* (URL → allowlist + cache, else local path) and furnish it."""
    if _looks_like_url(source):
        validate_kernel_url(source)
        kernel_cache = cache if cache is not None else default_kernel_cache()
        from_cache = kernel_cache.is_cached(source)
        local_path = await kernel_cache.fetch(source)
        furnish_target = str(local_path)
    else:
        furnish_target = _resolve_local_kernel(source)
        from_cache = False

    rows = await run_on_spice_thread(furnish_and_describe, furnish_target)
    return SpiceLoadKernelResponse(
        loaded=[
            SpiceKernelInfo(name=r.name, type=r.type, source=r.source, handle=r.handle)
            for r in rows
        ],
        from_cache=from_cache,
    )


async def _do_list_kernels(kind: list[SpiceKernelCategory] | None) -> SpiceListKernelsResponse:
    """Enumerate the pool, optionally filtered to the given CSPICE categories."""
    category = normalize_kind_filter(list(kind) if kind is not None else None)
    rows = await run_on_spice_thread(list_pool, category)
    return SpiceListKernelsResponse(
        kernels=[
            SpiceKernelInfo(name=r.name, type=r.type, source=r.source, handle=r.handle)
            for r in rows
        ],
    )


async def _do_unload_kernel(name: str) -> SpiceUnloadKernelResponse:
    """Unload the kernel named *name* and report the remaining pool size."""
    remaining = await run_on_spice_thread(unload_kernel, name)
    return SpiceUnloadKernelResponse(unloaded=name, remaining_count=remaining)


# ---------------------------------------------------------------------------
# Placeholder slots — reserved output shape; bodies raise loudly
# ---------------------------------------------------------------------------


class SpicePlaceholderResult(BaseModel):
    """Reserved output shape for the not-yet-implemented SPICE tool slots.

    Each placeholder slot declares this return type only so the server can
    derive an output schema for it; the bodies raise ``NotImplementedError`` and
    never construct one. Per-tool follow-up work replaces this with the tool's
    real response model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    detail: str = Field(
        ...,
        description="Placeholder marker; never populated at runtime.",
    )


# ---------------------------------------------------------------------------
# Tool descriptions
# ---------------------------------------------------------------------------

_LOAD_KERNEL_DESCRIPTION = (
    "Furnish a SPICE kernel into the process kernel pool from a local path or a NAIF "
    "https URL, so later spice_* queries can read it, e.g. "
    "spice_load_kernel('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls') "
    "furnishes a generic leap-second kernel. The pool is additive and persists across "
    "calls — load a leap-second kernel (LSK) before any time conversion, and a planetary "
    "SPK before a state query; both stay loaded together. A meta-kernel (.tm) furnishes "
    "everything it lists in one call, so `loaded` may contain several kernels of several "
    "types. URL sources must be on the NAIF allowlist (naif.jpl.nasa.gov, https only) — "
    "host your own mirror behind a local path otherwise; a repeat URL load is served from "
    "the on-disk cache (from_cache=true). Returns each furnished kernel's resolved name, "
    "type, and handle. Keep the returned `name` — it is what you pass to "
    "spice_unload_kernel."
)

_LIST_KERNELS_DESCRIPTION = (
    "List the SPICE kernels currently furnished in the process kernel pool, e.g. "
    "spice_list_kernels() to confirm a leap-second kernel and an SPK are both loaded "
    "before a state query, or spice_list_kernels(kind=['SPK','PCK']) to see only the "
    "ephemeris and planetary-constants kernels. Each row carries the kernel's name, type "
    "(SPK / CK / PCK / EK / DSK / META / TEXT), provenance, and handle. The pool is "
    "process-global, so on an HTTP deployment this reports every caller's kernels, not "
    "just yours."
)

_UNLOAD_KERNEL_DESCRIPTION = (
    "Unload a previously furnished SPICE kernel from the process kernel pool, e.g. "
    "spice_unload_kernel('/path/to/de440.bsp') to drop a stale ephemeris before "
    "furnishing a newer one. Unload by the `name` returned from spice_load_kernel (or "
    "shown by spice_list_kernels), not the original URL — a name that is not loaded is a "
    "typed error rather than a silent no-op. Returns the remaining pool count."
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

    Annotations are honest about each slot's semantics: the two pool-mutating
    tools (load / unload) are not read-only, and only ``spice_load_kernel``
    reaches the network (the NAIF furnish-from-URL path); the query tools read
    the in-process pool and touch nothing outside it.
    """

    @register_tool(
        name="spice_load_kernel",
        description=_LOAD_KERNEL_DESCRIPTION,
        annotations=ToolAnnotations(
            title="SPICE Load Kernel", readOnlyHint=False, openWorldHint=True
        ),
    )
    async def spice_load_kernel(
        source: Annotated[
            str,
            Field(
                description=(
                    "A local filesystem path to a kernel, or an https NAIF URL "
                    "(naif.jpl.nasa.gov). A meta-kernel path furnishes every kernel it "
                    "lists. e.g. '/data/de440.bsp' or "
                    "'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls'."
                ),
            ),
        ],
    ) -> SpiceLoadKernelResponse:
        return await _do_load_kernel(source)

    @register_tool(
        name="spice_list_kernels",
        description=_LIST_KERNELS_DESCRIPTION,
        annotations=ToolAnnotations(
            title="SPICE List Kernels", readOnlyHint=True, openWorldHint=False
        ),
    )
    async def spice_list_kernels(
        kind: Annotated[
            list[SpiceKernelCategory] | None,
            Field(
                description=(
                    "Optional category filter — list only kernels of these CSPICE types. "
                    "Omit to list every loaded kernel. e.g. ['SPK'] for ephemerides, or "
                    "['SPK','PCK'] for both. Valid categories: "
                    f"{list(SPICE_KERNEL_CATEGORIES)}."
                ),
            ),
        ] = None,
    ) -> SpiceListKernelsResponse:
        return await _do_list_kernels(kind)

    @register_tool(
        name="spice_unload_kernel",
        description=_UNLOAD_KERNEL_DESCRIPTION,
        annotations=ToolAnnotations(
            title="SPICE Unload Kernel", readOnlyHint=False, openWorldHint=False
        ),
    )
    async def spice_unload_kernel(
        name: Annotated[
            str,
            Field(
                description=(
                    "The name of the kernel to unload — the `name` returned by "
                    "spice_load_kernel or shown by spice_list_kernels, not the original "
                    "URL. e.g. '/data/de440.bsp'. A name that is not loaded returns a "
                    "typed error."
                ),
            ),
        ],
    ) -> SpiceUnloadKernelResponse:
        return await _do_unload_kernel(name)

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
