"""SPICE tool slots — registered only when ``spiceypy`` is importable.

The SPICE surface ships behind the optional ``[spice]`` extra. When
``spiceypy`` (the Python binding to NASA NAIF's CSPICE) is installed the seven
``spice_*`` tool slots register; on a bare install they are absent and the rest
of the tool surface is unaffected — the same gate the ``[gmat]`` tools use.

Four slots are implemented — the kernel-management trio
(``spice_load_kernel`` / ``spice_list_kernels`` / ``spice_unload_kernel``) that
furnishes, enumerates, and unloads kernels in the process-global pool, plus
``spice_state`` (SPK state / ephemeris queries). The other three are
``NotImplementedError`` placeholders; each per-tool follow-up replaces one slot
the way these and the GMAT slots graduated.

Per the locked SPICE integration contract (``docs/spice-integration.md``) the
slots register identically on stdio and Streamable HTTP — there is no
transport-specific gating; the kernel pool is process-global and the trust
boundary of an HTTP deployment is the operator's. Every CSPICE call is
serialised onto one dedicated worker thread (:mod:`astrodynamics_mcp.spice_runtime`)
and URL loads route through the NAIF allowlist + XDG cache
(:mod:`astrodynamics_mcp.spice_kernels`).
"""

from __future__ import annotations

import os
from datetime import timezone
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlparse

from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, field_validator

from astrodynamics_mcp.errors import InvalidInputError
from astrodynamics_mcp.schemas.base import Epoch, _epoch_to_instant
from astrodynamics_mcp.server import register_tool
from astrodynamics_mcp.spice_kernels import (
    KernelCache,
    default_kernel_cache,
    validate_kernel_url,
)
from astrodynamics_mcp.spice_runtime import (
    SPICE_ABERRATION_CORRECTIONS,
    SPICE_KERNEL_CATEGORIES,
    furnish_and_describe,
    list_pool,
    normalize_aberration,
    normalize_kind_filter,
    query_frame_transform,
    query_state,
    run_on_spice_thread,
    unload_kernel,
)
from astrodynamics_mcp.units import Quantity, QuantityVector

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


class SpiceStateAtEpoch(BaseModel):
    """A target's state relative to an observer at one epoch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    epoch: str = Field(
        ...,
        description=(
            "The UTC ISO 8601 epoch this state is for, echoed verbatim from the "
            "requested `epochs` so each entry is self-describing regardless of order."
        ),
    )
    position: QuantityVector = Field(
        ...,
        description=(
            "Cartesian position [x, y, z] of the target relative to the observer, "
            "in the requested frame (km)."
        ),
    )
    velocity: QuantityVector = Field(
        ...,
        description=(
            "Cartesian velocity [vx, vy, vz] of the target relative to the observer, "
            "in the requested frame (km/s)."
        ),
    )
    light_time: Quantity | None = Field(
        None,
        description=(
            "One-way light time between target and observer (s). Present only when "
            "the aberration correction is not 'NONE'; a geometric ('NONE') query "
            "returns null here because no light-time correction was requested."
        ),
    )


class SpiceStateResponse(BaseModel):
    """States of a target relative to an observer at one or more epochs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: str = Field(
        ...,
        description="The target body, echoed from the request (name or NAIF ID as supplied).",
    )
    observer: str = Field(
        ...,
        description="The observer body, echoed from the request (name or NAIF ID as supplied).",
    )
    frame: str = Field(
        ...,
        description="The reference frame the states are expressed in, echoed from the request.",
    )
    aberration: str = Field(
        ...,
        description=(
            "The aberration correction applied, upper-cased and echoed from the request "
            "(e.g. 'NONE', 'LT', 'LT+S')."
        ),
    )
    states: list[SpiceStateAtEpoch] = Field(
        ...,
        description=("One state per requested epoch, in the same order as the `epochs` input."),
    )


# Length / velocity unit sets a rotatable vector may carry on the wire. A frame
# rotation is unit-agnostic, but the {value, unit} discipline still requires a
# declared unit; these mirror the schemas.base StateVector conventions so the
# rotated output can echo the input unit.
_LENGTH_UNITS = frozenset({"km", "m", "AU"})
_VELOCITY_UNITS = frozenset({"km/s", "m/s"})


class RotatableState(BaseModel):
    """A 3- or 6-vector to rotate between SPICE frames.

    ``position`` alone is a 3-vector, rotated by ``pxform``; adding ``velocity``
    makes it a 6-vector state, rotated by ``sxform`` (which carries the target
    frame's rotation rate into the rotated velocity). Omit the whole object on
    the tool call to request the rotation matrix alone — and to rotate any
    vector that is not a position (a pointing direction, an angular-momentum
    vector), request the matrix and apply it yourself.
    """

    model_config = ConfigDict(extra="forbid")

    position: QuantityVector = Field(
        ...,
        description=(
            "Cartesian position [x, y, z] in the source frame, length unit "
            "(km / m / AU). Rotated into `to_frame`. e.g. "
            "{value: [4000, 5000, 6000], unit: 'km'}."
        ),
        examples=[{"value": [4000.0, 5000.0, 6000.0], "unit": "km"}],
    )
    velocity: QuantityVector | None = Field(
        default=None,
        description=(
            "Optional Cartesian velocity [vx, vy, vz] in the source frame, velocity "
            "unit (km/s / m/s). When present the rotation uses the full state "
            "transform (sxform), so the rotated velocity includes the target frame's "
            "rotation rate; omit it to rotate position only (pxform). e.g. "
            "{value: [-1.0, 2.0, 0.5], unit: 'km/s'}."
        ),
        examples=[{"value": [-1.0, 2.0, 0.5], "unit": "km/s"}],
    )

    @field_validator("position")
    @classmethod
    def _position_unit(cls, v: QuantityVector) -> QuantityVector:
        if v.unit not in _LENGTH_UNITS:
            raise InvalidInputError(
                f"position unit must be a length ({sorted(_LENGTH_UNITS)}), got {v.unit!r}",
                code="invalid_input.wrong_unit_category",
            )
        if len(v.value) != 3:
            raise InvalidInputError(
                f"position must have exactly 3 components, got {len(v.value)}",
                code="invalid_input.wrong_vector_length",
            )
        return v

    @field_validator("velocity")
    @classmethod
    def _velocity_unit(cls, v: QuantityVector | None) -> QuantityVector | None:
        if v is None:
            return v
        if v.unit not in _VELOCITY_UNITS:
            raise InvalidInputError(
                f"velocity unit must be a velocity ({sorted(_VELOCITY_UNITS)}), got {v.unit!r}",
                code="invalid_input.wrong_unit_category",
            )
        if len(v.value) != 3:
            raise InvalidInputError(
                f"velocity must have exactly 3 components, got {len(v.value)}",
                code="invalid_input.wrong_vector_length",
            )
        return v


class SpiceFrameTransformResponse(BaseModel):
    """A frame-to-frame rotation, plus the rotated state when one was supplied."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    from_frame: str = Field(..., description="Source frame, echoed verbatim from the request.")
    to_frame: str = Field(..., description="Target frame, echoed verbatim from the request.")
    epoch: str = Field(
        ...,
        description=(
            "The UTC ISO 8601 epoch the rotation is evaluated at, echoed verbatim from the request."
        ),
    )
    rotation: list[QuantityVector] = Field(
        ...,
        description=(
            "The 3x3 orientation matrix from pxform, as three row vectors: row i is "
            "the i-th row of R, where a source-frame vector v maps to R @ v in the "
            "target frame. Dimensionless (unit '1'). Always present, including for a "
            "rotation-only request."
        ),
    )
    position: QuantityVector | None = Field(
        None,
        description=(
            "The input position rotated into `to_frame`, in the same length unit as "
            "the input. Null when no state was supplied (a rotation-only request)."
        ),
    )
    velocity: QuantityVector | None = Field(
        None,
        description=(
            "The input velocity rotated into `to_frame` via the full state transform "
            "(sxform), in the same velocity unit as the input. Null when no velocity "
            "was supplied. For a rotating target frame this differs from rotation @ "
            "velocity, because sxform also folds in the frame's rotation rate."
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
    """Confirm *source* is a readable local kernel file; return its absolute path.

    CSPICE keys the kernel pool on the literal path string furnished and does not
    canonicalise it, so a relative path would make the kernel's name — and thus
    the unload key — depend on the process working directory, and two distinct
    files furnished under the same relative name would collide on one pool entry.
    Absolutising here makes the name working-directory-independent; the
    existence check turns a missing or non-file path into a typed input error
    rather than a CSPICE abort. ``abspath``, deliberately not ``realpath``: it
    must not resolve symlinks, or the name would diverge from the caller's path
    on platforms whose temp dirs live under a symlinked root.
    """
    if not Path(source).is_file():
        raise InvalidInputError(
            f"no readable kernel file at {source!r}; pass a local filesystem path to a "
            "kernel, or an https NAIF URL",
            code="invalid_input.spice_kernel_not_found",
            data={"source": source},
        )
    return os.path.abspath(source)


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


def _to_cspice_utc(epoch: str) -> str:
    """Render a validated ISO 8601 epoch as a CSPICE-parseable UTC string.

    CSPICE ``str2et`` reads an ISO calendar string as UTC but does not parse a
    trailing ``Z`` or a ``±HH:MM`` offset designator. The epoch has already
    passed the :data:`~astrodynamics_mcp.schemas.base.Epoch` shape check, so we
    convert it to a timezone-aware instant (honouring whatever offset it
    carried), shift to UTC, and emit an offset-free ISO string CSPICE accepts.
    """
    instant = _epoch_to_instant(epoch).astimezone(timezone.utc)
    return instant.strftime("%Y-%m-%dT%H:%M:%S.%f")


async def _do_state(
    target: str,
    observer: str,
    epochs: list[str],
    frame: str,
    aberration: str,
) -> SpiceStateResponse:
    """Query the state of *target* relative to *observer* at each of *epochs*.

    Validates the aberration correction up front (a malformed one never reaches
    CSPICE), then runs one ``str2et`` + ``spkezr`` per epoch on the worker
    thread. Light time is surfaced only for a non-``NONE`` correction, per the
    tool contract; a geometric query reports null light time.
    """
    abcorr = normalize_aberration(aberration)
    report_light_time = abcorr != "NONE"

    states: list[SpiceStateAtEpoch] = []
    for epoch in epochs:
        result = await run_on_spice_thread(
            query_state, target, observer, _to_cspice_utc(epoch), frame, abcorr
        )
        states.append(
            SpiceStateAtEpoch(
                epoch=epoch,
                position=QuantityVector(value=list(result.position), unit="km"),
                velocity=QuantityVector(value=list(result.velocity), unit="km/s"),
                light_time=(
                    Quantity(value=result.light_time, unit="s") if report_light_time else None
                ),
            )
        )

    return SpiceStateResponse(
        target=target,
        observer=observer,
        frame=frame,
        aberration=abcorr,
        states=states,
    )


async def _do_frame_transform(
    from_frame: str,
    to_frame: str,
    epoch: str,
    state: RotatableState | None,
) -> SpiceFrameTransformResponse:
    """Rotate *state* (or just compute the matrix) from *from_frame* to *to_frame*.

    Normalises the ISO 8601 epoch to a CSPICE-parseable UTC string, then runs
    ``str2et`` + ``pxform`` (and ``sxform`` when a velocity is present) on the
    worker thread. The rotated position / velocity echo the input units; the
    3x3 orientation matrix is always returned, dimensionless.
    """
    position_unit = state.position.unit if state is not None else None
    velocity_unit = (
        state.velocity.unit if state is not None and state.velocity is not None else None
    )
    position = [float(x) for x in state.position.value] if state is not None else None
    velocity = (
        [float(x) for x in state.velocity.value]
        if state is not None and state.velocity is not None
        else None
    )

    result = await run_on_spice_thread(
        query_frame_transform, from_frame, to_frame, _to_cspice_utc(epoch), position, velocity
    )

    rotation_rows = [QuantityVector(value=list(row), unit="1") for row in result.rotation]
    rotated_position = (
        QuantityVector(value=list(result.rotated_position), unit=position_unit)
        if result.rotated_position is not None and position_unit is not None
        else None
    )
    rotated_velocity = (
        QuantityVector(value=list(result.rotated_velocity), unit=velocity_unit)
        if result.rotated_velocity is not None and velocity_unit is not None
        else None
    )
    return SpiceFrameTransformResponse(
        from_frame=from_frame,
        to_frame=to_frame,
        epoch=epoch,
        rotation=rotation_rows,
        position=rotated_position,
        velocity=rotated_velocity,
    )


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
    "Query the state — position (km) and velocity (km/s) — of a target body relative to "
    "an observer body at one or more epochs, read from furnished SPK kernels, e.g. "
    "spice_state(target='MOON', observer='EARTH', epochs=['2026-01-01T00:00:00Z']) returns "
    "the Moon's geocentric state in J2000 (CSPICE's name for the Earth-mean-equator/equinox-"
    "of-J2000 inertial frame, aligned with ICRF to milliarcsecond level). Requires the "
    "relevant SPK *and* a leap-second kernel (LSK) furnished first via spice_load_kernel — a "
    "missing kernel returns a typed error, never a silent empty state. Each epoch must be UTC "
    "ISO 8601 with a time component (e.g. '2026-01-01T00:00:00Z'), not a bare date. `target` "
    "and `observer` accept body names ('MOON', 'MARS') or NAIF integer IDs as strings ('301', "
    "'499'), not arbitrary labels. `aberration` selects the correction (NONE for the geometric "
    "state, or LT / LT+S / CN / CN+S and their X-prefixed forms for light-time and stellar-"
    "aberration corrections); light time is returned only when a correction other than NONE is "
    "requested. Use the kernel-free frame_transform tool for Earth-centred frame changes; this "
    "tool is for SPK-backed ephemeris states."
)

_FRAME_TRANSFORM_DESCRIPTION = (
    "Rotate a vector between SPICE reference frames defined by furnished FK / PCK "
    "kernels at an epoch — in particular the non-Earth body-fixed frames the "
    "kernel-free frame_transform tool cannot provide. e.g. "
    "spice_frame_transform(from_frame='J2000', to_frame='IAU_MARS', "
    "epoch='2026-01-01T00:00:00Z', state={position: {value: [4000, 5000, 6000], "
    "unit: 'km'}}) rotates a position into the Mars body-fixed frame (a Mars PCK "
    "must be furnished first). Omit `state` to get just the 3x3 rotation matrix; "
    "pass position only for a pxform rotation, or position+velocity for the full "
    "sxform state rotation (which folds the target frame's rotation rate into the "
    "rotated velocity). The FK / PCK defining the frame must be furnished first via "
    "spice_load_kernel, plus a leap-second kernel (LSK) for the epoch — a missing "
    "kernel or an unrecognised frame returns a typed error, never a silent result. "
    "`epoch` is UTC ISO 8601 with a time component (e.g. '2026-01-01T00:00:00Z'). "
    "Use this for body-fixed frames like IAU_MARS / IAU_MOON; for ICRF / ITRS / "
    "GCRS / TEME prefer the kernel-free frame_transform."
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
    async def spice_state(
        target: Annotated[
            str,
            Field(
                description=(
                    "The body whose state to query — a body name ('MOON', 'MARS') or a "
                    "NAIF integer ID as a string ('301', '499'). Resolved by CSPICE against "
                    "the furnished kernels; an unrecognised name returns a typed error."
                ),
            ),
        ],
        observer: Annotated[
            str,
            Field(
                description=(
                    "The body the state is measured relative to — a body name ('EARTH', "
                    "'SOLAR SYSTEM BARYCENTER') or a NAIF integer ID as a string ('399', "
                    "'0'). Same name/ID resolution as `target`."
                ),
            ),
        ],
        epochs: Annotated[
            list[Epoch],
            Field(
                description=(
                    "One or more UTC ISO 8601 epochs with a mandatory time component "
                    "(e.g. ['2026-01-01T00:00:00Z']); a bare date is rejected. Each is "
                    "queried independently and returned in the same order."
                ),
            ),
        ],
        frame: Annotated[
            str,
            Field(
                description=(
                    "Reference frame the state is expressed in. Defaults to 'J2000' (CSPICE's "
                    "Earth-mean-equator/equinox-of-J2000 inertial frame, aligned with ICRF). "
                    "Any frame the furnished kernels define is accepted (e.g. 'ECLIPJ2000', "
                    "'IAU_MARS')."
                ),
            ),
        ] = "J2000",
        aberration: Annotated[
            str,
            Field(
                description=(
                    "Aberration correction: 'NONE' for the geometric state, or 'LT' / 'LT+S' "
                    "/ 'CN' / 'CN+S' (and the X-prefixed transmission forms) for light-time "
                    "and stellar-aberration corrections. Light time is returned only for a "
                    f"non-NONE correction. Valid values: {list(SPICE_ABERRATION_CORRECTIONS)}."
                ),
            ),
        ] = "NONE",
    ) -> SpiceStateResponse:
        return await _do_state(target, observer, epochs, frame, aberration)

    @register_tool(
        name="spice_frame_transform",
        description=_FRAME_TRANSFORM_DESCRIPTION,
        annotations=ToolAnnotations(
            title="SPICE Frame Transform", readOnlyHint=True, openWorldHint=False
        ),
    )
    async def spice_frame_transform(
        from_frame: Annotated[
            str,
            Field(
                description=(
                    "The source SPICE frame the input is currently expressed in — any "
                    "frame name CSPICE recognises once the defining kernels are furnished "
                    "(e.g. 'J2000', 'ECLIPJ2000', 'IAU_MARS'). An unrecognised frame "
                    "returns a typed error."
                ),
            ),
        ],
        to_frame: Annotated[
            str,
            Field(
                description=(
                    "The target SPICE frame to rotate into (e.g. 'IAU_MARS', 'IAU_MOON', "
                    "'ITRF93'). The FK / PCK defining a body-fixed target must be furnished "
                    "first via spice_load_kernel."
                ),
            ),
        ],
        epoch: Annotated[
            Epoch,
            Field(
                description=(
                    "UTC ISO 8601 epoch with a mandatory time component "
                    "(e.g. '2026-01-01T00:00:00Z') at which the rotation is evaluated; a "
                    "bare date is rejected. A leap-second kernel (LSK) must be furnished to "
                    "resolve it."
                ),
            ),
        ],
        state: Annotated[
            RotatableState | None,
            Field(
                description=(
                    "Optional 3- or 6-vector to rotate: {position} for a pxform rotation, "
                    "or {position, velocity} for the full sxform state rotation. Omit "
                    "entirely to return just the 3x3 rotation matrix. e.g. "
                    "{position: {value: [4000, 5000, 6000], unit: 'km'}}."
                ),
            ),
        ] = None,
    ) -> SpiceFrameTransformResponse:
        return await _do_frame_transform(from_frame, to_frame, epoch, state)

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
