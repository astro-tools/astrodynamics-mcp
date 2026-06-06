"""Single-worker CSPICE executor and kernel-pool primitives.

CSPICE is not thread-safe and its kernel pool is a single block of
process-global state, while the server is asyncio-based and serves many tool
calls. The SPICE integration design note (``docs/spice-integration.md``)
reconciles the two: **every CSPICE call runs on one dedicated worker thread.**
:func:`run_on_spice_thread` marshals each call onto that single-worker
executor, so CSPICE is only ever entered from one thread — serialised, never
concurrent — while the asyncio event loop never blocks on it and the kernel
pool persists in-process across calls.

The pool primitives (:func:`furnish_and_describe`, :func:`list_pool`,
:func:`unload_kernel`) run *inside* that worker; the tool bodies in
:mod:`astrodynamics_mcp.tools.spice` reach them through
:func:`run_on_spice_thread`. Each tool dispatches all of its CSPICE work in a
single worker call — the batch helpers (:func:`query_states`,
:func:`query_body_constants`) loop over their epochs / parameters *inside* the
worker rather than dispatching once per item — so each tool's whole CSPICE
interaction runs to completion without another tool's calls interleaving into
it.

``spiceypy`` is imported lazily, inside the worker, so importing this module on
a bare install (no ``[spice]`` extra) does not require CSPICE — matching the
conditional-registration gate the tool module uses.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

from astrodynamics_mcp.errors import InvalidInputError, UpstreamError

_logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# The pool-category keywords CSPICE's ``ktotal`` / ``kdata`` recognise and
# report as a kernel's type. The tool surface exposes exactly these; "no
# filter" maps to CSPICE's own catch-all ``"ALL"`` keyword here, never on the
# wire. Leap-second (LSK), frame (FK), and spacecraft-clock (SCLK) kernels all
# report as ``TEXT`` — CSPICE does not distinguish them at this layer.
SPICE_KERNEL_CATEGORIES: tuple[str, ...] = ("SPK", "CK", "PCK", "EK", "DSK", "META", "TEXT")
_ALL_CATEGORIES = "ALL"

# The aberration-correction keywords CSPICE ``spkezr`` accepts. ``NONE`` is the
# geometric (true relative) state; the remainder apply light-time (``LT``),
# light-time + stellar aberration (``LT+S``), their converged-Newtonian
# (``CN``) variants, and the transmission-side (``X``-prefixed) forms. CSPICE is
# case-insensitive here; the tool surface upper-cases and validates against this
# set so a malformed correction never reaches CSPICE.
SPICE_ABERRATION_CORRECTIONS: tuple[str, ...] = (
    "NONE",
    "LT",
    "LT+S",
    "CN",
    "CN+S",
    "XLT",
    "XLT+S",
    "XCN",
    "XCN+S",
)

# Upper bound on the element count ``bodvcd`` may return for a single body
# constant. RADII has 3, GM has 1, and orientation coefficient arrays
# (POLE_RA / POLE_DEC / PM) are short; 256 is far above any real value while
# still bounding the CSPICE call.
_MAX_BODY_CONSTANT_VALUES = 256

# The kernel-defined time systems ``spice_time_convert`` bridges. ``ET`` is TDB
# seconds past J2000 (the leap-second-kernel-defined zero), ``UTC`` a calendar
# string, and ``SCLK`` a spacecraft-clock string — each meaningful only with the
# relevant kernel furnished. The tool surface exposes exactly these; the
# normaliser upper-cases and validates against the set so a malformed system
# never reaches CSPICE.
SPICE_TIME_SYSTEMS: tuple[str, ...] = ("ET", "UTC", "SCLK")

# CSPICE ``et2utc`` output format and precision. ``ISOC`` is the ISO 8601
# calendar form (``YYYY-MM-DDTHH:MM:SS.ffffff``); six fractional digits mirror
# the microsecond precision the tool layer feeds ``str2et`` on the way in, so a
# UTC round-trip neither gains nor loses resolution.
_ET2UTC_FORMAT = "ISOC"
_ET2UTC_PRECISION = 6


@dataclass(frozen=True)
class KernelRow:
    """One entry in the CSPICE kernel pool, as ``kdata`` reports it.

    ``name`` is the path CSPICE knows the kernel by (the unload key); ``type``
    is one of :data:`SPICE_KERNEL_CATEGORIES`; ``source`` is the meta-kernel
    that furnished it (empty when furnished directly); ``handle`` is the DAF /
    DAS file handle, ``0`` for text kernels loaded into the pool.
    """

    name: str
    type: str
    source: str
    handle: int


@dataclass(frozen=True)
class SpiceState:
    """A target's state relative to an observer, as ``spkezr`` reports it.

    ``position`` is the Cartesian position in km and ``velocity`` the Cartesian
    velocity in km/s, both in the requested reference frame. ``light_time`` is
    the one-way light time in seconds CSPICE returned for the query; it is the
    geometric light time when the aberration correction was ``NONE`` and the
    corrected value otherwise. The tool layer decides whether to surface it.
    """

    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    light_time: float


@dataclass(frozen=True)
class FrameRotation:
    """A frame-to-frame rotation, as ``pxform`` / ``sxform`` report it.

    ``rotation`` is the 3x3 orientation matrix (row-major, dimensionless) such
    that ``v_to = rotation @ v_from`` for any vector expressed in the source
    frame. ``rotated_position`` and ``rotated_velocity`` are the supplied state
    rotated into the target frame; each is ``None`` when its input was not
    given. The position is rotated by ``rotation``; the velocity, when present,
    is rotated by the full 6x6 state transform (``sxform``), which additionally
    carries the target frame's rotation rate into the velocity — so for a
    rotating target frame the rotated velocity is not simply ``rotation`` times
    the input velocity.
    """

    rotation: tuple[tuple[float, float, float], ...]
    rotated_position: tuple[float, float, float] | None
    rotated_velocity: tuple[float, float, float] | None


@dataclass(frozen=True)
class BodyConstant:
    """One body constant read from the kernel pool, as ``bodvcd`` reports it.

    ``source`` is the pool variable CSPICE read the value from (e.g.
    ``BODY499_RADII``) — the authoritative provenance the pool exposes; CSPICE
    does not attribute a pool variable to its source file. ``values`` is the raw
    constant array: one element for a scalar like GM, three for RADII, the
    polynomial coefficients for an orientation item (POLE_RA / POLE_DEC / PM).
    The tool layer assigns the per-element units.
    """

    source: str
    values: tuple[float, ...]


# ---------------------------------------------------------------------------
# The single CSPICE worker thread
# ---------------------------------------------------------------------------

_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()
_worker_configured = False


def _get_executor() -> ThreadPoolExecutor:
    """Return the process-wide single-worker CSPICE executor, creating it once."""
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spice")
    return _executor


def _configure_cspice() -> None:
    """Mute CSPICE's own error output and make it return rather than abort.

    Runs once, on the worker thread, before the first CSPICE call. CSPICE's
    default error device is standard output — on the stdio transport that would
    corrupt the JSON-RPC stream — so we point the error device at ``NULL`` and
    suppress message printing. With the error action set to ``RETURN``, spiceypy
    surfaces each failure as a Python exception (which we translate to a typed
    error) instead of the process aborting.
    """
    global _worker_configured
    if _worker_configured:
        return
    import spiceypy

    spiceypy.erract("SET", "RETURN")
    spiceypy.errdev("SET", "NULL")
    spiceypy.errprt("SET", "NONE")
    _worker_configured = True


async def run_on_spice_thread(fn: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:
    """Run *fn* on the single CSPICE worker thread and await its result.

    Serialises every CSPICE entry onto one thread (CSPICE is not thread-safe)
    while leaving the event loop free. The first call configures CSPICE error
    handling on the worker; *fn* and its arguments then run there to completion.
    """
    loop = asyncio.get_running_loop()

    def _runner() -> _T:
        _configure_cspice()
        return fn(*args, **kwargs)

    return await loop.run_in_executor(_get_executor(), _runner)


# ---------------------------------------------------------------------------
# Pool primitives — these run inside the worker thread
# ---------------------------------------------------------------------------


@contextmanager
def _cspice_call(*, code: str, action: str, hint: str, data: dict[str, Any]) -> Iterator[None]:
    """Run a CSPICE call, turning any ``spiceypy`` failure into a typed error.

    Centralises the one error-translation shape every pool query helper needs:
    a ``spiceypy`` exception (``SpiceyError`` / subclasses) becomes an
    :class:`~astrodynamics_mcp.errors.UpstreamError` carrying the stable *code*,
    the original exception, and the *data* context — so a missing kernel, an
    unrecognised name, or an out-of-coverage epoch surfaces as a typed error
    rather than a raw CSPICE abort or a silent result. The raised message is
    ``f"{action}: {exc}. {hint}"`` — *action* describes what CSPICE was asked to
    do, *hint* the remediation (which kernel to furnish) — so the consumer sees
    both the CSPICE diagnostic and how to fix it.
    """
    try:
        yield
    except Exception as exc:  # spiceypy raises SpiceyError / subclasses
        raise UpstreamError(
            f"{action}: {exc}. {hint}",
            code=code,
            original_exception=exc,
            data=data,
        ) from exc


def _spiceypy() -> Any:
    """Import and return the ``spiceypy`` module (lazy; worker-thread only)."""
    import spiceypy

    return spiceypy


def _same_path(a: str, b: str) -> bool:
    """Whether two kernel paths name the same file (exact or normalised match)."""
    if a == b:
        return True
    try:
        return os.path.normpath(a) == os.path.normpath(b)
    except (ValueError, TypeError):  # pragma: no cover - defensive
        return False


def list_pool(kind: str | None = None) -> list[KernelRow]:
    """Return the kernel-pool rows, optionally filtered to *kind* categories.

    *kind* is a CSPICE category keyword or a space-joined set of them (e.g.
    ``"SPK"`` or ``"SPK PCK"``); ``None`` lists every kernel. Iterates
    ``ktotal`` / ``kdata`` for the requested category and skips any slot CSPICE
    reports as not-found (a kernel unloaded mid-enumeration). Runs on the worker
    thread.
    """
    sp = _spiceypy()
    category = kind if kind else _ALL_CATEGORIES
    count = int(sp.ktotal(category))
    rows: list[KernelRow] = []
    for i in range(count):
        file, filtyp, srcfil, handle, found = sp.kdata(i, category)
        if not found:
            continue
        rows.append(KernelRow(name=file, type=filtyp, source=srcfil, handle=int(handle)))
    return rows


def furnish_and_describe(path: str) -> list[KernelRow]:
    """Furnish *path* into the pool and return the rows this call added.

    Snapshots the pool before and after the ``furnsh`` so a meta-kernel — which
    furnishes every kernel it lists — reports all of them, not just itself; a
    plain kernel yields a single row. A re-furnish of an already-loaded kernel
    adds nothing new, so the existing row(s) for *path* are returned instead of
    an empty list. A CSPICE failure (corrupt or unreadable kernel) becomes a
    typed :class:`~astrodynamics_mcp.errors.UpstreamError`. Runs on the worker
    thread.
    """
    sp = _spiceypy()
    before = {row.name for row in list_pool()}
    try:
        sp.furnsh(path)
    except Exception as exc:  # spiceypy raises SpiceyError / subclasses
        raise UpstreamError(
            f"CSPICE could not furnish the kernel at {path!r}: {exc}",
            code="upstream.spice_furnish_failed",
            original_exception=exc,
            data={"path": path},
        ) from exc
    after = list_pool()
    added = [row for row in after if row.name not in before]
    if added:
        return added
    # Idempotent re-furnish: nothing new entered the pool. Report the existing
    # row(s) for this path so the caller still gets a meaningful description.
    return [row for row in after if _same_path(row.name, path)]


def unload_kernel(name: str) -> int:
    """Unload the kernel known by *name*; return the remaining pool count.

    CSPICE ``unload`` silently no-ops on a file that is not furnished, so we
    pre-check pool membership and raise a typed not-loaded error rather than
    let an unload-of-missing succeed vacuously. The membership check matches by
    :func:`_same_path` (so a normalised-but-equal form of the furnished path is
    accepted), but CSPICE ``unload`` keys on the *literal* furnished string — so
    we hand it the matched pool row's stored name, not the caller's argument. A
    normpath-equal-but-not-identical name would otherwise pass the check yet
    ``unload`` would no-op on it, leaving the kernel loaded while the tool
    reported success. Runs on the worker thread.
    """
    sp = _spiceypy()
    matched = next((row.name for row in list_pool() if _same_path(row.name, name)), None)
    if matched is None:
        raise InvalidInputError(
            f"no kernel named {name!r} is loaded; call spice_list_kernels to see the "
            "loaded names, and unload by the name returned from spice_load_kernel "
            "(not the original URL)",
            code="invalid_input.spice_kernel_not_loaded",
            data={"name": name},
        )
    try:
        sp.unload(matched)
    except Exception as exc:  # spiceypy raises SpiceyError / subclasses
        raise UpstreamError(
            f"CSPICE could not unload the kernel {name!r}: {exc}",
            code="upstream.spice_unload_failed",
            original_exception=exc,
            data={"name": name},
        ) from exc
    return int(sp.ktotal(_ALL_CATEGORIES))


def query_state(target: str, observer: str, utc_epoch: str, frame: str, abcorr: str) -> SpiceState:
    """Return *target*'s state relative to *observer* at *utc_epoch*.

    Resolves the UTC epoch to ephemeris time with ``str2et`` (which needs a
    furnished leap-second kernel), then reads the state with ``spkezr`` (which
    needs the relevant SPK). Both CSPICE failures — a missing kernel, an
    unrecognised body name, an epoch outside the SPK's coverage — surface as a
    typed :class:`~astrodynamics_mcp.errors.UpstreamError` with a stable code
    rather than a silent empty result, so the consumer never mistakes a
    not-loaded pool for a degenerate state. Runs on the worker thread.

    *utc_epoch* must already be a CSPICE-parseable UTC string (the tool layer
    normalises the ISO 8601 input — stripping the ``Z`` / offset designator —
    before handing it here). *abcorr* must be one of
    :data:`SPICE_ABERRATION_CORRECTIONS`; the tool layer validates it first.
    """
    sp = _spiceypy()
    with _cspice_call(
        code="upstream.spice_state_failed",
        action=f"CSPICE could not resolve the epoch {utc_epoch!r} to ephemeris time",
        hint="A leap-second kernel (LSK) must be furnished first (spice_load_kernel).",
        data={"target": target, "observer": observer, "epoch": utc_epoch},
    ):
        et = sp.str2et(utc_epoch)
    with _cspice_call(
        code="upstream.spice_state_failed",
        action=(
            f"CSPICE could not compute the state of {target!r} relative to {observer!r} "
            f"at {utc_epoch!r} in frame {frame!r}"
        ),
        hint=(
            "Confirm the relevant SPK is furnished (spice_load_kernel) and the body "
            "names / NAIF IDs are valid."
        ),
        data={
            "target": target,
            "observer": observer,
            "epoch": utc_epoch,
            "frame": frame,
            "aberration": abcorr,
        },
    ):
        state, light_time = sp.spkezr(target, et, frame, abcorr, observer)
    return SpiceState(
        position=(float(state[0]), float(state[1]), float(state[2])),
        velocity=(float(state[3]), float(state[4]), float(state[5])),
        light_time=float(light_time),
    )


def query_states(
    target: str, observer: str, utc_epochs: Sequence[str], frame: str, abcorr: str
) -> list[SpiceState]:
    """Return *target*'s state relative to *observer* at each of *utc_epochs*.

    Loops :func:`query_state` over the epochs *inside* one worker call, so a
    multi-epoch query is a single atomic CSPICE interaction — no other tool's
    calls interleave between epochs — and the per-epoch ``str2et`` + ``spkezr``
    cost one thread round-trip rather than one per epoch. Each epoch must
    already be a CSPICE-parseable UTC string (the tool layer normalises the
    ISO 8601 input). Runs on the worker thread.
    """
    return [query_state(target, observer, utc_epoch, frame, abcorr) for utc_epoch in utc_epochs]


def query_frame_transform(
    from_frame: str,
    to_frame: str,
    utc_epoch: str,
    position: Sequence[float] | None,
    velocity: Sequence[float] | None,
) -> FrameRotation:
    """Return the *from_frame* → *to_frame* rotation at *utc_epoch*.

    Resolves the UTC epoch to ephemeris time with ``str2et`` (which needs a
    furnished leap-second kernel), then reads the orientation in a single CSPICE
    rotation call: ``pxform`` (the 3x3) for a rotation-only or position-only
    request, or — when a velocity is supplied — the 6x6 state transform
    ``sxform``, whose upper-left block is that same 3x3 orientation, so the
    rotated velocity carries the target frame's rotation rate without a second
    ``pxform`` call. Both need the FK / PCK defining any frame CSPICE does not
    build in. Every CSPICE failure — a missing LSK / FK / PCK, an unrecognised
    frame name, an epoch the frame data does not cover — surfaces as a typed
    :class:`~astrodynamics_mcp.errors.UpstreamError` with a stable code rather
    than a silent result, so the consumer never mistakes an unfurnished pool for
    a degenerate rotation. Runs on the worker thread.

    *utc_epoch* must already be a CSPICE-parseable UTC string (the tool layer
    normalises the ISO 8601 input — stripping the ``Z`` / offset designator —
    before handing it here). *position* / *velocity* are the source-frame
    vectors to rotate, or ``None`` to request the rotation matrix alone.
    """
    sp = _spiceypy()
    with _cspice_call(
        code="upstream.spice_frame_transform_failed",
        action=f"CSPICE could not resolve the epoch {utc_epoch!r} to ephemeris time",
        hint="A leap-second kernel (LSK) must be furnished first (spice_load_kernel).",
        data={"from_frame": from_frame, "to_frame": to_frame, "epoch": utc_epoch},
    ):
        et = sp.str2et(utc_epoch)
    rotated_position: tuple[float, float, float] | None = None
    rotated_velocity: tuple[float, float, float] | None = None

    if position is not None and velocity is not None:
        # The state path: sxform yields both the rotated 6-vector and the 3x3
        # orientation (its upper-left block), so read the matrix from here rather
        # than making a separate pxform call.
        with _cspice_call(
            code="upstream.spice_frame_transform_failed",
            action=(
                f"CSPICE could not build the state transform from frame {from_frame!r} "
                f"to {to_frame!r} at {utc_epoch!r}"
            ),
            hint=(
                "Confirm the FK / PCK defining the frame is furnished (spice_load_kernel) "
                "and both frame names are recognised."
            ),
            data={"from_frame": from_frame, "to_frame": to_frame, "epoch": utc_epoch},
        ):
            xform = sp.sxform(from_frame, to_frame, et)
        rotation = (
            (float(xform[0][0]), float(xform[0][1]), float(xform[0][2])),
            (float(xform[1][0]), float(xform[1][1]), float(xform[1][2])),
            (float(xform[2][0]), float(xform[2][1]), float(xform[2][2])),
        )
        state: tuple[float, ...] = (*position, *velocity)
        out = [sum((float(xform[i][j]) * state[j] for j in range(6)), 0.0) for i in range(6)]
        rotated_position = (out[0], out[1], out[2])
        rotated_velocity = (out[3], out[4], out[5])
    else:
        # Rotation-only or position-only: pxform yields the 3x3 orientation.
        with _cspice_call(
            code="upstream.spice_frame_transform_failed",
            action=(
                f"CSPICE could not rotate from frame {from_frame!r} to {to_frame!r} "
                f"at {utc_epoch!r}"
            ),
            hint=(
                "Confirm the FK / PCK defining the frame is furnished (spice_load_kernel) "
                "and both frame names are recognised."
            ),
            data={"from_frame": from_frame, "to_frame": to_frame, "epoch": utc_epoch},
        ):
            matrix = sp.pxform(from_frame, to_frame, et)
        rotation = (
            (float(matrix[0][0]), float(matrix[0][1]), float(matrix[0][2])),
            (float(matrix[1][0]), float(matrix[1][1]), float(matrix[1][2])),
            (float(matrix[2][0]), float(matrix[2][1]), float(matrix[2][2])),
        )
        if position is not None:
            rotated_position = (
                rotation[0][0] * position[0]
                + rotation[0][1] * position[1]
                + rotation[0][2] * position[2],
                rotation[1][0] * position[0]
                + rotation[1][1] * position[1]
                + rotation[1][2] * position[2],
                rotation[2][0] * position[0]
                + rotation[2][1] * position[1]
                + rotation[2][2] * position[2],
            )

    return FrameRotation(
        rotation=rotation,
        rotated_position=rotated_position,
        rotated_velocity=rotated_velocity,
    )


def query_body_constant(body: str, item: str) -> BodyConstant:
    """Read body constant *item* for *body* from the furnished PCK pool.

    Resolves *body* (a name like ``"MARS"`` or a NAIF ID string like ``"499"``)
    to its integer code with ``bods2c`` — an unrecognised body is a typed
    :class:`~astrodynamics_mcp.errors.InvalidInputError`, distinct from a missing
    constant — then reads ``BODY<code>_<item>`` with ``bodvcd``. A constant no
    furnished kernel supplies makes CSPICE raise, which becomes a typed
    :class:`~astrodynamics_mcp.errors.UpstreamError` rather than a silent gap, so
    the consumer never mistakes an unfurnished pool for an absent constant. Runs
    on the worker thread.
    """
    sp = _spiceypy()
    code, found = sp.bods2c(body)
    if not found:
        raise InvalidInputError(
            f"unknown body {body!r}; pass a body name ('MARS') or a NAIF ID ('499') "
            "CSPICE can resolve",
            code="invalid_input.spice_unknown_body",
            data={"body": body},
        )
    with _cspice_call(
        code="upstream.spice_body_parameters_failed",
        action=f"CSPICE has no value for {item!r} of body {body!r} (BODY{int(code)}_{item})",
        hint=(
            "Furnish the PCK that defines it first (spice_load_kernel) — radii / pole / PM "
            "constants come from a planetary-constants PCK, GM from a gravity PCK."
        ),
        data={"body": body, "item": item, "code": int(code)},
    ):
        _dim, values = sp.bodvcd(int(code), item, _MAX_BODY_CONSTANT_VALUES)
    return BodyConstant(
        source=f"BODY{int(code)}_{item}",
        values=tuple(float(v) for v in values),
    )


def query_body_constants(body: str, items: Sequence[str]) -> list[BodyConstant]:
    """Read each of *items* for *body* from the furnished PCK pool.

    Loops :func:`query_body_constant` over the items *inside* one worker call,
    so a multi-parameter lookup is a single atomic CSPICE interaction at one
    thread round-trip rather than one per item. Runs on the worker thread.
    """
    return [query_body_constant(body, item) for item in items]


def _resolve_spacecraft_id(sp: Any, spacecraft: str | int) -> int:
    """Resolve a spacecraft name or NAIF ID to the integer code SCLK calls need.

    An ``int`` is taken as the NAIF spacecraft ID directly (SCLK IDs are
    negative integers, e.g. ``-82`` for Cassini). A string is resolved with
    ``bods2c`` — which accepts both a NAIF-ID digit string and a body name a
    furnished kernel maps — so an unresolvable spacecraft is a typed
    :class:`~astrodynamics_mcp.errors.InvalidInputError`, distinct from a missing
    SCLK kernel. Runs on the worker thread.
    """
    if isinstance(spacecraft, bool):  # pragma: no cover - defensive; bool is an int subclass
        raise InvalidInputError(
            "spacecraft must be a NAIF spacecraft ID or name, not a boolean",
            code="invalid_input.spice_unknown_spacecraft",
        )
    if isinstance(spacecraft, int):
        return spacecraft
    code, found = sp.bods2c(spacecraft)
    if not found:
        raise InvalidInputError(
            f"unknown spacecraft {spacecraft!r}; pass a NAIF spacecraft ID "
            "(e.g. '-82' for Cassini) or a name a furnished kernel maps",
            code="invalid_input.spice_unknown_spacecraft",
            data={"spacecraft": spacecraft},
        )
    return int(code)


def _input_to_et(sp: Any, value: str | float, from_system: str, sc_id: int | None) -> float:
    """Resolve a *from_system* input to ephemeris time (TDB seconds past J2000)."""
    if from_system == "ET":
        return float(value)
    if from_system == "UTC":
        with _cspice_call(
            code="upstream.spice_time_convert_failed",
            action=f"CSPICE could not resolve the UTC epoch {value!r} to ephemeris time",
            hint="A leap-second kernel (LSK) must be furnished first (spice_load_kernel).",
            data={"value": value, "from_system": from_system},
        ):
            return float(sp.str2et(value))
    # SCLK — sc_id is guaranteed non-None by the tool layer for any SCLK leg.
    with _cspice_call(
        code="upstream.spice_time_convert_failed",
        action=(
            f"CSPICE could not resolve the spacecraft-clock string {value!r} to ephemeris "
            f"time for spacecraft {sc_id}"
        ),
        hint="An SCLK kernel for that spacecraft must be furnished first (spice_load_kernel).",
        data={"value": value, "from_system": from_system, "spacecraft": sc_id},
    ):
        return float(sp.scs2e(sc_id, value))


def _et_to_output(sp: Any, et: float, to_system: str, sc_id: int | None) -> str | float:
    """Render ephemeris time *et* into the *to_system* representation."""
    if to_system == "ET":
        return et
    if to_system == "UTC":
        with _cspice_call(
            code="upstream.spice_time_convert_failed",
            action=f"CSPICE could not render ephemeris time {et!r} as a UTC calendar string",
            hint="A leap-second kernel (LSK) must be furnished first (spice_load_kernel).",
            data={"et": et, "to_system": to_system},
        ):
            return str(sp.et2utc(et, _ET2UTC_FORMAT, _ET2UTC_PRECISION))
    # SCLK — sc_id is guaranteed non-None by the tool layer for any SCLK leg.
    with _cspice_call(
        code="upstream.spice_time_convert_failed",
        action=(
            f"CSPICE could not render ephemeris time {et!r} as a spacecraft-clock string "
            f"for spacecraft {sc_id}"
        ),
        hint="An SCLK kernel for that spacecraft must be furnished first (spice_load_kernel).",
        data={"et": et, "to_system": to_system, "spacecraft": sc_id},
    ):
        return str(sp.sce2s(sc_id, et))


def query_time_convert(
    value: str | float,
    from_system: str,
    to_system: str,
    spacecraft: str | int | None,
) -> str | float:
    """Convert *value* from *from_system* to *to_system* through ephemeris time.

    ET (TDB seconds past J2000) is the canonical intermediate: the input is
    resolved to ET with the routine its system needs — ``str2et`` for UTC (needs
    a furnished LSK), ``scs2e`` for SCLK (needs a furnished SCLK kernel), or a
    passthrough for an ET input — then ET is rendered into the target system with
    ``et2utc`` / ``sce2s`` / a passthrough. Every CSPICE failure (a missing LSK
    or SCLK kernel, an unparseable SCLK string, an epoch outside an SCLK's
    coverage) surfaces as a typed
    :class:`~astrodynamics_mcp.errors.UpstreamError` with a stable code rather
    than a silent result. Runs on the worker thread.

    *value* arrives normalised by the tool layer: an offset-free UTC string for
    ``UTC``, a float for ``ET``, the raw clock string for ``SCLK``. *spacecraft*
    is required (non-``None``) whenever either system is ``SCLK``; the tool layer
    enforces that before dispatching here, so the SCLK legs always have a
    spacecraft to resolve.
    """
    sp = _spiceypy()
    needs_spacecraft = from_system == "SCLK" or to_system == "SCLK"
    sc_id = (
        _resolve_spacecraft_id(sp, spacecraft)
        if needs_spacecraft and spacecraft is not None
        else None
    )
    et = _input_to_et(sp, value, from_system, sc_id)
    return _et_to_output(sp, et, to_system, sc_id)


def normalize_aberration(abcorr: str) -> str:
    """Validate and upper-case an aberration-correction keyword for ``spkezr``.

    CSPICE accepts the corrections case-insensitively; we upper-case so the
    echoed value is canonical and reject anything outside
    :data:`SPICE_ABERRATION_CORRECTIONS` as a typed input error, so a malformed
    correction (``"light-time"``, ``"lt s"``) never reaches CSPICE.
    """
    upper = abcorr.upper()
    if upper not in SPICE_ABERRATION_CORRECTIONS:
        raise InvalidInputError(
            f"unknown aberration correction {abcorr!r}; valid corrections are "
            f"{list(SPICE_ABERRATION_CORRECTIONS)}",
            code="invalid_input.spice_unknown_aberration",
        )
    return upper


def normalize_kind_filter(kinds: list[str] | None) -> str | None:
    """Validate a list of category keywords and join it into a CSPICE ``kind`` string.

    ``None`` returns ``None`` (no filter — the caller lists everything). A
    non-empty list returns its deduplicated, upper-cased keywords space-joined
    the way CSPICE expects (``["SPK", "PCK"] -> "SPK PCK"``). An empty list and
    an unknown category are typed input errors, so an ambiguous or malformed
    filter never reaches CSPICE.
    """
    if kinds is None:
        return None
    if not kinds:
        raise InvalidInputError(
            "kind filter must contain at least one kernel category, or be omitted "
            "to list every loaded kernel",
            code="invalid_input.spice_empty_kind_filter",
        )
    ordered: list[str] = []
    for kind in kinds:
        upper = kind.upper()
        if upper not in SPICE_KERNEL_CATEGORIES:
            raise InvalidInputError(
                f"unknown kernel category {kind!r}; valid categories are "
                f"{list(SPICE_KERNEL_CATEGORIES)}",
                code="invalid_input.spice_unknown_kind",
            )
        if upper not in ordered:
            ordered.append(upper)
    return " ".join(ordered)
