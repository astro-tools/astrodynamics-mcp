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
:mod:`astrodynamics_mcp.tools.spice` call them through
:func:`run_on_spice_thread`, one call per tool invocation so each tool's whole
CSPICE interaction is atomic against any other.

``spiceypy`` is imported lazily, inside the worker, so importing this module on
a bare install (no ``[spice]`` extra) does not require CSPICE — matching the
conditional-registration gate the tool module uses.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
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
    let an unload-of-missing succeed vacuously. Runs on the worker thread.
    """
    sp = _spiceypy()
    if not any(_same_path(row.name, name) for row in list_pool()):
        raise InvalidInputError(
            f"no kernel named {name!r} is loaded; call spice_list_kernels to see the "
            "loaded names, and unload by the name returned from spice_load_kernel "
            "(not the original URL)",
            code="invalid_input.spice_kernel_not_loaded",
            data={"name": name},
        )
    try:
        sp.unload(name)
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
    try:
        et = sp.str2et(utc_epoch)
    except Exception as exc:  # spiceypy raises SpiceyError / subclasses
        raise UpstreamError(
            f"CSPICE could not resolve the epoch {utc_epoch!r} to ephemeris time: {exc}. "
            "A leap-second kernel (LSK) must be furnished first (spice_load_kernel).",
            code="upstream.spice_state_failed",
            original_exception=exc,
            data={"target": target, "observer": observer, "epoch": utc_epoch},
        ) from exc
    try:
        state, light_time = sp.spkezr(target, et, frame, abcorr, observer)
    except Exception as exc:  # spiceypy raises SpiceyError / subclasses
        raise UpstreamError(
            f"CSPICE could not compute the state of {target!r} relative to {observer!r} "
            f"at {utc_epoch!r} in frame {frame!r}: {exc}. Confirm the relevant SPK is "
            "furnished (spice_load_kernel) and the body names / NAIF IDs are valid.",
            code="upstream.spice_state_failed",
            original_exception=exc,
            data={
                "target": target,
                "observer": observer,
                "epoch": utc_epoch,
                "frame": frame,
                "aberration": abcorr,
            },
        ) from exc
    return SpiceState(
        position=(float(state[0]), float(state[1]), float(state[2])),
        velocity=(float(state[3]), float(state[4]), float(state[5])),
        light_time=float(light_time),
    )


def query_frame_transform(
    from_frame: str,
    to_frame: str,
    utc_epoch: str,
    position: Sequence[float] | None,
    velocity: Sequence[float] | None,
) -> FrameRotation:
    """Return the *from_frame* → *to_frame* rotation at *utc_epoch*.

    Resolves the UTC epoch to ephemeris time with ``str2et`` (which needs a
    furnished leap-second kernel), reads the 3x3 orientation with ``pxform``
    (which needs the FK / PCK defining any frame CSPICE does not build in), and
    — when a velocity is supplied — the 6x6 state transform with ``sxform`` so
    the rotated velocity carries the target frame's rotation rate. Every CSPICE
    failure — a missing LSK / FK / PCK, an unrecognised frame name, an epoch the
    frame data does not cover — surfaces as a typed
    :class:`~astrodynamics_mcp.errors.UpstreamError` with a stable code rather
    than a silent result, so the consumer never mistakes an unfurnished pool for
    a degenerate rotation. Runs on the worker thread.

    *utc_epoch* must already be a CSPICE-parseable UTC string (the tool layer
    normalises the ISO 8601 input — stripping the ``Z`` / offset designator —
    before handing it here). *position* / *velocity* are the source-frame
    vectors to rotate, or ``None`` to request the rotation matrix alone.
    """
    sp = _spiceypy()
    try:
        et = sp.str2et(utc_epoch)
    except Exception as exc:  # spiceypy raises SpiceyError / subclasses
        raise UpstreamError(
            f"CSPICE could not resolve the epoch {utc_epoch!r} to ephemeris time: {exc}. "
            "A leap-second kernel (LSK) must be furnished first (spice_load_kernel).",
            code="upstream.spice_frame_transform_failed",
            original_exception=exc,
            data={"from_frame": from_frame, "to_frame": to_frame, "epoch": utc_epoch},
        ) from exc
    try:
        matrix = sp.pxform(from_frame, to_frame, et)
    except Exception as exc:  # spiceypy raises SpiceyError / subclasses
        raise UpstreamError(
            f"CSPICE could not rotate from frame {from_frame!r} to {to_frame!r} at "
            f"{utc_epoch!r}: {exc}. Confirm the FK / PCK defining the frame is furnished "
            "(spice_load_kernel) and both frame names are recognised.",
            code="upstream.spice_frame_transform_failed",
            original_exception=exc,
            data={"from_frame": from_frame, "to_frame": to_frame, "epoch": utc_epoch},
        ) from exc
    rotation = (
        (float(matrix[0][0]), float(matrix[0][1]), float(matrix[0][2])),
        (float(matrix[1][0]), float(matrix[1][1]), float(matrix[1][2])),
        (float(matrix[2][0]), float(matrix[2][1]), float(matrix[2][2])),
    )

    rotated_position: tuple[float, float, float] | None = None
    rotated_velocity: tuple[float, float, float] | None = None

    if position is not None and velocity is not None:
        try:
            xform = sp.sxform(from_frame, to_frame, et)
        except Exception as exc:  # spiceypy raises SpiceyError / subclasses
            raise UpstreamError(
                f"CSPICE could not build the state transform from frame {from_frame!r} to "
                f"{to_frame!r} at {utc_epoch!r}: {exc}. Confirm the FK / PCK defining the "
                "frame is furnished (spice_load_kernel) and both frame names are recognised.",
                code="upstream.spice_frame_transform_failed",
                original_exception=exc,
                data={"from_frame": from_frame, "to_frame": to_frame, "epoch": utc_epoch},
            ) from exc
        state: tuple[float, ...] = (*position, *velocity)
        out = [sum((float(xform[i][j]) * state[j] for j in range(6)), 0.0) for i in range(6)]
        rotated_position = (out[0], out[1], out[2])
        rotated_velocity = (out[3], out[4], out[5])
    elif position is not None:
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
