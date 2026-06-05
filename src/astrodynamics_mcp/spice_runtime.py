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
from collections.abc import Callable
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
