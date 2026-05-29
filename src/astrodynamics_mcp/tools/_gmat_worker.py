"""Out-of-process execution of gmatpy for the in-process GMAT tools.

``gmat_run_mission``, ``gmat_execute_script``, and ``gmat_validate_script``
cannot run gmatpy inside the server process: gmat-run bootstraps a single
global Moderator that ``LoadScript`` / ``RunScript`` / ``Clear`` all mutate
(so two runs in one interpreter corrupt each other's resource graph), and
``RunScript`` blocks in C++ holding the GIL (so a worker thread neither frees
the event loop nor delivers parallelism). The fix is to run each load+run in
its own *fresh* interpreter — the same isolation model ``gmat-sweep`` uses.

This module owns that boundary:

* The picklable :class:`GmatSpec` (parent → worker) and the picklable result
  payloads (:class:`ResultSnapshot`, :class:`ValidateSnapshot`, wrapped in
  :class:`WorkerResult`) the worker hands back.
* :func:`run_operation` — the engine work, executed in the worker interpreter.
  It is the *only* code that imports ``gmat_run``; every ``gmat_run`` import is
  lazy and inside a function so the server process can import this module
  (for the dataclasses and the dispatcher) without ever bootstrapping gmatpy.
* :func:`main` — the ``python -m astrodynamics_mcp.tools._gmat_worker`` entry
  point: read a spec, run it, write the result, exit.
* :func:`dispatch_subprocess` — the async parent-side dispatcher: spawn a fresh
  interpreter per call, bound by a semaphore, killed on a wall-clock timeout.

The server process stays gmatpy-free; multiple runs proceed in isolated
interpreters; and a GMAT run never blocks the asyncio event loop.
"""

from __future__ import annotations

import asyncio
import os
import pickle
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Wire types — must be picklable (plain dataclasses, no gmat_run objects).
# ---------------------------------------------------------------------------

# Operation the worker performs. "run" and "execute" both load + run a mission
# (run also captures the structured summary and the report DataFrames); only
# "validate" takes the load-time log-capture path.
Operation = Literal["run", "execute", "validate"]

# WorkerResult.status — how the parent reshapes the payload back into a typed
# response or a typed error. "timeout" / "crashed" are produced by the
# dispatcher, never by the worker itself.
Status = Literal[
    "run_ok",
    "execute_ok",
    "validate_ok",
    "load_error",
    "run_error",
    "field_error",
    "bootstrap_error",
    "timeout",
    "crashed",
]


@dataclass
class GmatSpec:
    """One unit of GMAT work, shipped from the parent to a worker interpreter.

    ``script_path`` is resolved on the shared filesystem by the parent (inline
    text is already written to a temp file); ``workspace`` is the parent-owned
    output directory the run writes into so the parent can read artefacts back
    after the worker exits.
    """

    operation: Operation
    script_path: str
    overrides: dict[str, Any] = field(default_factory=dict)
    workspace: str | None = None


@dataclass
class CommandData:
    """Picklable mirror of a ``gmat_run`` command outline node."""

    type_name: str
    summary: str
    children: list[CommandData] = field(default_factory=list)
    nested_count: int = 0


@dataclass
class ResourceGroupData:
    """Picklable mirror of a ``gmat_run`` summary resource group."""

    category: str
    names: list[str] = field(default_factory=list)


@dataclass
class SummaryData:
    """Picklable mirror of ``gmat_run.summary.MissionSummary``.

    Exposes the same attribute shape the response shapers in ``tools/gmat.py``
    duck-type on (``script_name`` / ``resource_groups`` / ``commands``), so the
    parent's ``_render_mission_summary`` consumes it unchanged.
    """

    script_name: str
    resource_groups: list[ResourceGroupData] = field(default_factory=list)
    commands: list[CommandData] = field(default_factory=list)


@dataclass
class ResultSnapshot:
    """A finished ``mission.run`` reshaped into picklable, gmatpy-free data.

    ``reports`` carries the parsed ReportFile frames (pandas DataFrames) only
    for the ``run`` operation — ``execute`` reads its report text straight off
    disk via ``report_paths``, so it leaves ``reports`` empty. The four
    ``*_paths`` maps and ``output_dir`` let the parent walk and register the
    artefacts the run wrote into the shared workspace.
    """

    report_paths: dict[str, str] = field(default_factory=dict)
    ephemeris_paths: dict[str, str] = field(default_factory=dict)
    contact_paths: dict[str, str] = field(default_factory=dict)
    solver_paths: dict[str, str] = field(default_factory=dict)
    converged: dict[str, bool] = field(default_factory=dict)
    log: str = ""
    output_dir: str = ""
    summary: SummaryData | None = None
    reports: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidateSnapshot:
    """The load-time validation result captured in the worker interpreter."""

    load_ok: bool
    init_error: str | None
    raw_log: str
    summary: SummaryData | None = None


@dataclass
class WorkerResult:
    """Tagged payload the parent maps back to a typed response or error."""

    status: Status
    snapshot: ResultSnapshot | None = None
    validate: ValidateSnapshot | None = None
    message: str = ""
    log: str = ""
    path: str = ""


# ---------------------------------------------------------------------------
# Worker-side engine work (imports gmat_run lazily — runs in a fresh process).
# ---------------------------------------------------------------------------


def _serialize_command(cmd: Any) -> CommandData:
    return CommandData(
        type_name=cmd.type_name,
        summary=cmd.summary,
        children=[_serialize_command(child) for child in cmd.children],
        nested_count=cmd.nested_count,
    )


def _serialize_summary(summary: Any) -> SummaryData:
    """Reshape a ``gmat_run`` MissionSummary into the picklable mirror."""
    return SummaryData(
        script_name=summary.script_name,
        resource_groups=[
            ResourceGroupData(category=group.category, names=list(group.names))
            for group in summary.resource_groups
        ],
        commands=[_serialize_command(cmd) for cmd in summary.commands],
    )


def _snapshot_result(result: Any, mission: Any, *, operation: Operation) -> ResultSnapshot:
    """Capture everything the parent needs from a finished run, gmatpy-free."""
    report_paths = {str(k): str(v) for k, v in getattr(result, "report_paths", {}).items()}
    # Only the curated `run` tool needs the parsed frames (it shapes rows into
    # the response); `execute` reads its report text off disk. Materialise the
    # frames here, in the worker, before it exits — accessing them lazily reads
    # files under the workspace, which the parent keeps alive.
    reports: dict[str, Any] = {}
    summary: SummaryData | None = None
    if operation == "run":
        reports = {name: result.reports[name] for name in report_paths}
        summary = _serialize_summary(mission.summary())
    return ResultSnapshot(
        report_paths=report_paths,
        ephemeris_paths={str(k): str(v) for k, v in getattr(result, "ephemeris_paths", {}).items()},
        contact_paths={str(k): str(v) for k, v in getattr(result, "contact_paths", {}).items()},
        solver_paths={str(k): str(v) for k, v in getattr(result, "solver_paths", {}).items()},
        converged=dict(getattr(result, "converged", {})),
        log=getattr(result, "log", "") or "",
        output_dir=str(getattr(result, "output_dir", "") or ""),
        summary=summary,
        reports=reports,
    )


def _run_mission(spec: GmatSpec) -> WorkerResult:
    """Load + (override) + run a mission; classify failures into a status."""
    from gmat_run import Mission
    from gmat_run.errors import GmatError, GmatFieldError, GmatLoadError, GmatRunError

    script_path = Path(spec.script_path)
    try:
        mission = Mission.load(script_path)
    except GmatLoadError as exc:
        return WorkerResult(status="load_error", message=str(exc))
    except GmatError as exc:
        return WorkerResult(status="bootstrap_error", message=str(exc))

    for dotted, value in (spec.overrides or {}).items():
        try:
            mission[dotted] = value
        except GmatFieldError as exc:
            return WorkerResult(status="field_error", message=str(exc), path=dotted)
        except (TypeError, AttributeError) as exc:
            # gmat-run's assignment surface raises TypeError / AttributeError
            # for some bad-shape inputs that don't go through GmatFieldError.
            return WorkerResult(status="field_error", message=str(exc), path=dotted)

    try:
        result = mission.run(working_dir=Path(spec.workspace) if spec.workspace else None)
    except GmatRunError as exc:
        return WorkerResult(status="run_error", message=str(exc), log=getattr(exc, "log", "") or "")

    snapshot = _snapshot_result(result, mission, operation=spec.operation)
    if not snapshot.output_dir and spec.workspace:
        snapshot.output_dir = spec.workspace
    return WorkerResult(
        status="run_ok" if spec.operation == "run" else "execute_ok",
        snapshot=snapshot,
    )


def _run_validate(spec: GmatSpec) -> WorkerResult:
    """Capture GMAT's load-time log + summary for ``gmat_validate_script``.

    Bypasses ``Mission.load`` to capture the GMAT log across the parse step
    (``Mission.load`` only redirects ``UseLogFile`` during ``run()``). Reaches
    into gmat-run private helpers intentionally; once upstream ships
    ``Mission.validate()`` this collapses to a single call into that helper.
    """
    from contextlib import suppress

    script_path = Path(spec.script_path)
    try:
        from gmat_run.install import locate_gmat
        from gmat_run.mission import _get_api_exception, _initialize_spacecraft
        from gmat_run.runtime import bootstrap
        from gmat_run.summary import build_mission_summary

        install = locate_gmat()
        gmat = bootstrap(install)
    except Exception as exc:
        # Any discovery / import / bootstrap failure is surfaced to the parent
        # as a status rather than a traceback.
        return WorkerResult(status="bootstrap_error", message=str(exc))

    # Restore the install's conventional log target after the parse so the
    # MessageInterface stops holding the temp log open before it is unlinked.
    default_log_path = install.bin_dir.parent / "output" / "GmatLog.txt"

    init_error: str | None = None
    load_ok = False
    with tempfile.TemporaryDirectory(prefix="astrodynamics-mcp-validate-") as tmp:
        log_path = Path(tmp) / "validate.log"
        # Fresh sandbox so a prior load can't leak resources into the summary.
        with suppress(Exception):
            gmat.Clear()
        gmat.UseLogFile(str(log_path))
        try:
            load_ok = bool(gmat.LoadScript(str(script_path)))
            if load_ok:
                api_exception = _get_api_exception(gmat)
                try:
                    _initialize_spacecraft(gmat)
                except api_exception as exc:
                    init_error = f"{type(exc).__name__}: {exc}"
        finally:
            with suppress(Exception):
                gmat.UseLogFile(str(default_log_path))
        raw_log = log_path.read_text(encoding="utf-8", errors="replace")

    ok = load_ok and init_error is None
    summary: SummaryData | None = None
    if ok:
        summary = _serialize_summary(build_mission_summary(gmat, script_path))
    return WorkerResult(
        status="validate_ok",
        validate=ValidateSnapshot(
            load_ok=load_ok,
            init_error=init_error,
            raw_log=raw_log,
            summary=summary,
        ),
    )


def run_operation(spec: GmatSpec) -> WorkerResult:
    """Execute one :class:`GmatSpec` in the current interpreter.

    The single entry point for both the subprocess :func:`main` and the
    in-process dispatch the unit tests install. Every ``gmat_run`` import lives
    below this call, so importing this module never bootstraps gmatpy.
    """
    if spec.operation == "validate":
        return _run_validate(spec)
    return _run_mission(spec)


# ---------------------------------------------------------------------------
# Subprocess entry point.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Read a pickled spec, run it, write a pickled result. Exit 0 on success.

    Invoked as ``python -m astrodynamics_mcp.tools._gmat_worker <spec> <out>``.
    Handled GMAT failures travel back as :class:`WorkerResult` statuses (exit
    0); only an unwritable result path or a malformed invocation exits nonzero,
    which the dispatcher maps to a ``crashed`` status.
    """
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: _gmat_worker <spec-path> <result-path>", file=sys.stderr)
        return 2
    spec_path, result_path = args
    with open(spec_path, "rb") as fh:
        spec = pickle.load(fh)
    result = run_operation(spec)
    with open(result_path, "wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return 0


# ---------------------------------------------------------------------------
# Parent-side async dispatcher.
# ---------------------------------------------------------------------------

# Concurrency cap on in-flight GMAT runs (each spawns a gmatpy interpreter, so
# memory- and CPU-heavy). Default min(4, cpu_count); override with the env var.
_WORKERS_ENV = "ASTRODYNAMICS_MCP_GMAT_WORKERS"
# Per-call wall-clock cap in seconds; subprocess isolation makes a hard kill
# possible. Generous default for solver-heavy missions; override with the env.
_TIMEOUT_ENV = "ASTRODYNAMICS_MCP_GMAT_TIMEOUT"
_DEFAULT_TIMEOUT_S = 600.0

_semaphore: asyncio.Semaphore | None = None
_semaphore_loop: asyncio.AbstractEventLoop | None = None


def _worker_count() -> int:
    raw = os.environ.get(_WORKERS_ENV)
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value >= 1:
            return value
    return min(4, os.cpu_count() or 1)


def _timeout_seconds() -> float | None:
    raw = os.environ.get(_TIMEOUT_ENV)
    if raw:
        try:
            value = float(raw)
        except ValueError:
            return _DEFAULT_TIMEOUT_S
        # A non-positive override disables the wall-clock cap entirely.
        return value if value > 0 else None
    return _DEFAULT_TIMEOUT_S


def _worker_command(spec_path: str, result_path: str) -> list[str]:
    """Build the argv that runs one spec in a fresh interpreter.

    Factored out so tests can substitute a controllable subprocess (a sleeper
    for the timeout branch, a nonzero exit for the crash branch) without a
    GMAT install.
    """
    return [
        sys.executable,
        "-m",
        "astrodynamics_mcp.tools._gmat_worker",
        spec_path,
        result_path,
    ]


def _get_semaphore() -> asyncio.Semaphore:
    """Lazily bind the concurrency semaphore to the running event loop.

    An :class:`asyncio.Semaphore` binds to whichever loop first awaits it;
    test harnesses spin a fresh loop per test, so rebuild it when the running
    loop changes rather than reusing one bound to a closed loop.
    """
    global _semaphore, _semaphore_loop
    running = asyncio.get_running_loop()
    if _semaphore is None or _semaphore_loop is not running:
        _semaphore = asyncio.Semaphore(_worker_count())
        _semaphore_loop = running
    return _semaphore


async def dispatch_subprocess(spec: GmatSpec) -> WorkerResult:
    """Run one spec in a fresh interpreter, bounded and wall-clock-capped.

    Spawns ``python -m astrodynamics_mcp.tools._gmat_worker`` so gmatpy loads
    in a throwaway process (a fresh, uncorrupted Moderator every time) while
    the event loop stays free. A run that overruns the timeout is killed and
    reported as ``status="timeout"``; a crashed worker (segfault, nonzero exit)
    is reported as ``status="crashed"`` with its captured stderr.
    """
    semaphore = _get_semaphore()
    timeout = _timeout_seconds()
    async with semaphore:
        with tempfile.TemporaryDirectory(prefix="astrodynamics-mcp-worker-") as tmp:
            spec_path = Path(tmp) / "spec.pkl"
            result_path = Path(tmp) / "result.pkl"
            with open(spec_path, "wb") as fh:
                pickle.dump(spec, fh, protocol=pickle.HIGHEST_PROTOCOL)

            command = _worker_command(str(spec_path), str(result_path))
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except (asyncio.TimeoutError, TimeoutError):
                proc.kill()
                await proc.wait()
                return WorkerResult(
                    status="timeout",
                    message=(
                        f"GMAT worker exceeded the {timeout:.0f}s wall-clock limit and was "
                        f"killed; raise {_TIMEOUT_ENV} for longer-running missions"
                    ),
                )

            if proc.returncode != 0 or not result_path.is_file():
                detail = stderr.decode("utf-8", errors="replace").strip()
                return WorkerResult(
                    status="crashed",
                    message=(
                        f"GMAT worker exited with code {proc.returncode} without a result"
                        + (f"; stderr tail: {detail[-2000:]}" if detail else "")
                    ),
                )

            with open(result_path, "rb") as fh:
                loaded: WorkerResult = pickle.load(fh)
            return loaded


if __name__ == "__main__":
    # ``python -m astrodynamics_mcp.tools._gmat_worker`` loads this file as the
    # ``__main__`` module, so classes defined here would pickle under
    # ``__main__.<name>`` — which the parent (which imported the module by its
    # canonical path) cannot resolve. Delegate to the canonically-imported
    # module so the result payload pickles with the dotted module path the
    # parent unpickles against.
    from astrodynamics_mcp.tools._gmat_worker import main as _canonical_main

    raise SystemExit(_canonical_main())
