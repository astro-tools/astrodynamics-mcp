"""Shared helpers for the GMAT test suite.

Two pieces:

* :func:`make_fresh_mcp` — the per-test FastMCP + monkeypatch dance that
  every ``tests/test_tool_gmat_*.py`` file copies under its own local
  ``_fresh_mcp(monkeypatch)``. The existing call sites delegate here for
  the shared body; new tests can use the :func:`gmat_mcp` fixture in
  ``conftest.py`` instead.

* :func:`install_minimal_gmat_run_modules` — the minimal set of fake
  ``gmat_run.*`` submodule injections (``gmat_run.errors`` carrying the
  typed exception classes) that's identical across the per-tool fakes in
  the test files. Per-tool ``_FakeMission`` / ``_FakeResult`` stay
  per-file because they shape behaviour around each tool's specific
  contract.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest
from mcp.server.fastmcp import FastMCP

from astrodynamics_mcp.tools import _gmat_worker
from astrodynamics_mcp.tools import gmat as gmat_tools


async def _in_process_dispatch(
    spec: _gmat_worker.GmatSpec, *, timeout_override: float | None = None
) -> _gmat_worker.WorkerResult:
    """Run the worker body in-process instead of spawning a subprocess.

    The three GMAT tools run gmatpy out-of-process in production. The unit
    suite drives them with a fake ``gmat_run`` injected into ``sys.modules``;
    a real subprocess would not see that fake. So every ``make_fresh_mcp``
    test flips the dispatch seam to call :func:`_gmat_worker.run_operation`
    directly — the worker body then imports the injected fake just as it would
    import the real engine in a worker interpreter.

    ``timeout_override`` is accepted (the handlers pass it) but ignored: the
    in-process path has no subprocess to time out.
    """
    del timeout_override
    return _gmat_worker.run_operation(spec)


def make_fresh_mcp(
    name: str, monkeypatch: pytest.MonkeyPatch, *, register_tools: bool = True
) -> FastMCP:
    """Stand up a fresh FastMCP + monkeypatch the singleton; optionally register tools.

    All ``test_tool_gmat_*.py`` files used to inline an identical
    five-line copy of this body; they now delegate here. Pass
    ``register_tools=False`` for tests that exercise the registration
    helpers themselves.

    Also flips the GMAT execution seam to in-process dispatch (see
    :func:`_in_process_dispatch`) so the injected fake ``gmat_run`` drives the
    worker body without spawning a real interpreter.
    """
    fresh = FastMCP(name)
    monkeypatch.setattr("astrodynamics_mcp.server.mcp", fresh)
    monkeypatch.setattr(gmat_tools, "_GMAT_RUN_AVAILABLE", True)
    monkeypatch.setattr(gmat_tools, "_dispatch_worker", _in_process_dispatch)
    if register_tools:
        gmat_tools._register_gmat_tools()
    return fresh


def install_minimal_gmat_run_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    field_error: type[Exception] | None = None,
    load_error: type[Exception] | None = None,
    run_error: type[Exception] | None = None,
    discovery_error: type[Exception] | None = None,
) -> None:
    """Inject a fake ``gmat_run.errors`` module carrying the typed exceptions.

    The tool bodies do ``from gmat_run.errors import GmatLoadError,
    GmatRunError, GmatError, GmatFieldError`` at call time; this helper
    builds a module with the requested subset and inserts it into
    :data:`sys.modules` for the duration of the test. Each test gets to
    supply its own exception subclasses (so ``isinstance`` checks against
    the fake propagate cleanly) -- pass ``None`` to use the default
    placeholder.
    """
    errors_mod = ModuleType("gmat_run.errors")

    class _DefaultGmatError(Exception):
        pass

    class _DefaultGmatLoadError(_DefaultGmatError):
        pass

    class _DefaultGmatRunError(_DefaultGmatError):
        def __init__(self, msg: str = "", *, log: str = "") -> None:
            super().__init__(msg)
            self.log = log

    class _DefaultGmatFieldError(Exception):
        pass

    errors_mod.GmatError = discovery_error or _DefaultGmatError  # type: ignore[attr-defined]
    errors_mod.GmatLoadError = load_error or _DefaultGmatLoadError  # type: ignore[attr-defined]
    errors_mod.GmatRunError = run_error or _DefaultGmatRunError  # type: ignore[attr-defined]
    errors_mod.GmatFieldError = field_error or _DefaultGmatFieldError  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "gmat_run.errors", errors_mod)
