"""Tests for the out-of-process GMAT execution layer (``_gmat_worker``).

Three layers, none needing a GMAT install:

1. :class:`TestErrorMapping` — the parent-side ``_raise_for_worker_failure``
   maps every worker error status to the documented typed error + code.
2. :class:`TestDispatcher` — ``dispatch_subprocess`` spawns a fresh
   interpreter, enforces the wall-clock timeout (kill → ``timeout``), and
   surfaces a nonzero-exit worker as ``crashed``. The worker command is
   swapped for a controllable stand-in so these branches run without GMAT.
3. :class:`TestConfig` / :class:`TestLockRetired` — env-knob parsing and the
   invariant that the obsolete in-process Moderator lock is gone.

The happy path (a real spec round-tripping through the spawned interpreter)
is covered end-to-end by the ``@pytest.mark.gmat_installed`` integration
tests in ``test_tool_gmat_*.py``.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from astrodynamics_mcp.errors import InvalidInputError, UpstreamError
from astrodynamics_mcp.tools import _gmat_worker
from astrodynamics_mcp.tools import gmat as gmat_tools


class TestErrorMapping:
    """``_raise_for_worker_failure`` turns a worker status into a typed error."""

    def test_load_error_maps_to_upstream_load_failed(self) -> None:
        result = _gmat_worker.WorkerResult(status="load_error", message="bad parse")
        with pytest.raises(UpstreamError) as excinfo:
            gmat_tools._raise_for_worker_failure(result)
        assert excinfo.value.code == "upstream.gmat_run_load_failed"

    def test_bootstrap_error_maps_to_upstream_bootstrap_failed(self) -> None:
        result = _gmat_worker.WorkerResult(status="bootstrap_error", message="no install")
        with pytest.raises(UpstreamError) as excinfo:
            gmat_tools._raise_for_worker_failure(result)
        assert excinfo.value.code == "upstream.gmat_run_bootstrap_failed"

    def test_field_error_maps_to_invalid_input(self) -> None:
        result = _gmat_worker.WorkerResult(status="field_error", message="nope", path="Sat.SMA")
        with pytest.raises(InvalidInputError) as excinfo:
            gmat_tools._raise_for_worker_failure(result)
        assert excinfo.value.code == "invalid_input.gmat_override_failed"
        assert excinfo.value.data["path"] == "Sat.SMA"

    def test_run_error_maps_to_upstream_run_failed(self) -> None:
        result = _gmat_worker.WorkerResult(status="run_error", message="RunScript -1", log="boom")
        with pytest.raises(UpstreamError) as excinfo:
            gmat_tools._raise_for_worker_failure(result)
        assert excinfo.value.code == "upstream.gmat_run_failed"

    def test_timeout_maps_to_upstream_timeout(self) -> None:
        result = _gmat_worker.WorkerResult(status="timeout", message="killed at 600s")
        with pytest.raises(UpstreamError) as excinfo:
            gmat_tools._raise_for_worker_failure(result)
        assert excinfo.value.code == "upstream.gmat_run_timeout"

    def test_crashed_maps_to_upstream_worker_crashed(self) -> None:
        result = _gmat_worker.WorkerResult(status="crashed", message="exit 139 (SIGSEGV)")
        with pytest.raises(UpstreamError) as excinfo:
            gmat_tools._raise_for_worker_failure(result)
        assert excinfo.value.code == "upstream.gmat_worker_crashed"

    def test_success_statuses_do_not_raise(self) -> None:
        for status in ("run_ok", "execute_ok", "validate_ok"):
            gmat_tools._raise_for_worker_failure(_gmat_worker.WorkerResult(status=status))


class TestDispatcher:
    """``dispatch_subprocess`` spawns, times out, and detects crashes."""

    @pytest.fixture(autouse=True)
    def _reset_semaphore(self) -> None:
        # The module caches the concurrency semaphore per loop; each async test
        # gets a fresh loop, and _get_semaphore rebinds, so no teardown needed.
        return None

    async def test_timeout_kills_worker_and_reports_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A worker that sleeps well past the cap; the dispatcher must kill it.
        monkeypatch.setenv("ASTRODYNAMICS_MCP_GMAT_TIMEOUT", "0.3")

        def _sleeper(spec_path: str, result_path: str) -> list[str]:
            import sys

            return [sys.executable, "-c", "import time; time.sleep(30)"]

        monkeypatch.setattr(_gmat_worker, "_worker_command", _sleeper)
        spec = _gmat_worker.GmatSpec(operation="run", script_path="/tmp/x.script")
        result = await _gmat_worker.dispatch_subprocess(spec)
        assert result.status == "timeout"
        assert "0" in result.message  # mentions the limit

    async def test_nonzero_exit_reports_crashed_with_stderr(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _crasher(spec_path: str, result_path: str) -> list[str]:
            import sys

            return [
                sys.executable,
                "-c",
                "import sys; sys.stderr.write('kaboom'); sys.exit(3)",
            ]

        monkeypatch.setattr(_gmat_worker, "_worker_command", _crasher)
        spec = _gmat_worker.GmatSpec(operation="validate", script_path="/tmp/x.script")
        result = await _gmat_worker.dispatch_subprocess(spec)
        assert result.status == "crashed"
        assert "kaboom" in result.message

    async def test_clean_exit_without_result_file_is_crashed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Exit 0 but never write the result payload — still a crash from the
        # parent's perspective (no usable result).
        def _silent(spec_path: str, result_path: str) -> list[str]:
            import sys

            return [sys.executable, "-c", "pass"]

        monkeypatch.setattr(_gmat_worker, "_worker_command", _silent)
        spec = _gmat_worker.GmatSpec(operation="run", script_path="/tmp/x.script")
        result = await _gmat_worker.dispatch_subprocess(spec)
        assert result.status == "crashed"

    async def test_dispatch_reads_back_result_pickle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Exercise the success branch (returncode 0 + result file present →
        # unpickle and return). The stand-in worker pickles a lightweight
        # stdlib object rather than importing astrodynamics_mcp, so the
        # subprocess starts in ~0.1s instead of paying the full package
        # import (~12s) — dispatch_subprocess just pickle.loads whatever the
        # worker wrote, so the object's concrete type is immaterial here.
        def _producer(spec_path: str, result_path: str) -> list[str]:
            import sys

            code = (
                "import pickle, sys, types\n"
                "res = types.SimpleNamespace(status='execute_ok', "
                "snapshot=types.SimpleNamespace(log='done'))\n"
                "open(sys.argv[1], 'wb').write(pickle.dumps(res))\n"
            )
            return [sys.executable, "-c", code, result_path]

        monkeypatch.setattr(_gmat_worker, "_worker_command", _producer)
        spec = _gmat_worker.GmatSpec(operation="execute", script_path="/tmp/x.script")
        result = await _gmat_worker.dispatch_subprocess(spec)
        assert result.status == "execute_ok"
        assert result.snapshot is not None
        assert result.snapshot.log == "done"


class TestWorkerMain:
    """``main`` reads a spec, runs the operation, and writes a result pickle."""

    def test_main_runs_spec_and_writes_result(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Inject a fake gmat_run (overriding any real install) so main() is
        # deterministic regardless of whether GMAT is present on the box.
        from tests._gmat_helpers import install_minimal_gmat_run_modules

        class _LoadError(Exception):
            pass

        install_minimal_gmat_run_modules(monkeypatch, load_error=_LoadError)

        gmat_run_mod = ModuleType("gmat_run")

        class _Factory:
            @staticmethod
            def load(_path: Any) -> Any:
                raise _LoadError("synthetic bad script")

        gmat_run_mod.__dict__["Mission"] = _Factory
        monkeypatch.setitem(sys.modules, "gmat_run", gmat_run_mod)

        spec = _gmat_worker.GmatSpec(operation="run", script_path=str(tmp_path / "x.script"))
        spec_path = tmp_path / "spec.pkl"
        result_path = tmp_path / "result.pkl"
        spec_path.write_bytes(pickle.dumps(spec))

        rc = _gmat_worker.main([str(spec_path), str(result_path)])
        assert rc == 0
        loaded = pickle.loads(result_path.read_bytes())
        assert isinstance(loaded, _gmat_worker.WorkerResult)
        assert loaded.status == "load_error"

    def test_main_bad_argv_returns_nonzero(self) -> None:
        assert _gmat_worker.main(["only-one-arg"]) == 2


class TestResolveTimeout:
    """``_resolve_timeout`` validates and clamps a caller-supplied cap."""

    def test_none_defers_to_env(self) -> None:
        assert gmat_tools._resolve_timeout(None) is None

    def test_positive_below_ceiling_is_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ASTRODYNAMICS_MCP_GMAT_TIMEOUT", "600")
        assert gmat_tools._resolve_timeout(120) == 120.0

    def test_above_ceiling_is_clamped_down(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ASTRODYNAMICS_MCP_GMAT_TIMEOUT", "600")
        assert gmat_tools._resolve_timeout(5000) == 600.0

    def test_non_positive_is_rejected(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            gmat_tools._resolve_timeout(0)
        assert excinfo.value.code == "invalid_input.gmat_timeout_not_positive"

    def test_disabled_cap_honors_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Operator disabled the cap (<=0) → no ceiling → honor the request.
        monkeypatch.setenv("ASTRODYNAMICS_MCP_GMAT_TIMEOUT", "0")
        assert gmat_tools._resolve_timeout(5000) == 5000.0


class TestConfig:
    """Env knobs parse with sensible fallbacks."""

    def test_worker_count_defaults_to_min_four(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ASTRODYNAMICS_MCP_GMAT_WORKERS", raising=False)
        assert _gmat_worker._worker_count() == min(4, __import__("os").cpu_count() or 1)

    def test_worker_count_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ASTRODYNAMICS_MCP_GMAT_WORKERS", "7")
        assert _gmat_worker._worker_count() == 7

    def test_worker_count_garbage_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ASTRODYNAMICS_MCP_GMAT_WORKERS", "not-a-number")
        assert _gmat_worker._worker_count() == min(4, __import__("os").cpu_count() or 1)

    def test_timeout_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ASTRODYNAMICS_MCP_GMAT_TIMEOUT", raising=False)
        assert _gmat_worker._timeout_seconds() == _gmat_worker._DEFAULT_TIMEOUT_S

    def test_timeout_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ASTRODYNAMICS_MCP_GMAT_TIMEOUT", "12.5")
        assert _gmat_worker._timeout_seconds() == 12.5

    def test_non_positive_timeout_disables_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ASTRODYNAMICS_MCP_GMAT_TIMEOUT", "0")
        assert _gmat_worker._timeout_seconds() is None

    def test_worker_command_targets_this_module(self) -> None:
        cmd = _gmat_worker._worker_command("/tmp/spec.pkl", "/tmp/result.pkl")
        assert cmd[1:] == [
            "-m",
            "astrodynamics_mcp.tools._gmat_worker",
            "/tmp/spec.pkl",
            "/tmp/result.pkl",
        ]


class TestLockRetired:
    """The obsolete in-process Moderator lock is gone after process isolation."""

    def test_global_lock_removed(self) -> None:
        assert not hasattr(gmat_tools, "_GMAT_GLOBAL_LOCK")

    def test_dispatch_seam_present(self) -> None:
        # The handlers route through this module-level seam; production points
        # it at the subprocess dispatcher.
        assert gmat_tools._dispatch_worker is _gmat_worker.dispatch_subprocess
