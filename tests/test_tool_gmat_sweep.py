"""Tests for the ``gmat_sweep`` tool.

Unit tests drive the tool with fakes for the ``gmat_sweep`` and
``pandas`` modules injected via :mod:`sys.modules` and the registered
slot on a per-test :class:`FastMCP` instance. The integration test
exercises the real backend through the ``[gmat]`` extra and is gated
on the ``gmat_installed`` marker so contributors without GMAT still
get green CI on the rest of the suite.
"""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from astrodynamics_mcp.server_lint import check_tool_descriptions
from astrodynamics_mcp.tools import gmat as gmat_tools
from astrodynamics_mcp.tools.gmat import (
    GmatSweepResponse,
    _build_sweep_response,
    _coerce_perturb,
    _frame_rows,
    _numeric_column_stats,
    _status_counts,
    _validate_sweep_payload,
)

# ---------------------------------------------------------------------------
# Fake gmat_sweep + pandas surfaces
# ---------------------------------------------------------------------------


class _FakeSweepConfigError(Exception):
    """Stand-in for ``gmat_sweep.errors.SweepConfigError``."""


class _FakeSeries:
    """Thin numpy-backed stand-in for the slice of pandas.Series we touch."""

    def __init__(self, values: list[Any]) -> None:
        self._values = list(values)

    def __eq__(self, other: object) -> _FakeBoolMask:  # type: ignore[override]
        return _FakeBoolMask([v == other for v in self._values])

    def astype(self, _dtype: Any) -> _FakeSeries:
        return _FakeSeries([float(v) for v in self._values])

    def to_numpy(self, *, dtype: Any = None) -> np.ndarray:
        del dtype
        return np.array(self._values, dtype=float)


class _FakeBoolMask:
    """Bool-indexing mask + ``.sum()`` for ``frame['__status'] == 'ok'`` cases."""

    def __init__(self, values: list[bool]) -> None:
        self._values = list(values)

    def sum(self) -> int:
        return int(sum(self._values))

    def __iter__(self) -> Iterator[bool]:
        return iter(self._values)


class _FakeSubFrame:
    """Sub-frame returned by ``frame[list_of_cols]`` for the ``.to_numpy`` path."""

    def __init__(self, rows: list[list[Any]]) -> None:
        self._rows = [list(r) for r in rows]

    def to_numpy(self, *, dtype: Any = None) -> Any:
        del dtype
        return self._rows


class _FakeFrame:
    """Minimal pandas-DataFrame stand-in for the sweep-shaping path.

    Columns are stored as parallel lists; the MultiIndex is a list of
    ``(run_id, time)`` tuples. Supports column access, bool-mask
    filtering, and the sub-frame ``.to_numpy`` slice — enough surface
    for ``_status_counts``, ``_numeric_column_stats``, and
    ``_frame_rows`` to operate.
    """

    def __init__(
        self,
        columns: list[str],
        index: list[tuple[Any, Any]],
        data: dict[str, list[Any]],
    ) -> None:
        self.columns = list(columns)
        self.index = list(index)
        self._data = {col: list(data[col]) for col in columns}

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return _FakeSeries(self._data[key])
        if isinstance(key, list):
            rows: list[list[Any]] = []
            for i in range(len(self.index)):
                rows.append([self._data[col][i] for col in key])
            return _FakeSubFrame(rows)
        if isinstance(key, _FakeBoolMask):
            keep = list(key)
            new_index = [self.index[i] for i in range(len(self.index)) if keep[i]]
            new_data = {
                col: [self._data[col][i] for i in range(len(self.index)) if keep[i]]
                for col in self.columns
            }
            return _FakeFrame(self.columns, new_index, new_data)
        raise KeyError(f"unsupported key: {key!r}")


class _FakeLocalJoblibPool:
    """No-op stand-in for ``gmat_sweep.backends.joblib.LocalJoblibPool``."""

    def __init__(self, *, max_workers: int = 1) -> None:
        self.max_workers = max_workers


class _FakeSweepCalls:
    """Recorder for the fake backend functions; one instance per fresh mcp."""

    def __init__(self, result: _FakeFrame) -> None:
        self.result = result
        self.sweep_calls: list[dict[str, Any]] = []
        self.monte_carlo_calls: list[dict[str, Any]] = []
        self.latin_hypercube_calls: list[dict[str, Any]] = []
        self.raise_config_error: bool = False
        self.raise_unexpected: bool = False

    def sweep(self, script: Any, **kwargs: Any) -> _FakeFrame:
        if self.raise_config_error:
            raise _FakeSweepConfigError("bad config")
        if self.raise_unexpected:
            raise RuntimeError("boom from sweep")
        self.sweep_calls.append({"script": script, **kwargs})
        return self.result

    def monte_carlo(self, script: Any, **kwargs: Any) -> _FakeFrame:
        if self.raise_config_error:
            raise _FakeSweepConfigError("bad mc config")
        if self.raise_unexpected:
            raise RuntimeError("boom from monte_carlo")
        self.monte_carlo_calls.append({"script": script, **kwargs})
        return self.result

    def latin_hypercube(self, script: Any, **kwargs: Any) -> _FakeFrame:
        if self.raise_config_error:
            raise _FakeSweepConfigError("bad lh config")
        if self.raise_unexpected:
            raise RuntimeError("boom from latin_hypercube")
        self.latin_hypercube_calls.append({"script": script, **kwargs})
        return self.result


def _install_fake_gmat_sweep(monkeypatch: pytest.MonkeyPatch, calls: _FakeSweepCalls) -> None:
    """Inject fake ``gmat_sweep`` + ``pandas`` modules for one test.

    The tool body imports ``gmat_sweep``, ``gmat_sweep.backends.joblib``,
    and ``gmat_sweep.errors`` inside its function body, plus ``pandas`` in
    ``_samples_to_dataframe``. Faking all four is enough to drive the
    code path end-to-end without any real install.
    """
    fake = ModuleType("gmat_sweep")
    fake_errors = ModuleType("gmat_sweep.errors")
    fake_errors.__dict__["SweepConfigError"] = _FakeSweepConfigError
    fake_backends = ModuleType("gmat_sweep.backends")
    fake_backends_joblib = ModuleType("gmat_sweep.backends.joblib")
    fake_backends_joblib.__dict__["LocalJoblibPool"] = _FakeLocalJoblibPool

    fake.__dict__["sweep"] = calls.sweep
    fake.__dict__["monte_carlo"] = calls.monte_carlo
    fake.__dict__["latin_hypercube"] = calls.latin_hypercube
    fake.__dict__["errors"] = fake_errors
    fake.__dict__["backends"] = fake_backends

    monkeypatch.setitem(sys.modules, "gmat_sweep", fake)
    monkeypatch.setitem(sys.modules, "gmat_sweep.errors", fake_errors)
    monkeypatch.setitem(sys.modules, "gmat_sweep.backends", fake_backends)
    monkeypatch.setitem(sys.modules, "gmat_sweep.backends.joblib", fake_backends_joblib)

    fake_pandas = ModuleType("pandas")

    def _fake_dataframe(rows: list[list[Any]], columns: list[str]) -> _FakeFrame:
        # Build a (run_id-only) MultiIndex-ish list so the sweep call's
        # samples-mode payload is recognisable when the test asserts.
        index = [(i, None) for i in range(len(rows))]
        data = {col: [row[ci] for row in rows] for ci, col in enumerate(columns)}
        return _FakeFrame(columns, index, data)

    fake_pandas.__dict__["DataFrame"] = _fake_dataframe
    monkeypatch.setitem(sys.modules, "pandas", fake_pandas)


def _fresh_mcp(monkeypatch: pytest.MonkeyPatch) -> FastMCP:
    """Stand up a per-test FastMCP and re-register the GMAT slots against it."""
    fresh = FastMCP("gmat-sweep-test")
    monkeypatch.setattr("astrodynamics_mcp.server.mcp", fresh)
    monkeypatch.setattr(gmat_tools, "_GMAT_RUN_AVAILABLE", True)
    gmat_tools._register_gmat_tools()
    return fresh


def _trivial_result(rows: int = 3, with_status: bool = False) -> _FakeFrame:
    """Build a small ``_FakeFrame`` with numeric columns plus an optional ``__status``."""
    columns: list[str] = ["Sat.SMA", "Sat.X"]
    index = [(rid, float(t)) for rid in range(rows) for t in (0.0, 60.0)]
    sma: list[Any] = [7000.0 + 0.1 * i for i in range(len(index))]
    x: list[Any] = [1.0 + 0.5 * i for i in range(len(index))]
    data: dict[str, list[Any]] = {"Sat.SMA": sma, "Sat.X": x}
    if with_status:
        columns = [*columns, "__status"]
        data["__status"] = ["ok"] * len(index)
    return _FakeFrame(columns, index, data)


# ---------------------------------------------------------------------------
# Pure helpers — perturb / samples coercion, payload validation, shaping
# ---------------------------------------------------------------------------


class TestCoercePerturb:
    def test_normal_tuple_is_accepted(self) -> None:
        out = _coerce_perturb({"Sat.SMA": ["normal", 7000.0, 5.0]})
        assert out == {"Sat.SMA": ("normal", 7000.0, 5.0)}

    def test_uniform_and_lognormal_accepted(self) -> None:
        out = _coerce_perturb({"a": ["uniform", 0.0, 1.0], "b": ["lognormal", 0.0, 0.5]})
        assert out["a"][0] == "uniform"
        assert out["b"][0] == "lognormal"

    def test_empty_payload_rejected(self) -> None:
        from astrodynamics_mcp.errors import InvalidInputError

        with pytest.raises(InvalidInputError) as excinfo:
            _coerce_perturb({})
        assert excinfo.value.code == "invalid_input.gmat_sweep_perturb_empty"

    def test_unknown_tag_rejected(self) -> None:
        from astrodynamics_mcp.errors import InvalidInputError

        with pytest.raises(InvalidInputError) as excinfo:
            _coerce_perturb({"Sat.SMA": ["beta", 1.0, 2.0]})
        assert excinfo.value.code == "invalid_input.gmat_sweep_perturb_tag"

    def test_non_list_value_rejected(self) -> None:
        from astrodynamics_mcp.errors import InvalidInputError

        with pytest.raises(InvalidInputError) as excinfo:
            _coerce_perturb({"Sat.SMA": "normal"})
        assert excinfo.value.code == "invalid_input.gmat_sweep_perturb_shape"


class TestValidateSweepPayload:
    def test_grid_requires_grid(self) -> None:
        from astrodynamics_mcp.errors import InvalidInputError

        with pytest.raises(InvalidInputError) as excinfo:
            _validate_sweep_payload(
                mode="grid", grid=None, samples=None, perturb=None, n=None, seed=None
            )
        assert excinfo.value.code == "invalid_input.gmat_sweep_grid_required"

    def test_grid_rejects_other_payload(self) -> None:
        from astrodynamics_mcp.errors import InvalidInputError

        with pytest.raises(InvalidInputError) as excinfo:
            _validate_sweep_payload(
                mode="grid",
                grid={"x": [1]},
                samples=None,
                perturb={"y": ["normal", 0, 1]},
                n=None,
                seed=None,
            )
        assert excinfo.value.code == "invalid_input.gmat_sweep_mode_payload_conflict"

    def test_samples_requires_samples(self) -> None:
        from astrodynamics_mcp.errors import InvalidInputError

        with pytest.raises(InvalidInputError) as excinfo:
            _validate_sweep_payload(
                mode="samples", grid=None, samples=None, perturb=None, n=None, seed=None
            )
        assert excinfo.value.code == "invalid_input.gmat_sweep_samples_required"

    def test_monte_carlo_requires_perturb_and_n_and_seed(self) -> None:
        from astrodynamics_mcp.errors import InvalidInputError

        # Missing perturb.
        with pytest.raises(InvalidInputError) as excinfo:
            _validate_sweep_payload(
                mode="monte_carlo",
                grid=None,
                samples=None,
                perturb=None,
                n=5,
                seed=1,
            )
        assert excinfo.value.code == "invalid_input.gmat_sweep_perturb_required"

        # Missing n.
        with pytest.raises(InvalidInputError) as excinfo:
            _validate_sweep_payload(
                mode="monte_carlo",
                grid=None,
                samples=None,
                perturb={"x": ["normal", 0, 1]},
                n=None,
                seed=1,
            )
        assert excinfo.value.code == "invalid_input.gmat_sweep_n_required"

        # Missing seed.
        with pytest.raises(InvalidInputError) as excinfo:
            _validate_sweep_payload(
                mode="monte_carlo",
                grid=None,
                samples=None,
                perturb={"x": ["normal", 0, 1]},
                n=5,
                seed=None,
            )
        assert excinfo.value.code == "invalid_input.gmat_sweep_seed_required"

    def test_latin_hypercube_same_required_set(self) -> None:
        from astrodynamics_mcp.errors import InvalidInputError

        with pytest.raises(InvalidInputError) as excinfo:
            _validate_sweep_payload(
                mode="latin_hypercube",
                grid=None,
                samples=None,
                perturb={"x": ["normal", 0, 1]},
                n=5,
                seed=None,
            )
        assert excinfo.value.code == "invalid_input.gmat_sweep_seed_required"

    def test_well_formed_grid_passes(self) -> None:
        _validate_sweep_payload(
            mode="grid",
            grid={"x": [1, 2]},
            samples=None,
            perturb=None,
            n=None,
            seed=None,
        )


class TestStatusCounts:
    def test_no_status_column_counts_all_ok(self) -> None:
        frame = _trivial_result(rows=2, with_status=False)
        assert _status_counts(frame) == (4, 0, 0)

    def test_mixed_status_counts(self) -> None:
        index = [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]
        data: dict[str, list[Any]] = {
            "Sat.X": [1.0, float("nan"), float("nan"), 4.0],
            "__status": ["ok", "failed", "skipped", "ok"],
        }
        frame = _FakeFrame(["Sat.X", "__status"], index, data)
        assert _status_counts(frame) == (2, 1, 1)


class TestNumericColumnStats:
    def test_stats_over_ok_rows_only(self) -> None:
        index = [(0, 0.0), (1, 0.0), (2, 0.0)]
        data: dict[str, list[Any]] = {
            "Sat.X": [1.0, 2.0, 99.0],  # last row failed, should not contribute
            "__status": ["ok", "ok", "failed"],
        }
        frame = _FakeFrame(["Sat.X", "__status"], index, data)
        stats = _numeric_column_stats(frame)
        assert len(stats) == 1
        s = stats[0]
        assert s.column == "Sat.X"
        assert s.count.value == 2.0
        assert s.mean == pytest.approx(1.5)
        assert s.min == 1.0
        assert s.max == 2.0

    def test_single_row_std_is_nan(self) -> None:
        index = [(0, 0.0)]
        data = {"Sat.X": [7.0]}
        frame = _FakeFrame(["Sat.X"], index, data)
        stats = _numeric_column_stats(frame)
        assert len(stats) == 1
        assert np.isnan(stats[0].std)


class TestFrameRows:
    def test_rows_carry_run_id_time_and_data(self) -> None:
        frame = _trivial_result(rows=2, with_status=False)
        rows = _frame_rows(frame)
        assert len(rows) == 4
        assert rows[0]["run_id"] == 0.0
        assert rows[0]["time"] == 0.0
        assert rows[0]["Sat.SMA"] == pytest.approx(7000.0)

    def test_status_column_excluded_from_row_dicts(self) -> None:
        frame = _trivial_result(rows=1, with_status=True)
        rows = _frame_rows(frame)
        assert "__status" not in rows[0]


# ---------------------------------------------------------------------------
# Response shaping via _build_sweep_response
# ---------------------------------------------------------------------------


class TestBuildSweepResponse:
    def test_small_frame_inlines_head_and_tail_in_summary(self, tmp_path: Path) -> None:
        frame = _trivial_result(rows=2, with_status=True)  # 4 rows total
        response = _build_sweep_response(
            run_id="0" * 32,
            mode="grid",
            script_name="fixture.script",
            frame=frame,
            wall_clock_s=0.1,
            manifest_path=tmp_path / "manifest.jsonl",
            output_dir=tmp_path,
            output="summary",
        )
        assert response.truncated is False
        assert response.rows == []
        assert len(response.head) == 4
        assert len(response.tail) == 4
        assert response.run_count.value == 4.0
        assert response.status_counts.ok.value == 4.0
        assert {s.column for s in response.summary_stats} == {"Sat.SMA", "Sat.X"}

    def test_large_frame_truncates_in_summary_mode(self, tmp_path: Path) -> None:
        # 6 run_ids x 2 time-steps = 12 rows, above the 10-row inline threshold.
        frame = _trivial_result(rows=6, with_status=False)
        response = _build_sweep_response(
            run_id="0" * 32,
            mode="monte_carlo",
            script_name="fixture.script",
            frame=frame,
            wall_clock_s=0.2,
            manifest_path=tmp_path / "manifest.jsonl",
            output_dir=tmp_path,
            output="summary",
        )
        assert response.truncated is True
        assert response.rows == []
        assert len(response.head) == 5
        assert len(response.tail) == 5

    def test_full_mode_returns_every_row(self, tmp_path: Path) -> None:
        frame = _trivial_result(rows=6, with_status=False)
        response = _build_sweep_response(
            run_id="0" * 32,
            mode="latin_hypercube",
            script_name="fixture.script",
            frame=frame,
            wall_clock_s=0.3,
            manifest_path=tmp_path / "manifest.jsonl",
            output_dir=tmp_path,
            output="full",
        )
        assert response.truncated is False
        assert len(response.rows) == 12

    def test_mode_echoed(self, tmp_path: Path) -> None:
        frame = _trivial_result(rows=1, with_status=False)
        response = _build_sweep_response(
            run_id="0" * 32,
            mode="samples",
            script_name="fixture.script",
            frame=frame,
            wall_clock_s=0.0,
            manifest_path=tmp_path / "manifest.jsonl",
            output_dir=tmp_path,
            output="summary",
        )
        assert response.mode == "samples"
        assert response.manifest_path.endswith("manifest.jsonl")


# ---------------------------------------------------------------------------
# End-to-end call via FastMCP
# ---------------------------------------------------------------------------


class TestToolThroughMcp:
    async def test_grid_mode_dispatches_sweep(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls = _FakeSweepCalls(result=_trivial_result(rows=2, with_status=False))
        _install_fake_gmat_sweep(monkeypatch, calls)
        fresh = _fresh_mcp(monkeypatch)
        script = tmp_path / "fixture.script"
        script.write_text("% noop\n")

        _content, structured = await fresh.call_tool(
            "gmat_sweep",
            {
                "script": str(script),
                "mode": "grid",
                "grid": {"Sat.SMA": [7000, 7100]},
            },
        )
        parsed = GmatSweepResponse.model_validate(structured)
        assert parsed.mode == "grid"
        assert parsed.script_name == "fixture.script"
        assert len(calls.sweep_calls) == 1
        assert calls.sweep_calls[0]["grid"] == {"Sat.SMA": [7000, 7100]}
        # Default max_workers=1 path constructs a single-worker pool.
        backend = calls.sweep_calls[0]["backend"]
        assert isinstance(backend, _FakeLocalJoblibPool)
        assert backend.max_workers == 1

    async def test_samples_mode_builds_dataframe(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls = _FakeSweepCalls(result=_trivial_result(rows=1, with_status=False))
        _install_fake_gmat_sweep(monkeypatch, calls)
        fresh = _fresh_mcp(monkeypatch)
        script = tmp_path / "fixture.script"
        script.write_text("% noop\n")

        await fresh.call_tool(
            "gmat_sweep",
            {
                "script": str(script),
                "mode": "samples",
                "samples": [
                    {"Sat.SMA": 7000.0, "Sat.INC": 28.5},
                    {"Sat.SMA": 7100.0, "Sat.INC": 51.6},
                ],
            },
        )
        assert len(calls.sweep_calls) == 1
        samples_df = calls.sweep_calls[0]["samples"]
        # The fake pandas.DataFrame returned a _FakeFrame with the two rows.
        assert isinstance(samples_df, _FakeFrame)
        assert samples_df.columns == ["Sat.SMA", "Sat.INC"]
        assert len(samples_df.index) == 2

    async def test_monte_carlo_mode_dispatches_monte_carlo(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls = _FakeSweepCalls(result=_trivial_result(rows=2, with_status=False))
        _install_fake_gmat_sweep(monkeypatch, calls)
        fresh = _fresh_mcp(monkeypatch)
        script = tmp_path / "fixture.script"
        script.write_text("% noop\n")

        await fresh.call_tool(
            "gmat_sweep",
            {
                "script": str(script),
                "mode": "monte_carlo",
                "perturb": {"Sat.SMA": ["normal", 7000, 5.0]},
                "n": 5,
                "seed": 42,
            },
        )
        assert len(calls.monte_carlo_calls) == 1
        assert calls.monte_carlo_calls[0]["n"] == 5
        assert calls.monte_carlo_calls[0]["seed"] == 42
        # Perturb was coerced from list to tuple at the boundary.
        assert calls.monte_carlo_calls[0]["perturb"] == {"Sat.SMA": ("normal", 7000, 5.0)}

    async def test_latin_hypercube_mode_dispatches_latin_hypercube(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls = _FakeSweepCalls(result=_trivial_result(rows=2, with_status=False))
        _install_fake_gmat_sweep(monkeypatch, calls)
        fresh = _fresh_mcp(monkeypatch)
        script = tmp_path / "fixture.script"
        script.write_text("% noop\n")

        await fresh.call_tool(
            "gmat_sweep",
            {
                "script": str(script),
                "mode": "latin_hypercube",
                "perturb": {"Sat.SMA": ["uniform", 6900, 7100]},
                "n": 8,
                "seed": 7,
            },
        )
        assert len(calls.latin_hypercube_calls) == 1
        assert calls.latin_hypercube_calls[0]["n"] == 8

    async def test_max_workers_threaded_through(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls = _FakeSweepCalls(result=_trivial_result(rows=1, with_status=False))
        _install_fake_gmat_sweep(monkeypatch, calls)
        fresh = _fresh_mcp(monkeypatch)
        script = tmp_path / "fixture.script"
        script.write_text("% noop\n")

        await fresh.call_tool(
            "gmat_sweep",
            {
                "script": str(script),
                "mode": "grid",
                "grid": {"Sat.SMA": [7000]},
                "max_workers": 4,
            },
        )
        assert calls.sweep_calls[0]["backend"].max_workers == 4

    async def test_output_full_returns_every_row(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls = _FakeSweepCalls(result=_trivial_result(rows=6, with_status=False))
        _install_fake_gmat_sweep(monkeypatch, calls)
        fresh = _fresh_mcp(monkeypatch)
        script = tmp_path / "fixture.script"
        script.write_text("% noop\n")

        _content, structured = await fresh.call_tool(
            "gmat_sweep",
            {
                "script": str(script),
                "mode": "grid",
                "grid": {"Sat.SMA": [7000, 7100, 7200, 7300, 7400, 7500]},
                "output": "full",
            },
        )
        parsed = GmatSweepResponse.model_validate(structured)
        assert parsed.truncated is False
        assert len(parsed.rows) == 12

    async def test_inline_script_writes_temp_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _FakeSweepCalls(result=_trivial_result(rows=1, with_status=False))
        _install_fake_gmat_sweep(monkeypatch, calls)
        fresh = _fresh_mcp(monkeypatch)

        inline = "% inline sweep fixture\nCreate Spacecraft Sat\n"
        _content, structured = await fresh.call_tool(
            "gmat_sweep",
            {
                "script": inline,
                "mode": "grid",
                "grid": {"Sat.SMA": [7000]},
            },
        )
        parsed = GmatSweepResponse.model_validate(structured)
        assert parsed.script_name.endswith(".script")
        # The temp file is unlinked after the tool returns; the path the
        # sweep saw lives on the calls record.
        assert calls.sweep_calls[0]["script"].suffix == ".script"

    async def test_mode_payload_conflict_surfaces_typed_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls = _FakeSweepCalls(result=_trivial_result(rows=1, with_status=False))
        _install_fake_gmat_sweep(monkeypatch, calls)
        fresh = _fresh_mcp(monkeypatch)
        script = tmp_path / "fixture.script"
        script.write_text("% noop\n")

        with pytest.raises(ToolError) as excinfo:
            await fresh.call_tool(
                "gmat_sweep",
                {
                    "script": str(script),
                    "mode": "grid",
                    "grid": {"Sat.SMA": [7000]},
                    "perturb": {"Sat.SMA": ["normal", 7000, 1]},
                },
            )
        assert "invalid_input.gmat_sweep_mode_payload_conflict" in str(excinfo.value)

    async def test_sweep_config_error_surfaces_invalid_input(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls = _FakeSweepCalls(result=_trivial_result(rows=1, with_status=False))
        calls.raise_config_error = True
        _install_fake_gmat_sweep(monkeypatch, calls)
        fresh = _fresh_mcp(monkeypatch)
        script = tmp_path / "fixture.script"
        script.write_text("% noop\n")

        with pytest.raises(ToolError) as excinfo:
            await fresh.call_tool(
                "gmat_sweep",
                {
                    "script": str(script),
                    "mode": "grid",
                    "grid": {"Sat.SMA": [7000]},
                },
            )
        assert "invalid_input.gmat_sweep_config" in str(excinfo.value)

    async def test_unexpected_sweep_error_surfaces_upstream(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls = _FakeSweepCalls(result=_trivial_result(rows=1, with_status=False))
        calls.raise_unexpected = True
        _install_fake_gmat_sweep(monkeypatch, calls)
        fresh = _fresh_mcp(monkeypatch)
        script = tmp_path / "fixture.script"
        script.write_text("% noop\n")

        with pytest.raises(ToolError) as excinfo:
            await fresh.call_tool(
                "gmat_sweep",
                {
                    "script": str(script),
                    "mode": "grid",
                    "grid": {"Sat.SMA": [7000]},
                },
            )
        assert "upstream.gmat_sweep_failed" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Schema + lint coverage
# ---------------------------------------------------------------------------


class TestResponseSchema:
    """Round-trip the response through JSON to catch schema drift."""

    def test_response_roundtrips_through_json(self, tmp_path: Path) -> None:
        frame = _trivial_result(rows=2, with_status=True)
        response = _build_sweep_response(
            run_id="0" * 32,
            mode="monte_carlo",
            script_name="fixture.script",
            frame=frame,
            wall_clock_s=0.42,
            manifest_path=tmp_path / "manifest.jsonl",
            output_dir=tmp_path,
            output="summary",
        )
        first = response.model_dump_json()
        rebuilt = GmatSweepResponse.model_validate_json(first)
        assert rebuilt.model_dump_json() == first
        # And the model_dump form is JSON-serialisable.
        json.dumps(response.model_dump(mode="json"), sort_keys=True)


class TestToolListing:
    """Description lint must accept the real `gmat_sweep` description."""

    async def test_description_passes_lint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fresh = _fresh_mcp(monkeypatch)
        tools = await fresh.list_tools()
        for tool in tools:
            if tool.name == "gmat_sweep":
                violations = check_tool_descriptions([tool])
                assert violations == []
                return
        pytest.fail("gmat_sweep slot missing from the fresh FastMCP surface")


# ---------------------------------------------------------------------------
# Integration: real GMAT install + real gmat-sweep
# ---------------------------------------------------------------------------


_FIXTURE_SCRIPT = Path(__file__).parent / "data" / "gmat_minimal_leo.script"


@pytest.mark.gmat_installed
@pytest.mark.skipif(
    importlib.util.find_spec("gmat_sweep") is None,
    reason="gmat_sweep is not installed; install the [gmat] extra to run this test",
)
class TestIntegrationAgainstRealGmat:
    """End-to-end: invoke the tool against the real Linux GMAT install + gmat-sweep."""

    async def test_monte_carlo_runs_end_to_end(self) -> None:
        # When this test runs gmat_run + gmat_sweep are installed; the slots
        # registered at module-load time own the singleton already.
        from astrodynamics_mcp.server import mcp

        _content, structured = await mcp.call_tool(
            "gmat_sweep",
            {
                "script": str(_FIXTURE_SCRIPT),
                "mode": "monte_carlo",
                "perturb": {"Sat.SMA": ["normal", 7000.0, 5.0]},
                "n": 2,
                "seed": 13,
            },
        )
        parsed = GmatSweepResponse.model_validate(structured)
        assert parsed.mode == "monte_carlo"
        assert parsed.script_name == "gmat_minimal_leo.script"
        # status_counts.ok + failed + skipped must equal run_count == 2.
        total = (
            parsed.status_counts.ok.value
            + parsed.status_counts.failed.value
            + parsed.status_counts.skipped.value
        )
        assert total == parsed.run_count.value
        # At least one head row must carry the run_id slot.
        assert parsed.head
        assert "run_id" in parsed.head[0]
        # Manifest pointer exists on disk after the tool returns.
        assert Path(parsed.manifest_path).is_file()


# ---------------------------------------------------------------------------
# Chained producer → read seam
# ---------------------------------------------------------------------------


class TestChainedReadback:
    """Drives gmat_sweep end-to-end, then reads ``manifest.jsonl`` back
    through gmat_read_run_artefact using the returned run_id. Sweep
    registers only the manifest (not per-run artefacts) so the chain
    check rounds through that file rather than a ReportFile — same seam
    semantics, different file.
    """

    async def test_sweep_then_read_manifest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from astrodynamics_mcp import runs as runs_module
        from astrodynamics_mcp.runs import RunRegistry
        from astrodynamics_mcp.tools.gmat import RawReportContent

        registry = RunRegistry(directory=tmp_path / "cache", limit=5)
        monkeypatch.setattr(runs_module, "_default_registry", registry)

        # The bare _FakeSweepCalls.sweep doesn't touch its ``out`` kwarg;
        # the producer only registers manifest.jsonl when it exists, so
        # we wrap sweep() to write a tiny manifest before returning.
        manifest_text = '{"run_id":0,"status":"ok","Sat.SMA":7000.0}\n'
        calls = _FakeSweepCalls(result=_trivial_result(rows=1, with_status=False))
        original_sweep = calls.sweep

        def writing_sweep(script: Any, **kwargs: Any) -> Any:
            out_dir = Path(kwargs["out"])
            # write_bytes, not write_text, so Windows doesn't translate
            # \n to \r\n — the byte-equality assertion below relies on
            # identity.
            (out_dir / "manifest.jsonl").write_bytes(manifest_text.encode("utf-8"))
            return original_sweep(script, **kwargs)

        calls.sweep = writing_sweep  # type: ignore[method-assign]
        _install_fake_gmat_sweep(monkeypatch, calls)
        fresh = _fresh_mcp(monkeypatch)
        script = tmp_path / "fixture.script"
        script.write_text("% noop\n")

        _content, structured = await fresh.call_tool(
            "gmat_sweep",
            {
                "script": str(script),
                "mode": "grid",
                "grid": {"Sat.SMA": [7000.0]},
            },
        )
        producer = GmatSweepResponse.model_validate(structured)
        # The seam: producer registered against this singleton.
        entry = registry.get(producer.run_id)
        assert entry is not None
        assert "manifest.jsonl" in entry.artefacts

        _content, structured = await fresh.call_tool(
            "gmat_read_run_artefact",
            {"run_id": producer.run_id, "name": "manifest.jsonl", "output": "full"},
        )
        readback = RawReportContent.model_validate(structured)
        assert readback.content == manifest_text
        assert readback.truncated is False
        assert readback.byte_count.value == float(
            (Path(producer.output_dir) / "manifest.jsonl").stat().st_size
        )
