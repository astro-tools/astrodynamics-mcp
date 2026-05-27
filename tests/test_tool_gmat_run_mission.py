"""Tests for the ``gmat_run_mission`` tool.

The unit tests drive the tool with a fake ``gmat_run`` injected via the
:mod:`astrodynamics_mcp.tools.gmat` module namespace and the registered
slot on a per-test :class:`FastMCP` instance. The integration tests
exercise the same body against a real GMAT install — gated on the
``gmat_installed`` marker so contributors without GMAT still get green
CI on the rest of the suite.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from astrodynamics_mcp.server_lint import check_tool_descriptions
from astrodynamics_mcp.tools import gmat as gmat_tools
from astrodynamics_mcp.tools.gmat import (
    GmatRunMissionResponse,
    _build_response,
    _looks_like_inline_script,
    _resolve_script_input,
    _shape_report,
)

# ---------------------------------------------------------------------------
# Fake gmat_run surface
# ---------------------------------------------------------------------------


class _FakeGmatFieldError(Exception):
    """Stand-in for ``gmat_run.errors.GmatFieldError``."""


class _FakeGmatLoadError(Exception):
    """Stand-in for ``gmat_run.errors.GmatLoadError``."""


class _FakeGmatRunError(Exception):
    """Stand-in for ``gmat_run.errors.GmatRunError``."""


class _FakeGmatError(Exception):
    """Stand-in for ``gmat_run.errors.GmatError`` (root)."""


class _FakeResourceGroup:
    def __init__(self, category: str, names: tuple[str, ...]) -> None:
        self.category = category
        self.names = names


class _FakeCommandOutline:
    def __init__(
        self,
        *,
        type_name: str,
        summary: str,
        children: tuple[_FakeCommandOutline, ...] = (),
        nested_count: int = 0,
    ) -> None:
        self.type_name = type_name
        self.summary = summary
        self.children = children
        self.nested_count = nested_count


class _FakeMissionSummary:
    def __init__(
        self,
        *,
        script_name: str,
        resource_groups: tuple[_FakeResourceGroup, ...],
        commands: tuple[_FakeCommandOutline, ...],
    ) -> None:
        self.script_name = script_name
        self.resource_groups = resource_groups
        self.commands = commands


class _FakeMission:
    """Behaviour-compatible stand-in for :class:`gmat_run.Mission`."""

    def __init__(
        self,
        *,
        summary: _FakeMissionSummary,
        run_result: _FakeResult,
        accept_overrides: tuple[str, ...] = (),
        reject_override: str | None = None,
    ) -> None:
        self._summary = summary
        self._run_result = run_result
        self._accept = set(accept_overrides)
        self._reject = reject_override
        self.writes: dict[str, Any] = {}

    def __setitem__(self, key: str, value: Any) -> None:
        if self._reject is not None and key == self._reject:
            raise _FakeGmatFieldError(f"rejected override on {key!r}")
        if self._accept and key not in self._accept:
            raise _FakeGmatFieldError(f"unknown override {key!r}")
        self.writes[key] = value

    def summary(self) -> _FakeMissionSummary:
        return self._summary

    def run(self, *, working_dir: Any = None, overwrite: bool = False) -> _FakeResult:
        del working_dir, overwrite
        return self._run_result


class _FakeResult:
    """Behaviour-compatible stand-in for :class:`gmat_run.Results`."""

    def __init__(
        self,
        *,
        reports: dict[str, Any],
        report_paths: dict[str, Path],
        ephemeris_paths: dict[str, Path] | None = None,
        contact_paths: dict[str, Path] | None = None,
        converged: dict[str, bool] | None = None,
    ) -> None:
        self.reports = reports
        self.report_paths = MappingProxyType(dict(report_paths))
        self.ephemeris_paths = MappingProxyType(dict(ephemeris_paths or {}))
        self.contact_paths = MappingProxyType(dict(contact_paths or {}))
        self.converged = dict(converged or {})


def _install_fake_gmat_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mission: _FakeMission,
    load_error: BaseException | None = None,
    run_error: BaseException | None = None,
) -> None:
    """Inject a fake ``gmat_run`` module into ``sys.modules`` for one test.

    The tool body imports ``gmat_run`` and ``gmat_run.errors`` inside its
    function bodies, so a fake plugged into ``sys.modules`` is enough to
    redirect the import — no need to monkeypatch the tool module itself.
    """
    fake = ModuleType("gmat_run")
    fake_errors = ModuleType("gmat_run.errors")
    fake_errors.__dict__["GmatFieldError"] = _FakeGmatFieldError
    fake_errors.__dict__["GmatLoadError"] = _FakeGmatLoadError
    fake_errors.__dict__["GmatRunError"] = _FakeGmatRunError
    fake_errors.__dict__["GmatError"] = _FakeGmatError

    class _MissionFactory:
        @staticmethod
        def load(path: Any) -> _FakeMission:
            if load_error is not None:
                raise load_error
            return mission

    fake.__dict__["Mission"] = _MissionFactory
    fake.__dict__["errors"] = fake_errors
    monkeypatch.setitem(sys.modules, "gmat_run", fake)
    monkeypatch.setitem(sys.modules, "gmat_run.errors", fake_errors)

    if run_error is not None:

        def _raise(*, working_dir: Any = None, overwrite: bool = False) -> _FakeResult:
            del working_dir, overwrite
            raise run_error

        mission.run = _raise  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeDataFrame:
    """Minimal pandas-DataFrame stand-in for the report-shaping path.

    The tool body reads ``.columns``, ``.index`` (only ``len`` is used), and
    ``.to_numpy(dtype=object)`` — pre-computing a 2D ``list[list[Any]]`` is
    enough to exercise every shaping branch without pulling pandas into the
    test env.
    """

    def __init__(self, columns: list[str], rows: list[list[Any]]) -> None:
        self.columns = list(columns)
        self._rows = [list(r) for r in rows]
        self.index = range(len(self._rows))

    def to_numpy(self, *, dtype: Any = None) -> Any:
        del dtype
        return self._rows


def _trivial_summary() -> _FakeMissionSummary:
    return _FakeMissionSummary(
        script_name="fixture.script",
        resource_groups=(
            _FakeResourceGroup("Spacecraft", ("Sat",)),
            _FakeResourceGroup("ReportFile", ("RF",)),
        ),
        commands=(_FakeCommandOutline(type_name="Propagate", summary="Propagate Prop(Sat)"),),
    )


def _small_report(rows: int = 5) -> _FakeDataFrame:
    """ReportFile-shaped fake with a string time column and numeric state columns."""
    columns = ["Sat.UTCGregorian", "Sat.X", "Sat.Y", "Sat.SMA"]
    data = [
        [
            f"01 Jan 2026 12:0{i}:00.000",
            7000.0 + i,
            0.0 + i,
            7000.0 + 0.01 * i,
        ]
        for i in range(rows)
    ]
    return _FakeDataFrame(columns, data)


def _large_report(rows: int = 100) -> _FakeDataFrame:
    columns = ["Sat.UTCGregorian", "Sat.X"]
    data = [[f"row-{i:04d}", 7000.0 + i] for i in range(rows)]
    return _FakeDataFrame(columns, data)


def _fresh_mcp(monkeypatch: pytest.MonkeyPatch) -> FastMCP:
    """Stand up a per-test FastMCP and re-register the GMAT slots against it."""
    fresh = FastMCP("gmat-run-mission-test")
    monkeypatch.setattr("astrodynamics_mcp.server.mcp", fresh)
    monkeypatch.setattr(gmat_tools, "_GMAT_RUN_AVAILABLE", True)
    gmat_tools._register_gmat_tools()
    return fresh


# ---------------------------------------------------------------------------
# Path / inline-text auto-detect
# ---------------------------------------------------------------------------


class TestScriptInputResolution:
    def test_inline_text_with_newlines_is_inline(self) -> None:
        assert _looks_like_inline_script("Create Spacecraft Sat\nSat.SMA = 7000\n")

    def test_leading_percent_comment_is_inline(self) -> None:
        assert _looks_like_inline_script("% just one commented line")

    def test_leading_create_is_inline(self) -> None:
        assert _looks_like_inline_script("Create Spacecraft Sat")

    def test_bare_path_is_path(self) -> None:
        assert not _looks_like_inline_script("/abs/path/to/file.script")

    def test_inline_text_writes_to_tempfile(self) -> None:
        text = "% sample\nCreate Spacecraft Sat\n"
        path, handle = _resolve_script_input(text)
        try:
            assert handle is not None
            assert path.exists()
            assert path.suffix == ".script"
            assert path.read_text() == text
        finally:
            if handle is not None:
                Path(handle.name).unlink(missing_ok=True)

    def test_relative_path_rejected(self) -> None:
        from astrodynamics_mcp.errors import InvalidInputError

        with pytest.raises(InvalidInputError) as excinfo:
            _resolve_script_input("relative/file.script")
        assert excinfo.value.code == "invalid_input.script_path_not_absolute"

    def test_missing_path_rejected(self, tmp_path: Path) -> None:
        from astrodynamics_mcp.errors import InvalidInputError

        missing = tmp_path / "does_not_exist.script"
        with pytest.raises(InvalidInputError) as excinfo:
            _resolve_script_input(str(missing))
        assert excinfo.value.code == "invalid_input.script_path_not_found"

    def test_existing_path_passes_through(self, tmp_path: Path) -> None:
        f = tmp_path / "ok.script"
        f.write_text("Create Spacecraft Sat\n")
        path, handle = _resolve_script_input(str(f))
        assert handle is None
        assert path == f


# ---------------------------------------------------------------------------
# Report shaping
# ---------------------------------------------------------------------------


class TestReportShape:
    def test_small_report_inlines_full_rows(self, tmp_path: Path) -> None:
        df = _small_report(rows=5)
        shape = _shape_report("RF", tmp_path / "leo.txt", df, output="summary")
        assert shape.truncated is False
        assert shape.head == []
        assert shape.tail == []
        assert len(shape.rows) == 5
        assert shape.rows[0]["Sat.UTCGregorian"] == "01 Jan 2026 12:00:00.000"
        assert shape.rows[0]["Sat.X"] == 7000.0
        assert shape.row_count.value == 5.0
        assert shape.row_count.unit == "1"
        assert shape.columns == ["Sat.UTCGregorian", "Sat.X", "Sat.Y", "Sat.SMA"]

    def test_large_report_emits_head_and_tail_in_summary_mode(self, tmp_path: Path) -> None:
        df = _large_report(rows=100)
        shape = _shape_report("RF", tmp_path / "leo.txt", df, output="summary")
        assert shape.truncated is True
        assert shape.rows == []
        assert len(shape.head) == 5
        assert len(shape.tail) == 5
        assert shape.head[0]["Sat.UTCGregorian"] == "row-0000"
        assert shape.tail[-1]["Sat.UTCGregorian"] == "row-0099"
        assert shape.row_count.value == 100.0

    def test_large_report_inlines_every_row_in_full_mode(self, tmp_path: Path) -> None:
        df = _large_report(rows=100)
        shape = _shape_report("RF", tmp_path / "leo.txt", df, output="full")
        assert shape.truncated is False
        assert shape.head == []
        assert shape.tail == []
        assert len(shape.rows) == 100
        assert shape.rows[0]["Sat.UTCGregorian"] == "row-0000"
        assert shape.rows[-1]["Sat.UTCGregorian"] == "row-0099"

    def test_threshold_boundary_inlines_in_summary_mode(self, tmp_path: Path) -> None:
        # row_count == 20 is the inclusive inline threshold.
        df = _large_report(rows=20)
        shape = _shape_report("RF", tmp_path / "leo.txt", df, output="summary")
        assert shape.truncated is False
        assert len(shape.rows) == 20

    def test_threshold_boundary_truncates_in_summary_mode(self, tmp_path: Path) -> None:
        df = _large_report(rows=21)
        shape = _shape_report("RF", tmp_path / "leo.txt", df, output="summary")
        assert shape.truncated is True
        assert len(shape.head) == 5
        assert len(shape.tail) == 5


# ---------------------------------------------------------------------------
# End-to-end call via FastMCP
# ---------------------------------------------------------------------------


class TestToolThroughMcp:
    async def test_callable_with_path_argument(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        script = tmp_path / "fixture.script"
        script.write_text("% noop fixture for the fake gmat_run path\n")
        report_path = tmp_path / "leo.txt"
        result = _FakeResult(
            reports={"RF": _small_report(3)},
            report_paths={"RF": report_path},
            converged={"DC": True},
        )
        mission = _FakeMission(summary=_trivial_summary(), run_result=result)
        _install_fake_gmat_run(monkeypatch, mission=mission)
        fresh = _fresh_mcp(monkeypatch)

        _content, structured = await fresh.call_tool("gmat_run_mission", {"script": str(script)})

        assert isinstance(structured, dict)
        parsed = GmatRunMissionResponse.model_validate(structured)
        assert parsed.summary.script_name == "fixture.script"
        assert parsed.reports[0].name == "RF"
        assert parsed.reports[0].rows[0]["Sat.X"] == 7000.0
        assert parsed.converged == {"DC": True}
        assert parsed.wall_clock.unit == "s"
        assert parsed.wall_clock.value >= 0.0

    async def test_callable_with_inline_text(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        result = _FakeResult(
            reports={"RF": _small_report(3)},
            report_paths={"RF": tmp_path / "leo.txt"},
        )
        mission = _FakeMission(summary=_trivial_summary(), run_result=result)
        _install_fake_gmat_run(monkeypatch, mission=mission)
        fresh = _fresh_mcp(monkeypatch)

        inline = "% inline fixture\nCreate Spacecraft Sat\n"
        _content, structured = await fresh.call_tool("gmat_run_mission", {"script": inline})
        assert isinstance(structured, dict)
        parsed = GmatRunMissionResponse.model_validate(structured)
        assert parsed.reports[0].columns[0] == "Sat.UTCGregorian"

    async def test_overrides_applied_before_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        script = tmp_path / "fixture.script"
        script.write_text("% noop\n")
        result = _FakeResult(reports={}, report_paths={}, converged={})
        mission = _FakeMission(
            summary=_trivial_summary(),
            run_result=result,
            accept_overrides=("Sat.SMA",),
        )
        _install_fake_gmat_run(monkeypatch, mission=mission)
        fresh = _fresh_mcp(monkeypatch)

        await fresh.call_tool(
            "gmat_run_mission",
            {"script": str(script), "overrides": {"Sat.SMA": 7100.0}},
        )
        assert mission.writes == {"Sat.SMA": 7100.0}

    async def test_invalid_override_surfaces_typed_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        script = tmp_path / "fixture.script"
        script.write_text("% noop\n")
        mission = _FakeMission(
            summary=_trivial_summary(),
            run_result=_FakeResult(reports={}, report_paths={}),
            accept_overrides=("Sat.SMA",),
            reject_override="Sat.SMA",
        )
        _install_fake_gmat_run(monkeypatch, mission=mission)
        fresh = _fresh_mcp(monkeypatch)

        with pytest.raises(ToolError) as excinfo:
            await fresh.call_tool(
                "gmat_run_mission",
                {"script": str(script), "overrides": {"Sat.SMA": 7100.0}},
            )
        raw = str(excinfo.value)
        assert "invalid_input.gmat_override_failed" in raw

    async def test_select_outputs_filters_reports(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        script = tmp_path / "fixture.script"
        script.write_text("% noop\n")
        result = _FakeResult(
            reports={"RF1": _small_report(2), "RF2": _small_report(3)},
            report_paths={"RF1": tmp_path / "r1.txt", "RF2": tmp_path / "r2.txt"},
        )
        mission = _FakeMission(summary=_trivial_summary(), run_result=result)
        _install_fake_gmat_run(monkeypatch, mission=mission)
        fresh = _fresh_mcp(monkeypatch)

        _content, structured = await fresh.call_tool(
            "gmat_run_mission", {"script": str(script), "select_outputs": ["RF2"]}
        )
        parsed = GmatRunMissionResponse.model_validate(structured)
        assert [r.name for r in parsed.reports] == ["RF2"]

    async def test_output_full_inlines_every_row(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        script = tmp_path / "fixture.script"
        script.write_text("% noop\n")
        result = _FakeResult(
            reports={"RF": _large_report(50)},
            report_paths={"RF": tmp_path / "leo.txt"},
        )
        mission = _FakeMission(summary=_trivial_summary(), run_result=result)
        _install_fake_gmat_run(monkeypatch, mission=mission)
        fresh = _fresh_mcp(monkeypatch)

        _content, structured = await fresh.call_tool(
            "gmat_run_mission", {"script": str(script), "output": "full"}
        )
        parsed = GmatRunMissionResponse.model_validate(structured)
        assert parsed.reports[0].truncated is False
        assert len(parsed.reports[0].rows) == 50

    async def test_select_outputs_rejects_unknown_names(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        script = tmp_path / "fixture.script"
        script.write_text("% noop\n")
        result = _FakeResult(
            reports={"RF1": _small_report(2)}, report_paths={"RF1": tmp_path / "r1.txt"}
        )
        mission = _FakeMission(summary=_trivial_summary(), run_result=result)
        _install_fake_gmat_run(monkeypatch, mission=mission)
        fresh = _fresh_mcp(monkeypatch)

        with pytest.raises(ToolError) as excinfo:
            await fresh.call_tool(
                "gmat_run_mission",
                {"script": str(script), "select_outputs": ["Missing"]},
            )
        assert "invalid_input.unknown_output_selection" in str(excinfo.value)

    async def test_load_error_surfaces_upstream_code(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        script = tmp_path / "fixture.script"
        script.write_text("% noop\n")
        mission = _FakeMission(
            summary=_trivial_summary(),
            run_result=_FakeResult(reports={}, report_paths={}),
        )
        _install_fake_gmat_run(
            monkeypatch,
            mission=mission,
            load_error=_FakeGmatLoadError("bad parse"),
        )
        fresh = _fresh_mcp(monkeypatch)

        with pytest.raises(ToolError) as excinfo:
            await fresh.call_tool("gmat_run_mission", {"script": str(script)})
        assert "upstream.gmat_run_load_failed" in str(excinfo.value)

    async def test_run_error_surfaces_upstream_code(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        script = tmp_path / "fixture.script"
        script.write_text("% noop\n")
        mission = _FakeMission(
            summary=_trivial_summary(),
            run_result=_FakeResult(reports={}, report_paths={}),
        )
        _install_fake_gmat_run(
            monkeypatch,
            mission=mission,
            run_error=_FakeGmatRunError("RunScript status -1"),
        )
        fresh = _fresh_mcp(monkeypatch)

        with pytest.raises(ToolError) as excinfo:
            await fresh.call_tool("gmat_run_mission", {"script": str(script)})
        assert "upstream.gmat_run_failed" in str(excinfo.value)

    async def test_ephemeris_and_contact_pointers_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        script = tmp_path / "fixture.script"
        script.write_text("% noop\n")
        eph_path = tmp_path / "eph.oem"
        contact_path = tmp_path / "contacts.txt"
        result = _FakeResult(
            reports={},
            report_paths={},
            ephemeris_paths={"EphemerisFile1": eph_path},
            contact_paths={"ContactLocator1": contact_path},
        )
        mission = _FakeMission(summary=_trivial_summary(), run_result=result)
        _install_fake_gmat_run(monkeypatch, mission=mission)
        fresh = _fresh_mcp(monkeypatch)

        _content, structured = await fresh.call_tool("gmat_run_mission", {"script": str(script)})
        parsed = GmatRunMissionResponse.model_validate(structured)
        assert [e.name for e in parsed.ephemerides] == ["EphemerisFile1"]
        assert parsed.ephemerides[0].path == str(eph_path)
        assert [c.name for c in parsed.contacts] == ["ContactLocator1"]
        assert parsed.contacts[0].path == str(contact_path)

    async def test_branch_command_emits_children_and_nesting_flag(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        script = tmp_path / "fixture.script"
        script.write_text("% noop\n")
        target = _FakeCommandOutline(
            type_name="Target",
            summary="Target DC",
            children=(_FakeCommandOutline(type_name="Vary", summary="Vary TOI.V"),),
            nested_count=3,
        )
        summary = _FakeMissionSummary(
            script_name="branch.script",
            resource_groups=(),
            commands=(target,),
        )
        mission = _FakeMission(
            summary=summary,
            run_result=_FakeResult(reports={}, report_paths={}),
        )
        _install_fake_gmat_run(monkeypatch, mission=mission)
        fresh = _fresh_mcp(monkeypatch)

        _content, structured = await fresh.call_tool("gmat_run_mission", {"script": str(script)})
        parsed = GmatRunMissionResponse.model_validate(structured)
        assert parsed.summary.commands[0].type_name == "Target"
        assert parsed.summary.commands[0].children[0].type_name == "Vary"
        assert parsed.summary.commands[0].has_deeper_nesting is True


# ---------------------------------------------------------------------------
# Schema + lint coverage
# ---------------------------------------------------------------------------


class TestResponseSchema:
    """Round-trip the response through JSON to catch schema drift."""

    def test_response_roundtrips_through_json(self, tmp_path: Path) -> None:
        result = _FakeResult(
            reports={"RF": _small_report(4)},
            report_paths={"RF": tmp_path / "leo.txt"},
            converged={"DC": True},
        )
        mission = _FakeMission(summary=_trivial_summary(), run_result=result)
        response = _build_response(
            run_id="0" * 32,
            mission=mission,
            result=result,
            wall_clock_s=0.05,
            select_outputs=None,
        )

        first = response.model_dump_json()
        rebuilt = GmatRunMissionResponse.model_validate_json(first)
        assert rebuilt.model_dump_json() == first
        # And the model_dump form is JSON-serialisable.
        json.dumps(response.model_dump(mode="json"), sort_keys=True)


class TestToolListing:
    """Description lint must accept the real `gmat_run_mission` description."""

    async def test_description_passes_lint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fresh = _fresh_mcp(monkeypatch)
        tools = await fresh.list_tools()
        for tool in tools:
            if tool.name == "gmat_run_mission":
                violations = check_tool_descriptions([tool])
                assert violations == []
                return
        pytest.fail("gmat_run_mission slot missing from the fresh FastMCP surface")


# ---------------------------------------------------------------------------
# Integration: real GMAT install
# ---------------------------------------------------------------------------


_FIXTURE_SCRIPT = Path(__file__).parent / "data" / "gmat_minimal_leo.script"


@pytest.mark.gmat_installed
@pytest.mark.skipif(
    importlib.util.find_spec("gmat_run") is None,
    reason="gmat_run is not installed; install the [gmat] extra to run this test",
)
class TestIntegrationAgainstRealGmat:
    """End-to-end: invoke the tool against the real Linux GMAT install."""

    async def test_minimal_leo_runs_end_to_end(self) -> None:
        # Re-import to land the real ``gmat_run`` import inside the tool body.
        # ``_GMAT_RUN_AVAILABLE`` was probed at module load time — when this
        # test runs, gmat_run is installed, so the singleton already has the
        # real registration.
        from astrodynamics_mcp.server import mcp

        # FastMCP keeps tools across a session; the placeholder fixtures stay
        # registered, but the real ``gmat_run_mission`` body owns the slot.
        _content, structured = await mcp.call_tool(
            "gmat_run_mission", {"script": str(_FIXTURE_SCRIPT)}
        )
        parsed = GmatRunMissionResponse.model_validate(structured)
        assert parsed.summary.script_name == "gmat_minimal_leo.script"
        report_names = {r.name for r in parsed.reports}
        assert "RF" in report_names
        rf = next(r for r in parsed.reports if r.name == "RF")
        assert "Sat.X" in rf.columns
        # The fixture propagates 600 s with a 60 s initial step, so we expect
        # a handful of report rows — small enough to fit inline.
        assert rf.row_count.value >= 1.0
        assert parsed.wall_clock.value > 0.0
