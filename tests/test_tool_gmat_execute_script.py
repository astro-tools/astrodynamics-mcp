"""Tests for the ``gmat_execute_script`` tool.

Drives the tool body with a fake ``gmat_run`` injected via ``sys.modules``
and a per-test :class:`FastMCP` instance. The fake :class:`_FakeMission`
writes synthetic report / ephemeris files into a real temp directory so
the artefact walk and the raw-text reader exercise actual filesystem
behaviour without needing a GMAT install.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any

import pytest
from mcp.server.fastmcp import FastMCP

from astrodynamics_mcp.tools.gmat import (
    GmatExecuteScriptResponse,
    _shape_raw_report,
    _walk_artefacts,
)

# ---------------------------------------------------------------------------
# Fake gmat_run surface
# ---------------------------------------------------------------------------


class _FakeGmatFieldError(Exception):
    pass


class _FakeGmatLoadError(Exception):
    pass


class _FakeGmatRunError(Exception):
    def __init__(self, message: str, log: str) -> None:
        self.log = log
        super().__init__(message)


class _FakeGmatError(Exception):
    pass


class _FakeResult:
    """Behaviour-compatible stand-in for :class:`gmat_run.Results`."""

    def __init__(
        self,
        *,
        output_dir: Path,
        log: str,
        report_paths: dict[str, Path] | None = None,
        ephemeris_paths: dict[str, Path] | None = None,
        contact_paths: dict[str, Path] | None = None,
        solver_paths: dict[str, Path] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.log = log
        self.report_paths = MappingProxyType(dict(report_paths or {}))
        self.ephemeris_paths = MappingProxyType(dict(ephemeris_paths or {}))
        self.contact_paths = MappingProxyType(dict(contact_paths or {}))
        self.solver_paths = MappingProxyType(dict(solver_paths or {}))


class _FakeMission:
    def __init__(
        self, *, run_result: _FakeResult | None = None, run_error: BaseException | None = None
    ) -> None:
        self._run_result = run_result
        self._run_error = run_error

    def summary(self) -> Any:  # pragma: no cover - escape hatch does not call this
        raise AssertionError("gmat_execute_script must not call mission.summary()")

    def run(self, *, working_dir: Any = None, overwrite: bool = False) -> _FakeResult:
        del working_dir, overwrite
        if self._run_error is not None:
            raise self._run_error
        assert self._run_result is not None
        return self._run_result


def _install_fake_gmat_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mission: _FakeMission | None = None,
    load_error: BaseException | None = None,
) -> None:
    """Inject a fake ``gmat_run`` module covering the symbols the tool body imports."""
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
            assert mission is not None
            return mission

    fake.__dict__["Mission"] = _MissionFactory
    fake.__dict__["errors"] = fake_errors
    monkeypatch.setitem(sys.modules, "gmat_run", fake)
    monkeypatch.setitem(sys.modules, "gmat_run.errors", fake_errors)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_mcp(monkeypatch: pytest.MonkeyPatch) -> FastMCP:
    from tests._gmat_helpers import make_fresh_mcp

    return make_fresh_mcp("gmat-execute-script-test", monkeypatch)


def _write_report(path: Path, *, lines: int) -> None:
    """Write a fixture ReportFile with a header row and ``lines`` data rows."""
    rows = ["Sat.UTCGregorian    Sat.X    Sat.Y    Sat.SMA"]
    for i in range(lines):
        row = f"01 Jan 2026 12:0{i % 10}:00.000    {7000.0 + i}    0.0    {7000.01 + i * 0.01}"
        rows.append(row)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


async def _call(mcp: FastMCP, **args: Any) -> GmatExecuteScriptResponse:
    # FastMCP returns (content, structured); the structured payload is the
    # pydantic-derived dict we round-trip back into the typed response.
    _content, structured = await mcp.call_tool("gmat_execute_script", args)
    return GmatExecuteScriptResponse.model_validate(structured)


# ---------------------------------------------------------------------------
# Raw-text shape helpers
# ---------------------------------------------------------------------------


class TestRawReportShape:
    def test_short_report_inlines_full_content(self, tmp_path: Path) -> None:
        report = tmp_path / "RF.txt"
        _write_report(report, lines=10)
        shape = _shape_raw_report("RF", report, output="summary")
        assert shape.truncated is False
        assert shape.head == ""
        assert shape.tail == ""
        # Header line plus 10 data lines = 11 lines.
        assert shape.line_count.value == 11.0
        assert shape.byte_count.value == float(report.stat().st_size)
        assert shape.content.startswith("Sat.UTCGregorian")
        # Last data row's first field is "01 Jan 2026 12:09:00.000".
        assert "12:09:00.000" in shape.content

    def test_long_report_emits_head_and_tail_in_summary(self, tmp_path: Path) -> None:
        report = tmp_path / "RF.txt"
        _write_report(report, lines=200)
        shape = _shape_raw_report("RF", report, output="summary")
        assert shape.truncated is True
        assert shape.content == ""
        head_lines = shape.head.split("\n")
        tail_lines = shape.tail.split("\n")
        assert len(head_lines) == 20
        assert len(tail_lines) == 20
        # The header is the first head line; the last data row is the last tail line.
        assert head_lines[0].startswith("Sat.UTCGregorian")
        assert tail_lines[-1].startswith("01 Jan 2026")
        # 1 header + 200 data lines.
        assert shape.line_count.value == 201.0

    def test_long_report_full_mode_returns_every_line(self, tmp_path: Path) -> None:
        report = tmp_path / "RF.txt"
        _write_report(report, lines=200)
        shape = _shape_raw_report("RF", report, output="full")
        assert shape.truncated is False
        assert shape.head == ""
        assert shape.tail == ""
        assert shape.content.count("\n") >= 200

    def test_at_threshold_is_not_truncated(self, tmp_path: Path) -> None:
        # 59 data rows + 1 header = 60 lines, which is the inline cap.
        report = tmp_path / "RF.txt"
        _write_report(report, lines=59)
        shape = _shape_raw_report("RF", report, output="summary")
        assert shape.line_count.value == 60.0
        assert shape.truncated is False

    def test_non_utf8_bytes_do_not_crash(self, tmp_path: Path) -> None:
        report = tmp_path / "RF.txt"
        report.write_bytes(b"header\n\xff\xfe data\n")
        shape = _shape_raw_report("RF", report, output="summary")
        assert shape.truncated is False
        assert shape.line_count.value == 2.0
        # The undecodable byte becomes U+FFFD; the rest survives.
        assert "header" in shape.content
        assert "data" in shape.content


# ---------------------------------------------------------------------------
# Artefact walk
# ---------------------------------------------------------------------------


class TestWalkArtefacts:
    def test_known_resources_carry_their_declared_names(self, tmp_path: Path) -> None:
        rf = tmp_path / "RF.txt"
        eph = tmp_path / "EF.oem"
        rf.write_text("header\nrow\n")
        eph.write_text("eph data\n")
        result = _FakeResult(
            output_dir=tmp_path,
            log="",
            report_paths={"ReportFile1": rf},
            ephemeris_paths={"EphemerisFile1": eph},
        )
        artefacts = _walk_artefacts(result)
        names = {a.name for a in artefacts}
        assert names == {"ReportFile1", "EphemerisFile1"}

    def test_stray_files_use_basename(self, tmp_path: Path) -> None:
        rf = tmp_path / "RF.txt"
        stray = tmp_path / "gmat.log"
        rf.write_text("header\nrow\n")
        stray.write_text("noise\n")
        result = _FakeResult(
            output_dir=tmp_path,
            log="",
            report_paths={"ReportFile1": rf},
        )
        artefacts = _walk_artefacts(result)
        by_name = {a.name: a.path for a in artefacts}
        assert by_name["ReportFile1"] == str(rf)
        assert by_name["gmat.log"] == str(stray)

    def test_nested_files_are_picked_up(self, tmp_path: Path) -> None:
        nested = tmp_path / "Solver" / "DC.data"
        nested.parent.mkdir()
        nested.write_text("iter 0\n")
        result = _FakeResult(
            output_dir=tmp_path,
            log="",
            solver_paths={"DC": nested},
        )
        artefacts = _walk_artefacts(result)
        assert [a.name for a in artefacts] == ["DC"]
        assert artefacts[0].path == str(nested)

    def test_sorted_by_path(self, tmp_path: Path) -> None:
        for name in ["c.txt", "a.txt", "b.txt"]:
            (tmp_path / name).write_text("x")
        result = _FakeResult(output_dir=tmp_path, log="")
        artefacts = _walk_artefacts(result)
        assert [Path(a.path).name for a in artefacts] == ["a.txt", "b.txt", "c.txt"]


# ---------------------------------------------------------------------------
# Tool body
# ---------------------------------------------------------------------------


class TestSuccessfulRun:
    async def test_inline_script_runs_and_reports_inline(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rf = tmp_path / "RF.txt"
        _write_report(rf, lines=5)
        result = _FakeResult(
            output_dir=tmp_path,
            log="GMAT log: mission complete\n",
            report_paths={"ReportFile1": rf},
        )
        _install_fake_gmat_run(monkeypatch, mission=_FakeMission(run_result=result))
        mcp = _fresh_mcp(monkeypatch)

        response = await _call(mcp, script="% inline\nCreate Spacecraft Sat\n")

        assert response.ok is True
        assert response.stderr == "GMAT log: mission complete\n"
        assert response.wall_clock.unit == "s"
        assert response.wall_clock.value >= 0.0
        assert len(response.reports) == 1
        rep = response.reports[0]
        assert rep.name == "ReportFile1"
        assert rep.truncated is False
        assert "01 Jan 2026" in rep.content
        assert {a.name for a in response.artefacts} == {"ReportFile1"}

    async def test_large_report_truncated_under_summary(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rf = tmp_path / "RF.txt"
        _write_report(rf, lines=200)
        result = _FakeResult(output_dir=tmp_path, log="", report_paths={"ReportFile1": rf})
        _install_fake_gmat_run(monkeypatch, mission=_FakeMission(run_result=result))
        mcp = _fresh_mcp(monkeypatch)

        response = await _call(mcp, script="% inline\nCreate Spacecraft Sat\n")
        rep = response.reports[0]
        assert rep.truncated is True
        assert rep.content == ""
        assert rep.head.count("\n") == 19  # 20 lines, 19 separators
        assert rep.tail.count("\n") == 19

    async def test_large_report_full_mode_returns_every_line(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rf = tmp_path / "RF.txt"
        _write_report(rf, lines=200)
        result = _FakeResult(output_dir=tmp_path, log="", report_paths={"ReportFile1": rf})
        _install_fake_gmat_run(monkeypatch, mission=_FakeMission(run_result=result))
        mcp = _fresh_mcp(monkeypatch)

        response = await _call(mcp, script="% inline\nCreate Spacecraft Sat\n", output="full")
        rep = response.reports[0]
        assert rep.truncated is False
        assert rep.content.count("\n") >= 200

    async def test_artefacts_include_ephemerides_and_strays(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rf = tmp_path / "RF.txt"
        eph = tmp_path / "EF.oem"
        log = tmp_path / "gmat.log"
        _write_report(rf, lines=5)
        eph.write_text("eph data\n")
        log.write_text("noise\n")
        result = _FakeResult(
            output_dir=tmp_path,
            log="",
            report_paths={"ReportFile1": rf},
            ephemeris_paths={"EphemerisFile1": eph},
        )
        _install_fake_gmat_run(monkeypatch, mission=_FakeMission(run_result=result))
        mcp = _fresh_mcp(monkeypatch)

        response = await _call(mcp, script="% inline\nCreate Spacecraft Sat\n")
        names = {a.name for a in response.artefacts}
        assert names == {"ReportFile1", "EphemerisFile1", "gmat.log"}
        # Ephemeris is pointer-only — never inlined into `reports`.
        assert [r.name for r in response.reports] == ["ReportFile1"]


class TestRunFailuresAreData:
    async def test_gmat_run_error_returns_ok_false_with_stderr(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        exc = _FakeGmatRunError("RunScript failed", log="GMAT error: bad burn duration\n")
        _install_fake_gmat_run(monkeypatch, mission=_FakeMission(run_error=exc))
        mcp = _fresh_mcp(monkeypatch)

        response = await _call(mcp, script="% inline\nCreate Spacecraft Sat\n")
        assert response.ok is False
        assert response.stderr == "GMAT error: bad burn duration\n"
        assert response.reports == []
        assert response.artefacts == []
        assert response.wall_clock.unit == "s"


class TestPreRunFailuresRaise:
    async def test_load_error_surfaces_as_upstream(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_gmat_run(
            monkeypatch,
            mission=_FakeMission(),  # never reached; load_error wins
            load_error=_FakeGmatLoadError("gmatpy missing"),
        )
        mcp = _fresh_mcp(monkeypatch)

        from mcp.server.fastmcp.exceptions import ToolError

        with pytest.raises(ToolError) as excinfo:
            await mcp.call_tool(
                "gmat_execute_script", {"script": "% inline\nCreate Spacecraft Sat\n"}
            )
        raw = str(excinfo.value)
        assert "upstream.gmat_run_load_failed" in raw

    async def test_bootstrap_error_surfaces_as_upstream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_gmat_run(
            monkeypatch,
            mission=_FakeMission(),
            load_error=_FakeGmatError("no install"),
        )
        mcp = _fresh_mcp(monkeypatch)

        from mcp.server.fastmcp.exceptions import ToolError

        with pytest.raises(ToolError) as excinfo:
            await mcp.call_tool(
                "gmat_execute_script", {"script": "% inline\nCreate Spacecraft Sat\n"}
            )
        assert "upstream.gmat_run_bootstrap_failed" in str(excinfo.value)

    async def test_relative_path_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_gmat_run(monkeypatch, mission=_FakeMission())
        mcp = _fresh_mcp(monkeypatch)

        from mcp.server.fastmcp.exceptions import ToolError

        with pytest.raises(ToolError) as excinfo:
            await mcp.call_tool("gmat_execute_script", {"script": "relative/path.script"})
        assert "invalid_input.script_path_not_absolute" in str(excinfo.value)

    async def test_non_positive_timeout_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_gmat_run(monkeypatch, mission=_FakeMission())
        mcp = _fresh_mcp(monkeypatch)

        from mcp.server.fastmcp.exceptions import ToolError

        with pytest.raises(ToolError) as excinfo:
            await mcp.call_tool(
                "gmat_execute_script",
                {"script": "% x\nCreate Spacecraft Sat\n", "timeout_seconds": -5},
            )
        assert "invalid_input.gmat_timeout_not_positive" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Round-trip parity with gmat_run_mission
# ---------------------------------------------------------------------------


class TestRoundTripWithRunMission:
    """The escape hatch and the curated tool must agree on the run-success
    boolean for the same fake mission. Successful run → both report success;
    failing run → both surface a failure (curated tool raises, escape hatch
    returns ok=False)."""

    async def test_both_agree_success(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rf = tmp_path / "RF.txt"
        _write_report(rf, lines=5)

        from astrodynamics_mcp.tools.gmat import GmatRunMissionResponse

        # Build a richer fake supporting both tools: gmat_run_mission needs
        # summary() / converged / reports-as-DataFrames, gmat_execute_script
        # needs output_dir / log / *_paths.
        class _DF:
            def __init__(self) -> None:
                self.columns = ["Sat.UTCGregorian", "Sat.X"]
                self.index = range(2)

            def to_numpy(self, *, dtype: Any = None) -> Any:
                del dtype
                return [["t0", 7000.0], ["t1", 7001.0]]

        class _RG:
            def __init__(self, category: str, names: tuple[str, ...]) -> None:
                self.category = category
                self.names = names

        class _Cmd:
            def __init__(self) -> None:
                self.type_name = "Propagate"
                self.summary = "Propagate Prop(Sat)"
                self.children = ()
                self.nested_count = 0

        class _Sum:
            def __init__(self) -> None:
                self.script_name = "fixture.script"
                self.resource_groups = (_RG("ReportFile", ("ReportFile1",)),)
                self.commands = (_Cmd(),)

        class _Result:
            def __init__(self) -> None:
                self.output_dir = tmp_path
                self.log = "ok\n"
                self.reports = {"ReportFile1": _DF()}
                self.report_paths = MappingProxyType({"ReportFile1": rf})
                empty: dict[str, Path] = {}
                self.ephemeris_paths = MappingProxyType(empty)
                self.contact_paths = MappingProxyType(empty)
                self.solver_paths = MappingProxyType(empty)
                self.converged: dict[str, bool] = {}

        result = _Result()

        class _Mission:
            def summary(self) -> _Sum:
                return _Sum()

            def run(self, *, working_dir: Any = None, overwrite: bool = False) -> _Result:
                del working_dir, overwrite
                return result

        _install_fake_gmat_run(monkeypatch, mission=_Mission())  # type: ignore[arg-type]
        mcp = _fresh_mcp(monkeypatch)

        execute_response = await _call(mcp, script="% inline\nCreate Spacecraft Sat\n")
        assert execute_response.ok is True

        _content, structured = await mcp.call_tool(
            "gmat_run_mission", {"script": "% inline\nCreate Spacecraft Sat\n"}
        )
        run_response = GmatRunMissionResponse.model_validate(structured)
        # The curated tool succeeded too — its converged dict is empty but the
        # response itself materialised, which is the run-success signal here.
        assert isinstance(run_response, GmatRunMissionResponse)

    async def test_both_agree_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        exc = _FakeGmatRunError("RunScript failed", log="GMAT error\n")

        class _Mission:
            def summary(self) -> Any:
                raise AssertionError("summary should not be called on the failure path")

            def run(self, *, working_dir: Any = None, overwrite: bool = False) -> Any:
                del working_dir, overwrite
                raise exc

        _install_fake_gmat_run(monkeypatch, mission=_Mission())  # type: ignore[arg-type]
        mcp = _fresh_mcp(monkeypatch)

        execute_response = await _call(mcp, script="% inline\nCreate Spacecraft Sat\n")
        assert execute_response.ok is False
        assert execute_response.stderr == "GMAT error\n"

        from mcp.server.fastmcp.exceptions import ToolError

        with pytest.raises(ToolError) as excinfo:
            await mcp.call_tool("gmat_run_mission", {"script": "% inline\nCreate Spacecraft Sat\n"})
        assert "upstream.gmat_run_failed" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Description / schema discipline (positive cases only — the negative
# placeholder paths are covered by the existing registration test).
# ---------------------------------------------------------------------------


class TestRegisteredSchema:
    async def test_response_schema_has_required_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_gmat_run(monkeypatch, mission=_FakeMission())
        mcp = _fresh_mcp(monkeypatch)
        tools = {t.name: t for t in await mcp.list_tools()}
        out_schema = tools["gmat_execute_script"].outputSchema
        assert out_schema is not None
        # Fish the response shape out of the schema's $defs, since FastMCP
        # wraps the pydantic model under a $ref by default.
        text = json.dumps(out_schema)
        for field in ("ok", "stderr", "wall_clock", "reports", "artefacts"):
            assert f'"{field}"' in text, f"missing {field} in output schema"


class TestSteeringText:
    """Regression guard for the description's 'prefer gmat_run_mission' framing.

    The escape hatch's whole point is to *not* be the LLM's first pick. If a
    future edit softens the steering, this test fails so the description gets
    re-tightened (or the test is updated deliberately).
    """

    def test_description_mentions_curated_alternative_first(self) -> None:
        from astrodynamics_mcp.tools.gmat import _EXECUTE_SCRIPT_DESCRIPTION as desc

        # The description must explicitly point at the curated tool by name
        # before describing the escape hatch's own use cases.
        assert "gmat_run_mission" in desc
        prefer_idx = desc.lower().find("prefer gmat_run_mission")
        assert prefer_idx >= 0, "description must say 'Prefer gmat_run_mission'"
        # The 'prefer' clause must come *before* the example, so the LLM reads
        # the steer before the call site.
        example_idx = desc.find("e.g. gmat_execute_script")
        assert example_idx > prefer_idx, (
            "steering text must precede the example call so the LLM weighs "
            "the curated alternative before reaching the example"
        )

    def test_description_documents_failures_as_data(self) -> None:
        from astrodynamics_mcp.tools.gmat import _EXECUTE_SCRIPT_DESCRIPTION as desc

        # The ok=False / stderr-as-data contract is the most surprising part of
        # the tool vs. the curated one — lock it into the description.
        assert "ok=False" in desc
        assert "stderr" in desc


# ---------------------------------------------------------------------------
# Chained producer → read seam
# ---------------------------------------------------------------------------


class TestChainedReadback:
    """Drives gmat_execute_script end-to-end, then reads a ReportFile back
    through gmat_read_run_artefact using the returned run_id. Validates
    that the producer registers in the same RunRegistry singleton the
    read tool consumes — a seam the schema-level checks can't see.
    """

    async def test_execute_script_then_read_reportfile(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from astrodynamics_mcp import runs as runs_module
        from astrodynamics_mcp.runs import RunRegistry
        from astrodynamics_mcp.tools.gmat import RawReportContent

        # Per-test registry installed as the singleton.
        registry = RunRegistry(directory=tmp_path / "cache", limit=5)
        monkeypatch.setattr(runs_module, "_default_registry", registry)

        fixture_text = "Sat.UTCGregorian Sat.X\n01 Jan 2026 12:00:00.000 7000.0\n"
        fixture_path = tmp_path / "ReportFile1.txt"
        # write_bytes, not write_text, so Windows doesn't translate \n
        # to \r\n — the byte-equality assertion below relies on identity.
        fixture_path.write_bytes(fixture_text.encode("utf-8"))

        result = _FakeResult(
            output_dir=tmp_path,
            log="GMAT log: mission complete\n",
            report_paths={"ReportFile1": fixture_path},
        )
        _install_fake_gmat_run(monkeypatch, mission=_FakeMission(run_result=result))
        mcp = _fresh_mcp(monkeypatch)

        _content, structured = await mcp.call_tool(
            "gmat_execute_script", {"script": "% inline\nCreate Spacecraft Sat\n"}
        )
        producer = GmatExecuteScriptResponse.model_validate(structured)
        assert producer.ok is True
        # The seam: the producer registered against this exact singleton.
        entry = registry.get(producer.run_id)
        assert entry is not None
        assert "ReportFile1" in entry.artefacts

        _content, structured = await mcp.call_tool(
            "gmat_read_run_artefact",
            {"run_id": producer.run_id, "name": "ReportFile1", "output": "full"},
        )
        readback = RawReportContent.model_validate(structured)
        assert readback.content == fixture_text
        assert readback.truncated is False
        assert readback.byte_count.value == float(fixture_path.stat().st_size)
