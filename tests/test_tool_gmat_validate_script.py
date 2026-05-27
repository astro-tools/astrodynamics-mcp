"""Tests for the ``gmat_validate_script`` tool.

Three layers:

1. :class:`TestParseGmatLog` — unit tests for ``_parse_gmat_log`` against
   golden log snippets captured from real GMAT R2026a runs. This is the
   primary CI signal because the integration tests below auto-skip when
   ``gmat_run`` is not installed (true in the bare CI env).

2. :class:`TestToolBody` — drives the tool body with fake
   ``gmat_run.install`` / ``gmat_run.runtime`` / ``gmat_run.mission`` /
   ``gmat_run.summary`` submodules injected via ``sys.modules``. The fake
   ``gmat`` module writes a synthetic log to the same temp path the body
   redirects ``UseLogFile`` to, so the parse pipeline exercises real file
   I/O without needing GMAT.

3. :class:`TestIntegrationAgainstRealGmat` — opt-in
   ``@pytest.mark.gmat_installed`` block: runs the registered tool against
   the real Linux GMAT install, covering the three acceptance cases from
   issue #82 (valid script → ok=True, unknown field → ok=False, missing
   BeginMissionSequence → ok=True with a warning).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from mcp.server.fastmcp import FastMCP

from astrodynamics_mcp.tools import gmat as gmat_tools
from astrodynamics_mcp.tools.gmat import (
    GmatValidateScriptResponse,
    ParseDiagnostic,
    _parse_gmat_log,
)

# ---------------------------------------------------------------------------
# Golden log snippets — captured from R2026a `UseLogFile` output across a
# handful of handcrafted scripts. Keep the literals raw so the regex contract
# is exercised against bytes-on-disk, not synthesised fixtures.
# ---------------------------------------------------------------------------


_LOG_VALID = """\
GMAT Build Date: Mar 30 2026 10:09:27

GMAT API Log: Created Wed May 27 16:38:58 2026


Logging to /tmp/foo/load.log

Interpreting scripts from the file.
***** file: /tmp/foo/probe.script
Successfully set Planetary Source to use: DE405
Successfully set Planetary Source to use: DE405
Successfully interpreted the script

Log closed: Wed May 27 16:38:58 2026

"""


_LOG_UNKNOWN_FIELD = (
    "GMAT Build Date: Mar 30 2026 10:09:27\n"
    "\n"
    "GMAT API Log: Created Wed May 27 16:39:44 2026\n"
    "\n"
    "\n"
    "Logging to /tmp/foo/load.log\n"
    "\n"
    "Interpreting scripts from the file.\n"
    "***** file: /tmp/foo/probe.script\n"
    "Successfully set Planetary Source to use: DE405\n"
    "Successfully set Planetary Source to use: DE405\n"
    "1: /tmp/foo/probe.script: **** ERROR **** Interpreter Exception: "
    'The field name "WidgetCount" on object "Sat" is not permitted in line:\n'
    '   "   3: Sat.WidgetCount = 7;"\n'
    "\n"
    "\n"
    "========================================\n"
    "\n"
    "Log closed: Wed May 27 16:39:44 2026\n"
    "\n"
)


_LOG_TWO_ERRORS = (
    "Interpreting scripts from the file.\n"
    "***** file: /tmp/foo/probe.script\n"
    "1: /tmp/foo/probe.script: **** ERROR **** Interpreter Exception: "
    'Cannot create an object "Sat". The "Spacecraf" is an unknown object '
    "type or invalid object name or dimension in line:\n"
    '   "   2: Create Spacecraf Sat;"\n'
    "\n"
    "2: /tmp/foo/probe.script: **** ERROR **** Interpreter Exception: "
    'Cannot find LHS object named "Sat" in line:\n'
    '   "   3: Sat.SMA = 7000;"\n'
    "\n"
    "\n"
    "========================================\n"
)


_LOG_UNDECLARED_REFERENCE = (
    "Interpreting scripts from the file.\n"
    "***** file: /tmp/foo/probe.script\n"
    "Interpreter Exception: /tmp/foo/probe.script: "
    'The ODEModel named "MissingFM", referenced by the Propagator "Prop" '
    "cannot be found\n"
    "\n"
    "Log closed: Wed May 27 16:39:44 2026\n"
)


_LOG_MISSING_BEGIN_SEQUENCE = (
    "Interpreting scripts from the file.\n"
    "***** file: /tmp/foo/probe.script\n"
    "Successfully interpreted the script\n"
    "*** WARNING ***  BeginMissionSequence command is missing. "
    "One will be required in future release. "
    "Command mode entered at 'Propagate Prop(Sat) {Sat.ElapsedSecs = 60};'\n"
    "\n"
    "\n"
    "Log closed: Wed May 27 16:39:44 2026\n"
)


class TestParseGmatLog:
    def test_valid_script_produces_no_diagnostics(self) -> None:
        errors, warnings = _parse_gmat_log(_LOG_VALID)
        assert errors == []
        assert warnings == []

    def test_unknown_field_carries_line_and_message(self) -> None:
        errors, warnings = _parse_gmat_log(_LOG_UNKNOWN_FIELD)
        assert warnings == []
        assert len(errors) == 1
        err = errors[0]
        assert err.line == 3
        assert 'field name "WidgetCount"' in err.message
        assert 'on object "Sat"' in err.message
        # ``raw`` carries the original marker line so callers can fall back on it.
        assert "**** ERROR ****" in err.raw

    def test_two_errors_both_surface_with_distinct_line_numbers(self) -> None:
        errors, warnings = _parse_gmat_log(_LOG_TWO_ERRORS)
        assert warnings == []
        assert len(errors) == 2
        assert errors[0].line == 2
        assert errors[1].line == 3
        assert '"Spacecraf"' in errors[0].message
        assert "LHS object" in errors[1].message

    def test_undeclared_reference_has_no_line_number(self) -> None:
        errors, warnings = _parse_gmat_log(_LOG_UNDECLARED_REFERENCE)
        assert warnings == []
        assert len(errors) == 1
        assert errors[0].line is None
        # Path prefix stripped — message starts at the substantive text.
        assert errors[0].message.startswith('The ODEModel named "MissingFM"')

    def test_warning_for_missing_begin_sequence(self) -> None:
        errors, warnings = _parse_gmat_log(_LOG_MISSING_BEGIN_SEQUENCE)
        assert errors == []
        assert len(warnings) == 1
        assert "BeginMissionSequence command is missing" in warnings[0].message
        assert warnings[0].line is None

    def test_empty_log_produces_no_diagnostics(self) -> None:
        errors, warnings = _parse_gmat_log("")
        assert errors == []
        assert warnings == []

    def test_header_only_log_produces_no_diagnostics(self) -> None:
        # Just header noise, no errors / warnings / interpreter output.
        errors, warnings = _parse_gmat_log(
            "GMAT Build Date: Mar 30 2026 10:09:27\n"
            "Logging to /tmp/foo/load.log\n"
            "Log closed: Wed May 27 16:39:44 2026\n"
        )
        assert errors == []
        assert warnings == []


# ---------------------------------------------------------------------------
# Tool-body unit tests — fake gmat_run.* submodules
# ---------------------------------------------------------------------------


class _FakeAPIException(Exception):
    """Stand-in for ``gmatpy.APIException`` — what the engine raises on bad init."""


class _FakeGmat:
    """Behaviour-compatible stand-in for the bootstrapped ``gmatpy`` module.

    Exposes the four entry points the tool body calls (``Clear``,
    ``UseLogFile``, ``LoadScript``, ``APIException``) plus an injectable
    log payload so each test can pin the parser's input.
    """

    APIException = _FakeAPIException

    def __init__(
        self,
        *,
        load_returns: bool = True,
        log_payload: str = "",
        raise_on_load: BaseException | None = None,
    ) -> None:
        self._load_returns = load_returns
        self._log_payload = log_payload
        self._raise_on_load = raise_on_load
        self._log_path: Path | None = None
        self.clear_called = False

    def Clear(self) -> None:
        self.clear_called = True

    def UseLogFile(self, path: str) -> None:
        # The tool body redirects to ``os.devnull`` after the load; only the
        # in-temp path is the one we want to materialise content into.
        self._log_path = Path(path)

    def LoadScript(self, path: str) -> bool:
        if self._raise_on_load is not None:
            raise self._raise_on_load
        # Materialise the canned log payload at the redirected path so the
        # tool body's ``read_text`` exercises real file I/O.
        if self._log_path is not None:
            self._log_path.write_text(self._log_payload, encoding="utf-8")
        return self._load_returns


def _fake_install() -> SimpleNamespace:
    """Minimal stand-in for ``gmat_run.install.GmatInstall``."""
    return SimpleNamespace(root=Path("/fake/gmat"))


def _fake_summary(script_path: Path) -> SimpleNamespace:
    """Minimal MissionSummary-shaped object for the response-render path."""
    return SimpleNamespace(
        script_name=script_path.name,
        resource_groups=[
            SimpleNamespace(category="Spacecraft", names=["Sat"]),
            SimpleNamespace(category="ForceModel", names=["FM"]),
        ],
        commands=[
            SimpleNamespace(
                type_name="Propagate",
                summary="Propagate Prop(Sat) {Sat.ElapsedSecs = 60}",
                children=[],
                nested_count=0,
            )
        ],
    )


def _install_fake_gmat_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fake_gmat: _FakeGmat,
    init_raises: BaseException | None = None,
    summary_factory: Any = _fake_summary,
) -> None:
    """Inject fake ``gmat_run.*`` submodules covering the tool body's imports."""
    fake_install_mod = ModuleType("gmat_run.install")
    fake_install_mod.__dict__["locate_gmat"] = lambda: _fake_install()

    fake_runtime_mod = ModuleType("gmat_run.runtime")
    fake_runtime_mod.__dict__["bootstrap"] = lambda _install: fake_gmat

    fake_mission_mod = ModuleType("gmat_run.mission")
    fake_mission_mod.__dict__["_get_api_exception"] = lambda _gmat: _FakeAPIException

    def _fake_initialize_spacecraft(_gmat: Any) -> None:
        if init_raises is not None:
            raise init_raises

    fake_mission_mod.__dict__["_initialize_spacecraft"] = _fake_initialize_spacecraft

    fake_summary_mod = ModuleType("gmat_run.summary")
    fake_summary_mod.__dict__["build_mission_summary"] = lambda _gmat, path: summary_factory(path)

    monkeypatch.setitem(sys.modules, "gmat_run.install", fake_install_mod)
    monkeypatch.setitem(sys.modules, "gmat_run.runtime", fake_runtime_mod)
    monkeypatch.setitem(sys.modules, "gmat_run.mission", fake_mission_mod)
    monkeypatch.setitem(sys.modules, "gmat_run.summary", fake_summary_mod)


def _fresh_mcp(monkeypatch: pytest.MonkeyPatch) -> FastMCP:
    fresh = FastMCP("gmat-validate-script-test")
    monkeypatch.setattr("astrodynamics_mcp.server.mcp", fresh)
    monkeypatch.setattr(gmat_tools, "_GMAT_RUN_AVAILABLE", True)
    gmat_tools._register_gmat_tools()
    return fresh


async def _call(mcp: FastMCP, **args: Any) -> GmatValidateScriptResponse:
    _content, structured = await mcp.call_tool("gmat_validate_script", args)
    return GmatValidateScriptResponse.model_validate(structured)


class TestToolBody:
    async def test_valid_script_returns_ok_with_summary(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        script = tmp_path / "ok.script"
        script.write_text("Create Spacecraft Sat\nBeginMissionSequence\n")
        fake = _FakeGmat(load_returns=True, log_payload=_LOG_VALID)
        _install_fake_gmat_run(monkeypatch, fake_gmat=fake)
        mcp = _fresh_mcp(monkeypatch)

        result = await _call(mcp, script=str(script))

        assert result.ok is True
        assert result.errors == []
        assert result.warnings == []
        assert result.summary is not None
        assert result.summary.script_name == "ok.script"
        assert any(g.category == "Spacecraft" for g in result.summary.resource_groups)
        assert result.raw_log == _LOG_VALID
        assert fake.clear_called is True

    async def test_load_failure_returns_ok_false_with_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        script = tmp_path / "bad.script"
        script.write_text("Create Spacecraft Sat\nSat.WidgetCount = 7\nBeginMissionSequence\n")
        fake = _FakeGmat(load_returns=False, log_payload=_LOG_UNKNOWN_FIELD)
        _install_fake_gmat_run(monkeypatch, fake_gmat=fake)
        mcp = _fresh_mcp(monkeypatch)

        result = await _call(mcp, script=str(script))

        assert result.ok is False
        # Summary deliberately suppressed on parse failure — moderator state
        # after a failed LoadScript is indeterminate.
        assert result.summary is None
        assert len(result.errors) == 1
        assert result.errors[0].line == 3
        assert "WidgetCount" in result.errors[0].message

    async def test_initialize_failure_marks_ok_false_and_appends_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        script = tmp_path / "ecc.script"
        script.write_text("Create Spacecraft Sat\nBeginMissionSequence\n")
        # LoadScript itself succeeds (clean log) — the failure happens in
        # _initialize_spacecraft, mimicking GMAT's "ECC > 1" APIException.
        fake = _FakeGmat(load_returns=True, log_payload=_LOG_VALID)
        _install_fake_gmat_run(
            monkeypatch,
            fake_gmat=fake,
            init_raises=_FakeAPIException("Spacecraft 'Sat': ECC > 1"),
        )
        mcp = _fresh_mcp(monkeypatch)

        result = await _call(mcp, script=str(script))

        assert result.ok is False
        assert result.summary is None
        assert len(result.errors) == 1
        assert "ECC > 1" in result.errors[0].message
        # The init-error entry uses the exception text as both ``message``
        # and ``raw`` since there is no log line to point at.
        assert result.errors[0].line is None

    async def test_warning_does_not_flip_ok_to_false(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        script = tmp_path / "warn.script"
        script.write_text("Create Spacecraft Sat\nPropagate Prop(Sat) {Sat.ElapsedSecs = 60}\n")
        fake = _FakeGmat(load_returns=True, log_payload=_LOG_MISSING_BEGIN_SEQUENCE)
        _install_fake_gmat_run(monkeypatch, fake_gmat=fake)
        mcp = _fresh_mcp(monkeypatch)

        result = await _call(mcp, script=str(script))

        # Warnings don't gate ``ok`` — the script still loaded.
        assert result.ok is True
        assert result.summary is not None
        assert len(result.warnings) == 1
        assert "BeginMissionSequence" in result.warnings[0].message

    async def test_inline_script_text_auto_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeGmat(load_returns=True, log_payload=_LOG_VALID)
        _install_fake_gmat_run(monkeypatch, fake_gmat=fake)
        mcp = _fresh_mcp(monkeypatch)

        inline = "% inline test\nCreate Spacecraft Sat\nBeginMissionSequence\n"
        result = await _call(mcp, script=inline)
        assert result.ok is True
        # The summary script_name is derived from the temp file the helper
        # writes; just confirm something landed there.
        assert result.summary is not None
        assert result.summary.script_name.endswith(".script")

    async def test_response_roundtrips_through_output_schema(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Mirror the cross-tool round-trip discipline locally.

        The cross-tool ``SAMPLE_CALLS`` table excludes every gmat_* tool
        because they need real GMAT to invoke. The contract still has to
        hold for the validate response — exercise it here against the fake.
        """
        script = tmp_path / "ok.script"
        script.write_text("Create Spacecraft Sat\nBeginMissionSequence\n")
        fake = _FakeGmat(load_returns=True, log_payload=_LOG_VALID)
        _install_fake_gmat_run(monkeypatch, fake_gmat=fake)
        mcp = _fresh_mcp(monkeypatch)

        result = await _call(mcp, script=str(script))
        first = result.model_dump_json()
        rebuilt = GmatValidateScriptResponse.model_validate_json(first)
        assert rebuilt.model_dump_json() == first


# ---------------------------------------------------------------------------
# Description-lint — registered description must satisfy the static rules
# ---------------------------------------------------------------------------


class TestDescriptionLint:
    async def test_description_passes_lint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from astrodynamics_mcp.server_lint import check_tool_descriptions

        fake = _FakeGmat(load_returns=True, log_payload=_LOG_VALID)
        _install_fake_gmat_run(monkeypatch, fake_gmat=fake)
        mcp = _fresh_mcp(monkeypatch)
        tools = [t for t in await mcp.list_tools() if t.name == "gmat_validate_script"]
        assert tools, "gmat_validate_script slot missing from the fresh surface"
        assert check_tool_descriptions(tools) == []


# ---------------------------------------------------------------------------
# Direct ParseDiagnostic round-trip — schema-level invariants
# ---------------------------------------------------------------------------


class TestParseDiagnosticModel:
    def test_line_can_be_none(self) -> None:
        d = ParseDiagnostic(line=None, message="msg", raw="raw")
        # frozen=True: assignment raises rather than silently mutating.
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            d.line = 1

    def test_extra_keys_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ParseDiagnostic.model_validate(
                {"line": None, "message": "m", "raw": "r", "unexpected": "x"}
            )


# ---------------------------------------------------------------------------
# Integration: real GMAT install (auto-skipped when gmat_run missing)
# ---------------------------------------------------------------------------


_VALID_FIXTURE_SCRIPT = Path(__file__).parent / "data" / "gmat_minimal_leo.script"
_UNKNOWN_FIELD_FIXTURE = Path(__file__).parent / "data" / "gmat_unknown_field.script"
_NO_BEGIN_SEQUENCE_FIXTURE = Path(__file__).parent / "data" / "gmat_missing_begin_sequence.script"


@pytest.mark.gmat_installed
@pytest.mark.skipif(
    importlib.util.find_spec("gmat_run") is None,
    reason="gmat_run is not installed; install the [gmat] extra to run this test",
)
class TestIntegrationAgainstRealGmat:
    """End-to-end: invoke the tool against the real Linux GMAT install."""

    async def test_valid_script_parses_ok(self) -> None:
        from astrodynamics_mcp.server import mcp

        _content, structured = await mcp.call_tool(
            "gmat_validate_script", {"script": str(_VALID_FIXTURE_SCRIPT)}
        )
        parsed = GmatValidateScriptResponse.model_validate(structured)
        assert parsed.ok is True
        assert parsed.errors == []
        assert parsed.summary is not None
        categories = {g.category for g in parsed.summary.resource_groups}
        assert {"Spacecraft", "ForceModel", "Propagator"}.issubset(categories)

    async def test_unknown_field_returns_error_with_line(self) -> None:
        from astrodynamics_mcp.server import mcp

        _content, structured = await mcp.call_tool(
            "gmat_validate_script", {"script": str(_UNKNOWN_FIELD_FIXTURE)}
        )
        parsed = GmatValidateScriptResponse.model_validate(structured)
        assert parsed.ok is False
        assert parsed.summary is None
        assert any(e.line is not None and "WidgetCount" in e.message for e in parsed.errors), (
            f"no actionable WidgetCount error in {parsed.errors!r}"
        )

    async def test_missing_begin_sequence_returns_warning(self) -> None:
        from astrodynamics_mcp.server import mcp

        _content, structured = await mcp.call_tool(
            "gmat_validate_script", {"script": str(_NO_BEGIN_SEQUENCE_FIXTURE)}
        )
        parsed = GmatValidateScriptResponse.model_validate(structured)
        # The script parses (no errors) but emits a warning.
        assert parsed.ok is True
        assert any("BeginMissionSequence" in w.message for w in parsed.warnings), (
            f"no BeginMissionSequence warning in {parsed.warnings!r}"
        )
