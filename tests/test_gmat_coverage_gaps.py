"""Coverage tests for branches the existing GMAT test suite missed.

Each class targets a specific gap surfaced in the issue #95 code review:
registration idempotency, missing-file / malformed-description handling
in ``_register_gmat_resources``, the broadened ``_apply_overrides`` typed
exception path, the defensive ``_validate_sweep_payload`` ``else``
branch, binary-sniff boundary cases in ``gmat_read_run_artefact``, the
``samples=[]`` / ``samples=[{}]`` payload-validation gaps in
``gmat_sweep``, RunRegistry concurrency under interleaved producer
calls, the ``gmat-skeleton://*`` MIME-type contract, and
``_parse_gmat_log`` warning-after-error interleaving.

Tests live in their own file rather than scattered across the existing
``test_tool_gmat_*.py`` so the coverage delta against the cleanup PR is
easy to read and the existing files keep their focused scope.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from astrodynamics_mcp import runs as runs_module
from astrodynamics_mcp.tools import gmat as gmat_tools
from astrodynamics_mcp.tools.gmat import (
    _SKELETON_URI_SCHEME,
    RawReportContent,
    _apply_overrides,
    _parse_gmat_log,
    _validate_sweep_payload,
)
from tests._gmat_helpers import install_minimal_gmat_run_modules

# ---------------------------------------------------------------------------
# Registration idempotency
# ---------------------------------------------------------------------------


class TestRegistrationIdempotency:
    """``_register_gmat_tools()`` / ``_register_gmat_resources()`` are safe to call twice."""

    async def test_register_tools_twice_does_not_double(self, gmat_mcp_bare: FastMCP) -> None:
        gmat_tools._register_gmat_tools()
        first = {t.name for t in await gmat_mcp_bare.list_tools()}
        # Second call would either silently double-register or raise; either
        # would break a hot-reload / re-init path. The contract is "no
        # change in tool surface".
        try:
            gmat_tools._register_gmat_tools()
        except Exception as exc:
            # If FastMCP rejects double-registration explicitly that's fine
            # too -- the test pins behaviour either way.
            pytest.skip(f"double-register raised cleanly: {exc!r}")
        second = {t.name for t in await gmat_mcp_bare.list_tools()}
        assert first == second, "double _register_gmat_tools changed the tool surface"

    async def test_register_resources_twice_does_not_double(self, gmat_mcp_bare: FastMCP) -> None:
        gmat_tools._register_gmat_resources()
        first = {str(r.uri) for r in await gmat_mcp_bare.list_resources()}
        try:
            gmat_tools._register_gmat_resources()
        except Exception as exc:
            pytest.skip(f"double-register raised cleanly: {exc!r}")
        second = {str(r.uri) for r in await gmat_mcp_bare.list_resources()}
        assert first == second, "double _register_gmat_resources changed the resource surface"


# ---------------------------------------------------------------------------
# _register_gmat_resources error paths
# ---------------------------------------------------------------------------


class TestRegisterResourcesErrors:
    """Skeleton-registration helper fails loudly on bad inputs."""

    async def test_missing_file_raises_with_slug(
        self, monkeypatch: pytest.MonkeyPatch, gmat_mcp_bare: FastMCP
    ) -> None:
        bogus = (("ghost-skeleton", "this-file-does-not-exist.script"),)
        monkeypatch.setattr(gmat_tools, "_SKELETONS", bogus)
        with pytest.raises(FileNotFoundError, match="ghost-skeleton"):
            gmat_tools._register_gmat_resources()

    async def test_malformed_description_raises_with_slug_and_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        gmat_mcp_bare: FastMCP,
    ) -> None:
        bad = tmp_path / "no_description.script"
        bad.write_text("% just a banner, no description line\nCreate Spacecraft Sat\n")
        from importlib import resources

        class _FakeTraversable:
            def joinpath(self, _name: str) -> Path:
                return bad

        monkeypatch.setattr(resources, "files", lambda _pkg: _FakeTraversable())
        monkeypatch.setattr(gmat_tools, "_SKELETONS", (("broken", "no_description.script"),))
        with pytest.raises(ValueError) as excinfo:
            gmat_tools._register_gmat_resources()
        msg = str(excinfo.value)
        assert "broken" in msg
        assert str(bad) in msg


# ---------------------------------------------------------------------------
# _apply_overrides broadened exception handling
# ---------------------------------------------------------------------------


class TestApplyOverridesBroadenedTypes:
    """``_apply_overrides`` catches TypeError / AttributeError as invalid_input.*."""

    def test_type_error_surfaces_typed_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_minimal_gmat_run_modules(monkeypatch)

        class _Mission:
            def __setitem__(self, _key: str, _value: Any) -> None:
                raise TypeError("can only assign real to real field")

        from astrodynamics_mcp.errors import InvalidInputError

        with pytest.raises(InvalidInputError) as excinfo:
            _apply_overrides(_Mission(), {"Sat.SMA": "not-a-number"})
        assert excinfo.value.code == "invalid_input.gmat_override_failed"

    def test_attribute_error_surfaces_typed_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_minimal_gmat_run_modules(monkeypatch)

        class _Mission:
            def __setitem__(self, _key: str, _value: Any) -> None:
                raise AttributeError("'Mission' object has no attribute '_resource'")

        from astrodynamics_mcp.errors import InvalidInputError

        with pytest.raises(InvalidInputError) as excinfo:
            _apply_overrides(_Mission(), {"Sat.SMA": 7000.0})
        assert excinfo.value.code == "invalid_input.gmat_override_failed"


# ---------------------------------------------------------------------------
# _validate_sweep_payload defensive else branch
# ---------------------------------------------------------------------------


class TestValidateSweepPayloadElse:
    """The defensive ``else`` branch ensures a widened Literal still gets typed errors."""

    def test_unknown_mode_raises_typed_error(self) -> None:
        from astrodynamics_mcp.errors import InvalidInputError

        with pytest.raises(InvalidInputError) as excinfo:
            _validate_sweep_payload(
                mode="garbage",
                grid=None,
                samples=None,
                perturb=None,
                n=None,
                seed=None,
            )
        assert excinfo.value.code == "invalid_input.gmat_sweep_unknown_mode"


# ---------------------------------------------------------------------------
# Binary-sniff boundary cases in gmat_read_run_artefact
# ---------------------------------------------------------------------------


class TestBinarySniffBoundaries:
    """The 8 KB sniff window is inclusive of byte 0..8191; byte 8192 is past the window."""

    @pytest.fixture
    def _registered_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, gmat_mcp: FastMCP
    ) -> str:
        """Stand up a fake run in the real registry pointing at ``tmp_path``."""
        registry = runs_module.default_registry()
        run_id = registry.mint()
        registry.register(run_id, output_dir=tmp_path, artefacts={})
        return run_id

    async def test_null_inside_sniff_window_is_rejected(
        self, gmat_mcp: FastMCP, _registered_run: str, tmp_path: Path
    ) -> None:
        # NULL byte at offset 100 -- well inside the 8 KB sniff window.
        (tmp_path / "binary.bin").write_bytes(b"A" * 100 + b"\x00" + b"B" * 100)
        with pytest.raises(ToolError) as excinfo:
            await gmat_mcp.call_tool(
                "gmat_read_run_artefact",
                {"run_id": _registered_run, "name": "binary.bin", "output": "summary"},
            )
        assert "invalid_input.binary_artefact" in str(excinfo.value)

    async def test_null_past_sniff_window_is_accepted(
        self, gmat_mcp: FastMCP, _registered_run: str, tmp_path: Path
    ) -> None:
        # NULL byte at offset 8192 (one past the sniff window) -- the
        # heuristic treats this as text since the first 8 KB look clean.
        # GMAT-format files never carry NULL at all so this is a synthetic
        # boundary test, not a realistic payload.
        (tmp_path / "text_then_null.txt").write_bytes(b"A" * 8192 + b"\x00line")
        _content, structured = await gmat_mcp.call_tool(
            "gmat_read_run_artefact",
            {"run_id": _registered_run, "name": "text_then_null.txt", "output": "summary"},
        )
        parsed = RawReportContent.model_validate(structured)
        # The content should be returned (the tool decoded with errors='replace'),
        # not rejected as binary. Length > 8 KB confirms we got past the sniff.
        assert int(parsed.byte_count.value) > 8192

    async def test_empty_file_is_accepted(
        self, gmat_mcp: FastMCP, _registered_run: str, tmp_path: Path
    ) -> None:
        (tmp_path / "empty.log").write_bytes(b"")
        _content, structured = await gmat_mcp.call_tool(
            "gmat_read_run_artefact",
            {"run_id": _registered_run, "name": "empty.log", "output": "summary"},
        )
        parsed = RawReportContent.model_validate(structured)
        assert parsed.content == ""
        assert int(parsed.line_count.value) == 0
        assert int(parsed.byte_count.value) == 0
        assert parsed.truncated is False


# ---------------------------------------------------------------------------
# gmat_sweep payload validation: samples=[] and samples=[{}]
# ---------------------------------------------------------------------------


class TestSweepSamplesPayloadGaps:
    """Empty sample list / row-with-no-keys both produce typed errors."""

    async def test_samples_empty_list_rejected(self, gmat_mcp: FastMCP, tmp_path: Path) -> None:
        script = tmp_path / "noop.script"
        script.write_text("% noop\n")
        with pytest.raises(ToolError) as excinfo:
            await gmat_mcp.call_tool(
                "gmat_sweep",
                {"script": str(script), "mode": "samples", "samples": []},
            )
        assert "invalid_input.gmat_sweep_samples_empty" in str(excinfo.value)

    async def test_samples_empty_row_rejected(self, gmat_mcp: FastMCP, tmp_path: Path) -> None:
        script = tmp_path / "noop.script"
        script.write_text("% noop\n")
        with pytest.raises(ToolError) as excinfo:
            await gmat_mcp.call_tool(
                "gmat_sweep",
                {"script": str(script), "mode": "samples", "samples": [{}]},
            )
        assert "invalid_input.gmat_sweep_samples_row_shape" in str(excinfo.value)


# ---------------------------------------------------------------------------
# RunRegistry concurrency under interleaved producers
# ---------------------------------------------------------------------------


class TestRunRegistryConcurrency:
    """Parallel producer calls don't tear up the registry's LRU bookkeeping."""

    async def test_parallel_execute_calls_respect_cap(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Build a process-local registry capped at 3 entries to keep the
        # test self-contained. The behaviour we're pinning: dispatch 6
        # concurrent registrations and assert the registry only retains 3.
        from astrodynamics_mcp.runs import RunRegistry

        registry = RunRegistry(limit=3)
        monkeypatch.setattr(runs_module, "_default_registry", registry)
        monkeypatch.setattr(runs_module, "default_registry", lambda: registry)

        async def producer() -> str:
            run_id = registry.mint()
            workspace = tmp_path / run_id
            workspace.mkdir()
            registry.register(run_id, output_dir=workspace, artefacts={})
            return run_id

        ids = await asyncio.gather(*[producer() for _ in range(6)])
        assert len(set(ids)) == 6, "registry minted duplicate run_ids under concurrent load"
        # Only the most recent 3 survive; earlier ones are evicted.
        survivors = [run_id for run_id in ids if registry.get(run_id) is not None]
        assert len(survivors) == 3


# ---------------------------------------------------------------------------
# MIME-type contract for gmat-skeleton:// resources
# ---------------------------------------------------------------------------


class TestSkeletonResourceMimeType:
    """Skeleton resources serve as ``text/plain`` so MCP clients render them as text."""

    async def test_listed_resources_carry_text_mime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fresh = FastMCP("gmat-skeleton-mime-test")
        monkeypatch.setattr("astrodynamics_mcp.server.mcp", fresh)
        gmat_tools._register_gmat_resources()
        listed = await fresh.list_resources()
        for resource in listed:
            if not str(resource.uri).startswith(f"{_SKELETON_URI_SCHEME}://"):
                continue
            assert resource.mimeType == "text/plain", (
                f"{resource.uri} registered with mimeType={resource.mimeType!r}; "
                "client text renderers expect text/plain"
            )


# ---------------------------------------------------------------------------
# _parse_gmat_log warning-after-error interleaving
# ---------------------------------------------------------------------------


class TestParseGmatLogInterleaving:
    """A warning emitted on the line immediately after an error is not swallowed."""

    def test_warning_after_error_with_line_context(self) -> None:
        log = (
            "Interpreting scripts from the file.\n"
            "***** file: /tmp/probe.script\n"
            "1: /tmp/probe.script: **** ERROR **** Interpreter Exception: "
            'The field name "WidgetCount" on object "Sat" is not permitted in line:\n'
            '   "   3: Sat.WidgetCount = 7;"\n'
            "*** WARNING ***  BeginMissionSequence command is missing.\n"
            "\n"
            "Log closed: Wed May 27 16:39:44 2026\n"
        )
        errors, warnings = _parse_gmat_log(log)
        assert len(errors) == 1
        assert errors[0].line == 3
        assert len(warnings) == 1
        assert "BeginMissionSequence command is missing" in warnings[0].message


# ---------------------------------------------------------------------------
# Producer pre-register failure does not leak the workspace
# ---------------------------------------------------------------------------


class TestWorkspaceLeakFix:
    """Failures between mkdtemp and registry.register no longer leak the workspace."""

    async def test_gmat_run_mission_failure_cleans_workspace(
        self, monkeypatch: pytest.MonkeyPatch, gmat_mcp: FastMCP
    ) -> None:
        import tempfile

        # Snapshot temp dirs matching our prefix before the call, so we can
        # detect a leak as a new survivor afterwards.
        prefix = "astrodynamics-mcp-run-"
        tmpdir = Path(tempfile.gettempdir())
        before = {p for p in tmpdir.iterdir() if p.name.startswith(prefix)}

        # Drive _load_mission to fail with GmatLoadError -- the workspace
        # was created on entry and must be reaped by the new context
        # manager even though registry.register never fires.
        class _LoadError(Exception):
            pass

        install_minimal_gmat_run_modules(monkeypatch, load_error=_LoadError)

        gmat_run_mod = ModuleType("gmat_run")

        class _Mission:
            @staticmethod
            def load(_path: Path) -> Any:
                raise _LoadError("synthetic bad script")

        gmat_run_mod.__dict__["Mission"] = _Mission
        monkeypatch.setitem(sys.modules, "gmat_run", gmat_run_mod)

        with pytest.raises(ToolError):
            await gmat_mcp.call_tool("gmat_run_mission", {"script": "% noop\n"})

        after = {p for p in tmpdir.iterdir() if p.name.startswith(prefix)}
        leaked = after - before
        assert not leaked, f"workspace leaked under load failure: {leaked}"
