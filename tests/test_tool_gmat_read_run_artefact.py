"""Tests for the ``gmat_read_run_artefact`` tool.

The producer tools are exercised with a fake ``gmat_run`` (see the sibling
``test_tool_gmat_*`` files). Here we drive the read tool directly against
a per-test :class:`RunRegistry` injected as the module singleton, so the
resolution paths (resource-name hit, basename fallback, unknown id,
unknown name, evicted-but-known) are covered without re-spinning the
producers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from astrodynamics_mcp import runs as runs_module
from astrodynamics_mcp.runs import RunRegistry
from astrodynamics_mcp.tools import gmat as gmat_tools
from astrodynamics_mcp.tools.gmat import RawReportContent

# ---------------------------------------------------------------------------
# Per-test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunRegistry:
    """A fresh registry rooted at tmp_path, installed as the singleton."""
    reg = RunRegistry(directory=tmp_path / "cache", limit=5)
    monkeypatch.setattr(runs_module, "_default_registry", reg)
    return reg


@pytest.fixture
def mcp_server(monkeypatch: pytest.MonkeyPatch) -> FastMCP:
    """Fresh FastMCP with the gmat tool slots registered against it."""
    fresh = FastMCP("read-run-artefact-test")
    monkeypatch.setattr("astrodynamics_mcp.server.mcp", fresh)
    monkeypatch.setattr(gmat_tools, "_GMAT_RUN_AVAILABLE", True)
    gmat_tools._register_gmat_tools()
    return fresh


def _make_run(parent: Path, slug: str, artefact_text: str = "row\nrow\n") -> Path:
    """Materialise a synthetic output directory under ``parent``.

    Returns the directory path. Always writes a ``ReportFile1.txt`` and a
    stray ``GMAT.log`` so both the declared-name and basename-fallback
    resolution paths can be exercised.
    """
    parent.mkdir(parents=True, exist_ok=True)
    out = parent / slug
    out.mkdir()
    (out / "ReportFile1.txt").write_text(f"Sat.UTCGregorian\n{artefact_text}", encoding="utf-8")
    (out / "GMAT.log").write_text("noise\n", encoding="utf-8")
    return out


async def _call(mcp: FastMCP, **args: Any) -> RawReportContent:
    """Invoke ``gmat_read_run_artefact`` through the FastMCP wire."""
    _content, structured = await mcp.call_tool("gmat_read_run_artefact", args)
    return RawReportContent.model_validate(structured)


# ---------------------------------------------------------------------------
# Resolution paths
# ---------------------------------------------------------------------------


class TestResourceNameResolution:
    async def test_declared_name_returns_content(
        self, tmp_path: Path, registry: RunRegistry, mcp_server: FastMCP
    ) -> None:
        out = _make_run(tmp_path / "workspaces", "run-a", artefact_text="alpha\nbeta\n")
        run_id = registry.mint()
        registry.register(
            run_id,
            output_dir=out,
            artefacts={"ReportFile1": out / "ReportFile1.txt"},
        )
        response = await _call(mcp_server, run_id=run_id, name="ReportFile1")
        assert response.name == "ReportFile1"
        assert response.path == str(out / "ReportFile1.txt")
        assert "alpha" in response.content
        assert "beta" in response.content
        assert response.truncated is False
        assert response.line_count.unit == "1"
        assert response.byte_count.unit == "1"

    async def test_full_mode_returns_every_line(
        self, tmp_path: Path, registry: RunRegistry, mcp_server: FastMCP
    ) -> None:
        out = tmp_path / "workspaces" / "run-a"
        out.mkdir(parents=True)
        # 1 header + 200 data rows: under summary the response carries head/tail.
        rows = "\n".join(f"row{i}" for i in range(200))
        (out / "ReportFile1.txt").write_text("Sat.UTCGregorian\n" + rows + "\n")
        run_id = registry.mint()
        registry.register(
            run_id,
            output_dir=out,
            artefacts={"ReportFile1": out / "ReportFile1.txt"},
        )
        summary = await _call(mcp_server, run_id=run_id, name="ReportFile1")
        assert summary.truncated is True
        assert summary.content == ""

        full = await _call(mcp_server, run_id=run_id, name="ReportFile1", output="full")
        assert full.truncated is False
        assert "row0" in full.content
        assert "row199" in full.content


class TestBasenameFallback:
    async def test_basename_resolves_undeclared_file(
        self, tmp_path: Path, registry: RunRegistry, mcp_server: FastMCP
    ) -> None:
        """``GMAT.log`` is not in the registered artefact map yet must resolve."""
        out = _make_run(tmp_path / "workspaces", "run-a")
        run_id = registry.mint()
        registry.register(
            run_id,
            output_dir=out,
            artefacts={"ReportFile1": out / "ReportFile1.txt"},
        )
        response = await _call(mcp_server, run_id=run_id, name="GMAT.log")
        assert response.name == "GMAT.log"
        assert response.path == str(out / "GMAT.log")
        assert "noise" in response.content

    async def test_basename_subdirectory_not_reached(
        self, tmp_path: Path, registry: RunRegistry, mcp_server: FastMCP
    ) -> None:
        """Files nested under a subdir need a declared name; basename is non-recursive."""
        out = _make_run(tmp_path / "workspaces", "run-a")
        nested = out / "Solver"
        nested.mkdir()
        (nested / "DC.data").write_text("iter\n")
        run_id = registry.mint()
        registry.register(run_id, output_dir=out, artefacts={})
        with pytest.raises(ToolError) as excinfo:
            await mcp_server.call_tool(
                "gmat_read_run_artefact",
                {"run_id": run_id, "name": "DC.data"},
            )
        assert "invalid_input.unknown_artefact_name" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Typed-error surfaces
# ---------------------------------------------------------------------------


class TestErrorShapes:
    async def test_unknown_run_id_surfaces_known_set(
        self, tmp_path: Path, registry: RunRegistry, mcp_server: FastMCP
    ) -> None:
        out = _make_run(tmp_path / "workspaces", "run-a")
        known = registry.mint()
        registry.register(known, output_dir=out, artefacts={})
        with pytest.raises(ToolError) as excinfo:
            await mcp_server.call_tool(
                "gmat_read_run_artefact",
                {"run_id": "deadbeef" * 4, "name": "ReportFile1"},
            )
        raw = str(excinfo.value)
        assert "invalid_input.unknown_run_id" in raw
        # The known set comes back in `data` for the LLM to recover from.
        payload_start = raw.find("{")
        envelope = json.loads(raw[payload_start:])
        assert known in envelope["data"]["known_run_ids"]

    async def test_unknown_name_surfaces_available_sets(
        self, tmp_path: Path, registry: RunRegistry, mcp_server: FastMCP
    ) -> None:
        out = _make_run(tmp_path / "workspaces", "run-a")
        run_id = registry.mint()
        registry.register(
            run_id,
            output_dir=out,
            artefacts={"ReportFile1": out / "ReportFile1.txt"},
        )
        with pytest.raises(ToolError) as excinfo:
            await mcp_server.call_tool(
                "gmat_read_run_artefact",
                {"run_id": run_id, "name": "NotARealResource"},
            )
        raw = str(excinfo.value)
        assert "invalid_input.unknown_artefact_name" in raw
        envelope = json.loads(raw[raw.find("{") :])
        assert envelope["data"]["available_resource_names"] == ["ReportFile1"]
        # Both files in the dir surface as available basenames.
        assert set(envelope["data"]["available_basenames"]) == {
            "ReportFile1.txt",
            "GMAT.log",
        }

    async def test_evicted_artefact_surfaces_typed_code(
        self, tmp_path: Path, registry: RunRegistry, mcp_server: FastMCP
    ) -> None:
        """A registered run whose temp directory was reaped between calls."""
        out = _make_run(tmp_path / "workspaces", "run-a")
        run_id = registry.mint()
        registry.register(
            run_id,
            output_dir=out,
            artefacts={"ReportFile1": out / "ReportFile1.txt"},
        )
        # Simulate an external reaper (systemd-tmpfiles, manual rm -rf):
        # remove just the registered artefact path, leaving the entry in
        # the registry pointing at a stale location.
        (out / "ReportFile1.txt").unlink()
        with pytest.raises(ToolError) as excinfo:
            await mcp_server.call_tool(
                "gmat_read_run_artefact",
                {"run_id": run_id, "name": "ReportFile1"},
            )
        assert "invalid_input.artefact_evicted" in str(excinfo.value)
        # Eager eviction: the dead entry is dropped right away so a
        # subsequent call surfaces unknown_run_id rather than spinning
        # on the same stale path.
        assert registry.get(run_id) is None
        assert run_id not in registry.known_run_ids()


# ---------------------------------------------------------------------------
# Issue #88 acceptance criteria
# ---------------------------------------------------------------------------


class TestIssueAcceptance:
    async def test_eviction_clears_artefacts_and_returns_unknown_run_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Acceptance: 51st run on a fresh server removes the 1st run's dir.

        Cap of 2 instead of 50 keeps the test fast; the LRU contract is
        the same.
        """
        reg = RunRegistry(directory=tmp_path / "cache", limit=2)
        monkeypatch.setattr(runs_module, "_default_registry", reg)
        fresh = FastMCP("acceptance-test")
        monkeypatch.setattr("astrodynamics_mcp.server.mcp", fresh)
        monkeypatch.setattr(gmat_tools, "_GMAT_RUN_AVAILABLE", True)
        gmat_tools._register_gmat_tools()

        first_id = reg.mint()
        first_dir = _make_run(tmp_path / "workspaces", "first")
        reg.register(
            first_id,
            output_dir=first_dir,
            artefacts={"ReportFile1": first_dir / "ReportFile1.txt"},
        )
        second_id = reg.mint()
        second_dir = _make_run(tmp_path / "workspaces", "second")
        reg.register(second_id, output_dir=second_dir, artefacts={})
        third_id = reg.mint()
        third_dir = _make_run(tmp_path / "workspaces", "third")
        reg.register(third_id, output_dir=third_dir, artefacts={})

        # The first run's output_dir is gone from disk.
        assert not first_dir.exists()

        # And the read tool returns unknown_run_id when the LLM tries to
        # follow up on it.
        with pytest.raises(ToolError) as excinfo:
            await fresh.call_tool(
                "gmat_read_run_artefact",
                {"run_id": first_id, "name": "ReportFile1"},
            )
        assert "invalid_input.unknown_run_id" in str(excinfo.value)

    async def test_round_trip_full_content_matches_disk(
        self, tmp_path: Path, registry: RunRegistry, mcp_server: FastMCP
    ) -> None:
        """Acceptance: read tool returns the same bytes the producer wrote."""
        out = tmp_path / "workspaces" / "run-a"
        out.mkdir(parents=True)
        body = "Sat.UTCGregorian Sat.X\n01 Jan 2026 12:00:00.000 7000.0\n"
        report_path = out / "ReportFile1.txt"
        report_path.write_text(body, encoding="utf-8")
        run_id = registry.mint()
        registry.register(run_id, output_dir=out, artefacts={"ReportFile1": report_path})

        response = await _call(mcp_server, run_id=run_id, name="ReportFile1", output="full")
        # The full-mode content reconstructs the file's text exactly
        # (modulo trailing newline normalisation, which the producer
        # tests already cover in `_shape_raw_report`).
        assert response.content.startswith("Sat.UTCGregorian Sat.X")
        assert "7000.0" in response.content
        assert response.byte_count.value == float(report_path.stat().st_size)
