"""Tests for the ``gmat-skeleton://*`` MCP resource catalogue.

Three layers, mirroring :mod:`tests.test_tool_gmat_validate_script`:

1. :class:`TestNegativeCase` — the bare environment has no GMAT skeletons on
   the real :data:`astrodynamics_mcp.server.mcp` singleton, since the
   ``_GMAT_RUN_AVAILABLE`` guard is ``False`` without the ``[gmat]`` extra.

2. :class:`TestPositiveCase` and :class:`TestDescriptionExtraction` — drive
   the registration helper against a fresh :class:`FastMCP` and exercise
   the description-extraction parser directly. Both run in the bare CI
   environment.

3. :class:`TestIntegrationAgainstRealGmat` — opt-in
   ``@pytest.mark.gmat_installed`` block that round-trips every registered
   skeleton through the registered :func:`gmat_validate_script` tool
   against the real Linux GMAT install. This is the pin that keeps
   skeleton validity tracked as the GMAT version moves.
"""

from __future__ import annotations

import importlib.util
from importlib import resources
from pathlib import Path

import pytest
from mcp.server.fastmcp import FastMCP
from pydantic import AnyUrl

from astrodynamics_mcp.server import mcp as real_mcp
from astrodynamics_mcp.tools import gmat as gmat_tools
from astrodynamics_mcp.tools.gmat import (
    _SKELETON_URI_SCHEME,
    _SKELETONS,
    _extract_description,
)


def _skeleton_path(filename: str) -> Path:
    return Path(str(resources.files("astrodynamics_mcp.skeletons").joinpath(filename)))


def _skeleton_uri(slug: str) -> str:
    return f"{_SKELETON_URI_SCHEME}://{slug}"


@pytest.mark.skipif(
    importlib.util.find_spec("gmat_run") is not None,
    reason="negative case requires a bare environment (no [gmat] extra installed)",
)
class TestNegativeCase:
    """Without ``gmat-run`` installed the real surface carries no skeletons."""

    async def test_module_guard_is_false_in_test_env(self) -> None:
        assert gmat_tools._GMAT_RUN_AVAILABLE is False

    async def test_real_surface_has_no_skeleton_resources(self) -> None:
        resources_listed = await real_mcp.list_resources()
        uris = {str(r.uri) for r in resources_listed}
        leaked = {u for u in uris if u.startswith(f"{_SKELETON_URI_SCHEME}://")}
        assert not leaked, f"skeleton resources leaked into the bare surface: {leaked}"


class TestPositiveCase:
    """When the guard is satisfied, every skeleton lands as a resource."""

    @pytest.fixture(autouse=True)
    def _isolated_mcp(self, monkeypatch: pytest.MonkeyPatch) -> FastMCP:
        fresh = FastMCP("gmat-skeleton-test")
        monkeypatch.setattr("astrodynamics_mcp.server.mcp", fresh)
        return fresh

    async def test_helper_registers_full_catalogue(self, _isolated_mcp: FastMCP) -> None:
        gmat_tools._register_gmat_resources()
        listed = await _isolated_mcp.list_resources()
        listed_uris = {str(r.uri) for r in listed}
        expected_uris = {_skeleton_uri(slug) for slug, _ in _SKELETONS}
        missing = expected_uris - listed_uris
        assert not missing, f"missing skeleton resources: {missing}"

    async def test_descriptions_populated(self, _isolated_mcp: FastMCP) -> None:
        gmat_tools._register_gmat_resources()
        listed = await _isolated_mcp.list_resources()
        for r in listed:
            if not str(r.uri).startswith(f"{_SKELETON_URI_SCHEME}://"):
                continue
            assert r.description, f"{r.uri} registered without a description"

    @pytest.mark.parametrize(("slug", "filename"), _SKELETONS)
    async def test_read_returns_file_verbatim(
        self, _isolated_mcp: FastMCP, slug: str, filename: str
    ) -> None:
        gmat_tools._register_gmat_resources()
        chunks = await _isolated_mcp.read_resource(AnyUrl(_skeleton_uri(slug)))
        # ``read_resource`` returns an iterable of ``ReadResourceContents``;
        # the FunctionResource wrapper joins them into one chunk per call.
        contents = list(chunks)
        assert len(contents) == 1
        assert contents[0].content == _skeleton_path(filename).read_text(encoding="utf-8")


class TestDescriptionExtraction:
    """Unit tests for the description-line scraper."""

    def test_first_match_wins(self) -> None:
        text = (
            "% banner\n% Description: hello world\n% Description: ignored\nCreate Spacecraft Sat\n"
        )
        assert _extract_description(text) == "hello world"

    def test_blank_and_comment_lines_skipped(self) -> None:
        text = (
            "\n\n%\n%   Description:   trimmed surrounding whitespace   \nCreate Spacecraft Sat\n"
        )
        assert _extract_description(text) == "trimmed surrounding whitespace"

    def test_missing_description_raises(self) -> None:
        text = "% only a banner\nCreate Spacecraft Sat\nBeginMissionSequence\n"
        with pytest.raises(ValueError, match="Description"):
            _extract_description(text)

    def test_description_after_non_comment_is_missed(self) -> None:
        # The scraper only inspects leading comments — anything after a
        # script line doesn't count, so a misplaced description still raises.
        text = "Create Spacecraft Sat\n% Description: too late\n"
        with pytest.raises(ValueError, match="Description"):
            _extract_description(text)


# ---------------------------------------------------------------------------
# Integration: real GMAT install (auto-skipped when gmat_run is missing)
# ---------------------------------------------------------------------------


@pytest.mark.gmat_installed
@pytest.mark.skipif(
    importlib.util.find_spec("gmat_run") is None,
    reason="gmat_run is not installed; install the [gmat] extra to run this test",
)
class TestIntegrationAgainstRealGmat:
    """Round-trip every skeleton through ``gmat_validate_script`` and assert ``ok=True``."""

    @pytest.mark.parametrize(("slug", "filename"), _SKELETONS)
    async def test_skeleton_parses_via_gmat_validate_script(self, slug: str, filename: str) -> None:
        from astrodynamics_mcp.server import mcp
        from astrodynamics_mcp.tools.gmat import GmatValidateScriptResponse

        text = _skeleton_path(filename).read_text(encoding="utf-8")
        _content, structured = await mcp.call_tool("gmat_validate_script", {"script": text})
        parsed = GmatValidateScriptResponse.model_validate(structured)
        assert parsed.ok, (
            f"skeleton {slug!r} failed to parse via gmat_validate_script: errors={parsed.errors!r}"
        )
        assert parsed.errors == [], (
            f"skeleton {slug!r} parsed but reported errors: {parsed.errors!r}"
        )

    @pytest.mark.parametrize(("slug", "filename"), _SKELETONS)
    async def test_skeleton_runs_via_gmat_execute_script(self, slug: str, filename: str) -> None:
        """Round-trip every skeleton end-to-end through the live GMAT engine.

        Parse-validation only confirms GMAT's interpreter built the object
        graph; this gate confirms the mission sequence actually executes
        (solvers converge, propagation completes, ReportFile output lands).
        Required-but-missing data files, divergent DC targets, and runtime
        engine failures all surface here as ``ok=False`` rather than the
        ``ok=True`` validate path.
        """
        from astrodynamics_mcp.server import mcp
        from astrodynamics_mcp.tools.gmat import GmatExecuteScriptResponse

        text = _skeleton_path(filename).read_text(encoding="utf-8")
        _content, structured = await mcp.call_tool("gmat_execute_script", {"script": text})
        parsed = GmatExecuteScriptResponse.model_validate(structured)
        assert parsed.ok, (
            f"skeleton {slug!r} failed to run via gmat_execute_script. "
            f"GMAT stderr:\n{parsed.stderr}"
        )
