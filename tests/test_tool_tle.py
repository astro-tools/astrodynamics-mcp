"""Tests for `astrodynamics_mcp.tools.tle`.

Drive the tool against an ``httpx.MockTransport``-backed adapter so the
tool tests stay end-to-end (function body → adapter → mock CelesTrak)
without hitting the network. Registration and lint coverage are exercised
through the real module-level :data:`astrodynamics_mcp.server.mcp`
singleton.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from mcp.server.fastmcp.exceptions import ToolError

from astrodynamics_mcp.cache import Cache
from astrodynamics_mcp.server import mcp
from astrodynamics_mcp.server_lint import check_tool_descriptions
from astrodynamics_mcp.tools.tle import TleLookupResponse, TleResult, tle_lookup

_SAMPLE_OMM_ISS: dict[str, Any] = {
    "OBJECT_NAME": "ISS (ZARYA)",
    "OBJECT_ID": "1998-067A",
    "EPOCH": "2024-01-01T12:00:00.000000",
    "MEAN_MOTION": 15.5,
    "ECCENTRICITY": 0.0001,
    "INCLINATION": 51.64,
    "RA_OF_ASC_NODE": 90.0,
    "ARG_OF_PERICENTER": 90.0,
    "MEAN_ANOMALY": 270.0,
    "EPHEMERIS_TYPE": 0,
    "CLASSIFICATION_TYPE": "U",
    "NORAD_CAT_ID": 25544,
    "ELEMENT_SET_NO": 999,
    "REV_AT_EPOCH": 0,
    "BSTAR": 0.00018,
    "MEAN_MOTION_DOT": 0.0001,
    "MEAN_MOTION_DDOT": 0.0,
}

_GOLDEN_PATH = Path(__file__).parent / "data" / "golden" / "tle_lookup_iss.json"


@pytest.fixture
def tmp_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Cache:
    """Per-test cache rooted at a tmp dir, wired in as the module singleton.

    The tool calls ``fetch_tle`` with no cache argument, so the adapter
    falls back to ``default_cache()``. We force the singleton to a tmp
    directory and reset it so each test gets a clean slate.
    """
    monkeypatch.setenv("ASTRODYNAMICS_MCP_CACHE_DIR", str(tmp_path))
    import astrodynamics_mcp.cache as cache_module

    monkeypatch.setattr(cache_module, "_default_cache", None)
    return Cache(directory=tmp_path)


def _patched_async_client(handler: Any) -> Any:
    """Patch the celestrak adapter's ``httpx.AsyncClient`` with a MockTransport-backed one."""
    return patch(
        "astrodynamics_mcp.data.celestrak.httpx.AsyncClient",
        return_value=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _patched_spacetrack_client(handler: Any) -> Any:
    """Patch the spacetrack adapter's singleton builder to use a MockTransport client.

    The spacetrack adapter uses a module-level singleton client (so the
    session cookie persists across calls in production). Tests bypass that
    by replacing the builder so each test gets its own isolated client.
    ``_BASE_URL`` must be set on the test client so the adapter's relative
    URLs resolve correctly.
    """
    from astrodynamics_mcp.data.spacetrack import _BASE_URL

    return patch(
        "astrodynamics_mcp.data.spacetrack._build_singleton_client",
        return_value=httpx.AsyncClient(
            base_url=_BASE_URL,
            transport=httpx.MockTransport(handler),
        ),
    )


@pytest.fixture
def reset_spacetrack_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the spacetrack singleton client between tests that exercise it."""
    import astrodynamics_mcp.data.spacetrack as spacetrack_module

    monkeypatch.setattr(spacetrack_module, "_singleton_client", None)


class TestHappyPaths:
    async def test_norad_id_lookup_returns_one_result(self, tmp_cache: Cache) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        with _patched_async_client(handler):
            response = await tle_lookup(query="25544")

        assert isinstance(response, TleLookupResponse)
        assert len(response.results) == 1
        result = response.results[0]
        assert isinstance(result, TleResult)
        assert result.name == "ISS (ZARYA)"
        assert result.norad_id == "25544"
        assert len(result.tle_line1) == 69
        assert len(result.tle_line2) == 69
        assert result.tle_line1.startswith("1 25544")
        assert result.tle_line2.startswith("2 25544")
        assert result.omm["NORAD_CAT_ID"] == 25544
        assert result.stale is False

    async def test_name_lookup_uses_name_param(self, tmp_cache: Cache) -> None:
        seen_params: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen_params.update(request.url.params)
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        with _patched_async_client(handler):
            response = await tle_lookup(query="HUBBLE")

        assert seen_params.get("NAME") == "HUBBLE"
        assert response.results[0].name == "ISS (ZARYA)"

    async def test_group_lookup_returns_multiple_results(self, tmp_cache: Cache) -> None:
        second = dict(_SAMPLE_OMM_ISS)
        second["OBJECT_NAME"] = "NOAA 19"
        second["NORAD_CAT_ID"] = 33591

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS, second])

        with _patched_async_client(handler):
            response = await tle_lookup(query="weather")

        assert len(response.results) == 2
        names = {r.name for r in response.results}
        assert names == {"ISS (ZARYA)", "NOAA 19"}
        norad_ids = {r.norad_id for r in response.results}
        assert norad_ids == {"25544", "33591"}

    async def test_fetched_at_is_iso_8601_with_time_component(self, tmp_cache: Cache) -> None:
        """The Epoch type alias validates the string format on TleResult.fetched_at."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        with _patched_async_client(handler):
            response = await tle_lookup(query="25544")

        # Parse it back — should be a valid datetime with timezone.
        parsed = datetime.fromisoformat(response.results[0].fetched_at)
        assert parsed.tzinfo is not None


class TestCacheHit:
    async def test_second_call_within_ttl_serves_from_cache(self, tmp_cache: Cache) -> None:
        """Two `tle_lookup` calls with the same query → exactly one HTTP request.

        Proves the tool's `fetch_tle` call (which passes no `cache` arg) does
        fall through to the module-level singleton and that the singleton's
        6-hour CelesTrak TTL keeps the second call off the wire.
        """
        call_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        with _patched_async_client(handler):
            first = await tle_lookup(query="25544")
            second = await tle_lookup(query="25544")

        assert call_count["n"] == 1
        assert first.results[0].stale is False
        assert second.results[0].stale is False
        # The cache files live under the tmp dir the fixture wired in.
        assert tmp_cache.directory is not None
        assert (tmp_cache.directory / "celestrak").is_dir()


class TestStaleFallback:
    async def test_stale_flag_propagates_per_result(self, tmp_cache: Cache) -> None:
        """When the adapter returns stale=True, every TleResult carries stale=True."""

        # Seed the cache via a successful first fetch.
        def good_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        with _patched_async_client(good_handler):
            await tle_lookup(query="25544")

        # Backdate the cache entry past TTL, then re-issue against an outage.
        from datetime import timedelta

        cache_path = tmp_cache._path("celestrak", "catnr:25544")
        payload = json.loads(cache_path.read_text())
        payload["fetched_at"] = (datetime.now(tz=timezone.utc) - timedelta(days=1)).isoformat()
        cache_path.write_text(json.dumps(payload))

        def fail_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("simulated outage")

        with _patched_async_client(fail_handler):
            response = await tle_lookup(query="25544")

        assert len(response.results) == 1
        assert response.results[0].stale is True


class TestErrorPropagation:
    async def test_data_source_error_surfaces_as_typed_envelope(self, tmp_cache: Cache) -> None:
        """When CelesTrak is unreachable with no cache, the typed error envelope hits the wire.

        The ``register_tool`` wrapper translates every ``AstrodynamicsMCPError``
        into a ``ToolError`` whose message is the JSON envelope — the LLM
        consumer parses the stable string code out of that body.
        """

        def fail_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("simulated outage")

        with _patched_async_client(fail_handler), pytest.raises(ToolError) as excinfo:
            await tle_lookup(query="25544")

        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "data_source.celestrak_unreachable"
        assert envelope["data"]["source"] == "celestrak"


class TestRegistration:
    """The tool registers against the module-level FastMCP singleton on import."""

    async def test_tool_is_listed(self) -> None:
        tools = await mcp.list_tools()
        names = {t.name for t in tools}
        assert "tle_lookup" in names

    async def test_tool_description_passes_lint(self) -> None:
        tools = await mcp.list_tools()
        violations = [v for v in check_tool_descriptions(tools) if v.tool_name == "tle_lookup"]
        assert violations == []

    async def test_tool_callable_via_mcp(self, tmp_cache: Cache) -> None:
        """End-to-end: invoking through ``mcp.call_tool`` returns the structured response."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        with _patched_async_client(handler):
            content, structured = await mcp.call_tool("tle_lookup", {"query": "25544"})
        del content  # FastMCP also emits a content list; only the structured dict matters here.
        assert isinstance(structured, dict)
        assert "results" in structured
        assert len(structured["results"]) == 1
        assert structured["results"][0]["norad_id"] == "25544"


class TestGoldenSnapshot:
    """The ISS-lookup response shape is locked against a committed JSON snapshot.

    The snapshot was generated from the deterministic ``_SAMPLE_OMM_ISS`` input
    through ``sgp4.omm.initialize`` + ``sgp4.exporter.export_tle``. Bumping
    the sgp4 minor version may produce a different line1/line2 — regenerate
    the snapshot deliberately and review the diff per the issue's contract.
    """

    async def test_iss_response_matches_golden(self, tmp_cache: Cache) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        with _patched_async_client(handler):
            response = await tle_lookup(query="25544")

        live = response.model_dump()
        # ``fetched_at`` varies per run; mask before comparing.
        for r in live["results"]:
            r["fetched_at"] = "<masked-in-test>"

        golden = json.loads(_GOLDEN_PATH.read_text())
        assert live == golden


class TestSchemaInvariants:
    async def test_response_round_trips_through_json(self, tmp_cache: Cache) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        with _patched_async_client(handler):
            response = await tle_lookup(query="25544")

        as_json = response.model_dump_json()
        rebuilt = TleLookupResponse.model_validate_json(as_json)
        assert rebuilt == response


_SPACETRACK_LOGIN_COOKIE = "chocolatechip=session-token; Path=/"


def _spacetrack_handler(payload: Any) -> Any:
    """Build a handler that satisfies Space-Track login then returns *payload*."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/ajaxauth/login":
            return httpx.Response(200, headers={"set-cookie": _SPACETRACK_LOGIN_COOKIE})
        return httpx.Response(200, json=payload)

    return handler


class TestSpaceTrackDispatch:
    """``source='space-track'`` goes through the spacetrack adapter and credential gate."""

    async def test_creds_in_env_returns_response(
        self,
        tmp_cache: Cache,
        monkeypatch: pytest.MonkeyPatch,
        reset_spacetrack_singleton: None,
    ) -> None:
        monkeypatch.setenv("ASTRODYNAMICS_MCP_SPACETRACK_USERNAME", "alice@example.org")
        monkeypatch.setenv("ASTRODYNAMICS_MCP_SPACETRACK_PASSWORD", "hunter2")

        with _patched_spacetrack_client(_spacetrack_handler([_SAMPLE_OMM_ISS])):
            response = await tle_lookup(query="25544", source="space-track")

        assert isinstance(response, TleLookupResponse)
        assert len(response.results) == 1
        result = response.results[0]
        assert result.norad_id == "25544"
        assert result.name == "ISS (ZARYA)"
        assert result.stale is False

    async def test_missing_creds_raises_credential_required_envelope(
        self,
        tmp_cache: Cache,
        monkeypatch: pytest.MonkeyPatch,
        reset_spacetrack_singleton: None,
    ) -> None:
        """No creds → typed envelope, and the HTTP layer is never touched."""
        monkeypatch.delenv("ASTRODYNAMICS_MCP_SPACETRACK_USERNAME", raising=False)
        monkeypatch.delenv("ASTRODYNAMICS_MCP_SPACETRACK_PASSWORD", raising=False)

        handler_called = {"hit": False}

        def handler(request: httpx.Request) -> httpx.Response:
            handler_called["hit"] = True
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        with _patched_spacetrack_client(handler), pytest.raises(ToolError) as excinfo:
            await tle_lookup(query="25544", source="space-track")

        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "credential_required.spacetrack"
        assert envelope["data"]["source"] == "spacetrack"
        assert set(envelope["data"]["missing_fields"]) == {"username", "password"}
        # The credential gate fires before any network/login attempt.
        assert handler_called["hit"] is False

    async def test_partial_creds_treated_as_missing(
        self,
        tmp_cache: Cache,
        monkeypatch: pytest.MonkeyPatch,
        reset_spacetrack_singleton: None,
    ) -> None:
        """Only one of username/password set → still raises, names the missing field."""
        monkeypatch.setenv("ASTRODYNAMICS_MCP_SPACETRACK_USERNAME", "alice@example.org")
        monkeypatch.delenv("ASTRODYNAMICS_MCP_SPACETRACK_PASSWORD", raising=False)

        with (
            _patched_spacetrack_client(_spacetrack_handler([_SAMPLE_OMM_ISS])),
            pytest.raises(ToolError) as excinfo,
        ):
            await tle_lookup(query="25544", source="space-track")

        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "credential_required.spacetrack"
        assert envelope["data"]["missing_fields"] == ["password"]

    async def test_default_source_is_celestrak(self, tmp_cache: Cache) -> None:
        """Sanity: omitting source still goes through the CelesTrak path, unchanged."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        with _patched_async_client(handler):
            response = await tle_lookup(query="25544")  # no source arg
        assert response.results[0].norad_id == "25544"
