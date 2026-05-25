"""Tests for `astrodynamics_mcp.data.celestrak`.

Unit tests drive the adapter against ``httpx.MockTransport`` for the five
failure modes specified in the issue (happy path, cache hit, stale
fallback, hard-failure-no-cache, HTTP 4xx). One ``integration``-marked
test hits the live CelesTrak endpoint for NORAD 25544 (ISS).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from astrodynamics_mcp.cache import Cache
from astrodynamics_mcp.data.celestrak import (
    CelestrakResponse,
    OmmRecord,
    fetch_tle,
)
from astrodynamics_mcp.errors import DataSourceError, UpstreamError

_SAMPLE_OMM_ISS = {
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


@pytest.fixture
def cache(tmp_path: Path) -> Cache:
    """Per-test cache rooted at a tmp dir; never touches the real XDG cache."""
    return Cache(directory=tmp_path)


def _mock_client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestCacheMissThenFetch:
    async def test_happy_path_caches_response(self, cache: Cache) -> None:
        call_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            assert "CATNR=25544" in str(request.url)
            assert "FORMAT=json" in str(request.url)
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        client = _mock_client(handler)
        try:
            response = await fetch_tle("25544", client=client, cache=cache)
        finally:
            await client.aclose()

        assert isinstance(response, CelestrakResponse)
        assert response.stale is False
        assert len(response.results) == 1
        assert response.results[0].NORAD_CAT_ID == 25544
        assert response.results[0].OBJECT_NAME == "ISS (ZARYA)"
        assert len(response.tle_lines) == 1
        assert len(response.tle_lines[0].line1) == 69
        assert len(response.tle_lines[0].line2) == 69
        assert response.fetched_at.tzinfo is not None

        # Second call hits the cache; no second HTTP request.
        response2 = await fetch_tle("25544", client=client, cache=cache)
        assert call_count["n"] == 1
        assert response2.stale is False

    async def test_name_query_uses_name_param(self, cache: Cache) -> None:
        seen_params: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen_params.update(request.url.params)
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        client = _mock_client(handler)
        try:
            await fetch_tle("HUBBLE", client=client, cache=cache)
        finally:
            await client.aclose()
        assert seen_params.get("NAME") == "HUBBLE"
        assert "CATNR" not in seen_params

    async def test_norad_id_query_uses_catnr_param(self, cache: Cache) -> None:
        seen_params: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen_params.update(request.url.params)
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        client = _mock_client(handler)
        try:
            await fetch_tle("20580", client=client, cache=cache)  # Hubble
        finally:
            await client.aclose()
        assert seen_params.get("CATNR") == "20580"
        assert "NAME" not in seen_params

    async def test_multi_record_group_response(self, cache: Cache) -> None:
        """A name query that returns several satellites still parses cleanly."""
        second_omm = dict(_SAMPLE_OMM_ISS)
        second_omm["OBJECT_NAME"] = "ISS DEB"
        second_omm["NORAD_CAT_ID"] = 99999

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS, second_omm])

        client = _mock_client(handler)
        try:
            response = await fetch_tle("ISS", client=client, cache=cache)
        finally:
            await client.aclose()
        assert len(response.results) == 2
        assert len(response.tle_lines) == 2


class TestStaleFallback:
    async def test_network_failure_with_cache_returns_stale(self, cache: Cache) -> None:
        # Seed the cache with a successful fetch first.
        def good_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        client = _mock_client(good_handler)
        try:
            first = await fetch_tle("25544", client=client, cache=cache)
        finally:
            await client.aclose()
        assert first.stale is False

        # Now force the cache TTL to "always expired" so the next call hits
        # the network. We do this by passing a fresh handler that raises.
        def fail_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("simulated outage")

        # Backdate the cache entry to 1 day old so the TTL gate kicks the
        # request out to the network, where the failure path engages.
        from datetime import timedelta

        path = cache._path("celestrak", "catnr:25544")
        import json as _json

        payload = _json.loads(path.read_text())
        payload["fetched_at"] = (datetime.now(tz=timezone.utc) - timedelta(days=1)).isoformat()
        path.write_text(_json.dumps(payload))

        client = _mock_client(fail_handler)
        try:
            second = await fetch_tle("25544", client=client, cache=cache)
        finally:
            await client.aclose()

        assert second.stale is True
        assert second.results[0].NORAD_CAT_ID == 25544
        # The fetched_at carried back is the original cache fetched_at, not
        # the failed-fetch attempt time. Operators see real freshness.
        age_days = (datetime.now(tz=timezone.utc) - second.fetched_at).days
        assert age_days >= 1


class TestHardFailureNoCache:
    async def test_network_failure_no_cache_raises_data_source_error(self, cache: Cache) -> None:
        def fail_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("simulated outage")

        client = _mock_client(fail_handler)
        try:
            with pytest.raises(DataSourceError) as excinfo:
                await fetch_tle("25544", client=client, cache=cache)
        finally:
            await client.aclose()
        assert excinfo.value.code == "data_source.celestrak_unreachable"
        assert excinfo.value.source == "celestrak"

    async def test_http_500_no_cache_raises_data_source_error(self, cache: Cache) -> None:
        def server_error_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="Service Unavailable")

        client = _mock_client(server_error_handler)
        try:
            with pytest.raises(DataSourceError) as excinfo:
                await fetch_tle("25544", client=client, cache=cache)
        finally:
            await client.aclose()
        assert excinfo.value.code == "data_source.celestrak_unreachable"

    async def test_http_404_no_cache_raises_data_source_error(self, cache: Cache) -> None:
        def not_found(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="Not Found")

        client = _mock_client(not_found)
        try:
            with pytest.raises(DataSourceError):
                await fetch_tle("99999999", client=client, cache=cache)
        finally:
            await client.aclose()


class TestUpstreamShapeGuard:
    async def test_non_list_payload_raises_upstream_error(self, cache: Cache) -> None:
        # CelesTrak occasionally returns a JSON object with a `message` field
        # instead of a list (e.g. "No GP data found"). We treat that as an
        # upstream-shape mismatch — the cache must not be poisoned.
        def odd_shape(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"message": "no data"})

        client = _mock_client(odd_shape)
        try:
            with pytest.raises(UpstreamError) as excinfo:
                await fetch_tle("25544", client=client, cache=cache)
        finally:
            await client.aclose()
        assert excinfo.value.code == "upstream.celestrak_unexpected_shape"
        # And the cache was not poisoned with the odd payload.
        assert cache.get_stale("celestrak", "catnr:25544") is None

    async def test_invalid_json_response_raises_data_source_error(self, cache: Cache) -> None:
        """A 200 OK with HTML body (e.g. captcha page) decodes as ValueError."""

        def html_response(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>nope</html>")

        client = _mock_client(html_response)
        try:
            with pytest.raises(DataSourceError):
                await fetch_tle("25544", client=client, cache=cache)
        finally:
            await client.aclose()


class TestOmmRecord:
    def test_extra_fields_allowed(self) -> None:
        # CelesTrak adds fields over time; the model accepts them so we
        # don't release on every upstream change.
        data = dict(_SAMPLE_OMM_ISS)
        data["NEW_FIELD"] = "future"
        record = OmmRecord.model_validate(data)
        assert record.OBJECT_NAME == "ISS (ZARYA)"


class TestGroupQueryDispatch:
    """Group/category keywords route to ``gp.php?GROUP=<name>``."""

    async def test_group_keyword_uses_group_param(self, cache: Cache) -> None:
        seen_params: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen_params.update(request.url.params)
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        client = _mock_client(handler)
        try:
            await fetch_tle("weather", client=client, cache=cache)
        finally:
            await client.aclose()
        assert seen_params.get("GROUP") == "weather"
        assert "NAME" not in seen_params
        assert "CATNR" not in seen_params

    async def test_group_keyword_case_insensitive(self, cache: Cache) -> None:
        seen_params: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen_params.update(request.url.params)
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        client = _mock_client(handler)
        try:
            await fetch_tle("Weather", client=client, cache=cache)
        finally:
            await client.aclose()
        # Case is normalised to lowercase before hitting the wire so the
        # cache key stays stable across "weather" / "Weather" / "WEATHER".
        assert seen_params.get("GROUP") == "weather"

    async def test_non_keyword_string_still_falls_through_to_name(self, cache: Cache) -> None:
        """A query that isn't a digit AND isn't a known group keyword is a NAME lookup."""
        seen_params: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen_params.update(request.url.params)
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        client = _mock_client(handler)
        try:
            await fetch_tle("not-a-group-name", client=client, cache=cache)
        finally:
            await client.aclose()
        assert seen_params.get("NAME") == "not-a-group-name"
        assert "GROUP" not in seen_params


class TestThinOmmRecord:
    """OMM records missing the CCSDS metadata fields parse with defaults."""

    @staticmethod
    def _thin_record() -> dict[str, Any]:
        """An OMM payload missing every CCSDS-metadata field with a default."""
        thin = dict(_SAMPLE_OMM_ISS)
        for field in (
            "OBJECT_ID",
            "EPHEMERIS_TYPE",
            "CLASSIFICATION_TYPE",
            "ELEMENT_SET_NO",
            "REV_AT_EPOCH",
        ):
            del thin[field]
        return thin

    def test_record_missing_metadata_parses_with_ccsds_defaults(self) -> None:
        record = OmmRecord.model_validate(self._thin_record())
        assert record.OBJECT_ID == ""
        assert record.EPHEMERIS_TYPE == 0
        assert record.CLASSIFICATION_TYPE == "U"
        assert record.ELEMENT_SET_NO == 0
        assert record.REV_AT_EPOCH == 0
        # The mean elements + identifiers are still required and present.
        assert record.NORAD_CAT_ID == 25544
        assert record.MEAN_MOTION == 15.5

    async def test_thin_record_round_trips_to_valid_tle(self, cache: Cache) -> None:
        """A thin CelesTrak response still derives 69-char line1 / line2."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[self._thin_record()])

        client = _mock_client(handler)
        try:
            response = await fetch_tle("25544", client=client, cache=cache)
        finally:
            await client.aclose()

        assert len(response.results) == 1
        assert len(response.tle_lines) == 1
        assert len(response.tle_lines[0].line1) == 69
        assert len(response.tle_lines[0].line2) == 69


class TestMalformedOmmRecord:
    """A record missing a mean-element field surfaces as UpstreamError, no cache poison."""

    @staticmethod
    def _broken_record() -> dict[str, Any]:
        """An OMM payload missing INCLINATION — can't reconstruct a TLE."""
        broken = dict(_SAMPLE_OMM_ISS)
        del broken["INCLINATION"]
        return broken

    async def test_missing_mean_element_raises_upstream_error(self, cache: Cache) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[self._broken_record()])

        client = _mock_client(handler)
        try:
            with pytest.raises(UpstreamError) as excinfo:
                await fetch_tle("25544", client=client, cache=cache)
        finally:
            await client.aclose()
        assert excinfo.value.code == "upstream.celestrak_invalid_record"
        assert excinfo.value.data["record_index"] == 0
        assert excinfo.value.data["norad_cat_id"] == 25544

    async def test_malformed_record_does_not_poison_cache(self, cache: Cache) -> None:
        """The cache must not retain a payload that contains an unparseable record."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[self._broken_record()])

        client = _mock_client(handler)
        try:
            with pytest.raises(UpstreamError):
                await fetch_tle("25544", client=client, cache=cache)
        finally:
            await client.aclose()
        # Cache must be empty for this key — validation runs before cache.put.
        assert cache.get_stale("celestrak", "catnr:25544") is None

    async def test_second_record_malformed_fails_whole_response(self, cache: Cache) -> None:
        """First good, second bad → whole response fails (no partial success)."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS, self._broken_record()])

        client = _mock_client(handler)
        try:
            with pytest.raises(UpstreamError) as excinfo:
                await fetch_tle("active", client=client, cache=cache)
        finally:
            await client.aclose()
        assert excinfo.value.code == "upstream.celestrak_invalid_record"
        assert excinfo.value.data["record_index"] == 1
        # Cache untouched even though the first record was valid.
        assert cache.get_stale("celestrak", "group:active") is None


class TestDefaultClientAndCachePaths:
    """Exercise the production code paths where no client / cache is injected."""

    async def test_default_client_path(self, cache: Cache) -> None:
        """When `client=None`, fetch_tle owns the AsyncClient via async-with."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        # Patch httpx.AsyncClient so the owned-client path uses our MockTransport.
        with patch(
            "astrodynamics_mcp.data.celestrak.httpx.AsyncClient",
            return_value=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ):
            response = await fetch_tle("25544", cache=cache)

        assert len(response.results) == 1

    async def test_default_cache_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When `cache=None`, fetch_tle pulls the module-level singleton."""
        # Force the singleton to use a tmp dir we control.
        monkeypatch.setenv("ASTRODYNAMICS_MCP_CACHE_DIR", str(tmp_path))
        # Reset any previously-built singleton.
        import astrodynamics_mcp.cache as cache_module

        monkeypatch.setattr(cache_module, "_default_cache", None)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        client = _mock_client(handler)
        try:
            response = await fetch_tle("25544", client=client)
        finally:
            await client.aclose()
        assert len(response.results) == 1


class TestDisabledCache:
    """When the cache is off the adapter still works; fetched_at is set inline."""

    async def test_fetched_at_set_when_cache_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ASTRODYNAMICS_MCP_CACHE_DIR", "")
        disabled_cache = Cache()
        assert not disabled_cache.enabled

        before = datetime.now(tz=timezone.utc)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        client = _mock_client(handler)
        try:
            response = await fetch_tle("25544", client=client, cache=disabled_cache)
        finally:
            await client.aclose()

        assert response.fetched_at >= before
        assert response.stale is False
        # Subsequent calls re-hit the network (no caching).
        try:
            client2 = _mock_client(handler)
            await fetch_tle("25544", client=client2, cache=disabled_cache)
        finally:
            await client2.aclose()


@pytest.mark.integration
class TestLiveCelestrak:
    """Hits the real CelesTrak endpoint; gated behind the integration marker."""

    async def test_live_iss_fetch(self, cache: Cache) -> None:
        try:
            response = await fetch_tle("25544", cache=cache)
        except DataSourceError as exc:
            # GitHub Actions runners occasionally lose outbound connectivity
            # to celestrak.org — the structural assertions below have no
            # signal when the network is the bottleneck. Skip rather than
            # fail a PR on a transient runner issue.
            pytest.skip(f"CelesTrak unreachable from this runner: {exc}")
        assert len(response.results) >= 1
        iss = next(r for r in response.results if r.NORAD_CAT_ID == 25544)
        assert iss.OBJECT_NAME.startswith("ISS")
        assert len(response.tle_lines) >= 1
        assert response.tle_lines[0].line1.startswith("1 ")
        assert response.tle_lines[0].line2.startswith("2 ")
        assert response.stale is False
