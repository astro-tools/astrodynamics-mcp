"""Tests for `astrodynamics_mcp.data.spacetrack`.

Drive the adapter against ``httpx.MockTransport`` for the login flow plus
the failure modes that mirror the CelesTrak adapter (happy path, cache
hit, stale fallback, hard failure with no cache, malformed record). The
Space-Track-specific behaviours — session-cookie reuse, 401-triggered
re-login, refused credentials surfacing as `data_source.spacetrack_auth_failed`
— get dedicated classes.

No live-integration class. Real Space-Track exercise happens locally with
maintainer credentials, not in CI (no test-creds path exists upstream
and per-account rate limits make CI a poor fit).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from astrodynamics_mcp.cache import Cache
from astrodynamics_mcp.data.spacetrack import (
    _BASE_URL,
    SpacetrackResponse,
    fetch_tle,
)
from astrodynamics_mcp.errors import DataSourceError, UpstreamError

_CREDENTIAL = {"username": "alice@example.org", "password": "hunter2"}

_LOGIN_COOKIE_HEADER = "chocolatechip=session-token-xyz; Path=/"

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


@pytest.fixture
def cache(tmp_path: Path) -> Cache:
    """Per-test cache rooted at a tmp dir; never touches the real XDG cache."""
    return Cache(directory=tmp_path)


@pytest.fixture(autouse=True)
def reset_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the module-level singleton + login lock state between tests.

    Tests that inject their own client never touch the singleton, but the
    singleton-path test does — clear it so a leaked client from one test
    can never pollute another.
    """
    import astrodynamics_mcp.data.spacetrack as spacetrack_module

    monkeypatch.setattr(spacetrack_module, "_singleton_client", None)


def _mock_client(handler: Any) -> httpx.AsyncClient:
    """Build a MockTransport-backed client with the production base_url.

    ``base_url`` is required so the adapter's relative-URL calls
    (``/ajaxauth/login``, ``/basicspacedata/...``) resolve to absolute
    URLs that bind cookies to the right domain.
    """
    return httpx.AsyncClient(
        base_url=_BASE_URL,
        transport=httpx.MockTransport(handler),
    )


def _login_then_query_handler(query_payload: Any) -> Any:
    """Build a handler that satisfies the login POST then returns *query_payload*."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/ajaxauth/login":
            return httpx.Response(200, headers={"set-cookie": _LOGIN_COOKIE_HEADER})
        return httpx.Response(200, json=query_payload)

    return handler


class TestLogin:
    async def test_login_posts_identity_and_password(self, cache: Cache) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/ajaxauth/login":
                seen["body"] = request.content.decode()
                seen["content_type"] = request.headers.get("content-type", "")
                return httpx.Response(200, headers={"set-cookie": _LOGIN_COOKIE_HEADER})
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        client = _mock_client(handler)
        try:
            await fetch_tle("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()

        # form-encoded with identity / password
        assert "identity=alice%40example.org" in seen["body"]
        assert "password=hunter2" in seen["body"]
        assert "application/x-www-form-urlencoded" in seen["content_type"]

    async def test_login_without_cookie_raises_auth_failed(self, cache: Cache) -> None:
        """Space-Track returns 200 with no Set-Cookie on a bad credential."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/ajaxauth/login":
                # 200 OK but no session cookie — Space-Track's "bad credential" shape.
                return httpx.Response(200, json={"Login": "Failed"})
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        client = _mock_client(handler)
        try:
            with pytest.raises(DataSourceError) as excinfo:
                await fetch_tle("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert excinfo.value.code == "data_source.spacetrack_auth_failed"
        assert excinfo.value.source == "spacetrack"

    async def test_login_network_error_raises_unreachable(self, cache: Cache) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("simulated outage")

        client = _mock_client(handler)
        try:
            with pytest.raises(DataSourceError) as excinfo:
                await fetch_tle("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert excinfo.value.code == "data_source.spacetrack_unreachable"


class TestQueryDispatch:
    async def test_norad_id_query_uses_norad_cat_id_predicate(self, cache: Cache) -> None:
        seen_paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/ajaxauth/login":
                return httpx.Response(200, headers={"set-cookie": _LOGIN_COOKIE_HEADER})
            seen_paths.append(request.url.path)
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        client = _mock_client(handler)
        try:
            await fetch_tle("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert any("NORAD_CAT_ID/25544" in p for p in seen_paths)

    async def test_name_query_uses_object_name_substring(self, cache: Cache) -> None:
        seen_paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/ajaxauth/login":
                return httpx.Response(200, headers={"set-cookie": _LOGIN_COOKIE_HEADER})
            seen_paths.append(request.url.path)
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        client = _mock_client(handler)
        try:
            await fetch_tle("HUBBLE", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        # The ~~text~~ operator is Space-Track's substring filter.
        assert any("OBJECT_NAME/~~HUBBLE~~" in p for p in seen_paths)

    async def test_group_keyword_falls_through_to_name_search(self, cache: Cache) -> None:
        """CelesTrak's group keywords have no Space-Track equivalent — treat as a name."""
        seen_paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/ajaxauth/login":
                return httpx.Response(200, headers={"set-cookie": _LOGIN_COOKIE_HEADER})
            seen_paths.append(request.url.path)
            return httpx.Response(200, json=[])

        client = _mock_client(handler)
        try:
            response = await fetch_tle("stations", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        # No special-case for "stations" — it goes through as a name substring.
        assert any("OBJECT_NAME/~~stations~~" in p for p in seen_paths)
        assert response.results == []


class TestHappyPath:
    async def test_returns_typed_response(self, cache: Cache) -> None:
        client = _mock_client(_login_then_query_handler([_SAMPLE_OMM_ISS]))
        try:
            response = await fetch_tle("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()

        assert isinstance(response, SpacetrackResponse)
        assert response.stale is False
        assert len(response.results) == 1
        assert response.results[0].NORAD_CAT_ID == 25544
        assert response.results[0].OBJECT_NAME == "ISS (ZARYA)"
        assert len(response.tle_lines) == 1
        assert len(response.tle_lines[0].line1) == 69
        assert len(response.tle_lines[0].line2) == 69
        assert response.fetched_at.tzinfo is not None

    async def test_caches_response_skipping_second_network_call(self, cache: Cache) -> None:
        login_count = {"n": 0}
        query_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/ajaxauth/login":
                login_count["n"] += 1
                return httpx.Response(200, headers={"set-cookie": _LOGIN_COOKIE_HEADER})
            query_count["n"] += 1
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        client = _mock_client(handler)
        try:
            await fetch_tle("25544", _CREDENTIAL, client=client, cache=cache)
            # Second call should hit cache — no login, no query.
            response2 = await fetch_tle("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert login_count["n"] == 1  # first call only
        assert query_count["n"] == 1  # first call only
        assert response2.stale is False


class TestSessionReuse:
    async def test_session_cookie_reused_across_calls(self, cache: Cache) -> None:
        """Two distinct queries share one login."""
        login_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/ajaxauth/login":
                login_count["n"] += 1
                return httpx.Response(200, headers={"set-cookie": _LOGIN_COOKIE_HEADER})
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        client = _mock_client(handler)
        try:
            await fetch_tle("25544", _CREDENTIAL, client=client, cache=cache)
            # Distinct query — cache miss, but cookie should still be valid.
            await fetch_tle("20580", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert login_count["n"] == 1


class TestSessionExpiry:
    async def test_401_triggers_relogin_and_retry(self, cache: Cache) -> None:
        login_count = {"n": 0}
        query_calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/ajaxauth/login":
                login_count["n"] += 1
                return httpx.Response(200, headers={"set-cookie": _LOGIN_COOKIE_HEADER})
            query_calls["n"] += 1
            # First query → 401 (expired session). Retry → 200.
            if query_calls["n"] == 1:
                return httpx.Response(401, text="session expired")
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        client = _mock_client(handler)
        try:
            response = await fetch_tle("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert login_count["n"] == 2  # initial + post-401 re-login
        assert query_calls["n"] == 2  # original + retry
        assert response.results[0].NORAD_CAT_ID == 25544

    async def test_persistent_401_surfaces_as_data_source_error(self, cache: Cache) -> None:
        """If 401s keep coming after re-login, give up rather than loop."""
        login_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/ajaxauth/login":
                login_count["n"] += 1
                return httpx.Response(200, headers={"set-cookie": _LOGIN_COOKIE_HEADER})
            return httpx.Response(401, text="forever expired")

        client = _mock_client(handler)
        try:
            with pytest.raises(DataSourceError) as excinfo:
                await fetch_tle("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert excinfo.value.code == "data_source.spacetrack_unreachable"
        # Exactly one retry — initial login + one re-login, no infinite loop.
        assert login_count["n"] == 2


class TestStaleFallback:
    async def test_network_failure_with_cache_returns_stale(self, cache: Cache) -> None:
        # Seed the cache via a successful fetch.
        client = _mock_client(_login_then_query_handler([_SAMPLE_OMM_ISS]))
        try:
            first = await fetch_tle("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert first.stale is False

        # Backdate the cache so the next call goes to the network.
        path = cache._path("spacetrack", "catnr:25544")
        import json as _json

        payload = _json.loads(path.read_text())
        payload["fetched_at"] = (datetime.now(tz=timezone.utc) - timedelta(days=1)).isoformat()
        path.write_text(_json.dumps(payload))

        def fail_handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/ajaxauth/login":
                return httpx.Response(200, headers={"set-cookie": _LOGIN_COOKIE_HEADER})
            raise httpx.ConnectError("simulated outage")

        # Fresh client → cookie jar empty → login happens, query fails, stale served.
        client = _mock_client(fail_handler)
        try:
            second = await fetch_tle("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert second.stale is True
        assert second.results[0].NORAD_CAT_ID == 25544
        age_days = (datetime.now(tz=timezone.utc) - second.fetched_at).days
        assert age_days >= 1


class TestHardFailureNoCache:
    async def test_network_failure_no_cache_raises(self, cache: Cache) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/ajaxauth/login":
                return httpx.Response(200, headers={"set-cookie": _LOGIN_COOKIE_HEADER})
            raise httpx.ConnectError("simulated outage")

        client = _mock_client(handler)
        try:
            with pytest.raises(DataSourceError) as excinfo:
                await fetch_tle("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert excinfo.value.code == "data_source.spacetrack_unreachable"
        assert excinfo.value.source == "spacetrack"

    async def test_http_500_no_cache_raises(self, cache: Cache) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/ajaxauth/login":
                return httpx.Response(200, headers={"set-cookie": _LOGIN_COOKIE_HEADER})
            return httpx.Response(503, text="Service Unavailable")

        client = _mock_client(handler)
        try:
            with pytest.raises(DataSourceError) as excinfo:
                await fetch_tle("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert excinfo.value.code == "data_source.spacetrack_unreachable"


class TestUpstreamShapeGuard:
    async def test_non_list_payload_raises_upstream_error(self, cache: Cache) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/ajaxauth/login":
                return httpx.Response(200, headers={"set-cookie": _LOGIN_COOKIE_HEADER})
            return httpx.Response(200, json={"error": "object not list"})

        client = _mock_client(handler)
        try:
            with pytest.raises(UpstreamError) as excinfo:
                await fetch_tle("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert excinfo.value.code == "upstream.spacetrack_unexpected_shape"
        assert cache.get_stale("spacetrack", "catnr:25544") is None

    async def test_invalid_json_response_raises_data_source_error(self, cache: Cache) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/ajaxauth/login":
                return httpx.Response(200, headers={"set-cookie": _LOGIN_COOKIE_HEADER})
            return httpx.Response(200, text="<html>maintenance</html>")

        client = _mock_client(handler)
        try:
            with pytest.raises(DataSourceError):
                await fetch_tle("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()


class TestMalformedRecord:
    @staticmethod
    def _broken_record() -> dict[str, Any]:
        broken = dict(_SAMPLE_OMM_ISS)
        del broken["INCLINATION"]
        return broken

    async def test_missing_mean_element_raises_upstream_error(self, cache: Cache) -> None:
        client = _mock_client(_login_then_query_handler([self._broken_record()]))
        try:
            with pytest.raises(UpstreamError) as excinfo:
                await fetch_tle("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert excinfo.value.code == "upstream.spacetrack_invalid_record"
        assert excinfo.value.data["record_index"] == 0
        assert excinfo.value.data["norad_cat_id"] == 25544

    async def test_malformed_record_does_not_poison_cache(self, cache: Cache) -> None:
        client = _mock_client(_login_then_query_handler([self._broken_record()]))
        try:
            with pytest.raises(UpstreamError):
                await fetch_tle("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert cache.get_stale("spacetrack", "catnr:25544") is None


class TestSingletonClientPath:
    """The production path (``client=None``) lazy-builds a module-level client."""

    async def test_singleton_is_built_and_reused(self, cache: Cache) -> None:
        login_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/ajaxauth/login":
                login_count["n"] += 1
                return httpx.Response(200, headers={"set-cookie": _LOGIN_COOKIE_HEADER})
            return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

        # Patch the singleton-builder so we get a MockTransport-backed client
        # instead of one that would actually hit space-track.org.
        mock_built = _mock_client(handler)
        with patch(
            "astrodynamics_mcp.data.spacetrack._build_singleton_client",
            return_value=mock_built,
        ):
            try:
                first = await fetch_tle("25544", _CREDENTIAL, cache=cache)
                # Different query (cache miss) — cookie should still be valid
                # because we're reusing the same singleton.
                second = await fetch_tle("20580", _CREDENTIAL, cache=cache)
            finally:
                await mock_built.aclose()

        assert first.results[0].NORAD_CAT_ID == 25544
        assert second.results[0].NORAD_CAT_ID == 25544  # mock returns same payload
        assert login_count["n"] == 1  # singleton's cookie jar survives the second call
