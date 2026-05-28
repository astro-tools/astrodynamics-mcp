"""Tests for `astrodynamics_mcp.data.horizons`."""

from __future__ import annotations

import json as _json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from astrodynamics_mcp.cache import Cache
from astrodynamics_mcp.data.horizons import (
    HorizonsResponse,
    _cache_key,
    _signature,
    fetch_ephemeris,
)
from astrodynamics_mcp.errors import DataSourceError, UpstreamError

_SAMPLE_HORIZONS_PAYLOAD = {
    "signature": {"source": "NAIF/JPL/SSD"},
    "result": "*****\nMars vectors at the requested epochs\n...\n",
}


@pytest.fixture
def cache(tmp_path: Path) -> Cache:
    return Cache(directory=tmp_path)


def _mock_client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestSignatureAndKey:
    def test_signature_is_canonical(self) -> None:
        sig = _signature("499", "@sun", "2026-01-01", "2026-12-31", "1d")
        assert sig == {
            "target": "499",
            "center": "@sun",
            "start": "2026-01-01",
            "stop": "2026-12-31",
            "step": "1d",
        }

    def test_cache_key_is_stable_under_key_order(self) -> None:
        sig1 = {"a": "1", "b": "2", "c": "3"}
        sig2 = {"c": "3", "a": "1", "b": "2"}
        assert _cache_key(sig1) == _cache_key(sig2)

    def test_cache_key_differs_per_request(self) -> None:
        sig1 = _signature("499", "@sun", "2026-01-01", "2026-12-31", "1d")
        sig2 = _signature("499", "@sun", "2026-01-01", "2026-12-31", "6h")
        assert _cache_key(sig1) != _cache_key(sig2)


class TestCacheMissThenFetch:
    async def test_happy_path_caches_response(self, cache: Cache) -> None:
        call_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            params = request.url.params
            assert params["COMMAND"] == "'499'"
            assert params["CENTER"] == "'@sun'"
            assert params["START_TIME"] == "'2026-01-01'"
            assert params["EPHEM_TYPE"] == "VECTORS"
            return httpx.Response(200, json=_SAMPLE_HORIZONS_PAYLOAD)

        client = _mock_client(handler)
        try:
            response = await fetch_ephemeris(
                "499",
                "@sun",
                "2026-01-01",
                "2026-12-31",
                "1d",
                client=client,
                cache=cache,
            )
        finally:
            await client.aclose()

        assert isinstance(response, HorizonsResponse)
        assert response.stale is False
        assert response.result == _SAMPLE_HORIZONS_PAYLOAD["result"]
        assert response.signature["target"] == "499"
        assert response.signature["center"] == "@sun"
        assert response.fetched_at.tzinfo is not None

        # Cache hit: no second HTTP request.
        response2 = await fetch_ephemeris(
            "499",
            "@sun",
            "2026-01-01",
            "2026-12-31",
            "1d",
            client=client,
            cache=cache,
        )
        assert call_count["n"] == 1
        assert response2.stale is False
        assert response2.signature == response.signature


class TestStaleFallback:
    async def test_network_failure_with_cache_returns_stale(self, cache: Cache) -> None:
        def good_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_SAMPLE_HORIZONS_PAYLOAD)

        client = _mock_client(good_handler)
        try:
            await fetch_ephemeris(
                "499",
                "@sun",
                "2026-01-01",
                "2026-12-31",
                "1d",
                client=client,
                cache=cache,
            )
        finally:
            await client.aclose()

        # Backdate the cache so the next call escapes the TTL into the
        # network path, where the failure stale-fallback engages.
        key = _cache_key(_signature("499", "@sun", "2026-01-01", "2026-12-31", "1d"))
        path = cache._path("horizons", key)
        payload = _json.loads(path.read_text())
        payload["fetched_at"] = (datetime.now(tz=timezone.utc) - timedelta(days=10)).isoformat()
        path.write_text(_json.dumps(payload))

        def fail_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("simulated outage")

        client = _mock_client(fail_handler)
        try:
            stale = await fetch_ephemeris(
                "499",
                "@sun",
                "2026-01-01",
                "2026-12-31",
                "1d",
                client=client,
                cache=cache,
            )
        finally:
            await client.aclose()

        assert stale.stale is True
        assert stale.result == _SAMPLE_HORIZONS_PAYLOAD["result"]


class TestHardFailureNoCache:
    async def test_503_no_cache_raises_data_source_error(self, cache: Cache) -> None:
        def server_error(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="Service Unavailable")

        client = _mock_client(server_error)
        try:
            with pytest.raises(DataSourceError) as excinfo:
                await fetch_ephemeris(
                    "499",
                    "@sun",
                    "2026-01-01",
                    "2026-12-31",
                    "1d",
                    client=client,
                    cache=cache,
                )
        finally:
            await client.aclose()
        assert excinfo.value.code == "data_source.horizons_unreachable"
        assert excinfo.value.source == "horizons"

    async def test_connect_error_no_cache_raises_data_source_error(self, cache: Cache) -> None:
        def fail_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("simulated outage")

        client = _mock_client(fail_handler)
        try:
            with pytest.raises(DataSourceError) as excinfo:
                await fetch_ephemeris(
                    "499",
                    "@sun",
                    "2026-01-01",
                    "2026-12-31",
                    "1d",
                    client=client,
                    cache=cache,
                )
        finally:
            await client.aclose()
        assert excinfo.value.code == "data_source.horizons_unreachable"


class TestUpstreamShapeGuard:
    async def test_missing_result_key_raises_upstream_error(self, cache: Cache) -> None:
        def bad_shape(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"signature": {"foo": "bar"}})

        client = _mock_client(bad_shape)
        try:
            with pytest.raises(UpstreamError) as excinfo:
                await fetch_ephemeris(
                    "499",
                    "@sun",
                    "2026-01-01",
                    "2026-12-31",
                    "1d",
                    client=client,
                    cache=cache,
                )
        finally:
            await client.aclose()
        assert excinfo.value.code == "upstream.horizons_unexpected_shape"

    async def test_array_payload_raises_upstream_error(self, cache: Cache) -> None:
        def array_shape(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[{"result": "no"}])

        client = _mock_client(array_shape)
        try:
            with pytest.raises(UpstreamError) as excinfo:
                await fetch_ephemeris(
                    "499",
                    "@sun",
                    "2026-01-01",
                    "2026-12-31",
                    "1d",
                    client=client,
                    cache=cache,
                )
        finally:
            await client.aclose()
        assert excinfo.value.code == "upstream.horizons_unexpected_shape"


class TestErrorBlobGuard:
    """Horizons returns HTTP 200 + an in-band ``error`` for bad inputs; that
    blob must surface as a typed error and never be cached."""

    async def test_error_blob_raises_typed_error(self, cache: Cache) -> None:
        def error_blob(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"error": "No matches found for COMMAND='nonsense'."},
            )

        client = _mock_client(error_blob)
        try:
            with pytest.raises(UpstreamError) as excinfo:
                await fetch_ephemeris(
                    "nonsense",
                    "@sun",
                    "2026-01-01",
                    "2026-12-31",
                    "1d",
                    client=client,
                    cache=cache,
                )
        finally:
            await client.aclose()
        assert excinfo.value.code == "upstream.horizons_error"
        assert "No matches found" in excinfo.value.data["horizons_error"]

    async def test_error_blob_is_not_cached(self, cache: Cache) -> None:
        key = _cache_key(_signature("nonsense", "@sun", "2026-01-01", "2026-12-31", "1d"))

        def error_blob(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"error": "bad CENTER"})

        client = _mock_client(error_blob)
        try:
            with pytest.raises(UpstreamError):
                await fetch_ephemeris(
                    "nonsense",
                    "@sun",
                    "2026-01-01",
                    "2026-12-31",
                    "1d",
                    client=client,
                    cache=cache,
                )
        finally:
            await client.aclose()
        # The error blob must not poison the cache under the valid signature key.
        assert cache.get_stale("horizons", key) is None


class TestStaleFallbackUnusable:
    """When the stale cached payload no longer rebuilds (e.g. an older entry
    with no 'result' key), fall through to the typed unreachable error."""

    async def test_unparseable_stale_payload_raises_unreachable(self, cache: Cache) -> None:
        key = _cache_key(_signature("499", "@sun", "2026-01-01", "2026-12-31", "1d"))
        # Seed the cache directly with a payload that lacks 'result', so the
        # stale rebuild raises a KeyError inside _build_response.
        path = cache._path("horizons", key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            _json.dumps(
                {
                    "key": key,
                    "fetched_at": (datetime.now(tz=timezone.utc) - timedelta(days=30)).isoformat(),
                    "value": {"signature": {"foo": "bar"}},
                }
            )
        )

        def outage(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("simulated outage")

        client = _mock_client(outage)
        try:
            with pytest.raises(DataSourceError) as excinfo:
                await fetch_ephemeris(
                    "499",
                    "@sun",
                    "2026-01-01",
                    "2026-12-31",
                    "1d",
                    client=client,
                    cache=cache,
                )
        finally:
            await client.aclose()
        assert excinfo.value.code == "data_source.horizons_unreachable"


class TestDefaultClientAndCachePaths:
    async def test_default_client_path(self, cache: Cache) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_SAMPLE_HORIZONS_PAYLOAD)

        with patch(
            "astrodynamics_mcp.data.horizons.httpx.AsyncClient",
            return_value=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ):
            response = await fetch_ephemeris(
                "499",
                "@sun",
                "2026-01-01",
                "2026-12-31",
                "1d",
                cache=cache,
            )
        assert response.signature["target"] == "499"

    async def test_default_cache_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ASTRODYNAMICS_MCP_CACHE_DIR", str(tmp_path))
        import astrodynamics_mcp.cache as cache_module

        monkeypatch.setattr(cache_module, "_default_cache", None)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_SAMPLE_HORIZONS_PAYLOAD)

        client = _mock_client(handler)
        try:
            response = await fetch_ephemeris(
                "499",
                "@sun",
                "2026-01-01",
                "2026-12-31",
                "1d",
                client=client,
            )
        finally:
            await client.aclose()
        assert response.signature["target"] == "499"


class TestDisabledCache:
    async def test_fetched_at_set_when_cache_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ASTRODYNAMICS_MCP_CACHE_DIR", "")
        disabled_cache = Cache()
        assert not disabled_cache.enabled

        before = datetime.now(tz=timezone.utc)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_SAMPLE_HORIZONS_PAYLOAD)

        client = _mock_client(handler)
        try:
            response = await fetch_ephemeris(
                "499",
                "@sun",
                "2026-01-01",
                "2026-12-31",
                "1d",
                client=client,
                cache=disabled_cache,
            )
        finally:
            await client.aclose()

        assert response.fetched_at >= before
        assert response.stale is False


@pytest.mark.integration
class TestLiveHorizons:
    """Hits the real Horizons endpoint; gated behind the integration marker."""

    async def test_live_mars_ephemeris(self, cache: Cache) -> None:
        response = await fetch_ephemeris(
            "499",
            "@sun",
            "2026-01-01",
            "2026-01-03",
            "1d",
            cache=cache,
        )
        assert isinstance(response, HorizonsResponse)
        # The real Horizons result is a text block — light sanity that it
        # mentions the target body somewhere.
        assert "Mars" in response.result or "499" in response.result
        assert response.stale is False
