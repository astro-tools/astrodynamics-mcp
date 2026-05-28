"""Tests for `astrodynamics_mcp.data.discosweb`.

Drive the adapter against ``httpx.MockTransport`` for the bearer-token
auth flow and the failure modes that mirror the Space-Track adapter
(happy path, cache hit, stale fallback, hard failure with no cache,
malformed payload). DISCOSweb-specific behaviours — empty-data response
for unknown NORAD IDs, refused token surfacing as
``data_source.discosweb_auth_failed`` — get dedicated classes.

No live-integration class. Real DISCOSweb exercise happens locally with
maintainer credentials, not in CI (per-account rate limits and the
free-tier daily quota make CI a poor fit, same posture as Space-Track).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from astrodynamics_mcp.cache import Cache
from astrodynamics_mcp.data.discosweb import (
    _BASE_URL,
    DiscoswebResponse,
    fetch_metadata,
)
from astrodynamics_mcp.errors import DataSourceError, UpstreamError

_CREDENTIAL = {"token": "iss-bearer-token-xyz"}


def _sample_payload_iss() -> dict[str, Any]:
    """A DISCOSweb response for the ISS (NORAD 25544), populated end-to-end.

    Carries every field the adapter looks at — mass, dimensions, launch
    relationship with inlined site, operator, and an absent reentry — so
    one fixture covers the happy-path mapping for every nullable axis.
    """
    return {
        "data": [
            {
                "type": "Object",
                "id": "61",
                "attributes": {
                    "name": "ISS (ZARYA)",
                    "satno": 25544,
                    "cosparId": "1998-067A",
                    "mass": 420000.0,
                    "width": 73.0,
                    "height": 45.0,
                    "depth": 27.5,
                    "objectClass": "Payload",
                },
                "relationships": {
                    "launch": {"data": {"type": "Launch", "id": "12"}},
                    "operators": {"data": [{"type": "Operator", "id": "17"}]},
                    "reentry": {"data": None},
                },
            }
        ],
        "included": [
            {
                "type": "Launch",
                "id": "12",
                "attributes": {"epoch": "1998-11-20T06:40:00.000Z"},
                "relationships": {"site": {"data": {"type": "LaunchSite", "id": "8"}}},
            },
            {
                "type": "LaunchSite",
                "id": "8",
                "attributes": {"name": "Baikonur Cosmodrome"},
            },
            {
                "type": "Operator",
                "id": "17",
                "attributes": {"name": "NASA"},
            },
        ],
    }


def _sample_payload_decayed() -> dict[str, Any]:
    """A DISCOSweb response with a populated reentry relationship."""
    return {
        "data": [
            {
                "type": "Object",
                "id": "16489",
                "attributes": {
                    "name": "MIR",
                    "satno": 16609,
                    "cosparId": "1986-017A",
                    "mass": 129700.0,
                    "width": None,
                    "height": None,
                    "depth": None,
                    "objectClass": "Payload",
                },
                "relationships": {
                    "launch": {"data": None},
                    "operators": {"data": []},
                    "reentry": {"data": {"type": "Reentry", "id": "777"}},
                },
            }
        ],
        "included": [
            {
                "type": "Reentry",
                "id": "777",
                "attributes": {"epoch": "2001-03-23T05:59:24Z"},
            },
        ],
    }


def _empty_payload() -> dict[str, Any]:
    """The DISCOSweb response shape for an unknown NORAD ID."""
    return {"data": [], "included": []}


@pytest.fixture
def cache(tmp_path: Path) -> Cache:
    """Per-test cache rooted at a tmp dir; never touches the real XDG cache."""
    return Cache(directory=tmp_path)


@pytest.fixture(autouse=True)
def reset_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the module-level singleton client between tests."""
    import astrodynamics_mcp.data.discosweb as discosweb_module

    monkeypatch.setattr(discosweb_module, "_singleton_client", None)


def _mock_client(handler: Any) -> httpx.AsyncClient:
    """Build a MockTransport-backed client with the production base_url."""
    return httpx.AsyncClient(
        base_url=_BASE_URL,
        transport=httpx.MockTransport(handler),
    )


def _static_handler(payload: Any, *, status: int = 200) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return handler


class TestAuthHeader:
    async def test_bearer_token_sent_in_authorization_header(self, cache: Cache) -> None:
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["authorization"] = request.headers.get("authorization", "")
            return httpx.Response(200, json=_sample_payload_iss())

        client = _mock_client(handler)
        try:
            await fetch_metadata("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert seen["authorization"] == "Bearer iss-bearer-token-xyz"

    async def test_401_raises_auth_failed(self, cache: Cache) -> None:
        client = _mock_client(_static_handler({"errors": ["bad token"]}, status=401))
        try:
            with pytest.raises(DataSourceError) as excinfo:
                await fetch_metadata("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert excinfo.value.code == "data_source.discosweb_auth_failed"
        assert excinfo.value.source == "discosweb"

    async def test_403_raises_auth_failed(self, cache: Cache) -> None:
        client = _mock_client(_static_handler({"errors": ["forbidden"]}, status=403))
        try:
            with pytest.raises(DataSourceError) as excinfo:
                await fetch_metadata("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert excinfo.value.code == "data_source.discosweb_auth_failed"

    async def test_auth_failure_does_not_serve_stale_cache(self, cache: Cache) -> None:
        """A refused credential is permanent — never fall back to a cached value."""
        # Seed the cache.
        client = _mock_client(_static_handler(_sample_payload_iss()))
        try:
            await fetch_metadata("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()

        # Now the upstream rejects the token. We must surface auth_failed,
        # not silently return the cached value as stale.
        client = _mock_client(_static_handler({"errors": ["unauthorized"]}, status=401))
        try:
            with pytest.raises(DataSourceError) as excinfo:
                # Force a network call by backdating the cache past its TTL.
                path = cache._path("discosweb", "norad:25544")
                import json as _json

                payload = _json.loads(path.read_text())
                payload["fetched_at"] = (
                    datetime.now(tz=timezone.utc) - timedelta(days=2)
                ).isoformat()
                path.write_text(_json.dumps(payload))
                await fetch_metadata("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert excinfo.value.code == "data_source.discosweb_auth_failed"


class TestQueryDispatch:
    async def test_norad_id_passed_via_satno_filter(self, cache: Cache) -> None:
        seen_params: list[dict[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_params.append(dict(request.url.params))
            return httpx.Response(200, json=_sample_payload_iss())

        client = _mock_client(handler)
        try:
            await fetch_metadata("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert seen_params[0]["filter"] == "eq(satno,25544)"
        assert "launch" in seen_params[0]["include"]
        assert "reentry" in seen_params[0]["include"]
        assert "operators" in seen_params[0]["include"]

    async def test_request_targets_objects_endpoint(self, cache: Cache) -> None:
        seen_paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_paths.append(request.url.path)
            return httpx.Response(200, json=_sample_payload_iss())

        client = _mock_client(handler)
        try:
            await fetch_metadata("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert seen_paths == ["/api/objects"]


class TestHappyPath:
    async def test_returns_typed_response_with_full_mapping(self, cache: Cache) -> None:
        client = _mock_client(_static_handler(_sample_payload_iss()))
        try:
            response = await fetch_metadata("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()

        assert isinstance(response, DiscoswebResponse)
        assert response.stale is False
        record = response.record
        assert record is not None
        assert record.norad_id == 25544
        assert record.name == "ISS (ZARYA)"
        assert record.cospar_id == "1998-067A"
        assert record.mass_kg == 420000.0
        assert record.width_m == 73.0
        assert record.height_m == 45.0
        assert record.depth_m == 27.5
        assert record.object_class == "Payload"
        assert record.launch_date == "1998-11-20T06:40:00.000Z"
        assert record.launch_site_name == "Baikonur Cosmodrome"
        assert record.operator_names == ["NASA"]
        assert record.decay_date is None

    async def test_decayed_object_carries_reentry_epoch(self, cache: Cache) -> None:
        client = _mock_client(_static_handler(_sample_payload_decayed()))
        try:
            response = await fetch_metadata("16609", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        record = response.record
        assert record is not None
        assert record.decay_date == "2001-03-23T05:59:24Z"
        # Launch absent → no launch_date / site.
        assert record.launch_date is None
        assert record.launch_site_name is None
        # Empty operators list.
        assert record.operator_names == []

    async def test_caches_response_skipping_second_network_call(self, cache: Cache) -> None:
        request_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            request_count["n"] += 1
            return httpx.Response(200, json=_sample_payload_iss())

        client = _mock_client(handler)
        try:
            await fetch_metadata("25544", _CREDENTIAL, client=client, cache=cache)
            response2 = await fetch_metadata("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert request_count["n"] == 1
        assert response2.stale is False
        assert response2.record is not None
        assert response2.record.norad_id == 25544


class TestUnknownNoradId:
    async def test_empty_data_array_returns_none_record(self, cache: Cache) -> None:
        client = _mock_client(_static_handler(_empty_payload()))
        try:
            response = await fetch_metadata("99999999", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert response.record is None
        assert response.stale is False

    async def test_empty_data_array_is_cached(self, cache: Cache) -> None:
        request_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            request_count["n"] += 1
            return httpx.Response(200, json=_empty_payload())

        client = _mock_client(handler)
        try:
            await fetch_metadata("99999999", _CREDENTIAL, client=client, cache=cache)
            await fetch_metadata("99999999", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert request_count["n"] == 1


class TestStaleFallback:
    async def test_network_failure_with_cache_returns_stale(self, cache: Cache) -> None:
        # Seed the cache via a successful fetch.
        client = _mock_client(_static_handler(_sample_payload_iss()))
        try:
            first = await fetch_metadata("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert first.stale is False

        # Backdate the cache so the next call goes to the network.
        path = cache._path("discosweb", "norad:25544")
        import json as _json

        payload = _json.loads(path.read_text())
        payload["fetched_at"] = (datetime.now(tz=timezone.utc) - timedelta(days=2)).isoformat()
        path.write_text(_json.dumps(payload))

        def fail_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("simulated outage")

        client = _mock_client(fail_handler)
        try:
            second = await fetch_metadata("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert second.stale is True
        assert second.record is not None
        assert second.record.norad_id == 25544
        age_days = (datetime.now(tz=timezone.utc) - second.fetched_at).days
        assert age_days >= 1


class TestHardFailureNoCache:
    async def test_network_failure_no_cache_raises(self, cache: Cache) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("simulated outage")

        client = _mock_client(handler)
        try:
            with pytest.raises(DataSourceError) as excinfo:
                await fetch_metadata("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert excinfo.value.code == "data_source.discosweb_unreachable"
        assert excinfo.value.source == "discosweb"

    async def test_http_500_no_cache_raises(self, cache: Cache) -> None:
        client = _mock_client(_static_handler({"error": "service unavailable"}, status=503))
        try:
            with pytest.raises(DataSourceError) as excinfo:
                await fetch_metadata("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert excinfo.value.code == "data_source.discosweb_unreachable"


class TestUpstreamShapeGuard:
    async def test_non_object_payload_raises_upstream_error(self, cache: Cache) -> None:
        client = _mock_client(_static_handler([1, 2, 3]))
        try:
            with pytest.raises(UpstreamError) as excinfo:
                await fetch_metadata("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert excinfo.value.code == "upstream.discosweb_unexpected_shape"
        assert cache.get_stale("discosweb", "norad:25544") is None

    async def test_missing_data_key_raises_upstream_error(self, cache: Cache) -> None:
        client = _mock_client(_static_handler({"meta": {"total": 0}}))
        try:
            with pytest.raises(UpstreamError) as excinfo:
                await fetch_metadata("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert excinfo.value.code == "upstream.discosweb_unexpected_shape"

    async def test_invalid_json_response_raises_data_source_error(self, cache: Cache) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>maintenance</html>")

        client = _mock_client(handler)
        try:
            with pytest.raises(DataSourceError):
                await fetch_metadata("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()


class TestMalformedRecord:
    async def test_missing_attributes_raises_upstream_error(self, cache: Cache) -> None:
        broken = {"data": [{"type": "Object", "id": "61"}]}  # no attributes
        client = _mock_client(_static_handler(broken))
        try:
            with pytest.raises(UpstreamError) as excinfo:
                await fetch_metadata("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert excinfo.value.code == "upstream.discosweb_invalid_record"
        assert cache.get_stale("discosweb", "norad:25544") is None

    async def test_non_numeric_satno_raises_upstream_error(self, cache: Cache) -> None:
        broken: dict[str, Any] = {
            "data": [
                {
                    "type": "Object",
                    "id": "61",
                    "attributes": {"name": "X", "satno": "not-a-number"},
                }
            ]
        }
        client = _mock_client(_static_handler(broken))
        try:
            with pytest.raises(UpstreamError) as excinfo:
                await fetch_metadata("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert excinfo.value.code == "upstream.discosweb_invalid_record"


class TestPartialPayloads:
    async def test_missing_relationships_section_is_tolerated(self, cache: Cache) -> None:
        """Some objects have no operators / launch / reentry data at all."""
        minimal = {
            "data": [
                {
                    "type": "Object",
                    "id": "9",
                    "attributes": {"name": "Frag-1", "satno": 12345, "mass": None},
                }
            ]
        }
        client = _mock_client(_static_handler(minimal))
        try:
            response = await fetch_metadata("12345", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert response.record is not None
        assert response.record.norad_id == 12345
        assert response.record.mass_kg is None
        assert response.record.operator_names == []
        assert response.record.launch_date is None
        assert response.record.decay_date is None


class TestDefensiveParsing:
    """Direct unit coverage for the JSON-API shape guards in ``_parse_payload``.

    DISCOSweb is well-behaved in practice, but the parser is the boundary
    between an untrusted upstream and our typed model. These cases exercise
    the defensive paths so a future upstream schema drift surfaces as a
    typed error rather than a leaked traceback.
    """

    async def test_non_object_data_element_raises(self, cache: Cache) -> None:
        client = _mock_client(_static_handler({"data": ["string-instead-of-object"]}))
        try:
            with pytest.raises(UpstreamError) as excinfo:
                await fetch_metadata("25544", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert excinfo.value.code == "upstream.discosweb_unexpected_shape"

    async def test_garbage_included_entries_are_ignored(self, cache: Cache) -> None:
        """Non-dict ``included`` entries and entries with bad type/id fields are skipped."""
        payload: dict[str, Any] = {
            "data": [
                {
                    "type": "Object",
                    "id": "1",
                    "attributes": {"name": "X", "satno": 999, "objectClass": "Payload"},
                    "relationships": {
                        "launch": {"data": {"type": "Launch", "id": "12"}},
                        "operators": {
                            "data": [
                                "not-a-dict",  # skipped
                                {"type": 5, "id": "9"},  # non-string type skipped
                                {"type": "Operator", "id": "missing"},  # not in index
                            ]
                        },
                    },
                }
            ],
            "included": [
                "not-a-dict",  # skipped
                {"type": 9, "id": "1"},  # non-string fields skipped
                {
                    "type": "Launch",
                    "id": "12",
                    "attributes": {"epoch": "2020-01-01T00:00:00Z"},
                },
            ],
        }
        client = _mock_client(_static_handler(payload))
        try:
            response = await fetch_metadata("999", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert response.record is not None
        assert response.record.launch_date == "2020-01-01T00:00:00Z"
        assert response.record.operator_names == []  # every operator ref skipped

    async def test_relationship_with_non_dict_data_is_ignored(self, cache: Cache) -> None:
        """``relationships.launch.data: "garbage"`` should not crash the parser."""
        payload: dict[str, Any] = {
            "data": [
                {
                    "type": "Object",
                    "id": "1",
                    "attributes": {"name": "X", "satno": 777, "objectClass": "Payload"},
                    "relationships": {
                        "launch": {"data": "garbage"},
                        "operators": {"data": "garbage"},
                        "reentry": {"data": "garbage"},
                    },
                }
            ]
        }
        client = _mock_client(_static_handler(payload))
        try:
            response = await fetch_metadata("777", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert response.record is not None
        assert response.record.launch_date is None
        assert response.record.operator_names == []
        assert response.record.decay_date is None

    async def test_operator_with_non_dict_attributes_is_ignored(self, cache: Cache) -> None:
        payload: dict[str, Any] = {
            "data": [
                {
                    "type": "Object",
                    "id": "1",
                    "attributes": {"name": "X", "satno": 555, "objectClass": "Payload"},
                    "relationships": {
                        "operators": {
                            "data": [
                                {"type": "Operator", "id": "1"},
                                {"type": "Operator", "id": "2"},
                            ]
                        },
                    },
                }
            ],
            "included": [
                {"type": "Operator", "id": "1", "attributes": "not-a-dict"},
                {"type": "Operator", "id": "2", "attributes": {"name": ""}},
            ],
        }
        client = _mock_client(_static_handler(payload))
        try:
            response = await fetch_metadata("555", _CREDENTIAL, client=client, cache=cache)
        finally:
            await client.aclose()
        assert response.record is not None
        assert response.record.operator_names == []


class TestSingletonClientPath:
    """The production path (``client=None``) lazy-builds a module-level client."""

    async def test_singleton_is_built_and_reused(self, cache: Cache) -> None:
        request_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            request_count["n"] += 1
            return httpx.Response(200, json=_sample_payload_iss())

        mock_built = _mock_client(handler)
        with patch(
            "astrodynamics_mcp.data.discosweb._build_singleton_client",
            return_value=mock_built,
        ):
            try:
                first = await fetch_metadata("25544", _CREDENTIAL, cache=cache)
                # Different NORAD ID — cache miss, but the singleton is reused.
                second = await fetch_metadata("20580", _CREDENTIAL, cache=cache)
            finally:
                await mock_built.aclose()

        assert first.record is not None
        assert second.record is not None
        assert request_count["n"] == 2
