"""Tests for `astrodynamics_mcp.tools.satellite_metadata`.

Drive the tool against an ``httpx.MockTransport``-backed DISCOSweb
adapter so the tool tests stay end-to-end (tool body → credential gate →
adapter → mock DISCOSweb) without hitting the network. Registration and
description-lint coverage are exercised through the real module-level
:data:`astrodynamics_mcp.server.mcp` singleton.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from mcp.server.fastmcp.exceptions import ToolError

from astrodynamics_mcp.cache import Cache
from astrodynamics_mcp.server import mcp
from astrodynamics_mcp.server_lint import check_tool_descriptions
from astrodynamics_mcp.tools.satellite_metadata import (
    SatelliteMetadataResponse,
    satellite_metadata,
)


def _sample_payload_iss() -> dict[str, Any]:
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


def _sample_payload_partial() -> dict[str, Any]:
    """A record where mass and dimensions are absent — both nullable on the wire."""
    return {
        "data": [
            {
                "type": "Object",
                "id": "9999",
                "attributes": {
                    "name": "FRAG-DEB",
                    "satno": 88888,
                    "cosparId": None,
                    "mass": None,
                    "width": None,
                    "height": None,
                    "depth": None,
                    "objectClass": "Debris",
                },
                "relationships": {
                    "launch": {"data": None},
                    "operators": {"data": []},
                    "reentry": {"data": None},
                },
            }
        ],
        "included": [],
    }


def _empty_payload() -> dict[str, Any]:
    return {"data": [], "included": []}


@pytest.fixture
def tmp_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Cache:
    """Per-test cache rooted at a tmp dir, wired in as the module singleton.

    Mirrors the pattern in ``test_tool_tle.py``: the tool calls
    ``fetch_metadata`` with no cache argument, so the adapter falls back
    to ``default_cache()``. We force the singleton to a tmp directory and
    reset it so each test gets a clean slate.
    """
    monkeypatch.setenv("ASTRODYNAMICS_MCP_CACHE_DIR", str(tmp_path))
    import astrodynamics_mcp.cache as cache_module

    monkeypatch.setattr(cache_module, "_default_cache", None)
    return Cache(directory=tmp_path)


@pytest.fixture
def discosweb_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a non-empty bearer token in the env so the credential gate passes."""
    monkeypatch.setenv("ASTRODYNAMICS_MCP_DISCOSWEB_TOKEN", "test-bearer-token")


@pytest.fixture
def reset_discosweb_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the DISCOSweb module-level singleton between tests."""
    import astrodynamics_mcp.data.discosweb as discosweb_module

    monkeypatch.setattr(discosweb_module, "_singleton_client", None)


def _patched_discosweb_client(handler: Any) -> Any:
    """Patch the DISCOSweb adapter's singleton builder to use a MockTransport client."""
    from astrodynamics_mcp.data.discosweb import _BASE_URL

    return patch(
        "astrodynamics_mcp.data.discosweb._build_singleton_client",
        return_value=httpx.AsyncClient(
            base_url=_BASE_URL,
            transport=httpx.MockTransport(handler),
        ),
    )


def _static_handler(payload: Any, *, status: int = 200) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return handler


class TestHappyPath:
    async def test_iss_lookup_returns_typed_response(
        self,
        tmp_cache: Cache,
        discosweb_token: None,
        reset_discosweb_singleton: None,
    ) -> None:
        with _patched_discosweb_client(_static_handler(_sample_payload_iss())):
            response = await satellite_metadata(norad_id="25544")

        assert isinstance(response, SatelliteMetadataResponse)
        assert response.name == "ISS (ZARYA)"
        assert response.norad_id == "25544"
        assert response.cospar_id == "1998-067A"
        assert response.mass_kg is not None
        assert response.mass_kg.value == 420000.0
        assert response.mass_kg.unit == "kg"
        assert response.dimensions_m is not None
        assert response.dimensions_m.x.value == 73.0
        assert response.dimensions_m.y.value == 45.0
        assert response.dimensions_m.z.value == 27.5
        assert response.dimensions_m.x.unit == "m"
        assert response.launch_date == "1998-11-20T06:40:00.000Z"
        assert response.launch_site == "Baikonur Cosmodrome"
        assert response.owner == "NASA"
        assert response.mission_type == "Payload"
        assert response.decay_status == "active"
        assert response.decay_date is None
        assert response.source == "discosweb"
        assert response.stale is False

    async def test_partial_record_nulls_mass_and_dimensions(
        self,
        tmp_cache: Cache,
        discosweb_token: None,
        reset_discosweb_singleton: None,
    ) -> None:
        with _patched_discosweb_client(_static_handler(_sample_payload_partial())):
            response = await satellite_metadata(norad_id="88888")
        assert response.mass_kg is None
        assert response.dimensions_m is None
        assert response.owner is None
        assert response.launch_date is None
        assert response.launch_site is None

    async def test_fetched_at_is_iso_8601_with_time_component(
        self,
        tmp_cache: Cache,
        discosweb_token: None,
        reset_discosweb_singleton: None,
    ) -> None:
        with _patched_discosweb_client(_static_handler(_sample_payload_iss())):
            response = await satellite_metadata(norad_id="25544")
        parsed = datetime.fromisoformat(response.fetched_at)
        assert parsed.tzinfo is not None


class TestDecayMapping:
    async def test_reentry_relationship_maps_to_decayed(
        self,
        tmp_cache: Cache,
        discosweb_token: None,
        reset_discosweb_singleton: None,
    ) -> None:
        payload: dict[str, Any] = {
            "data": [
                {
                    "type": "Object",
                    "id": "1",
                    "attributes": {"name": "MIR", "satno": 16609, "objectClass": "Payload"},
                    "relationships": {"reentry": {"data": {"type": "Reentry", "id": "7"}}},
                }
            ],
            "included": [
                {
                    "type": "Reentry",
                    "id": "7",
                    "attributes": {"epoch": "2001-03-23T05:59:24Z"},
                },
            ],
        }
        with _patched_discosweb_client(_static_handler(payload)):
            response = await satellite_metadata(norad_id="16609")
        assert response.decay_status == "decayed"
        assert response.decay_date == "2001-03-23T05:59:24Z"


class TestMultipleOperators:
    async def test_multiple_operators_joined_with_comma(
        self,
        tmp_cache: Cache,
        discosweb_token: None,
        reset_discosweb_singleton: None,
    ) -> None:
        payload: dict[str, Any] = {
            "data": [
                {
                    "type": "Object",
                    "id": "1",
                    "attributes": {"name": "X", "satno": 11111, "objectClass": "Payload"},
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
                {"type": "Operator", "id": "1", "attributes": {"name": "ESA"}},
                {"type": "Operator", "id": "2", "attributes": {"name": "JAXA"}},
            ],
        }
        with _patched_discosweb_client(_static_handler(payload)):
            response = await satellite_metadata(norad_id="11111")
        assert response.owner == "ESA, JAXA"


class TestInputValidation:
    async def test_non_digit_norad_id_raises_envelope(
        self,
        tmp_cache: Cache,
        discosweb_token: None,
        reset_discosweb_singleton: None,
    ) -> None:
        """A name or arbitrary string is rejected before the credential gate."""
        with pytest.raises(ToolError) as excinfo:
            await satellite_metadata(norad_id="ISS")
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.norad_id_not_digits"

    async def test_empty_string_norad_id_rejected(
        self,
        tmp_cache: Cache,
        discosweb_token: None,
        reset_discosweb_singleton: None,
    ) -> None:
        with pytest.raises(ToolError) as excinfo:
            await satellite_metadata(norad_id="")
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.norad_id_not_digits"


class TestCredentialGate:
    async def test_missing_token_raises_credential_required_envelope(
        self,
        tmp_cache: Cache,
        monkeypatch: pytest.MonkeyPatch,
        reset_discosweb_singleton: None,
    ) -> None:
        """No creds → typed envelope, and the HTTP layer is never touched."""
        monkeypatch.delenv("ASTRODYNAMICS_MCP_DISCOSWEB_TOKEN", raising=False)

        handler_called = {"hit": False}

        def handler(request: httpx.Request) -> httpx.Response:
            handler_called["hit"] = True
            return httpx.Response(200, json=_sample_payload_iss())

        with _patched_discosweb_client(handler), pytest.raises(ToolError) as excinfo:
            await satellite_metadata(norad_id="25544")

        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "credential_required.discosweb"
        assert envelope["data"]["source"] == "discosweb"
        assert envelope["data"]["missing_fields"] == ["token"]
        assert handler_called["hit"] is False


class TestUnknownNoradId:
    async def test_empty_response_raises_not_found_envelope(
        self,
        tmp_cache: Cache,
        discosweb_token: None,
        reset_discosweb_singleton: None,
    ) -> None:
        with (
            _patched_discosweb_client(_static_handler(_empty_payload())),
            pytest.raises(ToolError) as excinfo,
        ):
            await satellite_metadata(norad_id="99999999")
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "data_source.discosweb_norad_not_found"
        assert envelope["data"]["source"] == "discosweb"
        assert envelope["data"]["norad_id"] == "99999999"


class TestCacheHit:
    async def test_second_call_within_ttl_serves_from_cache(
        self,
        tmp_cache: Cache,
        discosweb_token: None,
        reset_discosweb_singleton: None,
    ) -> None:
        """Two identical calls → one network request, courtesy of the 24h DISCOSweb TTL."""
        call_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            return httpx.Response(200, json=_sample_payload_iss())

        with _patched_discosweb_client(handler):
            first = await satellite_metadata(norad_id="25544")
            second = await satellite_metadata(norad_id="25544")

        assert call_count["n"] == 1
        assert first.stale is False
        assert second.stale is False
        assert tmp_cache.directory is not None
        assert (tmp_cache.directory / "discosweb").is_dir()


class TestStaleFallback:
    async def test_stale_flag_propagates_to_response(
        self,
        tmp_cache: Cache,
        discosweb_token: None,
        reset_discosweb_singleton: None,
    ) -> None:
        import astrodynamics_mcp.data.discosweb as discosweb_module

        # Seed.
        with _patched_discosweb_client(_static_handler(_sample_payload_iss())):
            await satellite_metadata(norad_id="25544")

        # Backdate.
        cache_path = tmp_cache._path("discosweb", "norad:25544")
        payload = json.loads(cache_path.read_text())
        payload["fetched_at"] = (datetime.now(tz=timezone.utc) - timedelta(days=2)).isoformat()
        cache_path.write_text(json.dumps(payload))

        # The first ``with`` block already built a singleton via the patched
        # builder; clear it so the second ``with`` block's failure handler
        # actually wires in, instead of reusing the success client.
        discosweb_module._singleton_client = None

        def fail_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("simulated outage")

        with _patched_discosweb_client(fail_handler):
            response = await satellite_metadata(norad_id="25544")
        assert response.stale is True
        assert response.name == "ISS (ZARYA)"


class TestErrorPropagation:
    async def test_data_source_error_surfaces_as_typed_envelope(
        self,
        tmp_cache: Cache,
        discosweb_token: None,
        reset_discosweb_singleton: None,
    ) -> None:
        def fail_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("simulated outage")

        with (
            _patched_discosweb_client(fail_handler),
            pytest.raises(ToolError) as excinfo,
        ):
            await satellite_metadata(norad_id="25544")

        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "data_source.discosweb_unreachable"
        assert envelope["data"]["source"] == "discosweb"

    async def test_auth_failure_surfaces_as_typed_envelope(
        self,
        tmp_cache: Cache,
        discosweb_token: None,
        reset_discosweb_singleton: None,
    ) -> None:
        with (
            _patched_discosweb_client(_static_handler({"errors": ["bad token"]}, status=401)),
            pytest.raises(ToolError) as excinfo,
        ):
            await satellite_metadata(norad_id="25544")

        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "data_source.discosweb_auth_failed"
        assert envelope["data"]["source"] == "discosweb"


class TestRegistration:
    """The tool registers against the module-level FastMCP singleton on import."""

    async def test_tool_is_listed(self) -> None:
        tools = await mcp.list_tools()
        names = {t.name for t in tools}
        assert "satellite_metadata" in names

    async def test_tool_description_passes_lint(self) -> None:
        tools = await mcp.list_tools()
        violations = [
            v for v in check_tool_descriptions(tools) if v.tool_name == "satellite_metadata"
        ]
        assert violations == []

    async def test_tool_callable_via_mcp(
        self,
        tmp_cache: Cache,
        discosweb_token: None,
        reset_discosweb_singleton: None,
    ) -> None:
        """End-to-end: invoking through ``mcp.call_tool`` returns the structured response."""
        with _patched_discosweb_client(_static_handler(_sample_payload_iss())):
            content, structured = await mcp.call_tool("satellite_metadata", {"norad_id": "25544"})
        del content
        assert isinstance(structured, dict)
        assert structured["norad_id"] == "25544"
        assert structured["name"] == "ISS (ZARYA)"
        assert structured["mass_kg"]["value"] == 420000.0
        assert structured["dimensions_m"]["x"]["value"] == 73.0


class TestSchemaInvariants:
    async def test_response_round_trips_through_json(
        self,
        tmp_cache: Cache,
        discosweb_token: None,
        reset_discosweb_singleton: None,
    ) -> None:
        with _patched_discosweb_client(_static_handler(_sample_payload_iss())):
            response = await satellite_metadata(norad_id="25544")
        as_json = response.model_dump_json()
        rebuilt = SatelliteMetadataResponse.model_validate_json(as_json)
        assert rebuilt == response
