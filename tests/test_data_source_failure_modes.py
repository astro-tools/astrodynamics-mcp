"""Cross-tool failure-mode contract — tool-layer surfaces typed codes when upstream fails.

The per-adapter tests under ``test_data_celestrak.py`` /
``test_data_horizons.py`` / ``test_data_iers.py`` already cover the
network / cache / stale-fallback machinery inside each adapter. This
module is the *tool*-layer companion: it asserts that when one of those
adapters raises a typed error, the registered tool propagates the
typed code intact through the FastMCP boundary so the LLM consumer
sees the documented ``data_source.*`` / ``upstream.*`` code rather
than a leaked traceback.

Each case mocks the adapter to its documented failure mode and asserts
the tool surfaces the right code via ``mcp.call_tool``.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from mcp.server.fastmcp.exceptions import ToolError

from astrodynamics_mcp.schemas.base import TimeScale
from astrodynamics_mcp.tools.time import time_convert
from astrodynamics_mcp.tools.tle import tle_lookup


@contextmanager
def _empty_cache() -> Iterator[None]:
    """Run with an empty per-test cache so stale-fallback paths can't fire."""
    import astrodynamics_mcp.cache as cache_module

    with tempfile.TemporaryDirectory() as cache_dir:
        previous_default = cache_module._default_cache
        previous_env = os.environ.get("ASTRODYNAMICS_MCP_CACHE_DIR")
        os.environ["ASTRODYNAMICS_MCP_CACHE_DIR"] = cache_dir
        cache_module._default_cache = None
        try:
            yield
        finally:
            cache_module._default_cache = previous_default
            if previous_env is None:
                os.environ.pop("ASTRODYNAMICS_MCP_CACHE_DIR", None)
            else:
                os.environ["ASTRODYNAMICS_MCP_CACHE_DIR"] = previous_env


def _parse_tool_error_envelope(raw: str) -> dict[str, Any]:
    """The FastMCP wire layer carries our JSON envelope inside ToolError(...)."""
    envelope: dict[str, Any] = json.loads(raw)
    return envelope


class TestCelestrakUnreachableSurfaceAtToolLayer:
    """tle_lookup must surface `data_source.celestrak_unreachable` when CelesTrak is down."""

    async def test_tool_propagates_typed_data_source_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="celestrak temporarily unavailable")

        original_client = httpx.AsyncClient

        def factory(*_args: Any, **_kwargs: Any) -> httpx.AsyncClient:
            return original_client(transport=httpx.MockTransport(handler))

        with (
            _empty_cache(),
            patch(
                "astrodynamics_mcp.data.celestrak.httpx.AsyncClient",
                side_effect=factory,
            ),
            pytest.raises(ToolError) as excinfo,
        ):
            await tle_lookup(query="25544")

        envelope = _parse_tool_error_envelope(str(excinfo.value))
        assert envelope["code"] == "data_source.celestrak_unreachable"
        assert envelope["data"]["source"] == "celestrak"


class TestIersUnavailableSurfaceAtToolLayer:
    """time_convert(UTC ↔ UT1) must surface `upstream.iers_unavailable` when IERS load fails."""

    async def test_tool_propagates_typed_upstream_error_for_iers(self) -> None:
        def fake_iers_open(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("IERS Bulletin A unreachable in test")

        with (
            patch(
                "astropy.utils.iers.IERS_Auto.open",
                side_effect=fake_iers_open,
            ),
            pytest.raises(ToolError) as excinfo,
        ):
            await time_convert(
                value="2026-05-23T12:00:00",
                from_scale=TimeScale.UTC,
                to_scale=TimeScale.UT1,
            )

        envelope = _parse_tool_error_envelope(str(excinfo.value))
        assert envelope["code"] == "upstream.iers_unavailable"


class TestHorizonsUnreachableSurfaceAtToolLayer:
    """porkchop must surface `data_source.horizons_unreachable` when Horizons errors out."""

    async def test_tool_propagates_typed_data_source_error(self) -> None:
        from astrodynamics_mcp.tools.porkchop import porkchop

        def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Horizons unreachable in test")

        original_client = httpx.AsyncClient

        def factory(*_args: Any, **_kwargs: Any) -> httpx.AsyncClient:
            return original_client(transport=httpx.MockTransport(handler))

        with (
            _empty_cache(),
            patch(
                "astrodynamics_mcp.data.horizons.httpx.AsyncClient",
                side_effect=factory,
            ),
            pytest.raises(ToolError) as excinfo,
        ):
            await porkchop(
                departure_body="earth",
                arrival_body="mars",
                depart_window=["2026-11-01T00:00:00Z", "2026-12-31T00:00:00Z"],
                arrive_window=["2027-06-01T00:00:00Z", "2027-11-01T00:00:00Z"],
                samples_per_axis=3,
            )

        envelope = _parse_tool_error_envelope(str(excinfo.value))
        assert envelope["code"] == "data_source.horizons_unreachable"
        assert envelope["data"]["source"] == "horizons"
