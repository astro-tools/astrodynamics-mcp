"""Tests for the SPICE kernel-management tool bodies.

Drives the module-level ``_do_load_kernel`` / ``_do_list_kernels`` /
``_do_unload_kernel`` helpers (the thin registered slots wrap these) against the
in-memory ``FakeSpice`` and a ``tmp_path`` kernel cache. The test env has no real
``spiceypy``, so the worker's lazy ``import spiceypy`` resolves to the fake.

Covers the kernel-management acceptance contract: typed errors for a disallowed
URL / unreadable kernel / unload-of-missing, the allowlist + cache exercise (a
second URL load served from cache), meta-kernel fan-out, and the output-schema
round-trip per the v0.1 pattern.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest
from mcp.server.fastmcp import FastMCP

from astrodynamics_mcp.errors import InvalidInputError, UpstreamError
from astrodynamics_mcp.spice_kernels import KernelCache
from astrodynamics_mcp.tools import spice as spice_tools
from astrodynamics_mcp.tools.spice import (
    SpiceKernelInfo,
    SpiceListKernelsResponse,
    SpiceLoadKernelResponse,
    SpiceUnloadKernelResponse,
    _do_list_kernels,
    _do_load_kernel,
    _do_unload_kernel,
    _looks_like_url,
)
from tests._spice_fakes import FakeSpice, FakeSpiceyError

_NAIF = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels"
_LSK_URL = f"{_NAIF}/lsk/naif0012.tls"
_SPK_URL = f"{_NAIF}/spk/planets/de440.bsp"


@pytest.fixture
def fake_spice(monkeypatch: pytest.MonkeyPatch) -> FakeSpice:
    fake = FakeSpice()
    monkeypatch.setitem(sys.modules, "spiceypy", fake)
    return fake


@pytest.fixture
def cache(tmp_path: Path) -> KernelCache:
    return KernelCache(directory=tmp_path)


def _local_kernel(tmp_path: Path, name: str = "de440.bsp") -> str:
    path = tmp_path / name
    path.write_bytes(b"DAF/SPK fake kernel bytes")
    return str(path)


# ---------------------------------------------------------------------------
# URL detection
# ---------------------------------------------------------------------------


class TestLooksLikeUrl:
    @pytest.mark.parametrize(
        "source",
        [_SPK_URL, "http://naif.jpl.nasa.gov/x.bsp"],
    )
    def test_url_sources(self, source: str) -> None:
        assert _looks_like_url(source) is True

    @pytest.mark.parametrize(
        "source",
        ["/data/de440.bsp", "relative/de440.bsp", r"C:\kernels\de440.bsp"],
    )
    def test_local_paths(self, source: str) -> None:
        assert _looks_like_url(source) is False


# ---------------------------------------------------------------------------
# Load — local path
# ---------------------------------------------------------------------------


class TestLoadLocalKernel:
    async def test_loads_local_kernel(self, fake_spice: FakeSpice, tmp_path: Path) -> None:
        source = _local_kernel(tmp_path)
        response = await _do_load_kernel(source)
        assert isinstance(response, SpiceLoadKernelResponse)
        assert response.from_cache is False
        assert [k.name for k in response.loaded] == [source]
        assert response.loaded[0].type == "SPK"

    async def test_missing_local_kernel_is_typed_error(self, fake_spice: FakeSpice) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            await _do_load_kernel("/no/such/kernel.bsp")
        assert excinfo.value.code == "invalid_input.spice_kernel_not_found"
        assert fake_spice.calls["furnsh"] == []  # never reached CSPICE

    async def test_corrupt_kernel_is_typed_upstream_error(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        source = _local_kernel(tmp_path, "corrupt.bsp")
        fake_spice.fail_furnsh(source, FakeSpiceyError("bad DAF header"))
        with pytest.raises(UpstreamError) as excinfo:
            await _do_load_kernel(source)
        assert excinfo.value.code == "upstream.spice_furnish_failed"

    async def test_meta_kernel_fans_out(self, fake_spice: FakeSpice, tmp_path: Path) -> None:
        meta = tmp_path / "mission.tm"
        meta.write_text("\\begindata\nKERNELS_TO_LOAD = ( 'naif0012.tls', 'de440.bsp' )\n")
        fake_spice.plan_furnish(
            str(meta),
            [
                {"name": str(meta), "type": "META", "source": "", "handle": 0},
                {"name": "/k/naif0012.tls", "type": "TEXT", "source": str(meta), "handle": 0},
                {"name": "/k/de440.bsp", "type": "SPK", "source": str(meta), "handle": 9},
            ],
        )
        response = await _do_load_kernel(str(meta))
        assert {k.name for k in response.loaded} == {str(meta), "/k/naif0012.tls", "/k/de440.bsp"}
        assert {k.type for k in response.loaded} == {"META", "TEXT", "SPK"}


# ---------------------------------------------------------------------------
# Load — URL (allowlist + cache)
# ---------------------------------------------------------------------------


class TestLoadUrlKernel:
    @pytest.mark.parametrize(
        ("url", "code"),
        [
            ("http://naif.jpl.nasa.gov/x.bsp", "invalid_input.spice_kernel_url_scheme"),
            ("https://evil.example.com/x.bsp", "invalid_input.spice_kernel_url_host"),
        ],
    )
    async def test_disallowed_url_refused_before_fetch(
        self, fake_spice: FakeSpice, cache: KernelCache, url: str, code: str
    ) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            await _do_load_kernel(url, cache=cache)
        assert excinfo.value.code == code
        assert fake_spice.calls["furnsh"] == []

    async def test_second_url_load_served_from_cache(
        self, fake_spice: FakeSpice, cache: KernelCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = b"DAF/SPK de440 bytes"
        calls: list[int] = []

        def handler(_request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(200, content=body)

        real_client = httpx.AsyncClient

        def client_factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
            kwargs["transport"] = httpx.MockTransport(handler)
            return real_client(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr("astrodynamics_mcp.spice_kernels.httpx.AsyncClient", client_factory)

        first = await _do_load_kernel(_SPK_URL, cache=cache)
        second = await _do_load_kernel(_SPK_URL, cache=cache)

        # First load downloads (miss); second is served from the on-disk cache.
        assert first.from_cache is False
        assert second.from_cache is True
        assert len(calls) == 1, "second load must not hit the network"
        # Both resolve to the same cached blob path, furnished as an SPK.
        cached_path = str(cache.path_for(_SPK_URL))
        assert first.loaded[0].name == cached_path
        assert first.loaded[0].type == "SPK"

    async def test_cached_url_reports_from_cache_without_network(
        self, fake_spice: FakeSpice, cache: KernelCache
    ) -> None:
        # Pre-populate the cache so the load is a pure hit — no transport needed.
        blob = cache.path_for(_LSK_URL)
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(b"KPL/LSK fake leap seconds")

        response = await _do_load_kernel(_LSK_URL, cache=cache)
        assert response.from_cache is True
        assert response.loaded[0].name == str(blob)
        # `.tls` suffix is preserved on the blob name, so CSPICE-typed as TEXT.
        assert response.loaded[0].type == "TEXT"


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


class TestListKernels:
    async def test_lists_loaded_pool(self, fake_spice: FakeSpice, tmp_path: Path) -> None:
        await _do_load_kernel(_local_kernel(tmp_path, "naif0012.tls"))
        await _do_load_kernel(_local_kernel(tmp_path, "de440.bsp"))
        response = await _do_list_kernels(None)
        assert isinstance(response, SpiceListKernelsResponse)
        assert len(response.kernels) == 2

    async def test_category_filter(self, fake_spice: FakeSpice, tmp_path: Path) -> None:
        await _do_load_kernel(_local_kernel(tmp_path, "naif0012.tls"))  # TEXT
        await _do_load_kernel(_local_kernel(tmp_path, "de440.bsp"))  # SPK
        response = await _do_list_kernels(["SPK"])
        assert [k.type for k in response.kernels] == ["SPK"]

    async def test_empty_filter_is_typed_error(self, fake_spice: FakeSpice) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            await _do_list_kernels([])
        assert excinfo.value.code == "invalid_input.spice_empty_kind_filter"


# ---------------------------------------------------------------------------
# Unload
# ---------------------------------------------------------------------------


class TestUnloadKernel:
    async def test_unloads_and_reports_remaining(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        lsk = _local_kernel(tmp_path, "naif0012.tls")
        spk = _local_kernel(tmp_path, "de440.bsp")
        await _do_load_kernel(lsk)
        await _do_load_kernel(spk)
        response = await _do_unload_kernel(spk)
        assert isinstance(response, SpiceUnloadKernelResponse)
        assert response.unloaded == spk
        assert response.remaining_count == 1

    async def test_unload_missing_is_typed_error(self, fake_spice: FakeSpice) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            await _do_unload_kernel("/k/never-loaded.bsp")
        assert excinfo.value.code == "invalid_input.spice_kernel_not_loaded"


# ---------------------------------------------------------------------------
# Output-schema round-trip (the v0.1 pattern, sans spiceypy)
# ---------------------------------------------------------------------------


class TestOutputRoundTrip:
    @pytest.mark.parametrize(
        "response",
        [
            SpiceLoadKernelResponse(
                loaded=[SpiceKernelInfo(name="/k/de440.bsp", type="SPK", source="", handle=42)],
                from_cache=True,
            ),
            SpiceListKernelsResponse(
                kernels=[
                    SpiceKernelInfo(name="/k/naif0012.tls", type="TEXT", source="", handle=0),
                    SpiceKernelInfo(
                        name="/k/de440.bsp", type="SPK", source="/k/mission.tm", handle=3
                    ),
                ],
            ),
            SpiceUnloadKernelResponse(unloaded="/k/de440.bsp", remaining_count=2),
        ],
        ids=["load", "list", "unload"],
    )
    def test_response_roundtrips_through_schema(self, response: object) -> None:
        model = type(response)
        first = response.model_dump_json()  # type: ignore[attr-defined]
        rebuilt = model.model_validate_json(first)  # type: ignore[attr-defined]
        assert rebuilt.model_dump_json() == first

    def test_extra_keys_forbidden(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SpiceUnloadKernelResponse.model_validate(
                {"unloaded": "/k/x.bsp", "remaining_count": 1, "surprise": True}
            )


# ---------------------------------------------------------------------------
# End-to-end through the registered slots — exercises the thin wrappers and
# FastMCP's output-schema serialisation, the way an MCP client would call them.
# ---------------------------------------------------------------------------


class TestRegisteredToolCall:
    @pytest.fixture
    def registered_mcp(self, monkeypatch: pytest.MonkeyPatch) -> FastMCP:
        fresh = FastMCP("spice-kernels-test")
        monkeypatch.setattr("astrodynamics_mcp.server.mcp", fresh)
        spice_tools._register_spice_tools()
        return fresh

    async def test_load_list_unload_round_trip(
        self, registered_mcp: FastMCP, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        source = _local_kernel(tmp_path)

        _, loaded = await registered_mcp.call_tool("spice_load_kernel", {"source": source})
        assert isinstance(loaded, dict)
        assert loaded["from_cache"] is False
        assert loaded["loaded"][0]["name"] == source
        assert loaded["loaded"][0]["type"] == "SPK"

        _, listed = await registered_mcp.call_tool("spice_list_kernels", {})
        assert [k["name"] for k in listed["kernels"]] == [source]

        _, unloaded = await registered_mcp.call_tool("spice_unload_kernel", {"name": source})
        assert unloaded["unloaded"] == source
        assert unloaded["remaining_count"] == 0
