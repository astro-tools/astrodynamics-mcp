"""Tests for `astrodynamics_mcp.spice_kernels`.

Covers the NAIF furnish-from-URL policy: the URL allowlist, the on-disk kernel
cache (XDG ``kernels/`` subdir, TTL, atomic store), and the streaming
fetch-through-cache — redirects re-validated per hop, body capped, network
skipped on a cache hit. The fetch path is exercised with an in-memory
``httpx.MockTransport`` so no real NAIF traffic is made.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from astrodynamics_mcp.errors import InvalidInputError, UpstreamError
from astrodynamics_mcp.spice_kernels import (
    KernelCache,
    _default_max_bytes,
    _default_ttl_s,
    default_kernel_cache,
    validate_kernel_url,
)

_NAIF = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels"
_KERNEL_URL = f"{_NAIF}/spk/planets/de440.bsp"


@pytest.fixture
def cache(tmp_path: Path) -> KernelCache:
    return KernelCache(directory=tmp_path)


def _mock_client(handler: object) -> httpx.AsyncClient:
    """An async client whose every request is served by *handler* in memory.

    ``follow_redirects=False`` matches the production client config so the
    module's own per-hop redirect re-validation is what gets exercised.
    """
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
        follow_redirects=False,
    )


async def _astream(*chunks: bytes) -> AsyncIterator[bytes]:
    """A streaming (no Content-Length) response body for cap tests."""
    for chunk in chunks:
        yield chunk


# ---------------------------------------------------------------------------
# validate_kernel_url
# ---------------------------------------------------------------------------


class TestValidateKernelUrl:
    def test_accepts_naif_https(self) -> None:
        assert validate_kernel_url(_KERNEL_URL) == _KERNEL_URL

    def test_rejects_non_https_scheme(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            validate_kernel_url("http://naif.jpl.nasa.gov/x.bsp")
        assert excinfo.value.code == "invalid_input.spice_kernel_url_scheme"

    def test_rejects_off_allowlist_host(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            validate_kernel_url("https://evil.example.com/x.bsp")
        assert excinfo.value.code == "invalid_input.spice_kernel_url_host"

    def test_rejects_lookalike_host(self) -> None:
        # A suffix-attached lookalike must not pass a substring-style check.
        with pytest.raises(InvalidInputError) as excinfo:
            validate_kernel_url("https://naif.jpl.nasa.gov.evil.com/x.bsp")
        assert excinfo.value.code == "invalid_input.spice_kernel_url_host"


# ---------------------------------------------------------------------------
# KernelCache storage primitives
# ---------------------------------------------------------------------------


class TestKernelCacheStorage:
    def test_enabled_under_tmp_dir(self, cache: KernelCache) -> None:
        assert cache.enabled is True
        assert cache.directory is not None
        assert cache.directory.name == "kernels"

    def test_disabled_when_cache_dir_blank(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ASTRODYNAMICS_MCP_CACHE_DIR", "")
        disabled = KernelCache()
        assert disabled.enabled is False
        assert disabled.directory is None
        assert disabled.get(_KERNEL_URL, ttl_s=10) is None
        assert disabled.is_cached(_KERNEL_URL) is False
        with pytest.raises(RuntimeError):
            disabled.path_for(_KERNEL_URL)

    def test_path_for_is_under_kernels_dir_and_url_keyed(self, cache: KernelCache) -> None:
        path = cache.path_for(_KERNEL_URL)
        assert cache.directory is not None
        assert path.parent == cache.directory
        # The blob name preserves the URL's short suffix for grep-ability but is
        # keyed by the URL hash, so a different URL maps to a different file.
        assert path.name.endswith(".bsp")
        other = cache.path_for(f"{_NAIF}/lsk/naif0012.tls")
        assert other.name != path.name

    def test_get_miss_returns_none(self, cache: KernelCache) -> None:
        assert cache.get(_KERNEL_URL, ttl_s=10) is None

    def test_get_honours_ttl(self, cache: KernelCache) -> None:
        path = cache.path_for(_KERNEL_URL)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"DAF/SPK")
        old = time.time() - 100
        os.utime(path, (old, old))
        assert cache.get(_KERNEL_URL, ttl_s=50) is None  # stale
        assert cache.get(_KERNEL_URL, ttl_s=200) == path  # fresh

    def test_is_cached_tracks_freshness(self, cache: KernelCache) -> None:
        assert cache.is_cached(_KERNEL_URL, ttl_s=200) is False  # miss
        path = cache.path_for(_KERNEL_URL)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"DAF/SPK")
        assert cache.is_cached(_KERNEL_URL, ttl_s=200) is True  # fresh hit
        old = time.time() - 100
        os.utime(path, (old, old))
        assert cache.is_cached(_KERNEL_URL, ttl_s=50) is False  # stale → miss


# ---------------------------------------------------------------------------
# fetch-through-cache
# ---------------------------------------------------------------------------


class TestKernelCacheFetch:
    async def test_miss_downloads_and_stores(self, cache: KernelCache) -> None:
        body = b"DAF/SPK kernel bytes"

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body)

        async with _mock_client(handler) as client:
            path = await cache.fetch(_KERNEL_URL, client=client)
        assert path == cache.path_for(_KERNEL_URL)
        assert path.read_bytes() == body

    async def test_hit_skips_the_network(self, cache: KernelCache) -> None:
        body = b"cached kernel"
        calls: list[int] = []

        def handler(_request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(200, content=body)

        async with _mock_client(handler) as client:
            first = await cache.fetch(_KERNEL_URL, client=client)
            second = await cache.fetch(_KERNEL_URL, client=client)
        assert first == second
        assert len(calls) == 1  # second call served from disk

    async def test_follows_allowlisted_redirect(self, cache: KernelCache) -> None:
        final = b"redirected kernel bytes"
        start = f"{_NAIF}/spk/start.bsp"
        target = f"{_NAIF}/spk/final.bsp"

        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url) == start:
                return httpx.Response(302, headers={"location": target})
            return httpx.Response(200, content=final)

        async with _mock_client(handler) as client:
            path = await cache.fetch(start, client=client)
        # Cached under the *original* URL, with the redirect's body.
        assert path == cache.path_for(start)
        assert path.read_bytes() == final

    async def test_rejects_redirect_off_allowlist(self, cache: KernelCache) -> None:
        start = f"{_NAIF}/spk/escape.bsp"

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(302, headers={"location": "https://evil.example.com/x.bsp"})

        async with _mock_client(handler) as client:
            with pytest.raises(InvalidInputError) as excinfo:
                await cache.fetch(start, client=client)
        assert excinfo.value.code == "invalid_input.spice_kernel_url_host"

    async def test_redirect_without_location_fails(self, cache: KernelCache) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(302)

        async with _mock_client(handler) as client:
            with pytest.raises(UpstreamError) as excinfo:
                await cache.fetch(_KERNEL_URL, client=client)
        assert excinfo.value.code == "upstream.spice_kernel_fetch_failed"

    async def test_too_many_redirects_fails(self, cache: KernelCache) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            # Always bounce to another allowlisted path — a redirect loop.
            return httpx.Response(302, headers={"location": f"{_NAIF}/spk/hop.bsp?n={time.time()}"})

        async with _mock_client(handler) as client:
            with pytest.raises(UpstreamError) as excinfo:
                await cache.fetch(f"{_NAIF}/spk/loop.bsp", client=client)
        assert excinfo.value.code == "upstream.spice_kernel_fetch_failed"

    async def test_http_error_status_fails(self, cache: KernelCache) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(404)

        async with _mock_client(handler) as client:
            with pytest.raises(UpstreamError) as excinfo:
                await cache.fetch(_KERNEL_URL, client=client)
        assert excinfo.value.code == "upstream.spice_kernel_fetch_failed"

    async def test_cap_enforced_via_content_length(self, cache: KernelCache) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"x" * 100)  # sets Content-Length: 100

        async with _mock_client(handler) as client:
            with pytest.raises(UpstreamError) as excinfo:
                await cache.fetch(_KERNEL_URL, client=client, max_bytes=10)
        assert excinfo.value.code == "upstream.spice_kernel_too_large"
        # Nothing was written to the cache.
        assert not cache.path_for(_KERNEL_URL).exists()

    async def test_cap_enforced_mid_stream(self, cache: KernelCache) -> None:
        # A streaming body carries no Content-Length, so the cap is enforced as
        # bytes arrive rather than up front.
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_astream(b"x" * 10))

        async with _mock_client(handler) as client:
            with pytest.raises(UpstreamError) as excinfo:
                await cache.fetch(_KERNEL_URL, client=client, max_bytes=4)
        assert excinfo.value.code == "upstream.spice_kernel_too_large"
        assert not cache.path_for(_KERNEL_URL).exists()

    async def test_bad_url_rejected_before_network(self, cache: KernelCache) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("network must not be touched for an invalid URL")

        async with _mock_client(handler) as client:
            with pytest.raises(InvalidInputError) as excinfo:
                await cache.fetch("http://naif.jpl.nasa.gov/x.bsp", client=client)
        assert excinfo.value.code == "invalid_input.spice_kernel_url_scheme"

    async def test_disabled_cache_refuses_url_fetch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ASTRODYNAMICS_MCP_CACHE_DIR", "")
        disabled = KernelCache()
        with pytest.raises(UpstreamError) as excinfo:
            await disabled.fetch(_KERNEL_URL)
        assert excinfo.value.code == "upstream.spice_kernel_cache_disabled"


# ---------------------------------------------------------------------------
# env-configurable knobs + singleton
# ---------------------------------------------------------------------------


class TestConfigKnobs:
    def test_ttl_default_and_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ASTRODYNAMICS_MCP_KERNEL_CACHE_TTL", raising=False)
        assert _default_ttl_s() == 30 * 24 * 60 * 60.0
        monkeypatch.setenv("ASTRODYNAMICS_MCP_KERNEL_CACHE_TTL", "120")
        assert _default_ttl_s() == 120.0
        monkeypatch.setenv("ASTRODYNAMICS_MCP_KERNEL_CACHE_TTL", "not-a-number")
        assert _default_ttl_s() == 30 * 24 * 60 * 60.0  # falls back

    def test_max_bytes_default_and_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ASTRODYNAMICS_MCP_KERNEL_MAX_BYTES", raising=False)
        assert _default_max_bytes() == 512 * 1024 * 1024
        monkeypatch.setenv("ASTRODYNAMICS_MCP_KERNEL_MAX_BYTES", "2048")
        assert _default_max_bytes() == 2048
        monkeypatch.setenv("ASTRODYNAMICS_MCP_KERNEL_MAX_BYTES", "huge")
        assert _default_max_bytes() == 512 * 1024 * 1024  # falls back

    def test_default_kernel_cache_is_singleton(self) -> None:
        assert default_kernel_cache() is default_kernel_cache()
