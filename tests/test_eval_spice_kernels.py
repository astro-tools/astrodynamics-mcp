"""Tests for the SPICE eval-prompt kernel set and its cache-presence gate.

The SPICE prompts pre-seed a fixed set of NAIF generic kernels into the shared
on-disk cache so the spawned server's ``spice_load_kernel(URL)`` is served
``from_cache``. :mod:`eval._spice_kernels` is the single source of truth for
that set plus the stat-only presence check the ``requires_spice`` skip-gate
calls. These tests pin the URL contract, the all-present/any-missing logic, and
that provisioning only fetches what is absent — without touching the network.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import pytest
from astrodynamics_mcp.spice_kernels import NAIF_KERNEL_HOSTS, KernelCache
from eval._spice_kernels import CANONICAL_KERNEL_URLS, ensure_cached, kernels_cached


def _seed(cache: KernelCache, url: str) -> None:
    """Write a stand-in blob at the cache path for *url* so is_cached() is true."""
    assert cache.directory is not None
    cache.directory.mkdir(parents=True, exist_ok=True)
    cache.path_for(url).write_bytes(b"fake kernel bytes")


def test_canonical_urls_are_allowlisted_https_naif() -> None:
    assert CANONICAL_KERNEL_URLS, "the canonical kernel set must not be empty"
    for url in CANONICAL_KERNEL_URLS:
        parsed = urlparse(url)
        assert parsed.scheme == "https", f"{url} must be https"
        assert parsed.hostname in NAIF_KERNEL_HOSTS, f"{url} host not on the NAIF allowlist"


def test_kernels_cached_false_when_directory_empty(tmp_path: Path) -> None:
    cache = KernelCache(directory=tmp_path)
    assert kernels_cached(cache) is False


def test_kernels_cached_false_when_one_missing(tmp_path: Path) -> None:
    cache = KernelCache(directory=tmp_path)
    for url in CANONICAL_KERNEL_URLS[:-1]:
        _seed(cache, url)
    # All but the last seeded — the gate must still report not-cached.
    assert kernels_cached(cache) is False


def test_kernels_cached_true_when_all_present(tmp_path: Path) -> None:
    cache = KernelCache(directory=tmp_path)
    for url in CANONICAL_KERNEL_URLS:
        _seed(cache, url)
    assert kernels_cached(cache) is True


def test_kernels_cached_false_when_cache_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    # An empty ASTRODYNAMICS_MCP_CACHE_DIR disables the cache entirely, so the
    # SPICE prompts must skip rather than fail.
    monkeypatch.setenv("ASTRODYNAMICS_MCP_CACHE_DIR", "")
    cache = KernelCache()
    assert cache.enabled is False
    assert kernels_cached(cache) is False


async def test_ensure_cached_only_fetches_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = KernelCache(directory=tmp_path)
    # Pre-seed all but the last so only one URL needs fetching.
    already = CANONICAL_KERNEL_URLS[:-1]
    missing = CANONICAL_KERNEL_URLS[-1]
    for url in already:
        _seed(cache, url)

    fetched_urls: list[str] = []

    async def fake_fetch(url: str) -> Path:
        fetched_urls.append(url)
        _seed(cache, url)
        return cache.path_for(url)

    monkeypatch.setattr(cache, "fetch", fake_fetch)
    fetched = await ensure_cached(cache)

    assert fetched == [missing]
    assert fetched_urls == [missing]
    assert kernels_cached(cache) is True


async def test_ensure_cached_idempotent_when_all_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = KernelCache(directory=tmp_path)
    for url in CANONICAL_KERNEL_URLS:
        _seed(cache, url)

    async def fail_fetch(url: str) -> Path:  # pragma: no cover - must not run
        raise AssertionError(f"unexpected fetch of already-cached {url!r}")

    monkeypatch.setattr(cache, "fetch", fail_fetch)
    assert await ensure_cached(cache) == []
