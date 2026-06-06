"""Canonical generic-kernel set the SPICE eval prompts depend on.

The SPICE prompts under ``eval/prompts/spice_*.yaml`` instruct the model to
furnish a fixed set of NAIF generic kernels by URL. For those loads to be fast
and offline-safe in the eval, the kernels are pre-seeded into the **same**
on-disk kernel cache the spawned ``astrodynamics-mcp stdio`` server reads from
(both processes resolve it through
:func:`astrodynamics_mcp.cache.resolve_cache_dir`, and the server subprocess
inherits the runner's environment) — so a ``spice_load_kernel(URL)`` is served
``from_cache`` with no network hop.

This module is the single source of truth for that kernel set:

- :data:`CANONICAL_KERNEL_URLS` — the NAIF URLs the prompts load, also the
  exact strings the prompt text hands the model so the cache key matches.
- :func:`kernels_cached` — a stat-only presence check used by the
  ``requires_spice`` skip-gate (see :func:`eval._prompts._spice_available`):
  when any kernel is missing the SPICE prompts skip rather than fail.
- :func:`ensure_cached` — the provisioning helper that downloads the set into
  the cache. Run it once before a ``[spice]``-extra eval (or from a future CI
  provisioning step); it is deliberately **not** invoked on the eval's hot
  path, so a normal run never blocks on a 30 MB download.

The kernels are the smallest generic set that covers the prompts' needs: a
leap-second kernel (time), a planetary-constants PCK (radii / pole / PM), a
gravity PCK (GM), and the compact ``de440s`` planetary SPK (states, ~32 MB,
1849-2150). All four live on the NAIF allowlist
(:data:`astrodynamics_mcp.spice_kernels.NAIF_KERNEL_HOSTS`).
"""

from __future__ import annotations

from astrodynamics_mcp.spice_kernels import KernelCache

_NAIF_GENERIC = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels"

# Leap-second kernel — every UTC<->ET conversion and every str2et-backed query
# needs it.
LSK_URL = f"{_NAIF_GENERIC}/lsk/naif0012.tls"
# Planetary-constants PCK — triaxial radii and the pole / prime-meridian
# orientation coefficients, plus the IAU body-fixed frames (IAU_MARS, …).
PCK_URL = f"{_NAIF_GENERIC}/pck/pck00011.tpc"
# Gravity PCK — the GM constants pck00011 does not carry.
GM_PCK_URL = f"{_NAIF_GENERIC}/pck/gm_de440.tpc"
# Compact planetary SPK — body states over 1849-2150 at ~32 MB, vs. de440's
# ~114 MB. Enough for every state / agreement prompt in the suite.
SPK_URL = f"{_NAIF_GENERIC}/spk/planets/de440s.bsp"

CANONICAL_KERNEL_URLS: tuple[str, ...] = (LSK_URL, PCK_URL, GM_PCK_URL, SPK_URL)


def kernels_cached(cache: KernelCache | None = None) -> bool:
    """True when every canonical kernel is present and fresh in the cache.

    Stat-only — never touches the network — so it is safe to call from the
    ``requires_spice`` skip-gate on every prompt-load. A disabled cache
    (``ASTRODYNAMICS_MCP_CACHE_DIR=""``) makes :meth:`KernelCache.is_cached`
    return ``False`` for every URL, so the SPICE prompts skip there too.
    """
    resolved = cache if cache is not None else KernelCache()
    return all(resolved.is_cached(url) for url in CANONICAL_KERNEL_URLS)


async def ensure_cached(cache: KernelCache | None = None) -> list[str]:
    """Download any missing canonical kernels into the cache; return their URLs.

    The provisioning entrypoint: run once before an eval that installs the
    ``[spice]`` extra so the SPICE prompts run rather than skip. Idempotent — a
    kernel already cached and fresh is left untouched and not reported. Routes
    every fetch through :meth:`KernelCache.fetch`, which enforces the NAIF
    allowlist and the size cap.
    """
    resolved = cache if cache is not None else KernelCache()
    fetched: list[str] = []
    for url in CANONICAL_KERNEL_URLS:
        if not resolved.is_cached(url):
            await resolved.fetch(url)
            fetched.append(url)
    return fetched
