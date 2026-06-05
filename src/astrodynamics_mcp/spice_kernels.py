"""NAIF kernel cache and furnish-from-URL policy for the SPICE tool surface.

A SPICE kernel can be furnished from a local filesystem path or fetched from a
URL. An arbitrary-URL fetch is an SSRF / path-traversal surface, so URL loads
are constrained exactly as the SPICE integration design note records:

- the scheme must be ``https``;
- the host must be on the NAIF allowlist (:data:`NAIF_KERNEL_HOSTS`), and any
  redirect that leaves the allowlist is refused rather than followed;
- the download is routed through an on-disk cache, keyed by a hash of the URL
  and written with the same atomic-rename discipline the response cache uses,
  so the URL never names a destination path directly (no path-traversal vector)
  and a repeat load is served from disk;
- oversized downloads are capped.

The cache lives **under the same XDG cache root** the response cache
(:mod:`astrodynamics_mcp.cache`) manages — it reuses
:func:`astrodynamics_mcp.cache.resolve_cache_dir` for root resolution — but
stores raw kernel blobs under a ``kernels/`` subdirectory rather than the
JSON-per-entry layout the response cache uses.

A local path is furnished as-is: the caller already has filesystem access, so
no allowlist applies there. The tool that calls :meth:`KernelCache.fetch` for
a URL load lands in follow-up work; this module is the wiring it builds on.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlparse

import httpx

from astrodynamics_mcp import __version__
from astrodynamics_mcp.cache import resolve_cache_dir
from astrodynamics_mcp.errors import InvalidInputError, UpstreamError

_logger = logging.getLogger(__name__)

# NAIF is the canonical, trusted source of generic SPICE kernels. Only hosts in
# this set may be fetched from; everything else (and any redirect that leaves
# it) is refused before a request is made.
NAIF_KERNEL_HOSTS: frozenset[str] = frozenset({"naif.jpl.nasa.gov"})

# Subdirectory under the XDG cache root that holds cached kernel blobs.
_KERNELS_SUBDIR = "kernels"

# Env-configurable knobs. Read at call time so an operator (or a test) can
# override them per-process without re-importing the module.
_TTL_ENV_VAR = "ASTRODYNAMICS_MCP_KERNEL_CACHE_TTL"
_MAX_BYTES_ENV_VAR = "ASTRODYNAMICS_MCP_KERNEL_MAX_BYTES"

# Cached kernels at a given NAIF URL are effectively immutable (NAIF versions
# filenames), so a long default TTL mostly governs how often we re-validate a
# long-lived deployment's cache. 30 days is conservative.
_DEFAULT_TTL_S = 30 * 24 * 60 * 60.0
# A safety rail against a runaway / hostile download, not a per-kernel size
# limit; generous enough for a planetary SPK (e.g. de440 ~ 114 MB) yet far
# below a pathological multi-gigabyte fetch. Operators that need a larger
# kernel raise the cap explicitly.
_DEFAULT_MAX_BYTES = 512 * 1024 * 1024

# Bound the redirect chain we will follow; each hop is re-validated against the
# allowlist before it is fetched.
_MAX_REDIRECTS = 5
_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})

_HTTP_TIMEOUT = 60.0
_USER_AGENT = f"astrodynamics-mcp/{__version__} (+https://github.com/astro-tools/astrodynamics-mcp)"
_HTTP_HEADERS: dict[str, str] = {"User-Agent": _USER_AGENT}

# Only a short, alphanumeric URL suffix is preserved on the cached blob name —
# purely so an operator grep-ing the cache dir can tell a `.bsp` from a `.tls`.
# The blob is keyed by the URL hash regardless; the suffix carries no meaning to
# CSPICE, which identifies a kernel from its file contents, not its extension.
_SAFE_SUFFIX_RE = re.compile(r"^\.[A-Za-z0-9]{1,8}$")


def validate_kernel_url(url: str) -> str:
    """Return *url* unchanged if it is a fetchable NAIF kernel URL, else raise.

    Enforces the two static furnish-from-URL constraints: an ``https`` scheme
    and a host on :data:`NAIF_KERNEL_HOSTS`. e.g. ``validate_kernel_url(
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp")``
    returns the URL; an ``http://`` URL or any non-NAIF host raises
    :class:`~astrodynamics_mcp.errors.InvalidInputError`. Called once per
    redirect hop so a redirect leaving the allowlist is refused, not followed.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise InvalidInputError(
            f"SPICE kernel URL must use https, got scheme {parsed.scheme!r} in {url!r}",
            code="invalid_input.spice_kernel_url_scheme",
            data={"url": url, "scheme": parsed.scheme},
        )
    host = (parsed.hostname or "").lower()
    if host not in NAIF_KERNEL_HOSTS:
        raise InvalidInputError(
            f"SPICE kernel URL host {host!r} is not on the NAIF allowlist "
            f"{sorted(NAIF_KERNEL_HOSTS)}; refusing to fetch {url!r}",
            code="invalid_input.spice_kernel_url_host",
            data={"url": url, "host": host, "allowlist": sorted(NAIF_KERNEL_HOSTS)},
        )
    return url


def _default_ttl_s() -> float:
    raw = os.environ.get(_TTL_ENV_VAR)
    if not raw:
        return _DEFAULT_TTL_S
    try:
        return float(raw)
    except ValueError:
        _logger.warning("ignoring non-numeric %s=%r; using default", _TTL_ENV_VAR, raw)
        return _DEFAULT_TTL_S


def _default_max_bytes() -> int:
    raw = os.environ.get(_MAX_BYTES_ENV_VAR)
    if not raw:
        return _DEFAULT_MAX_BYTES
    try:
        return int(raw)
    except ValueError:
        _logger.warning("ignoring non-integer %s=%r; using default", _MAX_BYTES_ENV_VAR, raw)
        return _DEFAULT_MAX_BYTES


def _hash_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _blob_name(url: str) -> str:
    """Filesystem-safe cache filename for *url* — ``<sha256>[<suffix>]``."""
    suffix = Path(urlparse(url).path).suffix
    if not _SAFE_SUFFIX_RE.match(suffix):
        suffix = ""
    return f"{_hash_url(url)}{suffix.lower()}"


class KernelCache:
    """On-disk cache of NAIF kernel blobs under the XDG ``kernels/`` subdir.

    Stores one file per ``url``, keyed by a hash of the URL and written with
    the response cache's atomic-rename discipline (tempfile in the destination
    directory → ``os.fsync`` → ``os.replace``). Disabled when the shared cache
    is disabled (``ASTRODYNAMICS_MCP_CACHE_DIR=""``).
    """

    def __init__(self, directory: Path | None = None) -> None:
        root = resolve_cache_dir(directory)
        self._dir: Path | None = (root / _KERNELS_SUBDIR) if root is not None else None

    @property
    def enabled(self) -> bool:
        return self._dir is not None

    @property
    def directory(self) -> Path | None:
        """Resolved kernel-cache directory, or ``None`` when caching is disabled."""
        return self._dir

    def path_for(self, url: str) -> Path:
        if self._dir is None:
            raise RuntimeError("kernel cache is disabled; check `enabled` before calling path_for")
        return self._dir / _blob_name(url)

    def get(self, url: str, *, ttl_s: float) -> Path | None:
        """Return the cached kernel path if present and younger than ``ttl_s``."""
        if self._dir is None:
            return None
        path = self.path_for(url)
        try:
            age = time.time() - path.stat().st_mtime
        except OSError:
            return None
        if age > ttl_s:
            return None
        return path

    async def fetch(
        self,
        url: str,
        *,
        client: httpx.AsyncClient | None = None,
        ttl_s: float | None = None,
        max_bytes: int | None = None,
    ) -> Path:
        """Furnish-from-URL: validate, serve from cache, else download + store.

        Returns the local path of the cached kernel. A fresh cache hit skips the
        network entirely. On a miss the kernel is streamed from NAIF — each
        redirect hop re-validated against the allowlist, the body capped at
        ``max_bytes`` — and written atomically into the cache. The optional
        ``client`` lets a caller (or a test) inject a configured
        :class:`httpx.AsyncClient`; production callers pass none.
        """
        validate_kernel_url(url)
        if self._dir is None:
            raise UpstreamError(
                "SPICE kernel cache is disabled (ASTRODYNAMICS_MCP_CACHE_DIR=''); "
                "furnish-from-URL routes through the cache, so furnish a local "
                "kernel path instead when the cache is off",
                code="upstream.spice_kernel_cache_disabled",
            )
        ttl = _default_ttl_s() if ttl_s is None else ttl_s
        cap = _default_max_bytes() if max_bytes is None else max_bytes

        hit = self.get(url, ttl_s=ttl)
        if hit is not None:
            return hit

        if client is None:
            async with httpx.AsyncClient(
                timeout=_HTTP_TIMEOUT, headers=_HTTP_HEADERS, follow_redirects=False
            ) as owned_client:
                return await self._download(owned_client, url, cap=cap)
        return await self._download(client, url, cap=cap)

    async def _download(self, client: httpx.AsyncClient, original_url: str, *, cap: int) -> Path:
        """Stream *original_url* into the cache, following allowlisted redirects."""
        current = original_url
        for _ in range(_MAX_REDIRECTS + 1):
            # Re-validate every hop: an off-allowlist redirect target is refused
            # here, before any request is issued against it.
            validate_kernel_url(current)
            try:
                async with client.stream("GET", current) as response:
                    if response.status_code in _REDIRECT_CODES:
                        location = response.headers.get("location")
                        if not location:
                            raise UpstreamError(
                                f"redirect from {current!r} carried no Location header",
                                code="upstream.spice_kernel_fetch_failed",
                                data={"url": original_url},
                            )
                        current = str(httpx.URL(current).join(location))
                        continue
                    response.raise_for_status()
                    return await self._stream_to_cache(original_url, response, cap=cap)
            except httpx.HTTPError as exc:
                raise UpstreamError(
                    f"failed to fetch SPICE kernel from {current!r}: {exc}",
                    code="upstream.spice_kernel_fetch_failed",
                    original_exception=exc,
                    data={"url": original_url},
                ) from exc
        raise UpstreamError(
            f"too many redirects fetching SPICE kernel from {original_url!r} "
            f"(exceeded {_MAX_REDIRECTS})",
            code="upstream.spice_kernel_fetch_failed",
            data={"url": original_url, "max_redirects": _MAX_REDIRECTS},
        )

    async def _stream_to_cache(self, url: str, response: httpx.Response, *, cap: int) -> Path:
        """Write the streamed response body into the cache atomically, capped."""
        assert self._dir is not None  # fetch() guards the disabled case
        self._raise_if_content_length_exceeds(url, response, cap=cap)
        self._dir.mkdir(parents=True, exist_ok=True)
        dest = self.path_for(url)
        # The tempfile MUST live in the destination dir so the final rename
        # stays on the same filesystem (where it is atomic).
        fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=str(self._dir))
        total = 0
        try:
            with os.fdopen(fd, "wb") as fh:
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > cap:
                        raise UpstreamError(
                            f"SPICE kernel at {url!r} exceeds the {cap}-byte cap "
                            f"({_MAX_BYTES_ENV_VAR}); download aborted",
                            code="upstream.spice_kernel_too_large",
                            data={"url": url, "max_bytes": cap},
                        )
                    fh.write(chunk)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, dest)
        except BaseException:
            with suppress(FileNotFoundError):
                os.unlink(tmp_name)
            raise
        return dest

    @staticmethod
    def _raise_if_content_length_exceeds(url: str, response: httpx.Response, *, cap: int) -> None:
        """Reject before downloading when the server's Content-Length blows the cap."""
        raw = response.headers.get("content-length")
        if raw is None:
            return
        try:
            declared = int(raw)
        except ValueError:
            return
        if declared > cap:
            raise UpstreamError(
                f"SPICE kernel at {url!r} declares {declared} bytes, over the "
                f"{cap}-byte cap ({_MAX_BYTES_ENV_VAR})",
                code="upstream.spice_kernel_too_large",
                data={"url": url, "max_bytes": cap, "content_length": declared},
            )


_default_kernel_cache: KernelCache | None = None


def default_kernel_cache() -> KernelCache:
    """Return the module-level lazy-initialised :class:`KernelCache` singleton.

    The SPICE tools share one instance per process so a kernel one call fetched
    is on disk for the next. Tests construct their own :class:`KernelCache`
    with a ``tmp_path`` directory rather than relying on this singleton.
    """
    global _default_kernel_cache
    if _default_kernel_cache is None:
        _default_kernel_cache = KernelCache()
    return _default_kernel_cache
