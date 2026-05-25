"""XDG-aware on-disk cache for upstream-data responses.

One JSON file per ``(source, key)`` entry under a platformdirs-resolved cache
root (``~/.cache/astrodynamics-mcp/`` on Linux,
``%LOCALAPPDATA%\\astrodynamics-mcp\\Cache\\`` on Windows).

Multi-process safety
--------------------

The cache supports parallel ``astrodynamics-mcp stdio`` / ``... http``
processes reading and writing the same XDG dir at the same time. Two
invariants make that work:

1. **Atomic-rename writes.** Every write goes to a tempfile in the *same
   directory* as its destination, then ``os.replace`` swaps it into place.
   On POSIX ``rename(2)`` is atomic; on Windows ``os.replace`` is atomic
   via ``MoveFileExW(MOVEFILE_REPLACE_EXISTING)`` since Python 3.3. A
   reader will see either the prior file or the new one, never a torn
   half-write. The same-directory invariant is non-negotiable — a
   tempfile in ``/tmp`` and a destination in ``/home/user/.cache/...``
   would make the rename a cross-filesystem move, which is not atomic.

2. **No locking.** Two writers racing on the same key both create their
   own tempfile and both ``os.replace``; whichever rename lands second
   wins, neither write is torn, and no reader sees an intermediate state.
   We deliberately do not use a lockfile — a crashed writer leaving a
   stale lock would block future writes indefinitely; the rename pattern
   is self-healing instead.

Disabled mode
-------------

Setting ``ASTRODYNAMICS_MCP_CACHE_DIR=""`` (empty string) disables the
cache: ``get`` / ``get_stale`` always return ``None`` and ``put`` is a
no-op. Useful in tests and CI cells that want a pristine cache miss.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import platformdirs

_logger = logging.getLogger(__name__)

_CACHE_DIR_ENV_VAR = "ASTRODYNAMICS_MCP_CACHE_DIR"
_APP_NAME = "astrodynamics-mcp"


# Per-source default TTLs (seconds). Convenience for adapters — Cache.get
# always takes an explicit `ttl_s`, so the source-aware default lives at
# the call site. The registry exists so the value is one place, not
# scattered across data adapters.
DEFAULT_TTLS: Mapping[str, float] = {
    # CelesTrak honours If-Modified-Since for the bulk of its catalogues.
    # The 6h TTL is the soft-revalidate window; tools that need fresher
    # data can override.
    "celestrak": 6 * 60 * 60,
    # IERS Bulletin A refreshes Thursdays ~20:00 UTC. A 24h TTL guarantees
    # we re-fetch within one cycle.
    "iers": 24 * 60 * 60,
    # JPL Horizons planetary ephemerides drift on geological scales. 7d is
    # plenty conservative for porkchop / B-plane tools.
    "horizons": 7 * 24 * 60 * 60,
}


@dataclass(frozen=True)
class CacheHit:
    """The result of a successful cache lookup."""

    value: Any  # JSON-serialisable
    fetched_at: datetime  # tz-aware UTC


def _resolve_cache_dir(explicit: Path | None) -> Path | None:
    """Resolve the cache directory.

    Precedence (highest first): constructor arg → env var → platformdirs
    default. The env var being set to an *empty* string disables the
    cache (returns ``None``).
    """
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get(_CACHE_DIR_ENV_VAR)
    if env is not None:
        if env == "":
            return None
        return Path(env)
    return Path(platformdirs.user_cache_dir(_APP_NAME))


def _hash_key(key: str) -> str:
    """Hash an arbitrary key to a filesystem-safe, fixed-length filename."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class Cache:
    """On-disk cache for upstream-data responses.

    See module docstring for the multi-process-safety story.
    """

    _DEFAULT_TTLS: ClassVar[Mapping[str, float]] = DEFAULT_TTLS

    def __init__(self, directory: Path | None = None) -> None:
        self._dir = _resolve_cache_dir(directory)

    @property
    def enabled(self) -> bool:
        return self._dir is not None

    @property
    def directory(self) -> Path | None:
        """Resolved cache directory, or ``None`` when caching is disabled."""
        return self._dir

    def get(self, source: str, key: str, *, ttl_s: float) -> CacheHit | None:
        """Return the cached value if it exists and is younger than ``ttl_s``."""
        hit = self.get_stale(source, key)
        if hit is None:
            return None
        age = (datetime.now(tz=timezone.utc) - hit.fetched_at).total_seconds()
        if age > ttl_s:
            return None
        return hit

    def get_stale(self, source: str, key: str) -> CacheHit | None:
        """Return the cached value regardless of age. ``None`` if missing."""
        if self._dir is None:
            return None
        path = self._path(source, key)
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            fetched_at = datetime.fromisoformat(payload["fetched_at"])
            value = payload["value"]
        except (OSError, ValueError, KeyError) as exc:
            # Corrupted or truncated cache file. Treat as a miss; the
            # next `put` overwrites it cleanly.
            _logger.warning("ignoring corrupt cache entry at %s: %s", path, exc)
            return None
        return CacheHit(value=value, fetched_at=fetched_at)

    def put(self, source: str, key: str, value: Any) -> None:
        """Write ``value`` to the cache atomically. No-op when disabled."""
        if self._dir is None:
            return
        path = self._path(source, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            # Stored alongside the value as a debugging aid — without it the
            # sha256-named files in the cache dir are opaque to a curious
            # operator running `grep` against them.
            "key": key,
            "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
            "value": value,
        }
        # The tempfile MUST live in the destination directory so the final
        # rename stays on the same filesystem (where it is atomic). A
        # tempfile in `/tmp` plus a destination in `/home/user/.cache/...`
        # would silently degrade to a non-atomic cross-filesystem move.
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            # Clean up the tempfile on any failure (including KeyboardInterrupt).
            # The destination file, if it existed, is untouched because the
            # rename never happened.
            with suppress(FileNotFoundError):
                os.unlink(tmp_path)
            raise

    def _path(self, source: str, key: str) -> Path:
        if self._dir is None:
            raise RuntimeError("cache is disabled; check `enabled` before calling _path")
        return self._dir / source / f"{_hash_key(key)}.json"


_default_cache: Cache | None = None


def default_cache() -> Cache:
    """Return the module-level lazy-initialised cache singleton.

    Data adapters call this so they share a single cache instance per
    process. Tests should construct their own :class:`Cache` with a
    ``tmp_path`` directory and pass it into the code under test
    explicitly — do not rely on the singleton in tests.
    """
    global _default_cache
    if _default_cache is None:
        _default_cache = Cache()
    return _default_cache
