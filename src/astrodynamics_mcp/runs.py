"""Per-run artefact registry with best-effort cross-restart resolution.

Producer tools (``gmat_run_mission``, ``gmat_sweep``, ``gmat_execute_script``)
mint a UUID4 ``run_id`` per call, create their own ``tempfile.mkdtemp()``
workspace, run GMAT into it, and register the resulting name → path map
here. A follow-up ``gmat_read_run_artefact(run_id, name)`` call resolves
the path through this registry and reads the file fresh — bytes that
were too large to inline in the producer's response are still readable
in subsequent tool turns within the same MCP server process.

Lifetime
--------

In-memory dict capped at the last N runs (env var
``ASTRODYNAMICS_MCP_RUN_REGISTRY_LIMIT``, default 50). Eviction pops the
oldest entry by *creation* time, ``shutil.rmtree``s its ``output_dir``,
and deletes its disk-index file. There is no access-time LRU bump — the
contract is "last N runs", not "last N reads", matching the issue's
working terminology.

Cross-restart (best-effort)
---------------------------

A tiny JSON index file is written under ``<cache_root>/runs/<run_id>.json``
mirroring the atomic-rename / no-lock multi-process pattern from
:class:`~astrodynamics_mcp.cache.Cache`. On first access in a new
process the registry replays the index, dropping any entry whose
``output_dir`` no longer exists on disk. The output directories
themselves stay in ``tempfile.mkdtemp()`` (typically ``/tmp``) and are
*not* moved under the cache — the OS retains its usual reaping
semantics for the bytes; the index is just a directory lookup. If the
temp dir is gone after a restart, the read tool surfaces a typed
``invalid_input.artefact_evicted`` so the caller sees a clean "files
were reaped" signal instead of a mysterious miss.

Disabled mode
-------------

Setting ``ASTRODYNAMICS_MCP_CACHE_DIR=""`` (the same knob the upstream-
data cache honours) disables the disk index. The in-memory registry
still works exactly as before; nothing survives restart.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import platformdirs

_logger = logging.getLogger(__name__)

_APP_NAME = "astrodynamics-mcp"
_CACHE_DIR_ENV_VAR = "ASTRODYNAMICS_MCP_CACHE_DIR"
_LIMIT_ENV_VAR = "ASTRODYNAMICS_MCP_RUN_REGISTRY_LIMIT"
_DEFAULT_LIMIT = 50
_INDEX_SUBDIR = "runs"


@dataclass(frozen=True)
class RunArtefacts:
    """One run's registry snapshot: id, output dir, name→path map, created_at."""

    run_id: str
    output_dir: Path
    artefacts: Mapping[str, Path]
    created_at: datetime


def _resolve_index_dir(explicit: Path | None) -> Path | None:
    """Resolve the directory holding per-entry index JSONs.

    Precedence (highest first): explicit constructor arg →
    ``ASTRODYNAMICS_MCP_CACHE_DIR`` env var → platformdirs default. The
    env var being set to an *empty* string disables persistence
    (returns ``None``); the registry then runs memory-only.
    """
    if explicit is not None:
        return Path(explicit) / _INDEX_SUBDIR
    env = os.environ.get(_CACHE_DIR_ENV_VAR)
    if env is not None:
        if env == "":
            return None
        return Path(env) / _INDEX_SUBDIR
    return Path(platformdirs.user_cache_dir(_APP_NAME)) / _INDEX_SUBDIR


def _resolve_limit(explicit: int | None) -> int:
    """Resolve the maximum number of runs to retain.

    Precedence: explicit constructor arg → ``ASTRODYNAMICS_MCP_RUN_REGISTRY_LIMIT``
    env var → :data:`_DEFAULT_LIMIT` (50). Values < 1 are rejected up
    front so the runtime invariant ``len(entries) <= limit`` holds with
    eviction-by-popleft.
    """
    if explicit is not None:
        if explicit < 1:
            raise ValueError(f"run registry limit must be >= 1, got {explicit}")
        return explicit
    env = os.environ.get(_LIMIT_ENV_VAR)
    if env is not None and env != "":
        try:
            value = int(env)
        except ValueError as exc:
            raise ValueError(f"{_LIMIT_ENV_VAR}={env!r} is not an integer") from exc
        if value < 1:
            raise ValueError(f"{_LIMIT_ENV_VAR}={env!r} must be >= 1, got {value}")
        return value
    return _DEFAULT_LIMIT


class RunRegistry:
    """In-memory LRU registry with best-effort disk-backed index.

    See module docstring for the lifetime / cross-restart contract.
    """

    def __init__(
        self,
        *,
        directory: Path | None = None,
        limit: int | None = None,
    ) -> None:
        self._index_dir = _resolve_index_dir(directory)
        self._limit = _resolve_limit(limit)
        # OrderedDict ordered by creation time. Eviction pops the head
        # when count exceeds the limit. Reads do *not* bump position —
        # the contract is "retain the last N runs by creation", not
        # access-time LRU.
        self._entries: OrderedDict[str, RunArtefacts] = OrderedDict()
        self._loaded = False

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def directory(self) -> Path | None:
        """The index directory, or ``None`` when persistence is disabled."""
        return self._index_dir

    def mint(self) -> str:
        """Return a fresh UUID4 hex string for use as a ``run_id``."""
        return uuid.uuid4().hex

    def register(
        self,
        run_id: str,
        *,
        output_dir: Path,
        artefacts: Mapping[str, Path],
    ) -> RunArtefacts:
        """Record a new run, evicting the oldest if over the cap.

        Eviction removes both the disk-index entry and the
        ``output_dir`` itself (``shutil.rmtree`` with ``ignore_errors``
        — a concurrent process may have already reaped the dir).
        """
        self._ensure_loaded()
        entry = RunArtefacts(
            run_id=run_id,
            output_dir=Path(output_dir),
            artefacts={str(k): Path(v) for k, v in artefacts.items()},
            created_at=datetime.now(tz=timezone.utc),
        )
        # Vanishingly unlikely with UUID4, but a duplicate id would
        # otherwise leak the prior entry's bytes — drop them first.
        if run_id in self._entries:
            old = self._entries.pop(run_id)
            self._drop_payload(old)
        self._entries[run_id] = entry
        self._write_index(entry)
        while len(self._entries) > self._limit:
            _, oldest = self._entries.popitem(last=False)
            self._drop_payload(oldest)
        return entry

    def get(self, run_id: str) -> RunArtefacts | None:
        """Return the snapshot for ``run_id`` or ``None`` when unknown."""
        self._ensure_loaded()
        return self._entries.get(run_id)

    def drop(self, run_id: str) -> bool:
        """Remove ``run_id`` eagerly. Returns whether anything was removed.

        Symmetric with cap-driven eviction: drops the in-memory entry,
        unlinks its disk-index JSON, and ``shutil.rmtree``s the run's
        ``output_dir`` (``ignore_errors=True`` so a partially-reaped dir
        is fine). Called from the read tool when an external reaper
        (systemd-tmpfiles, manual cleanup) made the registered path
        unreadable mid-process — keeping the dead entry around would
        otherwise occupy a slot in the LRU cap until the next process
        restart.
        """
        self._ensure_loaded()
        entry = self._entries.pop(run_id, None)
        if entry is None:
            return False
        self._drop_payload(entry)
        return True

    def known_run_ids(self) -> list[str]:
        """Return all known ``run_id``s in creation order (oldest first)."""
        self._ensure_loaded()
        return list(self._entries.keys())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lazy-replay the disk index on first access. No-op when disabled."""
        if self._loaded:
            return
        self._loaded = True
        if self._index_dir is None or not self._index_dir.is_dir():
            return
        replayed: list[RunArtefacts] = []
        for path in sorted(self._index_dir.glob("*.json")):
            entry = self._read_index(path)
            if entry is None:
                continue
            if not entry.output_dir.is_dir():
                # OS-level reaping (systemd-tmpfiles, manual cleanup, a
                # peer process's eviction) leaves orphan JSON behind. Drop
                # quietly — the read tool will surface a clean
                # ``unknown_run_id`` to any caller chasing that id.
                with suppress(FileNotFoundError):
                    path.unlink()
                continue
            replayed.append(entry)
        replayed.sort(key=lambda e: e.created_at)
        for entry in replayed:
            self._entries[entry.run_id] = entry
        # If the limit shrank between processes, evict the surplus.
        while len(self._entries) > self._limit:
            _, oldest = self._entries.popitem(last=False)
            self._drop_payload(oldest)

    def _write_index(self, entry: RunArtefacts) -> None:
        """Atomically write ``entry``'s JSON index file. No-op when disabled.

        Same atomic-rename invariant as :class:`~astrodynamics_mcp.cache.Cache`:
        tempfile lives in the destination directory so the final
        ``os.replace`` stays on one filesystem and is therefore atomic.
        """
        if self._index_dir is None:
            return
        self._index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": entry.run_id,
            "output_dir": str(entry.output_dir),
            "artefacts": {k: str(v) for k, v in entry.artefacts.items()},
            "created_at": entry.created_at.isoformat(),
        }
        path = self._index_dir / f"{entry.run_id}.json"
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(self._index_dir),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            with suppress(FileNotFoundError):
                os.unlink(tmp_path)
            raise

    def _read_index(self, path: Path) -> RunArtefacts | None:
        """Parse one index file, treating any corruption as a miss."""
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            return RunArtefacts(
                run_id=payload["run_id"],
                output_dir=Path(payload["output_dir"]),
                artefacts={k: Path(v) for k, v in payload["artefacts"].items()},
                created_at=datetime.fromisoformat(payload["created_at"]),
            )
        except (OSError, ValueError, KeyError) as exc:
            _logger.warning("ignoring corrupt run-index entry at %s: %s", path, exc)
            return None

    def _drop_payload(self, entry: RunArtefacts) -> None:
        """Remove ``entry``'s index file and ``rmtree`` its output dir."""
        if self._index_dir is not None:
            with suppress(FileNotFoundError):
                (self._index_dir / f"{entry.run_id}.json").unlink()
        shutil.rmtree(entry.output_dir, ignore_errors=True)


_default_registry: RunRegistry | None = None


def default_registry() -> RunRegistry:
    """Return the module-level lazy-initialised registry singleton.

    Producer tools call this so they share a single registry per
    process. Tests should construct their own :class:`RunRegistry` with
    a ``tmp_path`` directory and inject it via
    ``monkeypatch.setattr(astrodynamics_mcp.runs, "_default_registry", ...)``
    rather than relying on the singleton.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = RunRegistry()
    return _default_registry
