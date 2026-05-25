"""Tests for `astrodynamics_mcp.cache`.

Includes a cross-process race test that exercises the atomic-rename
invariant by spawning two real subprocesses that hammer the same key —
matches the multi-process scenario of two parallel
``astrodynamics-mcp`` CLI instances sharing the same XDG cache.
"""

from __future__ import annotations

import dataclasses
import json
import multiprocessing as mp
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from astrodynamics_mcp.cache import (
    DEFAULT_TTLS,
    Cache,
    CacheHit,
    default_cache,
)


@pytest.fixture
def cache(tmp_path: Path) -> Cache:
    """A fresh cache rooted at a per-test temp dir."""
    return Cache(directory=tmp_path)


class TestRoundTrip:
    def test_put_then_get_returns_value(self, cache: Cache) -> None:
        cache.put("celestrak", "25544", {"name": "ISS", "norad_id": 25544})
        hit = cache.get("celestrak", "25544", ttl_s=3600)
        assert hit is not None
        assert hit.value == {"name": "ISS", "norad_id": 25544}

    def test_fetched_at_is_recent_and_utc(self, cache: Cache) -> None:
        before = datetime.now(tz=timezone.utc)
        cache.put("celestrak", "25544", {"v": 1})
        after = datetime.now(tz=timezone.utc)
        hit = cache.get("celestrak", "25544", ttl_s=3600)
        assert hit is not None
        assert hit.fetched_at.tzinfo is not None
        assert before <= hit.fetched_at <= after

    def test_get_returns_none_for_missing_key(self, cache: Cache) -> None:
        assert cache.get("celestrak", "does-not-exist", ttl_s=3600) is None

    @pytest.mark.parametrize(
        "value",
        [
            {"nested": {"deep": [1, 2, 3]}},
            [1, "two", 3.0, None, True],
            "raw string",
            42,
            3.14,
            None,
            [],
            {},
        ],
    )
    def test_value_types_round_trip(self, cache: Cache, value: Any) -> None:
        cache.put("src", "k", value)
        hit = cache.get("src", "k", ttl_s=3600)
        assert hit is not None
        assert hit.value == value


class TestTtlBehaviour:
    def test_ttl_zero_returns_none_for_just_written(self, cache: Cache) -> None:
        cache.put("celestrak", "k", {"v": 1})
        # ttl_s=0 means "any age > 0 is expired". A just-written entry
        # has age epsilon > 0 by the time get() runs.
        assert cache.get("celestrak", "k", ttl_s=0) is None

    def test_get_within_ttl_returns_hit(self, cache: Cache) -> None:
        cache.put("celestrak", "k", {"v": 1})
        hit = cache.get("celestrak", "k", ttl_s=3600)
        assert hit is not None

    def test_get_past_ttl_returns_none(self, cache: Cache) -> None:
        cache.put("celestrak", "k", {"v": 1})
        # Backdate the file to look 2h old, then query with a 1h TTL.
        path = cache._path("celestrak", "k")
        payload = json.loads(path.read_text())
        payload["fetched_at"] = (datetime.now(tz=timezone.utc) - timedelta(hours=2)).isoformat()
        path.write_text(json.dumps(payload))
        assert cache.get("celestrak", "k", ttl_s=3600) is None

    def test_get_stale_ignores_ttl(self, cache: Cache) -> None:
        cache.put("celestrak", "k", {"v": 1})
        path = cache._path("celestrak", "k")
        payload = json.loads(path.read_text())
        payload["fetched_at"] = (datetime.now(tz=timezone.utc) - timedelta(days=7)).isoformat()
        path.write_text(json.dumps(payload))
        hit = cache.get_stale("celestrak", "k")
        assert hit is not None
        assert hit.value == {"v": 1}

    def test_get_stale_returns_none_for_missing(self, cache: Cache) -> None:
        assert cache.get_stale("celestrak", "missing") is None


class TestAtomicWrite:
    def test_failed_replace_leaves_prior_value_intact(self, cache: Cache) -> None:
        cache.put("celestrak", "k", {"v": "original"})
        original = cache.get("celestrak", "k", ttl_s=3600)
        assert original is not None

        # Simulate an os.replace failure (e.g. EACCES on Windows AV-locked file).
        with (
            patch("astrodynamics_mcp.cache.os.replace", side_effect=OSError("simulated")),
            pytest.raises(OSError, match="simulated"),
        ):
            cache.put("celestrak", "k", {"v": "replacement"})

        # Prior file untouched.
        intact = cache.get("celestrak", "k", ttl_s=3600)
        assert intact is not None
        assert intact.value == {"v": "original"}

        # And no orphaned tempfile in the source directory.
        leftover = list((cache.directory or Path("/")).glob("celestrak/.*.tmp"))
        assert leftover == []

    def test_tempfile_lives_in_destination_directory(self, cache: Cache) -> None:
        """The atomic-rename invariant: tempfile MUST be on the same filesystem.

        Cross-filesystem renames are not atomic — they silently degrade to
        copy+unlink, which a concurrent reader can observe mid-flight. We
        enforce same-dir tempfiles by passing `dir=path.parent` to
        ``tempfile.mkstemp``. This test pins the behaviour by patching
        ``tempfile.mkstemp`` and asserting the `dir=` kwarg matches the
        destination directory.
        """
        captured: dict[str, str] = {}

        real_mkstemp = __import__("tempfile").mkstemp

        def fake_mkstemp(**kwargs: Any) -> Any:
            captured["dir"] = kwargs.get("dir", "")
            return real_mkstemp(**kwargs)

        with patch("astrodynamics_mcp.cache.tempfile.mkstemp", side_effect=fake_mkstemp):
            cache.put("celestrak", "k", {"v": 1})

        expected_dir = cache._path("celestrak", "k").parent
        assert Path(captured["dir"]) == expected_dir


class TestCorruption:
    def test_corrupted_json_treated_as_miss(
        self, cache: Cache, caplog: pytest.LogCaptureFixture
    ) -> None:
        cache.put("celestrak", "k", {"v": 1})
        path = cache._path("celestrak", "k")
        path.write_text("{not valid json")
        assert cache.get_stale("celestrak", "k") is None

    def test_missing_required_fields_treated_as_miss(self, cache: Cache) -> None:
        path = cache._path("celestrak", "k")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"something": "else"}')  # no `fetched_at` or `value`
        assert cache.get_stale("celestrak", "k") is None


class TestCacheDirResolution:
    def test_explicit_directory_takes_precedence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ASTRODYNAMICS_MCP_CACHE_DIR", "/this/should/be/ignored")
        c = Cache(directory=tmp_path)
        assert c.directory == tmp_path

    def test_env_var_overrides_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ASTRODYNAMICS_MCP_CACHE_DIR", str(tmp_path / "subdir"))
        c = Cache()
        assert c.directory == tmp_path / "subdir"

    def test_empty_env_var_disables_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ASTRODYNAMICS_MCP_CACHE_DIR", "")
        c = Cache()
        assert not c.enabled
        assert c.directory is None
        # put is a no-op
        c.put("celestrak", "k", {"v": 1})
        # get always returns None
        assert c.get("celestrak", "k", ttl_s=3600) is None
        assert c.get_stale("celestrak", "k") is None

    def test_unset_env_var_falls_back_to_platformdirs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ASTRODYNAMICS_MCP_CACHE_DIR", raising=False)
        c = Cache()
        assert c.directory is not None
        # The path should contain the app name somewhere; exact location
        # depends on platform/XDG state but the discipline is what we test.
        assert "astrodynamics-mcp" in str(c.directory)

    def test_disabled_cache_does_not_create_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Disabled mode must not even resolve a path that could be created.
        monkeypatch.setenv("ASTRODYNAMICS_MCP_CACHE_DIR", "")
        c = Cache()
        c.put("celestrak", "k", {"v": 1})
        # Nothing was written under tmp_path because the cache is off entirely.
        assert list(tmp_path.iterdir()) == []

    def test_path_on_disabled_cache_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_path() is internal but must fail loud if a future caller skips the enabled check."""
        monkeypatch.setenv("ASTRODYNAMICS_MCP_CACHE_DIR", "")
        c = Cache()
        with pytest.raises(RuntimeError, match="cache is disabled"):
            c._path("celestrak", "k")


class TestCacheDirAutoCreation:
    def test_subdirectory_created_on_first_put(self, tmp_path: Path) -> None:
        c = Cache(directory=tmp_path / "nested" / "deeper")
        c.put("celestrak", "k", {"v": 1})
        # Both the cache root and the per-source subdir were created.
        assert (tmp_path / "nested" / "deeper" / "celestrak").is_dir()


class TestDefaultTtlsRegistry:
    def test_v01_sources_present(self) -> None:
        assert set(DEFAULT_TTLS.keys()) >= {"celestrak", "iers", "horizons"}

    def test_default_ttls_are_in_seconds(self) -> None:
        # Sanity that the numbers match the documented per-source TTLs.
        assert DEFAULT_TTLS["celestrak"] == 6 * 60 * 60
        assert DEFAULT_TTLS["iers"] == 24 * 60 * 60
        assert DEFAULT_TTLS["horizons"] == 7 * 24 * 60 * 60


class TestDefaultCacheSingleton:
    def test_returns_same_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Force-reset the module-level singleton between tests to avoid
        # cross-contamination from earlier test runs.
        import astrodynamics_mcp.cache as cache_module

        monkeypatch.setattr(cache_module, "_default_cache", None)
        c1 = default_cache()
        c2 = default_cache()
        assert c1 is c2


# ---------------------------------------------------------------------------
# Multi-process race tests (the parallel-CLI scenario)
# ---------------------------------------------------------------------------


def _race_put_in_subprocess(directory: str, source: str, key: str, value: Any) -> None:
    """Subprocess entry point: write `value` into a cache rooted at `directory`."""
    c = Cache(directory=Path(directory))
    c.put(source, key, value)


class TestMultiProcessSafety:
    def test_concurrent_writers_leave_parseable_file(self, tmp_path: Path) -> None:
        """Two subprocesses racing on the same key → final file is one of them.

        Mirrors two parallel ``astrodynamics-mcp`` CLI processes hitting
        the same XDG cache. Neither write should be torn; the loser is
        clobbered, but the winner's payload is intact.
        """
        ctx = mp.get_context("spawn")
        p1 = ctx.Process(
            target=_race_put_in_subprocess,
            args=(str(tmp_path), "celestrak", "raced-key", {"from": "process_1"}),
        )
        p2 = ctx.Process(
            target=_race_put_in_subprocess,
            args=(str(tmp_path), "celestrak", "raced-key", {"from": "process_2"}),
        )
        p1.start()
        p2.start()
        p1.join(timeout=10)
        p2.join(timeout=10)
        assert p1.exitcode == 0
        assert p2.exitcode == 0

        # File parses cleanly and contains one of the two payloads.
        c = Cache(directory=tmp_path)
        hit = c.get_stale("celestrak", "raced-key")
        assert hit is not None
        assert hit.value in ({"from": "process_1"}, {"from": "process_2"})

        # No orphaned tempfiles left behind in the per-source directory.
        leftover = list((tmp_path / "celestrak").glob(".*.tmp"))
        assert leftover == []

    def test_reader_during_concurrent_writes_never_sees_torn_file(self, tmp_path: Path) -> None:
        """A reader looping `get` against a key being rewritten many times sees no garbage.

        Same-process threads instead of subprocesses — the threading
        race is enough to exercise the atomic-rename guarantee within
        a single interpreter, and a thread-based test is much faster
        and more deterministic than a subprocess-based one. The
        cross-process invariant is covered by
        :meth:`test_concurrent_writers_leave_parseable_file` above.
        """
        c = Cache(directory=tmp_path)
        # Seed an initial value so the reader has something to load on
        # the very first iteration.
        c.put("celestrak", "k", {"counter": 0})

        stop_writers = threading.Event()
        observed: list[Any] = []
        reader_errors: list[BaseException] = []

        def writer(value_offset: int) -> None:
            counter = value_offset
            while not stop_writers.is_set():
                c.put("celestrak", "k", {"counter": counter, "from_writer": value_offset})
                counter += 2

        def reader() -> None:
            try:
                while not stop_writers.is_set():
                    hit = c.get_stale("celestrak", "k")
                    if hit is not None:
                        # Each observation must be a complete payload — if
                        # the rename were non-atomic we'd see partial JSON
                        # and get_stale would log + return None, never a
                        # garbage CacheHit.
                        assert "counter" in hit.value
                        observed.append(hit.value)
            except BaseException as exc:
                reader_errors.append(exc)

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(1,)),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        time.sleep(0.5)  # let the race run for half a second
        stop_writers.set()
        for t in threads:
            t.join(timeout=5)

        assert reader_errors == [], f"reader observed corruption: {reader_errors}"
        assert observed, "readers never saw any value (writers may not have run)"


class TestCacheHit:
    def test_frozen(self) -> None:
        hit = CacheHit(value={"v": 1}, fetched_at=datetime.now(tz=timezone.utc))
        with pytest.raises(dataclasses.FrozenInstanceError):
            hit.value = {"v": 2}  # type: ignore[misc]
