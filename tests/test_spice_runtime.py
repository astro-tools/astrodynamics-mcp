"""Tests for :mod:`astrodynamics_mcp.spice_runtime`.

Covers the single-worker CSPICE executor (serialisation onto one thread, the
once-only error-handling configuration that keeps CSPICE off stdout) and the
kernel-pool primitives (furnish-delta, meta-kernel fan-out, list filtering,
unload membership-check) — all driven against the in-memory ``FakeSpice`` from
``tests/_spice_fakes.py``, since the test env has no real ``spiceypy``.
"""

from __future__ import annotations

import asyncio
import sys
import threading

import pytest

from astrodynamics_mcp import spice_runtime
from astrodynamics_mcp.errors import InvalidInputError, UpstreamError
from astrodynamics_mcp.spice_runtime import (
    _same_path,
    furnish_and_describe,
    list_pool,
    normalize_kind_filter,
    run_on_spice_thread,
    unload_kernel,
)
from tests._spice_fakes import FakeSpice, FakeSpiceyError


@pytest.fixture
def fake_spice(monkeypatch: pytest.MonkeyPatch) -> FakeSpice:
    """Inject a fresh ``FakeSpice`` as ``spiceypy`` for the duration of a test."""
    fake = FakeSpice()
    monkeypatch.setitem(sys.modules, "spiceypy", fake)
    return fake


# ---------------------------------------------------------------------------
# normalize_kind_filter — pure, no CSPICE
# ---------------------------------------------------------------------------


class TestNormalizeKindFilter:
    def test_none_is_no_filter(self) -> None:
        assert normalize_kind_filter(None) is None

    def test_single_category(self) -> None:
        assert normalize_kind_filter(["SPK"]) == "SPK"

    def test_multiple_categories_space_joined(self) -> None:
        assert normalize_kind_filter(["SPK", "PCK"]) == "SPK PCK"

    def test_dedupes_preserving_order(self) -> None:
        assert normalize_kind_filter(["SPK", "PCK", "SPK"]) == "SPK PCK"

    def test_uppercases(self) -> None:
        assert normalize_kind_filter(["spk", "pck"]) == "SPK PCK"

    def test_empty_list_is_typed_error(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            normalize_kind_filter([])
        assert excinfo.value.code == "invalid_input.spice_empty_kind_filter"

    def test_unknown_category_is_typed_error(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            normalize_kind_filter(["SPK", "WIDGET"])
        assert excinfo.value.code == "invalid_input.spice_unknown_kind"


class TestSamePath:
    def test_exact_match(self) -> None:
        assert _same_path("/k/de440.bsp", "/k/de440.bsp") is True

    def test_normalised_match(self) -> None:
        # CSPICE stores the furnished path; a caller passing a non-normalised
        # form of the same file (e.g. an unload by name with "./") still matches.
        assert _same_path("/k/de440.bsp", "/k/./de440.bsp") is True

    def test_distinct_paths_do_not_match(self) -> None:
        assert _same_path("/k/a.bsp", "/k/b.bsp") is False


# ---------------------------------------------------------------------------
# The single CSPICE worker
# ---------------------------------------------------------------------------


class TestSingleWorker:
    async def test_runs_on_a_single_spice_named_thread(self, fake_spice: FakeSpice) -> None:
        def probe() -> tuple[int, str]:
            return threading.get_ident(), threading.current_thread().name

        results = await asyncio.gather(*(run_on_spice_thread(probe) for _ in range(5)))
        idents = {ident for ident, _ in results}
        names = {name for _, name in results}
        assert len(idents) == 1, "CSPICE calls must all run on one worker thread"
        assert all(name.startswith("spice") for name in names)

    async def test_calls_never_overlap(self, fake_spice: FakeSpice) -> None:
        """Serialisation guarantee: the worker is never entered concurrently."""
        in_flight = 0
        max_in_flight = 0
        lock = threading.Lock()

        def body() -> None:
            nonlocal in_flight, max_in_flight
            with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            # Busy a moment so an overlapping call would be caught.
            for _ in range(10000):
                pass
            with lock:
                in_flight -= 1

        await asyncio.gather(*(run_on_spice_thread(body) for _ in range(8)))
        assert max_in_flight == 1

    async def test_result_and_args_round_trip(self, fake_spice: FakeSpice) -> None:
        def add(a: int, b: int) -> int:
            return a + b

        assert await run_on_spice_thread(add, 2, 3) == 5

    async def test_configures_cspice_error_handling_once(
        self, fake_spice: FakeSpice, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force re-configuration on this worker so the fake records the setters.
        monkeypatch.setattr(spice_runtime, "_worker_configured", False)

        await run_on_spice_thread(lambda: None)
        await run_on_spice_thread(lambda: None)

        # CSPICE muted off stdout (NULL device, no printing) and set to RETURN so
        # spiceypy raises instead of the process aborting — configured exactly once.
        assert fake_spice.calls["erract"] == [("SET", "RETURN")]
        assert fake_spice.calls["errdev"] == [("SET", "NULL")]
        assert fake_spice.calls["errprt"] == [("SET", "NONE")]


# ---------------------------------------------------------------------------
# list_pool
# ---------------------------------------------------------------------------


class TestListPool:
    def test_empty_pool(self, fake_spice: FakeSpice) -> None:
        assert list_pool() == []

    def test_lists_all_with_fields(self, fake_spice: FakeSpice) -> None:
        fake_spice.furnsh("/k/naif0012.tls")
        fake_spice.furnsh("/k/de440.bsp")
        rows = list_pool()
        assert {r.name for r in rows} == {"/k/naif0012.tls", "/k/de440.bsp"}
        bsp = next(r for r in rows if r.name == "/k/de440.bsp")
        assert bsp.type == "SPK"
        assert bsp.handle != 0  # binary kernel carries a DAF handle
        tls = next(r for r in rows if r.name == "/k/naif0012.tls")
        assert tls.type == "TEXT"
        assert tls.handle == 0  # text kernel loads into the pool, no handle

    def test_category_filter(self, fake_spice: FakeSpice) -> None:
        fake_spice.furnsh("/k/naif0012.tls")  # TEXT
        fake_spice.furnsh("/k/de440.bsp")  # SPK
        fake_spice.furnsh("/k/pck00011.tpc")  # PCK
        spk_only = list_pool("SPK")
        assert [r.name for r in spk_only] == ["/k/de440.bsp"]
        spk_pck = {r.name for r in list_pool("SPK PCK")}
        assert spk_pck == {"/k/de440.bsp", "/k/pck00011.tpc"}

    def test_skips_not_found_slot(
        self, fake_spice: FakeSpice, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A kernel vanishing mid-enumeration (ktotal says N, kdata reports
        not-found) is skipped, never emitted as a blank row."""
        fake_spice.furnsh("/k/de440.bsp")
        monkeypatch.setattr(fake_spice, "ktotal", lambda kind: 2)  # claim one extra
        rows = list_pool()
        assert [r.name for r in rows] == ["/k/de440.bsp"]


# ---------------------------------------------------------------------------
# furnish_and_describe
# ---------------------------------------------------------------------------


class TestFurnishAndDescribe:
    def test_single_kernel_returns_one_row(self, fake_spice: FakeSpice) -> None:
        rows = furnish_and_describe("/k/de440.bsp")
        assert len(rows) == 1
        assert rows[0].name == "/k/de440.bsp"
        assert rows[0].type == "SPK"

    def test_meta_kernel_returns_every_pulled_in_kernel(self, fake_spice: FakeSpice) -> None:
        fake_spice.plan_furnish(
            "/k/mission.tm",
            [
                {"name": "/k/mission.tm", "type": "META", "source": "", "handle": 0},
                {"name": "/k/naif0012.tls", "type": "TEXT", "source": "/k/mission.tm", "handle": 0},
                {"name": "/k/de440.bsp", "type": "SPK", "source": "/k/mission.tm", "handle": 7},
            ],
        )
        rows = furnish_and_describe("/k/mission.tm")
        assert {r.name for r in rows} == {"/k/mission.tm", "/k/naif0012.tls", "/k/de440.bsp"}
        assert {r.type for r in rows} == {"META", "TEXT", "SPK"}

    def test_only_the_delta_is_returned(self, fake_spice: FakeSpice) -> None:
        furnish_and_describe("/k/naif0012.tls")
        rows = furnish_and_describe("/k/de440.bsp")  # second, distinct furnish
        assert [r.name for r in rows] == ["/k/de440.bsp"]  # not the LSK already loaded

    def test_idempotent_refurnish_returns_existing_row(self, fake_spice: FakeSpice) -> None:
        furnish_and_describe("/k/de440.bsp")
        rows = furnish_and_describe("/k/de440.bsp")  # already loaded → no delta
        assert [r.name for r in rows] == ["/k/de440.bsp"]

    def test_furnish_failure_is_typed_upstream_error(self, fake_spice: FakeSpice) -> None:
        fake_spice.fail_furnsh("/k/corrupt.bsp", FakeSpiceyError("bad DAF"))
        with pytest.raises(UpstreamError) as excinfo:
            furnish_and_describe("/k/corrupt.bsp")
        assert excinfo.value.code == "upstream.spice_furnish_failed"
        assert excinfo.value.data["path"] == "/k/corrupt.bsp"


# ---------------------------------------------------------------------------
# unload_kernel
# ---------------------------------------------------------------------------


class TestUnloadKernel:
    def test_unloads_and_returns_remaining_count(self, fake_spice: FakeSpice) -> None:
        furnish_and_describe("/k/naif0012.tls")
        furnish_and_describe("/k/de440.bsp")
        remaining = unload_kernel("/k/de440.bsp")
        assert remaining == 1
        assert {r.name for r in list_pool()} == {"/k/naif0012.tls"}

    def test_unload_missing_is_typed_error(self, fake_spice: FakeSpice) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            unload_kernel("/k/never-loaded.bsp")
        assert excinfo.value.code == "invalid_input.spice_kernel_not_loaded"
        assert excinfo.value.data["name"] == "/k/never-loaded.bsp"

    def test_unload_does_not_touch_cspice_when_missing(self, fake_spice: FakeSpice) -> None:
        with pytest.raises(InvalidInputError):
            unload_kernel("/k/never-loaded.bsp")
        assert fake_spice.calls["unload"] == []  # membership pre-check guards the call

    def test_unload_failure_is_typed_upstream_error(
        self, fake_spice: FakeSpice, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        furnish_and_describe("/k/de440.bsp")

        def boom(_path: str) -> None:
            raise FakeSpiceyError("unload exploded")

        monkeypatch.setattr(fake_spice, "unload", boom)
        with pytest.raises(UpstreamError) as excinfo:
            unload_kernel("/k/de440.bsp")
        assert excinfo.value.code == "upstream.spice_unload_failed"
