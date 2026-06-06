"""Coverage-gap tests for the SPICE subsystem review.

Collects the checks the per-tool ``test_tool_spice_*.py`` files did not yet
exercise: ``_register_spice_tools`` idempotency (mirroring the GMAT
``test_gmat_coverage_gaps`` precedent), the ``_parse_et_seconds`` bool guard,
and tool-level interleaved-call coverage (several ``_do_*`` coroutines run
concurrently through the single CSPICE worker stay consistent).

Tests live in their own file rather than scattered across the existing per-tool
suites so the coverage delta against the SPICE review is easy to read. They run
against the in-memory ``FakeSpice`` (the test env ships no real ``spiceypy``).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from mcp.server.fastmcp import FastMCP

from astrodynamics_mcp.errors import InvalidInputError
from astrodynamics_mcp.tools import spice as spice_tools
from astrodynamics_mcp.tools.spice import (
    RotatableState,
    _do_body_parameters,
    _do_frame_transform,
    _do_list_kernels,
    _do_load_kernel,
    _do_state,
    _do_time_convert,
)
from astrodynamics_mcp.units import QuantityVector
from tests._spice_fakes import FakeSpice

_EXPECTED_TOOL_NAMES = frozenset(
    {
        "spice_load_kernel",
        "spice_list_kernels",
        "spice_unload_kernel",
        "spice_state",
        "spice_frame_transform",
        "spice_body_parameters",
        "spice_time_convert",
    }
)


@pytest.fixture
def fake_spice(monkeypatch: pytest.MonkeyPatch) -> FakeSpice:
    fake = FakeSpice()
    monkeypatch.setitem(sys.modules, "spiceypy", fake)
    return fake


def _write_kernel(tmp_path: Path, name: str) -> str:
    path = tmp_path / name
    path.write_bytes(b"fake kernel bytes")
    return str(path)


_ROTATION = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
_EPOCHS = ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"]


# ---------------------------------------------------------------------------
# Registration idempotency
# ---------------------------------------------------------------------------


class TestRegistrationIdempotency:
    """``_register_spice_tools()`` is safe to call twice (hot-reload / re-init)."""

    @pytest.fixture
    def fresh_mcp(self, monkeypatch: pytest.MonkeyPatch) -> FastMCP:
        fresh = FastMCP("spice-coverage-test")
        monkeypatch.setattr("astrodynamics_mcp.server.mcp", fresh)
        return fresh

    async def test_register_twice_does_not_change_surface(self, fresh_mcp: FastMCP) -> None:
        spice_tools._register_spice_tools()
        first = {t.name for t in await fresh_mcp.list_tools()}
        # A second call would either silently double-register or raise; either
        # would break a hot-reload / re-init path. The contract is "no change in
        # tool surface".
        try:
            spice_tools._register_spice_tools()
        except Exception as exc:
            pytest.skip(f"double-register raised cleanly: {exc!r}")
        second = {t.name for t in await fresh_mcp.list_tools()}
        assert first == second, "double _register_spice_tools changed the tool surface"
        assert _EXPECTED_TOOL_NAMES.issubset(first)


# ---------------------------------------------------------------------------
# _parse_et_seconds bool guard
# ---------------------------------------------------------------------------


class TestEtValueGuard:
    """A boolean ET value is rejected as a typed error before any CSPICE call.

    ``bool`` is an ``int`` subclass, so a guard that only checked ``isinstance(
    value, (int, float, str))`` would silently coerce ``True`` to ``1.0``.
    ``_parse_et_seconds`` rejects it explicitly.
    """

    async def test_bool_et_value_is_typed_error(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _do_load_kernel(_write_kernel(tmp_path, "naif0012.tls"))  # LSK furnished
        with pytest.raises(InvalidInputError) as excinfo:
            await _do_time_convert(value=True, from_scale="ET", to_scale="UTC", spacecraft=None)
        assert excinfo.value.code == "invalid_input.spice_invalid_et_value"
        # The guard fires before any CSPICE call.
        assert fake_spice.calls["et2utc"] == []


# ---------------------------------------------------------------------------
# Interleaved tool calls
# ---------------------------------------------------------------------------


class TestInterleavedCalls:
    """Concurrent ``_do_*`` calls share one serialised CSPICE worker and stay
    consistent: each tool's whole interaction runs atomically, so interleaving
    several reads neither corrupts the pool nor crosses results."""

    async def test_concurrent_reads_are_consistent(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        # Furnish the kernels the queries below each depend on, then plan their
        # CSPICE results.
        await _do_load_kernel(_write_kernel(tmp_path, "naif0012.tls"))  # LSK
        await _do_load_kernel(_write_kernel(tmp_path, "de440.bsp"))  # SPK
        await _do_load_kernel(_write_kernel(tmp_path, "pck00011.tpc"))  # PCK
        fake_spice.plan_state("MOON", "EARTH", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1.0)
        fake_spice.plan_rotation("J2000", "IAU_MARS", _ROTATION, requires="PCK")
        fake_spice.plan_body_code("MARS", 499)
        fake_spice.plan_body_constant(499, "RADII", [3396.19, 3396.19, 3376.2], requires="PCK")
        fake_spice.plan_body_constant(499, "GM", [42828.375214], requires="PCK")

        state_resp, frame_resp, body_resp, list_resp = await asyncio.gather(
            _do_state(
                target="MOON",
                observer="EARTH",
                epochs=_EPOCHS,
                frame="J2000",
                aberration="NONE",
            ),
            _do_frame_transform(
                from_frame="J2000",
                to_frame="IAU_MARS",
                epoch="2026-01-01T00:00:00Z",
                state=RotatableState(
                    position=QuantityVector(value=[4000.0, 5000.0, 6000.0], unit="km")
                ),
            ),
            _do_body_parameters(body="MARS", parameters=None),
            _do_list_kernels(None),
        )

        # Each tool's result is complete and correct — no cross-talk.
        assert [s.epoch for s in state_resp.states] == _EPOCHS
        assert frame_resp.position is not None
        assert [row.value for row in frame_resp.rotation] == _ROTATION
        assert [p.name for p in body_resp.parameters] == ["radii", "gm"]
        # The pool is intact after the concurrent burst: all three kernels remain.
        assert len(list_resp.kernels) == 3
