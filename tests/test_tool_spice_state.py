"""Tests for the ``spice_state`` SPK state-query tool.

Drives the module-level ``_do_state`` helper (the registered slot is a thin
wrapper) against the in-memory ``FakeSpice``, which models CSPICE's kernel
dependence: ``str2et`` needs a leap-second (TEXT) kernel furnished and
``spkezr`` needs an SPK, so the missing-kernel acceptance path is exercised
through the real pool state rather than a bolted-on flag.

Covers the acceptance contract: position/velocity returned as ``{value, unit}``
quantities with the frame / observer / epoch echoed; light time only for a
non-NONE aberration; typed errors for a missing kernel and an unknown
aberration; the output schema round-trips; and a committed reference golden
(Moon relative to Earth in J2000).

Per the chosen v0.3 strategy, the golden validates the tool's packaging of a
known reference state — the ``FakeSpice`` is fed the same raw vector — not
CSPICE's own ephemeris math, which no CI cell runs (the test env ships no
``spiceypy`` and a real planetary SPK is too large to commit).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from astrodynamics_mcp.errors import InvalidInputError, UpstreamError
from astrodynamics_mcp.tools import spice as spice_tools
from astrodynamics_mcp.tools.spice import (
    SpiceStateAtEpoch,
    SpiceStateResponse,
    _do_load_kernel,
    _do_state,
    _to_cspice_utc,
)
from astrodynamics_mcp.units import Quantity, QuantityVector
from tests._spice_fakes import FakeSpice

# A representative Moon-geocentric-J2000 reference state (km, km/s). The exact
# values are the committed golden's; the fake is fed this same raw vector.
_MOON_POSITION = [-291800.0, 214600.0, 118300.0]
_MOON_VELOCITY = [-0.6062, -0.6779, -0.1067]
_MOON_LIGHT_TIME = 1.28425  # one-way light time (s); dropped for a NONE query
_EPOCH = "2026-01-01T00:00:00Z"

_GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "spice_state_golden.json"


@pytest.fixture
def fake_spice(monkeypatch: pytest.MonkeyPatch) -> FakeSpice:
    fake = FakeSpice()
    monkeypatch.setitem(sys.modules, "spiceypy", fake)
    return fake


def _write_kernel(tmp_path: Path, name: str, payload: bytes = b"fake kernel bytes") -> str:
    path = tmp_path / name
    path.write_bytes(payload)
    return str(path)


async def _furnish_lsk_and_spk(tmp_path: Path) -> None:
    """Furnish a leap-second kernel and an SPK so a state query can run."""
    await _do_load_kernel(_write_kernel(tmp_path, "naif0012.tls"))
    await _do_load_kernel(_write_kernel(tmp_path, "de440.bsp"))


# ---------------------------------------------------------------------------
# UTC normalisation for CSPICE str2et
# ---------------------------------------------------------------------------


class TestToCspiceUtc:
    def test_strips_z_designator(self) -> None:
        # CSPICE str2et does not parse a trailing Z; it must be converted to a
        # plain UTC calendar string.
        assert _to_cspice_utc("2026-01-01T00:00:00Z") == "2026-01-01T00:00:00.000000"

    def test_converts_offset_to_utc(self) -> None:
        # A +02:00 epoch is the same instant as 10:00:00 UTC.
        assert _to_cspice_utc("2026-01-01T12:00:00+02:00") == "2026-01-01T10:00:00.000000"

    def test_offset_free_epoch_treated_as_utc(self) -> None:
        assert _to_cspice_utc("2026-01-01T00:00:00") == "2026-01-01T00:00:00.000000"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestStateQuery:
    async def test_geometric_state_echoes_and_wraps(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_lsk_and_spk(tmp_path)
        fake_spice.plan_state("MOON", "EARTH", _MOON_POSITION + _MOON_VELOCITY, _MOON_LIGHT_TIME)

        response = await _do_state(
            target="MOON", observer="EARTH", epochs=[_EPOCH], frame="J2000", aberration="NONE"
        )

        assert isinstance(response, SpiceStateResponse)
        assert response.target == "MOON"
        assert response.observer == "EARTH"
        assert response.frame == "J2000"
        assert response.aberration == "NONE"
        assert len(response.states) == 1

        state = response.states[0]
        assert state.epoch == _EPOCH
        assert state.position.value == _MOON_POSITION
        assert state.position.unit == "km"
        assert state.velocity.value == _MOON_VELOCITY
        assert state.velocity.unit == "km/s"
        # NONE correction → no light time reported.
        assert state.light_time is None

    async def test_str2et_receives_offset_free_utc(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_lsk_and_spk(tmp_path)
        await _do_state(
            target="MOON", observer="EARTH", epochs=[_EPOCH], frame="J2000", aberration="NONE"
        )
        # The Z designator was stripped before reaching CSPICE.
        assert fake_spice.calls["str2et"] == ["2026-01-01T00:00:00.000000"]

    async def test_light_time_reported_for_non_none_correction(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_lsk_and_spk(tmp_path)
        fake_spice.plan_state("MOON", "EARTH", _MOON_POSITION + _MOON_VELOCITY, _MOON_LIGHT_TIME)

        response = await _do_state(
            target="MOON", observer="EARTH", epochs=[_EPOCH], frame="J2000", aberration="LT"
        )

        assert response.aberration == "LT"
        light_time = response.states[0].light_time
        assert light_time is not None
        assert light_time.value == _MOON_LIGHT_TIME
        assert light_time.unit == "s"

    async def test_aberration_is_upper_cased(self, fake_spice: FakeSpice, tmp_path: Path) -> None:
        await _furnish_lsk_and_spk(tmp_path)
        response = await _do_state(
            target="MOON", observer="EARTH", epochs=[_EPOCH], frame="J2000", aberration="lt+s"
        )
        assert response.aberration == "LT+S"
        assert response.states[0].light_time is not None

    async def test_multiple_epochs_preserve_order(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_lsk_and_spk(tmp_path)
        epochs = [
            "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z",
            "2026-01-03T00:00:00Z",
        ]
        response = await _do_state(
            target="MOON", observer="EARTH", epochs=epochs, frame="J2000", aberration="NONE"
        )
        assert [s.epoch for s in response.states] == epochs

    async def test_multi_epoch_query_is_a_single_worker_dispatch(
        self, fake_spice: FakeSpice, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # All epochs run in one worker call (query_states), so the multi-epoch
        # query is one atomic CSPICE interaction — not one dispatch per epoch.
        from astrodynamics_mcp.spice_runtime import run_on_spice_thread as real

        await _furnish_lsk_and_spk(tmp_path)
        dispatched: list[str] = []

        async def counting(fn: object, *args: object, **kwargs: object) -> object:
            dispatched.append(fn.__name__)  # type: ignore[attr-defined]
            return await real(fn, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(spice_tools, "run_on_spice_thread", counting)
        await _do_state(
            target="MOON",
            observer="EARTH",
            epochs=["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"],
            frame="J2000",
            aberration="NONE",
        )
        assert dispatched == ["query_states"]


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------


class TestTypedErrors:
    async def test_missing_leapseconds_kernel_is_typed_error(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        # SPK furnished but no LSK — str2et fails first.
        await _do_load_kernel(_write_kernel(tmp_path, "de440.bsp"))
        with pytest.raises(UpstreamError) as excinfo:
            await _do_state(
                target="MOON", observer="EARTH", epochs=[_EPOCH], frame="J2000", aberration="NONE"
            )
        assert excinfo.value.code == "upstream.spice_state_failed"

    async def test_missing_spk_is_typed_error(self, fake_spice: FakeSpice, tmp_path: Path) -> None:
        # LSK furnished but no SPK — str2et succeeds, spkezr fails.
        await _do_load_kernel(_write_kernel(tmp_path, "naif0012.tls"))
        with pytest.raises(UpstreamError) as excinfo:
            await _do_state(
                target="MOON", observer="EARTH", epochs=[_EPOCH], frame="J2000", aberration="NONE"
            )
        assert excinfo.value.code == "upstream.spice_state_failed"

    async def test_no_kernels_at_all_is_typed_error_not_empty(self, fake_spice: FakeSpice) -> None:
        # The acceptance guard: a missing-kernel query never returns a silent
        # empty result — it raises a typed error.
        with pytest.raises(UpstreamError) as excinfo:
            await _do_state(
                target="MOON", observer="EARTH", epochs=[_EPOCH], frame="J2000", aberration="NONE"
            )
        assert excinfo.value.code == "upstream.spice_state_failed"

    async def test_unknown_aberration_is_typed_error_before_cspice(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_lsk_and_spk(tmp_path)
        with pytest.raises(InvalidInputError) as excinfo:
            await _do_state(
                target="MOON",
                observer="EARTH",
                epochs=[_EPOCH],
                frame="J2000",
                aberration="light-time",
            )
        assert excinfo.value.code == "invalid_input.spice_unknown_aberration"
        # Validation happens before any CSPICE call.
        assert fake_spice.calls["str2et"] == []
        assert fake_spice.calls["spkezr"] == []


# ---------------------------------------------------------------------------
# Output-schema round-trip
# ---------------------------------------------------------------------------


class TestOutputRoundTrip:
    def test_response_roundtrips_through_schema(self) -> None:
        response = SpiceStateResponse(
            target="MOON",
            observer="EARTH",
            frame="J2000",
            aberration="LT",
            states=[
                SpiceStateAtEpoch(
                    epoch=_EPOCH,
                    position=QuantityVector(value=_MOON_POSITION, unit="km"),
                    velocity=QuantityVector(value=_MOON_VELOCITY, unit="km/s"),
                    light_time=Quantity(value=_MOON_LIGHT_TIME, unit="s"),
                )
            ],
        )
        first = response.model_dump_json()
        rebuilt = SpiceStateResponse.model_validate_json(first)
        assert rebuilt.model_dump_json() == first

    def test_extra_keys_forbidden(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SpiceStateResponse.model_validate(
                {
                    "target": "MOON",
                    "observer": "EARTH",
                    "frame": "J2000",
                    "aberration": "NONE",
                    "states": [],
                    "surprise": True,
                }
            )


# ---------------------------------------------------------------------------
# Committed reference golden
# ---------------------------------------------------------------------------


class TestReferenceGolden:
    async def test_state_matches_committed_golden(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_lsk_and_spk(tmp_path)
        fake_spice.plan_state("MOON", "EARTH", _MOON_POSITION + _MOON_VELOCITY, _MOON_LIGHT_TIME)

        response = await _do_state(
            target="MOON", observer="EARTH", epochs=[_EPOCH], frame="J2000", aberration="NONE"
        )
        actual = response.model_dump(mode="json")
        expected = json.loads(_GOLDEN_PATH.read_text())
        assert actual == expected


# ---------------------------------------------------------------------------
# End-to-end through the registered slot
# ---------------------------------------------------------------------------


class TestRegisteredToolCall:
    @pytest.fixture
    def registered_mcp(self, monkeypatch: pytest.MonkeyPatch) -> FastMCP:
        fresh = FastMCP("spice-state-test")
        monkeypatch.setattr("astrodynamics_mcp.server.mcp", fresh)
        spice_tools._register_spice_tools()
        return fresh

    async def test_state_round_trip(
        self, registered_mcp: FastMCP, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_lsk_and_spk(tmp_path)
        fake_spice.plan_state("MOON", "EARTH", _MOON_POSITION + _MOON_VELOCITY, _MOON_LIGHT_TIME)

        _, result = await registered_mcp.call_tool(
            "spice_state",
            {"target": "MOON", "observer": "EARTH", "epochs": [_EPOCH]},
        )
        assert isinstance(result, dict)
        assert result["frame"] == "J2000"
        assert result["aberration"] == "NONE"
        assert result["states"][0]["position"]["value"] == _MOON_POSITION
        assert result["states"][0]["light_time"] is None

    async def test_bare_date_epoch_rejected(
        self, registered_mcp: FastMCP, fake_spice: FakeSpice
    ) -> None:
        # The Epoch type's BeforeValidator rejects a bare date during the slot's
        # argument validation, before any CSPICE call. FastMCP.call_tool surfaces
        # the validator's message (the low-level wire handler additionally
        # recovers the typed `invalid_input.epoch_missing_time_component` code —
        # see tests/test_server.py).
        with pytest.raises(ToolError) as excinfo:
            await registered_mcp.call_tool(
                "spice_state",
                {"target": "MOON", "observer": "EARTH", "epochs": ["2026-01-01"]},
            )
        assert "include a time component" in str(excinfo.value)
        assert fake_spice.calls["str2et"] == []
