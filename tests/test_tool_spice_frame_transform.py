"""Tests for the ``spice_frame_transform`` FK / PCK frame-rotation tool.

Drives the module-level ``_do_frame_transform`` helper (the registered slot is a
thin wrapper) against the in-memory ``FakeSpice``, which models CSPICE's kernel
dependence: ``str2et`` needs a leap-second (TEXT) kernel furnished, and a
pinned ``pxform`` / ``sxform`` raises unless its required kernel category (e.g.
a PCK for a body-fixed frame) is in the pool — so the missing-kernel acceptance
paths are exercised through real pool state rather than a bolted-on flag.

Covers the acceptance contract: the 3x3 rotation and the rotated position /
velocity returned as ``{value, unit}`` quantities with from-frame, to-frame, and
epoch echoed; pxform for a 3-vector and sxform for a 6-vector; typed errors with
stable codes for a missing LSK, a missing PCK, and an unknown frame; the output
schema round-trips; a committed reference golden; and a frame-equivalence check
against the kernel-free astropy ``frame_transform`` where both define the same
Earth-fixed frame.

Per the v0.3 strategy (the test env ships no ``spiceypy``), the golden validates
the tool's packaging of a known rotation matrix — the ``FakeSpice`` is fed the
same matrix — not CSPICE's own orientation math, which no CI cell runs. The
equivalence test sources its rotation from astropy, so it cross-checks the
tool's matrix-application convention against a real independent rotation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from astrodynamics_mcp.errors import InvalidInputError, UpstreamError
from astrodynamics_mcp.schemas.base import Frame, StateVector
from astrodynamics_mcp.tools import spice as spice_tools
from astrodynamics_mcp.tools.frames import frame_transform
from astrodynamics_mcp.tools.spice import (
    RotatableState,
    SpiceFrameTransformResponse,
    _do_frame_transform,
    _do_load_kernel,
)
from astrodynamics_mcp.units import QuantityVector
from tests._spice_fakes import FakeSpice

# A 90°-about-z rotation — orthonormal, non-symmetric (so a transpose bug shows),
# and exactly representable so the golden carries no float noise.
_ROTATION_Z90: list[list[float]] = [
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
]
# The matching 6x6 state transform: the same rotation on both the position and
# velocity blocks, with a zero coupling block (an inertial-to-inertial pair, no
# frame-rotation-rate term). sxform([r, v]) = [R r, R v].
_STATE_TRANSFORM_Z90: list[list[float]] = [
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]
_POSITION = [4000.0, 5000.0, 6000.0]
_VELOCITY = [1.0, 2.0, 3.0]
# R @ position / R @ velocity.
_ROTATED_POSITION = [5000.0, -4000.0, 6000.0]
_ROTATED_VELOCITY = [2.0, -1.0, 3.0]
_EPOCH = "2026-01-01T00:00:00Z"
_FROM = "J2000"
_TO = "IAU_MARS"

_GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "spice_frame_transform_golden.json"


@pytest.fixture
def fake_spice(monkeypatch: pytest.MonkeyPatch) -> FakeSpice:
    fake = FakeSpice()
    monkeypatch.setitem(sys.modules, "spiceypy", fake)
    return fake


def _write_kernel(tmp_path: Path, name: str, payload: bytes = b"fake kernel bytes") -> str:
    path = tmp_path / name
    path.write_bytes(payload)
    return str(path)


async def _furnish_lsk_and_pck(tmp_path: Path) -> None:
    """Furnish a leap-second kernel and a PCK so a body-fixed rotation can run."""
    await _do_load_kernel(_write_kernel(tmp_path, "naif0012.tls"))
    await _do_load_kernel(_write_kernel(tmp_path, "pck00011.tpc"))


def _position_state() -> RotatableState:
    return RotatableState(position=QuantityVector(value=_POSITION, unit="km"))


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestRotationOnly:
    async def test_omitted_state_returns_matrix_only(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_lsk_and_pck(tmp_path)
        fake_spice.plan_rotation(_FROM, _TO, _ROTATION_Z90, requires="PCK")

        response = await _do_frame_transform(
            from_frame=_FROM, to_frame=_TO, epoch=_EPOCH, state=None
        )

        assert isinstance(response, SpiceFrameTransformResponse)
        assert response.from_frame == _FROM
        assert response.to_frame == _TO
        assert response.epoch == _EPOCH
        assert [row.value for row in response.rotation] == _ROTATION_Z90
        assert all(row.unit == "1" for row in response.rotation)
        assert response.position is None
        assert response.velocity is None
        # No state to rotate → the 6x6 path is never touched.
        assert fake_spice.calls["sxform"] == []

    async def test_str2et_receives_offset_free_utc(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_lsk_and_pck(tmp_path)
        fake_spice.plan_rotation(_FROM, _TO, _ROTATION_Z90, requires="PCK")
        await _do_frame_transform(from_frame=_FROM, to_frame=_TO, epoch=_EPOCH, state=None)
        # The Z designator was stripped before reaching CSPICE.
        assert fake_spice.calls["str2et"] == ["2026-01-01T00:00:00.000000"]


class TestPositionRotation:
    async def test_position_only_uses_pxform(self, fake_spice: FakeSpice, tmp_path: Path) -> None:
        await _furnish_lsk_and_pck(tmp_path)
        fake_spice.plan_rotation(_FROM, _TO, _ROTATION_Z90, requires="PCK")

        response = await _do_frame_transform(
            from_frame=_FROM, to_frame=_TO, epoch=_EPOCH, state=_position_state()
        )

        assert response.position is not None
        assert response.position.value == _ROTATED_POSITION
        assert response.position.unit == "km"
        assert response.velocity is None
        # A 3-vector rotation goes through pxform, never sxform.
        assert fake_spice.calls["pxform"]
        assert fake_spice.calls["sxform"] == []

    async def test_position_unit_is_echoed(self, fake_spice: FakeSpice, tmp_path: Path) -> None:
        await _furnish_lsk_and_pck(tmp_path)
        fake_spice.plan_rotation(_FROM, _TO, _ROTATION_Z90, requires="PCK")
        response = await _do_frame_transform(
            from_frame=_FROM,
            to_frame=_TO,
            epoch=_EPOCH,
            state=RotatableState(position=QuantityVector(value=[4_000_000.0, 0.0, 0.0], unit="m")),
        )
        assert response.position is not None
        assert response.position.unit == "m"
        assert response.position.value == [0.0, -4_000_000.0, 0.0]


class TestStateRotation:
    async def test_position_and_velocity_use_sxform(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_lsk_and_pck(tmp_path)
        fake_spice.plan_rotation(
            _FROM, _TO, _ROTATION_Z90, state_transform=_STATE_TRANSFORM_Z90, requires="PCK"
        )

        response = await _do_frame_transform(
            from_frame=_FROM,
            to_frame=_TO,
            epoch=_EPOCH,
            state=RotatableState(
                position=QuantityVector(value=_POSITION, unit="km"),
                velocity=QuantityVector(value=_VELOCITY, unit="km/s"),
            ),
        )

        assert response.position is not None
        assert response.position.value == _ROTATED_POSITION
        assert response.velocity is not None
        assert response.velocity.value == _ROTATED_VELOCITY
        assert response.velocity.unit == "km/s"
        # The 6-vector rotation goes through the state transform — and only that:
        # the 3x3 orientation is read from sxform's upper-left block, so no
        # separate pxform call is made.
        assert fake_spice.calls["sxform"]
        assert fake_spice.calls["pxform"] == []
        # The 3x3 orientation is still reported alongside the rotated state.
        assert [row.value for row in response.rotation] == _ROTATION_Z90


# ---------------------------------------------------------------------------
# Matrix-application convention
# ---------------------------------------------------------------------------


class TestRotationConvention:
    async def test_rotated_position_is_matrix_times_vector(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        """The tool must apply R as out = R @ v (row-major), not its transpose."""
        await _furnish_lsk_and_pck(tmp_path)
        matrix = [
            [0.6, -0.8, 0.0],
            [0.8, 0.6, 0.0],
            [0.0, 0.0, 1.0],
        ]
        fake_spice.plan_rotation(_FROM, _TO, matrix, requires="PCK")
        vector = [100.0, 200.0, 300.0]

        response = await _do_frame_transform(
            from_frame=_FROM,
            to_frame=_TO,
            epoch=_EPOCH,
            state=RotatableState(position=QuantityVector(value=vector, unit="km")),
        )

        expected = [sum(matrix[i][j] * vector[j] for j in range(3)) for i in range(3)]
        assert response.position is not None
        for axis in range(3):
            assert response.position.value[axis] == pytest.approx(expected[axis], abs=1e-12)
        # The transpose would give a different result for this non-symmetric R.
        transposed = [sum(matrix[j][i] * vector[j] for j in range(3)) for i in range(3)]
        assert expected != transposed


# ---------------------------------------------------------------------------
# Frame-equivalence with the kernel-free astropy frame_transform
# ---------------------------------------------------------------------------


class TestAstropyFrameEquivalence:
    """Where SPICE and astropy both define an Earth-fixed frame, the two paths agree.

    The rotation is sourced from astropy's GCRS→ITRS transform (a pure
    geocentric rotation: same origin, no offset), reconstructed from the images
    of the three basis vectors, and fed to ``FakeSpice`` as the SPICE-side
    ``J2000``→``ITRF93`` rotation (J2000 axes ≈ GCRS, ITRF93 ≈ ITRS). The SPICE
    tool's rotated position must then match the astropy tool's for the same
    vector — a genuine cross-path check of the tool's matrix-application
    convention against a real, independent rotation.
    """

    async def test_earth_fixed_rotation_matches_astropy(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        epoch = "2026-03-20T12:00:00Z"

        async def astropy_gcrs_to_itrs(vector: list[float]) -> list[float]:
            out = await frame_transform(
                state=StateVector(
                    r=QuantityVector(value=vector, unit="km"),
                    v=QuantityVector(value=[0.0, 0.0, 0.0], unit="km/s"),
                    frame=Frame.GCRS,
                    epoch=epoch,
                ),
                to_frame=Frame.ITRS,
            )
            return list(out.state.r.value)

        # Column j of R is the image of basis vector e_j (positions rotate purely).
        cols = [
            await astropy_gcrs_to_itrs([1.0, 0.0, 0.0]),
            await astropy_gcrs_to_itrs([0.0, 1.0, 0.0]),
            await astropy_gcrs_to_itrs([0.0, 0.0, 1.0]),
        ]
        matrix = [[cols[j][i] for j in range(3)] for i in range(3)]

        # str2et still needs an LSK; the J2000↔ITRF93 rotation itself needs no
        # body kernel, so the plan carries no `requires` gate.
        await _do_load_kernel(_write_kernel(tmp_path, "naif0012.tls"))
        fake_spice.plan_rotation("J2000", "ITRF93", matrix)

        test_vector = [4000.0, 5000.0, 6000.0]
        response = await _do_frame_transform(
            from_frame="J2000",
            to_frame="ITRF93",
            epoch=epoch,
            state=RotatableState(position=QuantityVector(value=test_vector, unit="km")),
        )
        reference = await astropy_gcrs_to_itrs(test_vector)

        assert response.position is not None
        for axis in range(3):
            assert response.position.value[axis] == pytest.approx(reference[axis], abs=1e-6)


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------


class TestTypedErrors:
    async def test_missing_leapseconds_kernel_is_typed_error(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        # PCK furnished and the rotation pinned, but no LSK — str2et fails first.
        await _do_load_kernel(_write_kernel(tmp_path, "pck00011.tpc"))
        fake_spice.plan_rotation(_FROM, _TO, _ROTATION_Z90, requires="PCK")
        with pytest.raises(UpstreamError) as excinfo:
            await _do_frame_transform(
                from_frame=_FROM, to_frame=_TO, epoch=_EPOCH, state=_position_state()
            )
        assert excinfo.value.code == "upstream.spice_frame_transform_failed"

    async def test_missing_pck_is_typed_error(self, fake_spice: FakeSpice, tmp_path: Path) -> None:
        # LSK furnished and the rotation pinned, but the required PCK is absent —
        # str2et succeeds, pxform fails.
        await _do_load_kernel(_write_kernel(tmp_path, "naif0012.tls"))
        fake_spice.plan_rotation(_FROM, _TO, _ROTATION_Z90, requires="PCK")
        with pytest.raises(UpstreamError) as excinfo:
            await _do_frame_transform(
                from_frame=_FROM, to_frame=_TO, epoch=_EPOCH, state=_position_state()
            )
        assert excinfo.value.code == "upstream.spice_frame_transform_failed"

    async def test_unknown_frame_is_typed_error(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        # LSK + PCK furnished, but no rotation pinned for the pair — an unknown /
        # unconnectable frame.
        await _furnish_lsk_and_pck(tmp_path)
        with pytest.raises(UpstreamError) as excinfo:
            await _do_frame_transform(
                from_frame=_FROM, to_frame="IAU_PLUTO", epoch=_EPOCH, state=_position_state()
            )
        assert excinfo.value.code == "upstream.spice_frame_transform_failed"

    async def test_missing_state_transform_is_typed_error(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        # A 6-vector request where only the 3x3 (not the 6x6) is available —
        # sxform raises.
        await _furnish_lsk_and_pck(tmp_path)
        fake_spice.plan_rotation(_FROM, _TO, _ROTATION_Z90, requires="PCK")
        with pytest.raises(UpstreamError) as excinfo:
            await _do_frame_transform(
                from_frame=_FROM,
                to_frame=_TO,
                epoch=_EPOCH,
                state=RotatableState(
                    position=QuantityVector(value=_POSITION, unit="km"),
                    velocity=QuantityVector(value=_VELOCITY, unit="km/s"),
                ),
            )
        assert excinfo.value.code == "upstream.spice_frame_transform_failed"

    async def test_bad_position_unit_is_typed_error_before_cspice(
        self, fake_spice: FakeSpice
    ) -> None:
        # A velocity unit on the position field is rejected by the input model,
        # before any CSPICE call.
        with pytest.raises(InvalidInputError) as excinfo:
            RotatableState(position=QuantityVector(value=_POSITION, unit="km/s"))
        assert excinfo.value.code == "invalid_input.wrong_unit_category"
        assert fake_spice.calls["pxform"] == []


# ---------------------------------------------------------------------------
# Output-schema round-trip
# ---------------------------------------------------------------------------


class TestOutputRoundTrip:
    def test_response_roundtrips_through_schema(self) -> None:
        response = SpiceFrameTransformResponse(
            from_frame=_FROM,
            to_frame=_TO,
            epoch=_EPOCH,
            rotation=[QuantityVector(value=row, unit="1") for row in _ROTATION_Z90],
            position=QuantityVector(value=_ROTATED_POSITION, unit="km"),
            velocity=QuantityVector(value=_ROTATED_VELOCITY, unit="km/s"),
        )
        first = response.model_dump_json()
        rebuilt = SpiceFrameTransformResponse.model_validate_json(first)
        assert rebuilt.model_dump_json() == first

    def test_extra_keys_forbidden(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SpiceFrameTransformResponse.model_validate(
                {
                    "from_frame": _FROM,
                    "to_frame": _TO,
                    "epoch": _EPOCH,
                    "rotation": [],
                    "surprise": True,
                }
            )


# ---------------------------------------------------------------------------
# Committed reference golden
# ---------------------------------------------------------------------------


class TestReferenceGolden:
    async def test_rotation_matches_committed_golden(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_lsk_and_pck(tmp_path)
        fake_spice.plan_rotation(_FROM, _TO, _ROTATION_Z90, requires="PCK")

        response = await _do_frame_transform(
            from_frame=_FROM, to_frame=_TO, epoch=_EPOCH, state=_position_state()
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
        fresh = FastMCP("spice-frame-transform-test")
        monkeypatch.setattr("astrodynamics_mcp.server.mcp", fresh)
        spice_tools._register_spice_tools()
        return fresh

    async def test_frame_transform_round_trip(
        self, registered_mcp: FastMCP, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_lsk_and_pck(tmp_path)
        fake_spice.plan_rotation(_FROM, _TO, _ROTATION_Z90, requires="PCK")

        _, result = await registered_mcp.call_tool(
            "spice_frame_transform",
            {
                "from_frame": _FROM,
                "to_frame": _TO,
                "epoch": _EPOCH,
                "state": {"position": {"value": _POSITION, "unit": "km"}},
            },
        )
        assert isinstance(result, dict)
        assert result["from_frame"] == _FROM
        assert result["to_frame"] == _TO
        assert result["position"]["value"] == _ROTATED_POSITION
        assert result["velocity"] is None
        assert [row["value"] for row in result["rotation"]] == _ROTATION_Z90

    async def test_bare_date_epoch_rejected(
        self, registered_mcp: FastMCP, fake_spice: FakeSpice
    ) -> None:
        # The Epoch type's BeforeValidator rejects a bare date during the slot's
        # argument validation, before any CSPICE call.
        with pytest.raises(ToolError) as excinfo:
            await registered_mcp.call_tool(
                "spice_frame_transform",
                {"from_frame": _FROM, "to_frame": _TO, "epoch": "2026-01-01"},
            )
        assert "include a time component" in str(excinfo.value)
        assert fake_spice.calls["pxform"] == []
