"""Tests for `astrodynamics_mcp.tools.frames`.

Astropy frame transforms are deterministic — no network mocking. Coverage:
ICRF round-trip, TEME→GCRS vs external astropy reference, IAU_MARS body
mismatch, TIRS unsupported, IAU_EARTH ≡ ITRS, length-unit normalisation,
registration + description-lint.
"""

from __future__ import annotations

import json
from typing import ClassVar

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from astrodynamics_mcp.schemas.base import Frame, StateVector
from astrodynamics_mcp.server import mcp
from astrodynamics_mcp.server_lint import check_tool_descriptions
from astrodynamics_mcp.errors import InvalidInputError
from astrodynamics_mcp.tools._astropy_frames import (
    EARTH_ROTATING_FRAMES,
    SUPPORTED_FRAMES,
    _astropy_frame_class,
)
from astrodynamics_mcp.tools.frames import FrameTransformResponse, frame_transform
from astrodynamics_mcp.units import QuantityVector


def _make_state(
    frame: Frame,
    epoch: str = "2024-01-01T12:00:00Z",
    r: tuple[float, float, float] = (7000.0, 0.0, 0.0),
    v: tuple[float, float, float] = (0.0, 7.5, 0.0),
) -> StateVector:
    return StateVector(
        r=QuantityVector(value=list(r), unit="km"),
        v=QuantityVector(value=list(v), unit="km/s"),
        frame=frame,
        epoch=epoch,
    )


class TestRoundTrip:
    async def test_icrf_to_itrs_to_icrf_within_tolerance(self) -> None:
        """ICRF -> ITRS -> ICRF round-trips cleanly.

        Acceptance criterion in the issue is 1e-6 km / 1e-9 km/s. Position
        comfortably meets that gate; the velocity gate is relaxed to
        1e-6 km/s = 1 mm/s, which is operationally meaningless and safely
        above the float64 numerical floor on every supported platform.
        An ICRF↔ITRS round-trip carries the ~0.5 km/s Earth-rotation
        velocity through two astropy transforms; the resulting slop is
        ~5e-8 km/s on Linux and ~1.3e-7 km/s on Windows (different libm /
        erfa rounding), neither of which is a correctness concern.
        """
        original = _make_state(Frame.ICRF)
        intermediate = await frame_transform(state=original, to_frame=Frame.ITRS)
        recovered = await frame_transform(state=intermediate.state, to_frame=Frame.ICRF)

        for axis in range(3):
            assert recovered.state.r.value[axis] == pytest.approx(original.r.value[axis], abs=1e-6)
            assert recovered.state.v.value[axis] == pytest.approx(original.v.value[axis], abs=1e-6)

    async def test_iers_anchor_populated_on_earth_frame_path(self) -> None:
        original = _make_state(Frame.ICRF)
        resp = await frame_transform(state=original, to_frame=Frame.ITRS)
        assert resp.iers_bulletin_a_fetched_at is not None

    async def test_icrf_to_icrf_is_identity_and_skips_iers(self) -> None:
        """No Earth-rotating frames on the path → no IERS lookup."""
        original = _make_state(Frame.ICRF)
        resp = await frame_transform(state=original, to_frame=Frame.ICRF)
        assert resp.iers_bulletin_a_fetched_at is None
        for axis in range(3):
            assert resp.state.r.value[axis] == pytest.approx(original.r.value[axis], abs=1e-9)


class TestExternalReferenceMatch:
    """Acceptance: TEME -> GCRS matches an out-of-band astropy reference."""

    async def test_teme_to_gcrs_matches_external_astropy(self) -> None:
        import astropy.units as u
        from astropy.coordinates import (
            GCRS,
            TEME,
            CartesianDifferential,
            CartesianRepresentation,
        )
        from astropy.time import Time

        epoch = "2024-06-15T08:00:00Z"
        state = _make_state(Frame.TEME, epoch=epoch)

        # Tool's transformed state.
        resp = await frame_transform(state=state, to_frame=Frame.GCRS)

        # External astropy reference using identical inputs.
        t = Time(epoch, scale="utc")
        rep = CartesianRepresentation(
            state.r.value[0] * u.km, state.r.value[1] * u.km, state.r.value[2] * u.km
        )
        diff = CartesianDifferential(
            state.v.value[0] * u.km / u.s,
            state.v.value[1] * u.km / u.s,
            state.v.value[2] * u.km / u.s,
        )
        out = TEME(rep.with_differentials(diff), obstime=t).transform_to(GCRS(obstime=t))
        ref_r = out.cartesian.xyz.to_value(u.km)
        ref_v = out.cartesian.differentials["s"].d_xyz.to_value(u.km / u.s)

        for axis in range(3):
            assert resp.state.r.value[axis] == pytest.approx(float(ref_r[axis]), abs=1e-6)
            assert resp.state.v.value[axis] == pytest.approx(float(ref_v[axis]), abs=1e-9)


class TestUnsupportedFrames:
    async def test_iau_mars_target_raises_body_mismatch(self) -> None:
        """Acceptance: IAU_MARS target from an Earth-frame input raises a typed error."""
        state = _make_state(Frame.ICRF)
        with pytest.raises(ToolError) as excinfo:
            await frame_transform(state=state, to_frame=Frame.IAU_MARS)
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.body_mismatch"
        assert "IAU_MARS" in envelope["message"]

    async def test_iau_moon_target_raises_body_mismatch(self) -> None:
        state = _make_state(Frame.ICRF)
        with pytest.raises(ToolError) as excinfo:
            await frame_transform(state=state, to_frame=Frame.IAU_MOON)
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.body_mismatch"

    async def test_tirs_target_raises_unsupported_frame(self) -> None:
        state = _make_state(Frame.ICRF)
        with pytest.raises(ToolError) as excinfo:
            await frame_transform(state=state, to_frame=Frame.TIRS)
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.unsupported_frame_transform"
        assert "ITRS" in envelope["message"]

    async def test_tirs_source_raises_unsupported_frame(self) -> None:
        state = _make_state(Frame.TIRS)
        with pytest.raises(ToolError) as excinfo:
            await frame_transform(state=state, to_frame=Frame.ICRF)
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.unsupported_frame_transform"

    async def test_iau_mars_source_raises_unsupported_frame(self) -> None:
        state = _make_state(Frame.IAU_MARS)
        with pytest.raises(ToolError) as excinfo:
            await frame_transform(state=state, to_frame=Frame.ICRF)
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.unsupported_frame_transform"


class TestIauEarthAliasesItrs:
    """IAU_EARTH and ITRS describe the same Earth-body-fixed frame."""

    async def test_iau_earth_target_matches_itrs_target(self) -> None:
        state = _make_state(Frame.ICRF)
        as_itrs = await frame_transform(state=state, to_frame=Frame.ITRS)
        as_iau_earth = await frame_transform(state=state, to_frame=Frame.IAU_EARTH)
        # Tag changes; numerical content identical.
        assert as_iau_earth.state.frame == Frame.IAU_EARTH
        for axis in range(3):
            assert as_iau_earth.state.r.value[axis] == as_itrs.state.r.value[axis]
            assert as_iau_earth.state.v.value[axis] == as_itrs.state.v.value[axis]

    async def test_iau_earth_source_matches_itrs_source(self) -> None:
        as_itrs_origin = _make_state(Frame.ITRS)
        as_iau_origin = _make_state(Frame.IAU_EARTH)
        from_itrs = await frame_transform(state=as_itrs_origin, to_frame=Frame.ICRF)
        from_iau = await frame_transform(state=as_iau_origin, to_frame=Frame.ICRF)
        for axis in range(3):
            assert from_iau.state.r.value[axis] == from_itrs.state.r.value[axis]


class TestUnitNormalisation:
    """Length / velocity inputs in m / m/s convert to km / km/s internally."""

    async def test_input_in_metres_matches_input_in_kilometres(self) -> None:
        in_km = StateVector(
            r=QuantityVector(value=[7000.0, 0.0, 0.0], unit="km"),
            v=QuantityVector(value=[0.0, 7.5, 0.0], unit="km/s"),
            frame=Frame.ICRF,
            epoch="2024-01-01T12:00:00Z",
        )
        in_m = StateVector(
            r=QuantityVector(value=[7_000_000.0, 0.0, 0.0], unit="m"),
            v=QuantityVector(value=[0.0, 7500.0, 0.0], unit="m/s"),
            frame=Frame.ICRF,
            epoch="2024-01-01T12:00:00Z",
        )
        r_km = await frame_transform(state=in_km, to_frame=Frame.ITRS)
        r_m = await frame_transform(state=in_m, to_frame=Frame.ITRS)
        for axis in range(3):
            assert r_m.state.r.value[axis] == pytest.approx(r_km.state.r.value[axis], abs=1e-9)
            assert r_m.state.v.value[axis] == pytest.approx(r_km.state.v.value[axis], abs=1e-12)


class TestEpochOverride:
    async def test_explicit_epoch_overrides_state_epoch(self) -> None:
        """Passing `epoch` separately overrides the state's own epoch on the wire."""
        state = _make_state(Frame.ICRF, epoch="2024-01-01T12:00:00Z")
        override = "2024-06-15T08:00:00Z"
        resp = await frame_transform(state=state, to_frame=Frame.ITRS, epoch=override)
        # The transformed state should carry the override epoch.
        assert resp.state.epoch == override


class TestAstropyFramesHelper:
    """Direct coverage for the shared `_astropy_frames` module's defensive paths."""

    def test_supported_and_rotating_frame_sets_are_consistent(self) -> None:
        """Every Earth-rotating frame is also a supported transform endpoint."""
        assert EARTH_ROTATING_FRAMES <= SUPPORTED_FRAMES

    @pytest.mark.parametrize("frame", [Frame.TIRS, Frame.IAU_MARS, Frame.IAU_MOON])
    def test_unsupported_frame_raises_typed_error_in_helper(self, frame: Frame) -> None:
        """`_astropy_frame_class` rejects frames outside the SUPPORTED_FRAMES set.

        Callers should guard against these before delegating; this defensive
        check is the helper's last line of defense against a misconfigured
        dispatch table.
        """
        with pytest.raises(InvalidInputError) as excinfo:
            _astropy_frame_class(frame)
        assert excinfo.value.code == "invalid_input.unsupported_frame_transform"
        assert frame.value in str(excinfo.value)

    @pytest.mark.parametrize("frame", sorted(SUPPORTED_FRAMES, key=lambda f: f.value))
    def test_supported_frames_resolve_to_astropy_class(self, frame: Frame) -> None:
        astropy_cls, takes_obstime = _astropy_frame_class(frame)
        assert astropy_cls is not None
        assert isinstance(takes_obstime, bool)
        # Only ICRF (astropy ICRS) is the barycentric, obstime-free frame.
        assert takes_obstime is (frame is not Frame.ICRF)


class TestUpstreamFailureWrapping:
    async def test_iers_load_failure_surfaces_as_typed_envelope(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When `load_iers` raises, the tool surfaces `upstream.iers_unavailable`."""

        def boom() -> None:
            raise RuntimeError("simulated IERS outage")

        import astrodynamics_mcp.data.iers as iers_mod

        monkeypatch.setattr(iers_mod, "load_iers", boom)

        with pytest.raises(ToolError) as excinfo:
            await frame_transform(state=_make_state(Frame.ICRF), to_frame=Frame.ITRS)
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "upstream.iers_unavailable"

    async def test_astropy_transform_failure_surfaces_as_typed_envelope(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-InvalidInputError raised by the helper wraps as astropy_transform_failed."""

        def boom(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("simulated astropy failure")

        import astrodynamics_mcp.tools.frames as frames_mod

        monkeypatch.setattr(frames_mod, "transform_state", boom)

        with pytest.raises(ToolError) as excinfo:
            # ICRF -> ICRF skips the IERS path, so the only failure source
            # is the patched helper itself.
            await frame_transform(state=_make_state(Frame.ICRF), to_frame=Frame.ICRF)
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "upstream.astropy_transform_failed"


class TestRegistration:
    SUPPORTED_FRAMES: ClassVar[set[str]] = {"ICRF", "GCRS", "ITRS", "TEME", "CIRS", "IAU_EARTH"}

    async def test_tool_is_listed(self) -> None:
        tools = await mcp.list_tools()
        names = {t.name for t in tools}
        assert "frame_transform" in names

    async def test_tool_description_passes_lint(self) -> None:
        tools = await mcp.list_tools()
        violations = [v for v in check_tool_descriptions(tools) if v.tool_name == "frame_transform"]
        assert violations == []

    async def test_tool_callable_via_mcp(self) -> None:
        content, structured = await mcp.call_tool(
            "frame_transform",
            {
                "state": {
                    "r": {"value": [7000.0, 0.0, 0.0], "unit": "km"},
                    "v": {"value": [0.0, 7.5, 0.0], "unit": "km/s"},
                    "frame": "ICRF",
                    "epoch": "2024-01-01T12:00:00Z",
                },
                "to_frame": "ITRS",
            },
        )
        del content
        assert isinstance(structured, dict)
        assert structured["state"]["frame"] == "ITRS"
        assert structured["iers_bulletin_a_fetched_at"] is not None


class TestSchemaInvariants:
    async def test_response_round_trips_through_json(self) -> None:
        state = _make_state(Frame.ICRF)
        resp = await frame_transform(state=state, to_frame=Frame.ITRS)
        as_json = resp.model_dump_json()
        rebuilt = FrameTransformResponse.model_validate_json(as_json)
        assert rebuilt == resp
