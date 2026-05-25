"""Tests for `astrodynamics_mcp.tools.propagation`.

SGP4 propagation is pure computation — no network mocking needed. Tests
drive the tool directly, plus one end-to-end roundtrip through
``mcp.call_tool``.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from astrodynamics_mcp.schemas.base import Frame, TleLines, TleOmm
from astrodynamics_mcp.server import mcp
from astrodynamics_mcp.server_lint import check_tool_descriptions
from astrodynamics_mcp.tools.propagation import Sgp4PropagateResponse, sgp4_propagate

# Fixed ISS-like TLE used across tests and the golden snapshot. Lines are
# 69 chars by construction; OMM payload mirrors the celestrak tests' sample.
_ISS_LINE1 = "1 25544U 98067A   24001.50000000  .00010000  00000-0  18000-3 0  9995"
_ISS_LINE2 = "2 25544  51.6400  90.0000 0001000  90.0000 270.0000 15.50000000    07"

_ISS_OMM: dict[str, Any] = {
    "OBJECT_NAME": "ISS (ZARYA)",
    "OBJECT_ID": "1998-067A",
    "EPOCH": "2024-01-01T12:00:00.000000",
    "MEAN_MOTION": 15.5,
    "ECCENTRICITY": 0.0001,
    "INCLINATION": 51.64,
    "RA_OF_ASC_NODE": 90.0,
    "ARG_OF_PERICENTER": 90.0,
    "MEAN_ANOMALY": 270.0,
    "EPHEMERIS_TYPE": 0,
    "CLASSIFICATION_TYPE": "U",
    "NORAD_CAT_ID": 25544,
    "ELEMENT_SET_NO": 999,
    "REV_AT_EPOCH": 0,
    "BSTAR": 0.00018,
    "MEAN_MOTION_DOT": 0.0001,
    "MEAN_MOTION_DDOT": 0.0,
}

_GOLDEN_PATH = Path(__file__).parent / "data" / "golden" / "sgp4_iss_24h.json"


@pytest.fixture(autouse=True)
def _silence_erfa_dubious_year() -> Any:
    """Astropy's ERFA layer warns on dates far from current IERS data.

    Far-future propagation tests deliberately hit those years to trigger
    SGP4 failure modes; the warning is informational and not actionable.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="ERFA function .* yielded .* dubious year")
        yield


class TestHappyPaths:
    async def test_tle_lines_input_returns_one_state(self) -> None:
        tle = TleLines(line1=_ISS_LINE1, line2=_ISS_LINE2)
        resp = await sgp4_propagate(tle=tle, epochs=["2024-01-01T12:00:00Z"], frame=Frame.TEME)
        assert isinstance(resp, Sgp4PropagateResponse)
        assert len(resp.states) == 1
        state = resp.states[0]
        assert state.frame == Frame.TEME
        assert state.epoch == "2024-01-01T12:00:00Z"
        assert state.r.unit == "km"
        assert state.v.unit == "km/s"
        assert len(state.r.value) == 3
        assert len(state.v.value) == 3
        # ISS-like orbit: |r| ≈ Earth radius + altitude (~6.8e3 km).
        r_mag = sum(x * x for x in state.r.value) ** 0.5
        assert 6500 < r_mag < 7500

    async def test_omm_input_matches_tle_lines_input(self) -> None:
        """Both TLE shapes are accepted and produce the same TEME state."""
        from_lines = await sgp4_propagate(
            tle=TleLines(line1=_ISS_LINE1, line2=_ISS_LINE2),
            epochs=["2024-01-01T13:00:00Z"],
            frame=Frame.TEME,
        )
        from_omm = await sgp4_propagate(
            tle=TleOmm(omm=_ISS_OMM),
            epochs=["2024-01-01T13:00:00Z"],
            frame=Frame.TEME,
        )
        # OMM round-trips through sgp4 with the same orbital elements, so the
        # propagated state should match to numerical noise.
        for axis in range(3):
            assert from_lines.states[0].r.value[axis] == pytest.approx(
                from_omm.states[0].r.value[axis], abs=1e-3
            )
            assert from_lines.states[0].v.value[axis] == pytest.approx(
                from_omm.states[0].v.value[axis], abs=1e-6
            )

    async def test_multi_epoch_returns_matching_count(self) -> None:
        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        epochs = [
            (start + timedelta(minutes=5 * i)).isoformat().replace("+00:00", "Z") for i in range(12)
        ]
        resp = await sgp4_propagate(
            tle=TleLines(line1=_ISS_LINE1, line2=_ISS_LINE2),
            epochs=epochs,
            frame=Frame.TEME,
        )
        assert len(resp.states) == 12
        # Epochs preserve order.
        assert [s.epoch for s in resp.states] == epochs


class TestFrameConversion:
    """Each non-TEME frame goes through astropy with per-epoch obstime."""

    EPOCH = "2024-06-15T08:00:00Z"

    @pytest.mark.parametrize("frame", [Frame.ICRF, Frame.GCRS, Frame.ITRS, Frame.CIRS])
    async def test_frame_tag_propagates_to_output(self, frame: Frame) -> None:
        resp = await sgp4_propagate(
            tle=TleLines(line1=_ISS_LINE1, line2=_ISS_LINE2),
            epochs=[self.EPOCH],
            frame=frame,
        )
        assert resp.states[0].frame == frame

    async def test_tool_icrf_matches_external_astropy_path(self) -> None:
        """Tool's TEME→ICRF must agree with the same astropy transform done out-of-band."""
        import astropy.units as u
        from astropy.coordinates import (
            ICRS,
            TEME,
            CartesianDifferential,
            CartesianRepresentation,
        )
        from astropy.time import Time

        epoch = self.EPOCH
        # Tool's TEME state at the epoch.
        teme_resp = await sgp4_propagate(
            tle=TleLines(line1=_ISS_LINE1, line2=_ISS_LINE2),
            epochs=[epoch],
            frame=Frame.TEME,
        )
        # Tool's ICRF state at the same epoch.
        icrf_resp = await sgp4_propagate(
            tle=TleLines(line1=_ISS_LINE1, line2=_ISS_LINE2),
            epochs=[epoch],
            frame=Frame.ICRF,
        )

        # External astropy reference transform from the tool's TEME output.
        r_teme = teme_resp.states[0].r.value
        v_teme = teme_resp.states[0].v.value
        t = Time(epoch, scale="utc")
        rep = CartesianRepresentation(r_teme[0] * u.km, r_teme[1] * u.km, r_teme[2] * u.km)
        diff = CartesianDifferential(
            v_teme[0] * u.km / u.s, v_teme[1] * u.km / u.s, v_teme[2] * u.km / u.s
        )
        out = TEME(rep.with_differentials(diff), obstime=t).transform_to(ICRS())
        ref_r = out.cartesian.xyz.to_value(u.km)
        ref_v = out.cartesian.differentials["s"].d_xyz.to_value(u.km / u.s)

        for axis in range(3):
            assert icrf_resp.states[0].r.value[axis] == pytest.approx(float(ref_r[axis]), abs=1e-3)
            assert icrf_resp.states[0].v.value[axis] == pytest.approx(float(ref_v[axis]), abs=1e-6)


class TestUnsupportedFrame:
    @pytest.mark.parametrize("frame", [Frame.TIRS, Frame.IAU_EARTH, Frame.IAU_MARS, Frame.IAU_MOON])
    async def test_unsupported_frame_raises_typed_envelope(self, frame: Frame) -> None:
        """Frames outside the 5-frame v0.1 cap surface as InvalidInputError."""
        with pytest.raises(ToolError) as excinfo:
            await sgp4_propagate(
                tle=TleLines(line1=_ISS_LINE1, line2=_ISS_LINE2),
                epochs=["2024-01-01T12:00:00Z"],
                frame=frame,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.unsupported_frame"
        assert "frame_transform" in envelope["message"]


class TestSgp4FailureModes:
    async def test_propagation_far_past_epoch_raises_sgp4_failure(self) -> None:
        """Propagating ISS to 2300 triggers SGP4 error code 1 (eccentricity OOR)."""
        with pytest.raises(ToolError) as excinfo:
            await sgp4_propagate(
                tle=TleLines(line1=_ISS_LINE1, line2=_ISS_LINE2),
                epochs=["2300-01-01T00:00:00Z"],
                frame=Frame.TEME,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "upstream.sgp4_failure"
        assert envelope["data"]["sgp4_error_code"] != 0
        assert envelope["data"]["epoch"] == "2300-01-01T00:00:00Z"

    async def test_omm_with_eccentricity_above_one_fails_at_init(self) -> None:
        bad = dict(_ISS_OMM)
        bad["ECCENTRICITY"] = 1.5
        with pytest.raises(ToolError) as excinfo:
            await sgp4_propagate(
                tle=TleOmm(omm=bad),
                epochs=["2024-01-01T12:00:00Z"],
                frame=Frame.TEME,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "upstream.sgp4_failure"
        assert envelope["data"]["sgp4_error_code"] != 0
        # Init failure → no "epoch" key (we haven't propagated yet).
        assert "epoch" not in envelope["data"]

    async def test_omm_missing_required_fields_wraps_keyerror(self) -> None:
        """An OMM that drops a load-bearing field → KeyError → upstream.sgp4_failure."""
        thin = {"OBJECT_NAME": "STUB"}  # nothing else; sgp4.omm.initialize raises KeyError
        with pytest.raises(ToolError) as excinfo:
            await sgp4_propagate(
                tle=TleOmm(omm=thin),
                epochs=["2024-01-01T12:00:00Z"],
                frame=Frame.TEME,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "upstream.sgp4_failure"


class TestRegistration:
    async def test_tool_is_listed(self) -> None:
        tools = await mcp.list_tools()
        names = {t.name for t in tools}
        assert "sgp4_propagate" in names

    async def test_tool_description_passes_lint(self) -> None:
        tools = await mcp.list_tools()
        violations = [v for v in check_tool_descriptions(tools) if v.tool_name == "sgp4_propagate"]
        assert violations == []

    async def test_tool_callable_via_mcp(self) -> None:
        content, structured = await mcp.call_tool(
            "sgp4_propagate",
            {
                "tle": {"line1": _ISS_LINE1, "line2": _ISS_LINE2},
                "epochs": ["2024-01-01T12:00:00Z"],
                "frame": "TEME",
            },
        )
        del content
        assert isinstance(structured, dict)
        assert "states" in structured
        assert len(structured["states"]) == 1
        assert structured["states"][0]["frame"] == "TEME"


class TestGoldenSnapshot:
    """ISS propagated 24h at 10-min steps must match the committed golden.

    Tolerance is 1e-6 km / 1e-9 km/s per the issue's acceptance criterion.
    The golden was generated from the deterministic `_ISS_LINE1` / `_ISS_LINE2`
    inputs; regenerate via a one-liner script and review the diff deliberately
    if a sgp4 minor-version bump shifts numerics.
    """

    async def test_iss_24h_matches_golden(self) -> None:
        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        epochs = [
            (start + timedelta(minutes=10 * i)).isoformat().replace("+00:00", "Z")
            for i in range(145)
        ]
        resp = await sgp4_propagate(
            tle=TleLines(line1=_ISS_LINE1, line2=_ISS_LINE2),
            epochs=epochs,
            frame=Frame.TEME,
        )

        golden = json.loads(_GOLDEN_PATH.read_text())
        assert len(resp.states) == len(golden["states"])
        for live, expected in zip(resp.states, golden["states"], strict=True):
            assert live.epoch == expected["epoch"]
            assert live.frame.value == expected["frame"]
            assert live.r.unit == expected["r"]["unit"]
            assert live.v.unit == expected["v"]["unit"]
            for axis in range(3):
                assert live.r.value[axis] == pytest.approx(expected["r"]["value"][axis], abs=1e-6)
                assert live.v.value[axis] == pytest.approx(expected["v"]["value"][axis], abs=1e-9)


class TestSchemaInvariants:
    async def test_response_round_trips_through_json(self) -> None:
        resp = await sgp4_propagate(
            tle=TleLines(line1=_ISS_LINE1, line2=_ISS_LINE2),
            epochs=["2024-01-01T12:00:00Z"],
            frame=Frame.TEME,
        )
        as_json = resp.model_dump_json()
        rebuilt = Sgp4PropagateResponse.model_validate_json(as_json)
        assert rebuilt == resp
