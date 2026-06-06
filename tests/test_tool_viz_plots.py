"""Tests for the static-plot tools (plot_ground_track / plot_trajectory / plot_porkchop).

Split into two layers:

- The pure geometry / summary helpers (sub-satellite lat/lon, revolution count,
  arc length, porkchop grid reconstruction, dateline splitting) are exercised
  directly. They depend only on numpy + astropy (a base dependency), never
  matplotlib, so they run in the standard test environment and carry the real
  numeric coverage.
- The end-to-end tool bodies need matplotlib, which ships only with the ``[viz]``
  extra, so that block self-skips where matplotlib is absent (mirroring
  ``test_viz_render.py``). It is exercised in CI's ``[viz]`` extra-install job.
"""

from __future__ import annotations

import base64
import math
from typing import Any

import numpy as np
import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import CallToolResult, ImageContent, TextContent
from pydantic import BaseModel

from astrodynamics_mcp.errors import InvalidInputError
from astrodynamics_mcp.schemas.base import Frame, StateVector
from astrodynamics_mcp.tools import viz
from astrodynamics_mcp.tools.porkchop import PorkchopCell, PorkchopResponse
from astrodynamics_mcp.tools.viz import (
    GroundTrackResponse,
    PorkchopPlotResponse,
    TrajectoryResponse,
)
from astrodynamics_mcp.units import Quantity, QuantityVector

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _itrs_series(n: int = 24, inclination_deg: float = 51.6) -> list[StateVector]:
    """A simple inclined circular state series in ITRS, one point every 15 min.

    ITRS so the ground-track path needs no inertial→fixed rotation (and thus no
    IERS data) — the tests stay hermetic and byte-deterministic.
    """
    inc = math.radians(inclination_deg)
    states: list[StateVector] = []
    for i in range(n):
        ang = math.radians(i * (720.0 / n))  # two full turns over the series
        r = [
            7000.0 * math.cos(ang),
            7000.0 * math.sin(ang) * math.cos(inc),
            7000.0 * math.sin(ang) * math.sin(inc),
        ]
        hh, mm = divmod(i * 15, 60)
        states.append(
            StateVector(
                r=QuantityVector(value=r, unit="km"),
                v=QuantityVector(value=[0.0, 7.5, 0.0], unit="km/s"),
                frame=Frame.ITRS,
                epoch=f"2024-01-01T{hh:02d}:{mm:02d}:00Z",
            )
        )
    return states


def _porkchop_result(*, with_grid: bool = True, drop_one: bool = False) -> PorkchopResponse:
    """A small synthetic porkchop result with a populated (optionally holey) grid."""
    departs = ["2026-11-01T00:00:00Z", "2026-11-15T00:00:00Z", "2026-12-01T00:00:00Z"]
    arrives = ["2027-06-01T00:00:00Z", "2027-07-01T00:00:00Z", "2027-08-01T00:00:00Z"]

    def cell(d: str, a: str, c3: float, dv: float) -> PorkchopCell:
        return PorkchopCell(
            depart_epoch=d,
            arrive_epoch=a,
            tof=Quantity(value=210.0, unit="days"),
            c3=Quantity(value=c3, unit="km^2/s^2"),
            v_inf_arrival=Quantity(value=3.0, unit="km/s"),
            dec_dep_asymptote=Quantity(value=-12.0, unit="deg"),
            total_dv=Quantity(value=dv, unit="km/s"),
        )

    grid: list[PorkchopCell] = []
    for i, d in enumerate(departs):
        for j, a in enumerate(arrives):
            if drop_one and i == 1 and j == 1:
                continue  # leave one infeasible hole
            grid.append(cell(d, a, 10.0 + i + j, 6.0 + 0.1 * (i + j)))
    best = min(grid, key=lambda c: c.total_dv.value)
    return PorkchopResponse(
        best=best,
        top_cells=sorted(grid, key=lambda c: c.total_dv.value)[:5],
        grid=grid if with_grid else [],
        ascii_summary="....",
    )


# ---------------------------------------------------------------------------
# Pure helpers — no matplotlib required
# ---------------------------------------------------------------------------


class TestRequireStates:
    def test_empty_list_raises(self) -> None:
        with pytest.raises(InvalidInputError) as exc:
            viz._require_states([], minimum=2, what="plot_x")
        assert exc.value.code == "invalid_input.too_few_states"

    def test_single_state_raises_when_two_needed(self) -> None:
        with pytest.raises(InvalidInputError) as exc:
            viz._require_states(_itrs_series(1), minimum=2, what="plot_x")
        assert exc.value.code == "invalid_input.too_few_states"

    def test_two_states_pass(self) -> None:
        viz._require_states(_itrs_series(2), minimum=2, what="plot_x")  # no raise


class TestPositionsKm:
    def test_metres_input_converted_to_km(self) -> None:
        states = [
            StateVector(
                r=QuantityVector(value=[7_000_000.0, 0.0, 0.0], unit="m"),
                v=QuantityVector(value=[0.0, 7500.0, 0.0], unit="m/s"),
                frame=Frame.ITRS,
                epoch="2024-01-01T00:00:00Z",
            )
        ]
        positions = viz._positions_km(states)
        assert positions[0][0] == pytest.approx(7000.0)


class TestEarthFixedPositions:
    def test_itrs_passthrough(self) -> None:
        states = _itrs_series(3)
        out = viz._earth_fixed_positions_km(states)
        assert out.shape == (3, 3)
        # ITRS input is used directly — first row equals the declared position.
        assert out[0] == pytest.approx(states[0].r.value)

    def test_non_earth_frame_raises(self) -> None:
        s = StateVector(
            r=QuantityVector(value=[7000.0, 0.0, 0.0], unit="km"),
            v=QuantityVector(value=[0.0, 7.5, 0.0], unit="km/s"),
            frame=Frame.IAU_MARS,
            epoch="2024-01-01T00:00:00Z",
        )
        with pytest.raises(InvalidInputError) as exc:
            viz._earth_fixed_positions_km([s, s])
        assert exc.value.code == "invalid_input.non_earth_frame"

    def test_inertial_teme_is_rotated_to_itrs(self) -> None:
        """A TEME series rotates to ITRS (exercises the astropy path, needs no network)."""
        states = [
            StateVector(
                r=QuantityVector(value=[7000.0, 0.0, 0.0], unit="km"),
                v=QuantityVector(value=[0.0, 7.5, 0.0], unit="km/s"),
                frame=Frame.TEME,
                epoch="2024-06-15T08:00:00Z",
            )
        ] * 2
        out = viz._earth_fixed_positions_km(states)
        # The rotation changes the vector but preserves its magnitude.
        assert np.linalg.norm(out[0]) == pytest.approx(7000.0, abs=1e-3)
        assert out[0] != pytest.approx([7000.0, 0.0, 0.0])


class TestSubsatelliteLatLon:
    def test_equatorial_point_on_prime_meridian(self) -> None:
        # A point on the +x ITRS axis is sub-satellite at lat 0, lon 0.
        positions = np.array([[7000.0, 0.0, 0.0]])
        lat, lon = viz._subsatellite_latlon(positions)
        assert lat[0] == pytest.approx(0.0, abs=1e-6)
        assert lon[0] == pytest.approx(0.0, abs=1e-6)

    def test_point_on_plus_y_is_lon_90(self) -> None:
        positions = np.array([[0.0, 7000.0, 0.0]])
        lat, lon = viz._subsatellite_latlon(positions)
        assert lat[0] == pytest.approx(0.0, abs=1e-6)
        assert lon[0] == pytest.approx(90.0, abs=1e-6)

    def test_longitude_folded_to_signed_range(self) -> None:
        positions = np.array([[-7000.0, -1.0, 0.0]])  # just past -x → near -180
        _lat, lon = viz._subsatellite_latlon(positions)
        assert -180.0 < lon[0] <= 180.0


class TestRevolutions:
    def test_two_periods_is_two_revs(self) -> None:
        lat = np.sin(np.linspace(0.0, 4.0 * np.pi, 400))
        assert viz._count_revolutions(lat) == pytest.approx(2.0)

    def test_flat_equatorial_track_is_zero(self) -> None:
        assert viz._count_revolutions(np.zeros(10)) == 0.0

    def test_single_point_is_zero(self) -> None:
        assert viz._count_revolutions(np.array([1.0])) == 0.0


class TestArcAndSpan:
    def test_arc_length_of_right_triangle_leg(self) -> None:
        positions = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
        assert viz._arc_length_km(positions) == pytest.approx(5.0)

    def test_arc_length_single_point_is_zero(self) -> None:
        assert viz._arc_length_km(np.array([[1.0, 2.0, 3.0]])) == 0.0

    def test_time_span_hours(self) -> None:
        states = _itrs_series(5)  # 4 gaps of 15 min = 1.0 h
        assert viz._time_span_hours(states) == pytest.approx(1.0)


class TestPorkchopGridArrays:
    def test_reshape_full_grid(self) -> None:
        departs, arrives, c3 = viz._porkchop_grid_arrays(_porkchop_result())
        assert len(departs) == 3
        assert len(arrives) == 3
        assert c3.shape == (3, 3)
        assert not np.isnan(c3).any()

    def test_infeasible_cell_left_nan(self) -> None:
        _d, _a, c3 = viz._porkchop_grid_arrays(_porkchop_result(drop_one=True))
        assert np.isnan(c3).sum() == 1

    def test_empty_grid_raises(self) -> None:
        with pytest.raises(InvalidInputError) as exc:
            viz._porkchop_grid_arrays(_porkchop_result(with_grid=False))
        assert exc.value.code == "invalid_input.porkchop_grid_empty"


class TestDatelineSplit:
    def test_break_inserted_at_wrap(self) -> None:
        lon = np.array([170.0, 179.0, -179.0, -170.0])
        lat = np.array([0.0, 1.0, 2.0, 3.0])
        lon_out, lat_out = viz._split_at_dateline(lon, lat)
        # One NaN break inserted → length grows by one.
        assert lon_out.size == lon.size + 1
        assert np.isnan(lon_out).sum() == 1
        assert np.isnan(lat_out).sum() == 1

    def test_no_break_without_wrap(self) -> None:
        lon = np.array([10.0, 20.0, 30.0])
        lat = np.array([0.0, 1.0, 2.0])
        lon_out, _ = viz._split_at_dateline(lon, lat)
        assert lon_out.size == lon.size
        assert not np.isnan(lon_out).any()


class TestDaysFromFirst:
    def test_offsets_from_earliest(self) -> None:
        out = viz._days_from_first(["2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z"])
        assert out[0] == pytest.approx(0.0)
        assert out[1] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# End-to-end tool bodies — require matplotlib (the [viz] extra)
#
# The skip is gated in the fixture, not at module scope: a module-level
# importorskip would skip the pure-helper tests above too, but those depend only
# on numpy + astropy and must run in the standard (no-matplotlib) test job.
# ---------------------------------------------------------------------------


@pytest.fixture
def viz_mcp(monkeypatch: pytest.MonkeyPatch) -> FastMCP:
    """A fresh FastMCP with the viz slots registered against it.

    Drives ``_register_viz_tools`` directly (the import-time guard also needs
    gmat-czml, which the plot tools do not), so the plot bodies are reachable
    with only matplotlib present. Skips when matplotlib is absent — only the
    rendering layer needs it.
    """
    pytest.importorskip("matplotlib", reason="[viz] extra not installed")
    fresh = FastMCP("viz-plots-test")
    monkeypatch.setattr("astrodynamics_mcp.server.mcp", fresh)
    viz._register_viz_tools()
    return fresh


async def _call(mcp: FastMCP, name: str, args: dict[str, Any]) -> CallToolResult:
    """Call a tool and narrow the result to a CallToolResult for the assertions.

    ``FastMCP.call_tool`` returns a union (content sequence or dict, depending on
    ``convert_result``); the viz tools return an attachment-bearing
    CallToolResult, so narrow it once here.
    """
    result = await mcp.call_tool(name, args)
    assert isinstance(result, CallToolResult)
    return result


def _png_bytes(result: CallToolResult) -> bytes:
    """Pull the single PNG attachment's raw bytes out of a tool result."""
    images = [c for c in result.content if isinstance(c, ImageContent)]
    assert len(images) == 1, f"expected exactly one image attachment, got {len(images)}"
    return base64.b64decode(images[0].data)


def _assert_summary_first(result: CallToolResult) -> str:
    assert isinstance(result.content[0], TextContent), "ASCII summary must lead the content list"
    return result.content[0].text


def _structured(result: CallToolResult) -> dict[str, Any]:
    assert result.structuredContent is not None, "expected structuredContent on the result"
    return result.structuredContent


def _roundtrips(structured: dict[str, Any], model: type[BaseModel]) -> None:
    """The structured content must validate against and round-trip through the model."""
    first = model.model_validate(structured).model_dump_json()
    second = model.model_validate_json(first).model_dump_json()
    assert first == second


def _png_dimensions(raw: bytes) -> tuple[int, int]:
    """Read (width, height) from a PNG's IHDR chunk — bytes 16:20 and 20:24."""
    return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")


class TestPlotGroundTrackEndToEnd:
    async def test_returns_png_and_summary(self, viz_mcp: FastMCP) -> None:
        states = [s.model_dump(mode="json") for s in _itrs_series()]
        result = await _call(viz_mcp, "plot_ground_track", {"states": states})
        assert result.isError is False
        assert _png_bytes(result).startswith(_PNG_MAGIC)
        text = _assert_summary_first(result)
        assert "Ground track" in text and "revs" in text
        _roundtrips(_structured(result), GroundTrackResponse)

    async def test_image_dimensions_reported(self, viz_mcp: FastMCP) -> None:
        states = [s.model_dump(mode="json") for s in _itrs_series()]
        result = await _call(viz_mcp, "plot_ground_track", {"states": states})
        image = _structured(result)["image"]
        assert image["width_px"] == viz._WIDTH_PX
        assert image["height_px"] == viz._HEIGHT_PX
        assert image["format"] == "png"
        # The reported dimensions must match the actual rendered PNG — catches any
        # drift between the declared canvas and the renderer's DPI.
        width, height = _png_dimensions(_png_bytes(result))
        assert (width, height) == (image["width_px"], image["height_px"])

    async def test_station_overlay_accepted(self, viz_mcp: FastMCP) -> None:
        states = [s.model_dump(mode="json") for s in _itrs_series()]
        result = await _call(
            viz_mcp, "plot_ground_track", {"states": states, "stations": [{"name": "madrid"}]}
        )
        assert result.isError is False
        assert _png_bytes(result).startswith(_PNG_MAGIC)

    async def test_render_is_deterministic(self, viz_mcp: FastMCP) -> None:
        states = [s.model_dump(mode="json") for s in _itrs_series()]
        first = await _call(viz_mcp, "plot_ground_track", {"states": states})
        second = await _call(viz_mcp, "plot_ground_track", {"states": states})
        assert _png_bytes(first) == _png_bytes(second)

    async def test_empty_states_is_typed_error(self, viz_mcp: FastMCP) -> None:
        # FastMCP.call_tool (the method) re-raises the registration wrapper's
        # ToolError; the typed code rides in the message. The real server's
        # low-level handler turns this into an isError envelope on the wire —
        # that conversion is covered in test_server / test_transport_equivalence.
        with pytest.raises(ToolError) as exc:
            await viz_mcp.call_tool("plot_ground_track", {"states": []})
        assert "invalid_input.too_few_states" in str(exc.value)


class TestPlotTrajectoryEndToEnd:
    async def test_2d_returns_png_and_summary(self, viz_mcp: FastMCP) -> None:
        states = [s.model_dump(mode="json") for s in _itrs_series()]
        result = await _call(viz_mcp, "plot_trajectory", {"states": states, "projection": "2D"})
        assert result.isError is False
        assert _png_bytes(result).startswith(_PNG_MAGIC)
        assert _structured(result)["projection"] == "2D"
        _roundtrips(_structured(result), TrajectoryResponse)

    async def test_3d_returns_png(self, viz_mcp: FastMCP) -> None:
        states = [s.model_dump(mode="json") for s in _itrs_series()]
        result = await _call(
            viz_mcp,
            "plot_trajectory",
            {"states": states, "projection": "3D", "central_body": "mars"},
        )
        assert result.isError is False
        assert _png_bytes(result).startswith(_PNG_MAGIC)
        assert _structured(result)["central_body"] == "mars"

    async def test_render_is_deterministic(self, viz_mcp: FastMCP) -> None:
        states = [s.model_dump(mode="json") for s in _itrs_series()]
        first = await _call(viz_mcp, "plot_trajectory", {"states": states})
        second = await _call(viz_mcp, "plot_trajectory", {"states": states})
        assert _png_bytes(first) == _png_bytes(second)

    async def test_single_state_is_typed_error(self, viz_mcp: FastMCP) -> None:
        states = [s.model_dump(mode="json") for s in _itrs_series(1)]
        with pytest.raises(ToolError) as exc:
            await viz_mcp.call_tool("plot_trajectory", {"states": states})
        assert "invalid_input.too_few_states" in str(exc.value)


class TestPlotPorkchopEndToEnd:
    async def test_returns_png_and_summary(self, viz_mcp: FastMCP) -> None:
        result = await _call(
            viz_mcp,
            "plot_porkchop",
            {"porkchop_result": _porkchop_result().model_dump(mode="json")},
        )
        assert result.isError is False
        assert _png_bytes(result).startswith(_PNG_MAGIC)
        text = _assert_summary_first(result)
        assert "Porkchop" in text
        assert _structured(result)["feasible_cells"] == 9
        _roundtrips(_structured(result), PorkchopPlotResponse)

    async def test_holey_grid_renders(self, viz_mcp: FastMCP) -> None:
        result = await _call(
            viz_mcp,
            "plot_porkchop",
            {"porkchop_result": _porkchop_result(drop_one=True).model_dump(mode="json")},
        )
        assert result.isError is False
        assert _structured(result)["feasible_cells"] == 8

    async def test_render_is_deterministic(self, viz_mcp: FastMCP) -> None:
        payload = _porkchop_result().model_dump(mode="json")
        first = await _call(viz_mcp, "plot_porkchop", {"porkchop_result": payload})
        second = await _call(viz_mcp, "plot_porkchop", {"porkchop_result": payload})
        assert _png_bytes(first) == _png_bytes(second)

    async def test_summary_only_grid_is_typed_error(self, viz_mcp: FastMCP) -> None:
        with pytest.raises(ToolError) as exc:
            await viz_mcp.call_tool(
                "plot_porkchop",
                {"porkchop_result": _porkchop_result(with_grid=False).model_dump(mode="json")},
            )
        assert "invalid_input.porkchop_grid_empty" in str(exc.value)
