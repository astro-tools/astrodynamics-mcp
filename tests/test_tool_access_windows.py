"""Tests for `astrodynamics_mcp.tools.access`.

Skyfield runs deterministically against a fixed TLE — no network mocking.
Tests cover the Madrid+ISS smoke path (proxy for the issue's Hubble
acceptance, since both are LEO with regular passes), the elevation/range
filters, observer-shape dispatch (NamedStation vs ObserverCoordinates),
OMM-input route, error paths, and end-to-end MCP invocation.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from astrodynamics_mcp.errors import InvalidInputError
from astrodynamics_mcp.schemas.base import (
    NamedStation,
    ObserverCoordinates,
    TleLines,
    TleOmm,
)
from astrodynamics_mcp.server import mcp
from astrodynamics_mcp.server_lint import check_tool_descriptions
from astrodynamics_mcp.tools.access import (
    _STATION_COORDS,
    AccessWindow,
    AccessWindowsResponse,
    _resolve_observer,
    access_windows,
)
from astrodynamics_mcp.units import Quantity

# Fixed ISS-like TLE — same lines used across the SGP4 tests, deterministic
# for ground-station passes.
_ISS_LINE1 = "1 25544U 98067A   24001.50000000  .00010000  00000-0  18000-3 0  9995"
_ISS_LINE2 = "2 25544  51.6400  90.0000 0001000  90.0000 270.0000 15.50000000    07"
_ISS_TLE = TleLines(line1=_ISS_LINE1, line2=_ISS_LINE2)

_WINDOW_START = "2024-01-01T00:00:00Z"
_WINDOW_END = "2024-01-02T00:00:00Z"

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
    "ELEMENT_SET_NO": 0,
    "REV_AT_EPOCH": 0,
    "BSTAR": 0.00018,
    "MEAN_MOTION_DOT": 0.0001,
    "MEAN_MOTION_DDOT": 0.0,
}


class TestHappyPath:
    """Madrid + ISS over 24 h — covers the Hubble-shaped acceptance criterion."""

    async def test_madrid_finds_passes_with_valid_aos_peak_los(self) -> None:
        """At least one pass with AOS < peak < LOS and positive peak elevation."""
        resp = await access_windows(
            observer=NamedStation(name="madrid"),
            target_tle=_ISS_TLE,
            start=_WINDOW_START,
            end=_WINDOW_END,
            min_elevation_deg=10.0,
        )
        assert isinstance(resp, AccessWindowsResponse)
        assert len(resp.windows) >= 1
        for window in resp.windows:
            assert isinstance(window, AccessWindow)
            assert window.aos < window.peak_elevation_time < window.los
            assert window.peak_elevation.value > 10.0  # at or above the mask
            assert window.peak_elevation.unit == "deg"
            assert window.range_at_aos.unit == "km"
            assert window.range_at_peak.unit == "km"
            assert window.range_at_los.unit == "km"
            assert window.duration.unit == "s"
            assert window.duration.value > 0

    async def test_peak_range_is_at_or_below_aos_and_los_range(self) -> None:
        """Peak elevation is closest approach — range at peak should be the minimum."""
        resp = await access_windows(
            observer=NamedStation(name="madrid"),
            target_tle=_ISS_TLE,
            start=_WINDOW_START,
            end=_WINDOW_END,
            min_elevation_deg=10.0,
        )
        for window in resp.windows:
            assert window.range_at_peak.value <= window.range_at_aos.value
            assert window.range_at_peak.value <= window.range_at_los.value


class TestObserverDispatch:
    """NamedStation vs ObserverCoordinates input must produce the same windows."""

    @staticmethod
    def _madrid_coords() -> ObserverCoordinates:
        lat, lon, alt_km = _STATION_COORDS["madrid"]
        return ObserverCoordinates(
            lat=Quantity(value=lat, unit="deg"),
            lon=Quantity(value=lon, unit="deg"),
            alt=Quantity(value=alt_km, unit="km"),
        )

    async def test_observer_coordinates_route_matches_named_station(self) -> None:
        from_name = await access_windows(
            observer=NamedStation(name="madrid"),
            target_tle=_ISS_TLE,
            start=_WINDOW_START,
            end=_WINDOW_END,
            min_elevation_deg=10.0,
        )
        from_coords = await access_windows(
            observer=self._madrid_coords(),
            target_tle=_ISS_TLE,
            start=_WINDOW_START,
            end=_WINDOW_END,
            min_elevation_deg=10.0,
        )
        assert len(from_name.windows) == len(from_coords.windows)
        for n, c in zip(from_name.windows, from_coords.windows, strict=True):
            assert n.aos == c.aos
            assert n.los == c.los

    async def test_observer_coordinates_with_metres_alt(self) -> None:
        """Length unit `m` must convert to km internally without changing results."""
        lat, lon, alt_km = _STATION_COORDS["madrid"]
        from_metres = await access_windows(
            observer=ObserverCoordinates(
                lat=Quantity(value=lat, unit="deg"),
                lon=Quantity(value=lon, unit="deg"),
                alt=Quantity(value=alt_km * 1000.0, unit="m"),
            ),
            target_tle=_ISS_TLE,
            start=_WINDOW_START,
            end=_WINDOW_END,
            min_elevation_deg=10.0,
        )
        from_km = await access_windows(
            observer=ObserverCoordinates(
                lat=Quantity(value=lat, unit="deg"),
                lon=Quantity(value=lon, unit="deg"),
                alt=Quantity(value=alt_km, unit="km"),
            ),
            target_tle=_ISS_TLE,
            start=_WINDOW_START,
            end=_WINDOW_END,
            min_elevation_deg=10.0,
        )
        assert len(from_metres.windows) == len(from_km.windows)


class TestTleInputDispatch:
    async def test_omm_input_matches_tle_lines_input(self) -> None:
        from_lines = await access_windows(
            observer=NamedStation(name="madrid"),
            target_tle=_ISS_TLE,
            start=_WINDOW_START,
            end=_WINDOW_END,
            min_elevation_deg=10.0,
        )
        from_omm = await access_windows(
            observer=NamedStation(name="madrid"),
            target_tle=TleOmm(omm=_ISS_OMM),
            start=_WINDOW_START,
            end=_WINDOW_END,
            min_elevation_deg=10.0,
        )
        assert len(from_lines.windows) == len(from_omm.windows)


class TestElevationFilter:
    async def test_min_elevation_90_yields_empty_windows(self) -> None:
        """Acceptance: min_elevation_deg=90 returns an empty list, not an error."""
        resp = await access_windows(
            observer=NamedStation(name="madrid"),
            target_tle=_ISS_TLE,
            start=_WINDOW_START,
            end=_WINDOW_END,
            min_elevation_deg=90.0,
        )
        assert resp.windows == []

    async def test_low_elevation_returns_more_passes_than_high(self) -> None:
        """Loosening the elevation mask must monotonically yield more (or equal) passes."""
        low = await access_windows(
            observer=NamedStation(name="madrid"),
            target_tle=_ISS_TLE,
            start=_WINDOW_START,
            end=_WINDOW_END,
            min_elevation_deg=5.0,
        )
        high = await access_windows(
            observer=NamedStation(name="madrid"),
            target_tle=_ISS_TLE,
            start=_WINDOW_START,
            end=_WINDOW_END,
            min_elevation_deg=30.0,
        )
        assert len(low.windows) >= len(high.windows)


class TestRangeFilter:
    async def test_max_range_km_drops_apogee_passes(self) -> None:
        """A tight max_range_km strictly drops the higher-altitude passes.

        10°-mask ISS passes from Madrid run roughly 600 km to 1800 km at
        peak elevation; capping at 700 km strictly excludes the apogee
        cluster, exercising the ``continue`` skip in the range filter.
        """
        unfiltered = await access_windows(
            observer=NamedStation(name="madrid"),
            target_tle=_ISS_TLE,
            start=_WINDOW_START,
            end=_WINDOW_END,
            min_elevation_deg=10.0,
        )
        filtered = await access_windows(
            observer=NamedStation(name="madrid"),
            target_tle=_ISS_TLE,
            start=_WINDOW_START,
            end=_WINDOW_END,
            min_elevation_deg=10.0,
            max_range_km=700.0,
        )
        assert len(filtered.windows) < len(unfiltered.windows)
        for window in filtered.windows:
            assert window.range_at_peak.value <= 700.0

    async def test_min_range_km_drops_close_passes(self) -> None:
        # Pick a min that excludes everything — the ISS rarely passes overhead
        # within 100 km of Madrid because of orbital inclination.
        resp = await access_windows(
            observer=NamedStation(name="madrid"),
            target_tle=_ISS_TLE,
            start=_WINDOW_START,
            end=_WINDOW_END,
            min_elevation_deg=10.0,
            min_range_km=100000.0,  # 100,000 km — absurd, excludes everything
        )
        assert resp.windows == []


class TestErrorPaths:
    async def test_unknown_station_raises_typed_envelope(self) -> None:
        """A NamedStation pointing at a name absent from the runtime registry."""
        # Pydantic's Literal validation normally prevents this — bypass via
        # model_construct so the registry-miss code path is exercised.
        rogue = NamedStation.model_construct(name="atlantis")
        with pytest.raises(ToolError) as excinfo:
            await access_windows(
                observer=rogue,
                target_tle=_ISS_TLE,
                start=_WINDOW_START,
                end=_WINDOW_END,
                min_elevation_deg=10.0,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.unknown_station"
        # Message lists the canonical set.
        for name in _STATION_COORDS:
            assert name in envelope["message"]

    async def test_negative_elevation_raises_typed_envelope(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await access_windows(
                observer=NamedStation(name="madrid"),
                target_tle=_ISS_TLE,
                start=_WINDOW_START,
                end=_WINDOW_END,
                min_elevation_deg=-1.0,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.elevation_out_of_range"

    async def test_elevation_above_90_returns_empty_not_error(self) -> None:
        # Boundary check: 90 → empty; > 90 also → empty (it's physically
        # impossible but should not error).
        resp = await access_windows(
            observer=NamedStation(name="madrid"),
            target_tle=_ISS_TLE,
            start=_WINDOW_START,
            end=_WINDOW_END,
            min_elevation_deg=120.0,
        )
        assert resp.windows == []

    async def test_negative_range_filter_raises_typed_envelope(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await access_windows(
                observer=NamedStation(name="madrid"),
                target_tle=_ISS_TLE,
                start=_WINDOW_START,
                end=_WINDOW_END,
                min_elevation_deg=10.0,
                max_range_km=-1.0,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.range_filter_invalid"

    async def test_start_after_end_raises_typed_envelope(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await access_windows(
                observer=NamedStation(name="madrid"),
                target_tle=_ISS_TLE,
                start=_WINDOW_END,
                end=_WINDOW_START,
                min_elevation_deg=10.0,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.interval_end_not_after_start"

    async def test_broken_omm_raises_upstream_envelope(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await access_windows(
                observer=NamedStation(name="madrid"),
                target_tle=TleOmm(omm={"OBJECT_NAME": "X"}),  # missing nearly everything
                start=_WINDOW_START,
                end=_WINDOW_END,
                min_elevation_deg=10.0,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "upstream.sgp4_failure"

    async def test_non_number_min_elevation_raises_typed_envelope(self) -> None:
        """A string elevation trips the type-guard before the range check."""
        with pytest.raises(ToolError) as excinfo:
            await access_windows(
                observer=NamedStation(name="madrid"),
                target_tle=_ISS_TLE,
                start=_WINDOW_START,
                end=_WINDOW_END,
                min_elevation_deg="high",  # type: ignore[arg-type]
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.value_not_a_number"

    async def test_skyfield_find_events_internal_exception_wrapped(self) -> None:
        """An exception inside skyfield's find_events surfaces as upstream.sgp4_failure.

        skyfield occasionally raises a bare Exception on degenerate TLEs
        that pass `Satrec` init but blow up during pass enumeration. The
        path is hard to trip with real geometry, so we patch the method.
        """
        from unittest.mock import patch

        def fail_find_events(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("synthetic skyfield failure inside find_events")

        with (
            patch(
                "skyfield.sgp4lib.EarthSatellite.find_events",
                side_effect=fail_find_events,
            ),
            pytest.raises(ToolError) as excinfo,
        ):
            await access_windows(
                observer=NamedStation(name="madrid"),
                target_tle=_ISS_TLE,
                start=_WINDOW_START,
                end=_WINDOW_END,
                min_elevation_deg=10.0,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "upstream.sgp4_failure"


class TestRegistry:
    """The named-station registry must match the wire-side NamedStationName Literal."""

    def test_resolve_observer_named_station(self) -> None:
        lat, lon, alt_km = _resolve_observer(NamedStation(name="madrid"))
        assert (lat, lon, alt_km) == _STATION_COORDS["madrid"]

    def test_resolve_unknown_station_raises_invalid_input_error(self) -> None:
        rogue = NamedStation.model_construct(name="nowhere")
        with pytest.raises(InvalidInputError) as excinfo:
            _resolve_observer(rogue)
        assert excinfo.value.code == "invalid_input.unknown_station"


class TestRegistration:
    EXPECTED_NAMES: ClassVar[set[str]] = {
        "madrid",
        "goldstone",
        "canberra",
        "svalbard",
        "wallops",
        "esrange",
        "gsfc",
        "jpl",
    }

    def test_registry_covers_all_named_stations(self) -> None:
        assert set(_STATION_COORDS) == self.EXPECTED_NAMES

    async def test_tool_is_listed(self) -> None:
        tools = await mcp.list_tools()
        names = {t.name for t in tools}
        assert "access_windows" in names

    async def test_tool_description_passes_lint(self) -> None:
        tools = await mcp.list_tools()
        violations = [v for v in check_tool_descriptions(tools) if v.tool_name == "access_windows"]
        assert violations == []

    async def test_tool_callable_via_mcp(self) -> None:
        content, structured = await mcp.call_tool(
            "access_windows",
            {
                "observer": {"name": "madrid"},
                "target_tle": {"line1": _ISS_LINE1, "line2": _ISS_LINE2},
                "start": _WINDOW_START,
                "end": _WINDOW_END,
                "min_elevation_deg": 10.0,
            },
        )
        del content
        assert isinstance(structured, dict)
        assert "windows" in structured
        assert len(structured["windows"]) >= 1


class TestSchemaInvariants:
    async def test_response_round_trips_through_json(self) -> None:
        resp = await access_windows(
            observer=NamedStation(name="madrid"),
            target_tle=_ISS_TLE,
            start=_WINDOW_START,
            end=_WINDOW_END,
            min_elevation_deg=10.0,
        )
        as_json = resp.model_dump_json()
        rebuilt = AccessWindowsResponse.model_validate_json(as_json)
        assert rebuilt == resp


class TestGroupedTriplesHelper:
    """Direct tests for the `_grouped_triples` partial-pass dropper.

    skyfield can emit interleaved event codes (0/1/2 = rise/culminate/set)
    where a window edge leaves a partial sequence — e.g. a `set` event
    without a prior `rise`, or two adjacent `set` events. The helper
    resets its buffer in those cases so only complete passes propagate.
    """

    def test_out_of_order_set_event_resets_buffer(self) -> None:
        """A LOS event arriving without a prior peak resets the buffer."""
        from astrodynamics_mcp.tools.access import _grouped_triples

        # Clean triple [0, 1, 2] followed by an out-of-order [2, 1, 2]:
        # the 4th element (a second LOS) is not preceded by a peak, so
        # the partial buffer is dropped.
        events = [0, 1, 2, 2, 1, 2]
        times = list(range(6))
        triples = _grouped_triples(times, events)
        assert len(triples) == 1
        assert triples[0] == (0, 1, 2)

    def test_peak_without_rise_resets_buffer(self) -> None:
        """A culmination without a preceding rise is dropped."""
        from astrodynamics_mcp.tools.access import _grouped_triples

        # [1] alone (peak without rise) → buffer reset; no triples.
        events = [1, 2]
        times = list(range(2))
        assert _grouped_triples(times, events) == []
