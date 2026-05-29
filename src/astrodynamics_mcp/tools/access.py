"""`access_windows` tool — ground-station / observer access intervals via Skyfield.

Wraps `skyfield`'s `find_events` and `altaz` machinery to enumerate the
times at which a TLE-defined satellite is visible above a horizon mask
from a fixed observer (named station from the closed registry or explicit
geodetic coordinates). Per-window altitude and range are reported at AOS,
peak, and LOS so downstream filters (range bounds) and LLM consumers see
the full pass geometry.

Skyfield's bundled timescale data (leap seconds, deltaT) is sufficient
here — we do not share astropy's IERS cache. The IERS shim in
``astrodynamics_mcp.data.iers`` is for tools that touch UT1 or
EOP-dependent frame transforms, which access windows do not.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, get_args

from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from astrodynamics_mcp.errors import InvalidInputError, UpstreamError
from astrodynamics_mcp.schemas.base import (
    Epoch,
    NamedStation,
    NamedStationName,
    ObserverCoordinates,
    TleLines,
    TleOmm,
)
from astrodynamics_mcp.server import register_tool
from astrodynamics_mcp.units import Quantity

# Geodetic coordinates (lat_deg, lon_deg, alt_km) for the named-station
# registry. Values are approximate civilian-published coordinates for the
# primary site each name refers to (DSS-63 at Madrid, DSS-14 at Goldstone,
# DSS-43 at Canberra, SvalSat antenna farm, etc.) — accurate enough for
# access-pass geometry to the second.
_STATION_COORDS: dict[str, tuple[float, float, float]] = {
    "madrid": (40.4256, -4.2503, 0.834),
    "goldstone": (35.4267, -116.8900, 1.036),
    "canberra": (-35.4014, 148.9819, 0.700),
    "svalbard": (78.2300, 15.4070, 0.501),
    "wallops": (37.9333, -75.4667, 0.013),
    "esrange": (67.8847, 21.1063, 0.302),
    "gsfc": (38.9967, -76.8483, 0.054),
    "jpl": (34.2014, -118.1717, 0.345),
}
# Schema-registry parity. `NamedStationName` is the Literal source of truth
# for valid station names at the wire boundary; the runtime registry below
# must cover every value exactly. Mismatches fail on import rather than at
# tool-call time.
assert set(_STATION_COORDS) == set(get_args(NamedStationName)), (
    "Station-name registry diverged from NamedStationName Literal: "
    f"missing={set(get_args(NamedStationName)) - set(_STATION_COORDS)}, "
    f"extra={set(_STATION_COORDS) - set(get_args(NamedStationName))}"
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class AccessWindow(BaseModel):
    """A single ground-station pass: AOS to LOS with peak-elevation details."""

    model_config = ConfigDict(extra="forbid")

    aos: Epoch = Field(..., description="Acquisition-of-signal epoch (UTC ISO 8601).")
    los: Epoch = Field(..., description="Loss-of-signal epoch (UTC ISO 8601).")
    peak_elevation_time: Epoch = Field(
        ..., description="Epoch of maximum elevation during the pass (UTC ISO 8601)."
    )
    peak_elevation: Quantity = Field(
        ...,
        description="Maximum elevation above the local horizon during the pass (deg).",
        examples=[{"value": 35.0, "unit": "deg"}],
    )
    range_at_aos: Quantity = Field(
        ...,
        description=(
            "Observer-to-satellite range at AOS (km). Always at the horizon-mask elevation."
        ),
        examples=[{"value": 1483.0, "unit": "km"}],
    )
    range_at_peak: Quantity = Field(
        ...,
        description=(
            "Observer-to-satellite range at peak elevation (km). Typically the closest "
            "approach during the pass; used by the `min_range_km` / `max_range_km` filters."
        ),
        examples=[{"value": 600.0, "unit": "km"}],
    )
    range_at_los: Quantity = Field(
        ...,
        description=(
            "Observer-to-satellite range at LOS (km). Always at the horizon-mask elevation."
        ),
        examples=[{"value": 1505.0, "unit": "km"}],
    )
    duration: Quantity = Field(
        ...,
        description="Time between AOS and LOS, in seconds.",
        examples=[{"value": 480.0, "unit": "s"}],
    )


class AccessWindowsResponse(BaseModel):
    """List of `AccessWindow`s for the requested observer and time range."""

    model_config = ConfigDict(extra="forbid")

    windows: list[AccessWindow] = Field(
        ...,
        description=(
            "Complete (AOS, peak, LOS) passes inside the requested window. Passes that "
            "begin before `start` or end after `end` are omitted — only complete "
            "triples are emitted. Range-filtered passes are also omitted."
        ),
    )


# ---------------------------------------------------------------------------
# Tool description (subject to server_lint)
# ---------------------------------------------------------------------------

_DESCRIPTION = (
    "Compute ground-station / observer access intervals for a TLE-defined satellite "
    "over a UTC time window. e.g. access_windows(observer={'name': 'madrid'}, "
    "target_tle={'line1': '...', 'line2': '...'}, start='2024-01-01T00:00:00Z', "
    "end='2024-01-02T00:00:00Z', min_elevation_deg=10) returns the satellite's passes "
    "above 10° as seen from Madrid. `observer` is either a named-station dict "
    "({'name': 'madrid' | 'goldstone' | 'canberra' | 'svalbard' | 'wallops' | "
    "'esrange' | 'gsfc' | 'jpl'}) or explicit geodetic coordinates "
    "({lat, lon, alt}). Epochs are UTC ISO 8601 with a mandatory time component "
    "('2024-01-01T00:00:00Z'). `min_elevation_deg` is in degrees, not radians; "
    "below 5° passes are usually operationally useless thanks to horizon refraction "
    "and terrain masking — 10° is the standard amateur threshold, 15° for DSN-style "
    "large-dish operations. Optional `min_range_km` / `max_range_km` filter on the "
    "satellite's range at peak elevation (closest approach during the pass). Frame "
    "is implicit: ECEF observer + TEME-derived satellite position, transformed "
    "internally via skyfield."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_LENGTH_KM: dict[str, float] = {"km": 1.0, "m": 1e-3, "AU": 149597870.7}


def _resolve_observer(
    observer: NamedStation | ObserverCoordinates,
) -> tuple[float, float, float]:
    """Return (lat_deg, lon_deg, alt_km) for the given observer."""
    if isinstance(observer, NamedStation):
        name = observer.name
        if name not in _STATION_COORDS:
            raise InvalidInputError(
                f"unknown station {name!r}; supported: {sorted(_STATION_COORDS)}",
                code="invalid_input.unknown_station",
            )
        return _STATION_COORDS[name]
    # ObserverCoordinates carries Quantity-wrapped lat/lon/alt. Schema-level
    # validators already constrained units; lat/lon are deg, alt is a length.
    lat_deg = float(observer.lat.value)
    lon_deg = float(observer.lon.value)
    alt_km = float(observer.alt.value) * _LENGTH_KM[observer.alt.unit]
    return lat_deg, lon_deg, alt_km


def _build_earth_satellite(tle: TleLines | TleOmm, ts: Any, earth_satellite_cls: Any) -> Any:
    """Construct a Skyfield EarthSatellite from either TLE shape."""
    try:
        if isinstance(tle, TleLines):
            return earth_satellite_cls(tle.line1, tle.line2, "target", ts)
        return earth_satellite_cls.from_omm(ts, tle.omm)
    except (KeyError, TypeError, ValueError, AssertionError) as exc:
        raise UpstreamError(
            f"failed to build satellite from TLE input: {exc}",
            code="upstream.sgp4_failure",
            original_exception=exc,
        ) from exc


def _parse_epoch_to_datetime(epoch: str) -> datetime:
    """Parse our Epoch (ISO 8601 with `Z` or `+HH:MM`) into a tz-aware datetime."""
    return datetime.fromisoformat(epoch.replace("Z", "+00:00"))


def _grouped_triples(times: Any, events: Any) -> list[tuple[Any, Any, Any]]:
    """Group skyfield find_events output into complete (rise, culminate, set) triples.

    skyfield interleaves event codes 0 / 1 / 2 (rise / culminate / set) but
    can emit partial sequences at window edges — e.g. starting with a `set`
    if the satellite was already up at the window's start. Only complete
    passes are emitted; partial passes are dropped silently.
    """
    triples: list[tuple[Any, Any, Any]] = []
    buffer: list[tuple[Any, int]] = []
    for time_obj, event in zip(times, events, strict=True):
        event_code = int(event)
        if event_code == 0:
            buffer = [(time_obj, event_code)]
        elif event_code == 1 and len(buffer) == 1 and buffer[0][1] == 0:
            buffer.append((time_obj, event_code))
        elif event_code == 2 and len(buffer) == 2 and buffer[1][1] == 1:
            triples.append((buffer[0][0], buffer[1][0], time_obj))
            buffer = []
        else:
            buffer = []
    return triples


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@register_tool(
    name="access_windows",
    description=_DESCRIPTION,
    annotations=ToolAnnotations(
        title="Ground-Station Access Windows", readOnlyHint=True, openWorldHint=False
    ),
)
async def access_windows(
    observer: Annotated[
        NamedStation | ObserverCoordinates,
        Field(
            description=(
                "Ground station / observer location. Either a NamedStation "
                "({name: ...}) where name is one of "
                "'madrid', 'goldstone', 'canberra', 'svalbard', 'wallops', "
                "'esrange', 'gsfc', 'jpl' — resolved to lat/lon/alt against "
                "the v0.1 registry — or explicit ObserverCoordinates "
                "({lat_deg, lon_deg, height_km}) for arbitrary sites."
            ),
        ),
    ],
    target_tle: Annotated[
        TleLines | TleOmm,
        Field(
            description=(
                "The satellite to track, supplied as a TLE line pair "
                "({line1, line2}) or as an OMM payload (the CCSDS-standard JSON "
                "returned by tle_lookup). Either form is accepted."
            ),
        ),
    ],
    start: Annotated[
        Epoch,
        Field(
            description=(
                "Start of the search window, UTC ISO 8601 with a time component "
                "(e.g. '2024-01-01T00:00:00Z')."
            ),
        ),
    ],
    end: Annotated[
        Epoch,
        Field(
            description=(
                "End of the search window, UTC ISO 8601 with a time component. "
                "Must be strictly after `start`."
            ),
        ),
    ],
    min_elevation_deg: Annotated[
        float,
        Field(
            description=(
                "Horizon mask, in degrees (not radians). Passes peaking below "
                "this elevation are filtered out. Conventional thresholds: 10° "
                "for amateur ground stations, 15° for DSN-style large-dish ops; "
                "values below 5° are usually noisy due to refraction and "
                "terrain. Must be in [0, 90]."
            ),
        ),
    ],
    min_range_km: Annotated[
        float | None,
        Field(
            description=(
                "Optional minimum range from observer to satellite at the pass "
                "peak (km). Passes whose closest approach is nearer than this "
                "are dropped. Useful for excluding very-low-altitude horizon "
                "noise."
            ),
        ),
    ] = None,
    max_range_km: Annotated[
        float | None,
        Field(
            description=(
                "Optional maximum range from observer to satellite at the pass "
                "peak (km). Passes whose closest approach is farther than this "
                "are dropped."
            ),
        ),
    ] = None,
) -> AccessWindowsResponse:
    # Input validation.
    if isinstance(min_elevation_deg, bool) or not isinstance(min_elevation_deg, (int, float)):
        raise InvalidInputError(
            f"min_elevation_deg must be a number, got {type(min_elevation_deg).__name__}",
            code="invalid_input.value_not_a_number",
        )
    if not 0.0 <= float(min_elevation_deg) <= 90.0:
        # The acceptance criterion says elevation=90 must yield an empty list,
        # not an error — so 90 is permitted at the boundary.
        if float(min_elevation_deg) > 90.0:
            return AccessWindowsResponse(windows=[])
        raise InvalidInputError(
            f"min_elevation_deg must be in [0, 90] degrees, got {min_elevation_deg}",
            code="invalid_input.elevation_out_of_range",
        )

    for name, value in (("min_range_km", min_range_km), ("max_range_km", max_range_km)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise InvalidInputError(
                f"{name} must be a non-negative number when supplied, got {value!r}",
                code="invalid_input.range_filter_invalid",
            )

    if start >= end:
        raise InvalidInputError(
            f"`start` must be strictly before `end`, got {start!r} >= {end!r}",
            code="invalid_input.interval_end_not_after_start",
        )

    lat_deg, lon_deg, alt_km = _resolve_observer(observer)

    # Lazy skyfield import — keeps the package-level boot under the
    # multi-process cache test's 10 s subprocess-spawn budget.
    from skyfield.api import EarthSatellite, load, wgs84

    ts = load.timescale()
    topos = wgs84.latlon(lat_deg, lon_deg, elevation_m=alt_km * 1000.0)
    sat = _build_earth_satellite(target_tle, ts, EarthSatellite)

    t_start = ts.from_datetime(_parse_epoch_to_datetime(start))
    t_end = ts.from_datetime(_parse_epoch_to_datetime(end))

    try:
        times, events = sat.find_events(
            topos, t_start, t_end, altitude_degrees=float(min_elevation_deg)
        )
    except Exception as exc:
        # skyfield occasionally surfaces a bare `Exception` on degenerate
        # TLE inputs that pass init but blow up inside `find_events`.
        raise UpstreamError(
            f"skyfield find_events failed: {exc}",
            code="upstream.sgp4_failure",
            original_exception=exc,
        ) from exc

    triples = _grouped_triples(times, events)
    difference = sat - topos

    windows: list[AccessWindow] = []
    for rise_t, culm_t, set_t in triples:
        peak_alt, _, peak_dist = difference.at(culm_t).altaz()
        _, _, aos_dist = difference.at(rise_t).altaz()
        _, _, los_dist = difference.at(set_t).altaz()

        range_peak_km = float(peak_dist.km)
        if min_range_km is not None and range_peak_km < float(min_range_km):
            continue
        if max_range_km is not None and range_peak_km > float(max_range_km):
            continue

        duration_s = (set_t.utc_datetime() - rise_t.utc_datetime()).total_seconds()

        windows.append(
            AccessWindow(
                aos=rise_t.utc_iso(),
                los=set_t.utc_iso(),
                peak_elevation_time=culm_t.utc_iso(),
                peak_elevation=Quantity(value=float(peak_alt.degrees), unit="deg"),
                range_at_aos=Quantity(value=float(aos_dist.km), unit="km"),
                range_at_peak=Quantity(value=range_peak_km, unit="km"),
                range_at_los=Quantity(value=float(los_dist.km), unit="km"),
                duration=Quantity(value=duration_s, unit="s"),
            )
        )

    return AccessWindowsResponse(windows=windows)
