"""Visualisation tool slots — registered only when the ``[viz]`` extra is present.

The visualisation surface ships behind the optional ``[viz]`` extra (matplotlib
for the static raster plots, gmat-czml for the CZML 3D view). When both are
importable the four ``plot_* / czml_*`` tool slots register; on a bare install
they are absent and the rest of the tool surface is unaffected — the same gate
the ``[gmat]`` and ``[spice]`` tools use.

The three static-plot tools (``plot_ground_track`` / ``plot_trajectory`` /
``plot_porkchop``) each return a deterministic PNG carried *alongside* a
structured summary model and a leading ASCII text block, via
:mod:`astrodynamics_mcp.attachments`. ``czml_trajectory`` is still a
``NotImplementedError`` placeholder — it graduates in the CZML follow-up the way
the GMAT and SPICE slots graduated one at a time.

Import-cost discipline: this module is imported unconditionally by
:mod:`astrodynamics_mcp.tools` (the registration only happens behind the guard),
so the response models and the pure geometry helpers stay matplotlib-free at
module scope — every matplotlib import lives inside a figure-builder function,
and astropy is imported lazily too. Importing this module on a base install is
free and never pulls the plotting stack in.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from astrodynamics_mcp.attachments import png_image_content, tool_result_with_attachments
from astrodynamics_mcp.errors import InvalidInputError
from astrodynamics_mcp.schemas.base import Epoch, Frame, Observer, StateVector, _epoch_to_instant
from astrodynamics_mcp.server import register_tool
from astrodynamics_mcp.tools.porkchop import PorkchopResponse
from astrodynamics_mcp.units import Quantity
from astrodynamics_mcp.viz_render import render_png, use_agg_backend

if TYPE_CHECKING:
    from matplotlib.figure import Figure

try:
    import gmat_czml  # noqa: F401  # availability probe; the symbol itself isn't used here
    import matplotlib  # noqa: F401  # availability probe; imported for real lazily in bodies

    _VIZ_AVAILABLE = True
except ImportError:
    _VIZ_AVAILABLE = False


# Earth-fixed (body-rotating) frames a ground track can use directly without a
# frame rotation. The inertial Earth frames (TEME / ICRF / GCRS / CIRS) are
# rotated into ITRS per-epoch via the shared astropy helper; the non-Earth IAU
# frames and TIRS are rejected — a ground track is Earth-specific.
_EARTH_FIXED_FRAMES: frozenset[Frame] = frozenset({Frame.ITRS, Frame.IAU_EARTH})
_INERTIAL_EARTH_FRAMES: frozenset[Frame] = frozenset(
    {Frame.TEME, Frame.ICRF, Frame.GCRS, Frame.CIRS}
)

# Equatorial radii (km) for the central bodies the trajectory plot can draw to
# scale. An unknown body is still plotted — the origin just gets a marker, not a
# filled disc. Values are mean-equatorial; the plot is schematic, not metric.
_BODY_RADII_KM: dict[str, float] = {
    "earth": 6378.137,
    "moon": 1737.4,
    "mars": 3396.2,
    "sun": 695700.0,
    "venus": 6051.8,
    "mercury": 2439.7,
    "jupiter": 71492.0,
}

# Fixed render canvas. The figure size in inches times the DPI fixes the PNG
# pixel grid, which the byte-for-byte determinism contract (stdio == HTTP)
# depends on. DPI is pinned inside viz_render.render_png; the pixel dimensions
# reported in the summary are derived from these so a client knows the image
# size without decoding it.
_FIGSIZE_IN: tuple[float, float] = (8.0, 4.0)
_RENDER_DPI = 100
_WIDTH_PX = round(_FIGSIZE_IN[0] * _RENDER_DPI)
_HEIGHT_PX = round(_FIGSIZE_IN[1] * _RENDER_DPI)


# ---------------------------------------------------------------------------
# Response models — structured summaries carried alongside the PNG attachment
# ---------------------------------------------------------------------------


class VizPlaceholderResult(BaseModel):
    """Reserved output shape for the not-yet-implemented ``czml_trajectory`` slot.

    The slot declares this return type only so the server can derive an output
    schema for it; the body raises ``NotImplementedError`` and never constructs
    one. The CZML follow-up replaces this with the tool's real response model — a
    structured summary carried alongside the CZML attachment via
    :mod:`astrodynamics_mcp.attachments`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    detail: str = Field(
        ...,
        description="Placeholder marker; never populated at runtime.",
    )


class PngImageInfo(BaseModel):
    """Pixel dimensions of the attached PNG.

    These are rendering *cardinalities*, not physical quantities, so they sit
    outside the ``{value, unit}`` envelope and are declared exempt where the
    unit-discipline meta-test polices the attachment-bearing schemas. The image
    bytes themselves ride as a separate ``ImageContent`` block, not in this
    summary.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    width_px: int = Field(..., description="PNG width in pixels (a rendering cardinality).")
    height_px: int = Field(..., description="PNG height in pixels (a rendering cardinality).")
    format: Literal["png"] = Field(
        default="png",
        description="Attachment image format; always 'png' for the static-plot tools.",
    )


class GroundTrackResponse(BaseModel):
    """Structured summary accompanying a ``plot_ground_track`` PNG.

    The PNG is additive: this summary is always present and is what a text-only
    client reads. Every physical field carries an explicit unit; ``revolutions``
    is dimensionless (unit ``"1"``).
    """

    model_config = ConfigDict(extra="forbid")

    revolutions: Quantity = Field(
        ...,
        description=(
            "Number of orbital revolutions the track spans, derived as equator "
            "crossings / 2. Dimensionless (unit '1')."
        ),
        examples=[{"value": 15.5, "unit": "1"}],
    )
    lat_min: Quantity = Field(
        ...,
        description="Minimum sub-satellite geodetic latitude, deg.",
        examples=[{"value": -51.6, "unit": "deg"}],
    )
    lat_max: Quantity = Field(
        ...,
        description="Maximum sub-satellite geodetic latitude, deg.",
        examples=[{"value": 51.6, "unit": "deg"}],
    )
    lon_min: Quantity = Field(
        ...,
        description="Minimum sub-satellite longitude (east-positive), deg.",
        examples=[{"value": -179.4, "unit": "deg"}],
    )
    lon_max: Quantity = Field(
        ...,
        description="Maximum sub-satellite longitude (east-positive), deg.",
        examples=[{"value": 178.9, "unit": "deg"}],
    )
    image: PngImageInfo = Field(..., description="Pixel dimensions of the attached PNG.")


class TrajectoryResponse(BaseModel):
    """Structured summary accompanying a ``plot_trajectory`` PNG."""

    model_config = ConfigDict(extra="forbid")

    arc_length: Quantity = Field(
        ...,
        description=(
            "Path length of the plotted state series, summed over straight "
            "segments between consecutive states, km."
        ),
        examples=[{"value": 41600.0, "unit": "km"}],
    )
    periapsis_radius: Quantity = Field(
        ...,
        description="Minimum position magnitude |r| over the series, km.",
        examples=[{"value": 6678.0, "unit": "km"}],
    )
    apoapsis_radius: Quantity = Field(
        ...,
        description="Maximum position magnitude |r| over the series, km.",
        examples=[{"value": 42164.0, "unit": "km"}],
    )
    time_span: Quantity = Field(
        ...,
        description="Wall-clock span from the first to the last state epoch, hours.",
        examples=[{"value": 5.27, "unit": "hours"}],
    )
    projection: Literal["2D", "3D"] = Field(
        ..., description="Projection the PNG was rendered with."
    )
    central_body: str = Field(
        ..., description="Central body the trajectory was drawn about (echo of the input)."
    )
    image: PngImageInfo = Field(..., description="Pixel dimensions of the attached PNG.")


class PorkchopPlotResponse(BaseModel):
    """Structured summary accompanying a ``plot_porkchop`` PNG.

    Echoes the minimum-Δv 'best' cell of the supplied porkchop grid and the
    grid's shape. ``feasible_cells`` / ``n_depart_samples`` / ``n_arrive_samples``
    are grid cardinalities (counts, not physical quantities) and sit outside the
    ``{value, unit}`` envelope.
    """

    model_config = ConfigDict(extra="forbid")

    best_c3: Quantity = Field(
        ...,
        description="Departure C3 of the minimum-total_dv cell, km^2/s^2.",
        examples=[{"value": 12.5, "unit": "km^2/s^2"}],
    )
    best_total_dv: Quantity = Field(
        ...,
        description="Two-impulse total-Δv proxy of the best cell, km/s.",
        examples=[{"value": 6.4, "unit": "km/s"}],
    )
    best_tof: Quantity = Field(
        ...,
        description="Time of flight of the best cell, days.",
        examples=[{"value": 210.0, "unit": "days"}],
    )
    best_depart_epoch: Epoch = Field(
        ..., description="Departure epoch of the best cell (UTC ISO 8601)."
    )
    best_arrive_epoch: Epoch = Field(
        ..., description="Arrival epoch of the best cell (UTC ISO 8601)."
    )
    feasible_cells: int = Field(
        ..., description="Number of feasible cells in the supplied grid (a cardinality)."
    )
    n_depart_samples: int = Field(
        ..., description="Distinct departure epochs across the grid (a cardinality)."
    )
    n_arrive_samples: int = Field(
        ..., description="Distinct arrival epochs across the grid (a cardinality)."
    )
    image: PngImageInfo = Field(..., description="Pixel dimensions of the attached PNG.")


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------


def _require_states(states: list[StateVector], *, minimum: int, what: str) -> None:
    """Reject an empty or too-short state series with a typed error."""
    if not isinstance(states, list) or len(states) < minimum:
        raise InvalidInputError(
            f"{what} needs at least {minimum} state{'s' if minimum != 1 else ''}, "
            f"got {len(states) if isinstance(states, list) else type(states).__name__}",
            code="invalid_input.too_few_states",
        )


def _positions_km(states: list[StateVector]) -> np.ndarray:
    """Stack the position vectors as an (N, 3) array in km.

    StateVector enforces a length unit (km / m / AU) on ``r``; convert to km so
    the plotted axes and the summary radii share one unit regardless of how the
    caller declared the input.
    """
    rows: list[list[float]] = []
    for state in states:
        scale = _LENGTH_TO_KM[state.r.unit]
        rows.append([component * scale for component in state.r.value])
    return np.asarray(rows, dtype=float)


# StateVector.r is validated to one of these length units; convert to km.
_LENGTH_TO_KM: dict[str, float] = {"km": 1.0, "m": 1.0e-3, "AU": 149597870.7}


# ---------------------------------------------------------------------------
# Ground-track geometry (pure; astropy imported lazily)
# ---------------------------------------------------------------------------


def _earth_fixed_positions_km(states: list[StateVector]) -> np.ndarray:
    """Return each state's position in Earth-fixed (ITRS) km, as an (N, 3) array.

    States already in an Earth-fixed frame are used directly; inertial Earth
    frames are rotated to ITRS per-epoch via the shared astropy helper (so the
    sub-satellite longitude tracks Earth's rotation correctly). Non-Earth frames
    (IAU_MARS / IAU_MOON) and TIRS are rejected — a ground track is Earth-fixed
    by construction.
    """
    from astrodynamics_mcp.tools._astropy_frames import transform_state

    rows: list[list[float]] = []
    for state in states:
        frame = state.frame
        scale = _LENGTH_TO_KM[state.r.unit]
        r_km = [component * scale for component in state.r.value]
        if frame in _EARTH_FIXED_FRAMES:
            rows.append(r_km)
        elif frame in _INERTIAL_EARTH_FRAMES:
            from astropy.time import Time

            v_scale = _VELOCITY_TO_KMPS[state.v.unit]
            v_kmps = [component * v_scale for component in state.v.value]
            r_out, _ = transform_state(
                r_km,
                v_kmps,
                from_frame=frame,
                to_frame=Frame.ITRS,
                epoch_time=Time(state.epoch, scale="utc"),
            )
            rows.append(r_out)
        else:
            raise InvalidInputError(
                f"ground track needs an Earth frame; state {state.epoch} is in "
                f"{frame.value!r}. Use TEME / ICRF / GCRS / CIRS / ITRS — for an "
                "Earth-fixed series call sgp4_propagate(frame='ITRS').",
                code="invalid_input.non_earth_frame",
            )
    return np.asarray(rows, dtype=float)


_VELOCITY_TO_KMPS: dict[str, float] = {"km/s": 1.0, "m/s": 1.0e-3}


def _subsatellite_latlon(positions_ecef_km: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Geodetic sub-satellite (lat, lon) in degrees for ITRS positions.

    Uses ``astropy.EarthLocation.from_geocentric`` (WGS-84 ellipsoid) for the
    geodetic latitude — pure ellipsoid geometry, no IERS / Earth-orientation
    data involved. Longitude is east-positive, wrapped to (-180, 180].
    """
    import astropy.units as u
    from astropy.coordinates import EarthLocation

    location = EarthLocation.from_geocentric(
        positions_ecef_km[:, 0] * u.km,
        positions_ecef_km[:, 1] * u.km,
        positions_ecef_km[:, 2] * u.km,
    )
    geodetic = location.geodetic
    lat = np.asarray(geodetic.lat.to_value(u.deg), dtype=float)
    lon = np.asarray(geodetic.lon.to_value(u.deg), dtype=float)
    # astropy returns longitude in [0, 360); fold to (-180, 180].
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lat, lon


def _count_revolutions(lat: np.ndarray) -> float:
    """Revolutions spanned by a ground track = equator crossings / 2.

    Counts sign changes in latitude (each ascending or descending equator
    crossing) and halves them — two crossings per revolution. Samples that land
    exactly on the equator are treated as the side they came from, so a track
    that merely grazes lat=0 does not over-count.
    """
    if lat.size < 2:
        return 0.0
    signs = np.sign(lat)
    # Carry the previous non-zero sign across exact-zero samples.
    nonzero = signs != 0
    if not nonzero.any():
        return 0.0
    last = 0.0
    filled = np.empty_like(signs)
    for i, (s, nz) in enumerate(zip(signs, nonzero, strict=True)):
        last = float(s) if nz else last
        filled[i] = last
    crossings = int(np.count_nonzero(np.diff(filled) != 0))
    return crossings / 2.0


# ---------------------------------------------------------------------------
# Trajectory geometry (pure)
# ---------------------------------------------------------------------------


def _arc_length_km(positions_km: np.ndarray) -> float:
    """Sum of straight-segment lengths between consecutive positions, km."""
    if positions_km.shape[0] < 2:
        return 0.0
    segments = np.diff(positions_km, axis=0)
    return float(np.sum(np.linalg.norm(segments, axis=1)))


def _time_span_hours(states: list[StateVector]) -> float:
    """Wall-clock span from the first to the last state epoch, hours."""
    first = _epoch_to_instant(states[0].epoch)
    last = _epoch_to_instant(states[-1].epoch)
    return abs((last - first).total_seconds()) / 3600.0


# ---------------------------------------------------------------------------
# Porkchop grid reconstruction (pure)
# ---------------------------------------------------------------------------


def _porkchop_grid_arrays(
    result: PorkchopResponse,
) -> tuple[list[str], list[str], np.ndarray]:
    """Reconstruct (depart_epochs, arrive_epochs, c3_grid) from a porkchop result.

    The porkchop response carries feasible cells as a flat list, each tagged
    with its depart / arrive epoch; infeasible cells are simply absent. Rebuild
    the rectangular grid by taking the sorted distinct depart / arrive epochs as
    the axes and filling a (n_arrive, n_depart) C3 array, leaving missing
    (infeasible) cells as NaN so the contour leaves them blank.
    """
    if not result.grid:
        raise InvalidInputError(
            "porkchop_result.grid is empty — plot_porkchop needs the full grid. "
            "Re-run porkchop with output='full' and pass that result.",
            code="invalid_input.porkchop_grid_empty",
        )
    departs = sorted({cell.depart_epoch for cell in result.grid})
    arrives = sorted({cell.arrive_epoch for cell in result.grid})
    depart_index = {epoch: i for i, epoch in enumerate(departs)}
    arrive_index = {epoch: i for i, epoch in enumerate(arrives)}
    c3 = np.full((len(arrives), len(departs)), np.nan, dtype=float)
    for cell in result.grid:
        c3[arrive_index[cell.arrive_epoch], depart_index[cell.depart_epoch]] = cell.c3.value
    return departs, arrives, c3


def _days_from_first(epochs: list[str]) -> np.ndarray:
    """Day offsets of each epoch from the earliest, as a float axis.

    Plotting against numeric day-offsets (rather than datetime objects) keeps
    the rendered axes locale- and timezone-independent, which the byte-for-byte
    determinism contract needs.
    """
    instants = [_epoch_to_instant(e) for e in epochs]
    base = instants[0]
    return np.asarray([(t - base).total_seconds() / 86400.0 for t in instants], dtype=float)


# ---------------------------------------------------------------------------
# Figure builders (matplotlib imported lazily; never at module scope)
# ---------------------------------------------------------------------------


def _new_figure() -> Figure:
    """Construct a fixed-size headless figure for deterministic rendering."""
    from matplotlib.figure import Figure as MplFigure

    use_agg_backend()
    return MplFigure(figsize=_FIGSIZE_IN)


def _split_at_dateline(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaN breaks where the longitude wraps the ±180° dateline.

    A raw lon/lat line plot draws a spurious horizontal streak across the map
    whenever the track crosses the antimeridian. Splitting the polyline at those
    jumps (|Δlon| > 180°) keeps each pass a separate stroke.
    """
    lon_out: list[float] = [float(lon[0])]
    lat_out: list[float] = [float(lat[0])]
    for i in range(1, lon.size):
        if abs(lon[i] - lon[i - 1]) > 180.0:
            lon_out.append(math.nan)
            lat_out.append(math.nan)
        lon_out.append(float(lon[i]))
        lat_out.append(float(lat[i]))
    return np.asarray(lon_out), np.asarray(lat_out)


def _render_ground_track(
    lat: np.ndarray, lon: np.ndarray, stations: list[tuple[float, float, str]]
) -> bytes:
    """Render a sub-satellite ground track over a lat/lon graticule to PNG bytes."""
    figure = _new_figure()
    axes = figure.add_subplot(111)
    axes.set_xlim(-180.0, 180.0)
    axes.set_ylim(-90.0, 90.0)
    axes.set_xticks(range(-180, 181, 60))
    axes.set_yticks(range(-90, 91, 30))
    axes.grid(True, linewidth=0.4, color="0.8")
    axes.axhline(0.0, color="0.6", linewidth=0.6)  # equator
    axes.set_xlabel("Longitude (deg, east-positive)")
    axes.set_ylabel("Latitude (deg)")
    axes.set_title("Sub-satellite ground track")

    lon_split, lat_split = _split_at_dateline(lon, lat)
    axes.plot(lon_split, lat_split, color="C0", linewidth=1.2)
    axes.plot(lon[0], lat[0], marker="o", color="C2", markersize=5)  # start
    axes.plot(lon[-1], lat[-1], marker="s", color="C3", markersize=5)  # end
    for st_lat, st_lon, name in stations:
        axes.plot(st_lon, st_lat, marker="^", color="C1", markersize=7)
        axes.annotate(name, (st_lon, st_lat), textcoords="offset points", xytext=(4, 4), fontsize=7)

    png = render_png(figure)
    _close(figure)
    return png


def _render_trajectory(
    positions_km: np.ndarray, projection: Literal["2D", "3D"], central_body: str
) -> bytes:
    """Render an orbit / transfer arc (2D or 3D) about the central body to PNG bytes."""
    figure = _new_figure()
    body = central_body.lower()
    radius = _BODY_RADII_KM.get(body)

    if projection == "3D":
        axes = figure.add_subplot(111, projection="3d")
        axes.plot(positions_km[:, 0], positions_km[:, 1], positions_km[:, 2], color="C0")
        axes.scatter([0.0], [0.0], [0.0], color="C1", s=30)
        axes.set_xlabel("x (km)")
        axes.set_ylabel("y (km)")
        axes.set_zlabel("z (km)")
        axes.view_init(elev=25.0, azim=-60.0)  # fixed view → deterministic
    else:
        axes = figure.add_subplot(111)
        axes.plot(positions_km[:, 0], positions_km[:, 1], color="C0")
        if radius is not None:
            axes.add_patch(_circle(radius))
        else:
            axes.plot(0.0, 0.0, marker="o", color="C1", markersize=6)
        axes.set_xlabel("x (km)")
        axes.set_ylabel("y (km)")
        axes.set_aspect("equal", adjustable="datalim")
    axes.plot(positions_km[0, 0], positions_km[0, 1], marker="o", color="C2", markersize=5)
    axes.set_title(f"Trajectory about {central_body} ({projection})")

    png = render_png(figure)
    _close(figure)
    return png


def _render_porkchop(
    depart_days: np.ndarray,
    arrive_days: np.ndarray,
    c3_grid: np.ndarray,
    depart_labels: list[str],
    arrive_labels: list[str],
    best_depart_day: float,
    best_arrive_day: float,
) -> bytes:
    """Render a filled C3 contour over the (depart, arrive) grid to PNG bytes."""
    figure = _new_figure()
    axes = figure.add_subplot(111)
    masked = np.ma.masked_invalid(c3_grid)
    mesh_x, mesh_y = np.meshgrid(depart_days, arrive_days)
    contour = axes.contourf(mesh_x, mesh_y, masked, levels=12, cmap="viridis")
    figure.colorbar(contour, ax=axes, label="C3 (km^2/s^2)")
    axes.plot(best_depart_day, best_arrive_day, marker="*", color="white", markersize=14)
    axes.set_xlabel(f"Departure (days from {depart_labels[0][:10]})")
    axes.set_ylabel(f"Arrival (days from {arrive_labels[0][:10]})")
    axes.set_title("Porkchop C3 contour")

    png = render_png(figure)
    _close(figure)
    return png


def _circle(radius_km: float) -> Any:
    """A filled circle patch of the given radius at the origin (central body)."""
    from matplotlib.patches import Circle

    return Circle((0.0, 0.0), radius_km, color="C1", alpha=0.6)


def _close(figure: Figure) -> None:
    """Release the figure's resources after rendering."""
    import matplotlib.pyplot as plt

    plt.close(figure)


# ---------------------------------------------------------------------------
# Tool descriptions (subject to server_lint)
# ---------------------------------------------------------------------------


_PLOT_GROUND_TRACK_DESCRIPTION = (
    "Render a satellite's sub-satellite ground track as a PNG over a lon/lat "
    "graticule, with an inline summary (revolutions, lat / lon extent) carried "
    "alongside the image. e.g. plot_ground_track(states=<sgp4_propagate output "
    "series>) for a 24-hour ISS track. Pass the whole state SERIES, not a single "
    "state — the track needs many epochs to trace. States may be in any Earth "
    "frame (TEME, ICRF, GCRS, CIRS, ITRS); inertial frames are rotated to "
    "Earth-fixed internally, so for the cheapest path propagate with frame='ITRS'. "
    "The PNG is additive — the numeric summary is always inline, so a text-only "
    "client still sees revolutions and extent. Optional ground stations are "
    "overlaid as markers. A graticule is drawn (no coastline basemap). Empty or "
    "single-state input, or a non-Earth frame, returns a typed error — never an "
    "empty image. Client renders the returned PNG."
)

_PLOT_TRAJECTORY_DESCRIPTION = (
    "Render an orbit or transfer trajectory as a 2D or 3D PNG about a central "
    "body, with an inline summary (arc length, periapsis / apoapsis, time span) "
    "carried alongside the image. e.g. plot_trajectory(states=<state series>, "
    "projection='2D', central_body='earth') for a GTO arc. Pass the whole state "
    "SERIES, not a single state. Positions are plotted in the states' own frame "
    "axes (km); the central body is drawn at the origin to scale for known "
    "bodies. The PNG is additive — the numeric summary (arc length km, apsides "
    "km) is always inline. Empty or single-state input returns a typed error, "
    "never an empty image. Client renders the returned PNG."
)

_PLOT_PORKCHOP_DESCRIPTION = (
    "Render a porkchop C3 contour as a PNG from an existing porkchop grid "
    "result, reusing the computed grid with no recompute, and keep the inline "
    "grid summary (best cell C3 / total delta-v / epochs). e.g. "
    "plot_porkchop(porkchop_result=<porkchop output with output='full'>) for an "
    "Earth-Mars 2026 window. The result MUST carry the full grid — call porkchop "
    "with output='full' first; a summary-only result (empty grid) returns a "
    "typed error, never an empty image. The minimum-delta-v 'best' cell is marked "
    "on the contour. The PNG is additive — the numeric summary is always inline. "
    "Client renders the returned PNG."
)

_CZML_TRAJECTORY_DESCRIPTION = (
    "Emit a trajectory as a CZML document for a Cesium 3D client, e.g. an SGP4 "
    "propagation series rendered as an animated orbit, returned as an embedded "
    "resource with an inline summary (packet count, time span). Reserved slot "
    "— not yet implemented; lands in follow-up work."
)


# ---------------------------------------------------------------------------
# Tool bodies
# ---------------------------------------------------------------------------


def _resolve_stations(stations: list[Observer] | None) -> list[tuple[float, float, str]]:
    """Resolve overlaid ground stations to (lat_deg, lon_deg, label) triples.

    Named stations resolve through the access tool's coordinate registry so the
    overlay matches access_windows; explicit coordinates are used as given.
    """
    if not stations:
        return []
    from astrodynamics_mcp.schemas.base import NamedStation
    from astrodynamics_mcp.tools.access import _resolve_observer

    resolved: list[tuple[float, float, str]] = []
    for station in stations:
        lat_deg, lon_deg, _alt_km = _resolve_observer(station)
        # Named stations carry a label; explicit coordinates are unlabelled.
        label = station.name if isinstance(station, NamedStation) else ""
        resolved.append((lat_deg, lon_deg, label))
    return resolved


def _register_viz_tools() -> None:
    """Attach the four visualisation tool slots to ``astrodynamics_mcp.server.mcp``.

    Factored out of module top-level — like :func:`_register_spice_tools` and
    :func:`_register_gmat_tools` — so unit tests can drive registration against
    a fresh :class:`~mcp.server.fastmcp.FastMCP` instance without relying on the
    import-time guard being satisfied.

    Every slot is read-only and touches nothing outside the process: a plot
    consumes the state / grid series it is handed and renders locally, so none
    of them reach the network.
    """

    @register_tool(
        name="plot_ground_track",
        description=_PLOT_GROUND_TRACK_DESCRIPTION,
        annotations=ToolAnnotations(
            title="Plot Ground Track", readOnlyHint=True, openWorldHint=False
        ),
    )
    async def plot_ground_track(
        states: Annotated[
            list[StateVector],
            Field(
                description=(
                    "The state SERIES to trace, e.g. the `states` list from an "
                    "sgp4_propagate(frame='ITRS', output='full') call. Each entry is a "
                    "{r, v, frame, epoch}; the track needs many epochs (at least two). "
                    "Earth frames only (TEME / ICRF / GCRS / CIRS / ITRS)."
                ),
            ),
        ],
        stations: Annotated[
            list[Observer] | None,
            Field(
                description=(
                    "Optional ground stations to overlay as markers — either named "
                    "stations ({name: 'madrid'}) or explicit coordinates "
                    "({lat, lon, alt}). Omit for none."
                ),
            ),
        ] = None,
    ) -> GroundTrackResponse:
        _require_states(states, minimum=2, what="plot_ground_track")
        ecef_km = _earth_fixed_positions_km(states)
        lat, lon = _subsatellite_latlon(ecef_km)
        revs = _count_revolutions(lat)
        resolved_stations = _resolve_stations(stations)

        png = _render_ground_track(lat, lon, resolved_stations)
        summary_model = GroundTrackResponse(
            revolutions=Quantity(value=revs, unit="1"),
            lat_min=Quantity(value=float(np.min(lat)), unit="deg"),
            lat_max=Quantity(value=float(np.max(lat)), unit="deg"),
            lon_min=Quantity(value=float(np.min(lon)), unit="deg"),
            lon_max=Quantity(value=float(np.max(lon)), unit="deg"),
            image=PngImageInfo(width_px=_WIDTH_PX, height_px=_HEIGHT_PX),
        )
        summary = (
            f"Ground track: {len(states)} points, {revs:.2f} revs; "
            f"lat [{float(np.min(lat)):.1f}, {float(np.max(lat)):.1f}] deg, "
            f"lon [{float(np.min(lon)):.1f}, {float(np.max(lon)):.1f}] deg. PNG attached."
        )
        return tool_result_with_attachments(  # type: ignore[return-value]
            structured=summary_model, summary=summary, attachments=[png_image_content(png)]
        )

    @register_tool(
        name="plot_trajectory",
        description=_PLOT_TRAJECTORY_DESCRIPTION,
        annotations=ToolAnnotations(
            title="Plot Trajectory", readOnlyHint=True, openWorldHint=False
        ),
    )
    async def plot_trajectory(
        states: Annotated[
            list[StateVector],
            Field(
                description=(
                    "The state SERIES forming the orbit / transfer arc, e.g. the "
                    "`states` from an sgp4_propagate or lambert/porkchop-derived "
                    "propagation. Each entry is a {r, v, frame, epoch}; at least two "
                    "states are required. Positions are plotted in the states' frame."
                ),
            ),
        ],
        projection: Annotated[
            Literal["2D", "3D"],
            Field(
                description=(
                    "Plot projection: '2D' (x-y plane, default) or '3D' (x-y-z with a "
                    "fixed viewing angle)."
                ),
            ),
        ] = "2D",
        central_body: Annotated[
            str,
            Field(
                description=(
                    "Central body drawn at the origin, e.g. 'earth' (default), 'mars', "
                    "'moon', 'sun'. Known bodies are drawn to scale; an unknown name "
                    "still plots, with the origin marked."
                ),
            ),
        ] = "earth",
    ) -> TrajectoryResponse:
        _require_states(states, minimum=2, what="plot_trajectory")
        positions_km = _positions_km(states)
        radii = np.linalg.norm(positions_km, axis=1)

        png = _render_trajectory(positions_km, projection, central_body)
        summary_model = TrajectoryResponse(
            arc_length=Quantity(value=_arc_length_km(positions_km), unit="km"),
            periapsis_radius=Quantity(value=float(np.min(radii)), unit="km"),
            apoapsis_radius=Quantity(value=float(np.max(radii)), unit="km"),
            time_span=Quantity(value=_time_span_hours(states), unit="hours"),
            projection=projection,
            central_body=central_body,
            image=PngImageInfo(width_px=_WIDTH_PX, height_px=_HEIGHT_PX),
        )
        summary = (
            f"Trajectory ({projection}) about {central_body}: "
            f"arc {_arc_length_km(positions_km):.0f} km over {_time_span_hours(states):.2f} h; "
            f"periapsis {float(np.min(radii)):.0f} km, apoapsis {float(np.max(radii)):.0f} km. "
            "PNG attached."
        )
        return tool_result_with_attachments(  # type: ignore[return-value]
            structured=summary_model, summary=summary, attachments=[png_image_content(png)]
        )

    @register_tool(
        name="plot_porkchop",
        description=_PLOT_PORKCHOP_DESCRIPTION,
        annotations=ToolAnnotations(title="Plot Porkchop", readOnlyHint=True, openWorldHint=False),
    )
    async def plot_porkchop(
        porkchop_result: Annotated[
            PorkchopResponse,
            Field(
                description=(
                    "A porkchop tool result carrying the FULL grid — call porkchop "
                    "with output='full' and pass that object back here. The grid's "
                    "feasible cells (each tagged with depart / arrive epoch and C3) are "
                    "contoured; the 'best' cell is marked. A summary-only result "
                    "(empty grid) is rejected with a typed error."
                ),
            ),
        ],
    ) -> PorkchopPlotResponse:
        departs, arrives, c3_grid = _porkchop_grid_arrays(porkchop_result)
        depart_days = _days_from_first(departs)
        arrive_days = _days_from_first(arrives)
        best = porkchop_result.best
        best_depart_day = float(_days_from_first([departs[0], best.depart_epoch])[1])
        best_arrive_day = float(_days_from_first([arrives[0], best.arrive_epoch])[1])

        png = _render_porkchop(
            depart_days,
            arrive_days,
            c3_grid,
            departs,
            arrives,
            best_depart_day,
            best_arrive_day,
        )
        summary_model = PorkchopPlotResponse(
            best_c3=Quantity(value=best.c3.value, unit="km^2/s^2"),
            best_total_dv=Quantity(value=best.total_dv.value, unit="km/s"),
            best_tof=Quantity(value=best.tof.value, unit="days"),
            best_depart_epoch=best.depart_epoch,
            best_arrive_epoch=best.arrive_epoch,
            feasible_cells=len(porkchop_result.grid),
            n_depart_samples=len(departs),
            n_arrive_samples=len(arrives),
            image=PngImageInfo(width_px=_WIDTH_PX, height_px=_HEIGHT_PX),
        )
        summary = (
            f"Porkchop contour: {len(porkchop_result.grid)} feasible cells, "
            f"{len(departs)}x{len(arrives)} grid; best C3 {best.c3.value:.2f} km^2/s^2, "
            f"total delta-v {best.total_dv.value:.2f} km/s, depart {best.depart_epoch} "
            f"arrive {best.arrive_epoch}. PNG attached."
        )
        return tool_result_with_attachments(  # type: ignore[return-value]
            structured=summary_model, summary=summary, attachments=[png_image_content(png)]
        )

    @register_tool(
        name="czml_trajectory",
        description=_CZML_TRAJECTORY_DESCRIPTION,
        annotations=ToolAnnotations(
            title="CZML Trajectory", readOnlyHint=True, openWorldHint=False
        ),
    )
    async def czml_trajectory() -> VizPlaceholderResult:
        raise NotImplementedError("czml_trajectory lands in follow-up work")


if _VIZ_AVAILABLE:
    _register_viz_tools()
