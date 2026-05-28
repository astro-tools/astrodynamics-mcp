"""`porkchop` tool — interplanetary porkchop generation.

Two-axis sweep of departure and arrival epochs against the Sun-centred
Lambert problem. Body ephemerides come from JPL Horizons (via
``data.horizons``), the Lambert solver is ``lamberthub.izzo2015``, and
the response carries the full grid, the minimum-Δv "best" cell, and a
small ASCII contour of C3 over the grid intended for inline LLM display.

Frame and units: positions and velocities live in ICRF ecliptic km / km·s⁻¹
(the OUT_UNITS / REF_PLANE / REF_SYSTEM the Horizons adapter requests).
The Lambert solve is heliocentric; ``mu`` is the Sun barycentric μ.

The reported ``total_dv`` is the v0.1 two-impulse proxy
``|v_inf_dep| + |v_inf_arr|`` — both legs treated as hyperbolic injections.
Mission-specific maneuver design (powered flybys, parking-orbit insertion
losses, mid-course corrections) lives downstream of this scan.
"""

from __future__ import annotations

import bisect
import re
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Literal

import numpy as np
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from astrodynamics_mcp.data.horizons import HorizonsResponse, fetch_ephemeris
from astrodynamics_mcp.errors import InvalidInputError, UpstreamError
from astrodynamics_mcp.schemas.base import Epoch
from astrodynamics_mcp.server import register_tool
from astrodynamics_mcp.units import Quantity

# Heliocentric μ in km³/s². Sun barycentric value — Lambert solves in the
# Sun-centred inertial frame.
_MU_SUN = 1.32712440018e11

# Major-body → Horizons COMMAND. Planet centres (399, 499, …) rather than
# system barycentres — the textbook porkchop convention. Sun and Moon are
# omitted as origins/targets: porkchop is interplanetary by construction
# and Earth-Moon transfers belong in a separate tool.
_BODY_TO_HORIZONS: dict[str, str] = {
    "mercury": "199",
    "venus": "299",
    "earth": "399",
    "mars": "499",
    "jupiter": "599",
    "saturn": "699",
    "uranus": "799",
    "neptune": "899",
    "pluto": "999",
}

_HORIZONS_CENTER = "@sun"
_HORIZONS_STEP = "1d"

# Mean obliquity of the J2000 ecliptic relative to the ICRF equator
# (IAU 2006, ε0 = 84381.406″). Horizons returns states in the ICRF ecliptic
# frame; rotating about the X-axis by this angle gives ICRF equatorial
# components, needed for a true declination of the launch asymptote (DLA).
_J2000_OBLIQUITY_RAD = float(np.radians(84381.406 / 3600.0))

# JD epoch of 1970-01-01 00:00:00 UTC. JD-TDB vs JD-UTC differs by ~70 s in
# the modern era; for porkchop at 1d Horizons cadence the resulting
# angular-position error is <0.01° (well under any cell-level resolution).
_JD_UNIX_EPOCH = 2440587.5

# Glyph ramp for the ASCII contour: low C3 → '.', high C3 → 'X'.
_CONTOUR_RAMP = ".:-+*#@X"

# Tolerant Horizons VECTORS row pattern. One match per epoch block.
_VECTOR_BLOCK_RE = re.compile(
    r"""
    (?P<jd>\d+\.\d+)\s*=\s*A\.D\.[^\n]*?TDB
    [^X]*?
    X\s*=\s*(?P<x>[-+]?\s*\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)
    \s+Y\s*=\s*(?P<y>[-+]?\s*\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)
    \s+Z\s*=\s*(?P<z>[-+]?\s*\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)
    \s+VX\s*=\s*(?P<vx>[-+]?\s*\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)
    \s+VY\s*=\s*(?P<vy>[-+]?\s*\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)
    \s+VZ\s*=\s*(?P<vz>[-+]?\s*\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)
    """,
    re.VERBOSE | re.DOTALL,
)


# Lazy-import lamberthub: it pulls scipy.special at import time. Keeping
# the import deferred matches `lambert.py` so subprocess-spawn paths stay
# cheap (cache tests notice).
def _izzo2015() -> Any:
    import lamberthub

    return lamberthub.izzo2015


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class PorkchopCell(BaseModel):
    """One (depart_epoch, arrive_epoch) cell of the porkchop grid."""

    model_config = ConfigDict(extra="forbid")

    depart_epoch: Epoch = Field(..., description="UTC ISO 8601 departure epoch for this cell.")
    arrive_epoch: Epoch = Field(..., description="UTC ISO 8601 arrival epoch for this cell.")
    tof: Quantity = Field(
        ...,
        description="Time of flight, days.",
        examples=[{"value": 180.0, "unit": "days"}],
    )
    c3: Quantity = Field(
        ...,
        description=(
            "Departure characteristic energy, km^2/s^2. C3 = |v_inf_dep|^2; the launch "
            "vehicle's required excess energy above the departure body's escape velocity."
        ),
        examples=[{"value": 12.5, "unit": "km^2/s^2"}],
    )
    v_inf_arrival: Quantity = Field(
        ...,
        description="Magnitude of arrival V-infinity (hyperbolic excess speed at arrival), km/s.",
        examples=[{"value": 3.0, "unit": "km/s"}],
    )
    dec_dep_asymptote: Quantity = Field(
        ...,
        description=(
            "Declination of the departure asymptote (DLA) in the ICRF equatorial "
            "frame, deg — the v_infinity direction rotated from the ecliptic working "
            "frame by the J2000 obliquity. This is the equatorial declination launch "
            "vehicles target, not ecliptic latitude."
        ),
        examples=[{"value": -12.0, "unit": "deg"}],
    )
    total_dv: Quantity = Field(
        ...,
        description=(
            "Two-impulse Δv proxy = |v_inf_dep| + |v_inf_arr|, km/s. Both legs treated as "
            "hyperbolic injections; mission-specific maneuver design lives downstream."
        ),
        examples=[{"value": 6.4, "unit": "km/s"}],
    )


# How many of the lowest-total_dv cells the summary response carries.
# Five cells (~6 fields each, ~1.2 KB JSON) leaves headroom under the
# ~2 KB / ~2k-token default-response target.
_SUMMARY_TOP_CELLS = 5


class PorkchopResponse(BaseModel):
    """Response from :func:`porkchop`.

    The shape is the same regardless of ``output`` — the parameter only
    selects how much of the grid travels back to the caller:

    - ``output="summary"`` (default): ``grid`` is empty; ``top_cells``
      carries up to five lowest-``total_dv`` cells so the response fits
      small-model input caps.
    - ``output="full"``: ``grid`` carries every feasible cell in row-major
      order (outer: arrive-epoch, inner: depart-epoch); ``top_cells`` is
      still populated for the canonical "show me the alternatives" case.

    Infeasible cells (``tof <= 0``, Lambert no-solution) are skipped
    silently rather than carried as NaN — the ASCII summary marks them
    with a space glyph for the visual.
    """

    model_config = ConfigDict(extra="forbid")

    best: PorkchopCell = Field(
        ...,
        description="The feasible cell with the minimum total_dv across the scan.",
    )
    top_cells: list[PorkchopCell] = Field(
        ...,
        description=(
            "Up to five lowest-total_dv cells, sorted ascending. Always includes `best` "
            "as the first entry; size is capped to keep the default response under small-"
            "model input limits."
        ),
    )
    grid: list[PorkchopCell] = Field(
        default_factory=list,
        description=(
            "Every feasible cell in row-major order (outer: arrive-epoch, inner: depart-epoch). "
            "Populated only when the caller passes output='full'; empty otherwise."
        ),
    )
    ascii_summary: str = Field(
        ...,
        description=(
            "Compact text contour of C3 over the grid. samples_per_axis rows x "
            "samples_per_axis columns; each glyph is one of `.:-+*#@X` binned by C3 decile, "
            "infeasible cells rendered as a space. Rows are arrive-epoch indices "
            "(top = earliest), columns depart-epoch indices (left = earliest)."
        ),
    )


# ---------------------------------------------------------------------------
# Tool description (subject to server_lint)
# ---------------------------------------------------------------------------


_DESCRIPTION = (
    "Generate an interplanetary porkchop scan — the (depart_epoch x arrive_epoch) grid of C3, "
    "arrival V-infinity, declination of the departure asymptote, and total-Δv proxy for a "
    "heliocentric transfer. e.g. porkchop(departure_body='earth', arrival_body='mars', "
    "depart_window=['2026-10-01T00:00:00Z','2026-12-01T00:00:00Z'], "
    "arrive_window=['2027-04-01T00:00:00Z','2027-10-01T00:00:00Z'], samples_per_axis=20) "
    "returns the minimum-total_dv 'best' cell, the five lowest-total_dv cells, and a compact "
    "ASCII contour for inline LLM display. Both windows are UTC ISO 8601 pairs [start, end]; "
    "the grid is samples_per_axis x samples_per_axis linspace-sampled across each window. "
    "Output shaping: the default output='summary' trims the response to best/top_cells/"
    "ascii_summary so it fits small-model context windows; pass output='full' to receive "
    "every feasible cell in `grid` (a 30x30 scan is ~250 KB and will overflow tight input "
    "caps). Body ephemerides come from JPL Horizons — the first call after a cold cache "
    "takes minutes (Horizons is slow), subsequent calls within the 7-day TTL are local. For "
    "broad exploratory scans, bring samples_per_axis down to 15-20 before launching a 50x50 "
    "grid. mu='sun' is the only v0.1 mu — heliocentric Lambert in ICRF ecliptic km / km/s. "
    "Misordered windows (arrive entirely before depart) raise "
    "invalid_input.porkchop_window_order. Horizons unreachable mid-grid raises "
    "data_source.horizons_unreachable with no partial results."
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_body(name: str, *, field: str) -> str:
    if not isinstance(name, str):
        raise InvalidInputError(
            f"{field} must be a string body name, got {type(name).__name__}",
            code="invalid_input.body_not_a_string",
        )
    key = name.lower()
    if key not in _BODY_TO_HORIZONS:
        raise InvalidInputError(
            f"unknown body {name!r} for porkchop; supported: {sorted(_BODY_TO_HORIZONS)}",
            code="invalid_input.unknown_body",
        )
    return key


def _parse_iso_epoch(value: str, *, field: str) -> datetime:
    """Parse an ISO 8601 UTC epoch string into a tz-aware UTC datetime."""
    if not isinstance(value, str):
        raise InvalidInputError(
            f"{field} must be a string ISO 8601 epoch, got {type(value).__name__}",
            code="invalid_input.epoch_not_a_string",
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InvalidInputError(
            f"{field} {value!r} is not a valid ISO 8601 timestamp",
            code="invalid_input.epoch_malformed",
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _validate_window(window: Any, *, field: str) -> tuple[datetime, datetime]:
    if not isinstance(window, list) or len(window) != 2:
        raise InvalidInputError(
            f"{field} must be a list of two ISO 8601 epoch strings [start, end], got {window!r}",
            code="invalid_input.window_wrong_shape",
        )
    start = _parse_iso_epoch(window[0], field=f"{field}[0]")
    end = _parse_iso_epoch(window[1], field=f"{field}[1]")
    if end <= start:
        raise InvalidInputError(
            f"{field} end {window[1]!r} must be strictly after start {window[0]!r}",
            code="invalid_input.window_end_not_after_start",
        )
    return start, end


def _validate_samples(samples_per_axis: Any) -> int:
    if isinstance(samples_per_axis, bool) or not isinstance(samples_per_axis, int):
        raise InvalidInputError(
            f"samples_per_axis must be an int, got {type(samples_per_axis).__name__}",
            code="invalid_input.samples_not_an_int",
        )
    if samples_per_axis < 2:
        raise InvalidInputError(
            f"samples_per_axis must be ≥ 2 to define a grid, got {samples_per_axis}",
            code="invalid_input.samples_too_small",
        )
    return samples_per_axis


def _validate_mu(mu: str) -> None:
    if mu != "sun":
        raise InvalidInputError(
            f"mu must be 'sun' at v0.1 (heliocentric porkchop only), got {mu!r}",
            code="invalid_input.unsupported_mu",
        )


# ---------------------------------------------------------------------------
# Horizons VECTORS parsing & interpolation
# ---------------------------------------------------------------------------


def _jd_to_utc(jd: float) -> datetime:
    """Treat a Horizons JD-TDB row as JD-UTC. See module docstring."""
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(
        seconds=(jd - _JD_UNIX_EPOCH) * 86400.0
    )


def _strip_signed_number(text: str) -> float:
    """Convert a Horizons capture group (may carry a sign with intervening space)."""
    return float(text.replace(" ", ""))


def _parse_horizons_vectors(
    response: HorizonsResponse,
) -> tuple[list[datetime], np.ndarray, np.ndarray]:
    """Parse a Horizons VECTORS response into (epochs, positions_km, velocities_km_s)."""
    matches = list(_VECTOR_BLOCK_RE.finditer(response.result))
    if not matches:
        raise UpstreamError(
            "Horizons response contained no parseable VECTORS blocks",
            code="upstream.horizons_unexpected_shape",
            data={"signature": response.signature},
        )
    epochs: list[datetime] = []
    positions: list[list[float]] = []
    velocities: list[list[float]] = []
    for match in matches:
        try:
            jd = float(match.group("jd"))
            x = _strip_signed_number(match.group("x"))
            y = _strip_signed_number(match.group("y"))
            z = _strip_signed_number(match.group("z"))
            vx = _strip_signed_number(match.group("vx"))
            vy = _strip_signed_number(match.group("vy"))
            vz = _strip_signed_number(match.group("vz"))
        except ValueError as exc:
            raise UpstreamError(
                f"Horizons row at JD {match.group('jd')!r} carries malformed numbers",
                code="upstream.horizons_unexpected_shape",
                original_exception=exc,
            ) from exc
        epochs.append(_jd_to_utc(jd))
        positions.append([x, y, z])
        velocities.append([vx, vy, vz])
    # Horizons output is already epoch-ascending, but defensive sort keeps
    # interpolation honest if a future API revision starts paginating.
    order = sorted(range(len(epochs)), key=lambda i: epochs[i])
    epochs_sorted = [epochs[i] for i in order]
    positions_sorted = np.asarray([positions[i] for i in order], dtype=float)
    velocities_sorted = np.asarray([velocities[i] for i in order], dtype=float)
    return epochs_sorted, positions_sorted, velocities_sorted


def _interp_state(
    target: datetime,
    epochs: list[datetime],
    positions: np.ndarray,
    velocities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear-interpolate (r, v) at ``target`` between bracketing Horizons rows.

    Linear interpolation at 1-day cadence introduces sub-km position noise
    over heliocentric distances of 10⁸ km — well below porkchop resolution.
    """
    if target < epochs[0] or target > epochs[-1]:
        raise UpstreamError(
            f"requested epoch {target.isoformat()} falls outside the Horizons window "
            f"[{epochs[0].isoformat()}, {epochs[-1].isoformat()}]",
            code="upstream.horizons_window_too_narrow",
        )
    idx = bisect.bisect_left(epochs, target)
    if idx == 0:
        return positions[0].copy(), velocities[0].copy()
    if idx >= len(epochs):
        return positions[-1].copy(), velocities[-1].copy()
    t_left, t_right = epochs[idx - 1], epochs[idx]
    span = (t_right - t_left).total_seconds()
    if span <= 0:
        return positions[idx].copy(), velocities[idx].copy()
    frac = (target - t_left).total_seconds() / span
    r = positions[idx - 1] + frac * (positions[idx] - positions[idx - 1])
    v = velocities[idx - 1] + frac * (velocities[idx] - velocities[idx - 1])
    return r, v


async def _fetch_body_ephemeris(
    body_key: str,
    window_start: datetime,
    window_end: datetime,
) -> tuple[list[datetime], np.ndarray, np.ndarray]:
    """Fetch and parse Horizons VECTORS for a body across the window.

    The Horizons request is padded by one day on each side so the sampled
    grid endpoints sit safely inside the parsed table (avoiding edge-case
    interpolation failures when the window exactly hits the table bounds).
    """
    pad = timedelta(days=1)
    start = (window_start - pad).strftime("%Y-%m-%d")
    stop = (window_end + pad).strftime("%Y-%m-%d")
    response = await fetch_ephemeris(
        _BODY_TO_HORIZONS[body_key],
        _HORIZONS_CENTER,
        start,
        stop,
        _HORIZONS_STEP,
    )
    return _parse_horizons_vectors(response)


# ---------------------------------------------------------------------------
# Grid evaluation
# ---------------------------------------------------------------------------


def _linspace_epochs(start: datetime, end: datetime, n: int) -> list[datetime]:
    span = (end - start).total_seconds()
    return [start + timedelta(seconds=span * i / (n - 1)) for i in range(n)]


def _solve_cell(
    depart_t: datetime,
    arrive_t: datetime,
    r_dep_body: np.ndarray,
    v_dep_body: np.ndarray,
    r_arr_body: np.ndarray,
    v_arr_body: np.ndarray,
    izzo: Any,
) -> PorkchopCell | None:
    """Evaluate one (depart, arrive) cell. Returns None on infeasible geometry."""
    tof_seconds = (arrive_t - depart_t).total_seconds()
    if tof_seconds <= 0:
        return None
    try:
        result = izzo(_MU_SUN, r_dep_body, r_arr_body, tof_seconds, prograde=True, M=0)
    except (AssertionError, ValueError, RuntimeError):
        return None
    v_transfer_dep = np.asarray(result[0], dtype=float)
    v_transfer_arr = np.asarray(result[1], dtype=float)
    # A marginal solve can converge to a non-finite velocity rather than
    # raising; treat that cell as infeasible (skipped) so no NaN/inf reaches
    # the wire — consistent with the tof <= 0 and no-solution skips above.
    if not (np.all(np.isfinite(v_transfer_dep)) and np.all(np.isfinite(v_transfer_arr))):
        return None

    v_inf_dep_vec = v_transfer_dep - v_dep_body
    v_inf_dep_mag = float(np.linalg.norm(v_inf_dep_vec))
    v_inf_arr_mag = float(np.linalg.norm(v_transfer_arr - v_arr_body))
    c3 = v_inf_dep_mag * v_inf_dep_mag
    total_dv = v_inf_dep_mag + v_inf_arr_mag

    if v_inf_dep_mag > 0:
        # v_inf is in the ICRF ecliptic frame; rotate its z-component into the
        # ICRF equatorial frame (rotation about X by the J2000 obliquity) so
        # this is a true equatorial declination (the DLA launch designers use),
        # not ecliptic latitude.
        sin_eps = np.sin(_J2000_OBLIQUITY_RAD)
        cos_eps = np.cos(_J2000_OBLIQUITY_RAD)
        z_equatorial = v_inf_dep_vec[1] * sin_eps + v_inf_dep_vec[2] * cos_eps
        dec_rad = float(np.arcsin(np.clip(z_equatorial / v_inf_dep_mag, -1.0, 1.0)))
    else:
        dec_rad = 0.0
    dec_deg = float(np.degrees(dec_rad))

    return PorkchopCell(
        depart_epoch=depart_t.strftime("%Y-%m-%dT%H:%M:%SZ"),
        arrive_epoch=arrive_t.strftime("%Y-%m-%dT%H:%M:%SZ"),
        tof=Quantity(value=tof_seconds / 86400.0, unit="days"),
        c3=Quantity(value=c3, unit="km^2/s^2"),
        v_inf_arrival=Quantity(value=v_inf_arr_mag, unit="km/s"),
        dec_dep_asymptote=Quantity(value=dec_deg, unit="deg"),
        total_dv=Quantity(value=total_dv, unit="km/s"),
    )


def _ascii_contour(rows_of_c3: list[list[float | None]]) -> str:
    """Render the C3 grid as a small ASCII contour for LLM consumption."""
    finite = [c for row in rows_of_c3 for c in row if c is not None and np.isfinite(c)]
    if not finite:
        return ""
    c_min = min(finite)
    c_max = max(finite)
    span = max(c_max - c_min, 1e-9)
    last_idx = len(_CONTOUR_RAMP) - 1
    lines: list[str] = []
    for row in rows_of_c3:
        chars: list[str] = []
        for c in row:
            if c is None or not np.isfinite(c):
                chars.append(" ")
                continue
            idx = int((c - c_min) / span * last_idx)
            chars.append(_CONTOUR_RAMP[max(0, min(idx, last_idx))])
        lines.append("".join(chars))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@register_tool(
    name="porkchop",
    description=_DESCRIPTION,
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
)
async def porkchop(
    departure_body: Annotated[
        str,
        Field(
            description=(
                "Body the spacecraft departs from. One of the JPL Horizons "
                "major-body names: 'mercury', 'venus', 'earth', 'mars', "
                "'jupiter', 'saturn', 'uranus', 'neptune'. Must differ from "
                "`arrival_body`."
            ),
        ),
    ],
    arrival_body: Annotated[
        str,
        Field(
            description=(
                "Body the spacecraft arrives at. Same body-name enum as "
                "`departure_body` and must be different from it."
            ),
        ),
    ],
    depart_window: Annotated[
        list[str],
        Field(
            description=(
                "Departure-epoch range as a two-element [start, end] list of "
                "UTC ISO 8601 strings, e.g. "
                "['2028-03-01T00:00:00Z', '2028-06-01T00:00:00Z']. The axis is "
                "sampled uniformly across this range."
            ),
        ),
    ],
    arrive_window: Annotated[
        list[str],
        Field(
            description=(
                "Arrival-epoch range as a two-element [start, end] list of UTC "
                "ISO 8601 strings, sampled uniformly along the other grid axis. "
                "Must end strictly after `depart_window` starts so at least one "
                "positive-time-of-flight cell exists."
            ),
        ),
    ],
    mu: Annotated[
        Literal["sun"],
        Field(
            description=(
                "Central-body gravitational parameter for the heliocentric "
                "Lambert solve. Only 'sun' is supported at v0.1; barycentric μ "
                "is used."
            ),
        ),
    ] = "sun",
    samples_per_axis: Annotated[
        int,
        Field(
            description=(
                "Grid resolution per axis — the full grid has "
                "samples_per_axis² cells. Default 30 gives a 900-cell grid; "
                "higher resolves fine structure but costs more Horizons calls "
                "and Lambert solves. Capped at the upper end to keep the "
                "response size sane."
            ),
        ),
    ] = 30,
    output: Annotated[
        Literal["summary", "full"],
        Field(
            description=(
                "'summary' (default) returns the minimum-Δv 'best' cell, the "
                "ASCII C3 contour, and grid metadata only — the MCP-payload-"
                "friendly form. 'full' adds the per-cell grid; only request "
                "this when downstream really needs every cell."
            ),
        ),
    ] = "summary",
) -> PorkchopResponse:
    # Validation.
    dep_body = _validate_body(departure_body, field="departure_body")
    arr_body = _validate_body(arrival_body, field="arrival_body")
    if dep_body == arr_body:
        raise InvalidInputError(
            f"departure_body and arrival_body must differ, both were {dep_body!r}",
            code="invalid_input.same_body",
        )
    _validate_mu(mu)
    n = _validate_samples(samples_per_axis)
    depart_start, depart_end = _validate_window(depart_window, field="depart_window")
    arrive_start, arrive_end = _validate_window(arrive_window, field="arrive_window")
    if arrive_end <= depart_start:
        raise InvalidInputError(
            f"arrive_window {arrive_window!r} ends at or before depart_window "
            f"{depart_window!r} starts — no positive-tof cell is possible",
            code="invalid_input.porkchop_window_order",
        )

    # Fetch ephemerides. DataSourceError from Horizons propagates unchanged
    # (no partial grid; the tool is request-response).
    dep_epochs, dep_positions, dep_velocities = await _fetch_body_ephemeris(
        dep_body, depart_start, depart_end
    )
    arr_epochs, arr_positions, arr_velocities = await _fetch_body_ephemeris(
        arr_body, arrive_start, arrive_end
    )

    # Sample windows and pre-resolve body states at every sample epoch.
    depart_samples = _linspace_epochs(depart_start, depart_end, n)
    arrive_samples = _linspace_epochs(arrive_start, arrive_end, n)
    dep_states = [
        _interp_state(t, dep_epochs, dep_positions, dep_velocities) for t in depart_samples
    ]
    arr_states = [
        _interp_state(t, arr_epochs, arr_positions, arr_velocities) for t in arrive_samples
    ]

    izzo = _izzo2015()
    flat_grid: list[PorkchopCell] = []
    c3_rows: list[list[float | None]] = []
    for ai, arrive_t in enumerate(arrive_samples):
        r_arr_body, v_arr_body = arr_states[ai]
        row_c3: list[float | None] = []
        for di, depart_t in enumerate(depart_samples):
            r_dep_body, v_dep_body = dep_states[di]
            cell = _solve_cell(
                depart_t, arrive_t, r_dep_body, v_dep_body, r_arr_body, v_arr_body, izzo
            )
            if cell is None:
                row_c3.append(None)
                continue
            row_c3.append(float(cell.c3.value))
            flat_grid.append(cell)
        c3_rows.append(row_c3)

    if not flat_grid:
        raise UpstreamError(
            "porkchop grid is entirely infeasible — every (depart, arrive) cell either has "
            "non-positive tof or Lambert produced no solution",
            code="upstream.porkchop_grid_empty",
            data={
                "depart_window": depart_window,
                "arrive_window": arrive_window,
                "samples_per_axis": n,
            },
        )

    sorted_by_total_dv = sorted(flat_grid, key=lambda c: c.total_dv.value)
    return PorkchopResponse(
        best=sorted_by_total_dv[0],
        top_cells=sorted_by_total_dv[:_SUMMARY_TOP_CELLS],
        grid=flat_grid if output == "full" else [],
        ascii_summary=_ascii_contour(c3_rows),
    )
