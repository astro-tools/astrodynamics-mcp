"""Shared pydantic base models for every tool's input and output schema.

This module is the single source of truth for the JSON-schema shape the MCP
SDK exposes to LLM consumers — every tool composes its input/output
schema from these primitives so naming, units, and frame conventions stay
consistent across the surface.

Every model carries rich :class:`~pydantic.Field` descriptions and example
values; those are what the LLM reads when deciding how to call the tool.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from astrodynamics_mcp.errors import InvalidInputError
from astrodynamics_mcp.units import Quantity, QuantityVector

# ---------------------------------------------------------------------------
# Closed enums
# ---------------------------------------------------------------------------


class TimeScale(str, Enum):
    """Time-scale identifiers used across the time/frame/access tool surfaces.

    Inherits from ``str`` so pydantic round-trips the enum as its string
    value in JSON (``"UTC"`` rather than ``"TimeScale.UTC"``).
    """

    UTC = "UTC"
    TAI = "TAI"
    TT = "TT"
    TDB = "TDB"
    UT1 = "UT1"
    GPS = "GPS"
    TCB = "TCB"
    TCG = "TCG"


class Frame(str, Enum):
    """Reference frames used for state vectors and frame conversions.

    The v0.1 set covers the inertial, Earth-rotating, and IAU body-fixed
    frames the wrapped upstreams (sgp4, astropy.coordinates, skyfield) all
    speak. Adding a new frame here means adding the corresponding transform
    path to the frame_transform tool.
    """

    TEME = "TEME"
    ICRF = "ICRF"
    GCRS = "GCRS"
    ITRS = "ITRS"
    CIRS = "CIRS"
    TIRS = "TIRS"
    IAU_EARTH = "IAU_EARTH"
    IAU_MARS = "IAU_MARS"
    IAU_MOON = "IAU_MOON"


# ---------------------------------------------------------------------------
# Epoch (string-with-validator alias)
# ---------------------------------------------------------------------------


# ISO 8601 with a mandatory time component: YYYY-MM-DDThh:mm:ss[.frac][Z|±HH:MM].
# Permissive about timezone designator and fractional seconds; the goal here
# is to catch the "LLM emits a bare date" mistake, not to be a full ISO 8601
# parser. The actual datetime resolution happens in the time_convert tool
# against astropy, which is far more permissive than we need to be.
_EPOCH_ISO8601_RE = re.compile(
    r"""
    ^
    \d{4}-\d{2}-\d{2}        # YYYY-MM-DD
    T                        # mandatory time separator
    \d{2}:\d{2}:\d{2}        # hh:mm:ss
    (?:\.\d+)?               # optional fractional seconds
    (?:Z|[+-]\d{2}:?\d{2})?  # optional timezone designator (Z, +HHMM, +HH:MM)
    $
    """,
    re.VERBOSE,
)


def _validate_epoch(value: object) -> str:
    """Validate an ISO 8601 epoch string with a time component."""
    if not isinstance(value, str):
        raise InvalidInputError(
            f"epoch must be a string, got {type(value).__name__}",
            code="invalid_input.epoch_not_a_string",
        )
    # Bare date (YYYY-MM-DD) is the canonical mistake: catch it with a
    # specific code so the LLM sees an actionable error.
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise InvalidInputError(
            f"epoch must be UTC ISO 8601, not just {value!r} — include a "
            f"time component, e.g. {value}T00:00:00Z",
            code="invalid_input.epoch_missing_time_component",
        )
    if not _EPOCH_ISO8601_RE.match(value):
        raise InvalidInputError(
            f"epoch {value!r} is not a valid ISO 8601 timestamp; expected "
            "YYYY-MM-DDThh:mm:ss[.fraction][Z or ±HH:MM]",
            code="invalid_input.epoch_malformed",
        )
    return value


def _epoch_to_instant(value: str) -> datetime:
    """Parse a validated :data:`Epoch` string to a timezone-aware ``datetime``.

    The string has already passed ``_validate_epoch`` (and thus the
    ``_EPOCH_ISO8601_RE`` shape), so the only normalization needed here is to
    feed ``datetime.fromisoformat`` a form it accepts on the 3.10 floor:
    ``Z`` → ``+00:00`` and ``±HHMM`` → ``±HH:MM``. A timezone-naive epoch is
    interpreted as UTC, matching the surface's documented convention.
    """
    text = value
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    else:
        # Insert the colon into a compact ±HHMM offset (3.10 fromisoformat
        # requires it); ±HH:MM and offset-less forms are left untouched.
        text = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", text)
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


Epoch = Annotated[
    str,
    BeforeValidator(_validate_epoch),
    Field(
        description=(
            "ISO 8601 UTC timestamp with a mandatory time component. "
            "Examples: '2026-05-23T12:00:00Z', '2026-05-23T12:00:00.500+00:00'. "
            "A bare date like '2026-05-23' is rejected."
        ),
        examples=["2026-05-23T12:00:00Z", "2026-01-01T00:00:00.500Z"],
    ),
]


# ---------------------------------------------------------------------------
# Body (string alias with examples)
# ---------------------------------------------------------------------------


Body = Annotated[
    str,
    Field(
        description=(
            "Celestial body or satellite name. Accepts SPICE-style identifiers "
            "(e.g. '399' for Earth, '301' for Moon), common English names "
            "('earth', 'sun', 'mars'), and well-known satellite names "
            "('hubble', 'iss'). Resolution happens inside the tool that "
            "consumes this — schema validation here is intentionally permissive."
        ),
        examples=["earth", "sun", "mars", "hubble", "399"],
    ),
]


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------


# Closed v0.1 named-station registry. Coordinate resolution lives in the
# access_windows tool; this enum is just the user-facing name surface.
NamedStationName = Literal[
    "madrid",
    "goldstone",
    "canberra",
    "svalbard",
    "wallops",
    "esrange",
    "gsfc",
    "jpl",
]


class NamedStation(BaseModel):
    """An observer identified by a name from the v0.1 station registry."""

    model_config = ConfigDict(extra="forbid")

    name: NamedStationName = Field(
        ...,
        description=(
            "Short name of a known ground station. Resolved to lat/lon/alt "
            "inside the consuming tool. The v0.1 registry is intentionally "
            "small; pass explicit ObserverCoordinates for anything else."
        ),
        examples=["madrid", "goldstone"],
    )


_LENGTH_UNITS: frozenset[str] = frozenset({"km", "m", "AU"})
_VELOCITY_UNITS: frozenset[str] = frozenset({"km/s", "m/s"})
_ANGLE_UNITS: frozenset[str] = frozenset({"deg", "rad"})
_TIME_UNITS: frozenset[str] = frozenset({"s", "min", "hours", "days"})


def _require_unit_in(quantity: Quantity, allowed: frozenset[str], *, field: str) -> Quantity:
    if quantity.unit not in allowed:
        raise InvalidInputError(
            f"{field} unit must be one of {sorted(allowed)}, got {quantity.unit!r}",
            code="invalid_input.wrong_unit_category",
        )
    return quantity


def _require_vector_unit_in(
    quantity: QuantityVector,
    allowed: frozenset[str],
    *,
    field: str,
    expected_length: int | None = None,
) -> QuantityVector:
    if quantity.unit not in allowed:
        raise InvalidInputError(
            f"{field} unit must be one of {sorted(allowed)}, got {quantity.unit!r}",
            code="invalid_input.wrong_unit_category",
        )
    if expected_length is not None and len(quantity.value) != expected_length:
        raise InvalidInputError(
            f"{field} must have exactly {expected_length} components, got {len(quantity.value)}",
            code="invalid_input.wrong_vector_length",
        )
    return quantity


class ObserverCoordinates(BaseModel):
    """An observer specified by explicit geodetic coordinates."""

    model_config = ConfigDict(extra="forbid")

    lat: Quantity = Field(
        ...,
        description="Geodetic latitude in degrees.",
        examples=[{"value": 40.4168, "unit": "deg"}],
    )
    lon: Quantity = Field(
        ...,
        description="Geodetic longitude in degrees (east-positive).",
        examples=[{"value": -3.7038, "unit": "deg"}],
    )
    alt: Quantity = Field(
        ...,
        description="Altitude above the WGS-84 ellipsoid in km.",
        examples=[{"value": 0.667, "unit": "km"}],
    )

    @field_validator("lat", "lon")
    @classmethod
    def _angle_unit(cls, v: Quantity) -> Quantity:
        return _require_unit_in(v, _ANGLE_UNITS, field="lat/lon")

    @field_validator("alt")
    @classmethod
    def _length_unit(cls, v: Quantity) -> Quantity:
        return _require_unit_in(v, _LENGTH_UNITS, field="alt")


Observer = Annotated[
    NamedStation | ObserverCoordinates,
    Field(
        description=(
            "Observer location. Either a named station "
            '(`{"name": "madrid"}`) or explicit coordinates '
            '(`{"lat": {...}, "lon": {...}, "alt": {...}}`).'
        ),
    ),
]


# ---------------------------------------------------------------------------
# TLE
# ---------------------------------------------------------------------------


_TLE_LINE_LENGTH = 69


class TleLines(BaseModel):
    """Two-line TLE as raw strings.

    Both lines must be exactly 69 characters (the canonical TLE width).
    Sub-69-character lines are the most common LLM mistake — they almost
    always come from a chat client stripping trailing whitespace.
    """

    model_config = ConfigDict(extra="forbid")

    line1: str = Field(
        ...,
        description="First TLE line (begins with '1 '), exactly 69 characters.",
        examples=["1 25544U 98067A   24001.50000000  .00010000  00000-0  18000-3 0  9990"],
    )
    line2: str = Field(
        ...,
        description="Second TLE line (begins with '2 '), exactly 69 characters.",
        examples=["2 25544  51.6400  90.0000 0001000  90.0000 270.0000 15.50000000000000"],
    )

    @field_validator("line1", "line2")
    @classmethod
    def _length(cls, v: str) -> str:
        if len(v) != _TLE_LINE_LENGTH:
            raise InvalidInputError(
                f"TLE line must be exactly {_TLE_LINE_LENGTH} characters, got {len(v)}",
                code="invalid_input.tle_line_wrong_length",
            )
        return v


class TleOmm(BaseModel):
    """Parsed OMM (Orbit Mean Elements Message) JSON.

    Schema is intentionally loose at v0.1: we accept whatever the upstream
    OMM source (CelesTrak) emitted. Downstream tools that need specific
    fields raise typed errors when those fields are missing.
    """

    model_config = ConfigDict(extra="forbid")

    omm: dict[str, Any] = Field(
        ...,
        description=(
            "Parsed OMM JSON object (CCSDS standard fields like "
            "CCSDS_OMM_VERS, EPOCH, MEAN_ELEMENTS, …). Fetched from CelesTrak "
            "by the tle_lookup tool or supplied directly."
        ),
        examples=[
            {
                "CCSDS_OMM_VERS": "2.0",
                "EPOCH": "2024-01-01T12:00:00.000000",
                "MEAN_MOTION": 15.5,
            },
        ],
    )


Tle = Annotated[
    TleLines | TleOmm,
    Field(
        description=(
            "A TLE in one of two shapes: two raw 69-character lines "
            "(`{line1, line2}`) or a parsed OMM JSON object (`{omm: {...}}`)."
        ),
    ),
]


# ---------------------------------------------------------------------------
# StateVector
# ---------------------------------------------------------------------------


class StateVector(BaseModel):
    """Cartesian state in a named frame at a named epoch.

    The position and velocity are :class:`QuantityVector` instances so the
    unit (km vs m, km/s vs m/s) is on the wire — never implicit.
    """

    model_config = ConfigDict(extra="forbid")

    r: QuantityVector = Field(
        ...,
        description="Cartesian position vector [x, y, z]. Unit must be a length (km / m / AU).",
        examples=[{"value": [7000.0, 0.0, 0.0], "unit": "km"}],
    )
    v: QuantityVector = Field(
        ...,
        description="Cartesian velocity vector [vx, vy, vz]. Unit must be a velocity (km/s / m/s).",
        examples=[{"value": [0.0, 7.5, 0.0], "unit": "km/s"}],
    )
    frame: Frame = Field(
        ...,
        description="Reference frame in which r and v are expressed.",
        examples=["TEME", "ICRF"],
    )
    epoch: Epoch = Field(
        ...,
        description="UTC ISO 8601 epoch at which the state is valid.",
    )

    @field_validator("r")
    @classmethod
    def _position(cls, v: QuantityVector) -> QuantityVector:
        return _require_vector_unit_in(v, _LENGTH_UNITS, field="r", expected_length=3)

    @field_validator("v")
    @classmethod
    def _velocity(cls, v: QuantityVector) -> QuantityVector:
        return _require_vector_unit_in(v, _VELOCITY_UNITS, field="v", expected_length=3)


# ---------------------------------------------------------------------------
# Interval (start, end, duration_s)
# ---------------------------------------------------------------------------


class Interval(BaseModel):
    """A time interval bounded by two epochs.

    `start` and `end` must be UTC ISO 8601; `duration_s` carries the unit
    explicitly. The `end > start` ordering check compares the two epochs as
    parsed instants, so it is correct across mixed timezone designators
    (e.g. an `end` of `...Z` against a `start` of `...+00:00`).
    """

    model_config = ConfigDict(extra="forbid")

    start: Epoch = Field(..., description="Start of the interval (UTC ISO 8601).")
    end: Epoch = Field(..., description="End of the interval (UTC ISO 8601).")
    duration_s: Quantity = Field(
        ...,
        description="Duration of the interval. Unit must be a time (s / min / hours / days).",
        examples=[{"value": 600.0, "unit": "s"}],
    )

    @field_validator("duration_s")
    @classmethod
    def _time_unit(cls, v: Quantity) -> Quantity:
        return _require_unit_in(v, _TIME_UNITS, field="duration_s")

    @model_validator(mode="after")
    def _end_after_start(self) -> Interval:
        if _epoch_to_instant(self.end) <= _epoch_to_instant(self.start):
            raise InvalidInputError(
                f"interval end {self.end!r} must be strictly after start {self.start!r}",
                code="invalid_input.interval_end_not_after_start",
            )
        return self


# ---------------------------------------------------------------------------
# Keplerian elements
# ---------------------------------------------------------------------------


class KeplerianElements(BaseModel):
    """Classical Keplerian orbital elements (a, e, i, RAAN, argp, nu).

    Shared across any tool that emits an orbit's elements — Lambert solve
    (transfer arc), porkchop (transfer-elements grid), bplane (hyperbolic
    approach orbit). True anomaly ``nu`` is at the reference epoch (e.g.
    the start of the Lambert transfer for ``lambert_solve``).

    Semi-major axis ``a`` is negative for hyperbolic orbits and undefined
    for parabolic ones — callers that need to handle the parabolic edge
    should branch on eccentricity rather than ``a``.
    """

    model_config = ConfigDict(extra="forbid")

    a: Quantity = Field(
        ...,
        description="Semi-major axis. Length unit (km / m / AU).",
        examples=[{"value": 24371.0, "unit": "km"}],
    )
    e: Quantity = Field(
        ...,
        description="Eccentricity (dimensionless; unit must be the dimensionless '1').",
        examples=[{"value": 0.7, "unit": "1"}],
    )
    i: Quantity = Field(
        ...,
        description="Inclination, deg.",
        examples=[{"value": 28.5, "unit": "deg"}],
    )
    raan: Quantity = Field(
        ...,
        description="Right ascension of the ascending node, deg.",
        examples=[{"value": 45.0, "unit": "deg"}],
    )
    argp: Quantity = Field(
        ...,
        description="Argument of periapsis, deg.",
        examples=[{"value": 90.0, "unit": "deg"}],
    )
    nu: Quantity = Field(
        ...,
        description="True anomaly at the reference epoch, deg.",
        examples=[{"value": 30.0, "unit": "deg"}],
    )

    @field_validator("a")
    @classmethod
    def _length(cls, v: Quantity) -> Quantity:
        return _require_unit_in(v, _LENGTH_UNITS, field="a")

    @field_validator("e")
    @classmethod
    def _dimensionless(cls, v: Quantity) -> Quantity:
        if v.unit != "1":
            raise InvalidInputError(
                f"eccentricity unit must be '1' (dimensionless), got {v.unit!r}",
                code="invalid_input.wrong_unit_category",
            )
        return v

    @field_validator("i", "raan", "argp", "nu")
    @classmethod
    def _angle(cls, v: Quantity) -> Quantity:
        return _require_unit_in(v, _ANGLE_UNITS, field="angle")
