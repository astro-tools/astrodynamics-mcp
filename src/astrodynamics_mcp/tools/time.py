"""`time_convert` tool — UTC/TAI/TT/TDB/UT1/GPS/TCB/TCG conversions.

Wraps ``astropy.time.Time`` with two small extensions:

- GPS scale: astropy's ``Time`` does not accept ``scale="gps"`` as an
  input scale and exposes GPS only as a ``.gps`` attribute that returns
  seconds since the GPS epoch. We bridge by treating a GPS-scale value
  as TAI offset by +19 s (the constant GPS-TAI relationship) and emit
  the inverse shift on output.
- ``j2000_seconds`` format: seconds since J2000 (2000-01-01T12:00:00 TT)
  is not a native astropy format. We compute it from ``Time.tt.jd``.

When the conversion path touches ``UT1``, the tool warms astropy's IERS
Bulletin A cache via ``astrodynamics_mcp.data.iers.load_iers`` and
surfaces the bulletin's freshness anchor + the per-call UT1-UTC offset.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from astrodynamics_mcp.errors import InvalidInputError, UpstreamError
from astrodynamics_mcp.schemas.base import TimeScale
from astrodynamics_mcp.server import register_tool
from astrodynamics_mcp.units import Quantity

TimeFormat = Literal["iso", "jd", "mjd", "j2000_seconds", "unix"]

# Astropy's accepted Time scale strings. GPS is not in this list — astropy
# treats GPS as a derived attribute only.
_ASTROPY_SCALE: dict[TimeScale, str] = {
    TimeScale.UTC: "utc",
    TimeScale.TAI: "tai",
    TimeScale.TT: "tt",
    TimeScale.TDB: "tdb",
    TimeScale.UT1: "ut1",
    TimeScale.TCB: "tcb",
    TimeScale.TCG: "tcg",
}

# Astropy's native format strings for the scale-bound subset of our formats.
# `unix` and `j2000_seconds` are scale-independent absolute counters and are
# handled separately. "iso" uses astropy's isot (ISO 8601 with `T` separator,
# no timezone suffix) — non-UTC scales would be wrong to tag with `Z`, so we
# leave the suffix off and rely on the response's `scale` field to
# disambiguate.
_ASTROPY_FORMAT: dict[str, str] = {
    "iso": "isot",
    "jd": "jd",
    "mjd": "mjd",
}

# GPS time is defined as TAI offset by a constant 19 s. Used to bridge
# astropy's TAI surface with the GPS scale our enum exposes.
_GPS_TAI_OFFSET_SEC: float = 19.0

# J2000 reference epoch in TT Julian Days. Used for `j2000_seconds`.
_J2000_TT_JD: float = 2451545.0


class TimeConvertResponse(BaseModel):
    """Converted time value plus UT1/IERS metadata when relevant."""

    model_config = ConfigDict(extra="forbid")

    value: str | float = Field(
        ...,
        description=(
            "Converted time value in the requested out_format and to_scale. A string "
            "for `iso`; a float for `jd`, `mjd`, `j2000_seconds`, `unix`."
        ),
    )
    scale: TimeScale = Field(..., description="The output time scale.")
    format: TimeFormat = Field(..., description="The output format.")
    ut1_utc_seconds: Quantity | None = Field(
        None,
        description=(
            "UT1-UTC offset used by the conversion when `from_scale` or `to_scale` "
            "is UT1 (s). None for conversions that do not touch UT1."
        ),
    )
    iers_fetched_at: str | None = Field(
        None,
        description=(
            "IERS Bulletin A freshness anchor (ISO 8601 UTC). Non-null when the "
            "conversion path touched UT1 and consulted IERS."
        ),
    )


_DESCRIPTION = (
    "Convert a time value between scales (UTC / TAI / TT / TDB / UT1 / GPS / TCB / TCG) "
    "and formats (iso / jd / mjd / j2000_seconds / unix). e.g. time_convert("
    "value='2026-05-23T12:00:00', from_scale='UTC', to_scale='TAI') returns "
    "'2026-05-23T12:00:37' (the current leap-second-aware offset). UTC -> TAI is exactly "
    "TAI - UTC = 37 s as of 2026 but historical conversions need leap-second-aware "
    "machinery — don't subtract 37 yourself; the tool does it right. GPS -> UTC respects "
    "the GPS-UTC offset (currently 18 s). UTC -> UT1 sources the small (<1 s) offset from "
    "IERS Bulletin A — the response includes `ut1_utc_seconds` and an `iers_fetched_at` "
    "anchor (ISO 8601). Format hint: `iso` is the safest interchange; `jd` is the Julian "
    "Date as a float (limited precision near present-day dates); `mjd` is Modified Julian "
    "Date; `j2000_seconds` is seconds since J2000 (2000-01-01T12:00:00 TT); `unix` is "
    "seconds since 1970-01-01T00:00:00 UTC."
)


def _build_input_time(value: str | float, from_scale: TimeScale, in_format: TimeFormat) -> Any:
    """Construct an astropy.time.Time from value + scale + format.

    Format semantics:
      - ``iso`` / ``jd`` / ``mjd`` are scale-bound: the value is interpreted in
        ``from_scale``, and GPS-scale input is shifted by +19 s to recover the
        true TAI representation.
      - ``unix`` is absolute UTC by definition: ``from_scale`` is informational
        only; the value is always parsed as a UTC-rooted unix timestamp.
      - ``j2000_seconds`` is absolute and TT-anchored: ``from_scale`` is also
        informational; the value is always interpreted as TT-rooted seconds
        since the J2000 instant.
    """
    from astropy.time import Time, TimeDelta

    if in_format == "j2000_seconds":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidInputError(
                f"j2000_seconds value must be a number, got {type(value).__name__}",
                code="invalid_input.invalid_time_value",
            )
        try:
            t_tt: Any = Time(_J2000_TT_JD, format="jd", scale="tt") + TimeDelta(
                float(value), format="sec"
            )
        except (ValueError, TypeError) as exc:
            raise InvalidInputError(
                f"could not parse value {value!r} as j2000_seconds: {exc}",
                code="invalid_input.invalid_time_value",
            ) from exc
        # j2000_seconds is absolute (TT-rooted) — return as-is regardless of
        # from_scale. Conversion to GPS / TAI / UTC happens at emit time.
        return t_tt

    if in_format == "unix":
        # unix is absolute UTC by definition — parse as UTC regardless of
        # from_scale.
        try:
            return Time(value, format="unix")
        except (ValueError, TypeError) as exc:
            raise InvalidInputError(
                f"could not parse value {value!r} as unix time: {exc}",
                code="invalid_input.invalid_time_value",
            ) from exc

    # By elimination in_format is now one of iso / jd / mjd — Literal
    # validation at the wire boundary catches anything else.
    astropy_format = _ASTROPY_FORMAT[in_format]

    if from_scale is TimeScale.GPS:
        # Construct as TAI, then add 19 s to recover the true TAI time
        # (TAI = GPS + 19 s).
        try:
            t_pseudo_tai: Any = Time(value, format=astropy_format, scale="tai")
        except (ValueError, TypeError) as exc:
            raise InvalidInputError(
                f"could not parse value {value!r} as {in_format} GPS time: {exc}",
                code="invalid_input.invalid_time_value",
            ) from exc
        return t_pseudo_tai + TimeDelta(_GPS_TAI_OFFSET_SEC, format="sec")

    try:
        return Time(value, format=astropy_format, scale=_ASTROPY_SCALE[from_scale])
    except (ValueError, TypeError) as exc:
        raise InvalidInputError(
            f"could not parse value {value!r} as {in_format} time in {from_scale.value}: {exc}",
            code="invalid_input.invalid_time_value",
        ) from exc


def _emit_output(t: Any, to_scale: TimeScale, out_format: TimeFormat) -> str | float:
    """Format `t` in `(to_scale, out_format)`.

    Format semantics mirror the input side:
      - ``unix`` and ``j2000_seconds`` are absolute (UTC- and TT-rooted
        respectively) — ``to_scale`` is informational and the emitted value
        does not depend on it.
      - ``iso`` / ``jd`` / ``mjd`` are scale-bound; GPS output goes through a
        shifted-TAI proxy whose ``.isot`` / ``.jd`` read as the GPS-scale value.
    """
    from astropy.time import TimeDelta

    if out_format == "j2000_seconds":
        # TT-anchored, absolute. `t.tt.jd` is scale-invariant.
        return float((t.tt.jd - _J2000_TT_JD) * 86400.0)
    if out_format == "unix":
        # UTC-anchored, absolute. astropy's `.unix` is scale-invariant.
        return float(t.unix)

    # By elimination, out_format is now one of iso / jd / mjd — `unix` and
    # `j2000_seconds` were handled above.
    if to_scale is TimeScale.GPS:
        t_gps_proxy = t.tai - TimeDelta(_GPS_TAI_OFFSET_SEC, format="sec")
        if out_format == "iso":
            return str(t_gps_proxy.isot)
        return float(getattr(t_gps_proxy, _ASTROPY_FORMAT[out_format]))

    t_scaled = getattr(t, _ASTROPY_SCALE[to_scale])
    if out_format == "iso":
        return str(t_scaled.isot)
    return float(getattr(t_scaled, _ASTROPY_FORMAT[out_format]))


def _ut1_utc_quantity(t_in: Any) -> Quantity:
    """Extract the UT1-UTC offset (s) astropy applied for this Time."""
    raw = t_in.utc.delta_ut1_utc
    # astropy returns a 0-d ndarray for a scalar Time; coerce to float.
    return Quantity(value=float(raw), unit="s")


@register_tool(name="time_convert", description=_DESCRIPTION)
async def time_convert(
    value: str | float,
    from_scale: TimeScale,
    to_scale: TimeScale,
    in_format: TimeFormat = "iso",
    out_format: TimeFormat = "iso",
) -> TimeConvertResponse:
    needs_iers = TimeScale.UT1 in (from_scale, to_scale)

    iers_fetched_at: str | None = None
    if needs_iers:
        from astrodynamics_mcp.data.iers import load_iers

        try:
            iers_status = load_iers()
        except Exception as exc:
            raise UpstreamError(
                f"IERS Bulletin A unavailable: {exc}",
                code="upstream.iers_unavailable",
                original_exception=exc,
            ) from exc
        iers_fetched_at = iers_status.last_updated

    t_in = _build_input_time(value, from_scale, in_format)
    out_value = _emit_output(t_in, to_scale, out_format)

    ut1_utc: Quantity | None = None
    if needs_iers:
        try:
            ut1_utc = _ut1_utc_quantity(t_in)
        except Exception as exc:
            raise UpstreamError(
                f"failed to read UT1-UTC offset from astropy: {exc}",
                code="upstream.iers_unavailable",
                original_exception=exc,
            ) from exc

    return TimeConvertResponse(
        value=out_value,
        scale=to_scale,
        format=out_format,
        ut1_utc_seconds=ut1_utc,
        iers_fetched_at=iers_fetched_at,
    )
