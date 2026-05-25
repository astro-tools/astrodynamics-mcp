"""Shared `astropy.coordinates` Cartesian state-vector transform helper.

Used by ``sgp4_propagate`` (TEME-only source) and ``frame_transform``
(arbitrary source / arbitrary target). Centralises the
``CartesianRepresentation`` + ``CartesianDifferential`` construction, the
Frame enum -> astropy class mapping, and the per-frame ``obstime``
handling so velocity-bearing transforms stay correct across every
caller.

The helper only handles the astropy-supported subset of the Frame enum:
ICRF, GCRS, ITRS, CIRS, TEME, plus IAU_EARTH routed through ITRS (same
Earth-body-fixed frame, different naming convention). TIRS, IAU_MARS,
and IAU_MOON are not exposed as transform endpoints by astropy and are
the caller's responsibility to reject with a typed error.
"""

from __future__ import annotations

from typing import Any

from astrodynamics_mcp.errors import InvalidInputError
from astrodynamics_mcp.schemas.base import Frame

# Frames this helper can transform between. The wider Frame enum includes
# TIRS / IAU_MARS / IAU_MOON which astropy does not expose as transform
# endpoints — callers must guard those before delegating here.
SUPPORTED_FRAMES: frozenset[Frame] = frozenset(
    {Frame.ICRF, Frame.GCRS, Frame.ITRS, Frame.CIRS, Frame.TEME, Frame.IAU_EARTH}
)

# Frames whose astropy transform paths route through Earth-orientation
# parameters (IERS Bulletin A). Callers can use this to decide whether
# to surface an `iers_fetched_at` freshness anchor with the response.
EARTH_ROTATING_FRAMES: frozenset[Frame] = frozenset(
    {Frame.ITRS, Frame.GCRS, Frame.CIRS, Frame.IAU_EARTH}
)


def _astropy_frame_class(frame: Frame) -> tuple[Any, bool]:
    """Return (astropy_class, takes_obstime) for the given Frame.

    ``IAU_EARTH`` is routed through ``ITRS`` — the IAU body-fixed Earth
    frame and the IERS standard ITRS describe the same Earth-fixed
    rotating frame under different naming conventions; routing through
    ITRS keeps the astropy transform chain canonical and avoids having
    to maintain a parallel `IAU_EARTH` axis path.
    """
    from astropy.coordinates import CIRS, GCRS, ICRS, ITRS, TEME

    mapping: dict[Frame, tuple[Any, bool]] = {
        Frame.TEME: (TEME, True),
        Frame.ICRF: (ICRS, False),
        Frame.GCRS: (GCRS, True),
        Frame.ITRS: (ITRS, True),
        Frame.CIRS: (CIRS, True),
        Frame.IAU_EARTH: (ITRS, True),
    }
    if frame not in mapping:
        raise InvalidInputError(
            f"frame {frame.value!r} is not supported as a transform endpoint; "
            f"supported: {sorted(f.value for f in SUPPORTED_FRAMES)}",
            code="invalid_input.unsupported_frame_transform",
        )
    return mapping[frame]


def transform_state(
    r_km: list[float],
    v_kmps: list[float],
    *,
    from_frame: Frame,
    to_frame: Frame,
    epoch_time: Any,
) -> tuple[list[float], list[float]]:
    """Transform a Cartesian state from `from_frame` to `to_frame` via astropy.

    Returns the transformed ``(r_km, v_kmps)`` tuple. ``obstime`` is set
    on both source and target frames where applicable (every astropy
    frame except ``ICRS``, which is barycentric and epoch-agnostic).
    """
    import astropy.units as u
    from astropy.coordinates import CartesianDifferential, CartesianRepresentation

    source_cls, source_takes_obstime = _astropy_frame_class(from_frame)
    target_cls, target_takes_obstime = _astropy_frame_class(to_frame)

    # Build a Cartesian representation carrying both position and velocity
    # so the frame transform carries velocity through. A position-only
    # representation silently drops velocity at the target end.
    r_rep = CartesianRepresentation(r_km[0] * u.km, r_km[1] * u.km, r_km[2] * u.km)
    v_diff = CartesianDifferential(
        v_kmps[0] * u.km / u.s, v_kmps[1] * u.km / u.s, v_kmps[2] * u.km / u.s
    )
    rep = r_rep.with_differentials(v_diff)

    source = source_cls(rep, obstime=epoch_time) if source_takes_obstime else source_cls(rep)
    target = (
        source.transform_to(target_cls(obstime=epoch_time))
        if target_takes_obstime
        else source.transform_to(target_cls())
    )

    cartesian = target.cartesian
    r_out = cartesian.xyz.to_value(u.km)
    v_out = cartesian.differentials["s"].d_xyz.to_value(u.km / u.s)
    return (
        [float(r_out[0]), float(r_out[1]), float(r_out[2])],
        [float(v_out[0]), float(v_out[1]), float(v_out[2])],
    )
