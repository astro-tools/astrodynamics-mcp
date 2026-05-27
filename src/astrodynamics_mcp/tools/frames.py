"""`frame_transform` tool — state-vector frame transforms via astropy.

Delegates the actual ICRF / GCRS / ITRS / TEME / CIRS / IAU_EARTH math to
:mod:`astrodynamics_mcp.tools._astropy_frames` (shared with
``sgp4_propagate``). Adds the input-frame validation, the IAU body-fixed
guard rails, the IERS freshness anchor, and the input position/velocity
unit normalisation that the helper assumes.

TIRS, IAU_MARS, and IAU_MOON are deliberately rejected with typed errors:
astropy does not expose TIRS as a transform endpoint (it lives inside
the CIRS↔ITRS chain), and there is no astropy support for body-fixed
frames of Mars or the Moon — those need a SPICE-backed rotation matrix
that is not yet available here.
"""

from __future__ import annotations

from typing import Annotated, Any

from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from astrodynamics_mcp.errors import InvalidInputError, UpstreamError
from astrodynamics_mcp.schemas.base import Epoch, Frame, StateVector
from astrodynamics_mcp.server import register_tool
from astrodynamics_mcp.tools._astropy_frames import (
    EARTH_ROTATING_FRAMES,
    SUPPORTED_FRAMES,
    transform_state,
)
from astrodynamics_mcp.units import QuantityVector

# Conversion factors from a supported length / velocity unit into the
# canonical km / km/s that the shared helper assumes.
_KM_PER_UNIT: dict[str, float] = {"km": 1.0, "m": 1e-3, "AU": 149597870.7}
_KMPS_PER_UNIT: dict[str, float] = {"km/s": 1.0, "m/s": 1e-3}

# Body-fixed frames that exist in the Frame enum but cannot serve as
# transform endpoints from Earth-centered inputs (the only inputs the v0.1
# surface accepts). Mars and Moon body-fixed rotations need SPICE kernels.
_BODY_MISMATCH_FRAMES: frozenset[Frame] = frozenset({Frame.IAU_MARS, Frame.IAU_MOON})


class FrameTransformResponse(BaseModel):
    """Transformed state plus an IERS freshness anchor when EOP data was used."""

    model_config = ConfigDict(extra="forbid")

    state: StateVector = Field(
        ..., description="Transformed state vector in the requested `to_frame`."
    )
    iers_bulletin_a_fetched_at: str | None = Field(
        None,
        description=(
            "IERS Bulletin A freshness anchor (ISO 8601 UTC). Non-null whenever the "
            "transform path touched an Earth-rotating frame (ITRS, GCRS, CIRS, "
            "IAU_EARTH), which depend on Earth-orientation parameters."
        ),
    )


_DESCRIPTION = (
    "Transform a Cartesian state vector from its current frame into a different "
    "reference frame. e.g. frame_transform(state={r: ..., v: ..., frame: 'TEME', "
    "epoch: '2024-01-01T12:00:00Z'}, to_frame='ICRF') re-expresses the state in "
    "ICRF. Supported frames: ICRF (barycentric inertial), GCRS (Earth-centred "
    "inertial), ITRS (Earth-fixed rotating), TEME (SGP4's native output), CIRS "
    "(Earth-rotating intermediate), IAU_EARTH (alias for ITRS — same Earth-body-"
    "fixed frame, different naming). The `epoch` arg defaults to the state's own "
    "epoch; pass an override only when you deliberately want to transform 'as if "
    "the state were at a different time'. Earth-fixed frames (ITRS / IAU_EARTH) "
    "rotate at sidereal rate (~7.292e-5 rad/s); the epoch must match the state's "
    "epoch to within a few seconds or the transformed velocity will be off by ~1 "
    "km/s — don't pass `epoch` to override a stale state's own epoch. Epochs are "
    "UTC ISO 8601 with a mandatory time component. TIRS is not yet supported as a "
    "transform endpoint (use ITRS instead); IAU_MARS and IAU_MOON require SPICE-"
    "backed body-fixed rotations not available in this tool."
)


def _normalize_state_to_km(state: StateVector) -> tuple[list[float], list[float]]:
    """Return (r_km, v_kmps) from the state's wire units."""
    r_factor = _KM_PER_UNIT[state.r.unit]
    v_factor = _KMPS_PER_UNIT[state.v.unit]
    return (
        [float(x) * r_factor for x in state.r.value],
        [float(x) * v_factor for x in state.v.value],
    )


def _validate_frame_endpoints(from_frame: Frame, to_frame: Frame) -> None:
    """Raise InvalidInputError for unsupported source / destination frames."""
    if to_frame in _BODY_MISMATCH_FRAMES:
        raise InvalidInputError(
            f"cannot transform an Earth-centred state to body-fixed frame "
            f"{to_frame.value} (body mismatch); Mars and Moon body-fixed frames "
            "need SPICE-backed rotation matrices that are not available in this "
            "tool. Convert to ICRF or GCRS for the inertial equivalent.",
            code="invalid_input.body_mismatch",
        )
    if to_frame is Frame.TIRS or from_frame is Frame.TIRS:
        raise InvalidInputError(
            "TIRS is not currently supported as a transform endpoint; use ITRS "
            "(or, for the intermediate rotating frame, CIRS).",
            code="invalid_input.unsupported_frame_transform",
        )
    if from_frame in _BODY_MISMATCH_FRAMES:
        raise InvalidInputError(
            f"state frame {from_frame.value} is not supported as a transform source",
            code="invalid_input.unsupported_frame_transform",
        )
    if from_frame not in SUPPORTED_FRAMES:
        raise InvalidInputError(
            f"state frame {from_frame.value!r} is not a supported transform source; "
            f"supported: {sorted(f.value for f in SUPPORTED_FRAMES)}",
            code="invalid_input.unsupported_frame_transform",
        )


@register_tool(
    name="frame_transform",
    description=_DESCRIPTION,
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
)
async def frame_transform(
    state: Annotated[
        StateVector,
        Field(
            description=(
                "Input state vector carrying position (km), velocity (km/s), "
                "the source `frame`, and the `epoch` at which it is valid. "
                "Supported frames: ICRF, GCRS, ITRS, TEME, CIRS, IAU_EARTH. "
                "TIRS, IAU_MARS, and IAU_MOON are deliberately rejected — "
                "astropy does not expose them as transform endpoints at v0.1."
            ),
        ),
    ],
    to_frame: Annotated[
        Frame,
        Field(
            description=(
                "Target frame for the output state. Same supported-frame list "
                "as the input. Identity (to_frame == state.frame) is a valid "
                "no-op that still returns a structured response."
            ),
        ),
    ],
    epoch: Annotated[
        Epoch | None,
        Field(
            description=(
                "Optional override for the epoch used in the transform, UTC ISO "
                "8601 with a mandatory time component. When omitted, the tool "
                "uses `state.epoch`. Override is useful when re-evaluating an "
                "Earth-rotating-frame state at a different time than the state's "
                "native validity epoch."
            ),
        ),
    ] = None,
) -> FrameTransformResponse:
    from_frame = state.frame
    target_epoch: str = epoch if epoch is not None else state.epoch

    _validate_frame_endpoints(from_frame, to_frame)

    needs_iers = (from_frame in EARTH_ROTATING_FRAMES) or (to_frame in EARTH_ROTATING_FRAMES)
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

    r_km, v_kmps = _normalize_state_to_km(state)

    from astropy.time import Time

    epoch_time: Any = Time(target_epoch, scale="utc")

    try:
        r_out, v_out = transform_state(
            r_km,
            v_kmps,
            from_frame=from_frame,
            to_frame=to_frame,
            epoch_time=epoch_time,
        )
    except InvalidInputError:
        raise
    except Exception as exc:
        raise UpstreamError(
            f"astropy frame transform {from_frame.value} -> {to_frame.value} failed: {exc}",
            code="upstream.astropy_transform_failed",
            original_exception=exc,
        ) from exc

    return FrameTransformResponse(
        state=StateVector(
            r=QuantityVector(value=r_out, unit="km"),
            v=QuantityVector(value=v_out, unit="km/s"),
            frame=to_frame,
            epoch=target_epoch,
        ),
        iers_bulletin_a_fetched_at=iers_fetched_at,
    )
