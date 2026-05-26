"""`sgp4_propagate` tool — SGP4/SDP4 propagation against the `sgp4` library.

TEME is the native SGP4 output frame. Other frames (`ICRF`, `GCRS`, `ITRS`,
`CIRS`) are reached via :mod:`astropy.coordinates`; each propagated state
carries its own ``obstime`` so the rotation between Earth-fixed and inertial
frames stays correct per-epoch.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sgp4 import omm
from sgp4.api import SGP4_ERRORS, Satrec

from astrodynamics_mcp.errors import InvalidInputError, UpstreamError
from astrodynamics_mcp.schemas.base import Epoch, Frame, StateVector, TleLines, TleOmm
from astrodynamics_mcp.server import register_tool
from astrodynamics_mcp.units import QuantityVector

# Frames this tool can emit. The full astrodynamics_mcp.Frame enum is wider
# (TIRS, IAU body-fixed) — those lookups land in the dedicated frame_transform
# tool; we cap this tool to the five frames the SGP4 → astropy path is
# well-defined for so the error surface stays predictable.
_SUPPORTED_FRAMES: frozenset[Frame] = frozenset(
    {Frame.TEME, Frame.ICRF, Frame.GCRS, Frame.ITRS, Frame.CIRS}
)

# How many states the default output='summary' response carries when the
# caller passes more than this many epochs. A 12-state cap (~2 KB JSON)
# keeps the response under the small-model 8000-token input limit while
# still characterising the trajectory shape — first and last epoch always
# retained, the rest evenly spaced.
_SUMMARY_STATE_CAP = 12


class Sgp4PropagateResponse(BaseModel):
    """A list of propagated state vectors.

    In the default ``output="summary"`` mode the list is capped to
    :data:`_SUMMARY_STATE_CAP` evenly-spaced entries (always including
    the first and last requested epoch). Pass ``output="full"`` to receive
    one state per input epoch.
    """

    model_config = ConfigDict(extra="forbid")

    states: list[StateVector] = Field(
        ...,
        description=(
            "Propagated state vectors in the requested frame. In output='full' there is one "
            "entry per input epoch; in the default output='summary' the list is subsampled to "
            "at most twelve evenly-spaced entries (first and last always retained). Position "
            "in km, velocity in km/s, units explicit on every numeric field."
        ),
    )


_DESCRIPTION = (
    "Propagate a TLE forward in time via SGP4/SDP4, returning Cartesian state "
    "vectors at the requested epochs. TLE input is either two raw 69-char lines "
    "(`{line1, line2}`) or a parsed OMM JSON object (`{omm: {...}}`). "
    "e.g. sgp4_propagate(tle={line1: ..., line2: ...}, epochs=['2026-05-23T12:00:00Z'], "
    "frame='TEME') returns one state vector in TEME. Epochs are UTC ISO 8601 with a "
    "mandatory time component (e.g. '2026-05-23T12:00:00Z') — a bare date is rejected. "
    "Default frame is TEME (SGP4's native output); for ICRF, GCRS, ITRS, or CIRS the "
    "tool transforms via astropy with per-epoch obstime. A list of one epoch is fine; "
    "1000+ epochs is also fine — propagation cost scales linearly. Output shaping: "
    "the default output='summary' returns at most twelve evenly-spaced states (first "
    "and last epoch always included) so the response fits small-model input caps; pass "
    "output='full' to receive one state per epoch when you need the dense series. "
    "SGP4 propagation failures (decayed satellite, deep-space epoch beyond SDP4 validity) "
    "surface as `upstream.sgp4_failure`."
)


def _build_satrec(tle: TleLines | TleOmm) -> Satrec:
    """Construct a Satrec from either TLE shape and assert init succeeded."""
    if isinstance(tle, TleLines):
        satrec = Satrec.twoline2rv(tle.line1, tle.line2)
    else:
        satrec = Satrec()
        try:
            omm.initialize(satrec, tle.omm)
        except (KeyError, TypeError, ValueError) as exc:
            # An OMM dict missing required fields (or carrying wrong types)
            # would crash deep inside the upstream initialiser — wrap as a
            # typed upstream error so the LLM sees a stable code.
            raise UpstreamError(
                f"failed to initialise Satrec from OMM input: {exc}",
                code="upstream.sgp4_failure",
                original_exception=exc,
            ) from exc

    err_code: int = int(satrec.error)
    if err_code != 0:
        raise UpstreamError(
            f"SGP4 initialisation failed: {SGP4_ERRORS.get(err_code, 'unknown error')}",
            code="upstream.sgp4_failure",
            data={"sgp4_error_code": err_code},
        )
    return satrec


def _propagate_one_teme(
    satrec: Satrec,
    epoch: str,
) -> tuple[list[float], list[float], Any]:
    """Propagate to *epoch* and return (r_teme_km, v_teme_kmps, astropy_time)."""
    from astropy.time import Time

    time = Time(epoch, scale="utc")
    jd1 = float(time.jd1)
    jd2 = float(time.jd2)
    err_code, r, v = satrec.sgp4(jd1, jd2)
    err_code = int(err_code)
    if err_code != 0:
        raise UpstreamError(
            f"SGP4 propagation failed at epoch {epoch}: "
            f"{SGP4_ERRORS.get(err_code, 'unknown error')}",
            code="upstream.sgp4_failure",
            data={"sgp4_error_code": err_code, "epoch": epoch},
        )
    return [float(r[0]), float(r[1]), float(r[2])], [float(v[0]), float(v[1]), float(v[2])], time


def _state_from_teme(
    r_teme: list[float],
    v_teme: list[float],
    epoch: str,
) -> StateVector:
    return StateVector(
        r=QuantityVector(value=r_teme, unit="km"),
        v=QuantityVector(value=v_teme, unit="km/s"),
        frame=Frame.TEME,
        epoch=epoch,
    )


def _transform_to_frame(
    r_teme: list[float],
    v_teme: list[float],
    epoch_time: Any,
    target: Frame,
    epoch: str,
) -> StateVector:
    """Convert a TEME (r, v) at *epoch_time* into *target* via the shared helper."""
    from astrodynamics_mcp.tools._astropy_frames import transform_state

    r_out, v_out = transform_state(
        r_teme,
        v_teme,
        from_frame=Frame.TEME,
        to_frame=target,
        epoch_time=epoch_time,
    )
    return StateVector(
        r=QuantityVector(value=r_out, unit="km"),
        v=QuantityVector(value=v_out, unit="km/s"),
        frame=target,
        epoch=epoch,
    )


def _summary_indices(n: int, cap: int) -> list[int]:
    """Pick up to *cap* indices from ``range(n)`` for the summary response.

    Always includes 0 and ``n - 1`` so the caller sees the trajectory's
    endpoints; the remaining slots are evenly spaced between them. Returns
    ``list(range(n))`` unchanged when ``n <= cap``.
    """
    if n <= cap:
        return list(range(n))
    if cap == 1:
        return [0]
    return [round(i * (n - 1) / (cap - 1)) for i in range(cap)]


@register_tool(name="sgp4_propagate", description=_DESCRIPTION)
async def sgp4_propagate(
    tle: TleLines | TleOmm,
    epochs: list[Epoch],
    frame: Frame = Frame.TEME,
    output: Literal["summary", "full"] = "summary",
) -> Sgp4PropagateResponse:
    if frame not in _SUPPORTED_FRAMES:
        raise InvalidInputError(
            f"frame {frame.value!r} is not supported by sgp4_propagate; "
            f"supported frames are {sorted(f.value for f in _SUPPORTED_FRAMES)}. "
            "For TIRS / IAU body-fixed frames use the frame_transform tool.",
            code="invalid_input.unsupported_frame",
        )

    satrec = _build_satrec(tle)

    states: list[StateVector] = []
    for epoch in epochs:
        # ``Epoch`` is a string alias with a BeforeValidator on the schema
        # boundary; at the function body it is already a plain str.
        r_teme, v_teme, epoch_time = _propagate_one_teme(satrec, epoch)
        if frame is Frame.TEME:
            states.append(_state_from_teme(r_teme, v_teme, epoch))
        else:
            states.append(_transform_to_frame(r_teme, v_teme, epoch_time, frame, epoch))

    if output == "summary":
        kept = _summary_indices(len(states), _SUMMARY_STATE_CAP)
        states = [states[i] for i in kept]

    return Sgp4PropagateResponse(states=states)
