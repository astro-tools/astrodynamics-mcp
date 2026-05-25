"""`bplane_target` tool — B-plane element calculation and one-step targeting.

Given a hyperbolic planetocentric state, compute the B-plane coordinates
(B·T, B·R), the hyperbolic excess velocity vector v_∞, and the asymptote
declination. If the caller supplies B·T and/or B·R targets, the tool also
returns the linearised one-step Δv at the input epoch that drives the
B-plane toward those coordinates.

Frame: the input state's r and v must be planetocentric — relative to the
target body's centre — in an inertial reference frame. The tool does not
transform the state; it consumes the components as-supplied. The
T̂ projection reference K̂ is the +Z axis of the input frame, which matches
the Vallado / JPL Horizons B-plane convention for ICRF / GCRS frames.

The targeting solver is linearised about the input state. Apply the
returned Δv and re-call to converge if the residual is large; the linear
solver is accurate when ‖Δv‖ is small relative to ‖v_∞‖.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from astrodynamics_mcp.errors import InvalidInputError
from astrodynamics_mcp.schemas.base import Epoch, StateVector
from astrodynamics_mcp.server import register_tool
from astrodynamics_mcp.units import Quantity, QuantityVector

# v0.1 body table: μ (km³/s²) and equatorial radius (km). Values from
# JPL ssd.jpl.nasa.gov; radius is the IAU equatorial radius. Coverage is
# the same eight bodies the issue calls out — no Sun, no Mercury, no Pluto.
_BODY_PARAMETERS: dict[str, tuple[float, float]] = {
    "earth": (3.986004418e5, 6378.137),
    "mars": (4.282837e4, 3396.2),
    "venus": (3.24858592e5, 6051.8),
    "jupiter": (1.26686534e8, 71492.0),
    "saturn": (3.7931187e7, 60268.0),
    "uranus": (5.793939e6, 25559.0),
    "neptune": (6.836529e6, 24764.0),
    "moon": (4.9028e3, 1737.4),
}

# Unit-conversion table for the input StateVector. Pydantic enforces the
# unit is one of these at the schema layer; the tool normalises everything
# to km / km·s⁻¹ before touching the math.
_LENGTH_KM: dict[str, float] = {"km": 1.0, "m": 1e-3, "AU": 149597870.7}
_VELOCITY_KM_S: dict[str, float] = {"km/s": 1.0, "m/s": 1e-3}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class BplaneResidual(BaseModel):
    """Post-Δv recompute of the B-plane coords, plus the constraint-space error.

    ``actual_b_t_after_dv`` and ``actual_b_r_after_dv`` are computed by
    applying the linear Δv to the input velocity and recomputing the
    B-plane geometry nonlinearly — the ground truth the caller will see
    once the burn is executed. ``magnitude`` is the Euclidean distance
    from the requested target in the dimensions the caller constrained
    (1-D if only one target was supplied, 2-D if both).
    """

    model_config = ConfigDict(extra="forbid")

    actual_b_t_after_dv: Quantity = Field(
        ...,
        description=(
            "B·T after applying the returned Δv to the input velocity and recomputing the "
            "B-plane geometry exactly (no linearisation). Compare against the requested "
            "target_btt_km to gauge the linear solver's accuracy."
        ),
        examples=[{"value": 1995.0, "unit": "km"}],
    )
    actual_b_r_after_dv: Quantity = Field(
        ...,
        description=(
            "B·R after applying the returned Δv to the input velocity and recomputing the "
            "B-plane geometry exactly (no linearisation)."
        ),
        examples=[{"value": -3.0, "unit": "km"}],
    )
    magnitude: Quantity = Field(
        ...,
        description=(
            "Euclidean distance between the post-Δv (B·T, B·R) and the requested target "
            "in the constrained dimensions, km. A small magnitude means the linear solver "
            "hit the target; a large magnitude means a follow-up iteration is needed."
        ),
        examples=[{"value": 5.4, "unit": "km"}],
    )


class BplaneTargetResponse(BaseModel):
    """Response from :func:`bplane_target`."""

    model_config = ConfigDict(extra="forbid")

    b_r: Quantity = Field(
        ...,
        description="B·R component of the impact-parameter vector at the input state, km.",
        examples=[{"value": 0.0, "unit": "km"}],
    )
    b_t: Quantity = Field(
        ...,
        description="B·T component of the impact-parameter vector at the input state, km.",
        examples=[{"value": -8660.25, "unit": "km"}],
    )
    v_infinity: QuantityVector = Field(
        ...,
        description=(
            "Hyperbolic excess velocity vector at the incoming asymptote, km/s, in the "
            "input state's reference frame. Magnitude is the scalar v_∞; direction is the "
            "incoming asymptote unit vector S^."
        ),
        examples=[{"value": [1.464, 2.535, 0.0], "unit": "km/s"}],
    )
    asymptote_declination: Quantity = Field(
        ...,
        description=(
            "Declination of the incoming-asymptote unit vector S^ above the input frame's "
            "XY plane, deg. ICRF/GCRS users get an ecliptic-style declination; body-fixed "
            "frame users get the body's equatorial declination."
        ),
        examples=[{"value": 0.0, "unit": "deg"}],
    )
    dv_required: QuantityVector | None = Field(
        None,
        description=(
            "Linearised one-step Δv at the input state's epoch that drives (B·T, B·R) to "
            "the requested targets. None when the tool ran in read-only mode "
            "(both target_btr_km and target_btt_km omitted)."
        ),
        examples=[{"value": [0.0, 0.0006, 0.0], "unit": "km/s"}],
    )
    residual: BplaneResidual | None = Field(
        None,
        description=(
            "Nonlinear recompute of the B-plane coords after applying dv_required, plus "
            "the constraint-space error. None in read-only mode."
        ),
    )


# ---------------------------------------------------------------------------
# Tool description (subject to server_lint)
# ---------------------------------------------------------------------------


_DESCRIPTION = (
    "Compute B-plane coordinates (B·T, B·R), hyperbolic excess velocity v_∞, and asymptote "
    "declination for a hyperbolic planetocentric flyby, and optionally return the linearised "
    "one-step Δv at the input epoch that drives the B-plane to caller-supplied target "
    "coordinates. e.g. bplane_target(state={'r': {'value': [5000, 0, 0], 'unit': 'km'}, "
    "'v': {'value': [0, 5.069, 0], 'unit': 'km/s'}, 'frame': 'ICRF', 'epoch': "
    "'2026-12-01T00:00:00Z'}, target_body='mars', target_epoch='2026-12-01T00:00:00Z', "
    "target_btr_km=1000.0) returns the current B-plane state plus the Δv that drives "
    "B·R toward 1000 km. Read-only mode (both target_btr_km and target_btt_km omitted) "
    "skips the targeting step. The input state must be planetocentric — r and v relative "
    "to target_body's centre — in an inertial frame (ICRF / GCRS / CIRS etc.); the tool "
    "does not transform the state. Epochs are UTC ISO 8601 with a mandatory time component "
    "('2026-12-01T00:00:00Z'). target_epoch is the closest-approach epoch the caller is "
    "designing for; the Δv applies at state.epoch and is documented as such. The targeting "
    "solver is linearised about the input state; for large required Δv apply the result and "
    "re-call to converge, or hand off to a multi-iteration targeter. Non-hyperbolic states "
    "(specific energy ≤ 0) raise invalid_input.not_hyperbolic; unknown target_body raises "
    "invalid_input.unknown_body."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_body(name: str, *, field: str) -> str:
    if not isinstance(name, str):
        raise InvalidInputError(
            f"{field} must be a string body name, got {type(name).__name__}",
            code="invalid_input.body_not_a_string",
        )
    key = name.lower()
    if key not in _BODY_PARAMETERS:
        raise InvalidInputError(
            f"unknown body {name!r}; supported: {sorted(_BODY_PARAMETERS)}",
            code="invalid_input.unknown_body",
        )
    return key


def _normalise_state(state: StateVector) -> tuple[np.ndarray, np.ndarray]:
    """Return (r_km, v_km_s) numpy arrays from a StateVector, regardless of units."""
    r = np.asarray(state.r.value, dtype=float) * _LENGTH_KM[state.r.unit]
    v = np.asarray(state.v.value, dtype=float) * _VELOCITY_KM_S[state.v.unit]
    return r, v


def _validate_target_scalar(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInputError(
            f"{field} must be a number or null, got {type(value).__name__}",
            code="invalid_input.value_not_a_number",
        )
    if not np.isfinite(float(value)):
        raise InvalidInputError(
            f"{field} must be a finite number, got {value!r}",
            code="invalid_input.value_not_a_number",
        )
    return float(value)


def _bplane_elements(
    r_vec: np.ndarray, v_vec: np.ndarray, mu: float
) -> tuple[float, float, np.ndarray, float]:
    """Return (b_t, b_r, v_inf_vec_km_s, asymptote_declination_deg) for a hyperbolic state."""
    r_mag = float(np.linalg.norm(r_vec))
    v_sq = float(v_vec @ v_vec)
    energy = v_sq / 2.0 - mu / r_mag
    if energy <= 0.0:
        raise InvalidInputError(
            f"state is not hyperbolic — specific energy {energy:.6g} km^2/s^2 is non-positive "
            "(elliptic or parabolic). B-plane targeting requires a hyperbolic flyby.",
            code="invalid_input.not_hyperbolic",
        )
    v_inf_scalar = float(np.sqrt(2.0 * energy))

    h_vec = np.cross(r_vec, v_vec)
    h_mag = float(np.linalg.norm(h_vec))
    if h_mag == 0.0:
        raise InvalidInputError(
            "state has zero angular momentum (r and v parallel); B-plane is undefined.",
            code="invalid_input.zero_angular_momentum",
        )

    e_vec = ((v_sq - mu / r_mag) * r_vec - float(r_vec @ v_vec) * v_vec) / mu
    e_mag = float(np.linalg.norm(e_vec))
    if e_mag <= 1.0:
        # The energy check above already covers ε > 0 ⇒ e > 1 in two-body
        # mechanics, but floating-point noise on grazing-hyperbola geometries
        # can push e marginally below 1. Treat that as a typed error rather
        # than letting the subsequent sqrt(e²-1) silently blow up.
        raise InvalidInputError(
            f"computed eccentricity {e_mag:.6g} is not strictly hyperbolic (e > 1 required)",
            code="invalid_input.not_hyperbolic",
        )

    # Incoming-asymptote unit vector S^ from perifocal-frame (1/e, √(e²-1)/e, 0).
    p_hat = e_vec / e_mag
    q_hat = np.cross(h_vec, e_vec) / (h_mag * e_mag)
    sqrt_em1 = float(np.sqrt(e_mag * e_mag - 1.0))
    s_hat = p_hat / e_mag + q_hat * sqrt_em1 / e_mag

    # Impact-parameter vector: perpendicular to S^ in the orbit plane, pointing
    # from the focus to the incoming asymptote's closest approach.
    b_hat = -sqrt_em1 / e_mag * p_hat + 1.0 / e_mag * q_hat
    b_mag = h_mag / v_inf_scalar  # equivalent to |a|·√(e²-1)
    b_vec = b_mag * b_hat

    # B-plane T̂ / R̂ basis. K̂ = +Z of input frame; degenerate S^ ∥ K̂
    # falls back to K̂ = +X (matches Vallado §12).
    k_hat = np.array([0.0, 0.0, 1.0])
    cross_sk = np.cross(s_hat, k_hat)
    if float(np.linalg.norm(cross_sk)) < 1e-10:
        k_hat = np.array([1.0, 0.0, 0.0])
        cross_sk = np.cross(s_hat, k_hat)
    t_hat = cross_sk / float(np.linalg.norm(cross_sk))
    r_hat = np.cross(s_hat, t_hat)

    b_t = float(b_vec @ t_hat)
    b_r = float(b_vec @ r_hat)
    v_inf_vec = v_inf_scalar * s_hat
    dec_deg = float(np.degrees(np.arcsin(np.clip(s_hat[2], -1.0, 1.0))))
    return b_t, b_r, v_inf_vec, dec_deg


def _btr_btt(r_vec: np.ndarray, v_vec: np.ndarray, mu: float) -> tuple[float, float]:
    """Return (b_t, b_r) only — used inside the numerical Jacobian's hot loop."""
    b_t, b_r, _, _ = _bplane_elements(r_vec, v_vec, mu)
    return b_t, b_r


def _jacobian(r_vec: np.ndarray, v_vec: np.ndarray, mu: float, v_inf_scalar: float) -> np.ndarray:
    """Return the 2x3 Jacobian ∂(b_t, b_r)/∂v via central differences.

    Step size scales with v_∞ so the perturbation is dimensionally sensible
    across two orders of magnitude in flyby energy (Mars vs Jupiter).
    """
    delta = max(1e-3 * v_inf_scalar, 1e-6)
    jac = np.zeros((2, 3))
    for axis in range(3):
        v_plus = v_vec.copy()
        v_plus[axis] += delta
        bt_plus, br_plus = _btr_btt(r_vec, v_plus, mu)
        v_minus = v_vec.copy()
        v_minus[axis] -= delta
        bt_minus, br_minus = _btr_btt(r_vec, v_minus, mu)
        jac[0, axis] = (bt_plus - bt_minus) / (2.0 * delta)
        jac[1, axis] = (br_plus - br_minus) / (2.0 * delta)
    return jac


def _min_norm_solve(jac: np.ndarray, delta_b: np.ndarray) -> np.ndarray:
    """Least-squares solution to ``jac @ dv = delta_b`` with ‖dv‖ minimised.

    Routes through ``numpy.linalg.lstsq`` so the underdetermined rank-2-of-3
    case (full B-plane targeting) and the rank-1 case (single-axis targeting)
    both fall out of the same call.
    """
    dv, *_ = np.linalg.lstsq(jac, delta_b, rcond=None)
    return np.asarray(dv, dtype=float)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@register_tool(name="bplane_target", description=_DESCRIPTION)
async def bplane_target(
    state: StateVector,
    target_body: str,
    target_epoch: Epoch,
    target_btr_km: float | None = None,
    target_btt_km: float | None = None,
) -> BplaneTargetResponse:
    body_key = _validate_body(target_body, field="target_body")
    mu, _radius_km = _BODY_PARAMETERS[body_key]
    btr_target = _validate_target_scalar(target_btr_km, field="target_btr_km")
    btt_target = _validate_target_scalar(target_btt_km, field="target_btt_km")
    # `target_epoch` is consumed by the schema's Epoch validator; the v0.1
    # solver applies its Δv at state.epoch and carries target_epoch only
    # for traceability. Referenced via `del` so mypy/ruff see it used.
    del target_epoch

    r_vec, v_vec = _normalise_state(state)
    b_t, b_r, v_inf_vec, dec_deg = _bplane_elements(r_vec, v_vec, mu)
    v_inf_scalar = float(np.linalg.norm(v_inf_vec))

    dv_required: QuantityVector | None = None
    residual: BplaneResidual | None = None
    if btr_target is not None or btt_target is not None:
        constraints: list[tuple[int, float, float]] = []
        if btt_target is not None:
            constraints.append((0, btt_target, b_t))  # row 0 of full Jacobian → b_t
        if btr_target is not None:
            constraints.append((1, btr_target, b_r))  # row 1 of full Jacobian → b_r

        full_jac = _jacobian(r_vec, v_vec, mu, v_inf_scalar)
        rows = [c[0] for c in constraints]
        jac = full_jac[rows, :]
        delta_b = np.asarray([c[1] - c[2] for c in constraints], dtype=float)
        dv_vec = _min_norm_solve(jac, delta_b)

        v_after = v_vec + dv_vec
        b_t_after, b_r_after, _, _ = _bplane_elements(r_vec, v_after, mu)
        actual = np.asarray(
            [b_t_after if c[0] == 0 else b_r_after for c in constraints], dtype=float
        )
        target_arr = np.asarray([c[1] for c in constraints], dtype=float)
        magnitude_km = float(np.linalg.norm(actual - target_arr))

        dv_required = QuantityVector(
            value=[float(dv_vec[0]), float(dv_vec[1]), float(dv_vec[2])],
            unit="km/s",
        )
        residual = BplaneResidual(
            actual_b_t_after_dv=Quantity(value=float(b_t_after), unit="km"),
            actual_b_r_after_dv=Quantity(value=float(b_r_after), unit="km"),
            magnitude=Quantity(value=magnitude_km, unit="km"),
        )

    return BplaneTargetResponse(
        b_r=Quantity(value=b_r, unit="km"),
        b_t=Quantity(value=b_t, unit="km"),
        v_infinity=QuantityVector(
            value=[float(v_inf_vec[0]), float(v_inf_vec[1]), float(v_inf_vec[2])],
            unit="km/s",
        ),
        asymptote_declination=Quantity(value=dec_deg, unit="deg"),
        dv_required=dv_required,
        residual=residual,
    )
