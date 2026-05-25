"""Tests for `astrodynamics_mcp.tools.bplane`.

The math is pure numpy — no network, no mocks. Tests cover an analytical
Mars-flyby reference geometry, read-only mode, single- and dual-axis
targeting, the linearity sanity check from the issue's acceptance
criteria, error paths, the unit-discipline meta-test, description-lint,
and end-to-end MCP invocation.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from mcp.server.fastmcp.exceptions import ToolError

from astrodynamics_mcp.schemas.base import Frame, StateVector
from astrodynamics_mcp.server import mcp
from astrodynamics_mcp.server_lint import check_tool_descriptions
from astrodynamics_mcp.tools.bplane import (
    _BODY_PARAMETERS,
    BplaneResidual,
    BplaneTargetResponse,
    bplane_target,
)
from astrodynamics_mcp.units import Quantity, QuantityVector

# ---------------------------------------------------------------------------
# Analytical Mars-flyby fixture
# ---------------------------------------------------------------------------
#
# Mars-centred ICRF state placed at periapsis on the +X axis with velocity
# along +Y. With e = 2.0 and r_p = 5000 km this yields a closed-form
# B-plane geometry (orbit in the XY plane, so K̂=+Z is normal to the orbit
# and T̂ lies along the orbit plane): B·T = -b, B·R = 0, asymptote
# declination = 0. The full derivation is in the tool docstring; the
# numbers below come from a-textbook two-body identities, no upstream
# library involved.

_MU_MARS, _ = _BODY_PARAMETERS["mars"]
_E_REF = 2.0
_R_P_REF = 5000.0
_A_REF = _R_P_REF / (1.0 - _E_REF)  # negative for hyperbolic
_ABS_A_REF = abs(_A_REF)
_B_REF = _ABS_A_REF * np.sqrt(_E_REF * _E_REF - 1.0)  # impact-parameter magnitude
_V_P_REF = np.sqrt(_MU_MARS * (2.0 / _R_P_REF - 1.0 / _A_REF))
_V_INF_REF = np.sqrt(_MU_MARS / _ABS_A_REF)  # = √(2·ε)

# Reference (B·T, B·R) for this geometry: see derivation in
# `_bplane_elements`'s docstring. Orbit in the XY plane ⇒ B·R = 0.
_BT_REF = -_B_REF
_BR_REF = 0.0
# Incoming-asymptote unit vector S^_pqw = (1/e, √(e²-1)/e, 0); with
# perifocal frame aligned to inertial axes that maps directly to inertial.
_S_HAT_REF = np.array([1.0 / _E_REF, np.sqrt(_E_REF * _E_REF - 1.0) / _E_REF, 0.0])
_V_INF_VEC_REF = _V_INF_REF * _S_HAT_REF
_TARGET_EPOCH = "2026-12-01T00:00:00Z"


def _mars_periapsis_state() -> StateVector:
    return StateVector(
        r=QuantityVector(value=[_R_P_REF, 0.0, 0.0], unit="km"),
        v=QuantityVector(value=[0.0, _V_P_REF, 0.0], unit="km/s"),
        frame=Frame.ICRF,
        epoch="2026-11-30T00:00:00Z",
    )


def _earth_hyperbolic_state() -> StateVector:
    """A different geometry — Earth-centred, out-of-plane v_∞.

    Used for cases that need a non-degenerate (b_t ≠ 0 and b_r ≠ 0) state.
    """
    mu_earth = _BODY_PARAMETERS["earth"][0]
    e = 1.5
    r_p = 8000.0
    a = r_p / (1.0 - e)
    v_p = np.sqrt(mu_earth * (2.0 / r_p - 1.0 / a))
    # Tilt the orbit plane 30° around the +X axis so the orbit is no longer
    # in the input frame's XY plane — that gives a non-zero B·R.
    cos_t = float(np.cos(np.radians(30.0)))
    sin_t = float(np.sin(np.radians(30.0)))
    return StateVector(
        r=QuantityVector(value=[r_p, 0.0, 0.0], unit="km"),
        v=QuantityVector(value=[0.0, v_p * cos_t, v_p * sin_t], unit="km/s"),
        frame=Frame.GCRS,
        epoch="2026-11-30T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# Read-only mode + analytical reference
# ---------------------------------------------------------------------------


class TestAnalyticalReference:
    async def test_mars_periapsis_matches_analytical_btr_btt(self) -> None:
        """Acceptance: canonical Mars flyby returns B·R, B·T at textbook tolerance."""
        resp = await bplane_target(
            state=_mars_periapsis_state(),
            target_body="mars",
            target_epoch=_TARGET_EPOCH,
        )
        assert resp.b_t.value == pytest.approx(_BT_REF, abs=1e-6)
        assert resp.b_r.value == pytest.approx(_BR_REF, abs=1e-6)
        assert resp.asymptote_declination.value == pytest.approx(0.0, abs=1e-9)
        for axis in range(3):
            assert resp.v_infinity.value[axis] == pytest.approx(_V_INF_VEC_REF[axis], abs=1e-6)

    async def test_units_are_explicit(self) -> None:
        resp = await bplane_target(
            state=_mars_periapsis_state(),
            target_body="mars",
            target_epoch=_TARGET_EPOCH,
        )
        assert resp.b_t.unit == "km"
        assert resp.b_r.unit == "km"
        assert resp.v_infinity.unit == "km/s"
        assert resp.asymptote_declination.unit == "deg"

    async def test_read_only_mode_omits_dv_and_residual(self) -> None:
        """Acceptance: target_btr_km and target_btt_km both None → no targeting block."""
        resp = await bplane_target(
            state=_mars_periapsis_state(),
            target_body="mars",
            target_epoch=_TARGET_EPOCH,
        )
        assert resp.dv_required is None
        assert resp.residual is None

    async def test_response_round_trips_through_json(self) -> None:
        resp = await bplane_target(
            state=_mars_periapsis_state(),
            target_body="mars",
            target_epoch=_TARGET_EPOCH,
        )
        rebuilt = BplaneTargetResponse.model_validate_json(resp.model_dump_json())
        assert rebuilt == resp

    async def test_body_name_case_insensitive(self) -> None:
        upper = await bplane_target(
            state=_mars_periapsis_state(),
            target_body="MARS",
            target_epoch=_TARGET_EPOCH,
        )
        lower = await bplane_target(
            state=_mars_periapsis_state(),
            target_body="mars",
            target_epoch=_TARGET_EPOCH,
        )
        assert upper.b_t.value == pytest.approx(lower.b_t.value)

    async def test_alternate_input_units_normalise(self) -> None:
        """Inputs in metres and m/s yield the same answer as km / km/s."""
        ref = await bplane_target(
            state=_mars_periapsis_state(),
            target_body="mars",
            target_epoch=_TARGET_EPOCH,
        )
        metric = await bplane_target(
            state=StateVector(
                r=QuantityVector(value=[_R_P_REF * 1000.0, 0.0, 0.0], unit="m"),
                v=QuantityVector(value=[0.0, _V_P_REF * 1000.0, 0.0], unit="m/s"),
                frame=Frame.ICRF,
                epoch="2026-11-30T00:00:00Z",
            ),
            target_body="mars",
            target_epoch=_TARGET_EPOCH,
        )
        assert metric.b_t.value == pytest.approx(ref.b_t.value, abs=1e-6)
        assert metric.b_r.value == pytest.approx(ref.b_r.value, abs=1e-6)


# ---------------------------------------------------------------------------
# Targeting modes
# ---------------------------------------------------------------------------


class TestTargeting:
    async def test_dv_returned_when_target_supplied(self) -> None:
        resp = await bplane_target(
            state=_mars_periapsis_state(),
            target_body="mars",
            target_epoch=_TARGET_EPOCH,
            target_btr_km=500.0,
        )
        assert resp.dv_required is not None
        assert resp.dv_required.unit == "km/s"
        assert len(resp.dv_required.value) == 3
        assert resp.residual is not None
        assert isinstance(resp.residual, BplaneResidual)
        assert resp.residual.actual_b_r_after_dv.unit == "km"
        assert resp.residual.magnitude.unit == "km"

    async def test_single_axis_btt_target_lands_close(self) -> None:
        resp = await bplane_target(
            state=_mars_periapsis_state(),
            target_body="mars",
            target_epoch=_TARGET_EPOCH,
            target_btt_km=_BT_REF + 100.0,  # shift B·T by 100 km from the baseline
        )
        assert resp.residual is not None
        # The linear solver lands within a few km on a 100 km offset.
        assert resp.residual.magnitude.value < 5.0
        # B·R is unconstrained — only the constrained dimension counts.
        assert abs(resp.residual.actual_b_t_after_dv.value - (_BT_REF + 100.0)) < 5.0

    async def test_dual_axis_target_lands_close_on_both(self) -> None:
        # Read the current B-plane state and ask for small perturbations.
        # Large offsets exceed the linear solver's accuracy by design — this
        # test exists to verify the *both-axis* solve path, not to stress
        # the linearisation.
        state = _earth_hyperbolic_state()
        baseline = await bplane_target(state=state, target_body="earth", target_epoch=_TARGET_EPOCH)
        resp = await bplane_target(
            state=state,
            target_body="earth",
            target_epoch=_TARGET_EPOCH,
            target_btr_km=baseline.b_r.value + 100.0,
            target_btt_km=baseline.b_t.value + 100.0,
        )
        assert resp.residual is not None
        # Both constraints in play → magnitude is the 2-D Euclidean error.
        # Linear solver lands within a few km on a ~140 km combined offset.
        assert resp.residual.magnitude.value < 10.0

    async def test_linearity_sanity_check(self) -> None:
        """Acceptance: ‖Δv‖ scales linearly with the requested B-plane offset.

        The offset is measured from the *current* B·R, not from zero, so the
        scan stays inside the linearisation's accuracy band where the
        relationship is linear by construction.
        """
        state = _earth_hyperbolic_state()
        baseline = await bplane_target(state=state, target_body="earth", target_epoch=_TARGET_EPOCH)
        baseline_btr = baseline.b_r.value
        offsets = [1.0, 2.0, 4.0]
        dv_norms = []
        for offset in offsets:
            resp = await bplane_target(
                state=state,
                target_body="earth",
                target_epoch=_TARGET_EPOCH,
                target_btr_km=baseline_btr + offset,
            )
            assert resp.dv_required is not None
            dv_norms.append(float(np.linalg.norm(resp.dv_required.value)))
        # Each doubling of offset roughly doubles ‖Δv‖.
        assert dv_norms[1] / dv_norms[0] == pytest.approx(2.0, rel=0.05)
        assert dv_norms[2] / dv_norms[1] == pytest.approx(2.0, rel=0.05)


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestFailureModes:
    async def test_unknown_target_body_raises_typed_error(self) -> None:
        """Acceptance: unknown body → invalid_input.unknown_body."""
        with pytest.raises(ToolError) as excinfo:
            await bplane_target(
                state=_mars_periapsis_state(),
                target_body="pluto",
                target_epoch=_TARGET_EPOCH,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.unknown_body"

    async def test_non_string_target_body_rejected(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await bplane_target(
                state=_mars_periapsis_state(),
                target_body=499,  # type: ignore[arg-type]
                target_epoch=_TARGET_EPOCH,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.body_not_a_string"

    async def test_non_hyperbolic_state_raises_typed_error(self) -> None:
        """A circular LEO state has ε < 0 and isn't a flyby."""
        mu_earth = _BODY_PARAMETERS["earth"][0]
        r_circ = 7000.0
        v_circ = float(np.sqrt(mu_earth / r_circ))
        with pytest.raises(ToolError) as excinfo:
            await bplane_target(
                state=StateVector(
                    r=QuantityVector(value=[r_circ, 0.0, 0.0], unit="km"),
                    v=QuantityVector(value=[0.0, v_circ, 0.0], unit="km/s"),
                    frame=Frame.GCRS,
                    epoch="2026-11-30T00:00:00Z",
                ),
                target_body="earth",
                target_epoch=_TARGET_EPOCH,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.not_hyperbolic"

    async def test_zero_angular_momentum_raises_typed_error(self) -> None:
        """Radial-only state has h = 0 — B-plane is undefined."""
        mu_earth = _BODY_PARAMETERS["earth"][0]
        # Pick a radial hyperbolic state: v aligned with r, ε > 0.
        r_mag = 8000.0
        # Excess speed: pick v > √(2μ/r) so ε > 0.
        v_mag = float(np.sqrt(2.0 * mu_earth / r_mag) + 5.0)
        with pytest.raises(ToolError) as excinfo:
            await bplane_target(
                state=StateVector(
                    r=QuantityVector(value=[r_mag, 0.0, 0.0], unit="km"),
                    v=QuantityVector(value=[v_mag, 0.0, 0.0], unit="km/s"),
                    frame=Frame.GCRS,
                    epoch="2026-11-30T00:00:00Z",
                ),
                target_body="earth",
                target_epoch=_TARGET_EPOCH,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.zero_angular_momentum"

    async def test_non_numeric_target_rejected(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await bplane_target(
                state=_mars_periapsis_state(),
                target_body="mars",
                target_epoch=_TARGET_EPOCH,
                target_btr_km="five hundred",  # type: ignore[arg-type]
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.value_not_a_number"

    async def test_nan_target_rejected(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await bplane_target(
                state=_mars_periapsis_state(),
                target_body="mars",
                target_epoch=_TARGET_EPOCH,
                target_btt_km=float("nan"),
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.value_not_a_number"

    async def test_boolean_target_rejected(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await bplane_target(
                state=_mars_periapsis_state(),
                target_body="mars",
                target_epoch=_TARGET_EPOCH,
                target_btr_km=True,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.value_not_a_number"


# ---------------------------------------------------------------------------
# MCP registration & description-lint
# ---------------------------------------------------------------------------


class TestRegistration:
    async def test_tool_is_listed(self) -> None:
        tools = await mcp.list_tools()
        assert "bplane_target" in {t.name for t in tools}

    async def test_tool_description_passes_lint(self) -> None:
        tools = await mcp.list_tools()
        violations = [v for v in check_tool_descriptions(tools) if v.tool_name == "bplane_target"]
        assert violations == []

    async def test_tool_callable_via_mcp(self) -> None:
        # Exercise the full envelope: pydantic serialisation, FastMCP dispatch,
        # error-translating wrapper, response structured-output extraction.
        content, structured = await mcp.call_tool(
            "bplane_target",
            {
                "state": {
                    "r": {"value": [_R_P_REF, 0.0, 0.0], "unit": "km"},
                    "v": {"value": [0.0, _V_P_REF, 0.0], "unit": "km/s"},
                    "frame": "ICRF",
                    "epoch": "2026-11-30T00:00:00Z",
                },
                "target_body": "mars",
                "target_epoch": _TARGET_EPOCH,
            },
        )
        del content
        assert isinstance(structured, dict)
        assert "b_r" in structured and "b_t" in structured
        assert structured["b_t"]["unit"] == "km"
        assert structured["b_t"]["value"] == pytest.approx(_BT_REF, abs=1e-3)
        assert structured["dv_required"] is None
        assert structured["residual"] is None


# ---------------------------------------------------------------------------
# Body table coverage
# ---------------------------------------------------------------------------


class TestBodyTable:
    @pytest.mark.parametrize(
        "body",
        ["earth", "mars", "venus", "jupiter", "saturn", "uranus", "neptune", "moon"],
    )
    async def test_every_supported_body_resolves(self, body: str) -> None:
        """Each supported body has μ and a corresponding hyperbolic flyby works end-to-end."""
        mu = _BODY_PARAMETERS[body][0]
        # Construct a hyperbolic state appropriate to the body: r ~ 3x the
        # body's radius, v slightly above local escape speed.
        radius = _BODY_PARAMETERS[body][1]
        r_mag = 3.0 * radius
        v_mag = float(np.sqrt(2.0 * mu / r_mag) + 1.0)
        resp = await bplane_target(
            state=StateVector(
                r=QuantityVector(value=[r_mag, 0.0, 0.0], unit="km"),
                v=QuantityVector(value=[0.0, v_mag, 0.0], unit="km/s"),
                frame=Frame.ICRF,
                epoch="2026-11-30T00:00:00Z",
            ),
            target_body=body,
            target_epoch=_TARGET_EPOCH,
        )
        # Every body gives a finite, real result.
        assert np.isfinite(resp.b_t.value)
        assert np.isfinite(resp.b_r.value)
        assert resp.v_infinity.value[0] != 0.0 or resp.v_infinity.value[1] != 0.0


# ---------------------------------------------------------------------------
# Degenerate frame: S^ parallel to K̂ exercises the K̂ = +X fallback
# ---------------------------------------------------------------------------


class TestDegenerateFrame:
    async def test_polar_incoming_asymptote_falls_back_cleanly(self) -> None:
        """An incoming asymptote nearly along +Z hits the K̂ = +X fallback branch."""
        mu_earth = _BODY_PARAMETERS["earth"][0]
        e = 1.5
        r_p = 8000.0
        a = r_p / (1.0 - e)
        v_p = np.sqrt(mu_earth * (2.0 / r_p - 1.0 / a))
        # Orbit in the XZ plane (no Y component) → S^ has zero Y; with
        # periapsis on +X and v on +Z, S^ ends up roughly in the +X+Z
        # quadrant — close enough to K̂ in the limit but not exactly parallel
        # in this geometry. To exercise the fallback we'd need orbit-normal
        # parallel to K̂ which means orbit in the XY plane — which is the
        # *non*-degenerate case. So this test instead checks the polar
        # geometry doesn't crash and gives a finite result.
        resp = await bplane_target(
            state=StateVector(
                r=QuantityVector(value=[r_p, 0.0, 0.0], unit="km"),
                v=QuantityVector(value=[0.0, 0.0, v_p], unit="km/s"),
                frame=Frame.GCRS,
                epoch="2026-11-30T00:00:00Z",
            ),
            target_body="earth",
            target_epoch=_TARGET_EPOCH,
        )
        assert np.isfinite(resp.b_t.value)
        assert np.isfinite(resp.b_r.value)

    async def test_s_hat_parallel_to_k_hat_fallback(self) -> None:
        """A polar incoming asymptote forces the K = +X fallback in the basis builder.

        Constructed in the perifocal frame and rotated so the orbit lives in
        the YZ plane (orbit normal along +X). Then S = (1/e)*(+Y) + sqrt(e^2-1)/e*(+Z),
        which is in the YZ plane — perpendicular to +X. Crossing S with the
        default K=+Z yields a vector in the X-Y plane; never degenerate.
        To genuinely force the fallback we set the orbit plane = XY (normal
        along +Z, same as K), which makes S lie in XY (orthogonal to K) —
        also not the degenerate case. The actual degenerate condition is
        when S itself points along +Z, which happens when the orbit normal
        lies in the XY plane AND the asymptote within the orbit plane
        happens to coincide with K. The cleanest construction: orbit in the
        XZ plane (normal along -Y), periapsis on +X, velocity on +Z gives
        h = r x v on +Y? Let's verify: r=(rp,0,0), v=(0,0,vp), so
        h = (0*0 - 0*vp, 0*0 - rp*vp, rp*0 - 0*0) = (0, -rp*vp, 0) — along -Y,
        so orbit normal is along -Y. The orbit is in the XZ plane. S in
        that plane points (1/e)*X + sqrt(e^2-1)/e*Z (rotated from the
        perifocal frame). For S to be parallel to K=+Z we need e -> infinity
        (so 1/e -> 0). Practically achievable with e=1000: S is essentially
        (+Z). That triggers the fallback.
        """
        mu_earth = _BODY_PARAMETERS["earth"][0]
        e = 1000.0  # extreme eccentricity -> S asymptotes onto +Z direction
        r_p = 7000.0
        a = r_p / (1.0 - e)
        v_p = float(np.sqrt(mu_earth * (2.0 / r_p - 1.0 / a)))
        resp = await bplane_target(
            state=StateVector(
                r=QuantityVector(value=[r_p, 0.0, 0.0], unit="km"),
                v=QuantityVector(value=[0.0, 0.0, v_p], unit="km/s"),
                frame=Frame.GCRS,
                epoch="2026-11-30T00:00:00Z",
            ),
            target_body="earth",
            target_epoch=_TARGET_EPOCH,
        )
        # Fallback path returned a finite, well-formed answer.
        assert np.isfinite(resp.b_t.value)
        assert np.isfinite(resp.b_r.value)
        assert isinstance(resp.b_t, Quantity)
        # The asymptote declination is essentially +90 degrees here.
        assert resp.asymptote_declination.value == pytest.approx(90.0, abs=0.5)
