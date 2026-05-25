"""Tests for `astrodynamics_mcp.tools.lambert`.

Lambert solves are pure computation against `lamberthub` — no network
mocking. Tests cover textbook reference, multi-revolution enumeration,
the four allowed algorithms, body-name vs explicit-μ inputs, error paths
(degenerate geometry, partial Δv, bad inputs), Δv computation,
description-lint, and end-to-end MCP invocation.
"""

from __future__ import annotations

import json
from typing import ClassVar, Literal

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from astrodynamics_mcp.server import mcp
from astrodynamics_mcp.server_lint import check_tool_descriptions
from astrodynamics_mcp.tools.lambert import (
    _BODY_MU,
    LambertSolution,
    LambertSolveResponse,
    lambert_solve,
)

# Curtis Example 5.2 — Earth-orbit Lambert reference. The textbook v1/v2
# values below come straight from the worked example; lamberthub matches
# them to ~1e-4 km/s.
_CURTIS_R1 = [5000.0, 10000.0, 2100.0]
_CURTIS_R2 = [-14600.0, 2500.0, 7000.0]
_CURTIS_TOF = 3600.0
_CURTIS_V1_REF = [-5.9925, 1.9254, 3.2456]
_CURTIS_V2_REF = [-3.3125, -4.1966, -0.3853]


class TestTextbookReference:
    async def test_curtis_5_2_matches_textbook_within_0_1_mps(self) -> None:
        """Acceptance: textbook Earth Lambert returns v1, v2 within 0.1 m/s."""
        resp = await lambert_solve(r1=_CURTIS_R1, r2=_CURTIS_R2, tof=_CURTIS_TOF, mu="earth")
        # 0.1 m/s = 1e-4 km/s — tolerance from the issue's acceptance criterion.
        for axis in range(3):
            assert resp.v1.value[axis] == pytest.approx(_CURTIS_V1_REF[axis], abs=1e-4)
            assert resp.v2.value[axis] == pytest.approx(_CURTIS_V2_REF[axis], abs=1e-4)
        # The primary echo and the M=0 row in all_solutions agree.
        assert len(resp.all_solutions) == 1
        first = resp.all_solutions[0]
        assert first.revs.value == 0.0
        assert first.low_path is True

    async def test_response_units_are_explicit(self) -> None:
        resp = await lambert_solve(r1=_CURTIS_R1, r2=_CURTIS_R2, tof=_CURTIS_TOF, mu="earth")
        assert resp.v1.unit == "km/s"
        assert resp.v2.unit == "km/s"
        assert resp.transfer_elements.a.unit == "km"
        assert resp.transfer_elements.e.unit == "1"
        assert resp.transfer_elements.i.unit == "deg"


class TestBodyAndMu:
    async def test_body_name_matches_explicit_mu(self) -> None:
        """`mu='earth'` and `mu=3.986004418e5` must yield the same primary solution."""
        by_name = await lambert_solve(r1=_CURTIS_R1, r2=_CURTIS_R2, tof=_CURTIS_TOF, mu="earth")
        by_float = await lambert_solve(
            r1=_CURTIS_R1, r2=_CURTIS_R2, tof=_CURTIS_TOF, mu=_BODY_MU["earth"]
        )
        for axis in range(3):
            assert by_name.v1.value[axis] == pytest.approx(by_float.v1.value[axis])
            assert by_name.v2.value[axis] == pytest.approx(by_float.v2.value[axis])

    async def test_body_name_case_insensitive(self) -> None:
        upper = await lambert_solve(r1=_CURTIS_R1, r2=_CURTIS_R2, tof=_CURTIS_TOF, mu="EARTH")
        lower = await lambert_solve(r1=_CURTIS_R1, r2=_CURTIS_R2, tof=_CURTIS_TOF, mu="earth")
        for axis in range(3):
            assert upper.v1.value[axis] == pytest.approx(lower.v1.value[axis])

    async def test_unknown_body_raises_typed_error(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await lambert_solve(r1=_CURTIS_R1, r2=_CURTIS_R2, tof=_CURTIS_TOF, mu="kepler-186f")
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.unknown_body"

    async def test_negative_mu_raises_typed_error(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await lambert_solve(r1=_CURTIS_R1, r2=_CURTIS_R2, tof=_CURTIS_TOF, mu=-1.0)
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.wrong_mu_value"

    async def test_boolean_mu_rejected(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await lambert_solve(
                r1=_CURTIS_R1,
                r2=_CURTIS_R2,
                tof=_CURTIS_TOF,
                mu=True,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.wrong_mu_type"


class TestMultiRevolution:
    """Long-tof + close-r geometry admits revs=2 with low + high path branches."""

    R1: ClassVar[list[float]] = [7000.0, 0.0, 0.0]
    R2: ClassVar[list[float]] = [0.0, 7100.0, 0.0]
    TOF: ClassVar[float] = 12 * 3600.0  # 12 h — plenty for revs=2 in LEO (period ~98 min)

    async def test_revs_two_returns_at_least_two_entries(self) -> None:
        """Acceptance: revs=2 yields ≥2 entries in all_solutions."""
        resp = await lambert_solve(r1=self.R1, r2=self.R2, tof=self.TOF, mu="earth", revs=2)
        assert len(resp.all_solutions) >= 2

    async def test_revs_two_enumerates_low_and_high_path(self) -> None:
        resp = await lambert_solve(r1=self.R1, r2=self.R2, tof=self.TOF, mu="earth", revs=2)
        m2_branches = [s for s in resp.all_solutions if s.revs.value == 2.0]
        # Both low_path branches for M=2 must be present.
        assert {s.low_path for s in m2_branches} == {True, False}

    async def test_primary_echoes_requested_revs_low_path(self) -> None:
        """The top-level v1/v2 echo (revs=requested, low_path=True)."""
        resp = await lambert_solve(r1=self.R1, r2=self.R2, tof=self.TOF, mu="earth", revs=2)
        primary_row = next(
            s for s in resp.all_solutions if s.revs.value == 2.0 and s.low_path is True
        )
        for axis in range(3):
            assert resp.v1.value[axis] == pytest.approx(primary_row.v1.value[axis])
            assert resp.v2.value[axis] == pytest.approx(primary_row.v2.value[axis])

    async def test_zero_revs_emits_one_solution_only(self) -> None:
        resp = await lambert_solve(
            r1=_CURTIS_R1, r2=_CURTIS_R2, tof=_CURTIS_TOF, mu="earth", revs=0
        )
        assert len(resp.all_solutions) == 1
        assert resp.all_solutions[0].revs.value == 0.0
        # M=0 is degenerate in low_path; we emit only the low_path=True row.
        assert resp.all_solutions[0].low_path is True


class TestAlgorithms:
    @pytest.mark.parametrize("algorithm", ["izzo", "izzo_revisited", "gooding", "battin"])
    async def test_every_algorithm_solves_curtis(
        self,
        algorithm: Literal["izzo", "izzo_revisited", "gooding", "battin"],
    ) -> None:
        resp = await lambert_solve(
            r1=_CURTIS_R1, r2=_CURTIS_R2, tof=_CURTIS_TOF, mu="earth", algorithm=algorithm
        )
        # Loose tolerance: `battin1984` converges to ~20 m/s of slop on this
        # geometry with lamberthub's default maxiter=35. The point of this
        # test is "dispatch wiring works for every algorithm name", not that
        # every solver hits textbook precision — Curtis's textbook precision
        # is asserted against izzo in TestTextbookReference.
        for axis in range(3):
            assert resp.v1.value[axis] == pytest.approx(_CURTIS_V1_REF[axis], abs=5e-2)
            assert resp.v2.value[axis] == pytest.approx(_CURTIS_V2_REF[axis], abs=5e-2)

    async def test_izzo_and_izzo_revisited_match_exactly(self) -> None:
        """Aliases must map to the same underlying solver call."""
        a = await lambert_solve(
            r1=_CURTIS_R1, r2=_CURTIS_R2, tof=_CURTIS_TOF, mu="earth", algorithm="izzo"
        )
        b = await lambert_solve(
            r1=_CURTIS_R1,
            r2=_CURTIS_R2,
            tof=_CURTIS_TOF,
            mu="earth",
            algorithm="izzo_revisited",
        )
        for axis in range(3):
            assert a.v1.value[axis] == b.v1.value[axis]


class TestDeltaV:
    """Two-impulse dv = |v1 - depart_velocity| + |v2 - arrive_velocity|."""

    async def test_partial_dv_raises_invalid_input(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await lambert_solve(
                r1=_CURTIS_R1,
                r2=_CURTIS_R2,
                tof=_CURTIS_TOF,
                mu="earth",
                depart_velocity=[0.0, 0.0, 0.0],
                # arrive_velocity omitted → XOR violation
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.lambert_partial_dv"

    async def test_dv_zero_when_boundary_velocities_match_transfer(self) -> None:
        """Δv == 0 when the spacecraft already moves at v1 / v2."""
        resp_no_dv = await lambert_solve(r1=_CURTIS_R1, r2=_CURTIS_R2, tof=_CURTIS_TOF, mu="earth")
        v1 = list(resp_no_dv.v1.value)
        v2 = list(resp_no_dv.v2.value)
        resp = await lambert_solve(
            r1=_CURTIS_R1,
            r2=_CURTIS_R2,
            tof=_CURTIS_TOF,
            mu="earth",
            depart_velocity=v1,
            arrive_velocity=v2,
        )
        assert resp.dv is not None
        assert resp.dv.value == pytest.approx(0.0, abs=1e-9)
        assert resp.dv.unit == "km/s"

    async def test_dv_sums_two_impulse_magnitudes(self) -> None:
        """dv = |v1 - dep| + |arr - v2| with arbitrary boundary velocities."""
        resp_baseline = await lambert_solve(
            r1=_CURTIS_R1, r2=_CURTIS_R2, tof=_CURTIS_TOF, mu="earth"
        )
        # Arbitrary boundary velocities — pick something clearly different from v1, v2.
        dep = [0.0, 0.0, 0.0]
        arr = [0.0, 0.0, 0.0]
        v1 = resp_baseline.v1.value
        v2 = resp_baseline.v2.value
        expected_dv = sum(x * x for x in v1) ** 0.5 + sum(x * x for x in v2) ** 0.5

        resp = await lambert_solve(
            r1=_CURTIS_R1,
            r2=_CURTIS_R2,
            tof=_CURTIS_TOF,
            mu="earth",
            depart_velocity=dep,
            arrive_velocity=arr,
        )
        assert resp.dv is not None
        assert resp.dv.value == pytest.approx(expected_dv, abs=1e-6)


class TestFailureModes:
    async def test_degenerate_r1_equals_r2_raises_no_solution(self) -> None:
        """Acceptance: r1 == r2 with tof > 0 → upstream.lambert_no_solution."""
        with pytest.raises(ToolError) as excinfo:
            await lambert_solve(r1=_CURTIS_R1, r2=_CURTIS_R1, tof=_CURTIS_TOF, mu="earth")
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "upstream.lambert_no_solution"
        assert envelope["data"]["algorithm"] == "izzo"

    async def test_negative_tof_raises_invalid_input(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await lambert_solve(r1=_CURTIS_R1, r2=_CURTIS_R2, tof=-1.0, mu="earth")
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.tof_not_positive"

    async def test_wrong_length_r1_raises_invalid_input(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await lambert_solve(r1=[1.0, 2.0], r2=_CURTIS_R2, tof=_CURTIS_TOF, mu="earth")
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.wrong_vector_length"

    async def test_nan_in_r2_raises_invalid_input(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await lambert_solve(
                r1=_CURTIS_R1,
                r2=[float("nan"), 0.0, 0.0],
                tof=_CURTIS_TOF,
                mu="earth",
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.value_not_a_number"

    async def test_negative_revs_raises_invalid_input(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await lambert_solve(
                r1=_CURTIS_R1,
                r2=_CURTIS_R2,
                tof=_CURTIS_TOF,
                mu="earth",
                revs=-1,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.revs_not_a_non_negative_int"


class TestRegistration:
    async def test_tool_is_listed(self) -> None:
        tools = await mcp.list_tools()
        names = {t.name for t in tools}
        assert "lambert_solve" in names

    async def test_tool_description_passes_lint(self) -> None:
        tools = await mcp.list_tools()
        violations = [v for v in check_tool_descriptions(tools) if v.tool_name == "lambert_solve"]
        assert violations == []

    async def test_tool_callable_via_mcp(self) -> None:
        content, structured = await mcp.call_tool(
            "lambert_solve",
            {
                "r1": _CURTIS_R1,
                "r2": _CURTIS_R2,
                "tof": _CURTIS_TOF,
                "mu": "earth",
            },
        )
        del content
        assert isinstance(structured, dict)
        assert "v1" in structured and "v2" in structured
        assert structured["v1"]["unit"] == "km/s"
        for axis in range(3):
            assert structured["v1"]["value"][axis] == pytest.approx(_CURTIS_V1_REF[axis], abs=1e-4)


class TestSchemaInvariants:
    async def test_response_round_trips_through_json(self) -> None:
        # Use the multi-rev geometry so `all_solutions` carries multiple rows
        # — exercises more of the response's nested schema in the round-trip.
        resp = await lambert_solve(
            r1=TestMultiRevolution.R1,
            r2=TestMultiRevolution.R2,
            tof=TestMultiRevolution.TOF,
            mu="earth",
            revs=2,
        )
        as_json = resp.model_dump_json()
        rebuilt = LambertSolveResponse.model_validate_json(as_json)
        assert rebuilt == resp

    async def test_solution_row_has_quantity_wrapped_revs(self) -> None:
        """`revs` is `Quantity(value=..., unit='1')` to satisfy unit discipline."""
        resp = await lambert_solve(r1=_CURTIS_R1, r2=_CURTIS_R2, tof=_CURTIS_TOF, mu="earth")
        first = resp.all_solutions[0]
        assert isinstance(first, LambertSolution)
        assert first.revs.unit == "1"
