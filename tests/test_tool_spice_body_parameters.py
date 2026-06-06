"""Tests for the ``spice_body_parameters`` PCK constant-lookup tool.

Drives the module-level ``_do_body_parameters`` helper (the registered slot is a
thin wrapper) against the in-memory ``FakeSpice``, which models CSPICE's kernel
dependence: ``bodvcd`` raises ``SPICE(KERNELVARNOTFOUND)`` unless the constant is
pinned and its required kernel category (a PCK) is in the pool — so the
missing-constant acceptance path is exercised through real pool state.

Covers the acceptance contract: radii / GM returned as ``{value, unit}`` elements
with the source pool variable echoed; body resolved by name or NAIF ID;
orientation coefficients carried with per-element units; typed errors for an
unknown body, an unknown parameter, and a missing constant; the output schema
round-trips; and a committed reference golden (Mars radii + GM).

Per the v0.3 strategy (the test env ships no ``spiceypy``), the golden validates
the tool's packaging of known constants — the ``FakeSpice`` is fed the same
values — not CSPICE's own kernel reads, which no CI cell runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from mcp.server.fastmcp import FastMCP

from astrodynamics_mcp.errors import InvalidInputError, UpstreamError
from astrodynamics_mcp.tools import spice as spice_tools
from astrodynamics_mcp.tools.spice import (
    SpiceBodyParameter,
    SpiceBodyParametersResponse,
    _do_body_parameters,
    _do_load_kernel,
)
from astrodynamics_mcp.units import Quantity
from tests._spice_fakes import FakeSpice

# Representative Mars (NAIF 499) constants. The exact values are the committed
# golden's; the fake is fed the same.
_MARS_RADII = [3396.19, 3396.19, 3376.2]
_MARS_GM = 42828.375214
_MARS_POLE_RA = [317.68143, -0.1061, 0.0]
_MARS_POLE_DEC = [52.8865, -0.0609, 0.0]
_MARS_PM = [176.63, 350.89198226, 0.0]

_GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "spice_body_parameters_golden.json"


@pytest.fixture
def fake_spice(monkeypatch: pytest.MonkeyPatch) -> FakeSpice:
    fake = FakeSpice()
    monkeypatch.setitem(sys.modules, "spiceypy", fake)
    return fake


def _write_kernel(tmp_path: Path, name: str, payload: bytes = b"fake kernel bytes") -> str:
    path = tmp_path / name
    path.write_bytes(payload)
    return str(path)


async def _furnish_pck(tmp_path: Path) -> None:
    """Furnish a planetary-constants PCK so a constant lookup can resolve."""
    await _do_load_kernel(_write_kernel(tmp_path, "pck00011.tpc"))


def _plan_mars(fake: FakeSpice, *, requires: str | None = "PCK") -> None:
    """Pin Mars's code and the standard radii / GM / orientation constants."""
    fake.plan_body_code("MARS", 499)
    fake.plan_body_constant(499, "RADII", _MARS_RADII, requires=requires)
    fake.plan_body_constant(499, "GM", [_MARS_GM], requires=requires)
    fake.plan_body_constant(499, "POLE_RA", _MARS_POLE_RA, requires=requires)
    fake.plan_body_constant(499, "POLE_DEC", _MARS_POLE_DEC, requires=requires)
    fake.plan_body_constant(499, "PM", _MARS_PM, requires=requires)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestDefaultCommonSet:
    async def test_default_returns_radii_and_gm(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_pck(tmp_path)
        _plan_mars(fake_spice)

        response = await _do_body_parameters(body="MARS", parameters=None)

        assert isinstance(response, SpiceBodyParametersResponse)
        assert response.body == "MARS"
        assert [p.name for p in response.parameters] == ["radii", "gm"]

        radii = response.parameters[0]
        assert radii.source == "BODY499_RADII"
        assert [q.value for q in radii.values] == _MARS_RADII
        assert all(q.unit == "km" for q in radii.values)

        gm = response.parameters[1]
        assert gm.source == "BODY499_GM"
        assert len(gm.values) == 1
        assert gm.values[0].value == _MARS_GM
        assert gm.values[0].unit == "km^3/s^2"


class TestBodyResolution:
    async def test_naif_id_string_resolves_same_as_name(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_pck(tmp_path)
        _plan_mars(fake_spice)
        # bods2c resolves the digit string '499' to the same code, no name plan
        # needed; the constant is keyed on the code, so it reads identically.
        response = await _do_body_parameters(body="499", parameters=["radii"])
        assert response.body == "499"
        assert response.parameters[0].source == "BODY499_RADII"
        assert [q.value for q in response.parameters[0].values] == _MARS_RADII


class TestOrientationConstants:
    async def test_pole_and_pm_carry_per_coefficient_units(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_pck(tmp_path)
        _plan_mars(fake_spice)

        response = await _do_body_parameters(body="MARS", parameters=["pole_ra", "pole_dec", "pm"])

        by_name = {p.name: p for p in response.parameters}

        pole_ra = by_name["pole_ra"]
        assert pole_ra.source == "BODY499_POLE_RA"
        assert [q.value for q in pole_ra.values] == _MARS_POLE_RA
        assert [q.unit for q in pole_ra.values] == ["deg", "deg/century", "deg/century^2"]

        pole_dec = by_name["pole_dec"]
        assert [q.unit for q in pole_dec.values] == ["deg", "deg/century", "deg/century^2"]

        pm = by_name["pm"]
        assert pm.source == "BODY499_PM"
        assert [q.value for q in pm.values] == _MARS_PM
        assert [q.unit for q in pm.values] == ["deg", "deg/day", "deg/day^2"]


class TestParameterNormalisation:
    async def test_names_are_case_insensitive(self, fake_spice: FakeSpice, tmp_path: Path) -> None:
        await _furnish_pck(tmp_path)
        _plan_mars(fake_spice)
        response = await _do_body_parameters(body="MARS", parameters=["RADII", "Gm"])
        assert [p.name for p in response.parameters] == ["radii", "gm"]

    async def test_duplicates_collapse_preserving_order(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_pck(tmp_path)
        _plan_mars(fake_spice)
        response = await _do_body_parameters(body="MARS", parameters=["pm", "radii", "pm", "radii"])
        assert [p.name for p in response.parameters] == ["pm", "radii"]

    async def test_multi_parameter_lookup_is_a_single_worker_dispatch(
        self, fake_spice: FakeSpice, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # All requested constants are read in one worker call
        # (query_body_constants), so a multi-parameter lookup is one atomic
        # CSPICE interaction — not one dispatch per parameter.
        from astrodynamics_mcp.spice_runtime import run_on_spice_thread as real

        await _furnish_pck(tmp_path)
        _plan_mars(fake_spice)
        dispatched: list[str] = []

        async def counting(fn: object, *args: object, **kwargs: object) -> object:
            dispatched.append(fn.__name__)  # type: ignore[attr-defined]
            return await real(fn, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(spice_tools, "run_on_spice_thread", counting)
        await _do_body_parameters(body="MARS", parameters=["radii", "gm", "pole_ra"])
        assert dispatched == ["query_body_constants"]


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------


class TestTypedErrors:
    async def test_unknown_body_is_typed_error(self, fake_spice: FakeSpice, tmp_path: Path) -> None:
        await _furnish_pck(tmp_path)
        # 'PLANET_X' is neither a digit string nor a planned name → bods2c not found.
        with pytest.raises(InvalidInputError) as excinfo:
            await _do_body_parameters(body="PLANET_X", parameters=["radii"])
        assert excinfo.value.code == "invalid_input.spice_unknown_body"
        # The failure is at body resolution — no constant read attempted.
        assert fake_spice.calls["bodvcd"] == []

    async def test_unknown_parameter_is_typed_error_before_cspice(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_pck(tmp_path)
        _plan_mars(fake_spice)
        with pytest.raises(InvalidInputError) as excinfo:
            await _do_body_parameters(body="MARS", parameters=["mass"])
        assert excinfo.value.code == "invalid_input.spice_unknown_parameter"
        assert fake_spice.calls["bods2c"] == []
        assert fake_spice.calls["bodvcd"] == []

    async def test_empty_parameter_list_is_typed_error(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_pck(tmp_path)
        _plan_mars(fake_spice)
        with pytest.raises(InvalidInputError) as excinfo:
            await _do_body_parameters(body="MARS", parameters=[])
        assert excinfo.value.code == "invalid_input.spice_empty_parameters"

    async def test_missing_pck_is_typed_error(self, fake_spice: FakeSpice, tmp_path: Path) -> None:
        # The constants are pinned but require a PCK that is not furnished —
        # bodvcd raises, never a silent gap.
        _plan_mars(fake_spice)
        with pytest.raises(UpstreamError) as excinfo:
            await _do_body_parameters(body="MARS", parameters=["radii"])
        assert excinfo.value.code == "upstream.spice_body_parameters_failed"

    async def test_missing_constant_is_typed_error(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        # GM is not pinned for this body (only radii is) — requesting it raises
        # rather than returning a silent gap.
        await _furnish_pck(tmp_path)
        fake_spice.plan_body_code("MARS", 499)
        fake_spice.plan_body_constant(499, "RADII", _MARS_RADII, requires="PCK")
        with pytest.raises(UpstreamError) as excinfo:
            await _do_body_parameters(body="MARS", parameters=["gm"])
        assert excinfo.value.code == "upstream.spice_body_parameters_failed"

    async def test_higher_order_orientation_is_typed_error(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        # A PM array longer than the unit-mapped coefficients (constant + linear +
        # quadratic) is refused rather than mislabelled.
        await _furnish_pck(tmp_path)
        fake_spice.plan_body_code("MARS", 499)
        fake_spice.plan_body_constant(499, "PM", [1.0, 2.0, 3.0, 4.0], requires="PCK")
        with pytest.raises(InvalidInputError) as excinfo:
            await _do_body_parameters(body="MARS", parameters=["pm"])
        assert excinfo.value.code == "invalid_input.spice_unsupported_constant_order"


# ---------------------------------------------------------------------------
# Output-schema round-trip
# ---------------------------------------------------------------------------


class TestOutputRoundTrip:
    def test_response_roundtrips_through_schema(self) -> None:
        response = SpiceBodyParametersResponse(
            body="MARS",
            parameters=[
                SpiceBodyParameter(
                    name="radii",
                    source="BODY499_RADII",
                    values=[Quantity(value=v, unit="km") for v in _MARS_RADII],
                ),
                SpiceBodyParameter(
                    name="gm",
                    source="BODY499_GM",
                    values=[Quantity(value=_MARS_GM, unit="km^3/s^2")],
                ),
            ],
        )
        first = response.model_dump_json()
        rebuilt = SpiceBodyParametersResponse.model_validate_json(first)
        assert rebuilt.model_dump_json() == first

    def test_extra_keys_forbidden(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SpiceBodyParametersResponse.model_validate(
                {"body": "MARS", "parameters": [], "surprise": True}
            )


# ---------------------------------------------------------------------------
# Committed reference golden
# ---------------------------------------------------------------------------


class TestReferenceGolden:
    async def test_constants_match_committed_golden(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_pck(tmp_path)
        _plan_mars(fake_spice)

        response = await _do_body_parameters(body="MARS", parameters=None)
        actual = response.model_dump(mode="json")
        expected = json.loads(_GOLDEN_PATH.read_text())
        assert actual == expected


# ---------------------------------------------------------------------------
# End-to-end through the registered slot
# ---------------------------------------------------------------------------


class TestRegisteredToolCall:
    @pytest.fixture
    def registered_mcp(self, monkeypatch: pytest.MonkeyPatch) -> FastMCP:
        fresh = FastMCP("spice-body-parameters-test")
        monkeypatch.setattr("astrodynamics_mcp.server.mcp", fresh)
        spice_tools._register_spice_tools()
        return fresh

    async def test_body_parameters_round_trip(
        self, registered_mcp: FastMCP, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_pck(tmp_path)
        _plan_mars(fake_spice)

        _, result = await registered_mcp.call_tool(
            "spice_body_parameters",
            {"body": "MARS", "parameters": ["radii", "pm"]},
        )
        assert isinstance(result, dict)
        assert result["body"] == "MARS"
        assert [p["name"] for p in result["parameters"]] == ["radii", "pm"]
        assert result["parameters"][0]["values"][0]["unit"] == "km"
        assert result["parameters"][1]["values"][1]["unit"] == "deg/day"

    async def test_default_set_via_slot(
        self, registered_mcp: FastMCP, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_pck(tmp_path)
        _plan_mars(fake_spice)
        _, result = await registered_mcp.call_tool("spice_body_parameters", {"body": "MARS"})
        assert isinstance(result, dict)
        assert [p["name"] for p in result["parameters"]] == ["radii", "gm"]
