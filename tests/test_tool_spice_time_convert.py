"""Tests for the ``spice_time_convert`` ET / UTC / SCLK conversion tool.

Drives the module-level ``_do_time_convert`` helper (the registered slot is a
thin wrapper) against the in-memory ``FakeSpice``, which models CSPICE's kernel
dependence: ``str2et`` / ``et2utc`` raise unless a leap-second (TEXT) kernel is
in the pool and ``sce2s`` / ``scs2e`` raise unless a spacecraft-clock kernel
(``.tsc``) is — so the missing-kernel acceptance paths run through real pool
state rather than a bolted-on flag.

Covers the acceptance contract: UTC<->ET and SCLK round-trips against committed
references; ET output carried as a ``{value, unit}`` quantity in
``'s past J2000 TDB'`` while UTC / SCLK output are strings; the scales and (for
SCLK) the spacecraft echoed; typed errors for a missing LSK / SCLK kernel, a
SCLK conversion with no spacecraft, an unknown spacecraft, and malformed ET /
UTC / SCLK input; the output schema round-trips; and a committed reference
golden (the headline UTC -> ET conversion).

Per the v0.3 strategy (the test env ships no ``spiceypy``), the golden validates
the tool's packaging of a known reference conversion — the ``FakeSpice`` is fed
the same pinned values — not CSPICE's own leap-second / SCLK math, which no CI
cell runs.
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
    SpiceTimeConvertResponse,
    SpiceTimeScale,
    _do_load_kernel,
    _do_time_convert,
)
from astrodynamics_mcp.units import Quantity
from tests._spice_fakes import FakeSpice

# A representative UTC <-> ET <-> SCLK reference for one instant. The exact ET is
# the committed golden's; the fake is fed these same pinned values. `_UTC_CSPICE`
# is the offset-free form the tool hands str2et (the trailing `Z` stripped),
# which et2utc also returns for the inverse.
_UTC_INPUT = "2026-01-01T00:00:00Z"
_UTC_CSPICE = "2026-01-01T00:00:00.000000"
_ET_SECONDS = 820497669.184
_SC_ID = -82
_SC_NAME = "CASSINI"
_SCLK_STRING = "1/1465644281.171"

_GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "spice_time_convert_golden.json"


@pytest.fixture
def fake_spice(monkeypatch: pytest.MonkeyPatch) -> FakeSpice:
    fake = FakeSpice()
    monkeypatch.setitem(sys.modules, "spiceypy", fake)
    return fake


def _write_kernel(tmp_path: Path, name: str, payload: bytes = b"fake kernel bytes") -> str:
    path = tmp_path / name
    path.write_bytes(payload)
    return str(path)


async def _furnish_lsk(tmp_path: Path) -> None:
    """Furnish a leap-second kernel so ET <-> UTC conversions can run."""
    await _do_load_kernel(_write_kernel(tmp_path, "naif0012.tls"))


async def _furnish_sclk(tmp_path: Path) -> None:
    """Furnish a spacecraft-clock kernel so SCLK conversions can run."""
    await _do_load_kernel(_write_kernel(tmp_path, "cas00172.tsc"))


def _plan_reference(fake: FakeSpice) -> None:
    """Pin the UTC <-> ET and SCLK <-> ET reference, plus the spacecraft name."""
    fake.plan_time(_UTC_CSPICE, _ET_SECONDS)
    fake.plan_sclk(_SC_ID, _SCLK_STRING, _ET_SECONDS)
    fake.plan_body_code(_SC_NAME, _SC_ID)


# ---------------------------------------------------------------------------
# Happy paths — ET <-> UTC
# ---------------------------------------------------------------------------


class TestUtcEt:
    async def test_utc_to_et_wraps_as_quantity(self, fake_spice: FakeSpice, tmp_path: Path) -> None:
        await _furnish_lsk(tmp_path)
        _plan_reference(fake_spice)

        response = await _do_time_convert(
            value=_UTC_INPUT, from_scale="UTC", to_scale="ET", spacecraft=None
        )

        assert isinstance(response, SpiceTimeConvertResponse)
        assert response.from_scale == "UTC"
        assert response.to_scale == "ET"
        assert response.spacecraft is None
        assert isinstance(response.value, Quantity)
        assert response.value.value == _ET_SECONDS
        assert response.value.unit == "s past J2000 TDB"

    async def test_str2et_receives_offset_free_utc(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_lsk(tmp_path)
        _plan_reference(fake_spice)
        await _do_time_convert(value=_UTC_INPUT, from_scale="UTC", to_scale="ET", spacecraft=None)
        # The Z designator was stripped before reaching CSPICE.
        assert fake_spice.calls["str2et"] == [_UTC_CSPICE]

    async def test_et_to_utc_returns_calendar_string(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_lsk(tmp_path)
        _plan_reference(fake_spice)

        response = await _do_time_convert(
            value=_ET_SECONDS, from_scale="ET", to_scale="UTC", spacecraft=None
        )

        assert response.to_scale == "UTC"
        assert response.value == _UTC_CSPICE

    async def test_et_input_accepts_numeric_string(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_lsk(tmp_path)
        _plan_reference(fake_spice)
        # ET given as a string, not a number — the tool parses it.
        response = await _do_time_convert(
            value="820497669.184", from_scale="ET", to_scale="UTC", spacecraft=None
        )
        assert response.value == _UTC_CSPICE

    async def test_et_to_et_identity_needs_no_kernel(self, fake_spice: FakeSpice) -> None:
        # Neither leg calls a kernel-dependent routine, so an ET->ET round-trips
        # with an empty pool.
        response = await _do_time_convert(
            value=_ET_SECONDS, from_scale="ET", to_scale="ET", spacecraft=None
        )
        assert isinstance(response.value, Quantity)
        assert response.value.value == _ET_SECONDS
        assert response.value.unit == "s past J2000 TDB"
        assert fake_spice.calls["str2et"] == []
        assert fake_spice.calls["et2utc"] == []

    async def test_utc_round_trips_through_et(self, fake_spice: FakeSpice, tmp_path: Path) -> None:
        await _furnish_lsk(tmp_path)
        _plan_reference(fake_spice)

        to_et = await _do_time_convert(
            value=_UTC_INPUT, from_scale="UTC", to_scale="ET", spacecraft=None
        )
        assert isinstance(to_et.value, Quantity)
        back = await _do_time_convert(
            value=to_et.value.value, from_scale="ET", to_scale="UTC", spacecraft=None
        )
        assert back.value == _UTC_CSPICE


# ---------------------------------------------------------------------------
# Happy paths — SCLK
# ---------------------------------------------------------------------------


class TestSclk:
    async def test_utc_to_sclk(self, fake_spice: FakeSpice, tmp_path: Path) -> None:
        await _furnish_lsk(tmp_path)
        await _furnish_sclk(tmp_path)
        _plan_reference(fake_spice)

        response = await _do_time_convert(
            value=_UTC_INPUT, from_scale="UTC", to_scale="SCLK", spacecraft=_SC_ID
        )

        assert response.to_scale == "SCLK"
        assert response.value == _SCLK_STRING
        # The spacecraft is echoed as a string for an SCLK conversion.
        assert response.spacecraft == "-82"

    async def test_sclk_to_utc(self, fake_spice: FakeSpice, tmp_path: Path) -> None:
        await _furnish_lsk(tmp_path)
        await _furnish_sclk(tmp_path)
        _plan_reference(fake_spice)

        response = await _do_time_convert(
            value=_SCLK_STRING, from_scale="SCLK", to_scale="UTC", spacecraft=_SC_ID
        )

        assert response.from_scale == "SCLK"
        assert response.value == _UTC_CSPICE
        assert response.spacecraft == "-82"

    async def test_et_to_sclk_needs_only_sclk_kernel(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        # No LSK furnished: ET is already absolute, so ET->SCLK needs only the
        # SCLK kernel.
        await _furnish_sclk(tmp_path)
        _plan_reference(fake_spice)

        response = await _do_time_convert(
            value=_ET_SECONDS, from_scale="ET", to_scale="SCLK", spacecraft=_SC_ID
        )
        assert response.value == _SCLK_STRING

    async def test_sclk_to_et_needs_only_sclk_kernel(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_sclk(tmp_path)
        _plan_reference(fake_spice)

        response = await _do_time_convert(
            value=_SCLK_STRING, from_scale="SCLK", to_scale="ET", spacecraft=_SC_ID
        )
        assert isinstance(response.value, Quantity)
        assert response.value.value == _ET_SECONDS

    async def test_sclk_round_trips_through_et(self, fake_spice: FakeSpice, tmp_path: Path) -> None:
        await _furnish_sclk(tmp_path)
        _plan_reference(fake_spice)

        to_et = await _do_time_convert(
            value=_SCLK_STRING, from_scale="SCLK", to_scale="ET", spacecraft=_SC_ID
        )
        assert isinstance(to_et.value, Quantity)
        back = await _do_time_convert(
            value=to_et.value.value, from_scale="ET", to_scale="SCLK", spacecraft=_SC_ID
        )
        assert back.value == _SCLK_STRING

    async def test_spacecraft_resolved_by_name(self, fake_spice: FakeSpice, tmp_path: Path) -> None:
        await _furnish_sclk(tmp_path)
        _plan_reference(fake_spice)
        # A name (resolved via bods2c) works as well as the NAIF ID; the input is
        # echoed verbatim.
        response = await _do_time_convert(
            value=_ET_SECONDS, from_scale="ET", to_scale="SCLK", spacecraft=_SC_NAME
        )
        assert response.value == _SCLK_STRING
        assert response.spacecraft == _SC_NAME


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------


class TestTypedErrors:
    async def test_utc_to_et_without_lsk_is_typed_error(self, fake_spice: FakeSpice) -> None:
        with pytest.raises(UpstreamError) as excinfo:
            await _do_time_convert(
                value=_UTC_INPUT, from_scale="UTC", to_scale="ET", spacecraft=None
            )
        assert excinfo.value.code == "upstream.spice_time_convert_failed"

    async def test_et_to_utc_without_lsk_is_typed_error(self, fake_spice: FakeSpice) -> None:
        with pytest.raises(UpstreamError) as excinfo:
            await _do_time_convert(
                value=_ET_SECONDS, from_scale="ET", to_scale="UTC", spacecraft=None
            )
        assert excinfo.value.code == "upstream.spice_time_convert_failed"

    async def test_sclk_without_sclk_kernel_is_typed_error(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        # LSK furnished (so str2et succeeds) but no SCLK kernel — sce2s fails.
        await _furnish_lsk(tmp_path)
        _plan_reference(fake_spice)
        with pytest.raises(UpstreamError) as excinfo:
            await _do_time_convert(
                value=_UTC_INPUT, from_scale="UTC", to_scale="SCLK", spacecraft=_SC_ID
            )
        assert excinfo.value.code == "upstream.spice_time_convert_failed"

    @pytest.mark.parametrize(
        ("from_scale", "to_scale", "value"),
        [
            ("UTC", "SCLK", _UTC_INPUT),
            ("SCLK", "UTC", _SCLK_STRING),
        ],
    )
    async def test_sclk_without_spacecraft_is_typed_error_before_cspice(
        self,
        fake_spice: FakeSpice,
        tmp_path: Path,
        from_scale: SpiceTimeScale,
        to_scale: SpiceTimeScale,
        value: str,
    ) -> None:
        await _furnish_lsk(tmp_path)
        await _furnish_sclk(tmp_path)
        _plan_reference(fake_spice)
        with pytest.raises(InvalidInputError) as excinfo:
            await _do_time_convert(
                value=value, from_scale=from_scale, to_scale=to_scale, spacecraft=None
            )
        assert excinfo.value.code == "invalid_input.spice_sclk_requires_spacecraft"
        # The precondition fails before any CSPICE call.
        assert fake_spice.calls["str2et"] == []
        assert fake_spice.calls["scs2e"] == []
        assert fake_spice.calls["sce2s"] == []

    async def test_unknown_spacecraft_is_typed_error(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_sclk(tmp_path)
        _plan_reference(fake_spice)
        # 'VOYAGER_X' is neither a digit string nor a planned name → bods2c not found.
        with pytest.raises(InvalidInputError) as excinfo:
            await _do_time_convert(
                value=_ET_SECONDS, from_scale="ET", to_scale="SCLK", spacecraft="VOYAGER_X"
            )
        assert excinfo.value.code == "invalid_input.spice_unknown_spacecraft"

    async def test_invalid_et_value_is_typed_error_before_cspice(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_lsk(tmp_path)
        with pytest.raises(InvalidInputError) as excinfo:
            await _do_time_convert(
                value="not-a-number", from_scale="ET", to_scale="UTC", spacecraft=None
            )
        assert excinfo.value.code == "invalid_input.spice_invalid_et_value"
        assert fake_spice.calls["et2utc"] == []

    async def test_bare_date_utc_is_typed_error(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_lsk(tmp_path)
        with pytest.raises(InvalidInputError) as excinfo:
            await _do_time_convert(
                value="2026-01-01", from_scale="UTC", to_scale="ET", spacecraft=None
            )
        assert excinfo.value.code == "invalid_input.epoch_missing_time_component"
        assert fake_spice.calls["str2et"] == []

    async def test_non_string_sclk_value_is_typed_error(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_sclk(tmp_path)
        _plan_reference(fake_spice)
        with pytest.raises(InvalidInputError) as excinfo:
            await _do_time_convert(
                value=1465644281.0, from_scale="SCLK", to_scale="ET", spacecraft=_SC_ID
            )
        assert excinfo.value.code == "invalid_input.spice_invalid_sclk_value"
        assert fake_spice.calls["scs2e"] == []


# ---------------------------------------------------------------------------
# Output-schema round-trip
# ---------------------------------------------------------------------------


class TestOutputRoundTrip:
    def test_quantity_value_roundtrips_through_schema(self) -> None:
        response = SpiceTimeConvertResponse(
            value=Quantity(value=_ET_SECONDS, unit="s past J2000 TDB"),
            from_scale="UTC",
            to_scale="ET",
            spacecraft=None,
        )
        first = response.model_dump_json()
        rebuilt = SpiceTimeConvertResponse.model_validate_json(first)
        assert rebuilt.model_dump_json() == first
        assert isinstance(rebuilt.value, Quantity)

    def test_string_value_roundtrips_through_schema(self) -> None:
        response = SpiceTimeConvertResponse(
            value=_SCLK_STRING,
            from_scale="ET",
            to_scale="SCLK",
            spacecraft="-82",
        )
        first = response.model_dump_json()
        rebuilt = SpiceTimeConvertResponse.model_validate_json(first)
        assert rebuilt.model_dump_json() == first
        assert rebuilt.value == _SCLK_STRING

    def test_extra_keys_forbidden(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SpiceTimeConvertResponse.model_validate(
                {
                    "value": "2026-01-01T00:00:00.000000",
                    "from_scale": "ET",
                    "to_scale": "UTC",
                    "spacecraft": None,
                    "surprise": True,
                }
            )


# ---------------------------------------------------------------------------
# Committed reference golden
# ---------------------------------------------------------------------------


class TestReferenceGolden:
    async def test_utc_to_et_matches_committed_golden(
        self, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_lsk(tmp_path)
        _plan_reference(fake_spice)

        response = await _do_time_convert(
            value=_UTC_INPUT, from_scale="UTC", to_scale="ET", spacecraft=None
        )
        actual = response.model_dump(mode="json")
        expected = json.loads(_GOLDEN_PATH.read_text())
        assert actual == expected


# ---------------------------------------------------------------------------
# End-to-end through the registered slot
# ---------------------------------------------------------------------------


class TestRegisteredToolCall:
    @pytest.fixture
    def registered_mcp(self, monkeypatch: pytest.MonkeyPatch) -> FastMCP:
        fresh = FastMCP("spice-time-convert-test")
        monkeypatch.setattr("astrodynamics_mcp.server.mcp", fresh)
        spice_tools._register_spice_tools()
        return fresh

    async def test_utc_to_et_via_slot(
        self, registered_mcp: FastMCP, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_lsk(tmp_path)
        _plan_reference(fake_spice)

        _, result = await registered_mcp.call_tool(
            "spice_time_convert",
            {"value": _UTC_INPUT, "from_scale": "UTC", "to_scale": "ET"},
        )
        assert isinstance(result, dict)
        assert result["to_scale"] == "ET"
        assert result["value"]["value"] == _ET_SECONDS
        assert result["value"]["unit"] == "s past J2000 TDB"
        assert result["spacecraft"] is None

    async def test_utc_to_sclk_via_slot(
        self, registered_mcp: FastMCP, fake_spice: FakeSpice, tmp_path: Path
    ) -> None:
        await _furnish_lsk(tmp_path)
        await _furnish_sclk(tmp_path)
        _plan_reference(fake_spice)

        _, result = await registered_mcp.call_tool(
            "spice_time_convert",
            {
                "value": _UTC_INPUT,
                "from_scale": "UTC",
                "to_scale": "SCLK",
                "spacecraft": _SC_ID,
            },
        )
        assert isinstance(result, dict)
        assert result["to_scale"] == "SCLK"
        assert result["value"] == _SCLK_STRING
        assert result["spacecraft"] == "-82"
