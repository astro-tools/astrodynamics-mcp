"""Tests for `astrodynamics_mcp.tools.time`.

Astropy + the IERS shim run deterministically — no network mocking.
Coverage: leap-second-aware UTC/TAI, GPS-UTC, UT1 (with IERS metadata),
the five formats including the custom `j2000_seconds`, round-trip
preservation, validation errors, registration + description-lint.
"""

from __future__ import annotations

import json

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from astrodynamics_mcp.schemas.base import TimeScale
from astrodynamics_mcp.server import mcp
from astrodynamics_mcp.server_lint import check_tool_descriptions
from astrodynamics_mcp.tools.time import TimeConvertResponse, time_convert


class TestLeapSecondAware:
    async def test_utc_to_tai_offset_is_37_seconds(self) -> None:
        resp = await time_convert(
            value="2026-05-23T12:00:00",
            from_scale=TimeScale.UTC,
            to_scale=TimeScale.TAI,
        )
        assert isinstance(resp, TimeConvertResponse)
        assert resp.scale == TimeScale.TAI
        assert resp.format == "iso"
        assert isinstance(resp.value, str)
        # astropy's isot includes fractional milliseconds: '2026-05-23T12:00:37.000'.
        assert resp.value.startswith("2026-05-23T12:00:37")

    async def test_gps_to_utc_offset_is_minus_18_seconds(self) -> None:
        """GPS 12:00:00 → UTC 11:59:42 (GPS - UTC = 18 s currently)."""
        resp = await time_convert(
            value="2026-05-23T12:00:00",
            from_scale=TimeScale.GPS,
            to_scale=TimeScale.UTC,
        )
        assert isinstance(resp.value, str)
        assert resp.value.startswith("2026-05-23T11:59:42")

    async def test_utc_to_gps_offset_is_plus_18_seconds(self) -> None:
        """UTC 12:00:00 → GPS 12:00:18 (the inverse of the above)."""
        resp = await time_convert(
            value="2026-05-23T12:00:00",
            from_scale=TimeScale.UTC,
            to_scale=TimeScale.GPS,
        )
        assert isinstance(resp.value, str)
        assert resp.value.startswith("2026-05-23T12:00:18")


class TestUt1WithIers:
    async def test_utc_to_ut1_populates_offset_and_freshness(self) -> None:
        resp = await time_convert(
            value="2026-05-23T12:00:00",
            from_scale=TimeScale.UTC,
            to_scale=TimeScale.UT1,
        )
        assert resp.ut1_utc_seconds is not None
        assert resp.ut1_utc_seconds.unit == "s"
        # |UT1-UTC| is always < 0.9 s by design (IERS introduces leap seconds
        # to keep it bounded).
        assert abs(resp.ut1_utc_seconds.value) < 1.0
        assert resp.iers_fetched_at is not None
        # IERS freshness anchor is an ISO 8601 string.
        assert "T" in resp.iers_fetched_at

    async def test_non_ut1_conversion_leaves_iers_metadata_null(self) -> None:
        resp = await time_convert(
            value="2026-05-23T12:00:00",
            from_scale=TimeScale.UTC,
            to_scale=TimeScale.TAI,
        )
        assert resp.ut1_utc_seconds is None
        assert resp.iers_fetched_at is None


class TestFormatOutputs:
    async def test_mjd_output_is_a_finite_float(self) -> None:
        resp = await time_convert(
            value="2026-05-23T12:00:00",
            from_scale=TimeScale.UTC,
            to_scale=TimeScale.UTC,
            out_format="mjd",
        )
        assert resp.format == "mjd"
        assert isinstance(resp.value, float)
        # Modified Julian Day starts at 1858-11-17; 2026-05-23 ≈ 61183.
        assert 61000 < resp.value < 62000

    async def test_jd_output_is_a_finite_float(self) -> None:
        resp = await time_convert(
            value="2026-05-23T12:00:00",
            from_scale=TimeScale.UTC,
            to_scale=TimeScale.UTC,
            out_format="jd",
        )
        assert isinstance(resp.value, float)
        # JD for 2026-05-23T12:00:00 UTC is roughly 2460854.
        assert 2_460_000 < resp.value < 2_461_500

    async def test_j2000_seconds_at_the_j2000_epoch_is_zero(self) -> None:
        resp = await time_convert(
            value="2000-01-01T12:00:00",
            from_scale=TimeScale.TT,
            to_scale=TimeScale.TT,
            out_format="j2000_seconds",
        )
        assert isinstance(resp.value, float)
        assert abs(resp.value) < 1e-6

    async def test_j2000_seconds_round_trip_through_input(self) -> None:
        """j2000_seconds → TT iso → j2000_seconds preserves the value."""
        original_secs = 12345.6789
        as_iso = await time_convert(
            value=original_secs,
            from_scale=TimeScale.TT,
            to_scale=TimeScale.TT,
            in_format="j2000_seconds",
            out_format="iso",
        )
        back_to_secs = await time_convert(
            value=as_iso.value,
            from_scale=TimeScale.TT,
            to_scale=TimeScale.TT,
            in_format="iso",
            out_format="j2000_seconds",
        )
        # j2000_seconds is float64 → expect at most a few ms of slop near present-day.
        assert isinstance(back_to_secs.value, float)
        assert abs(back_to_secs.value - original_secs) < 1e-3

    async def test_unix_output(self) -> None:
        resp = await time_convert(
            value="2026-05-23T12:00:00",
            from_scale=TimeScale.UTC,
            to_scale=TimeScale.UTC,
            out_format="unix",
        )
        assert isinstance(resp.value, float)
        # 2026-05-23T12:00:00 UTC = unix 1779537600.
        assert 1_779_000_000 < resp.value < 1_780_000_000

    async def test_unix_output_reports_utc_anchor_regardless_of_to_scale(self) -> None:
        # `unix` is UTC-anchored; the value must not depend on `to_scale`, and
        # the response `scale` must report the true anchor (UTC), not the
        # inapplicable requested `to_scale`.
        resp = await time_convert(
            value="2026-05-23T12:00:00",
            from_scale=TimeScale.UTC,
            to_scale=TimeScale.TAI,
            out_format="unix",
        )
        assert resp.scale == TimeScale.UTC
        assert resp.value == pytest.approx(1779537600.0)

    async def test_j2000_seconds_reports_tt_anchor_regardless_of_to_scale(self) -> None:
        resp = await time_convert(
            value="2000-01-01T12:00:00",
            from_scale=TimeScale.TT,
            to_scale=TimeScale.UTC,
            out_format="j2000_seconds",
        )
        assert resp.scale == TimeScale.TT
        assert abs(resp.value) < 1e-6


class TestRoundTrip:
    async def test_utc_tai_utc_iso_preserves_value(self) -> None:
        r1 = await time_convert(
            value="2026-05-23T12:00:00",
            from_scale=TimeScale.UTC,
            to_scale=TimeScale.TAI,
        )
        r2 = await time_convert(
            value=r1.value,
            from_scale=TimeScale.TAI,
            to_scale=TimeScale.UTC,
        )
        assert isinstance(r2.value, str)
        # Round-trip back to the original second.
        assert r2.value.startswith("2026-05-23T12:00:00")

    async def test_jd_round_trip_preserves_value(self) -> None:
        original = 2460853.0
        r1 = await time_convert(
            value=original,
            from_scale=TimeScale.UTC,
            to_scale=TimeScale.TAI,
            in_format="jd",
            out_format="jd",
        )
        r2 = await time_convert(
            value=r1.value,
            from_scale=TimeScale.TAI,
            to_scale=TimeScale.UTC,
            in_format="jd",
            out_format="jd",
        )
        assert isinstance(r2.value, float)
        # JD round-trip via Time has ~µs precision.
        assert abs(r2.value - original) < 1e-9

    async def test_gps_utc_gps_iso_preserves_value(self) -> None:
        """GPS → UTC → GPS should round-trip cleanly through the 19 s shift."""
        original = "2026-05-23T12:00:00"
        r1 = await time_convert(
            value=original,
            from_scale=TimeScale.GPS,
            to_scale=TimeScale.UTC,
        )
        r2 = await time_convert(
            value=r1.value,
            from_scale=TimeScale.UTC,
            to_scale=TimeScale.GPS,
        )
        assert isinstance(r2.value, str)
        assert r2.value.startswith(original)


class TestErrorPaths:
    async def test_malformed_iso_raises_typed_envelope(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await time_convert(
                value="not a date",
                from_scale=TimeScale.UTC,
                to_scale=TimeScale.TAI,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.invalid_time_value"

    async def test_non_numeric_j2000_seconds_raises_typed_envelope(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await time_convert(
                value="not a number",
                from_scale=TimeScale.TT,
                to_scale=TimeScale.TT,
                in_format="j2000_seconds",
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.invalid_time_value"

    async def test_boolean_j2000_seconds_rejected(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await time_convert(
                value=True,
                from_scale=TimeScale.TT,
                to_scale=TimeScale.TT,
                in_format="j2000_seconds",
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.invalid_time_value"

    async def test_malformed_unix_value_rejected(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await time_convert(
                value="not a number",
                from_scale=TimeScale.UTC,
                to_scale=TimeScale.UTC,
                in_format="unix",
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.invalid_time_value"

    async def test_malformed_iso_in_gps_scale_rejected(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await time_convert(
                value="not a date",
                from_scale=TimeScale.GPS,
                to_scale=TimeScale.UTC,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.invalid_time_value"


class TestGpsOutputFormats:
    """GPS-scale output goes through the shifted-TAI proxy for scale-bound formats."""

    @pytest.mark.parametrize("out_format", ["jd", "mjd"])
    async def test_utc_to_gps_in_scale_bound_format_shifts_by_18s(self, out_format: str) -> None:
        """For JD / MJD the GPS-scale value is 18 s later than UTC at the same instant."""
        resp = await time_convert(
            value="2026-05-23T12:00:00",
            from_scale=TimeScale.UTC,
            to_scale=TimeScale.GPS,
            out_format=out_format,  # type: ignore[arg-type]
        )
        utc_resp = await time_convert(
            value="2026-05-23T12:00:00",
            from_scale=TimeScale.UTC,
            to_scale=TimeScale.UTC,
            out_format=out_format,  # type: ignore[arg-type]
        )
        assert isinstance(resp.value, float)
        assert isinstance(utc_resp.value, float)
        assert resp.value == pytest.approx(utc_resp.value + 18.0 / 86400.0, abs=1e-9)


class TestAbsoluteFormatInvariance:
    """`unix` and `j2000_seconds` are absolute counters — `to_scale` does not shift them."""

    @pytest.mark.parametrize("absolute_format", ["unix", "j2000_seconds"])
    async def test_to_scale_does_not_shift_absolute_format(self, absolute_format: str) -> None:
        as_utc = await time_convert(
            value="2026-05-23T12:00:00",
            from_scale=TimeScale.UTC,
            to_scale=TimeScale.UTC,
            out_format=absolute_format,  # type: ignore[arg-type]
        )
        as_gps = await time_convert(
            value="2026-05-23T12:00:00",
            from_scale=TimeScale.UTC,
            to_scale=TimeScale.GPS,
            out_format=absolute_format,  # type: ignore[arg-type]
        )
        as_tai = await time_convert(
            value="2026-05-23T12:00:00",
            from_scale=TimeScale.UTC,
            to_scale=TimeScale.TAI,
            out_format=absolute_format,  # type: ignore[arg-type]
        )
        assert as_utc.value == as_gps.value == as_tai.value

    async def test_j2000_seconds_input_is_scale_invariant(self) -> None:
        """`j2000_seconds` input ignores `from_scale` — same instant either way."""
        as_tai = await time_convert(
            value=12345.6789,
            from_scale=TimeScale.TAI,
            to_scale=TimeScale.TT,
            in_format="j2000_seconds",
            out_format="iso",
        )
        as_gps = await time_convert(
            value=12345.6789,
            from_scale=TimeScale.GPS,
            to_scale=TimeScale.TT,
            in_format="j2000_seconds",
            out_format="iso",
        )
        # j2000_seconds is TT-anchored by definition — the from_scale label
        # is informational only, so both produce the same TT ISO output.
        assert as_tai.value == as_gps.value

    async def test_unix_input_is_scale_invariant(self) -> None:
        """`unix` input ignores `from_scale` — value is always UTC-rooted."""
        as_utc = await time_convert(
            value=1779537600.0,
            from_scale=TimeScale.UTC,
            to_scale=TimeScale.TAI,
            in_format="unix",
            out_format="iso",
        )
        as_gps = await time_convert(
            value=1779537600.0,
            from_scale=TimeScale.GPS,
            to_scale=TimeScale.TAI,
            in_format="unix",
            out_format="iso",
        )
        assert as_utc.value == as_gps.value


class TestIersFailureWrapping:
    async def test_iers_load_failure_surfaces_as_typed_envelope(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When `load_iers` raises, the tool surfaces `upstream.iers_unavailable`."""

        def boom() -> None:
            raise RuntimeError("simulated IERS outage")

        # Patch the symbol the tool imports lazily; the tool does
        # `from astrodynamics_mcp.data.iers import load_iers` inside the
        # function body, so patching the source module is the right path.
        import astrodynamics_mcp.data.iers as iers_mod

        monkeypatch.setattr(iers_mod, "load_iers", boom)

        with pytest.raises(ToolError) as excinfo:
            await time_convert(
                value="2026-05-23T12:00:00",
                from_scale=TimeScale.UTC,
                to_scale=TimeScale.UT1,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "upstream.iers_unavailable"
        assert "simulated IERS outage" in envelope["message"]


class TestRegistration:
    async def test_tool_is_listed(self) -> None:
        tools = await mcp.list_tools()
        names = {t.name for t in tools}
        assert "time_convert" in names

    async def test_tool_description_passes_lint(self) -> None:
        tools = await mcp.list_tools()
        violations = [v for v in check_tool_descriptions(tools) if v.tool_name == "time_convert"]
        assert violations == []

    async def test_tool_callable_via_mcp(self) -> None:
        content, structured = await mcp.call_tool(
            "time_convert",
            {
                "value": "2026-05-23T12:00:00",
                "from_scale": "UTC",
                "to_scale": "TAI",
            },
        )
        del content
        assert isinstance(structured, dict)
        assert structured["scale"] == "TAI"
        assert structured["value"].startswith("2026-05-23T12:00:37")


class TestSchemaInvariants:
    async def test_response_round_trips_through_json(self) -> None:
        resp = await time_convert(
            value="2026-05-23T12:00:00",
            from_scale=TimeScale.UTC,
            to_scale=TimeScale.UT1,
        )
        as_json = resp.model_dump_json()
        rebuilt = TimeConvertResponse.model_validate_json(as_json)
        assert rebuilt == resp
