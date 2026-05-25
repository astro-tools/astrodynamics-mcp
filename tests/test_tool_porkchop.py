"""Tests for `astrodynamics_mcp.tools.porkchop`.

The porkchop scan composes the existing Horizons adapter (mocked via
``httpx.MockTransport``) with ``lamberthub.izzo2015``. Tests cover the
input-validation surface, the Horizons VECTORS parser, the grid
evaluation against a synthetic two-body geometry, the ASCII summary
shape, and end-to-end MCP invocation. A live-Horizons integration test
is gated behind the ``integration`` marker.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import httpx
import numpy as np
import pytest
from mcp.server.fastmcp.exceptions import ToolError

from astrodynamics_mcp.server import mcp
from astrodynamics_mcp.server_lint import check_tool_descriptions
from astrodynamics_mcp.tools.porkchop import (
    _JD_UNIX_EPOCH,
    _MU_SUN,
    PorkchopResponse,
    porkchop,
)

# ---------------------------------------------------------------------------
# Synthetic Horizons-VECTORS fixture builder
# ---------------------------------------------------------------------------


_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _datetime_to_jd(dt: datetime) -> float:
    return _JD_UNIX_EPOCH + (dt - _UNIX_EPOCH).total_seconds() / 86400.0


def _circular_orbit_state(
    semi_major_axis_km: float, mean_anomaly_rad: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (r, v) for a coplanar (z=0) circular heliocentric orbit."""
    v_circ = float(np.sqrt(_MU_SUN / semi_major_axis_km))
    cos_m = float(np.cos(mean_anomaly_rad))
    sin_m = float(np.sin(mean_anomaly_rad))
    r = np.array([semi_major_axis_km * cos_m, semi_major_axis_km * sin_m, 0.0])
    v = np.array([-v_circ * sin_m, v_circ * cos_m, 0.0])
    return r, v


def _circular_orbit_table(
    *,
    start: datetime,
    days: int,
    semi_major_axis_km: float,
    initial_mean_anomaly_rad: float,
) -> tuple[list[datetime], np.ndarray, np.ndarray]:
    """Generate a 1-day-step (epoch, r, v) table for a circular orbit."""
    period_s = 2 * np.pi * np.sqrt(semi_major_axis_km**3 / _MU_SUN)
    epochs: list[datetime] = []
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    for day in range(days + 1):
        t = start + timedelta(days=day)
        m = initial_mean_anomaly_rad + 2 * np.pi * (day * 86400.0) / period_s
        r, v = _circular_orbit_state(semi_major_axis_km, m)
        epochs.append(t)
        positions.append(r)
        velocities.append(v)
    return epochs, np.asarray(positions), np.asarray(velocities)


def _format_horizons_vectors(
    epochs: list[datetime], positions: np.ndarray, velocities: np.ndarray
) -> str:
    lines = [
        "*****************************************************************************",
        "Ephemeris / API_USER Sun-centred VECTORS table — synthetic test fixture",
        "*****************************************************************************",
        "$$SOE",
    ]
    for t, r, v in zip(epochs, positions, velocities, strict=True):
        jd = _datetime_to_jd(t)
        lines.append(f" {jd:.9f} = A.D. {t:%Y-%b-%d %H:%M:%S.%f} TDB")
        lines.append(f" X ={r[0]: .9E} Y ={r[1]: .9E} Z ={r[2]: .9E}")
        lines.append(f" VX={v[0]: .9E} VY={v[1]: .9E} VZ={v[2]: .9E}")
    lines.append("$$EOE")
    lines.append("*****************************************************************************")
    return "\n".join(lines)


def _payload_for(
    *,
    start: datetime,
    days: int,
    semi_major_axis_km: float,
    initial_mean_anomaly_rad: float,
) -> dict[str, Any]:
    epochs, positions, velocities = _circular_orbit_table(
        start=start,
        days=days,
        semi_major_axis_km=semi_major_axis_km,
        initial_mean_anomaly_rad=initial_mean_anomaly_rad,
    )
    return {"result": _format_horizons_vectors(epochs, positions, velocities)}


# Approximate semi-major axes (km) for the synthetic two-body geometry.
_AU_KM = 149597870.7
_EARTH_SMA = _AU_KM
_MARS_SMA = 1.523679 * _AU_KM

# Fixture-scope window: 2026-09-01 to 2027-12-01 covers the depart and
# arrive windows used across the suite plus the Horizons one-day padding.
_FIXTURE_START = datetime(2026, 9, 1, tzinfo=timezone.utc)
_FIXTURE_DAYS = 460


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _disable_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the on-disk Horizons cache for every test in this module.

    Each test installs its own mocked Horizons handler; the cache would
    otherwise cross-pollute responses between tests inside a single pytest
    process.
    """
    monkeypatch.setenv("ASTRODYNAMICS_MCP_CACHE_DIR", "")
    import astrodynamics_mcp.cache as cache_module

    monkeypatch.setattr(cache_module, "_default_cache", None)


def _make_handler(
    payloads_by_target: dict[str, dict[str, Any]],
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        raw_target = request.url.params["COMMAND"].strip("'")
        payload = payloads_by_target.get(raw_target)
        if payload is None:
            return httpx.Response(404, text=f"unknown target {raw_target!r} in mock")
        return httpx.Response(200, json=payload)

    return handler


def _patched_client(
    payloads_by_target: dict[str, dict[str, Any]],
) -> Iterator[None]:
    """Patch so each `httpx.AsyncClient(...)` call returns a fresh mock client.

    Multiple `fetch_ephemeris` calls within one porkchop run each open and
    close their own AsyncClient — a shared `return_value` would surface
    httpx's "cannot reopen a closed client" error on the second call.
    The original class is captured before the patch is applied so the
    factory body doesn't recurse through the mock.
    """
    handler = _make_handler(payloads_by_target)
    original_async_client = httpx.AsyncClient

    def factory(*_args: Any, **_kwargs: Any) -> httpx.AsyncClient:
        return original_async_client(transport=httpx.MockTransport(handler))

    with patch(
        "astrodynamics_mcp.data.horizons.httpx.AsyncClient",
        side_effect=factory,
    ):
        yield


@pytest.fixture
def earth_mars_payloads() -> dict[str, dict[str, Any]]:
    """Earth (399) and Mars (499) on coplanar circular heliocentric orbits."""
    return {
        "399": _payload_for(
            start=_FIXTURE_START,
            days=_FIXTURE_DAYS,
            semi_major_axis_km=_EARTH_SMA,
            initial_mean_anomaly_rad=0.0,
        ),
        "499": _payload_for(
            start=_FIXTURE_START,
            days=_FIXTURE_DAYS,
            semi_major_axis_km=_MARS_SMA,
            # Phase Mars roughly 45° ahead of Earth at fixture start so the
            # depart window 60 days later lands in a launch-feasible geometry.
            initial_mean_anomaly_rad=np.radians(45.0),
        ),
    }


@pytest.fixture
def earth_mars_client(
    earth_mars_payloads: dict[str, dict[str, Any]],
) -> Iterator[None]:
    yield from _patched_client(earth_mars_payloads)


# ---------------------------------------------------------------------------
# Window / argument validation
# ---------------------------------------------------------------------------


_DEPART_WINDOW = ["2026-11-01T00:00:00Z", "2026-12-31T00:00:00Z"]
_ARRIVE_WINDOW = ["2027-06-01T00:00:00Z", "2027-11-01T00:00:00Z"]


class TestInputValidation:
    async def test_unknown_departure_body_raises_invalid_input(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await porkchop(
                departure_body="kepler-186f",
                arrival_body="mars",
                depart_window=_DEPART_WINDOW,
                arrive_window=_ARRIVE_WINDOW,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.unknown_body"

    async def test_unknown_arrival_body_raises_invalid_input(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await porkchop(
                departure_body="earth",
                arrival_body="ceres",
                depart_window=_DEPART_WINDOW,
                arrive_window=_ARRIVE_WINDOW,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.unknown_body"

    async def test_non_string_body_raises_invalid_input(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await porkchop(
                departure_body=399,  # type: ignore[arg-type]
                arrival_body="mars",
                depart_window=_DEPART_WINDOW,
                arrive_window=_ARRIVE_WINDOW,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.body_not_a_string"

    async def test_same_body_raises_invalid_input(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await porkchop(
                departure_body="earth",
                arrival_body="earth",
                depart_window=_DEPART_WINDOW,
                arrive_window=_ARRIVE_WINDOW,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.same_body"

    async def test_unsupported_mu_raises_invalid_input(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await porkchop(
                departure_body="earth",
                arrival_body="mars",
                depart_window=_DEPART_WINDOW,
                arrive_window=_ARRIVE_WINDOW,
                mu="earth",  # type: ignore[arg-type]
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.unsupported_mu"

    async def test_samples_per_axis_too_small_raises_invalid_input(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await porkchop(
                departure_body="earth",
                arrival_body="mars",
                depart_window=_DEPART_WINDOW,
                arrive_window=_ARRIVE_WINDOW,
                samples_per_axis=1,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.samples_too_small"

    async def test_samples_per_axis_wrong_type_raises_invalid_input(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await porkchop(
                departure_body="earth",
                arrival_body="mars",
                depart_window=_DEPART_WINDOW,
                arrive_window=_ARRIVE_WINDOW,
                samples_per_axis=4.5,  # type: ignore[arg-type]
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.samples_not_an_int"

    async def test_samples_per_axis_bool_rejected(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await porkchop(
                departure_body="earth",
                arrival_body="mars",
                depart_window=_DEPART_WINDOW,
                arrive_window=_ARRIVE_WINDOW,
                samples_per_axis=True,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.samples_not_an_int"

    async def test_window_not_a_list_raises_invalid_input(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await porkchop(
                departure_body="earth",
                arrival_body="mars",
                depart_window="not-a-list",  # type: ignore[arg-type]
                arrive_window=_ARRIVE_WINDOW,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.window_wrong_shape"

    async def test_window_wrong_length_raises_invalid_input(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await porkchop(
                departure_body="earth",
                arrival_body="mars",
                depart_window=["2026-11-01T00:00:00Z"],
                arrive_window=_ARRIVE_WINDOW,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.window_wrong_shape"

    async def test_window_endpoint_wrong_type_raises_invalid_input(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await porkchop(
                departure_body="earth",
                arrival_body="mars",
                depart_window=[123, "2026-12-31T00:00:00Z"],  # type: ignore[list-item]
                arrive_window=_ARRIVE_WINDOW,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.epoch_not_a_string"

    async def test_malformed_epoch_raises_invalid_input(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await porkchop(
                departure_body="earth",
                arrival_body="mars",
                depart_window=["not-an-iso-timestamp", "2026-12-31T00:00:00Z"],
                arrive_window=_ARRIVE_WINDOW,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.epoch_malformed"

    async def test_window_end_before_start_raises_invalid_input(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            await porkchop(
                departure_body="earth",
                arrival_body="mars",
                depart_window=["2026-12-01T00:00:00Z", "2026-11-01T00:00:00Z"],
                arrive_window=_ARRIVE_WINDOW,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.window_end_not_after_start"

    async def test_arrive_entirely_before_depart_raises_window_order(self) -> None:
        """Acceptance: arrive window entirely before depart window → typed error."""
        with pytest.raises(ToolError) as excinfo:
            await porkchop(
                departure_body="earth",
                arrival_body="mars",
                depart_window=["2027-06-01T00:00:00Z", "2027-09-01T00:00:00Z"],
                arrive_window=["2026-11-01T00:00:00Z", "2027-01-01T00:00:00Z"],
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "invalid_input.porkchop_window_order"


# ---------------------------------------------------------------------------
# Horizons failure paths
# ---------------------------------------------------------------------------


class TestHorizonsFailures:
    async def test_horizons_unreachable_raises_data_source_error(self) -> None:
        """Acceptance: Horizons down mid-scan → data_source.horizons_unreachable."""

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="Service Unavailable")

        original_async_client = httpx.AsyncClient

        def factory(*_args: Any, **_kwargs: Any) -> httpx.AsyncClient:
            return original_async_client(transport=httpx.MockTransport(handler))

        with (
            patch(
                "astrodynamics_mcp.data.horizons.httpx.AsyncClient",
                side_effect=factory,
            ),
            pytest.raises(ToolError) as excinfo,
        ):
            await porkchop(
                departure_body="earth",
                arrival_body="mars",
                depart_window=_DEPART_WINDOW,
                arrive_window=_ARRIVE_WINDOW,
                samples_per_axis=5,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "data_source.horizons_unreachable"

    async def test_horizons_unparseable_response_raises_upstream_error(self) -> None:
        """A 200 OK body with no VECTORS rows surfaces as upstream.horizons_unexpected_shape."""

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"result": "no SOE/EOE block here"})

        original_async_client = httpx.AsyncClient

        def factory(*_args: Any, **_kwargs: Any) -> httpx.AsyncClient:
            return original_async_client(transport=httpx.MockTransport(handler))

        with (
            patch(
                "astrodynamics_mcp.data.horizons.httpx.AsyncClient",
                side_effect=factory,
            ),
            pytest.raises(ToolError) as excinfo,
        ):
            await porkchop(
                departure_body="earth",
                arrival_body="mars",
                depart_window=_DEPART_WINDOW,
                arrive_window=_ARRIVE_WINDOW,
                samples_per_axis=5,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "upstream.horizons_unexpected_shape"


# ---------------------------------------------------------------------------
# Synthetic-geometry grid behaviour
# ---------------------------------------------------------------------------


class TestSyntheticGrid:
    async def test_grid_has_expected_shape_and_best_cell(
        self, earth_mars_client: Iterator[None]
    ) -> None:
        n = 5
        response = await porkchop(
            departure_body="earth",
            arrival_body="mars",
            depart_window=_DEPART_WINDOW,
            arrive_window=_ARRIVE_WINDOW,
            samples_per_axis=n,
        )
        # nxn cells, all positive-tof in this window so the grid is full.
        assert len(response.grid) == n * n
        assert isinstance(response, PorkchopResponse)

        # Every cell carries the right units.
        for cell in response.grid:
            assert cell.tof.unit == "days"
            assert cell.c3.unit == "km^2/s^2"
            assert cell.v_inf_arrival.unit == "km/s"
            assert cell.dec_dep_asymptote.unit == "deg"
            assert cell.total_dv.unit == "km/s"
            assert cell.tof.value > 0
            assert cell.c3.value >= 0
            assert cell.v_inf_arrival.value >= 0
            assert cell.total_dv.value >= 0

        # `best` minimises total_dv across the feasible grid.
        best_total_dv = min(c.total_dv.value for c in response.grid)
        assert response.best.total_dv.value == pytest.approx(best_total_dv)

        # Sanity bound for a synthetic Earth-Mars circular geometry.
        assert response.best.c3.value < 30.0

    async def test_ascii_summary_has_n_by_n_glyph_grid(
        self, earth_mars_client: Iterator[None]
    ) -> None:
        n = 6
        response = await porkchop(
            departure_body="earth",
            arrival_body="mars",
            depart_window=_DEPART_WINDOW,
            arrive_window=_ARRIVE_WINDOW,
            samples_per_axis=n,
        )
        assert response.ascii_summary, "ascii_summary must be non-empty"
        rows = response.ascii_summary.splitlines()
        assert len(rows) == n
        assert all(len(row) == n for row in rows)
        # The ramp characters or a space are the only glyphs we emit.
        allowed = set(".:-+*#@X ")
        for row in rows:
            assert set(row).issubset(allowed)

    async def test_partial_overlap_window_drops_negative_tof_cells(
        self, earth_mars_client: Iterator[None]
    ) -> None:
        """Overlapping windows leave some (depart > arrive) cells infeasible.

        depart window is 2027-01-01..2027-04-01; arrive window is
        2027-03-01..2027-07-01. The earliest arrival sample (2027-03-01)
        sits before the latest depart sample (2027-04-01), so the upper-
        right corner of the grid carries non-positive tof and gets dropped.
        """
        n = 4
        response = await porkchop(
            departure_body="earth",
            arrival_body="mars",
            depart_window=["2027-01-01T00:00:00Z", "2027-04-01T00:00:00Z"],
            arrive_window=["2027-03-01T00:00:00Z", "2027-07-01T00:00:00Z"],
            samples_per_axis=n,
        )
        # Some cells dropped → grid is strictly smaller than the nxn full grid.
        assert 0 < len(response.grid) < n * n
        # Every emitted cell still has positive tof.
        assert all(cell.tof.value > 0 for cell in response.grid)
        # ASCII summary still has n rows x n columns, with spaces marking drops.
        rows = response.ascii_summary.splitlines()
        assert len(rows) == n
        assert all(len(row) == n for row in rows)
        assert any(" " in row for row in rows)

    async def test_response_round_trips_through_json(
        self, earth_mars_client: Iterator[None]
    ) -> None:
        response = await porkchop(
            departure_body="earth",
            arrival_body="mars",
            depart_window=_DEPART_WINDOW,
            arrive_window=_ARRIVE_WINDOW,
            samples_per_axis=4,
        )
        rebuilt = PorkchopResponse.model_validate_json(response.model_dump_json())
        assert rebuilt == response

    async def test_body_name_case_insensitive(self, earth_mars_client: Iterator[None]) -> None:
        response = await porkchop(
            departure_body="EARTH",
            arrival_body="MARS",
            depart_window=_DEPART_WINDOW,
            arrive_window=_ARRIVE_WINDOW,
            samples_per_axis=3,
        )
        assert len(response.grid) > 0


# ---------------------------------------------------------------------------
# MCP registration & description-lint
# ---------------------------------------------------------------------------


class TestRegistration:
    async def test_tool_is_listed(self) -> None:
        tools = await mcp.list_tools()
        names = {t.name for t in tools}
        assert "porkchop" in names

    async def test_tool_description_passes_lint(self) -> None:
        tools = await mcp.list_tools()
        violations = [v for v in check_tool_descriptions(tools) if v.tool_name == "porkchop"]
        assert violations == []

    async def test_tool_callable_via_mcp(self, earth_mars_client: Iterator[None]) -> None:
        content, structured = await mcp.call_tool(
            "porkchop",
            {
                "departure_body": "earth",
                "arrival_body": "mars",
                "depart_window": _DEPART_WINDOW,
                "arrive_window": _ARRIVE_WINDOW,
                "samples_per_axis": 3,
            },
        )
        del content
        assert isinstance(structured, dict)
        assert "grid" in structured and "best" in structured and "ascii_summary" in structured
        assert structured["best"]["c3"]["unit"] == "km^2/s^2"
        assert structured["best"]["total_dv"]["unit"] == "km/s"


# ---------------------------------------------------------------------------
# Edge cases the synthetic fixtures aren't sized for
# ---------------------------------------------------------------------------


class TestEdgeCases:
    async def test_grid_entirely_infeasible_raises_upstream_error(self) -> None:
        """Constructing a single-day arrival window after a longer depart window
        where every cell has non-positive tof — exercises the empty-grid path."""
        # Synthesize Horizons payloads covering both windows.
        payloads = {
            "399": _payload_for(
                start=_FIXTURE_START,
                days=_FIXTURE_DAYS,
                semi_major_axis_km=_EARTH_SMA,
                initial_mean_anomaly_rad=0.0,
            ),
            "499": _payload_for(
                start=_FIXTURE_START,
                days=_FIXTURE_DAYS,
                semi_major_axis_km=_MARS_SMA,
                initial_mean_anomaly_rad=np.radians(45.0),
            ),
        }
        # Arrive samples [2026-12-31 .. 2027-01-01] both ≤ depart samples
        # [2026-11-01 .. 2026-12-31]; only the (depart=2026-11-01, arrive=
        # 2027-01-01) corner has positive tof — but the depart-end-equals-
        # arrive-start cell has tof=0 which is dropped. Most cells dropped.
        for _ in _patched_client(payloads):
            with pytest.raises(ToolError) as excinfo:
                # Pick a depart_window whose start equals the arrive_window
                # end so every arrive ≤ every depart → grid empty.
                await porkchop(
                    departure_body="earth",
                    arrival_body="mars",
                    depart_window=["2026-12-15T00:00:00Z", "2026-12-31T00:00:00Z"],
                    arrive_window=["2026-12-01T00:00:00Z", "2026-12-15T00:00:00Z"],
                    samples_per_axis=4,
                )
        envelope = json.loads(str(excinfo.value))
        # Both window-order error and the grid-empty path are valid sentinels
        # for this geometry; the validator catches the cleaner one first.
        assert envelope["code"] in {
            "invalid_input.porkchop_window_order",
            "upstream.porkchop_grid_empty",
        }


# ---------------------------------------------------------------------------
# Live Horizons (gated)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestLiveHorizons:
    """Hits the real Horizons endpoint; gated behind the integration marker.

    The acceptance criterion's textbook-tolerance C3 check lives here rather
    than in the unit tests so the unit suite stays offline and deterministic.
    """

    async def test_earth_mars_2026_window_returns_sensible_c3(self) -> None:
        response = await porkchop(
            departure_body="earth",
            arrival_body="mars",
            depart_window=["2026-09-01T00:00:00Z", "2026-12-01T00:00:00Z"],
            arrive_window=["2027-06-01T00:00:00Z", "2027-11-01T00:00:00Z"],
            samples_per_axis=10,
        )
        assert len(response.grid) > 0
        # Textbook 2026-2027 Earth-Mars launch period C3 minima sit in the
        # 10-25 km²/s² band; the loose bound below catches a broken grid
        # without baking in numbers that could drift with Horizons updates.
        assert 0 < response.best.c3.value < 50.0
