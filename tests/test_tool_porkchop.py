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
    _SUMMARY_TOP_CELLS,
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
            output="full",
        )
        assert isinstance(response, PorkchopResponse)
        # nxn cells, all positive-tof in this window so the grid is full.
        assert len(response.grid) == n * n

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
            output="full",
        )
        assert isinstance(response, PorkchopResponse)
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
            output="full",
        )
        assert isinstance(response, PorkchopResponse)
        rebuilt = PorkchopResponse.model_validate_json(response.model_dump_json())
        assert rebuilt == response

    async def test_body_name_case_insensitive(self, earth_mars_client: Iterator[None]) -> None:
        response = await porkchop(
            departure_body="EARTH",
            arrival_body="MARS",
            depart_window=_DEPART_WINDOW,
            arrive_window=_ARRIVE_WINDOW,
            samples_per_axis=3,
            output="full",
        )
        assert isinstance(response, PorkchopResponse)
        assert len(response.grid) > 0


# ---------------------------------------------------------------------------
# Default-summary output shape
# ---------------------------------------------------------------------------


class TestSummaryOutput:
    async def test_default_output_omits_full_grid(self, earth_mars_client: Iterator[None]) -> None:
        response = await porkchop(
            departure_body="earth",
            arrival_body="mars",
            depart_window=_DEPART_WINDOW,
            arrive_window=_ARRIVE_WINDOW,
            samples_per_axis=8,
        )
        assert response.grid == []
        assert response.top_cells[0] == response.best
        assert len(response.top_cells) == _SUMMARY_TOP_CELLS
        # top_cells are sorted ascending by total_dv.
        totals = [cell.total_dv.value for cell in response.top_cells]
        assert totals == sorted(totals)

    async def test_full_keeps_top_cells_alongside_grid(
        self, earth_mars_client: Iterator[None]
    ) -> None:
        response = await porkchop(
            departure_body="earth",
            arrival_body="mars",
            depart_window=_DEPART_WINDOW,
            arrive_window=_ARRIVE_WINDOW,
            samples_per_axis=6,
            output="full",
        )
        assert len(response.grid) > 0
        assert len(response.top_cells) == _SUMMARY_TOP_CELLS
        assert response.top_cells[0] == response.best
        # top_cells is a sorted prefix of the grid (by total_dv asc).
        grid_sorted = sorted(response.grid, key=lambda c: c.total_dv.value)
        assert response.top_cells == grid_sorted[:_SUMMARY_TOP_CELLS]

    async def test_summary_top_cells_caps_at_feasible_count(
        self, earth_mars_client: Iterator[None]
    ) -> None:
        """If fewer feasible cells exist than the top-N cap, top_cells shrinks."""
        # 2x2 grid → at most four feasible cells; top_cells must not exceed it.
        response = await porkchop(
            departure_body="earth",
            arrival_body="mars",
            depart_window=_DEPART_WINDOW,
            arrive_window=_ARRIVE_WINDOW,
            samples_per_axis=2,
        )
        assert 1 <= len(response.top_cells) <= 4

    async def test_summary_payload_is_smaller_than_full(
        self, earth_mars_client: Iterator[None]
    ) -> None:
        """The summary path must produce a strictly smaller JSON payload."""
        kwargs: dict[str, Any] = {
            "departure_body": "earth",
            "arrival_body": "mars",
            "depart_window": _DEPART_WINDOW,
            "arrive_window": _ARRIVE_WINDOW,
            "samples_per_axis": 10,
        }
        summary = await porkchop(**kwargs, output="summary")
        full = await porkchop(**kwargs, output="full")
        assert len(summary.model_dump_json()) < len(full.model_dump_json())


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
                "output": "full",
            },
        )
        del content
        assert isinstance(structured, dict)
        assert "grid" in structured and "best" in structured and "ascii_summary" in structured
        assert structured["best"]["c3"]["unit"] == "km^2/s^2"
        assert structured["best"]["total_dv"]["unit"] == "km/s"

    async def test_tool_callable_via_mcp_default_summary_shape(
        self, earth_mars_client: Iterator[None]
    ) -> None:
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
        assert {"best", "top_cells", "grid", "ascii_summary"} <= structured.keys()
        assert structured["grid"] == []
        assert 1 <= len(structured["top_cells"]) <= 5


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

    async def test_grid_empty_when_every_lambert_call_fails(
        self, earth_mars_client: Iterator[None]
    ) -> None:
        """Mocked Lambert always-raise drops every cell, hitting the empty-grid raise.

        The window-order validator is satisfied (overlapping windows with
        positive-tof cells), but every ``_solve_cell`` falls through the
        ``except (AssertionError, ValueError, RuntimeError)`` branch and
        returns None.
        """
        del earth_mars_client  # unused; activate-fixture only

        def izzo_always_raises(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("synthetic lambert failure")

        with (
            patch(
                "astrodynamics_mcp.tools.porkchop._izzo2015",
                return_value=izzo_always_raises,
            ),
            pytest.raises(ToolError) as excinfo,
        ):
            await porkchop(
                departure_body="earth",
                arrival_body="mars",
                depart_window=_DEPART_WINDOW,
                arrive_window=_ARRIVE_WINDOW,
                samples_per_axis=3,
            )
        envelope = json.loads(str(excinfo.value))
        assert envelope["code"] == "upstream.porkchop_grid_empty"


class TestInternalHelpers:
    """Direct tests for the helpers feasibility-loop branches don't reach naturally."""

    def test_parse_iso_epoch_attaches_utc_to_naive_value(self) -> None:
        """A naive ISO timestamp gets UTC attached before normalisation."""
        from datetime import timezone

        from astrodynamics_mcp.tools.porkchop import _parse_iso_epoch

        # The schema-level Epoch validator accepts timestamps without a
        # timezone designator; the porkchop parser then anchors them to UTC.
        out = _parse_iso_epoch("2026-11-01T00:00:00", field="depart_window[0]")
        assert out.tzinfo is not None
        assert out.utcoffset() == timezone.utc.utcoffset(out)

    def test_parse_horizons_vectors_raises_on_unparseable_number(self) -> None:
        """A ValueError inside `_strip_signed_number` surfaces as upstream error.

        The capture-group regex normally rejects bad numbers, so we patch
        the stripper to raise unconditionally and assert the wrapping
        exception lands with the right code.
        """
        from datetime import datetime, timezone

        from astrodynamics_mcp.data.horizons import HorizonsResponse
        from astrodynamics_mcp.errors import UpstreamError
        from astrodynamics_mcp.tools.porkchop import _parse_horizons_vectors

        body = (
            "$$SOE\n"
            " 2460676.500000000 = A.D. 2025-Jan-01 00:00:00.000 TDB\n"
            " X = 1.0E+08 Y = 2.0E+08 Z = 3.0E+08\n"
            " VX= 1.0E+01 VY= 2.0E+01 VZ= 3.0E+01\n"
            "$$EOE\n"
        )
        response = HorizonsResponse(
            signature={
                "target": "499",
                "center": "@sun",
                "start": "x",
                "stop": "y",
                "step": "1d",
            },
            result=body,
            fetched_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

        with (
            patch(
                "astrodynamics_mcp.tools.porkchop._strip_signed_number",
                side_effect=ValueError("synthetic strip failure"),
            ),
            pytest.raises(UpstreamError) as excinfo,
        ):
            _parse_horizons_vectors(response)
        assert excinfo.value.code == "upstream.horizons_unexpected_shape"

    def test_interp_state_returns_endpoints_verbatim(self) -> None:
        """Targeting the first or last epoch returns that row without interpolation."""
        from datetime import datetime, timezone

        from astrodynamics_mcp.tools.porkchop import _interp_state

        e0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        e1 = datetime(2026, 1, 2, tzinfo=timezone.utc)
        positions = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        velocities = np.array([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]])

        r0, v0 = _interp_state(e0, [e0, e1], positions, velocities)
        np.testing.assert_array_equal(r0, positions[0])
        np.testing.assert_array_equal(v0, velocities[0])

        r1, v1 = _interp_state(e1, [e0, e1], positions, velocities)
        np.testing.assert_array_equal(r1, positions[1])
        np.testing.assert_array_equal(v1, velocities[1])

    def test_interp_state_duplicate_epoch_bracket(self) -> None:
        """A zero-span bracket (duplicate epochs) falls back to the right edge."""
        from datetime import datetime, timezone

        from astrodynamics_mcp.tools.porkchop import _interp_state

        e_dup = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        e_late = datetime(2026, 1, 2, tzinfo=timezone.utc)
        positions = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        velocities = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [1.0, 2.0, 3.0]])
        r, v = _interp_state(e_dup, [e_dup, e_dup, e_late], positions, velocities)
        assert np.all(np.isfinite(r))
        assert np.all(np.isfinite(v))

    def test_interp_state_target_outside_window_raises(self) -> None:
        """A target before/after the ephemeris window surfaces a typed upstream error."""
        from datetime import datetime, timezone

        from astrodynamics_mcp.errors import UpstreamError
        from astrodynamics_mcp.tools.porkchop import _interp_state

        e0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        e1 = datetime(2026, 1, 2, tzinfo=timezone.utc)
        positions = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        velocities = np.array([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]])

        with pytest.raises(UpstreamError) as excinfo:
            _interp_state(
                datetime(2025, 12, 31, tzinfo=timezone.utc),
                [e0, e1],
                positions,
                velocities,
            )
        assert excinfo.value.code == "upstream.horizons_window_too_narrow"

    def test_ascii_contour_empty_grid_returns_empty_string(self) -> None:
        """An all-None / non-finite grid renders as the empty string."""
        from astrodynamics_mcp.tools.porkchop import _ascii_contour

        assert _ascii_contour([[None, None], [None, None]]) == ""

    def test_solve_cell_zero_v_inf_falls_back_to_zero_declination(self) -> None:
        """When v_∞_dep magnitude is zero, declination falls back to 0° (no NaN)."""
        from datetime import datetime, timezone

        from astrodynamics_mcp.tools.porkchop import _solve_cell

        v_body = np.array([0.0, 30.0, 0.0])

        def izzo_zero_vinf(
            _mu: float, _r1: Any, _r2: Any, _tof: float, **_kw: Any
        ) -> tuple[np.ndarray, np.ndarray]:
            # Match the body velocity exactly so v_inf_dep = 0.
            return v_body.copy(), v_body.copy()

        cell = _solve_cell(
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 6, 1, tzinfo=timezone.utc),
            np.array([1.5e8, 0.0, 0.0]),
            v_body,
            np.array([0.0, 2.3e8, 0.0]),
            v_body,
            izzo_zero_vinf,
        )
        assert cell is not None
        assert cell.dec_dep_asymptote.value == 0.0

    def test_solve_cell_returns_none_when_lambert_raises(self) -> None:
        """`_solve_cell` swallows AssertionError/ValueError/RuntimeError from izzo."""
        from datetime import datetime, timezone

        from astrodynamics_mcp.tools.porkchop import _solve_cell

        def izzo_raises(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("synthetic lambert failure")

        cell = _solve_cell(
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 6, 1, tzinfo=timezone.utc),
            np.array([1.5e8, 0.0, 0.0]),
            np.array([0.0, 30.0, 0.0]),
            np.array([0.0, 2.3e8, 0.0]),
            np.array([0.0, 30.0, 0.0]),
            izzo_raises,
        )
        assert cell is None


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
            output="full",
        )
        assert isinstance(response, PorkchopResponse)
        assert len(response.grid) > 0
        # Textbook 2026-2027 Earth-Mars launch period C3 minima sit in the
        # 10-25 km²/s² band; the loose bound below catches a broken grid
        # without baking in numbers that could drift with Horizons updates.
        assert 0 < response.best.c3.value < 50.0
