"""Shared per-tool sample-call infrastructure for the cross-tool validation tests.

Each entry in :data:`SAMPLE_CALLS` is a :class:`SampleCall` carrying the fixed
input the validation suite calls the tool with plus a context manager that
installs the deterministic mocks (CelesTrak, JPL Horizons) and cache isolation
the tool needs to produce a reproducible output. The roundtrip, reference-
output, and transport-equivalence modules all parametrize over this table.

The underscore prefix keeps pytest from collecting this module as a test file.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import httpx
import numpy as np
from pydantic import BaseModel

from astrodynamics_mcp.schemas.base import Frame, NamedStation, StateVector, TleLines
from astrodynamics_mcp.tools.access import (
    AccessWindowsResponse,
    access_windows,
)
from astrodynamics_mcp.tools.bplane import (
    BplaneTargetResponse,
    bplane_target,
)
from astrodynamics_mcp.tools.frames import (
    FrameTransformResponse,
    frame_transform,
)
from astrodynamics_mcp.tools.lambert import (
    LambertSolveResponse,
    lambert_solve,
)
from astrodynamics_mcp.tools.porkchop import (
    _JD_UNIX_EPOCH,
    _MU_SUN,
    PorkchopResponse,
    porkchop,
)
from astrodynamics_mcp.tools.propagation import (
    Sgp4PropagateResponse,
    sgp4_propagate,
)
from astrodynamics_mcp.tools.time import TimeConvertResponse, time_convert
from astrodynamics_mcp.tools.tle import TleLookupResponse, tle_lookup
from astrodynamics_mcp.units import QuantityVector

# ---------------------------------------------------------------------------
# Shared deterministic fixtures
# ---------------------------------------------------------------------------

ISS_LINE1 = "1 25544U 98067A   24001.50000000  .00010000  00000-0  18000-3 0  9995"
ISS_LINE2 = "2 25544  51.6400  90.0000 0001000  90.0000 270.0000 15.50000000    07"

_SAMPLE_OMM_ISS: dict[str, Any] = {
    "OBJECT_NAME": "ISS (ZARYA)",
    "OBJECT_ID": "1998-067A",
    "EPOCH": "2024-01-01T12:00:00.000000",
    "MEAN_MOTION": 15.5,
    "ECCENTRICITY": 0.0001,
    "INCLINATION": 51.64,
    "RA_OF_ASC_NODE": 90.0,
    "ARG_OF_PERICENTER": 90.0,
    "MEAN_ANOMALY": 270.0,
    "EPHEMERIS_TYPE": 0,
    "CLASSIFICATION_TYPE": "U",
    "NORAD_CAT_ID": 25544,
    "ELEMENT_SET_NO": 999,
    "REV_AT_EPOCH": 0,
    "BSTAR": 0.00018,
    "MEAN_MOTION_DOT": 0.0001,
    "MEAN_MOTION_DDOT": 0.0,
}


@contextmanager
def _temp_cache() -> Iterator[None]:
    """Point the on-disk cache at a temporary directory and reset the singleton."""
    import astrodynamics_mcp.cache as cache_module

    with tempfile.TemporaryDirectory() as cache_dir:
        previous_default = cache_module._default_cache
        previous_env = os.environ.get("ASTRODYNAMICS_MCP_CACHE_DIR")
        os.environ["ASTRODYNAMICS_MCP_CACHE_DIR"] = cache_dir
        cache_module._default_cache = None
        try:
            yield
        finally:
            cache_module._default_cache = previous_default
            if previous_env is None:
                os.environ.pop("ASTRODYNAMICS_MCP_CACHE_DIR", None)
            else:
                os.environ["ASTRODYNAMICS_MCP_CACHE_DIR"] = previous_env


@contextmanager
def _mock_celestrak_iss() -> Iterator[None]:
    """Patch the CelesTrak adapter's `httpx.AsyncClient` to return one ISS OMM record."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[_SAMPLE_OMM_ISS])

    original_client = httpx.AsyncClient

    def factory(*_args: Any, **_kwargs: Any) -> httpx.AsyncClient:
        return original_client(transport=httpx.MockTransport(handler))

    with (
        _temp_cache(),
        patch(
            "astrodynamics_mcp.data.celestrak.httpx.AsyncClient",
            side_effect=factory,
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# Porkchop synthetic Horizons fixture (mirrors the per-tool test's geometry)
# ---------------------------------------------------------------------------

_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_AU_KM = 149597870.7
_EARTH_SMA = _AU_KM
_MARS_SMA = 1.523679 * _AU_KM
_PORKCHOP_FIXTURE_START = datetime(2026, 9, 1, tzinfo=timezone.utc)
_PORKCHOP_FIXTURE_DAYS = 460


def _datetime_to_jd(dt: datetime) -> float:
    return _JD_UNIX_EPOCH + (dt - _UNIX_EPOCH).total_seconds() / 86400.0


def _circular_orbit_table(
    *,
    start: datetime,
    days: int,
    semi_major_axis_km: float,
    initial_mean_anomaly_rad: float,
) -> tuple[list[datetime], np.ndarray, np.ndarray]:
    period_s = 2 * np.pi * np.sqrt(semi_major_axis_km**3 / _MU_SUN)
    epochs: list[datetime] = []
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    v_circ = float(np.sqrt(_MU_SUN / semi_major_axis_km))
    for day in range(days + 1):
        t = start + timedelta(days=day)
        m = initial_mean_anomaly_rad + 2 * np.pi * (day * 86400.0) / period_s
        cos_m = float(np.cos(m))
        sin_m = float(np.sin(m))
        epochs.append(t)
        positions.append(np.array([semi_major_axis_km * cos_m, semi_major_axis_km * sin_m, 0.0]))
        velocities.append(np.array([-v_circ * sin_m, v_circ * cos_m, 0.0]))
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


def _porkchop_payloads() -> dict[str, dict[str, Any]]:
    earth_epochs, earth_pos, earth_vel = _circular_orbit_table(
        start=_PORKCHOP_FIXTURE_START,
        days=_PORKCHOP_FIXTURE_DAYS,
        semi_major_axis_km=_EARTH_SMA,
        initial_mean_anomaly_rad=0.0,
    )
    mars_epochs, mars_pos, mars_vel = _circular_orbit_table(
        start=_PORKCHOP_FIXTURE_START,
        days=_PORKCHOP_FIXTURE_DAYS,
        semi_major_axis_km=_MARS_SMA,
        initial_mean_anomaly_rad=float(np.radians(45.0)),
    )
    return {
        "399": {"result": _format_horizons_vectors(earth_epochs, earth_pos, earth_vel)},
        "499": {"result": _format_horizons_vectors(mars_epochs, mars_pos, mars_vel)},
    }


@contextmanager
def _mock_horizons_earth_mars() -> Iterator[None]:
    """Patch the Horizons adapter with a deterministic Earth/Mars synthetic geometry."""
    payloads = _porkchop_payloads()

    def handler(request: httpx.Request) -> httpx.Response:
        target = request.url.params["COMMAND"].strip("'")
        payload = payloads.get(target)
        if payload is None:
            return httpx.Response(404, text=f"unknown target {target!r} in mock")
        return httpx.Response(200, json=payload)

    original_client = httpx.AsyncClient

    def factory(*_args: Any, **_kwargs: Any) -> httpx.AsyncClient:
        return original_client(transport=httpx.MockTransport(handler))

    with (
        _temp_cache(),
        patch(
            "astrodynamics_mcp.data.horizons.httpx.AsyncClient",
            side_effect=factory,
        ),
    ):
        yield


@contextmanager
def _noop_context() -> Iterator[None]:
    yield


# ---------------------------------------------------------------------------
# Sample-call table — one entry per registered v0.1 tool
# ---------------------------------------------------------------------------


_PORKCHOP_DEPART_WINDOW: list[str] = [
    "2026-11-01T00:00:00Z",
    "2026-12-31T00:00:00Z",
]
_PORKCHOP_ARRIVE_WINDOW: list[str] = [
    "2027-06-01T00:00:00Z",
    "2027-11-01T00:00:00Z",
]


@dataclass(frozen=True)
class SampleCall:
    """One tool-name + sample-input + output-model entry."""

    tool_name: str
    output_model: type[BaseModel]
    setup: Callable[[], AbstractContextManager[None]]
    invoke: Callable[[], Any]
    """Async callable taking no args; returns the tool's :class:`BaseModel`."""

    mcp_arguments: dict[str, Any]
    """The same fixed input as a JSON-RPC-style argument dict for the MCP wire surface."""


async def _call_tle_lookup() -> TleLookupResponse:
    return await tle_lookup(query="25544")


async def _call_sgp4_propagate() -> Sgp4PropagateResponse:
    return await sgp4_propagate(
        tle=TleLines(line1=ISS_LINE1, line2=ISS_LINE2),
        epochs=["2024-01-01T12:00:00Z", "2024-01-01T12:10:00Z"],
        frame=Frame.TEME,
    )


async def _call_lambert_solve() -> LambertSolveResponse:
    return await lambert_solve(
        r1=[5000.0, 10000.0, 2100.0],
        r2=[-14600.0, 2500.0, 7000.0],
        tof=3600.0,
        mu="earth",
    )


async def _call_access_windows() -> AccessWindowsResponse:
    return await access_windows(
        observer=NamedStation(name="madrid"),
        target_tle=TleLines(line1=ISS_LINE1, line2=ISS_LINE2),
        start="2024-01-01T00:00:00Z",
        end="2024-01-02T00:00:00Z",
        min_elevation_deg=10.0,
    )


async def _call_time_convert() -> TimeConvertResponse:
    from astrodynamics_mcp.schemas.base import TimeScale

    return await time_convert(
        value="2026-05-23T12:00:00",
        from_scale=TimeScale.UTC,
        to_scale=TimeScale.TAI,
    )


async def _call_frame_transform() -> FrameTransformResponse:
    state = StateVector(
        r=QuantityVector(value=[7000.0, 0.0, 0.0], unit="km"),
        v=QuantityVector(value=[0.0, 7.5, 0.0], unit="km/s"),
        frame=Frame.TEME,
        epoch="2024-06-15T08:00:00Z",
    )
    return await frame_transform(state=state, to_frame=Frame.ICRF)


async def _call_porkchop() -> PorkchopResponse:
    return await porkchop(
        departure_body="earth",
        arrival_body="mars",
        depart_window=_PORKCHOP_DEPART_WINDOW,
        arrive_window=_PORKCHOP_ARRIVE_WINDOW,
        samples_per_axis=4,
    )


async def _call_bplane_target() -> BplaneTargetResponse:
    state = StateVector(
        r=QuantityVector(value=[5000.0, 0.0, 0.0], unit="km"),
        v=QuantityVector(value=[0.0, 5.069223017386393, 0.0], unit="km/s"),
        frame=Frame.ICRF,
        epoch="2026-11-30T00:00:00Z",
    )
    return await bplane_target(
        state=state,
        target_body="mars",
        target_epoch="2026-12-01T00:00:00Z",
    )


SAMPLE_CALLS: list[SampleCall] = [
    SampleCall(
        tool_name="tle_lookup",
        output_model=TleLookupResponse,
        setup=_mock_celestrak_iss,
        invoke=_call_tle_lookup,
        mcp_arguments={"query": "25544"},
    ),
    SampleCall(
        tool_name="sgp4_propagate",
        output_model=Sgp4PropagateResponse,
        setup=_noop_context,
        invoke=_call_sgp4_propagate,
        mcp_arguments={
            "tle": {"line1": ISS_LINE1, "line2": ISS_LINE2},
            "epochs": ["2024-01-01T12:00:00Z", "2024-01-01T12:10:00Z"],
            "frame": "TEME",
        },
    ),
    SampleCall(
        tool_name="lambert_solve",
        output_model=LambertSolveResponse,
        setup=_noop_context,
        invoke=_call_lambert_solve,
        mcp_arguments={
            "r1": [5000.0, 10000.0, 2100.0],
            "r2": [-14600.0, 2500.0, 7000.0],
            "tof": 3600.0,
            "mu": "earth",
        },
    ),
    SampleCall(
        tool_name="access_windows",
        output_model=AccessWindowsResponse,
        setup=_noop_context,
        invoke=_call_access_windows,
        mcp_arguments={
            "observer": {"name": "madrid"},
            "target_tle": {"line1": ISS_LINE1, "line2": ISS_LINE2},
            "start": "2024-01-01T00:00:00Z",
            "end": "2024-01-02T00:00:00Z",
            "min_elevation_deg": 10.0,
        },
    ),
    SampleCall(
        tool_name="time_convert",
        output_model=TimeConvertResponse,
        setup=_noop_context,
        invoke=_call_time_convert,
        mcp_arguments={
            "value": "2026-05-23T12:00:00",
            "from_scale": "UTC",
            "to_scale": "TAI",
        },
    ),
    SampleCall(
        tool_name="frame_transform",
        output_model=FrameTransformResponse,
        setup=_noop_context,
        invoke=_call_frame_transform,
        mcp_arguments={
            "state": {
                "r": {"value": [7000.0, 0.0, 0.0], "unit": "km"},
                "v": {"value": [0.0, 7.5, 0.0], "unit": "km/s"},
                "frame": "TEME",
                "epoch": "2024-06-15T08:00:00Z",
            },
            "to_frame": "ICRF",
        },
    ),
    SampleCall(
        tool_name="porkchop",
        output_model=PorkchopResponse,
        setup=_mock_horizons_earth_mars,
        invoke=_call_porkchop,
        mcp_arguments={
            "departure_body": "earth",
            "arrival_body": "mars",
            "depart_window": _PORKCHOP_DEPART_WINDOW,
            "arrive_window": _PORKCHOP_ARRIVE_WINDOW,
            "samples_per_axis": 4,
        },
    ),
    SampleCall(
        tool_name="bplane_target",
        output_model=BplaneTargetResponse,
        setup=_noop_context,
        invoke=_call_bplane_target,
        mcp_arguments={
            "state": {
                "r": {"value": [5000.0, 0.0, 0.0], "unit": "km"},
                "v": {"value": [0.0, 5.069223017386393, 0.0], "unit": "km/s"},
                "frame": "ICRF",
                "epoch": "2026-11-30T00:00:00Z",
            },
            "target_body": "mars",
            "target_epoch": "2026-12-01T00:00:00Z",
        },
    ),
]
