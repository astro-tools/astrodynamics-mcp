"""Shared deterministic fixtures and in-process MCP-client driver for the example sessions.

Each ``run_example_NN.py`` script imports from here so the mocks
(`CelesTrak`, `JPL Horizons`) and the MCP client/server pairing live in
one place. The same context managers back the smoke-test path under
`tests/test_examples.py`.

The MCP `ClientSession` is created in-process via
``mcp.shared.memory.create_connected_server_and_client_session`` against
the module-level `astrodynamics_mcp.server.mcp` singleton. This is
deliberately not a subprocess: the data-source mocks must reach the
tool functions, and ``httpx`` monkey-patches don't cross subprocess
boundaries. The user-facing transcripts still show the canonical stdio
client config — the deployment shape is identical at the MCP wire
layer.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import httpx
import numpy as np
from mcp import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session

# ---------------------------------------------------------------------------
# Cache isolation
# ---------------------------------------------------------------------------


@contextmanager
def temp_cache() -> Iterator[None]:
    """Point the on-disk cache at a temporary directory for the lifetime of the block.

    Each example needs a pristine cache so a stale entry from a prior run
    doesn't paper over a mock that didn't fire.
    """
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


# ---------------------------------------------------------------------------
# CelesTrak — fixed Hubble OMM
# ---------------------------------------------------------------------------


# Synthetic-but-CCSDS-conformant OMM for HST (NORAD 20580). The mean
# elements roughly match Hubble's actual orbit (~540 km altitude, 28.5°
# inclination); the BSTAR is a sane low-drag value. The fixture is
# deliberately *not* time-locked to a single calendar day so the example
# transcripts don't rot when a future re-run picks a different `start`.
_HUBBLE_OMM: dict[str, Any] = {
    "OBJECT_NAME": "HST",
    "OBJECT_ID": "1990-037B",
    "EPOCH": "2026-05-23T00:00:00.000000",
    "MEAN_MOTION": 15.09299,
    "ECCENTRICITY": 0.0002829,
    "INCLINATION": 28.4690,
    "RA_OF_ASC_NODE": 32.1234,
    "ARG_OF_PERICENTER": 80.0,
    "MEAN_ANOMALY": 280.0,
    "EPHEMERIS_TYPE": 0,
    "CLASSIFICATION_TYPE": "U",
    "NORAD_CAT_ID": 20580,
    "ELEMENT_SET_NO": 999,
    "REV_AT_EPOCH": 0,
    "BSTAR": 0.00012,
    "MEAN_MOTION_DOT": 0.0,
    "MEAN_MOTION_DDOT": 0.0,
}


@contextmanager
def mock_celestrak_hubble() -> Iterator[None]:
    """Patch CelesTrak's HTTP client to return one HST OMM record."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[_HUBBLE_OMM])

    original_client = httpx.AsyncClient

    def factory(*_args: Any, **_kwargs: Any) -> httpx.AsyncClient:
        return original_client(transport=httpx.MockTransport(handler))

    with (
        temp_cache(),
        patch(
            "astrodynamics_mcp.data.celestrak.httpx.AsyncClient",
            side_effect=factory,
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# Horizons — synthetic Earth / Mars 2028 geometry
# ---------------------------------------------------------------------------


_AU_KM = 149597870.7
_MU_SUN = 1.32712440018e11
_EARTH_SMA_KM = _AU_KM
_MARS_SMA_KM = 1.523679 * _AU_KM
_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_JD_UNIX_EPOCH = 2440587.5

# Coverage window for the Mars-2028 porkchop example. Earth and Mars are
# placed on coplanar circular orbits whose phasing at the start of the
# table puts the synthetic-Hohmann opportunity inside the depart / arrive
# windows the example calls porkchop with.
_FIXTURE_START = datetime(2028, 1, 1, tzinfo=timezone.utc)
_FIXTURE_DAYS = 700


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
        "Ephemeris / API_USER Sun-centred VECTORS table — synthetic example fixture",
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


def _build_horizons_payloads() -> dict[str, dict[str, Any]]:
    earth_epochs, earth_pos, earth_vel = _circular_orbit_table(
        start=_FIXTURE_START,
        days=_FIXTURE_DAYS,
        semi_major_axis_km=_EARTH_SMA_KM,
        initial_mean_anomaly_rad=0.0,
    )
    # Mars leads Earth by ~44° so the Hohmann opportunity lands inside the
    # depart window the example calls porkchop with. Tuned by hand against
    # the example's window pair.
    mars_epochs, mars_pos, mars_vel = _circular_orbit_table(
        start=_FIXTURE_START,
        days=_FIXTURE_DAYS,
        semi_major_axis_km=_MARS_SMA_KM,
        initial_mean_anomaly_rad=float(np.radians(44.0)),
    )
    return {
        "399": {"result": _format_horizons_vectors(earth_epochs, earth_pos, earth_vel)},
        "499": {"result": _format_horizons_vectors(mars_epochs, mars_pos, mars_vel)},
    }


@contextmanager
def mock_horizons_earth_mars_2028() -> Iterator[None]:
    """Patch Horizons' HTTP client with a deterministic Earth / Mars geometry covering 2028."""
    payloads = _build_horizons_payloads()

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
        temp_cache(),
        patch(
            "astrodynamics_mcp.data.horizons.httpx.AsyncClient",
            side_effect=factory,
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# SPICE — synthetic Mars heliocentric state matching the Horizons geometry
# ---------------------------------------------------------------------------


def mars_heliocentric_state_eclipj2000(epoch: datetime) -> tuple[list[float], list[float]]:
    """Mars's heliocentric state at *epoch* on the synthetic example orbit.

    Uses the *same* circular ecliptic-plane orbit the Horizons mock feeds
    porkchop (semi-major axis 1.523679 AU, phased 44° at 2028-01-01), so the
    SPICE SPK state the example queries and the porkchop's Mars ephemeris
    describe the same body. Returns ``([x, y, z] km, [vx, vy, vz] km/s)`` —
    expressed in the ecliptic-of-J2000 plane, which is what the example's
    ``spice_state(frame='ECLIPJ2000')`` query reports.
    """
    period_s = 2 * np.pi * np.sqrt(_MARS_SMA_KM**3 / _MU_SUN)
    v_circ = float(np.sqrt(_MU_SUN / _MARS_SMA_KM))
    dt_s = (epoch - _FIXTURE_START).total_seconds()
    m = float(np.radians(44.0)) + 2 * np.pi * dt_s / period_s
    cos_m, sin_m = float(np.cos(m)), float(np.sin(m))
    position = [_MARS_SMA_KM * cos_m, _MARS_SMA_KM * sin_m, 0.0]
    velocity = [-v_circ * sin_m, v_circ * cos_m, 0.0]
    return position, velocity


class _ExampleSpiceyError(Exception):
    """Stand-in for ``spiceypy.SpiceyError`` — what CSPICE raises on failure."""


class _ExampleSpice:
    """Minimal ``spiceypy`` stand-in for example session (e): furnish + SPK state.

    The test environment ships no ``spiceypy`` (it lives behind the ``[spice]``
    extra) and no real planetary SPK, so the example injects this fake via
    ``sys.modules`` exactly as the SPICE unit tests do. It implements only the
    surface ``spice_load_kernel`` and ``spice_state`` reach — ``furnsh`` /
    ``ktotal`` / ``kdata`` over a real in-process pool, ``str2et`` / ``spkezr``
    for the state, and the three error-handling setters — and returns the
    planned Mars heliocentric state. As with the unit-test golden, this
    validates the tool's *packaging* of a known reference state, not CSPICE's
    own ephemeris math.
    """

    SpiceyError = _ExampleSpiceyError

    def __init__(self, mars_state: tuple[float, ...]) -> None:
        self._pool: list[dict[str, Any]] = []
        self._next_handle = 1
        self._mars_state = list(mars_state)

    # Error-handling setters — recorded behaviour is irrelevant to the example.
    def erract(self, op: str, action: str | None = None) -> str:
        return action or "RETURN"

    def errdev(self, op: str, device: str | None = None) -> str:
        return device or "NULL"

    def errprt(self, op: str, value: str | None = None) -> str:
        return value or "NONE"

    # Kernel pool — an SPK gets a non-zero handle; text kernels (LSK) get 0.
    def furnsh(self, path: str) -> None:
        ktype = "SPK" if path.endswith(".bsp") else "TEXT"
        handle = 0
        if ktype == "SPK":
            handle = self._next_handle
            self._next_handle += 1
        if not any(entry["name"] == path for entry in self._pool):
            self._pool.append({"name": path, "type": ktype, "source": "", "handle": handle})

    def ktotal(self, kind: str) -> int:
        return len(self._pool)

    def kdata(
        self, which: int, kind: str, *args: Any, **kwargs: Any
    ) -> tuple[str, str, str, int, bool]:
        if which < 0 or which >= len(self._pool):
            return ("", "", "", 0, False)
        entry = self._pool[which]
        return (entry["name"], entry["type"], entry["source"], int(entry["handle"]), True)

    def _has_type(self, ktype: str) -> bool:
        return any(entry["type"] == ktype for entry in self._pool)

    def str2et(self, time: str) -> float:
        if not self._has_type("TEXT"):
            raise _ExampleSpiceyError("SPICE(NOLEAPSECONDS): no leapseconds kernel loaded")
        return 0.0

    def spkezr(
        self, targ: str, et: float, ref: str, abcorr: str, obs: str
    ) -> tuple[list[float], float]:
        if not self._has_type("SPK"):
            raise _ExampleSpiceyError("SPICE(SPKINSUFFDATA): no ephemeris data loaded")
        return (list(self._mars_state), 0.0)


@contextmanager
def mock_spice_mars_state(epoch_iso: str) -> Iterator[None]:
    """Inject the SPICE fake for the lifetime of the block, seeded for *epoch_iso*.

    The fake must be in ``sys.modules`` *before* ``astrodynamics_mcp.tools.spice``
    is first imported — its registration is gated on ``spiceypy`` being
    importable — so enter this context before opening :func:`mcp_session`.
    """
    epoch = datetime.fromisoformat(epoch_iso.replace("Z", "+00:00"))
    position, velocity = mars_heliocentric_state_eclipj2000(epoch)
    fake = _ExampleSpice(tuple(position) + tuple(velocity))
    previous = sys.modules.get("spiceypy")
    sys.modules["spiceypy"] = fake  # type: ignore[assignment]
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop("spiceypy", None)
        else:
            sys.modules["spiceypy"] = previous


# ---------------------------------------------------------------------------
# In-process MCP client session
# ---------------------------------------------------------------------------


@asynccontextmanager
async def mcp_session() -> AsyncIterator[ClientSession]:
    """Open an in-process MCP client connected to the live `astrodynamics_mcp` server.

    Wraps :func:`mcp.shared.memory.create_connected_server_and_client_session`
    around the module-level :data:`astrodynamics_mcp.server.mcp` singleton.
    Tool registration happens via the side effect of importing
    :mod:`astrodynamics_mcp.tools`.
    """
    # Importing the tools package triggers @register_tool side effects on
    # the shared `mcp` singleton.
    import astrodynamics_mcp.tools  # noqa: F401
    from astrodynamics_mcp.server import mcp

    async with create_connected_server_and_client_session(mcp) as session:
        await session.initialize()
        yield session


def first_text_content(call_tool_result: Any) -> dict[str, Any]:
    """Return the first text block of an MCP `tools/call` result as parsed JSON.

    FastMCP encodes a pydantic-model return value as a single
    ``TextContent`` whose ``.text`` is the JSON dump. The example
    scripts only ever care about the structured response, so we collapse
    that layer at one spot.
    """
    if not getattr(call_tool_result, "content", None):
        raise AssertionError("tool call returned no content")
    block = call_tool_result.content[0]
    text = getattr(block, "text", None)
    if text is None:
        raise AssertionError(f"first content block has no .text (got {type(block).__name__})")
    parsed: Any = json.loads(text)
    if not isinstance(parsed, dict):
        raise AssertionError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed
