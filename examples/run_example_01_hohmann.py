"""Reproducible smoke run for `examples/01_hohmann_dv.md`.

Drives an in-process MCP server with a single `lambert_solve` call
matching the canonical Hohmann LEO→GEO transfer described in the
accompanying transcript. Asserts the returned two-impulse Δv is within
±0.01 km/s of the textbook value (≈ 3.912 km/s for a 250 km circular
LEO → GEO transfer).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

# Make examples/_fixtures.py importable when this script is invoked as a
# bare path from anywhere in the workspace.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples._fixtures import first_text_content, mcp_session

# Earth gravitational parameter (km^3/s^2). Same value the `lambert_solve`
# `mu="earth"` lookup uses internally.
_MU_EARTH = 398600.4418
_R_EARTH = 6378.137  # km, equatorial — `mu="earth"` is geocentric.
_LEO_ALT_KM = 250.0
_R_GEO = 42164.0  # km, geostationary radius.

EXPECTED_DV_KMS = 3.912
DV_TOLERANCE_KMS = 0.01


_TRANSFER_ANGLE_DEG = 179.999


def _hohmann_geometry() -> dict[str, list[float] | float]:
    """Return the r1, r2, tof, depart-velocity, arrive-velocity inputs.

    r1 sits at perigee on the +x axis; r2 sits very nearly on the -x axis
    at the GEO radius. The transfer angle is 179.999° rather than exact
    180°: Izzo's algorithm hits a branch degeneracy at strictly collinear
    r1, r2 and returns NaN. A sub-millidegree offset avoids that branch
    without changing the answer to four significant figures.

    Circular tangent velocities at each end go in as
    `depart_velocity` / `arrive_velocity` so the tool's two-impulse Δv
    field is populated.
    """
    r1_mag = _R_EARTH + _LEO_ALT_KM
    a_transfer = (r1_mag + _R_GEO) / 2.0
    tof = float(np.pi * np.sqrt(a_transfer**3 / _MU_EARTH))

    v_leo = float(np.sqrt(_MU_EARTH / r1_mag))
    v_geo = float(np.sqrt(_MU_EARTH / _R_GEO))

    angle = np.radians(_TRANSFER_ANGLE_DEG)
    r2 = [_R_GEO * float(np.cos(angle)), _R_GEO * float(np.sin(angle)), 0.0]
    # Prograde circular-tangent velocities (perpendicular to r in the xy plane).
    arrive_velocity = [
        -v_geo * float(np.sin(angle)),
        v_geo * float(np.cos(angle)),
        0.0,
    ]

    return {
        "r1": [r1_mag, 0.0, 0.0],
        "r2": r2,
        "tof": tof,
        "depart_velocity": [0.0, v_leo, 0.0],
        "arrive_velocity": arrive_velocity,
    }


async def main() -> int:
    geometry = _hohmann_geometry()

    async with mcp_session() as session:
        result = await session.call_tool(
            "lambert_solve",
            arguments={
                **geometry,
                "mu": "earth",
                "direction": "prograde",
            },
        )

    if result.isError:
        print("lambert_solve returned an error:", result.content, file=sys.stderr)
        return 1

    payload = first_text_content(result)
    dv = payload["dv"]
    assert dv is not None, "expected dv to be populated when depart/arrive velocities are supplied"
    dv_value = float(dv["value"])
    dv_unit = dv["unit"]
    assert dv_unit == "km/s", f"expected km/s, got {dv_unit!r}"

    deviation = abs(dv_value - EXPECTED_DV_KMS)
    if deviation > DV_TOLERANCE_KMS:
        print(
            f"FAIL: dv={dv_value:.5f} km/s deviates from {EXPECTED_DV_KMS} km/s by "
            f"{deviation:.5f} km/s (tolerance {DV_TOLERANCE_KMS} km/s)",
            file=sys.stderr,
        )
        return 1

    transfer_a = payload["transfer_elements"]["a"]["value"]
    transfer_e = payload["transfer_elements"]["e"]["value"]
    print(f"lambert_solve dv = {dv_value:.5f} km/s (target {EXPECTED_DV_KMS} ± {DV_TOLERANCE_KMS})")
    print(f"transfer semi-major axis a = {transfer_a:.2f} km  eccentricity e = {transfer_e:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
