"""Reproducible smoke run for `examples/04_spice_mars_state.md`.

Example session (e): a SPICE / Horizons deep query. Furnishes a
leap-second kernel and a planetary SPK, asks SPICE for Mars's
heliocentric state, then runs a Mars porkchop (whose body ephemerides
come from JPL Horizons) and confirms the two sources put Mars on the
same orbit.

The test environment ships no `spiceypy` and no real planetary SPK, so
the SPICE half runs against an injected fake seeded with the *same*
synthetic Mars geometry the Horizons mock feeds porkchop — so the
agreement the example demonstrates is real within the fixture. The
deployment shape is identical at the MCP wire layer; against a live
install with the `[spice]` extra and the real de440s SPK, the same tool
calls return CSPICE's own ephemeris.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples._fixtures import (
    first_text_content,
    mars_heliocentric_state_eclipj2000,
    mcp_session,
    mock_horizons_earth_mars_2028,
    mock_spice_mars_state,
)

# Epoch inside the porkchop depart window so the SPICE query and the grid
# describe Mars over the same span.
_EPOCH = "2028-04-01T00:00:00Z"
_FRAME = "ECLIPJ2000"

_DEPART_WINDOW = ["2028-04-01T00:00:00Z", "2028-08-31T00:00:00Z"]
_ARRIVE_WINDOW = ["2028-12-01T00:00:00Z", "2029-06-30T00:00:00Z"]
_SAMPLES_PER_AXIS = 6

_AU_KM = 149597870.7
# Mars's semi-major axis (AU) is what both ephemerides must agree on for a
# point on the synthetic circular orbit; bracket loosely.
_MARS_SMA_AU_RANGE = (1.50, 1.55)
_BEST_DV_RANGE_KMS = (5.0, 15.0)


def _write_kernel(directory: Path, name: str) -> str:
    """Write a stand-in kernel file and return its path (the fake furnishes any path)."""
    path = directory / name
    path.write_bytes(b"fake kernel bytes")
    return str(path)


async def main() -> int:
    # The SPICE fake must be in sys.modules before mcp_session imports the
    # tools package (spice tools only register when spiceypy is importable),
    # so this context wraps the session.
    with (
        mock_spice_mars_state(_EPOCH),
        mock_horizons_earth_mars_2028(),
        tempfile.TemporaryDirectory() as kernel_dir,
    ):
        kdir = Path(kernel_dir)
        async with mcp_session() as session:
            for kernel in ("naif0012.tls", "de440s.bsp"):
                load_result = await session.call_tool(
                    "spice_load_kernel",
                    arguments={"source": _write_kernel(kdir, kernel)},
                )
                if load_result.isError:
                    print("spice_load_kernel error:", load_result.content, file=sys.stderr)
                    return 1

            state_result = await session.call_tool(
                "spice_state",
                arguments={
                    "target": "MARS",
                    "observer": "SUN",
                    "epochs": [_EPOCH],
                    "frame": _FRAME,
                },
            )
            if state_result.isError:
                print("spice_state error:", state_result.content, file=sys.stderr)
                return 1
            state_payload = first_text_content(state_result)

            porkchop_result = await session.call_tool(
                "porkchop",
                arguments={
                    "departure_body": "earth",
                    "arrival_body": "mars",
                    "depart_window": _DEPART_WINDOW,
                    "arrive_window": _ARRIVE_WINDOW,
                    "samples_per_axis": _SAMPLES_PER_AXIS,
                    "mu": "sun",
                },
            )

    if porkchop_result.isError:
        print("porkchop error:", porkchop_result.content, file=sys.stderr)
        return 1
    porkchop_payload = first_text_content(porkchop_result)

    # --- SPICE state checks ---------------------------------------------------
    position = state_payload["states"][0]["position"]["value"]
    if state_payload["states"][0]["position"]["unit"] != "km":
        print("FAIL: spice_state position not in km", file=sys.stderr)
        return 1
    spice_r_au = float(np.linalg.norm(position)) / _AU_KM
    if not _MARS_SMA_AU_RANGE[0] <= spice_r_au <= _MARS_SMA_AU_RANGE[1]:
        print(
            f"FAIL: SPICE Mars heliocentric distance {spice_r_au:.4f} AU outside "
            f"{_MARS_SMA_AU_RANGE}",
            file=sys.stderr,
        )
        return 1

    # --- Agreement check ------------------------------------------------------
    # The Horizons geometry the porkchop consumes places Mars on the same
    # circular orbit; the reference distance is its norm at this epoch.
    ref_position, _ = mars_heliocentric_state_eclipj2000(
        datetime.fromisoformat(_EPOCH.replace("Z", "+00:00"))
    )
    ref_r_au = float(np.linalg.norm(ref_position)) / _AU_KM
    if abs(spice_r_au - ref_r_au) > 1e-6:
        print(
            f"FAIL: SPICE distance {spice_r_au:.6f} AU disagrees with the Horizons "
            f"geometry {ref_r_au:.6f} AU",
            file=sys.stderr,
        )
        return 1

    # --- porkchop checks ------------------------------------------------------
    best = porkchop_payload.get("best")
    if best is None:
        print("FAIL: porkchop response carried no best cell", file=sys.stderr)
        return 1
    total_dv = float(best["total_dv"]["value"])
    if not _BEST_DV_RANGE_KMS[0] <= total_dv <= _BEST_DV_RANGE_KMS[1]:
        print(
            f"FAIL: best total_dv {total_dv:.2f} km/s outside {_BEST_DV_RANGE_KMS}",
            file=sys.stderr,
        )
        return 1

    # ASCII-only stdout: Windows' default cp1252 console can't encode Greek
    # (e.g. a delta), so the run script keeps prints plain (the .md transcript
    # carries the typeset symbols).
    print("SPICE / Horizons deep query - Mars ephemeris cross-check:")
    print(f"  spice_state MARS rel SUN ({_FRAME}) at {_EPOCH}:")
    print(f"    |r| = {spice_r_au:.4f} AU ({np.linalg.norm(position):.0f} km)")
    print(
        f"  porkchop EARTH->MARS best cell: depart {best['depart_epoch']}, "
        f"total dv {total_dv:.3f} km/s"
    )
    print(f"  agreement: SPICE and the Horizons-fed grid both place Mars at {spice_r_au:.4f} AU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
