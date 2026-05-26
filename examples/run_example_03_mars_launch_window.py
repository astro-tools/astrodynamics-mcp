"""Reproducible smoke run for `examples/03_mars_launch_window_2028.md`.

Drives a planning question — "what's the best Mars launch window in
2028?" — through the in-process MCP server. JPL Horizons is mocked
with a synthetic-but-self-consistent Earth/Mars geometry phased so the
Hohmann-like opportunity lands inside the depart window the script
asks about. Asserts the porkchop returns a best cell whose total Δv is
in a wide [5, 15] km/s neighbourhood — the synthetic geometry doesn't
match the actual JPL 2028 launch window; the example illustrates the
workflow, not flight planning.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples._fixtures import (
    first_text_content,
    mcp_session,
    mock_horizons_earth_mars_2028,
)

_DEPART_WINDOW = ["2028-04-01T00:00:00Z", "2028-08-31T00:00:00Z"]
_ARRIVE_WINDOW = ["2028-12-01T00:00:00Z", "2029-06-30T00:00:00Z"]
_SAMPLES_PER_AXIS = 6

_BEST_DV_RANGE_KMS = (5.0, 15.0)
_BEST_C3_RANGE_KM2_S2 = (5.0, 200.0)


async def main() -> int:
    with mock_horizons_earth_mars_2028():
        async with mcp_session() as session:
            result = await session.call_tool(
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

    if result.isError:
        print("porkchop returned an error:", result.content, file=sys.stderr)
        return 1
    payload = first_text_content(result)

    best = payload.get("best")
    if best is None:
        print("FAIL: porkchop response carried no best cell", file=sys.stderr)
        return 1

    total_dv = float(best["total_dv"]["value"])
    c3 = float(best["c3"]["value"])

    if not _BEST_DV_RANGE_KMS[0] <= total_dv <= _BEST_DV_RANGE_KMS[1]:
        print(
            f"FAIL: best total_dv {total_dv:.2f} km/s outside {_BEST_DV_RANGE_KMS}",
            file=sys.stderr,
        )
        return 1
    if not _BEST_C3_RANGE_KM2_S2[0] <= c3 <= _BEST_C3_RANGE_KM2_S2[1]:
        print(
            f"FAIL: best C3 {c3:.2f} km²/s² outside {_BEST_C3_RANGE_KM2_S2}",
            file=sys.stderr,
        )
        return 1

    print("porkchop best cell (synthetic Earth/Mars geometry):")
    print(f"  depart epoch: {best['depart_epoch']}")
    print(f"  arrive epoch: {best['arrive_epoch']}")
    print(f"  total Δv:     {total_dv:.3f} km/s")
    print(f"  C3:           {c3:.3f} km²/s²")
    top_cells = payload.get("top_cells", [])
    if top_cells:
        print(f"  top {len(top_cells)} cells returned in `top_cells`")
    if "ascii_summary" in payload:
        print()
        print(payload["ascii_summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
