"""Reproducible smoke run for `examples/05_ground_track_iss.md`.

Drives an `sgp4_propagate` -> `plot_ground_track` chain against the
in-process MCP server with an inline ISS TLE: propagate one orbit in the
Earth-fixed (ITRS) frame, then render the ground track as a PNG. Asserts
the propagation returns the requested epochs and that the plot tool comes
back with an ImageContent attachment plus a structured summary.

Requires the `[viz]` extra (matplotlib) — `tests/test_examples_viz.py`
skips this script when the extra is absent, and the `[viz] extra install`
CI job runs it with the extra present.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples._fixtures import (
    attachment_kinds,
    first_text_content,
    mcp_session,
    structured_content,
)

# Inline ISS TLE (NORAD 25544) — supplied directly so the run is fully
# deterministic and needs no CelesTrak / Space-Track fetch.
_TLE_LINE1 = "1 25544U 98067A   24001.50000000  .00010000  00000-0  18000-3 0  9990"
_TLE_LINE2 = "2 25544  51.6400  90.0000 0001000  90.0000 270.0000 15.50000000000000"

# One ISS revolution (~92 min) sampled every 3 minutes, in the Earth-fixed
# ITRS frame the ground track is drawn in.
_START = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_STEP_MINUTES = 3
_N_STEPS = 31  # 0, 3, ..., 90 minutes -> 31 epochs


def _epochs() -> list[str]:
    return [
        (_START + timedelta(minutes=_STEP_MINUTES * i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for i in range(_N_STEPS)
    ]


async def main() -> int:
    epochs = _epochs()
    async with mcp_session() as session:
        sgp4_result = await session.call_tool(
            "sgp4_propagate",
            arguments={
                "tle": {"line1": _TLE_LINE1, "line2": _TLE_LINE2},
                "epochs": epochs,
                "frame": "ITRS",
                "output": "full",
            },
        )
        if sgp4_result.isError:
            print("sgp4_propagate returned an error:", sgp4_result.content, file=sys.stderr)
            return 1
        sgp4_payload = first_text_content(sgp4_result)
        states = sgp4_payload.get("states", [])
        if len(states) != _N_STEPS:
            print(
                f"FAIL: expected {_N_STEPS} propagated states, got {len(states)}",
                file=sys.stderr,
            )
            return 1

        track_result = await session.call_tool(
            "plot_ground_track",
            arguments={"states": states},
        )

    if track_result.isError:
        print("plot_ground_track returned an error:", track_result.content, file=sys.stderr)
        return 1

    kinds = attachment_kinds(track_result)
    if "image" not in kinds:
        print(f"FAIL: expected an image attachment, got content kinds {kinds}", file=sys.stderr)
        return 1

    summary = structured_content(track_result)
    image = summary.get("image", {})
    if not (image.get("format") == "png" and image.get("width_px", 0) > 0):
        print(f"FAIL: plot summary carries no PNG dimensions: {image}", file=sys.stderr)
        return 1

    revs = summary["revolutions"]["value"]
    lat_min = summary["lat_min"]["value"]
    lat_max = summary["lat_max"]["value"]
    # A ~51.6 deg-inclination orbit should not stray far past its inclination
    # in sub-satellite latitude. A loose band catches a wildly wrong track.
    if not -90.0 <= lat_min <= lat_max <= 90.0:
        print(f"FAIL: latitude bounds out of range: [{lat_min}, {lat_max}]", file=sys.stderr)
        return 1

    print(
        f"sgp4_propagate -> plot_ground_track: {len(states)} ITRS states, "
        f"{revs:.2f} revs; lat [{lat_min:.1f}, {lat_max:.1f}] deg; "
        f"PNG {image['width_px']}x{image['height_px']} attached"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
