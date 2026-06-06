"""Reproducible smoke run for `examples/06_czml_export.md`.

Drives an `sgp4_propagate` -> `czml_trajectory` chain against the
in-process MCP server with an inline ISS TLE: propagate one orbit in the
TEME frame, then export it as a CZML document for a Cesium 3D view.
Asserts the export comes back with an EmbeddedResource attachment (the
CZML) plus a structured summary carrying non-zero packet / object counts.

Requires the `[viz]` extra (gmat-czml) — `tests/test_examples_viz.py`
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

# One ISS revolution (~92 min) sampled every 6 minutes, in TEME — the frame
# sgp4 emits by default and one gmat-czml renders directly.
_START = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_STEP_MINUTES = 6
_N_STEPS = 16  # 0, 6, ..., 90 minutes -> 16 epochs


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
                "frame": "TEME",
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

        czml_result = await session.call_tool(
            "czml_trajectory",
            arguments={"trajectory": states, "style": "default"},
        )

    if czml_result.isError:
        print("czml_trajectory returned an error:", czml_result.content, file=sys.stderr)
        return 1

    kinds = attachment_kinds(czml_result)
    if "resource" not in kinds:
        print(
            f"FAIL: expected a CZML resource attachment, got content kinds {kinds}",
            file=sys.stderr,
        )
        return 1

    summary = structured_content(czml_result)
    resource = summary.get("resource", {})
    if resource.get("format") != "czml" or resource.get("packet_count", 0) < 2:
        print(f"FAIL: CZML summary carries no document cardinalities: {resource}", file=sys.stderr)
        return 1
    if resource.get("object_count", 0) < 1:
        print(f"FAIL: CZML document rendered no objects: {resource}", file=sys.stderr)
        return 1

    print(
        f"sgp4_propagate -> czml_trajectory: {len(states)} TEME states "
        f"({summary['frame']}, style {summary['style']}); "
        f"CZML {resource['packet_count']} packets / {resource['object_count']} object(s) "
        f"attached at {resource['uri']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
