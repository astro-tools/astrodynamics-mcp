"""Reproducible smoke run for `examples/02_hubble_passes_madrid.md`.

Drives a `tle_lookup` → `access_windows` chain against the in-process
MCP server with CelesTrak mocked to return a fixed HST OMM record.
Asserts at least one pass is returned and that peak elevations and
durations sit in physically plausible ranges.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples._fixtures import (
    first_text_content,
    mcp_session,
    mock_celestrak_hubble,
)

# Fixed 7-day window. Choice of date is not load-bearing for the
# numerical results because the Hubble fixture's epoch lands inside the
# window — but pinning it makes the transcript reproducible.
_WINDOW_START = "2026-05-23T00:00:00Z"
_WINDOW_END = "2026-05-30T00:00:00Z"
_MIN_ELEVATION_DEG = 10.0

_MIN_PASSES = 1
_MAX_PASSES = 50
_PEAK_ELEVATION_RANGE_DEG = (10.0, 90.0)
_DURATION_RANGE_S = (60.0, 1800.0)


async def main() -> int:
    with mock_celestrak_hubble():
        async with mcp_session() as session:
            tle_result = await session.call_tool(
                "tle_lookup",
                arguments={"query": "HUBBLE"},
            )
            if tle_result.isError:
                print("tle_lookup returned an error:", tle_result.content, file=sys.stderr)
                return 1
            tle_payload = first_text_content(tle_result)

            if not tle_payload.get("results"):
                print("tle_lookup returned no results", file=sys.stderr)
                return 1
            tle_record = tle_payload["results"][0]
            target_tle = {
                "line1": tle_record["tle_line1"],
                "line2": tle_record["tle_line2"],
            }

            access_result = await session.call_tool(
                "access_windows",
                arguments={
                    "observer": {"name": "madrid"},
                    "target_tle": target_tle,
                    "start": _WINDOW_START,
                    "end": _WINDOW_END,
                    "min_elevation_deg": _MIN_ELEVATION_DEG,
                },
            )

    if access_result.isError:
        print("access_windows returned an error:", access_result.content, file=sys.stderr)
        return 1
    access_payload = first_text_content(access_result)
    windows = access_payload.get("windows", [])

    if not _MIN_PASSES <= len(windows) <= _MAX_PASSES:
        print(
            f"FAIL: expected {_MIN_PASSES}-{_MAX_PASSES} passes, got {len(windows)}",
            file=sys.stderr,
        )
        return 1

    for i, window in enumerate(windows):
        peak = float(window["peak_elevation"]["value"])
        duration = float(window["duration"]["value"])
        if not _PEAK_ELEVATION_RANGE_DEG[0] <= peak <= _PEAK_ELEVATION_RANGE_DEG[1]:
            print(
                f"FAIL: pass {i} peak elevation {peak}° outside {_PEAK_ELEVATION_RANGE_DEG}",
                file=sys.stderr,
            )
            return 1
        if not _DURATION_RANGE_S[0] <= duration <= _DURATION_RANGE_S[1]:
            print(
                f"FAIL: pass {i} duration {duration}s outside {_DURATION_RANGE_S}",
                file=sys.stderr,
            )
            return 1

    print(f"tle_lookup → access_windows: {len(windows)} HST passes from Madrid")
    for i, window in enumerate(windows[:5]):
        print(
            f"  pass {i}: AOS {window['aos']}  peak {window['peak_elevation']['value']:.1f}°  "
            f"duration {window['duration']['value']:.0f}s"
        )
    if len(windows) > 5:
        print(f"  ... ({len(windows) - 5} more)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
