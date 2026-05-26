# Example 3 — Mars launch window 2028 (planning)

A planning question: the LLM has to set up a two-parameter sweep, run
it, and read the answer off the best cell. Demonstrates the `porkchop`
tool with the default `output="summary"` shape — the response carries
the best cell, the top-5 cells, and an ASCII contour, omitting the full
grid (see [output shaping](../docs/output-shaping.md) for why).

The reproducible script lives at
[`examples/run_example_03_mars_launch_window.py`](run_example_03_mars_launch_window.py).
It mocks JPL Horizons with a synthetic-but-self-consistent Earth/Mars
geometry whose phasing puts a Hohmann-like opportunity inside the
depart window the example asks about. The synthetic geometry is *not*
the real 2028 launch window — the example illustrates the workflow,
not the flight-planning answer. Real-2028 Δv values run a few km/s
lower; for that, point the tool at live Horizons (the production
default) and don't apply the mock.

## Client config

```json
{
  "mcpServers": {
    "astrodynamics": {
      "command": "astrodynamics-mcp",
      "args": ["stdio"]
    }
  }
}
```

## Conversation

> **User:** What's the best Mars launch window in 2028?

The model interprets this as "scan a (depart × arrive) grid over the
2028 launch season and report the lowest-Δv cell". It calls `porkchop`
with sensible windows — depart April through August 2028 (the standard
US-window months around the 2028 opposition), arrive late 2028 through
mid 2029 — and the default summary shape.

```jsonc
// tools/call → porkchop
{
  "departure_body": "earth",
  "arrival_body": "mars",
  "depart_window": ["2028-04-01T00:00:00Z", "2028-08-31T00:00:00Z"],
  "arrive_window": ["2028-12-01T00:00:00Z", "2029-06-30T00:00:00Z"],
  "samples_per_axis": 6,
  "mu": "sun"
}
```

Response (default `output="summary"` — best cell, top-5 cells, ASCII
contour, no full grid):

```jsonc
{
  "best": {
    "depart_epoch": "2028-04-01T00:00:00Z",
    "arrive_epoch": "2029-05-18T19:12:00Z",
    "c3":              {"value": 17.249, "unit": "km^2/s^2"},
    "departure_dv":    {"value":  4.153, "unit": "km/s"},
    "arrival_v_inf":   {"value":  6.168, "unit": "km/s"},
    "total_dv":        {"value": 10.321, "unit": "km/s"},
    "tof_days":        {"value":  413.8, "unit": "days"}
  },
  "top_cells": [ /* 4 more cells ranked by total_dv ascending */ ],
  "ascii_summary":
"..:-*X\n...:+#\n...:-*\n....:+\n....:-\n.....:"
}
```

The ASCII summary is a compact 6×6 contour of the C3 grid. `X` marks
the best cell, `#` the second-best, and the gradient (`. : - + * # X`)
runs low-to-high C3 — the visual eye-check for whether the best cell
sits in a clean minimum or on a window edge.

> **Model:** With the synthetic test geometry, the porkchop tool finds
> its lowest-Δv cell at **depart 2028-04-01, arrive 2029-05-18**:
> total two-impulse Δv ≈ **10.3 km/s**, departure C3 ≈ 17.2 km²/s²,
> time-of-flight ≈ 414 days.
>
> ⚠ Note: this run is against a synthetic Earth/Mars geometry (the
> example fixture). The real 2028 launch window's Δv runs a few km/s
> lower than this — to get flight-planning numbers, ask me again
> against live JPL Horizons, not the example fixture.

(In production the LLM would not see the "synthetic test geometry"
caveat — that's narration of how the *example* was built. The point of
the example is that the workflow is identical against either live or
fixture data; only the numbers differ.)

### Follow-up — `lambert_solve` at the best cell

A natural next step the user often asks: "show me the actual transfer
velocities for the best cell". The LLM converts the best-cell epochs
to Earth and Mars heliocentric positions (these come from the same
Horizons data the porkchop call already fetched), then calls
`lambert_solve(mu="sun", r1=earth_pos, r2=mars_pos, tof=best.tof_days*86400)`.
That returns `v1` and `v2` at native (non-summary) precision — the
exact velocities the spacecraft needs at injection and at Mars arrival.
The chain is not shown in detail here; it's the same `lambert_solve`
mechanics demonstrated by [example 1](01_hohmann_dv.md).

## What this demonstrates

- The `porkchop` tool is sized to the planning question — one tool
  call, summary output by default, the LLM gets the best cell and an
  ASCII shape check without ingesting a 36-cell raw grid.
- The summary / full discriminated union is opt-in. The LLM can
  request `output="full"` if it needs every cell for a follow-up
  analysis; the default keeps small-context models inside their token
  budget (see [output shaping](../docs/output-shaping.md)).
- Body resolution accepts common names (`"earth"`, `"mars"`); the tool
  routes to the right Horizons body ID internally.
- The follow-up `lambert_solve` pulls the best-cell trajectory at full
  numerical precision — porkchop is the search, Lambert is the
  refinement.

## Out of scope here

- A real-launch trajectory: needs live JPL Horizons, the real C3 vs.
  declination of departure asymptote contour, and a launch-vehicle
  capability check. astrodynamics-mcp computes the orbital mechanics;
  matching that to a launcher's published C3 curve is the user's job.
- Multi-rev or low-thrust transfers — out of scope for v0.1; would
  need different tools.
- Arrival operations (capture, MOI ΔV) — same story; orbiter design
  is a separate analysis.
