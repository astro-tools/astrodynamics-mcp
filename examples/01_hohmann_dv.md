# Example 1 — Hohmann Δv (single-tool sanity)

The canonical introductory question. Demonstrates the `lambert_solve`
tool path with `tof` set to the Hohmann time-of-flight — the standard
recipe for "what's the Δv for transfer X → Y" when the chat is wired
into an MCP client.

The reproducible script lives at
[`examples/run_example_01_hohmann.py`](run_example_01_hohmann.py). It
exits 0 when `lambert_solve` produces a two-impulse Δv within 0.01 km/s
of the textbook ≈ 3.912 km/s.

## Client config

The user has wired the server into Claude Code:

```json
{
  "mcpServers": {
    "astrodynamics-mcp": {
      "command": "astrodynamics-mcp",
      "args": ["stdio"]
    }
  }
}
```

## Conversation

> **User:** Compute the Hohmann Δv from a 250 km circular LEO to GEO.

The model reaches for `lambert_solve`. The Hohmann transfer is the
half-ellipse joining perigee at the LEO radius to apogee at the GEO
radius; `tof` is half the transfer ellipse's period, and the two-impulse
Δv falls out when both the departure-circular and arrival-circular
velocities are supplied.

It computes the geometry:

- `r_LEO = 6378.137 + 250 = 6628.137 km` (Earth equatorial radius plus altitude).
- `r_GEO = 42164.0 km`.
- `a_transfer = (r_LEO + r_GEO) / 2 = 24396.069 km`.
- `tof = π · √(a_transfer³ / μ_Earth) ≈ 19036 s` (~5.3 h).
- `v_LEO = √(μ_Earth / r_LEO) ≈ 7.755 km/s`.
- `v_GEO = √(μ_Earth / r_GEO) ≈ 3.075 km/s`.

Then issues the tool call. The transfer geometry is offset by 0.001°
from a strict 180° so Lambert's algorithm doesn't hit the
collinear-r1/r2 degeneracy — see the script's `_TRANSFER_ANGLE_DEG`
constant for the reasoning.

```jsonc
// tools/call → lambert_solve
{
  "r1": [6628.137, 0.0, 0.0],
  "r2": [-42163.999999358, 0.0735926..., 0.0],
  "tof": 19035.51,
  "mu": "earth",
  "direction": "prograde",
  "depart_velocity": [0.0, 7.75483, 0.0],
  "arrive_velocity": [-0.000005368..., -3.07466..., 0.0]
}
```

Response (trimmed to the fields the LLM reads back):

```jsonc
{
  "v1":    {"value": [0.00769, 10.19492, 0.0],   "unit": "km/s"},
  "v2":    {"value": [-0.00261, -1.60263, 0.0],  "unit": "km/s"},
  "transfer_elements": {
    "a":   {"value": 24396.07, "unit": "km"},
    "e":   {"value": 0.72831,  "unit": "1"},
    "i":   {"value": 0.0,      "unit": "deg"},
    "raan":{"value": 0.0,      "unit": "deg"},
    "argp":{"value": 0.0,      "unit": "deg"},
    "nu":  {"value": 0.0,      "unit": "deg"}
  },
  "dv":    {"value": 3.91212, "unit": "km/s"},
  "all_solutions": [ /* one entry for the M=0 primary solution */ ]
}
```

The transfer arc's semi-major axis matches the textbook
`(r_LEO + r_GEO) / 2 = 24396 km`; the eccentricity `e ≈ 0.728` matches
`(r_GEO − r_LEO) / (r_GEO + r_LEO)`. The two-impulse Δv is `3.912 km/s`,
which decomposes as:

- `Δv₁ = |v1 − v_LEO_circular| ≈ 2.439 km/s` (LEO-circular → transfer perigee).
- `Δv₂ = |v_GEO_circular − v2| ≈ 1.473 km/s` (transfer apogee → GEO-circular).

> **Model:** A Hohmann transfer from a 250 km circular LEO to GEO needs
> ≈ **3.91 km/s** of total Δv, split into a 2.44 km/s prograde burn at
> LEO (raising apogee to GEO) and a 1.47 km/s prograde burn at apogee
> (circularising at GEO). The transfer takes ≈ 5.3 h.

## What this demonstrates

- A single `lambert_solve` call is enough for the canonical Hohmann
  question — no orbit propagation, no ephemerides, no ground-stations.
- `depart_velocity` + `arrive_velocity` (both supplied together)
  populate the response's `dv` field with the two-impulse total.
- The transfer's semi-major axis and eccentricity in the response let
  the LLM cross-check its own arithmetic against the textbook formulas.
- Lambert's collinear-r1/r2 degeneracy is sidestepped by a sub-millidegree
  offset — the conversation answer is unchanged.
