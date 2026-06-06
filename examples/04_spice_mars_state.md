# Example 4 — SPICE / Horizons Mars deep query (sequential)

Charter session (e): a *deep query* that cross-checks one body's
ephemeris two independent ways. The LLM furnishes SPICE kernels, asks
SPICE for Mars's heliocentric state from the planetary SPK, then runs a
Mars `porkchop` — whose body ephemerides come from JPL Horizons — and
confirms the two sources put Mars on the same orbit. It exercises the
`[spice]` tool surface (`spice_load_kernel` → `spice_state`) alongside
the Horizons-backed `porkchop`.

The reproducible script lives at
[`examples/run_example_04_spice_mars_state.py`](run_example_04_spice_mars_state.py).
The test environment ships no `spiceypy` (it lives behind the `[spice]`
extra) and no real planetary SPK, so the SPICE half runs against an
injected fake seeded with the *same* synthetic Mars geometry the
Horizons mock feeds `porkchop` — so the agreement the example
demonstrates is real within the fixture. Against a live install with the
`[spice]` extra and the real `de440s.bsp`, the identical tool calls
return CSPICE's own ephemeris.

## Client config

The SPICE tools register only when the `[spice]` extra is installed:

```bash
pip install "astrodynamics-mcp[spice]"
```

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

> **User:** I want to sanity-check Mars's position. Use SPICE to get
> Mars's heliocentric state on 2028-04-01, then run a Mars porkchop and
> confirm both agree on where Mars is.

The model first furnishes the two kernels SPICE needs for an SPK state
query — a leap-second kernel (to resolve the UTC epoch) and a planetary
ephemeris (the states themselves). Each `spice_load_kernel` call returns
the kernels it added to the process pool:

```jsonc
// tools/call → spice_load_kernel
{ "source": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls" }
```
```jsonc
// tools/call → spice_load_kernel
{ "source": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp" }
```

Then it queries the state — Mars relative to the Sun, in the
ecliptic-of-J2000 frame:

```jsonc
// tools/call → spice_state
{
  "target": "MARS",
  "observer": "SUN",
  "epochs": ["2028-04-01T00:00:00Z"],
  "frame": "ECLIPJ2000"
}
```

Response (geometric state — `aberration` defaults to `NONE`, so
`light_time` is null):

```jsonc
{
  "target": "MARS",
  "observer": "SUN",
  "frame": "ECLIPJ2000",
  "aberration": "NONE",
  "states": [
    {
      "epoch": "2028-04-01T00:00:00Z",
      "position": {"value": [-6712691.5, 227840269.9, 0.0], "unit": "km"},
      "velocity": {"value": [-24.119, -0.711, 0.0], "unit": "km/s"},
      "light_time": null
    }
  ]
}
```

The position's magnitude is **≈ 1.5237 AU** (227.9 million km) — Mars's
heliocentric distance. Next the model runs the porkchop over the 2028
launch season; the tool sources Earth and Mars from JPL Horizons:

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

```jsonc
{
  "best": {
    "depart_epoch": "2028-04-01T00:00:00Z",
    "arrive_epoch": "2029-05-18T19:12:00Z",
    "c3":       {"value": 17.249, "unit": "km^2/s^2"},
    "total_dv": {"value": 10.321, "unit": "km/s"},
    "tof_days": {"value":  413.8, "unit": "days"}
  },
  "top_cells": [ /* 4 more cells ranked by total_dv ascending */ ]
}
```

> **Model:** SPICE puts Mars at **1.5237 AU** from the Sun on
> 2028-04-01 (heliocentric position ≈ [-6.71e6, 2.278e8, 0] km in
> ECLIPJ2000). The Horizons-backed porkchop, which fetches Mars
> independently, places it on the same 1.52 AU orbit for the transfer
> search — the two ephemerides agree. The cheapest transfer in the
> window departs 2028-04-01 for a total Δv of ≈ 10.3 km/s.

(In production the LLM would not see the "synthetic fixture" caveat —
that's narration of how the *example* was built. Against a live `[spice]`
install the SPICE state is CSPICE's own `de440s` value and the porkchop
is live Horizons; the workflow is identical, only the numbers are the
real ones.)

## What this demonstrates

- **Kernel lifecycle.** `spice_load_kernel` furnishes kernels into a
  process-global pool that later queries read; an SPK state query needs
  both a leap-second kernel and a planetary SPK furnished first. URL
  loads come from the NAIF allowlist and are cached on disk, so a repeat
  load skips the network.
- **`spice_state` shape.** Position and velocity come back as
  `{value, unit}` quantities in the requested frame, each epoch
  self-describing; `light_time` is null for a geometric (`NONE`) query.
- **Cross-source agreement.** SPICE (SPK) and `porkchop` (Horizons) are
  independent ephemeris sources; reading the same body from both is the
  "deep query" sanity check — they should agree on where Mars is.

## Out of scope here

- A real-launch trajectory: the porkchop here illustrates the workflow,
  not the flight-planning answer (see [example 3](03_mars_launch_window_2028.md)).
- Body-fixed frames and constants: `spice_frame_transform` (e.g.
  `IAU_MARS`) and `spice_body_parameters` (radii / GM) are covered by
  the [tool reference](../docs/tool-reference.md) and the eval suite, not
  this walkthrough.
- Spacecraft-clock time (`SCLK`): `spice_time_convert` bridges ET / UTC /
  SCLK, but needs a mission SCLK kernel beyond this generic-kernel set.
