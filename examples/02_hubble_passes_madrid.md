# Example 2 — Hubble passes from Madrid (sequential)

A two-step chain: fetch a current TLE, then compute ground-station
access. Demonstrates the `tle_lookup` → `access_windows` flow against
the canonical "what passes can I see from a named station" question.

The reproducible script lives at
[`examples/run_example_02_hubble_passes.py`](run_example_02_hubble_passes.py).
It mocks CelesTrak so the same HST OMM record drives every run, then
walks the chain and asserts the returned passes sit in physically
plausible ranges (peak elevation 10°–90°, duration 60 s–30 min, at
least one pass returned for a Madrid-on-a-Hubble-pass-day window).

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

> **User:** I have TLEs for Hubble — plot ground-station passes from
> Madrid for the next seven days.

The model parses the prompt as a two-step problem: it needs a current
TLE first, then it needs the access-window solver. Two tool calls.

### Call 1 — fetch the TLE

```jsonc
// tools/call → tle_lookup
{
  "query": "HUBBLE"
}
```

Response (one OMM record, parsed and re-emitted alongside the raw
two-line strings):

```jsonc
{
  "results": [
    {
      "name": "HST",
      "norad_id": "20580",
      "tle_line1": "1 20580U 90037B   …",
      "tle_line2": "2 20580  28.4690 …",
      "omm": { "EPOCH": "2026-05-23T00:00:00.000000", "MEAN_MOTION": 15.09, … },
      "fetched_at": "2026-05-23T…",
      "stale": false
    }
  ]
}
```

The `stale=false` flag tells the LLM CelesTrak was reached on this
call. If the upstream had been unreachable and only an older cached
value were available, `stale=true` would surface here and the LLM
should mention "using a TLE from N hours ago" in its reply.

### Call 2 — compute access windows

```jsonc
// tools/call → access_windows
{
  "observer": {"name": "madrid"},
  "target_tle": {
    "line1": "1 20580U 90037B   …",
    "line2": "2 20580  28.4690 …"
  },
  "start": "2026-05-23T00:00:00Z",
  "end":   "2026-05-30T00:00:00Z",
  "min_elevation_deg": 10
}
```

The named-station registry resolves `madrid` to its WGS-84 geodetic
coordinates inside the tool — no need for the LLM to know lat/lon.
The eight v0.1 named stations cover Madrid, Goldstone, Canberra,
Svalbard, Wallops, Esrange, GSFC, and JPL; for anything else the user
passes explicit `{lat, lon, alt}` quantities (see
[the `access_windows` tool reference](../docs/tool-reference.md)).

Response (first 5 of 17 passes, schema trimmed):

```jsonc
{
  "windows": [
    {
      "aos": "2026-05-23T14:34:35Z",
      "los": "2026-05-23T14:38:00Z",
      "peak_elevation_time": "2026-05-23T14:36:17Z",
      "peak_elevation": {"value": 12.3, "unit": "deg"},
      "range_at_aos":  {"value": 1734.1, "unit": "km"},
      "range_at_peak": {"value": 1149.5, "unit": "km"},
      "range_at_los":  {"value": 1675.3, "unit": "km"},
      "duration":      {"value": 205.0,  "unit": "s"}
    },
    {
      "aos": "2026-05-23T16:14:18Z",
      "peak_elevation": {"value": 15.3, "unit": "deg"},
      "duration":       {"value": 293.0, "unit": "s"}
    },
    {
      "aos": "2026-05-23T17:56:33Z",
      "peak_elevation": {"value": 10.3, "unit": "deg"},
      "duration":       {"value":  72.0, "unit": "s"}
    },
    {
      "aos": "2026-05-24T14:23:32Z",
      "peak_elevation": {"value": 13.4, "unit": "deg"},
      "duration":       {"value": 245.0, "unit": "s"}
    },
    {
      "aos": "2026-05-24T16:03:40Z",
      "peak_elevation": {"value": 14.9, "unit": "deg"},
      "duration":       {"value": 284.0, "unit": "s"}
    }
    /* … 12 more passes through 2026-05-30 */
  ]
}
```

> **Model:** Hubble has **17 passes above 10° elevation** from Madrid
> over the next seven days. The brightest pass of the window peaks at
> 15.3° on May 23 at 16:14 UTC; the longest is roughly 5 minutes.
> Typical pass duration is 200–300 s. None of the passes clear 20°,
> consistent with Hubble's 28.47° inclination and Madrid's 40.4° N
> latitude — Madrid never sees Hubble overhead. Want me to filter to
> just the daytime / nighttime passes for visual observation?

## What this demonstrates

- The `tle_lookup` → `access_windows` chain is the canonical two-step
  flow for "what can I see from here". The LLM picks both tools without
  prompting.
- The named-station registry (`{"name": "madrid"}`) is the lowest-cost
  observer input — a single dict-with-name vs. three quantity-wrapped
  lat/lon/alt fields. Tool descriptions favour it so the LLM uses it
  whenever the prompt names a station the registry covers.
- `stale=true / false` on the TLE response gives the LLM the signal it
  needs to qualify the answer when CelesTrak is unreachable.
- Range and duration come back with explicit units, so the LLM doesn't
  guess at km vs. m. Same for peak elevation in deg vs. rad.

## Out of scope here

- Lighting (daylight / terminator) and Sun-angle filtering — would
  need a separate predicate over each pass's geometry; the v0.1 surface
  doesn't ship it.
- Visual brightness magnitudes — a different physics layer entirely;
  belongs in a satellite-photometry tool, not the geometric access
  tool.
