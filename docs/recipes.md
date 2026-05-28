# Recipes

Short worked examples covering the canonical workflows. Each recipe is
written as the natural-language exchange you'd have with an MCP-capable
chat client; the tool calls the LLM makes are shown inline.

Several recipes have a fully worked, reproducible counterpart under
[`examples/`](https://github.com/astro-tools/astrodynamics-mcp/tree/main/examples)
— see the **Worked example** link at the end of each section.

## Fetch a TLE and propagate it

The most common single-satellite question — current TLE for a named
object, propagate it to a few epochs.

> **You:** Fetch the latest TLE for the ISS and propagate it to
> 2026-05-23T12:00:00Z, 14:00, and 16:00.

The model calls `tle_lookup` with `query="25544"` (the ISS NORAD ID — or
the name "ISS (ZARYA)"). The CelesTrak adapter returns the current TLE
plus the parsed OMM JSON; the response is cached for six hours under the
XDG cache directory so a repeat call within that window is free.

It then calls `sgp4_propagate` with the fetched `{line1, line2}` and the
three epochs. The default `frame="TEME"` (SGP4's native output frame)
returns three `StateVector`s, each with explicit `km` and `km/s` units,
the frame string, and the epoch they're valid at. Pass `frame="ICRF"` or
`frame="GCRS"` to receive the same trajectory in an inertial Earth-centred
frame; the tool transforms via astropy with the correct per-epoch
`obstime`. See [`tle_lookup`](tool-reference.md) and
[`sgp4_propagate`](tool-reference.md) for the full schema.

## A TLE for a fresh launch (Space-Track)

> **You:** Get me the latest TLE for NORAD 62000 — it's a recent launch and I
> don't see it on CelesTrak.

CelesTrak's general-perturbations feed trails the catalogue for very recent
launches. The model retries `tle_lookup` against Space-Track, which carries
the deeper, more current GP records:

```jsonc
// tools/call → tle_lookup
{ "query": "62000", "source": "space-track" }
```

Space-Track requires a per-user account. If no credential is configured, the
call returns *before any network request* with the typed error
`credential_required.spacetrack`, listing the fields it needs. The LLM relays
that you must set `ASTRODYNAMICS_MCP_SPACETRACK_USERNAME` and
`ASTRODYNAMICS_MCP_SPACETRACK_PASSWORD` (stdio) or pass the credential in the
`initialize` `_meta` block (HTTP) — see [Credentials](credentials.md). Once
the credential is present, the same call returns the TLE plus parsed OMM JSON,
cached on disk for six hours so a repeat lookup within that window costs no
request against Space-Track's per-account rate limit.

A numeric query becomes a `NORAD_CAT_ID` lookup and anything else an
`OBJECT_NAME` substring search; Space-Track has no CelesTrak-style group
keywords. See [Data sources](data-sources.md#space-track-tle_lookup-with-credentials)
for the auth, caching, and rate-limit story.

## Ground-station passes for a named observer

> **You:** Show me Hubble passes above 10° from Madrid over the next
> seven days.

The model calls `tle_lookup` with `query="HUBBLE"` (or NORAD ID `20580`).
It then calls `access_windows` with:

- `observer={"name": "madrid"}` — the named-station registry covers
  Madrid, Goldstone, Canberra, Svalbard, Wallops, Esrange, GSFC, and JPL.
  Pass `observer={"lat": {...}, "lon": {...}, "alt": {...}}` for anything
  else.
- The fetched TLE.
- `start` / `end` epochs (UTC ISO 8601 with a mandatory time component —
  bare dates are rejected).
- `min_elevation_deg=10` (degrees, not radians).

The tool returns a list of `AccessWindow`s — each with AOS, LOS,
peak-elevation epoch and angle, range at AOS / peak / LOS, and pass
duration. Partial passes that straddle a window boundary are omitted; only
complete (AOS, peak, LOS) triples are emitted.

Filter on range at peak elevation by passing `min_range_km` /
`max_range_km`. See [`access_windows`](tool-reference.md) for the full
input list.

**Worked example:**
[Hubble passes from Madrid](https://github.com/astro-tools/astrodynamics-mcp/blob/main/examples/02_hubble_passes_madrid.md)
— full transcript plus a reproducible script that drives the
`tle_lookup` → `access_windows` chain against an in-process MCP
server.

## A simple porkchop

> **You:** Run a porkchop for Earth → Mars departure 2026-09 through
> 2026-12, arrival window 2027-04 through 2027-10.

The model calls `porkchop` with:

- `departure_body="earth"`, `arrival_body="mars"`.
- `depart_window=["2026-09-01T00:00:00Z", "2026-12-31T00:00:00Z"]`.
- `arrive_window=["2027-04-01T00:00:00Z", "2027-10-31T00:00:00Z"]`.
- Default `samples_per_axis` and `mu="sun"`.

The tool fetches both bodies' ephemerides from JPL Horizons (cached for a
week by default — planetary ephemerides drift on geological scales), runs
the Lambert solves across the (depart × arrive) grid, and by default
returns the `summary` shape: the best cell (lowest total Δv), the top
five cells, and an ASCII contour of the C3 grid. Pass `output="full"` to
receive every feasible cell. See
[Output shaping](output-shaping.md) for why the default trims the
grid.

Open the best cell from the summary and feed it straight back into a
Lambert solve to recover the depart / arrive velocity vectors at native
precision.

**Worked example:**
[Mars launch window 2028](https://github.com/astro-tools/astrodynamics-mcp/blob/main/examples/03_mars_launch_window_2028.md)
— planning-tier transcript driving `porkchop` over a 2028 window with
a synthetic Earth/Mars geometry; the run script asserts the best cell
sits inside a plausible Δv envelope.

## Hohmann Δv between circular orbits

> **You:** Compute the Hohmann Δv from a 250 km circular LEO to GEO.

The Hohmann transfer is the half-ellipse joining perigee at the
departure radius to apogee at the arrival radius. The model calls
`lambert_solve` with `tof` set to half the transfer ellipse's period
and supplies both circular-tangent velocities so the response's `dv`
field carries the two-impulse total directly.

```jsonc
// tools/call → lambert_solve
{
  "r1": [6628.137, 0.0, 0.0],
  "r2": [-42163.999..., 0.073..., 0.0],     // 179.999° offset dodges Lambert's strictly-collinear branch
  "tof": 19035.51,
  "mu": "earth",
  "depart_velocity": [0.0, 7.755, 0.0],
  "arrive_velocity": [-0.000005, -3.075, 0.0]
}
```

Response carries `dv ≈ 3.91 km/s` plus the transfer arc's semi-major
axis (`a = 24396 km`) and eccentricity (`e ≈ 0.73`) — the textbook
Hohmann values.

**Worked example:**
[Hohmann LEO → GEO](https://github.com/astro-tools/astrodynamics-mcp/blob/main/examples/01_hohmann_dv.md)
— full transcript with the geometric setup and the
collinear-degeneracy workaround explained, plus a reproducible script
asserting the Δv lands within ± 0.01 km/s of the textbook ≈ 3.912 km/s.

## Time-scale conversions for log analysis

> **You:** Convert these telemetry timestamps from GPS to UTC and TDB:
> 2026-05-23T12:00:00Z, 2026-05-23T12:00:01Z.

The model calls `time_convert` twice (or in a single multi-target call if
the prompt permits), once for each target scale. The tool wraps
`astropy.time` so leap-second handling, UT1-UTC corrections (sourced from
the cached IERS Bulletin A), and the TDB ↔ TT relativistic offset are all
correct.

The same tool handles ISO / JD / MJD / seconds-since-J2000 / Unix-time
representations on either side via the `in_format` / `out_format`
arguments. See [`time_convert`](tool-reference.md) for the full scale and
format lists; the time-scale Python type is
[`astrodynamics_mcp.schemas.base.TimeScale`](api.md#schemas).

## Run a GMAT mission from a skeleton

Requires the `[gmat]` extra — see [GMAT integration](gmat-integration.md) for
install and the full tool surface.

> **You:** Run a Hohmann transfer from a 7000 km circular LEO to GEO and give
> me the total Δv.

The client lists the GMAT skeleton resources, reads
`gmat-skeleton://hohmann-transfer`, and edits it for the requested geometry.
Before spending a full mission run, it parses the script with
`gmat_validate_script`:

```jsonc
// tools/call → gmat_validate_script
{ "script": "Create Spacecraft LEOSat; ... (edited skeleton text)" }
```

GMAT returns the declared resources and any parse errors. The model fixes a
mistyped field name GMAT flagged, re-validates clean, then runs it:

```jsonc
// tools/call → gmat_run_mission
{
  "script": "Create Spacecraft LEOSat; ... ",
  "overrides": { "LEOSat.SMA": 7000 }
}
```

The response carries a `run_id`, the mission summary, and the small report
inline; the model reads the two-impulse Δv from the report rows. Ephemerides
and oversized reports come back as pointers — pull the full bytes with
`gmat_read_run_artefact(run_id="…", name="EphemerisFile1")`. To explore how
the Δv responds across a range of burn magnitudes instead of a single run,
reach for `gmat_sweep`; see
[GMAT integration](gmat-integration.md#worked-recipes) for the sweep and
validate-then-run patterns in full.
