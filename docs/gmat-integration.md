# GMAT integration

[GMAT](https://gmat.gsfc.nasa.gov/) — NASA's General Mission Analysis Tool —
is a full mission-analysis environment: spacecraft, force models, propagators,
burns, targeters, and optimizers driven by a scripting language. The
astrodynamics-mcp GMAT surface lets an LLM client author a GMAT `.script`,
check that it parses, run it, sweep it over a parameter grid, and read back
the reports and ephemerides it produces — without leaving the chat.

The GMAT tools live behind an optional extra. Install it and they appear in
`tools/list`; skip it and the rest of the tool surface is unaffected.

```bash
uv tool install 'astrodynamics-mcp[gmat]'
# or
pipx install 'astrodynamics-mcp[gmat]'
```

The extra pulls in the [`gmat-run`](https://github.com/astro-tools/gmat-run)
and [`gmat-sweep`](https://github.com/astro-tools/gmat-sweep) drivers, which
locate a GMAT install at runtime. Without the extra, the GMAT tools do not
register; with it, each tool registers when its underlying driver imports
successfully.

## Tool surface

Five tools cover the GMAT use cases:

- `gmat_run_mission(script, overrides=…, select_outputs=…, output=…)` — the
  canonical "run my mission" verb. `script` is either an absolute path to a
  `.script` file or the inline script text. `overrides` applies dotted-path
  field writes (`Sat.SMA`, `FM.Drag.AtmosphereModel`) before the run, so one
  script can be re-run with tweaked inputs. The response inlines small
  reports and trims large ones; ephemerides and contact locators come back as
  pointers you re-read with `gmat_read_run_artefact`.
- `gmat_sweep(script, mode, grid|samples|perturb, …)` — parameter sweeps and
  Monte Carlo runs over a mission script. `mode` selects the variation
  strategy (`grid` for a full-factorial cartesian product, `samples` for
  explicit rows, `monte_carlo` / `latin_hypercube` for stochastic draws over
  `perturb` ranges) and dictates which payload field is required.
- `gmat_validate_script(script)` — parses the script through GMAT without
  executing the mission sequence. Returns parse errors, unknown resources or
  fields, and the list of declared resources. Built for a self-correction
  loop before `gmat_run_mission` — see [Script authoring](#script-authoring).
- `gmat_read_run_artefact(run_id, name, output=…)` — reads a single artefact
  from a prior `gmat_run_mission`, `gmat_sweep`, or `gmat_execute_script`
  call by its `run_id` and resource name (`ReportFile1`, `EphemerisFile1`, a
  solver name, or a plain file basename). This is how you pull the full bytes
  of an output that the run response returned only as a pointer.
- `gmat_execute_script(script)` — escape hatch for raw scripted output. The
  tool description steers callers toward `gmat_run_mission` first; this tool
  exists for cases the curated surface does not cover.

## When to pick which

- **Run a mission and read its results** → `gmat_run_mission`. This is the
  default. It loads the script, applies any overrides, runs the full mission
  sequence, and shapes the response for an LLM context window.
- **Same mission, many inputs** → `gmat_sweep`. Reach for it when the
  question is "how does apogee vary as I sweep the burn Δv from 1 to 3 km/s"
  rather than a single run.
- **Check a script before running it** → `gmat_validate_script`. Cheap,
  no mission execution; use it to close the author → fix → run loop.
- **Re-read a large output** → `gmat_read_run_artefact`. The run response
  points at ephemerides and oversized reports instead of inlining them;
  this tool fetches the full bytes on demand.
- **The curated tools don't fit** → `gmat_execute_script`. The raw escape
  hatch. Prefer `gmat_run_mission` unless you have a reason not to.

## Script authoring

LLMs producing a GMAT `.script` from scratch is the bottleneck for the
run-and-sweep tools above. The GMAT scripting DSL is case-sensitive,
strict about field and resource names, and silent about typos that
parse but produce a malformed mission. Two helpers narrow the gap.

`gmat_validate_script` gives the LLM a parse-only feedback loop. The
intended pattern is: author a script, call `gmat_validate_script`, fix
the errors GMAT itself surfaces, then call `gmat_run_mission`. This
keeps full mission runs reserved for scripts that already parse.

Vetted starter scripts ship as **MCP resources**, not tools. Each
skeleton has a stable `gmat-skeleton://<slug>` URI (for example
`gmat-skeleton://hohmann-transfer`) and is discoverable through the
client's resource-listing flow. Callers list the resources, read a
skeleton, edit it for their problem, and pass the edited script to
`gmat_run_mission`.

The catalogue spans the major mission archetypes — basic propagation,
force-model variations, impulsive and finite-burn transfers, lunar and
interplanetary missions, station-keeping, optimisation, libration-point
design, event location (ground-station contact, eclipse), attitude
modes, formation flying, and scripting control flow. Each skeleton is
derived from a GMAT R2026a bundled sample, stripped of GUI subscribers
(`OrbitView`, `GroundTrackPlot`, `XYPlot`) and any plugin-only resources
that aren't part of the default plugin load, and prefaced with a
`% Description: <text>` line that the client surfaces as the resource
description.

Resources register only when the `[gmat]` extra is installed (same
import guard as the tools), and every skeleton round-trips through
`gmat_validate_script` cleanly in CI, so a stripped-out plugin
reference that sneaks back in fails the build.

## Worked recipes

### Run a Hohmann transfer from a skeleton

> **You:** Run a Hohmann transfer from a 7000 km circular LEO to GEO and tell
> me the total Δv.

The client lists the skeleton resources, reads `gmat-skeleton://hohmann-transfer`,
and edits it for the requested geometry — or runs it as-is and adjusts via
`overrides`. It then calls `gmat_run_mission`:

```jsonc
// tools/call → gmat_run_mission
{
  "script": "Create Spacecraft LEOSat; ... (edited skeleton text)",
  "overrides": { "LEOSat.SMA": 7000 }
}
```

The response carries a `run_id`, the mission summary, and the small report
inline. The LLM reads the two-impulse Δv from the report rows and quotes it.
If the run produced an ephemeris you want in full, follow up with
`gmat_read_run_artefact(run_id="…", name="EphemerisFile1")`.

### Author, validate, then run

> **You:** Propagate a 500 km sun-synchronous orbit for one day with drag on
> and give me the decay in altitude.

The model writes a `.script` from scratch (or from `gmat-skeleton://minimal-leo`)
and calls `gmat_validate_script` first:

```jsonc
// tools/call → gmat_validate_script
{ "script": "Create Spacecraft Sat; ... " }
```

GMAT reports an unknown field — say the model wrote `FM.Drag.Model` instead of
`FM.Drag.AtmosphereModel`. The model fixes the line and re-validates; once the
script parses clean, it calls `gmat_run_mission` with the corrected text. The
validate step keeps a full propagation run reserved for a script GMAT has
already accepted.

### Sweep a burn over a Δv grid

> **You:** Sweep the apogee-raise burn from 1.0 to 2.0 km/s in 0.25 steps and
> show how final apogee responds.

The model starts from `gmat-skeleton://target-finite-burn-apogee-raise`, then
calls `gmat_sweep` in `grid` mode:

```jsonc
// tools/call → gmat_sweep
{
  "script": "Create Spacecraft Sat; ... ",
  "mode": "grid",
  "grid": { "Burn.Element1": [1.0, 1.25, 1.5, 1.75, 2.0] }
}
```

The tool runs the cartesian product (five runs here), returns one summary row
per case keyed by the swept value, and a `run_id` per case so a follow-up
`gmat_read_run_artefact` can pull any individual run's full report.

## Transports

The GMAT tools register on both stdio and Streamable HTTP transports —
there is no transport-specific gating. Operators own their HTTP
deployment's trust boundary — auth proxy, network controls, who can
reach the port — and the GMAT subset uses the same transport selection
as the rest of the tool surface. Because `gmat_execute_script` and
`gmat_run_mission` run arbitrary GMAT scripts, an HTTP deployment that
exposes the GMAT tools to untrusted callers is running untrusted code;
bind to `127.0.0.1` or front the server with your own auth unless the
network boundary is already trusted.
