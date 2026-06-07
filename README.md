# astrodynamics-mcp

A [Model Context Protocol](https://modelcontextprotocol.io) server that
gives any MCP-capable LLM client (Claude Code, Cursor, ChatGPT desktop,
custom agents) authoritative astrodynamics tools: TLE/SGP4 propagation,
Lambert solving, ground-station access, time-scale and coordinate-frame
conversions, porkchop scans, B-plane targeting, satellite metadata, and
— with optional extras — full NASA GMAT mission execution (`[gmat]`),
NASA SPICE / NAIF kernel queries (`[spice]`), and trajectory visualisation
(`[viz]`).

## Why

LLMs reason well about astrodynamics concepts but cannot do the
numerical work — they cannot propagate orbits, solve Lambert problems,
or query SPICE ephemerides. `astrodynamics-mcp` lets you plug
authoritative tools into any MCP-capable client so the LLM calls vetted
upstream libraries instead of fabricating numbers. Every result carries
explicit units; every tool description tunes against an
[Inspect AI eval suite](https://github.com/astro-tools/astrodynamics-mcp/tree/main/eval)
that measures whether the LLM picks the right tool and binds the right
arguments.

## Tools

| Tool              | What it does                                              | Backed by                |
| ----------------- | --------------------------------------------------------- | ------------------------ |
| `tle_lookup`      | Fetch current TLEs by NORAD ID, name, or group — from CelesTrak (default) or Space-Track. | CelesTrak `gp.php` API · Space-Track † |
| `sgp4_propagate`  | Propagate TLEs across UTC ISO 8601 epochs in TEME / ICRF / GCRS / ITRS / CIRS. | `sgp4`            |
| `lambert_solve`   | Solve Lambert's problem; multi-rev solutions enumerated; two-impulse Δv on demand. | `lamberthub`     |
| `access_windows`  | Ground-station / observer access intervals over a window, with AOS / LOS / peak elevation. | `skyfield`     |
| `time_convert`    | UTC / TAI / TT / TDB / UT1 / GPS / TCB / TCG conversions across ISO / JD / MJD / J2000-seconds / Unix. | `astropy.time`   |
| `frame_transform` | State-vector transforms across ICRF / ITRS / GCRS / TEME / CIRS / TIRS / IAU body-fixed frames. | `astropy.coordinates` |
| `porkchop`        | (depart × arrive) Δv / C3 grid for interplanetary transfers, ASCII contour, summary or full output. | `lamberthub` + JPL Horizons |
| `bplane_target`   | B-plane element calculation and impulsive targeting for hyperbolic flybys. | in-house, JPL Horizons fed |
| `satellite_metadata` | Physical & provenance metadata (mass, dimensions, COSPAR ID, launch, operator, decay status) for a NORAD ID. | ESA DISCOSweb † |

**†** Credentialed source. Pass credentials as environment variables for
the stdio transport, or in the session-init `_meta` block for HTTP — see
[Credentials](https://astro-tools.github.io/astrodynamics-mcp/credentials/).
A tool called without its credential returns a typed
`CredentialRequiredError`, never a silent failure.

### GMAT tools (optional `[gmat]` extra)

Install the `[gmat]` extra and have a local [NASA GMAT](https://sourceforge.net/projects/gmat/)
install, and five more tools register for driving real GMAT missions
(they stay hidden otherwise):

| Tool                    | What it does                                              | Backed by      |
| ----------------------- | --------------------------------------------------------- | -------------- |
| `gmat_run_mission`      | Run a complete GMAT mission; returns a parsed summary, report data, and pointers to large outputs. | `gmat-run`   |
| `gmat_sweep`            | Parameter sweeps and Monte Carlo (grid / samples / Monte Carlo / Latin hypercube) over a mission. | `gmat-sweep` |
| `gmat_execute_script`   | Escape hatch — run raw GMAT script text and return its reports verbatim; engine errors come back as data. | `gmat-run` |
| `gmat_validate_script`  | Parse-validate a script without running it; returns errors, warnings, and the resource/command structure. | `gmat-run` |
| `gmat_read_run_artefact`| Read the raw text of a file produced by a prior run (ephemerides, reports too large to inline). | run registry |

### SPICE tools (optional `[spice]` extra)

Install the `[spice]` extra and seven more tools register, backed by NASA
NAIF's CSPICE through [`spiceypy`](https://github.com/AndrewAnnex/SpiceyPy)
(they stay hidden otherwise). They furnish kernels into a process-global
pool and query whatever the pool holds:

| Tool                    | What it does                                              | Backed by      |
| ----------------------- | --------------------------------------------------------- | -------------- |
| `spice_load_kernel`     | Furnish a kernel into the pool from a local path or a NAIF `https` URL (allowlisted, cached); a meta-kernel furnishes all it lists. | `spiceypy` · NAIF |
| `spice_list_kernels`    | List the kernels currently furnished in the pool, optionally filtered by category. | `spiceypy` |
| `spice_unload_kernel`   | Drop a furnished kernel by the `name` `spice_load_kernel` returned. | `spiceypy` |
| `spice_state`           | Position / velocity of a target relative to an observer at one or more epochs, from furnished SPK kernels. | `spiceypy` (SPK) |
| `spice_frame_transform` | Rotate a vector between kernel-defined frames — in particular non-Earth body-fixed frames — or return the rotation matrix. | `spiceypy` (FK / PCK) |
| `spice_body_parameters` | Read a body's radii, GM, and pole / prime-meridian orientation constants from furnished PCK kernels. | `spiceypy` (PCK) |
| `spice_time_convert`    | Convert between the kernel-defined time systems ET / UTC / SCLK using furnished LSK / SCLK kernels. | `spiceypy` (LSK / SCLK) |

The kernel model, the NAIF furnish-from-URL allowlist, and the
process-global pool's trust boundary are covered on the
[SPICE integration](https://astro-tools.github.io/astrodynamics-mcp/spice-integration/)
page.

### Visualisation tools (optional `[viz]` extra)

Install the `[viz]` extra and four more tools register, backed by
[`matplotlib`](https://matplotlib.org/) (static PNG plots) and the
[`gmat-czml`](https://github.com/astro-tools/gmat-czml) sibling (CZML export)
— they stay hidden otherwise. Each returns its picture as an attachment
*alongside* a numeric summary, so a text-only client still gets the answer:

| Tool                | What it does                                              | Backed by      |
| ------------------- | --------------------------------------------------------- | -------------- |
| `plot_ground_track` | Render a satellite's sub-satellite ground track as a PNG over a lon/lat graticule, with the latitude / longitude extent inline. | `matplotlib` |
| `plot_trajectory`   | Render an orbit or transfer arc as a 2D or 3D PNG about a central body, with arc length and apsides inline. | `matplotlib` |
| `plot_porkchop`     | Render a porkchop C3 contour as a PNG from a full `porkchop` grid result — no recompute — with the best cell marked. | `matplotlib` |
| `czml_trajectory`   | Export a trajectory as a CZML document for a Cesium 3D client, returned as an embedded resource. | `gmat-czml`  |

The attachment model — additive PNG `ImageContent` / CZML `EmbeddedResource`
beside the structured summary — and which clients render each kind are covered
on the
[Visualisation](https://astro-tools.github.io/astrodynamics-mcp/visualisation/)
page.

Full input / output JSON schemas live on the
[Tool reference](https://astro-tools.github.io/astrodynamics-mcp/tool-reference/)
page of the docs site.

## Quick start

Install:

```bash
uv tool install astrodynamics-mcp            # or: pipx install astrodynamics-mcp
uv tool install "astrodynamics-mcp[gmat]"    # adds the GMAT mission tools (needs a local GMAT install)
uv tool install "astrodynamics-mcp[spice]"   # adds the SPICE tools (pulls spiceypy / bundled CSPICE)
uv tool install "astrodynamics-mcp[viz]"     # adds the visualisation tools (pulls matplotlib / gmat-czml)
```

### Claude Code

Add to your Claude Code MCP settings:

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

Restart Claude Code. In a chat:

> **You:** Compute the Hohmann Δv from a 250 km circular LEO to GEO.
>
> *(The model calls `lambert_solve` with the Hohmann geometry and
> answers ≈ 3.91 km/s, citing the tool output — not the LLM's own
> weights.)*

### Cursor

`~/.cursor/mcp.json` (or workspace-level `.cursor/mcp.json`):

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

Restart Cursor. The tools appear under the `astrodynamics` server group.

See
[Pick a client](https://astro-tools.github.io/astrodynamics-mcp/pick-a-client/)
in the docs for ChatGPT desktop, a raw Python MCP smoke client, and the
Streamable HTTP transport for remote agents.

## Supported clients

| Client                       | Transport       | Verified              |
| ---------------------------- | --------------- | --------------------- |
| Claude Code                  | stdio           | ✅ Yes                |
| Cursor                       | stdio           | ✅ Yes                |
| ChatGPT desktop              | stdio           | ⏳ Expected to work   |
| Raw Python (`mcp` SDK)       | stdio           | ✅ Yes                |
| Remote agents                | Streamable HTTP | ⏳ Expected to work   |
| LangGraph / AutoGen / CrewAI | any             | ⏳ Expected to work   |

## What this is not

- **Not a general-purpose astrodynamics framework.** Wraps vetted
  upstream libraries; does not re-implement propagators, integrators,
  or coordinate systems.
- **Not an agent framework.** Exposes MCP tools; LangGraph, AutoGen,
  CrewAI, and the LLM clients themselves consume them.
- **Not an ML / inference server.** Tools that need their own ML
  models (maneuver detection, neural propagators) belong in separate
  MCP servers — kept modular for dependency isolation.
- **Not a SaaS.** Runs locally or in your own infrastructure. No
  hosted multi-tenant deployment.
- **Not a web UI.** Tool consumption is via MCP clients; no browser
  frontend, no desktop app, no notebook widget.

For direct (non-MCP) Python use of the same surfaces, reach for the
upstream libraries:
[`sgp4`](https://github.com/brandon-rhodes/python-sgp4),
[`lamberthub`](https://github.com/jorgepiloto/lamberthub),
[`skyfield`](https://rhodesmill.org/skyfield/),
[`astropy`](https://www.astropy.org/),
[`interplanetary-porkchop`](https://github.com/mlewicki/interplanetary-porkchop),
[`spiceypy`](https://github.com/AndrewAnnex/SpiceyPy).

## Built on

The official Anthropic
[`modelcontextprotocol/python-sdk`](https://github.com/modelcontextprotocol/python-sdk)
(MIT). The bundled FastMCP server class is the server primitive;
stdio + Streamable HTTP transports are first-class.

## Docs and links

- **Docs site:** [astro-tools.github.io/astrodynamics-mcp](https://astro-tools.github.io/astrodynamics-mcp/)
  — getting started, per-client setup, tool reference, recipes,
  visualisation, data sources, eval suite, FAQ.
- **Issue tracker:** [astro-tools/astrodynamics-mcp/issues](https://github.com/astro-tools/astrodynamics-mcp/issues)
- **Discussions:** [orgs/astro-tools/discussions](https://github.com/orgs/astro-tools/discussions)
  — usage help and open-ended questions.
- **Eval suite:** [eval/README.md](https://github.com/astro-tools/astrodynamics-mcp/tree/main/eval#readme)
  — the regression contract on tool-description quality.

## Privacy

`astrodynamics-mcp` runs entirely on your own machine and collects
nothing — no telemetry, no analytics, no accounts. The only data that
leaves your machine is the query parameters a tool sends to the data
source it wraps (CelesTrak / JPL Horizons / IERS with no auth, and —
only if you configure their credentials — Space-Track and ESA DISCOSweb).
Credentials are read from local environment variables or the session
`_meta` block and are sent only to their own service over HTTPS. See the
[Privacy page](https://astro-tools.github.io/astrodynamics-mcp/privacy/)
for the full breakdown.

## License

MIT — see [LICENSE](https://github.com/astro-tools/astrodynamics-mcp/blob/main/LICENSE).

<!-- mcp-name: io.github.astro-tools/astrodynamics-mcp -->

