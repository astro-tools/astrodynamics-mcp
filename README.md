# astrodynamics-mcp

A [Model Context Protocol](https://modelcontextprotocol.io) server that
gives any MCP-capable LLM client (Claude Code, Cursor, ChatGPT desktop,
custom agents) authoritative astrodynamics tools — TLE/SGP4 propagation,
Lambert solving, ground-station access, time-scale and coordinate-frame
conversions, porkchop scans, B-plane targeting — instead of letting the
model hallucinate the math.

> **Status — pre-alpha.** The public surface is not yet usable. The
> [current milestone](https://github.com/astro-tools/astrodynamics-mcp/milestone/1)
> tracks the work needed to ship the first PyPI release.

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
| `tle_lookup`      | Fetch current TLEs from CelesTrak by NORAD ID, name, or group. | CelesTrak `gp.php` API |
| `sgp4_propagate`  | Propagate TLEs across UTC ISO 8601 epochs in TEME / ICRF / GCRS / ITRS / CIRS. | `sgp4`            |
| `lambert_solve`   | Solve Lambert's problem; multi-rev solutions enumerated; two-impulse Δv on demand. | `lamberthub`     |
| `access_windows`  | Ground-station / observer access intervals over a window, with AOS / LOS / peak elevation. | `skyfield`     |
| `time_convert`    | UTC / TAI / TT / TDB / UT1 / GPS / TCB / TCG conversions across ISO / JD / MJD / J2000-seconds / Unix. | `astropy.time`   |
| `frame_transform` | State-vector transforms across ICRF / ITRS / GCRS / TEME / CIRS / TIRS / IAU body-fixed frames. | `astropy.coordinates` |
| `porkchop`        | (depart × arrive) Δv / C3 grid for interplanetary transfers, ASCII contour, summary or full output. | `lamberthub` + JPL Horizons |
| `bplane_target`   | B-plane element calculation and impulsive targeting for hyperbolic flybys. | in-house, JPL Horizons fed |

Full input / output JSON schemas live on the
[Tool reference](https://astro-tools.github.io/astrodynamics-mcp/tool-reference/)
page of the docs site.

## Quick start

Install (pre-alpha — not yet on PyPI; track progress on the
[issue tracker](https://github.com/astro-tools/astrodynamics-mcp/issues)):

```bash
uv tool install astrodynamics-mcp     # or: pipx install astrodynamics-mcp
```

### Claude Code

Add to your Claude Code MCP settings:

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
    "astrodynamics": {
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
  — getting started, per-client setup, tool reference, recipes, data
  sources, eval suite, FAQ.
- **Issue tracker:** [astro-tools/astrodynamics-mcp/issues](https://github.com/astro-tools/astrodynamics-mcp/issues)
- **Discussions:** [orgs/astro-tools/discussions](https://github.com/orgs/astro-tools/discussions)
  — usage help and open-ended questions.
- **Eval suite:** [eval/README.md](https://github.com/astro-tools/astrodynamics-mcp/tree/main/eval#readme)
  — the regression contract on tool-description quality.

## License

MIT — see [LICENSE](https://github.com/astro-tools/astrodynamics-mcp/blob/main/LICENSE).
