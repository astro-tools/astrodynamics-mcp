# astrodynamics-mcp

Model Context Protocol server giving any LLM client (Claude, ChatGPT, Cursor,
custom agents) authoritative astrodynamics tools — TLE/SGP4, Lambert,
ground-station access, time/frame conversions, porkchop, B-plane.

## What it is

LLMs reason well about astrodynamics concepts but cannot do the numerical
work — they cannot propagate orbits, solve Lambert problems, or query SPICE
ephemerides. astrodynamics-mcp lets you plug authoritative tools into any
[Model Context Protocol](https://modelcontextprotocol.io)-capable client so
the LLM calls vetted upstream libraries instead of hallucinating numbers.

The current surface wraps eight no-auth tools across the most common
single-satellite questions: `tle_lookup`, `sgp4_propagate`, `lambert_solve`,
`access_windows`, `time_convert`, `frame_transform`, `porkchop`,
`bplane_target`.

## Quick start

Install the server:

```bash
uv tool install astrodynamics-mcp     # or: pipx install astrodynamics-mcp
```

Add it to your MCP client. Claude Code, for example:

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

Restart the client. The eight tools appear in its tool list. Ask:

> Compute the Hohmann Δv from a 250 km circular LEO to GEO.

The LLM calls `lambert_solve` instead of guessing, and quotes a number you
can reproduce.

> Now plot Hubble passes above 10° from Madrid for the next seven days.

`tle_lookup` fetches the current Hubble TLE from CelesTrak; `access_windows`
returns AOS/LOS/peak-elevation triples; the client formats them.

## Next

- [Getting started](getting-started.md) — install paths and the full vision conversation.
- [Pick a client](pick-a-client.md) — Claude Code, Cursor, ChatGPT desktop, raw Python.
- [Tool reference](tool-reference.md) — every tool with its current input / output schema.
- [Recipes](recipes.md) — worked examples covering the canonical workflows.

## What it is not

A general-purpose astrodynamics framework. astrodynamics-mcp wraps existing
libraries; it does not re-implement propagators, integrators, or coordinate
systems. For direct (non-MCP) Python access to the same surfaces, reach for
the upstream libraries:

- SGP4 / TLE propagation → [`sgp4`](https://github.com/brandon-rhodes/python-sgp4)
- Lambert's problem → [`lamberthub`](https://github.com/jorgepiloto/lamberthub)
- Ground-station / observer geometry → [`skyfield`](https://rhodesmill.org/skyfield/)
- Time scales and coordinate frames → [`astropy`](https://www.astropy.org/)

See the [FAQ](faq.md) for the full "what astrodynamics-mcp is not" list.

## License

MIT — see [LICENSE](https://github.com/astro-tools/astrodynamics-mcp/blob/main/LICENSE).
