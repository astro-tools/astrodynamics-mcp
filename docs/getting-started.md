# Getting started

This page walks through installing astrodynamics-mcp, wiring it into an MCP
client, and running the canonical vision conversation. For per-client setup
details (Claude Code, Cursor, ChatGPT desktop, raw Python) see
[Pick a client](pick-a-client.md).

## Install

The server is a pure-Python package; the runtime install pulls
[`mcp`](https://github.com/modelcontextprotocol/python-sdk),
[`sgp4`](https://github.com/brandon-rhodes/python-sgp4),
[`lamberthub`](https://github.com/jorgepiloto/lamberthub),
[`skyfield`](https://rhodesmill.org/skyfield/),
[`astropy`](https://www.astropy.org/), and a few smaller dependencies. No
external services are required for the current tool surface — every tool either
runs offline or calls a no-auth data source (CelesTrak, JPL Horizons, IERS).

The recommended install path is `uv` or `pipx` so the `astrodynamics-mcp`
console script lives in an isolated environment that your MCP client can
launch directly.

```bash
uv tool install astrodynamics-mcp     # preferred
# or
pipx install astrodynamics-mcp
# or, into the current venv
pip install astrodynamics-mcp
```

Verify the install:

```bash
astrodynamics-mcp --version
```

## Choose a transport

The server ships two transports over the same tool surface:

- **stdio.** The local default. Claude Code, Cursor, and ChatGPT desktop
  spawn the binary as a subprocess and speak MCP over its stdin/stdout.
  This is what the quick-start configures.
- **Streamable HTTP.** For remote agents and custom Python clients. Bind to
  a host and port; the client speaks MCP over a long-lived HTTP connection
  per the 2025-11-25 spec.

```bash
astrodynamics-mcp stdio              # local default; clients spawn this for you
astrodynamics-mcp http --port 8000   # remote agents over Streamable HTTP
```

The CLI is the only difference between the two — tool registration is shared
across transports.

## Wire it into Claude Code

The other clients work the same way with their own config formats; see
[Pick a client](pick-a-client.md). For Claude Code, add the server to
your MCP client config:

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

Restart the client. All eight tools appear in its tool list.

## The vision conversation

Once wired in, the canonical conversation looks like this:

> **You:** Compute the Hohmann Δv from a 250 km circular LEO to GEO.
>
> *(The LLM calls `lambert_solve` with the two-impulse depart/arrive
> geometry. Tool returns the two velocity vectors and the total Δv, units
> explicit. LLM quotes ≈ 3.93 km/s.)*
>
> **You:** Now plot Hubble passes above 10° from Madrid over the next
> seven days.
>
> *(The LLM calls `tle_lookup("HUBBLE")`, then `access_windows(...)` with
> the named station, the fetched TLE, the requested time window, and
> `min_elevation_deg=10`. Tool returns a list of AOS/peak/LOS triples.)*

This is the difference from a bare LLM session: the numbers are produced by
vetted upstream libraries and the conversation can cite them.

## Trust and reproducibility

Every numeric output carries an explicit unit string — never implicit km vs
m. Frame strings are always emitted alongside state vectors. Failures
surface as typed JSON error envelopes with stable string codes (see
[`astrodynamics_mcp.errors`](api.md)) — the LLM sees an actionable error,
not a leaked Python traceback.

Repeated tool calls with the same input are deterministic up to the
upstream library's own determinism guarantees (SGP4 is fully deterministic;
lamberthub solvers converge to the same root within numerical tolerance;
JPL Horizons responses are cached). See
[Data sources](data-sources.md) for cache and staleness behaviour.

## Where next

- [Pick a client](pick-a-client.md) — per-client setup details.
- [Tool reference](tool-reference.md) — every tool with its current input / output schema.
- [Recipes](recipes.md) — worked examples covering the canonical workflows.
- [Data sources](data-sources.md) — CelesTrak / Horizons / IERS, cache, staleness.
- [Eval suite](eval-suite.md) — how we measure tool-description quality on every PR.
