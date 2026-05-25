# astrodynamics-mcp

Model Context Protocol server giving any LLM client (Claude, ChatGPT, Cursor, custom agents) authoritative astrodynamics tools — TLE/SGP4, Lambert, ground-station access, time/frame conversions, porkchop, B-plane.

!!! warning "Pre-alpha"
    The public surface is not yet usable. The v0.1 milestone tracks the
    work needed to ship the first PyPI release; see the
    [issue tracker](https://github.com/astro-tools/astrodynamics-mcp/issues)
    for progress.

## What it is

LLMs reason well about astrodynamics concepts but cannot do the numerical work — they can't propagate orbits, solve Lambert problems, or query SPICE ephemerides. astrodynamics-mcp lets you plug authoritative tools into any [Model Context Protocol](https://modelcontextprotocol.io)-capable client so the LLM calls vetted upstream libraries instead of hallucinating numbers.

The v0.1 surface wraps eight no-auth tools across the most common single-satellite questions: `tle_lookup`, `sgp4_propagate`, `lambert_solve`, `access_windows`, `time_convert`, `frame_transform`, `porkchop`, `bplane_target`.

## What it is not

A general-purpose astrodynamics framework. astrodynamics-mcp wraps existing libraries; it does not re-implement propagators, integrators, or coordinate systems. For direct (non-MCP) Python access to the same surfaces, reach for the upstream libraries:

- SGP4 / TLE propagation → [`sgp4`](https://github.com/brandon-rhodes/python-sgp4)
- Lambert's problem → [`lamberthub`](https://github.com/jorgepiloto/lamberthub)
- Ground-station / observer geometry → [`skyfield`](https://rhodesmill.org/skyfield/)
- Time scales and coordinate frames → [`astropy`](https://www.astropy.org/)

## License

MIT — see [LICENSE](https://github.com/astro-tools/astrodynamics-mcp/blob/main/LICENSE).
