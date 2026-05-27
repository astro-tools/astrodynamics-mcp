# Changelog

All notable changes to astrodynamics-mcp are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.5] — 2026-05-27

### Added

- Per-parameter `description` on every tool input. Each parameter is now
  declared `Annotated[type, Field(description="…")]`, so the generated
  JSON Schema's `properties.<param>.description` carries semantics,
  units, and acceptable value ranges. Any MCP-connected LLM benefits
  from richer argument-binding metadata; Smithery's catalog UI surfaces
  the descriptions in its tool browser.
- `outputSchema` for every tool, surfaced through the published Smithery
  bundle. FastMCP already derived these from the pydantic response
  models — `scripts/dump-mcpb-tools.py` now includes them in each
  bundle `tools[]` entry alongside `inputSchema`.
- `annotations` for every tool — `readOnlyHint: true` on all eight (none
  mutate remote state) and `openWorldHint: true` on the three tools
  whose core function is fetching from an external service
  (`tle_lookup` from CelesTrak, `porkchop` and `bplane_target` from JPL
  Horizons). The `register_tool` decorator gains an `annotations` kwarg
  that passes through to FastMCP's `@mcp.tool()`.

## [0.1.4] — 2026-05-27

### Fixed

- Smithery publish, which v0.1.3 still rejected with
  `400 {"error":"Invalid input: expected object, received undefined"}` ×8 —
  Smithery's release API mirrors the MCP-protocol Tool type and requires a
  full `inputSchema` per tool, but the MCPB spec only accepts
  `{name, description}` and `mcpb pack` rejects `inputSchema` as an
  unrecognised key. Fix: `scripts/dump-mcpb-tools.py` generates a real
  tools array from the live server (`mcp.list_tools()` with full
  `inputSchema`); the workflow `jq`-injects it into the Smithery
  manifest and builds the `.mcpb` as a direct `zip` (bypassing
  `mcpb pack`'s validation). The canonical bundle stays spec-compliant
  with `"tools": []` — Claude Desktop and other MCPB-aware clients
  discover tools via the MCP protocol at runtime.

## [0.1.3] — 2026-05-27

### Fixed

- Smithery publish, which v0.1.2 left at "server entity created but no
  release" with `400 {"error":"No values to set"}`. Smithery's
  `releases.create` endpoint rejects a publish whose payload carries only
  `serverInfo.name` and `serverInfo.version`; at least one of `tools`,
  `prompts`, `resources`, or `user_config` must be present. Fix: declare
  the eight v0.1 tools as a top-level `tools[]` array (per the MCPB spec)
  in both the canonical and the Smithery MCPB manifests. Detailed
  input/output schemas remain discovered at runtime via the MCP protocol.

## [0.1.2] — 2026-05-27

### Fixed

- Release workflow's `publish-smithery` and `publish-mcp-registry` jobs, both
  of which failed on the v0.1.1 tag fire. The Official MCP Registry rejects
  any `$schema` URL outside the canonical static form
  (`https://static.modelcontextprotocol.io/schemas/<date>/server.schema.json`);
  the GitHub-raw URL the schema's own `$id` claims was the wrong source of
  truth. Smithery's CLI does not recognise MCPB `server.type: "uv"`. Fix:
  corrected the registry `$schema`, and added a Smithery-specific bundle
  under `packaging/mcpb-smithery/` with `type: "python"` plus
  `mcp_config.command: "uv"` and inline `--with` deps — the pattern
  Smithery's own MCPB bundling docs document. The canonical
  `packaging/mcpb/` bundle (`type: "uv"`) is unchanged and remains the
  asset attached to the GitHub Release.

## [0.1.1] — 2026-05-26

### Added

- MCPB bundle is built and published as a GitHub release asset on every
  `v*` tag, then auto-distributed to Smithery as
  `astro-tools/astrodynamics-mcp` and to the Official MCP Registry as
  `io.github.astro-tools/astrodynamics-mcp`. The bundle uses MCPB
  `manifest_version: "0.4"` with `type: "uv"` and pins the just-released
  PyPI version, so users can install the server through Claude Desktop's,
  Cursor's, or any MCP-aware client's catalog UI without hand-editing JSON
  config blocks. The release workflow grows three additive jobs
  (`pack-mcpb`, `publish-smithery`, `publish-mcp-registry`) downstream of
  the existing `publish-pypi` job.

## [0.1.0] — 2026-05-26

Initial public release. A Model Context Protocol server exposing eight
no-auth astrodynamics tools — TLE lookup, SGP4 propagation, Lambert
solving, ground-station access, time-scale and coordinate-frame
conversion, interplanetary porkchop scans, and B-plane targeting — over
stdio and Streamable HTTP transports, with a regression eval against
GitHub Models on every workflow dispatch.

### Added

- `tle_lookup` — fetch current TLEs from CelesTrak by NORAD catalog
  number, satellite name, or group. Routes through the cache layer with
  per-source TTL and a stale-fallback path when the upstream is
  unreachable (#9).
- `sgp4_propagate` — propagate a TLE across UTC ISO 8601 epochs and
  return position/velocity in TEME, ICRF, GCRS, ITRS, or CIRS, backed by
  the `sgp4` C implementation. Default response size capped to keep the
  MCP envelope small for the LLM consumer (#10).
- `lambert_solve` — solve Lambert's problem via `lamberthub`, with
  multi-revolution enumeration and optional two-impulse Δv computation.
  Four solvers exposed; defaults pick the one with the broadest
  convergence basin (#11).
- `access_windows` — ground-station / observer access intervals over a
  time window, returning AOS / LOS / peak-elevation triples. Named
  station registry plus arbitrary lat/lon/elevation, range filtering,
  and `min_elevation_deg` gating; backed by `skyfield` (#12).
- `time_convert` — convert epochs across UTC, TAI, TT, TDB, UT1, GPS,
  TCB, and TCG in ISO 8601, JD, MJD, J2000-seconds, and Unix
  representations, via `astropy.time` (#13).
- `frame_transform` — transform state vectors across ICRF, ITRS, GCRS,
  TEME, CIRS, TIRS, and IAU body-fixed frames, via
  `astropy.coordinates` (#14).
- `porkchop` — (depart × arrive) Δv / C3 grid for interplanetary
  transfers, composed from `lamberthub` and a JPL Horizons ephemeris
  adapter. Returns either a summary, full grid, or ASCII-contour view;
  default response size capped (#15).
- `bplane_target` — B-plane element calculation and impulsive targeting
  for hyperbolic flybys, fed by the same Horizons adapter (#16).
- FastMCP server primitive with a per-tool description lint that fails
  CI on missing units, undocumented arguments, or empty examples in any
  registered tool (#7).
- `astrodynamics-mcp` console script with `stdio` and `http`
  subcommands. The stdio path is what MCP clients launch as a
  subprocess; the HTTP path exposes the same tool surface over
  Streamable HTTP per the 2025-11-25 spec (#8).
- Typed error hierarchy under `astrodynamics_mcp.errors` rooted at
  `AstroMCPError`, with explicit units discipline across every tool
  output — every numeric field carries an explicit unit string and
  every state vector carries an explicit frame string (#3).
- Shared pydantic base schemas with JSON-schema export, so every tool's
  input and output surface is type-checked at the boundary and the same
  schemas drive the docs site's tool reference page (#4).
- XDG-aware on-disk cache layer at `~/.cache/astrodynamics-mcp/`, with
  per-source TTLs and a stale-value fallback when the upstream is
  unreachable. Cache key derivation is canonical so a re-run with the
  same arguments hits the same row (#5).
- Data adapters for CelesTrak (`gp.php`), JPL Horizons, and IERS
  Bulletin A, each cached through the layer above. The Horizons adapter
  speaks the small subset of the API the tools actually need; the IERS
  adapter resolves UT1-UTC for `time_convert` and `frame_transform`
  without requiring an astropy data download at runtime (#6).
- Cross-tool validation test suite — TLE round-trip across
  `tle_lookup` → `sgp4_propagate`, reference-output regressions for
  Lambert and porkchop, an equivalence test comparing
  `lambert_solve`'s two-impulse Δv against a Hohmann analytic, and
  failure-mode coverage for each typed error code (#17).
- Inspect AI eval suite with a hybrid scorer (trace-match + functional
  tolerance), exercised by `eval/tasks.py`. Goldens cover every tool at
  the single-tool tier and at least one tool at the sequential and
  planning tiers. The manual GitHub Models gate via `eval.yml`
  measures whether the LLM picks the right tool and binds the right
  arguments on Claude Sonnet 4.6 and `gpt-4.1-mini` (#18, #19).
- MkDocs-Material documentation site published at
  [astro-tools.github.io/astrodynamics-mcp](https://astro-tools.github.io/astrodynamics-mcp/)
  — getting started, per-client setup, tool reference, recipes, data
  sources, eval suite, supported clients, FAQ, and output shaping.
  Deployed to GitHub Pages on every `v*` tag push (#20).
- Three runnable example sessions — a Hohmann Δv single-tool prompt
  (`lambert_solve`), Hubble passes from Madrid as a sequential prompt
  (`tle_lookup` → `access_windows`), and a Mars 2028 launch-window
  planning prompt (`porkchop` + follow-up `lambert_solve`). Each ships
  with a markdown transcript and a Python script that drives the same
  sequence against an in-process MCP server and asserts numerical
  tolerance (#21).
- CI on Ubuntu and Windows × Python 3.10 / 3.11 / 3.12 (6 cells),
  separate `lint`, `typecheck`, and `minimal install smoke` jobs, and a
  release workflow that builds via `uv build`, publishes to PyPI via
  trusted publishing, and creates the GitHub Release on every `v*` tag.
  macOS is deferred to v0.2 (#1, #2).

[0.1.5]: https://github.com/astro-tools/astrodynamics-mcp/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/astro-tools/astrodynamics-mcp/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/astro-tools/astrodynamics-mcp/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/astro-tools/astrodynamics-mcp/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/astro-tools/astrodynamics-mcp/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/astro-tools/astrodynamics-mcp/releases/tag/v0.1.0
