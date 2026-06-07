# Changelog

All notable changes to astrodynamics-mcp are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] — 2026-06-06

The visualisation release. Adds four visualisation tools behind an
optional `[viz]` extra — three matplotlib static-plot tools
(`plot_ground_track`, `plot_trajectory`, `plot_porkchop`) returning PNG
images, and `czml_trajectory`, which exports a CZML document for a Cesium
3D client via the `gmat-czml` sibling. Every viz tool returns its picture
*alongside* the numeric summary — a PNG `ImageContent` or a CZML
`EmbeddedResource` added to, never replacing, the structured / ASCII
output — so a text-only client still gets the answer. The tools register
only when the extra is importable (`pip install astrodynamics-mcp[viz]`),
so the base install stays viz-free.

### Added

- Visualisation tools behind an optional `[viz]` extra — three matplotlib
  static-plot tools and one CZML export tool, plus the additive attachment
  output channel they share. `plot_ground_track` renders a sub-satellite
  ground track over a lon / lat graticule; `plot_trajectory` renders a 2D
  or 3D orbit / transfer arc about a central body; `plot_porkchop` renders
  a C3 contour from an existing `porkchop` grid with no recompute; and
  `czml_trajectory` exports a trajectory as a CZML document for a Cesium
  client. PNGs come back as `ImageContent`, CZML as an `EmbeddedResource`,
  each beside the inline numeric summary that stays the default. PNG
  rendering is deterministic — the headless Agg backend, a fixed DPI, and
  stripped version / timestamp metadata chunks make repeated renders
  byte-identical within a matplotlib version, so the same call returns an
  identical image over stdio and Streamable HTTP. The slots register only
  under the `[viz]` install, the same gate the `[gmat]` and `[spice]`
  extras use (#131, #132, #133).
- Eval-suite visualisation coverage — a single-tool prompt for each of the
  four viz tools plus a sequential `sgp4_propagate` → `plot_ground_track`
  prompt, scored by an attachment-aware hybrid scorer that asserts an
  attachment is produced and carries its declared type (`ImageContent`
  versus `EmbeddedResource`); a `requires_viz` gate skips these prompts
  when the extra is absent, mirroring `requires_spice` / `requires_gmat`.
  Plus two runnable example sessions — a static-plot session and a
  CZML-export session — each shipping a markdown transcript and a Python
  script that drives the same sequence against an in-process MCP server
  (#134).
- Documentation for the v0.4 surface: a Visualisation page covering the
  plot tools, `czml_trajectory`, the additive attachment model
  (`ImageContent` / `EmbeddedResource`), and which clients render images
  versus CZML; tool-reference entries for all four tools; output-shaping
  and supported-clients updates; the `[viz]` install line; and the viz
  tools table in the README (#136).

## [0.3.0] — 2026-06-06

The SPICE / NAIF release. Adds seven SPICE tools behind an optional
`[spice]` extra — kernel furnishing, SPK ephemeris state, frame
rotations, body parameters, and kernel-defined time conversions — backed
by NASA NAIF's CSPICE through `spiceypy`. The tools register only when
`spiceypy` is importable (`pip install astrodynamics-mcp[spice]`), so the
base install stays SPICE-free.

### Added

- SPICE tools behind an optional `[spice]` extra — the kernel-management
  trio (`spice_load_kernel`, `spice_list_kernels`, `spice_unload_kernel`)
  plus four query tools: `spice_state` (SPK position / velocity with
  light-time), `spice_frame_transform` (FK / PCK / TF frame rotations),
  `spice_body_parameters` (PCK radii / GM / pole and prime-meridian
  constants), and `spice_time_convert` (ET / UTC / SCLK via LSK / SCLK
  kernels). The tools furnish kernels into a process-global pool and
  query whatever the pool holds; every CSPICE call is serialised onto one
  dedicated worker thread, and `https` loads route through a NAIF-host
  allowlist and the XDG kernel cache. The slots register identically on
  stdio and Streamable HTTP (#120, #121, #122, #123, #124, #125, #126).
- Eval-suite SPICE coverage plus a fourth runnable example session — a
  Mars-state query that furnishes NAIF kernels and reads an SPK state —
  shipping a markdown transcript and a Python script that drives the same
  sequence against an in-process MCP server and asserts numerical
  tolerance (#127).
- Documentation for the v0.3 surface: a SPICE-integration design page
  covering the kernel model, the NAIF furnish-from-URL allowlist, and the
  process-global pool's trust boundary; tool-reference entries for all
  seven tools; and the `[spice]` install line and tools table in the
  README (#120, #129).

## [0.2.2] — 2026-05-29

Repository packaging only; no change to the installed package, tool
behaviour, or numerical output.

### Added

- A root `.mcp.json` and a `.plugin/plugin.json` manifest, following the
  [Open Plugin Specification](https://open-plugins.com), so the repository
  is auto-detectable — with real plugin metadata (name, description,
  author, homepage, license, keywords) instead of placeholders — by
  directory scanners such as the Cursor Directory. The `.mcp.json` runs
  the server zero-install via `uvx` with the `[gmat]` extra (so the GMAT
  tools are available when a local GMAT install is present) and doubles as
  a copy-paste client config.

### Changed

- Standardised the client-config alias to `astrodynamics-mcp` (from
  `astrodynamics`) across the README, docs, and example transcripts, so
  the alias the client groups the tools under matches the package and
  plugin name.

## [0.2.1] — 2026-05-29

Packaging and metadata polish for directory submission; no change to tool
behaviour or numerical output.

### Added

- A human-readable `title` on every tool's annotations (e.g. "TLE
  Lookup", "GMAT Run Mission"), so MCP clients and directories can show a
  display name alongside the tool id.
- A [Privacy page](https://astro-tools.github.io/astrodynamics-mcp/privacy/)
  and a README Privacy section documenting that the server runs locally
  and collects nothing; the MCPB manifest now declares a
  `privacy_policies` URL.
- Project logo committed to the repository.

### Changed

- Refreshed the MCPB bundle's `long_description` to cover the full v0.2
  tool surface (`satellite_metadata`, the GMAT tools, the Space-Track
  source); it previously described only the eight v0.1 tools.

## [0.2.0] — 2026-05-29

The GMAT integration release. Adds GMAT mission execution, parameter
sweeps, and script validation behind an optional `[gmat]` extra;
credential passthrough for Space-Track and ESA DISCOSweb; a
`satellite_metadata` tool; and a wider eval suite and CI matrix. Drops
the Smithery publishing path.

### Added

- GMAT integration tools behind an optional `[gmat]` extra —
  `gmat_run_mission`, `gmat_sweep`, `gmat_execute_script`,
  `gmat_validate_script`, and `gmat_read_run_artefact`, wrapping
  `gmat-run` and `gmat-sweep`. The tools register only when `gmat-run`
  is importable (`pip install astrodynamics-mcp[gmat]`), so the base
  install stays GMAT-free. Each call runs `gmatpy` in an isolated
  subprocess: gmatpy bootstraps a single global Moderator that
  successive runs in one interpreter would corrupt, and its C++ run loop
  holds the GIL, so isolation keeps runs from leaking state into each
  other or blocking the server's event loop (#70, #71, #72, #73, #82,
  #91).
- GMAT script skeletons shipped as MCP resources under
  `gmat-skeleton://` URIs — 20 vetted mission archetypes (LEO / GEO /
  lunar transfers, Hohmann, B-plane targeting, finite- and
  electric-propulsion burns, LEO and lunar station-keeping, contact and
  eclipse location, attitude pointing, constellations, control flow) to
  seed LLM-assisted script authoring (#83).
- Credential passthrough for credentialed data sources — environment
  variables for the stdio transport, session-init `_meta` metadata for
  HTTP. A tool that needs a credential it cannot find raises a typed
  `CredentialRequiredError` (stable `credential_required.<source>` code,
  naming the missing fields) rather than failing silently (#67).
- Space-Track as an alternate source for `tle_lookup` via
  `source="space-track"`, for recent launches and as a CelesTrak
  fallback. Requires Space-Track credentials; CelesTrak stays the
  no-auth default (#74).
- `satellite_metadata` — physical and provenance metadata (mass,
  bounding-box dimensions, COSPAR ID, launch date and site, operator,
  mission type, decay status) for a NORAD ID, backed by ESA DISCOSweb.
  Requires a DISCOSweb bearer token (#75).
- Eval suite expanded to 40 prompts with GMAT-tool and credentialed-tool
  coverage. GMAT and credentialed prompts skip — rather than fail — when
  their prerequisites are absent, so the suite runs without secrets or a
  GMAT install (#76).
- Multi-model offline eval workflow for release-cut comparison across
  the GitHub Models catalogue, separate from the per-PR gate (#69).
- macOS added to the CI test matrix — now Ubuntu, Windows, and macOS ×
  Python 3.10 / 3.11 / 3.12 (#68).
- Documentation for the v0.2 surface: a GMAT-integration design page
  recording the tool-granularity, HTTP-transport, and `[gmat]`-extra
  decisions, plus tool-reference and recipe updates (#66, #77).

### Changed

- Typed `AstroMCPError` envelopes now reach the wire on every tool — the
  `code`, `message`, and `data` fields serialise into the MCP error
  response so the LLM consumer can branch on a stable code, and
  non-finite (`NaN` / `Inf`) values are rejected at the `Quantity`
  boundary instead of leaking into a response (#108).
- Data-layer hardening across the adapters: per-event-loop client
  lifecycle with clean shutdown, stale-cache fallback on upstream
  outage, correct Space-Track query encoding, and caching of Horizons
  in-band errors (#110).

### Fixed

- Contained the basename fallback in `gmat_read_run_artefact` so a
  crafted `name` cannot escape the run's output directory — artefact
  reads are confined to direct children of the run workspace (#109).

### Removed

- Smithery publishing infrastructure. The canonical MCPB bundle ships
  with an empty `tools[]` (clients discover tools over the MCP protocol
  at runtime), but Smithery requires a populated `tools[]` to List a
  server — an irreconcilable conflict — so the Smithery-specific bundle
  and its publish job are removed. PyPI, the Official MCP Registry, and
  the GitHub Release MCPB asset remain the distribution path (#64).

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

[0.4.0]: https://github.com/astro-tools/astrodynamics-mcp/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/astro-tools/astrodynamics-mcp/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/astro-tools/astrodynamics-mcp/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/astro-tools/astrodynamics-mcp/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/astro-tools/astrodynamics-mcp/compare/v0.1.5...v0.2.0
[0.1.5]: https://github.com/astro-tools/astrodynamics-mcp/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/astro-tools/astrodynamics-mcp/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/astro-tools/astrodynamics-mcp/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/astro-tools/astrodynamics-mcp/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/astro-tools/astrodynamics-mcp/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/astro-tools/astrodynamics-mcp/releases/tag/v0.1.0
