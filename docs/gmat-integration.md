# GMAT integration

The GMAT mission-analysis surface is part of the v0.2 milestone. This page
documents the contract the integration lands against; subsequent GMAT work
fills it in.

## Tool surface

Four tools cover the GMAT use cases:

- `gmat_run_mission(script, overrides=..., select_outputs=...)` — the
  canonical "run my mission" verb. Curated input schema and rich tool
  description so the LLM picks it by default.
- `gmat_sweep(script, grid|samples|perturb, ...)` — parameter sweeps and
  Monte Carlo runs over a mission script. A tagged-union input picks
  between grid, samples, and perturbation modes.
- `gmat_execute_script(script_text)` — escape hatch for raw scripted
  output. The tool description points callers toward `gmat_run_mission`
  first; this tool exists for cases the curated surface does not cover.
- `gmat_validate_script(script_text)` — parses the script through GMAT
  without executing the mission sequence. Returns parse errors, unknown
  resources or fields, and the list of declared resources. Intended for
  use in a self-correction loop before `gmat_run_mission` — see
  [Script authoring](#script-authoring).

The four-tool split is the contract subsequent work builds against. It
can narrow or widen during implementation if the curated tools prove
too restrictive or the escape hatch redundant.

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
skeleton — Hohmann transfer, planetary flyby, station-keeping, and a
short list of similarly common patterns — has a stable URI and is
discoverable through the client's resource-listing flow. Callers read
a skeleton, edit it for their problem, and pass the edited script to
`gmat_run_mission`. The skeleton set starts small and grows as common
authoring gaps emerge.

## Transports

The GMAT tools register on both stdio and Streamable HTTP transports.
Operators own their HTTP deployment's trust boundary — auth proxy,
network controls, who can reach the port — and the GMAT subset uses
the same transport selection as the rest of the tool surface.

## Install

The GMAT tools live behind an optional extra:

```bash
uv tool install 'astrodynamics-mcp[gmat]'
# or
pipx install 'astrodynamics-mcp[gmat]'
```

Without the extra, the GMAT tools do not appear in `tools/list`. With
the extra installed, each tool registers when its underlying driver
imports successfully.
