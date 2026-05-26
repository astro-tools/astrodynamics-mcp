# Output shaping for grid- and series-shaped tools

Some MCP tools produce responses that are intrinsically large — a porkchop
grid scales with `samples_per_axis²`, an SGP4 propagation scales linearly
with the requested epoch count. On small-context LLM consumers (GH Models
Free Low tier `gpt-4.1-mini` is the eval gate's reference profile, with an
8000-token input cap on the *next* turn after a tool call) the full
payload from a 30×30 porkchop scan (~250 KB / ~63k tokens) or a 24-hour
1-minute SGP4 propagation (~64 KB / ~16k tokens) is enough on its own to
fail with `413 tokens_limit_reached` when the model is asked to interpret
the result.

The other v0.1 tools (`bplane_target`, `lambert_*`, `tle_lookup`,
`time_convert`, `frame_transform`, `access_windows`, single-state
`sgp4`) stay comfortably under that cap — none exceed ~6 KB per call —
so the convention below applies only to tools whose output is
intrinsically grid-shaped or series-shaped.

## Convention

Tools that can produce large responses expose an `output` parameter:

```python
output: Literal["summary", "full"] = "summary"
```

`summary` is the default. The response carries the parts a small-context
model needs to answer the canonical question — the best/minimum cell, a
compact ASCII visual, a small representative sample — and omits the bulk
of the raw grid or series.

`full` returns the complete shape. Callers opt into it when they need
every cell (e.g. cookbook recipes that drive their own analysis from the
raw numbers, or downstream batch workflows that can absorb the bytes).

The schema for the two modes is intentionally distinct (a discriminated
union, not a nullable `grid` field) so neither mode wastes bytes on the
other's data.

## How each grid/series tool applies the convention

### `porkchop`

| Mode | Response fields |
|------|----------------|
| `summary` (default) | `best`, `top_cells` (≤ 5 by `total_dv` asc, `best` first), `ascii_summary` |
| `full` | `grid` (every feasible cell, row-major), `best`, `ascii_summary` |

The grid sweep is still performed in summary mode — the Horizons fetch
and the Lambert solves dominate cost; the response trim is free. Callers
that need every cell pass `output="full"` on the same request and
receive the unsubsampled payload.

### `sgp4_propagate`

| Mode | Response shape |
|------|---------------|
| `summary` (default) | `states: list[StateVector]` capped at 12 evenly-spaced entries (first and last requested epoch always retained) |
| `full` | `states: list[StateVector]` with one entry per requested epoch |

Every requested epoch is propagated in both modes; summary mode only
drops entries at serialisation. Calling the same prompt twice with
matching `epochs` and switching `output` returns numerically identical
state vectors at the kept epochs — summary is a strict subset of full.

## Future grid/series tools

New tools whose output is grid-shaped (sweeps), series-shaped (per-epoch
samples), or otherwise intrinsically scales beyond ~2 KB should follow
the same convention: `output: Literal["summary", "full"] = "summary"`,
discriminated union for the two response shapes, summary chosen so a
~2000-token cap is comfortably respected.

A future option — MCP resource attachment — could let `output="full"`
attach the bulk payload as a separate resource the client fetches on
demand, rather than carrying it inline.
