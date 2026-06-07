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

The other tools (`bplane_target`, `lambert_*`, `tle_lookup`,
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

## The attachment channel

The `summary` / `full` split keeps a large payload off the wire by *omitting*
it. A second mechanism keeps a large payload off the *structured* response by
moving it into a side channel — an **attachment**. The
[visualisation tools](visualisation.md) use it: a rendered PNG or a CZML
document is bulky and not JSON-shaped, so it rides as an attachment beside the
structured summary rather than as a field inside it.

A single tool result can carry three things at once:

- **`structuredContent`** — the tool's declared response model, the same
  `{value, unit}`-disciplined JSON every other tool emits.
- **A leading `TextContent` block** — a one-line plain-text recap of the
  summary, so a text-only client always has something to read.
- **One or more attachment blocks** — an `ImageContent` (base64 PNG,
  `image/png`) for the static-plot tools, or an `EmbeddedResource` (an
  `application/json` CZML document keyed by a stable `uri`) for
  `czml_trajectory`.

```jsonc
// a plot tool's result on the wire
{
  "structuredContent": { "revolutions": {"value": 1.0, "unit": "1"}, "...": "..." },
  "content": [
    {"type": "text",  "text": "Ground track: 31 points, 1.00 revs; … PNG attached."},
    {"type": "image", "mimeType": "image/png", "data": "iVBORw0KGgo… (base64)"}
  ]
}
```

The defining property is that the attachment is **strictly additive**: the
structured summary and the ASCII line are always present, so a client that
cannot render the attachment — no image support, no CZML viewer — still gets a
complete, useful answer. The picture is never the *only* answer. This is the
same instinct as `summary` mode: never make the small-context or
limited-capability consumer pay for bulk it cannot use.

## Future grid/series tools

New tools whose output is grid-shaped (sweeps), series-shaped (per-epoch
samples), or otherwise intrinsically scales beyond ~2 KB should follow
the same convention: `output: Literal["summary", "full"] = "summary"`,
discriminated union for the two response shapes, summary chosen so a
~2000-token cap is comfortably respected.

The attachment channel above is the natural home for a future `output="full"`
that wants to ship the bulk grid out-of-band — as an `EmbeddedResource` the
client fetches on demand rather than an inline field. That is not wired today:
`output="full"` still carries the grid inline. The plumbing exists for the viz
payloads; extending it to the grid/series tools is a later step.
