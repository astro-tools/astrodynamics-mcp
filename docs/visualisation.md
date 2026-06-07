# Visualisation

The base tools answer astrodynamics questions in numbers and units. The
visualisation surface adds pictures: four tools that turn a state series or a
porkchop grid into a rendered artefact — a static PNG plot or a CZML document
for a Cesium 3D globe — while still carrying the numeric answer inline. An LLM
client can hand a user a ground-track image *and* the latitude band it spans,
in one tool call.

The viz tools live behind an optional extra. Install it and they appear in
`tools/list`; skip it and the rest of the tool surface is unaffected.

```bash
uv tool install 'astrodynamics-mcp[viz]'
# or
pipx install 'astrodynamics-mcp[viz]'
```

The extra pulls [`matplotlib`](https://matplotlib.org/) (the static-plot
backend) and the [`gmat-czml`](https://github.com/astro-tools/gmat-czml)
sibling (the CZML backend). The four tools register only when both are
importable — the same opt-in gate the `[gmat]` and `[spice]` extras use — so a
base install stays plot-free and pulls no rendering dependencies.

## The viz tools

The extra adds three static-plot tools that return a PNG and one export tool
that returns a CZML document:

- **`plot_ground_track`** — render a satellite's sub-satellite ground track as
  a PNG over a lon/lat graticule. Pass the whole state *series* (a single state
  cannot trace a track). States may be in any Earth frame — inertial frames are
  rotated to Earth-fixed internally, so the cheapest path is to propagate with
  `frame="ITRS"`. Optional ground stations are overlaid as markers.
- **`plot_trajectory`** — render an orbit or transfer arc as a 2D or 3D PNG
  about a central body. Positions are plotted in the states' own frame axes; a
  2D plot draws the central body to scale at the origin for known bodies, a 3D
  plot marks the origin without a sphere.
- **`plot_porkchop`** — render a porkchop C3 contour as a PNG from an existing
  grid result. It reuses the computed grid with no recompute, so call
  [`porkchop`](tool-reference.md) with `output="full"` first — a summary-only
  result has an empty grid and returns a typed error rather than an empty
  image. The minimum-Δv cell is marked on the contour.
- **`czml_trajectory`** — convert a trajectory into a
  [CZML](https://github.com/AnalyticalGraphicsInc/czml-writer/wiki/CZML-Guide)
  document for a Cesium 3D client, returned as an embedded resource. Pass the
  whole `{r, v, frame, epoch}` series; velocity is optional (a position-only
  series is rendered NaN-padded). Optional `intervals` annotate
  ground-station contacts as lines of sight shown only during each window.

Each tool's full input / output JSON schema — every argument, unit, and
default — lives on the [Tool reference](tool-reference.md) page, generated from
the live tool registry.

These tools render; they do not compute the trajectory. The canonical chain is
[`sgp4_propagate`](tool-reference.md) (or any tool that emits a state series) →
a viz tool. The two worked examples below follow exactly that shape.

## The attachment model

Every viz tool returns its picture as an **attachment** that rides *alongside*
the response, not instead of it. A single tool result carries three things:

1. **A structured summary** — the tool's declared response model
   (`structuredContent` on the wire). For `plot_ground_track` that is the
   revolution count and the latitude / longitude extent; for `czml_trajectory`
   it is the packet / object / contact counts and the time span. Every numeric
   field follows the same `{value, unit}` discipline as the rest of the surface
   (see [Data sources → Unit discipline](data-sources.md#unit-discipline)).
2. **A leading ASCII text block** — a one-line plain-text recap of the same
   summary.
3. **The attachment** — a PNG `ImageContent` for the three plot tools, or a
   CZML `EmbeddedResource` (an `application/json` document keyed by a stable
   `uri` such as `czml://trajectory/satellite`) for `czml_trajectory`.

The attachment is **strictly additive**. The summary and the ASCII line are
always present, so a client that cannot render images — or cannot render CZML —
is never left with nothing: it still reads the revolution count, the apsides,
the time span. A picture-capable client shows the picture *and* has the numbers
to describe or cross-check it without decoding pixels or parsing CZML.

This is the same trade the [output-shaping](output-shaping.md) convention makes
for grid- and series-shaped tools — keep a small, always-useful summary on the
wire and let the bulk ride in a channel the caller opts into — applied to
binary and resource output. The
[attachment-channel section](output-shaping.md#the-attachment-channel) covers
the wire shape in detail.

Because the picture is additive, an empty or single-state input never yields a
blank image: the plot and export tools return a typed error instead, the same
contract the rest of the surface follows.

## Frames

The two trajectory renderers care about frames in different ways.

`plot_ground_track` reads the sub-satellite point off the Earth-fixed frame, so
it accepts any Earth frame (TEME, ICRF, GCRS, CIRS, ITRS) and rotates inertial
input to Earth-fixed internally. Propagating directly in `ITRS` skips that
rotation — the cheapest path to a track — but any Earth frame produces the same
picture. A non-Earth frame is rejected with a typed error.

`czml_trajectory` renders in the frame Cesium animates the object in, and
accepts the CZML-renderable inertial and Earth-fixed frames — TEME, ICRF, GCRS
(mapped to CZML's GCRF), and ITRS (mapped to ITRF). All states in one call must
share a single frame. TEME is the natural choice for an SGP4-propagated orbit,
since it is sgp4's native output frame and needs no transform. Frames that CZML
cannot place — CIRS, TIRS, and the IAU body-fixed frames — are rejected with a
typed error.

`plot_trajectory` is frame-agnostic: it plots position components in whatever
axes the states arrive in, labelling them in their own units, and draws the
named `central_body` at the origin.

## Which clients render what

The MCP content blocks the viz tools emit are standard, but client support for
them varies — and the two attachment kinds are supported independently.

- **PNG `ImageContent`** — an image-capable client (Claude Code is the
  reference) shows the rendered plot inline. A text-only client shows the ASCII
  summary line and skips the image.
- **CZML `EmbeddedResource`** — CZML is a Cesium document format, not an image.
  A client that understands embedded resources can save the `resource.text` to
  a `.czml` file or stream it to a Cesium viewer, keyed by the stable `uri`; a
  client that does not still reads the structured summary. No mainstream chat
  client renders a CZML globe inline today — the document is meant to be handed
  to a Cesium viewer, which astrodynamics-mcp does not bundle (no web UI is in
  scope).

The [supported-clients matrix](supported-clients.md#attachment-rendering)
records which verified clients render each attachment kind. Whatever a client
does with the attachment, the additive summary guarantees it has a useful
answer either way.

## Worked examples

- [Plot a ground track](recipes.md#plot-a-satellite-ground-track) — the
  `sgp4_propagate` → `plot_ground_track` chain, with a reproducible script.
- [Export a trajectory as CZML](recipes.md#export-a-trajectory-as-czml) — the
  `sgp4_propagate` → `czml_trajectory` chain for a Cesium 3D view.
