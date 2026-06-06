# Example 5 — ISS ground track (static plot, sequential)

A visualisation walk-through: propagate the ISS over one orbit, then
render its ground track as a PNG. Demonstrates the `sgp4_propagate` →
`plot_ground_track` chain and the **attachment** channel — the plot tool
returns a structured summary *and* a PNG image block alongside it.

The reproducible script lives at
[`examples/run_example_05_ground_track.py`](run_example_05_ground_track.py).
It exits 0 when the chain returns 31 ITRS states and the plot comes back
with a PNG `ImageContent` plus a summary carrying the image dimensions.

> **Requires the `[viz]` extra** (`pip install 'astrodynamics-mcp[viz]'`).
> The four `plot_* / czml_*` tools register only when matplotlib and
> gmat-czml are importable; on a base install they are absent and this
> example does not apply.

## Client config

```json
{
  "mcpServers": {
    "astrodynamics-mcp": {
      "command": "astrodynamics-mcp",
      "args": ["stdio"]
    }
  }
}
```

## Conversation

> **User:** Here's the current ISS TLE. Plot its ground track over one
> orbit.
> ```
> 1 25544U 98067A   24001.50000000  .00010000  00000-0  18000-3 0  9990
> 2 25544  51.6400  90.0000 0001000  90.0000 270.0000 15.50000000000000
> ```

A ground track is the sub-satellite latitude/longitude path, so the model
first propagates the TLE in the **Earth-fixed (ITRS)** frame — the rotating
frame whose x-y plane the sub-satellite point is read off — then hands the
state series to `plot_ground_track`. It samples one ~92-minute revolution
every 3 minutes (31 epochs) and asks for the full series (`output="full"`)
so the track is smooth.

```jsonc
// tools/call → sgp4_propagate
{
  "tle": {
    "line1": "1 25544U 98067A   24001.50000000  .00010000  00000-0  18000-3 0  9990",
    "line2": "2 25544  51.6400  90.0000 0001000  90.0000 270.0000 15.50000000000000"
  },
  "epochs": ["2024-01-01T00:00:00Z", "2024-01-01T00:03:00Z", "… 31 epochs …", "2024-01-01T01:30:00Z"],
  "frame": "ITRS",
  "output": "full"
}
```

The response carries 31 ITRS states. The first:

```jsonc
{
  "states": [
    {
      "r": {"value": [778.18, 4144.97, 5317.76], "unit": "km"},
      "v": {"value": [-7.2697, 1.1172, 0.1925],  "unit": "km/s"},
      "frame": "ITRS",
      "epoch": "2024-01-01T00:00:00Z"
    }
    // … 30 more …
  ]
}
```

The model then feeds that `states` series straight into the plot tool:

```jsonc
// tools/call → plot_ground_track
{
  "states": [ /* the 31 ITRS states from sgp4_propagate */ ]
}
```

`plot_ground_track` returns **two content blocks plus a structured
summary**: a leading ASCII text block (what a text-only client reads), a
PNG `ImageContent` (the rendered track), and the `structuredContent`
below.

```jsonc
// structuredContent
{
  "revolutions": {"value": 1.0,     "unit": "1"},
  "lat_min":     {"value": -51.59,  "unit": "deg"},
  "lat_max":     {"value": 51.76,   "unit": "deg"},
  "lon_min":     {"value": -179.43, "unit": "deg"},
  "lon_max":     {"value": 173.60,  "unit": "deg"},
  "image":       {"width_px": 800, "height_px": 400, "format": "png"}
}
```

```jsonc
// content blocks (the attachment rides here, not in structuredContent)
[
  {"type": "text",  "text": "Ground track: 31 points, 1.00 revs; lat [-51.6, 51.8] deg, lon [-179.4, 173.6] deg. PNG attached."},
  {"type": "image", "mimeType": "image/png", "data": "iVBORw0KGgo… (base64 PNG)"}
]
```

The latitude extent `±51.6°` is exactly the orbit's inclination — the
ground track of a circular orbit reaches its inclination at the highest
and lowest points. An MCP client that renders images shows the PNG inline;
a text-only client still gets the summary line.

> **Model:** Here's the ISS ground track over one ~92-minute orbit (PNG
> attached). It covers one full revolution, reaching ±51.6° latitude — the
> orbit's inclination — and sweeps the full longitude band as Earth rotates
> beneath it.

## What this demonstrates

- The canonical `sgp4_propagate` → `plot_ground_track` visualisation chain:
  propagate in the Earth-fixed ITRS frame, then render.
- The **attachment channel** — a static-plot tool returns the PNG as an
  `ImageContent` block *alongside* a structured summary and an ASCII text
  block, so both image-capable and text-only clients get a useful answer.
- The structured summary (`revolutions`, latitude/longitude extent, image
  dimensions) lets a client describe or cross-check the plot without
  decoding the pixels.
