# Example 6 — CZML export for Cesium (3D view, sequential)

A visualisation walk-through: propagate the ISS over one orbit, then
export the trajectory as a **CZML** document for a Cesium 3D globe.
Demonstrates the `sgp4_propagate` → `czml_trajectory` chain and the
**embedded-resource** attachment — the CZML rides as an
`EmbeddedResource` (an `application/json` document) alongside a structured
summary.

The reproducible script lives at
[`examples/run_example_06_czml_export.py`](run_example_06_czml_export.py).
It exits 0 when the chain returns a CZML `EmbeddedResource` plus a summary
carrying non-zero packet / object counts.

> **Requires the `[viz]` extra** (`pip install 'astrodynamics-mcp[viz]'`).
> `czml_trajectory` wraps the gmat-czml sibling and registers only when
> gmat-czml (and matplotlib) are importable; on a base install it is
> absent and this example does not apply.

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

> **User:** Here's the current ISS TLE. Export one orbit as CZML so I can
> load it into Cesium.
> ```
> 1 25544U 98067A   24001.50000000  .00010000  00000-0  18000-3 0  9990
> 2 25544  51.6400  90.0000 0001000  90.0000 270.0000 15.50000000000000
> ```

CZML animates an object's position over time in an inertial frame, so the
model propagates the TLE in **TEME** (sgp4's native inertial frame, which
gmat-czml renders directly) and hands the state series to
`czml_trajectory`. It samples one ~92-minute revolution every 6 minutes
(16 epochs) with the full series so the path is smooth.

```jsonc
// tools/call → sgp4_propagate
{
  "tle": {
    "line1": "1 25544U 98067A   24001.50000000  .00010000  00000-0  18000-3 0  9990",
    "line2": "2 25544  51.6400  90.0000 0001000  90.0000 270.0000 15.50000000000000"
  },
  "epochs": ["2024-01-01T00:00:00Z", "2024-01-01T00:06:00Z", "… 16 epochs …", "2024-01-01T01:30:00Z"],
  "frame": "TEME",
  "output": "full"
}
```

The 16 TEME states feed straight into the export tool (the
`{r, v, frame, epoch}` shape is exactly what `czml_trajectory` consumes;
velocity is optional but carried here):

```jsonc
// tools/call → czml_trajectory
{
  "trajectory": [ /* the 16 TEME states from sgp4_propagate */ ],
  "style": "default"
}
```

`czml_trajectory` returns the structured summary below plus **two content
blocks**: a leading ASCII text block and the CZML `EmbeddedResource`.

```jsonc
// structuredContent
{
  "time_span":    {"value": 1.5, "unit": "hours"},
  "frame":        "TEME",
  "style":        "sat-default",
  "has_velocity": true,
  "resource": {
    "packet_count": 2,
    "object_count": 1,
    "contact_count": 0,
    "format": "czml",
    "media_type": "application/json",
    "uri": "czml://trajectory/satellite"
  }
}
```

```jsonc
// content blocks (the CZML document rides as the embedded resource)
[
  {"type": "text", "text": "CZML trajectory: 1 object over 1.50 h (TEME, style sat-default). 2 packets. Document attached."},
  {
    "type": "resource",
    "resource": {
      "uri": "czml://trajectory/satellite",
      "mimeType": "application/json",
      "text": "[{\"id\":\"document\",\"name\":\"gmat-czml\",\"version\":\"1.0\",\"clock\":{ … }},{\"id\":\"satellite\", … }]"
    }
  }
]
```

The document is a two-packet CZML stream: the preamble (with a `clock`
spanning the 90-minute window) and one `satellite` object whose
time-tagged Cartesian positions Cesium interpolates into an animated orbit.
A client saves the `resource.text` to a `.czml` file (or streams it to a
Cesium viewer) keyed by the stable `uri`.

> **Model:** Exported one ~92-minute ISS orbit as CZML (document attached,
> `czml://trajectory/satellite`). It's a 2-packet stream in the TEME frame —
> a preamble with a 90-minute clock and one animated satellite object. Drop
> the `.czml` into a Cesium viewer to play the orbit back.

## What this demonstrates

- The `sgp4_propagate` → `czml_trajectory` chain: propagate in an inertial
  frame, then export for a 3D view.
- The **embedded-resource attachment** — unlike the static-plot tools'
  PNG `ImageContent`, `czml_trajectory` returns the CZML as an
  `application/json` `EmbeddedResource` keyed by a stable `uri`, alongside
  the structured summary and ASCII text block.
- The summary's document cardinalities (`packet_count`, `object_count`,
  `contact_count`) and `time_span` let a client describe the export without
  parsing the CZML.
