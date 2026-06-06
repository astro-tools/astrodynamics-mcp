# Examples

Six runnable client sessions demonstrating the canonical
astrodynamics-mcp workflows.

| # | Tier | Question | Tool path |
|---|------|----------|-----------|
| [01](01_hohmann_dv.md) | single-tool | "Compute the Hohmann Δv from a 250 km circular LEO to GEO." | `lambert_solve` |
| [02](02_hubble_passes_madrid.md) | sequential | "Plot Hubble passes from Madrid for the next seven days." | `tle_lookup` → `access_windows` |
| [03](03_mars_launch_window_2028.md) | planning | "What's the best Mars launch window in 2028?" | `porkchop` (+ follow-up `lambert_solve`) |
| [04](04_spice_mars_state.md) | sequential | "Use SPICE to get Mars's state, then confirm a porkchop agrees on where Mars is." | `spice_load_kernel` → `spice_state` → `porkchop` |
| [05](05_ground_track_iss.md) | sequential | "Plot the ISS ground track over one orbit." | `sgp4_propagate` → `plot_ground_track` |
| [06](06_czml_export.md) | sequential | "Export one ISS orbit as CZML for Cesium." | `sgp4_propagate` → `czml_trajectory` |

Examples 05 and 06 are the **visualisation** sessions: they need the
`[viz]` extra (`pip install 'astrodynamics-mcp[viz]'`), which registers the
`plot_* / czml_*` tools. On a base install those tools are absent and the
two sessions do not apply.

## How each example is structured

Each example ships as a pair:

- `0N_<slug>.md` — the canonical conversation transcript. User prompt,
  the tool call(s) the LLM would issue, the JSON response, and the
  natural-language answer the LLM produces from it. Schemas are
  trimmed to the fields the LLM actually reads back; the
  [tool reference](../docs/tool-reference.md) carries the full input
  and output JSON schemas.
- `run_example_NN.py` — a reproducible Python script that drives the
  same tool sequence against an in-process MCP server, then asserts
  the numerical output is within tolerance. The scripts are exercised
  in CI as smoke tests via `tests/test_examples.py` (and, for the
  `[viz]`-gated sessions 05 / 06, `tests/test_examples_viz.py`).

## Running an example locally

```bash
uv run python examples/run_example_01_hohmann.py
uv run python examples/run_example_02_hubble_passes.py
uv run python examples/run_example_03_mars_launch_window.py
uv run python examples/run_example_04_spice_mars_state.py
# Sessions 05 / 06 need the [viz] extra (uv sync --extra viz):
uv run python examples/run_example_05_ground_track.py
uv run python examples/run_example_06_czml_export.py
```

Each script exits 0 on success and prints a short summary of the tool
output. Exit ≠ 0 means a tolerance assertion failed — investigate
before relying on the run.

## In-process driving

The run scripts drive the MCP server *in-process* via
`mcp.shared.memory.create_connected_server_and_client_session`, not as
a stdio subprocess. The wire-level tool surface is identical to the
production stdio path; the difference is that in-process driving lets
the script's CelesTrak / Horizons mocks reach the tool functions
without crossing a subprocess boundary.

Examples 02 (CelesTrak) and 03 (JPL Horizons) need deterministic
upstream responses to be reproducible — production behaviour against
live data is the same workflow but with results that change as the
upstream catalogues drift. Example 01 (Hohmann) is pure orbital
mechanics with no data source; it produces the same number every run
against any deployment shape.

Example 04 (SPICE / Horizons) additionally injects a small `spiceypy`
fake — the test environment ships no `[spice]` extra and no real
planetary SPK, so the SPICE half runs against a stand-in seeded with the
same synthetic Mars geometry the Horizons mock feeds `porkchop` (the
same approach the SPICE unit tests take). Against a live `[spice]`
install with the real kernels, the identical tool calls return CSPICE's
own ephemeris.

Examples 05 (ground track) and 06 (CZML) need **no** data-source mocks:
the ISS TLE is supplied inline, so `sgp4_propagate` and the visualisation
tools are fully deterministic. They do need the `[viz]` extra installed —
the run scripts assert the plot / CZML attachment came back, which only
happens when the `plot_* / czml_*` tools are registered. The
`tests/test_examples_viz.py` smoke test skips them when the extra is
absent.

## Linking out

These transcripts are the worked-example pairs for the recipes in
[`docs/recipes.md`](../docs/recipes.md). When adding a new recipe to
the docs site, check whether it warrants a worked example here and
cross-link both ways.
