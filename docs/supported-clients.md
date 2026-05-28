# Supported clients

astrodynamics-mcp is a standards-compliant MCP server — anything that
speaks the [Model Context Protocol](https://modelcontextprotocol.io)
either over stdio or Streamable HTTP can drive it. The matrix below
records which clients the maintainers have verified end-to-end.

## Matrix

| Client | Transport | Verified | Notes |
| --- | --- | --- | --- |
| Claude Code | stdio | ✅ Yes | Reference desktop integration. See [Pick a client → Claude Code](pick-a-client.md#claude-code). |
| Cursor | stdio | ✅ Yes | Same config shape as Claude Code. See [Pick a client → Cursor](pick-a-client.md#cursor). |
| ChatGPT desktop | stdio | ⏳ Expected to work | MCP support landed in late 2026; verified-working matrix entry pending. |
| Raw Python (`mcp` SDK) | stdio | ✅ Yes | Smoke client example in [Pick a client → Raw Python](pick-a-client.md#raw-python-smoke-client). |
| Remote agents | Streamable HTTP | ⏳ Expected to work | Stdio + HTTP share a single registered tool surface; transport-equivalence is exercised under `pytest`. Production deployments not yet documented. |
| LangGraph / AutoGen / CrewAI | (any) | ⏳ Expected to work | astrodynamics-mcp exposes tools; these frameworks consume MCP tools. A cookbook page lands in a later release. |

Legend:

- ✅ **Verified** — the maintainers have run the full base tool surface
  through this client and confirmed all calls round-trip. Reported in
  the project's release notes.
- ⏳ **Expected to work** — the client speaks compliant MCP; the
  maintainers have not yet exercised it. Reports of working setups
  welcome via [Discussions](https://github.com/orgs/astro-tools/discussions).

## Reporting a working setup

If you have astrodynamics-mcp working in a client not on this list, open
a Discussion in the
[astro-tools community space](https://github.com/orgs/astro-tools/discussions)
with:

- Client name and version.
- Transport used (stdio / HTTP).
- A snippet of your client config or initialisation code.
- Whether the tools appear and round-trip.

Verified entries land in the next release's matrix.

## Reporting a broken setup

If a tool call fails against an otherwise-compliant client, open an issue
with the [client-compatibility issue
template](https://github.com/astro-tools/astrodynamics-mcp/issues/new?template=client-compatibility.md)
when available, or a plain bug report otherwise. Include the failing
tool's input arguments and the error envelope the client received — the
typed error code (e.g. `upstream.sgp4_failure`,
`data_source.celestrak_unreachable`) is the fastest path to a diagnosis.
