# Pick a client

astrodynamics-mcp speaks the MCP protocol over either stdio or Streamable
HTTP. Most desktop chat clients run it over stdio — they spawn the
`astrodynamics-mcp` console script as a subprocess and talk to it on
stdin/stdout. Remote agents reach the same tool surface over HTTP.

The configs below assume the `astrodynamics-mcp` binary is on the client's
`PATH`. If you installed via `uv tool install` or `pipx`, that should be the
case automatically; check with `which astrodynamics-mcp`. If it isn't, use
the absolute path everywhere `astrodynamics-mcp` appears below.

## Install from a registry

If your client has a registry browser, the one-click install paths are
preferable to hand-editing JSON:

- **[Official MCP Registry](https://registry.modelcontextprotocol.io/v0/servers/io.github.astro-tools/astrodynamics-mcp)** —
  `io.github.astro-tools/astrodynamics-mcp`. Read by `mcp-cli`, GitHub
  Copilot's MCP UI, and any other registry-aware client.
- **Anthropic's Claude Desktop directory** — surfaces inside Claude
  Desktop's *Browse extensions* UI once the v0.1.1 bundle clears
  Anthropic's review queue.
- **[Cursor Directory](https://cursor.directory/plugins)** — surfaces inside
  Cursor's MCP picker once the v0.1.1 listing is approved.

For everything else — and for any client where the JSON config is the
fastest path — the per-client snippets below cover the same ground.

## Credential plumbing

Most tools need no credential. Two sources do — Space-Track (deeper TLE
records) and ESA DISCOSweb (`satellite_metadata`). For a stdio client the
credential travels as an environment variable on the server process, which
every client below sets the same way: an `env` block alongside `command` /
`args`. The keys follow `ASTRODYNAMICS_MCP_<SOURCE>_<FIELD>`:

```json
{
  "mcpServers": {
    "astrodynamics-mcp": {
      "command": "astrodynamics-mcp",
      "args": ["stdio"],
      "env": {
        "ASTRODYNAMICS_MCP_SPACETRACK_USERNAME": "alice@example.org",
        "ASTRODYNAMICS_MCP_SPACETRACK_PASSWORD": "...",
        "ASTRODYNAMICS_MCP_DISCOSWEB_TOKEN": "..."
      }
    }
  }
}
```

Claude Code and Cursor both read the `env` block from their `mcp.json`;
ChatGPT desktop exposes the same fields through its server-settings UI. A
remote HTTP agent does not use `env` — it sends the credential in the
`initialize` request's `_meta` block instead. Either way, the server reads
the credential once and never echoes it back to the model. The full matrix,
the `_meta` shape, and the security guarantees live in
[Credentials](credentials.md); skip this section entirely if you only use the
no-auth tools.

## Claude Code

Claude Code reads MCP server definitions from its `mcp` settings. Add:

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

Restart Claude Code. The no-auth core (`tle_lookup`, `sgp4_propagate`,
`lambert_solve`, `access_windows`, `time_convert`, `frame_transform`,
`porkchop`, `bplane_target`) plus `satellite_metadata` appears in the tool
list; the GMAT tools join it when the `[gmat]` extra is installed (see
[GMAT integration](gmat-integration.md)), the SPICE tools when the `[spice]`
extra is (see [SPICE integration](spice-integration.md)), and the
visualisation tools when the `[viz]` extra is (see
[Visualisation](visualisation.md)). The [tool reference](tool-reference.md) is
the live catalogue. Claude Code renders the `[viz]` plot tools' PNG output
inline; how other clients handle the image and CZML attachments varies — see
[Supported clients → Attachment rendering](supported-clients.md#attachment-rendering).

To enable verbose server-side logging while debugging, swap the `args`:

```json
"args": ["--log-level", "info", "stdio"]
```

The server writes logs to stderr so they appear in Claude Code's MCP server
log pane without polluting the protocol stream.

## Cursor

Cursor's `~/.cursor/mcp.json` (or workspace-level `.cursor/mcp.json`) uses
the same shape as Claude Code:

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

Restart Cursor. The tools appear in the agent's tool list under the
`astrodynamics` server group.

## ChatGPT desktop

ChatGPT desktop reads MCP servers from its app settings. Use the same
`command` / `args` shape as the other stdio clients; the exact path through
the settings UI varies by ChatGPT desktop version.

```json
{
  "command": "astrodynamics-mcp",
  "args": ["stdio"]
}
```

## Raw Python (smoke client)

Useful for verifying the server end-to-end without a chat client in the
loop — and as the shape of an embedded usage from a custom agent.

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    params = StdioServerParameters(command="astrodynamics-mcp", args=["stdio"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("tools:", [t.name for t in tools.tools])

            result = await session.call_tool(
                "time_convert",
                arguments={
                    "value": "2026-05-23T12:00:00Z",
                    "from_scale": "UTC",
                    "to_scale": "TT",
                },
            )
            print(result.content)


asyncio.run(main())
```

## Streamable HTTP

For remote agents and any client that prefers HTTP over stdio. Start the
server with the `http` subcommand:

```bash
astrodynamics-mcp http --host 0.0.0.0 --port 8000
```

Then point the client at `http://<host>:8000/mcp` using the official MCP
SDK's Streamable HTTP transport. The tool surface is byte-for-byte the same
across stdio and HTTP — see the [transport-equivalence
section](eval-suite.md#how-this-is-validated) of the eval suite docs for
how this is enforced.

!!! warning "Trust boundary"
    Streamable HTTP exposes every registered tool to any caller that can
    reach the port. The server ships no built-in auth. When the `[gmat]`
    extra is installed, `gmat_run_mission` and `gmat_execute_script` run
    arbitrary GMAT scripts, so an exposed HTTP port is an arbitrary-code
    surface — bind to `127.0.0.1` (the default) unless you intentionally
    want the server reachable across the network, and put it behind your
    own auth proxy when you do. The operator owns this trust boundary; see
    [GMAT integration](gmat-integration.md#transports).
