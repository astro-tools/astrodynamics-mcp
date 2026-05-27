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

## Claude Code

Claude Code reads MCP server definitions from its `mcp` settings. Add:

```json
{
  "mcpServers": {
    "astrodynamics": {
      "command": "astrodynamics-mcp",
      "args": ["stdio"]
    }
  }
}
```

Restart Claude Code. The registered tools (`tle_lookup`, `sgp4_propagate`,
`lambert_solve`, `access_windows`, `time_convert`, `frame_transform`,
`porkchop`, `bplane_target`) appear in the tool list.

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
    "astrodynamics": {
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
    Streamable HTTP exposes every tool to any caller that can reach the
    port. The server ships no built-in auth and the current tool surface
    does not execute arbitrary scripts — but bind to `127.0.0.1` (the
    default) unless you intentionally want the server reachable across
    the network, and put it behind your own auth proxy when you do.
    Sandboxing for arbitrary-script tools is one of the open design
    questions for the planned GMAT integration.
