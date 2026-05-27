"""Dump astrodynamics-mcp tool definitions as a JSON array for the MCPB manifest.

Each entry carries name, description, and the real `inputSchema` JSON Schema
that FastMCP derives from the tool function's signature and pydantic types.
Output goes to stdout; the release workflow `jq`-injects it into
`packaging/mcpb*/manifest.json` before `mcpb pack`.

The MCPB spec at `anthropics/mcpb` allows `tools[]` entries to be the minimal
`{name, description}`. Smithery's listing API, however, type-checks entries as
full MCP-protocol Tool objects (`inputSchema` required). Generating from the
live server bridges that gap and keeps the catalog in sync with the code.
"""

from __future__ import annotations

import asyncio
import json
import sys

from astrodynamics_mcp.server import mcp


async def main() -> int:
    tools = await mcp.list_tools()
    out = [
        {
            "name": t.name,
            "description": t.description or "",
            "inputSchema": t.inputSchema,
        }
        for t in tools
    ]
    json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
