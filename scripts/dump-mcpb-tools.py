"""Dump astrodynamics-mcp tool definitions as a JSON array for the MCPB manifest.

Each entry carries `name`, `description`, `inputSchema`, `outputSchema` (when
FastMCP derives one), and `annotations` (`readOnlyHint`, `openWorldHint`, …)
— the full MCP-protocol Tool shape. Output goes to stdout; the release
workflow `jq`-injects it into `packaging/mcpb*/manifest.json` before pack.

The MCPB spec at `anthropics/mcpb` allows `tools[]` entries to be the minimal
`{name, description}`. Smithery's listing API, however, type-checks entries as
full MCP-protocol Tool objects and scores against the richer fields
(parameter descriptions inside `inputSchema`, plus `outputSchema` and
`annotations`). Generating from the live server bridges that gap and keeps
the catalog in sync with the code.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from astrodynamics_mcp.server import mcp


async def main() -> int:
    tools = await mcp.list_tools()
    out: list[dict[str, Any]] = []
    for t in tools:
        entry: dict[str, Any] = {
            "name": t.name,
            "description": t.description or "",
            "inputSchema": t.inputSchema,
        }
        if t.outputSchema is not None:
            entry["outputSchema"] = t.outputSchema
        if t.annotations is not None:
            ann = t.annotations.model_dump(exclude_none=True)
            if ann:
                entry["annotations"] = ann
        out.append(entry)
    json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
