"""MkDocs build-time hook that renders the live tool catalogue.

The tool reference page (``docs/tool-reference.md``) contains a single
``<!-- AUTOGEN:tool-reference -->`` marker. At build time we ask the
``FastMCP`` singleton for the registered tool list and substitute the
marker with one section per tool — name, LLM-facing description, and
the input / output JSON schemas the MCP wire actually carries.

Run side effects intentionally: importing :mod:`astrodynamics_mcp.tools`
triggers ``@register_tool`` registration on the shared ``mcp`` singleton.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

_MARKER = "<!-- AUTOGEN:tool-reference -->"


def _tool_records() -> list[Any]:
    """Return the live list of registered MCP tools."""
    # Import inside the hook so a missing dev dependency surfaces as a
    # build error on the docs page, not at MkDocs config-load time.
    from astrodynamics_mcp import tools as _tools_pkg  # noqa: F401  # registration side effect
    from astrodynamics_mcp.server import mcp

    return sorted(asyncio.run(mcp.list_tools()), key=lambda t: t.name)


def _render_schema(label: str, schema: dict[str, Any] | None) -> str:
    if not schema:
        return f"**{label}:** *(none)*\n"
    body = json.dumps(schema, indent=2, sort_keys=False)
    return f"**{label}:**\n\n```json\n{body}\n```\n"


def _render_tool(tool: Any) -> str:
    lines: list[str] = []
    lines.append(f"### `{tool.name}`")
    lines.append("")
    description = (tool.description or "").strip()
    if description:
        lines.append(description)
        lines.append("")
    lines.append(_render_schema("Input schema", tool.inputSchema))
    output_schema = getattr(tool, "outputSchema", None)
    lines.append(_render_schema("Output schema", output_schema))
    return "\n".join(lines)


def _render_catalogue() -> str:
    tools = _tool_records()
    sections = [_render_tool(t) for t in tools]
    return "\n".join(sections).rstrip() + "\n"


def on_page_markdown(
    markdown: str,
    *,
    page: Any,
    config: Any,
    files: Any,
) -> str:
    """MkDocs hook: substitute the AUTOGEN marker on the tool-reference page."""
    del page, config, files
    if _MARKER not in markdown:
        return markdown
    return markdown.replace(_MARKER, _render_catalogue())
