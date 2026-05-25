"""MCP tool modules — one module per tool family. Tools register against `server.mcp` on import."""

from __future__ import annotations

# Side-effect imports — each module's @register_tool decorators attach
# the tool to astrodynamics_mcp.server.mcp at import time.
from astrodynamics_mcp.tools import tle as _tle  # noqa: F401
