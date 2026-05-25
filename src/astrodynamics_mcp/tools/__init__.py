"""MCP tool modules — one module per tool family. Tools register against `server.mcp` on import."""

from __future__ import annotations

# Side-effect imports — each module's @register_tool decorators attach
# the tool to astrodynamics_mcp.server.mcp at import time.
from astrodynamics_mcp.tools import access as _access  # noqa: F401
from astrodynamics_mcp.tools import bplane as _bplane  # noqa: F401
from astrodynamics_mcp.tools import frames as _frames  # noqa: F401
from astrodynamics_mcp.tools import lambert as _lambert  # noqa: F401
from astrodynamics_mcp.tools import porkchop as _porkchop  # noqa: F401
from astrodynamics_mcp.tools import propagation as _propagation  # noqa: F401
from astrodynamics_mcp.tools import time as _time  # noqa: F401
from astrodynamics_mcp.tools import tle as _tle  # noqa: F401
