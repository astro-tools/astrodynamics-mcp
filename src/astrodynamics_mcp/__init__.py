"""astrodynamics-mcp: Model Context Protocol server exposing astrodynamics tools to LLM clients."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("astrodynamics-mcp")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]

# Trigger every tool module's side-effect @register_tool decorators against
# astrodynamics_mcp.server.mcp. Imported last so __version__ resolution is
# unaffected if the tool layer ever raises during import.
from astrodynamics_mcp import tools as _tools  # noqa: F401
