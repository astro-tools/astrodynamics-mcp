"""astrodynamics-mcp: Model Context Protocol server exposing astrodynamics tools to LLM clients."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("astrodynamics-mcp")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
