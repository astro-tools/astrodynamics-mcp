"""The single ``FastMCP`` server instance plus the error-translating tool decorator.

Tool modules register against :data:`mcp` via :func:`register_tool` (not
``mcp.tool`` directly) so any :class:`~astrodynamics_mcp.errors.AstrodynamicsMCPError`
raised in a tool body surfaces on the MCP wire as a parseable JSON
envelope carrying our stable string error code. Other unhandled
exceptions are wrapped in an :class:`UpstreamError` so the LLM consumer
always sees one of our typed codes — never a leaked Python traceback.
"""

from __future__ import annotations

import asyncio
import functools
import json
from collections.abc import Callable
from typing import Any, TypeVar

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from astrodynamics_mcp.errors import AstrodynamicsMCPError, UpstreamError

_SERVER_NAME = "astrodynamics-mcp"
_SERVER_INSTRUCTIONS = (
    "Authoritative astrodynamics tools for any MCP client. Pass real numerical "
    "values with explicit units; the server speaks SI-adjacent astrodynamics "
    "units (km, km/s, deg, s, UTC ISO 8601 epochs). Failures surface as JSON "
    "error envelopes with stable string codes — see docs for the taxonomy."
)

mcp: FastMCP = FastMCP(_SERVER_NAME, instructions=_SERVER_INSTRUCTIONS)
"""Module-level singleton — tool modules register against this instance on import."""


_F = TypeVar("_F", bound=Callable[..., Any])


def _envelope_to_tool_error(err: AstrodynamicsMCPError) -> ToolError:
    """Serialise our typed error onto a :class:`ToolError`.

    FastMCP's ``ToolError(msg)`` lands ``msg`` in the MCP protocol error's
    ``message`` field. We JSON-encode the full ``{code, message, data}``
    envelope so structured consumers (and the LLM, with prompting) can
    extract the typed code from the message body.
    """
    return ToolError(json.dumps(err.to_mcp_error(), separators=(",", ":")))


def _wrap_unexpected(exc: BaseException) -> AstrodynamicsMCPError:
    return UpstreamError(
        f"unexpected error in tool: {exc!r}",
        code="upstream.unexpected_exception",
        original_exception=exc,
    )


def register_tool(
    *,
    name: str | None = None,
    title: str | None = None,
    description: str | None = None,
    structured_output: bool | None = None,
) -> Callable[[_F], _F]:
    """Register a tool against the module-level :data:`mcp` instance.

    Wraps the inner function so that:

    - :class:`AstrodynamicsMCPError` subclasses serialise to a ``ToolError``
      carrying the JSON envelope (typed string ``code`` preserved).
    - Other exceptions are caught and converted to
      :class:`UpstreamError` (``code="upstream.unexpected_exception"``) — the
      LLM consumer always sees one of our typed codes.

    Both sync and async tool functions are supported; the wrapper picks
    the right branch from :func:`asyncio.iscoroutinefunction`.
    """

    def decorator(fn: _F) -> _F:
        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapped(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await fn(*args, **kwargs)
                except AstrodynamicsMCPError as err:
                    raise _envelope_to_tool_error(err) from err
                except Exception as exc:
                    raise _envelope_to_tool_error(_wrap_unexpected(exc)) from exc

            registered = mcp.tool(
                name=name,
                title=title,
                description=description,
                structured_output=structured_output,
            )(async_wrapped)
        else:

            @functools.wraps(fn)
            def sync_wrapped(*args: Any, **kwargs: Any) -> Any:
                try:
                    return fn(*args, **kwargs)
                except AstrodynamicsMCPError as err:
                    raise _envelope_to_tool_error(err) from err
                except Exception as exc:
                    raise _envelope_to_tool_error(_wrap_unexpected(exc)) from exc

            registered = mcp.tool(
                name=name,
                title=title,
                description=description,
                structured_output=structured_output,
            )(sync_wrapped)
        return registered  # type: ignore[return-value]

    return decorator
