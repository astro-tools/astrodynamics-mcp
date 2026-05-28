"""The single ``FastMCP`` server instance plus the error-translating tool decorator.

Tool modules register against :data:`mcp` via :func:`register_tool` (not
``mcp.tool`` directly) so any :class:`~astrodynamics_mcp.errors.AstrodynamicsMCPError`
raised in a tool body surfaces on the MCP wire as a parseable JSON
envelope carrying our stable string error code. Other unhandled
exceptions are wrapped in an :class:`UpstreamError` so the LLM consumer
always sees one of our typed codes — never a leaked Python traceback.

Wire-format error contract
--------------------------
FastMCP's default ``call_tool`` path turns any error into a
``CallToolResult(isError=True)`` whose only content is the string
``"Error executing tool <name>: <message>"`` — and crucially it does this
*after* argument validation, so a typed error raised inside a pydantic
validator never reaches :func:`register_tool`'s wrapper and the stable
``code`` is lost. We install our own low-level ``call_tool`` handler
(:func:`_wire_call_tool`) that delegates to FastMCP, resolves the real
:class:`AstrodynamicsMCPError` out of the wrapped exception's ``__cause__``
chain (covering both tool-body errors and argument-validation errors),
and returns a ``CallToolResult`` carrying the ``{code, message, data}``
envelope in both ``structuredContent`` (a real structured channel) and a
JSON ``TextContent`` block — with no ``"Error executing tool"`` prefix.

The :meth:`FastMCP.call_tool` *method* still raises ``ToolError`` as before;
only the wire-facing low-level handler is replaced, so in-process callers
(and the existing method-level tests) are unaffected.
"""

from __future__ import annotations

import asyncio
import functools
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import ValidationError

from astrodynamics_mcp.errors import AstrodynamicsMCPError, InvalidInputError, UpstreamError

_SERVER_NAME = "astrodynamics-mcp"
_SERVER_INSTRUCTIONS = (
    "Authoritative astrodynamics tools for any MCP client. Pass real numerical "
    "values with explicit units; the server speaks SI-adjacent astrodynamics "
    "units (km, km/s, deg, s, UTC ISO 8601 epochs). Failures surface as JSON "
    "error envelopes with stable string codes — see docs for the taxonomy."
)


@asynccontextmanager
async def _lifespan(_server: FastMCP) -> AsyncIterator[None]:
    """Release long-lived data-adapter clients on server shutdown.

    The Space-Track and DISCOSweb adapters keep a module-level
    :class:`httpx.AsyncClient` singleton alive across tool calls (to preserve
    the Space-Track session cookie and keep both HTTPS connection pools warm).
    Closing them here — inside the running event loop, after request handling
    stops — releases their sockets / SSL contexts instead of leaking them on
    shutdown. Imported lazily so importing :mod:`server` does not pull the
    data layer (and its transitive deps) at registration time.
    """
    from astrodynamics_mcp.data import discosweb, spacetrack

    try:
        yield
    finally:
        await spacetrack.aclose()
        await discosweb.aclose()


mcp: FastMCP = FastMCP(_SERVER_NAME, instructions=_SERVER_INSTRUCTIONS, lifespan=_lifespan)
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
    annotations: ToolAnnotations | None = None,
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
                annotations=annotations,
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
                annotations=annotations,
            )(sync_wrapped)
        return registered  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Wire-facing error serialisation
# ---------------------------------------------------------------------------

# Cap the __cause__ walk so a pathological self-referential chain can't loop.
_MAX_CAUSE_DEPTH = 16


def _validation_to_invalid_input(exc: ValidationError) -> InvalidInputError:
    """Map a pydantic ``ValidationError`` (argument schema failure) to a typed code.

    These come from FastMCP validating the call arguments *before* the tool
    body runs, so they never carry one of our codes on their own. The first
    error's location + message is enough for an LLM caller to repair the call.
    """
    errors = exc.errors()
    if errors:
        first = errors[0]
        loc = ".".join(str(part) for part in first.get("loc", ()))
        detail = str(first.get("msg", ""))
    else:
        loc = ""
        detail = str(exc)
    where = f" for {loc!r}" if loc else ""
    message = f"argument validation failed{where}: {detail}"
    return InvalidInputError(
        message,
        code="invalid_input.schema_validation",
        data={"error_count": len(errors)},
    )


def _resolve_error(exc: BaseException) -> AstrodynamicsMCPError:
    """Resolve the typed error behind a FastMCP-wrapped ``ToolError``.

    FastMCP re-wraps everything (tool-body errors *and* argument-validation
    failures) into a generic ``ToolError`` via ``raise ... from e``, so the
    real error is reachable through the ``__cause__`` chain. Walk it for the
    first :class:`AstrodynamicsMCPError` (our typed errors) or pydantic
    :class:`ValidationError` (raw schema failures); anything else is an
    unexpected exception that becomes ``upstream.unexpected_exception``.
    """
    cause: BaseException | None = exc
    original: BaseException = exc
    for _ in range(_MAX_CAUSE_DEPTH):
        if cause is None:
            break
        if isinstance(cause, AstrodynamicsMCPError):
            return cause
        if isinstance(cause, ValidationError):
            return _validation_to_invalid_input(cause)
        original = cause
        cause = cause.__cause__
    return _wrap_unexpected(original)


def _error_call_result(err: AstrodynamicsMCPError) -> CallToolResult:
    """Build the wire ``CallToolResult`` carrying our typed error envelope.

    The ``{code, message, data}`` envelope rides in ``structuredContent``
    (the structured channel a programmatic consumer reads) *and* as a JSON
    ``TextContent`` block (so prompt-driven and text-only clients still see
    the code). No ``"Error executing tool"`` prefix is added.
    """
    payload = err.to_mcp_error()
    return CallToolResult(
        isError=True,
        content=[TextContent(type="text", text=json.dumps(payload, separators=(",", ":")))],
        structuredContent=payload,
    )


@mcp._mcp_server.call_tool(validate_input=False)  # type: ignore[untyped-decorator]
async def _wire_call_tool(name: str, arguments: dict[str, Any]) -> Any:
    """Low-level ``call_tool`` handler replacing FastMCP's default.

    Delegates to :meth:`FastMCP.call_tool` (which still raises ``ToolError``)
    and translates any failure into our structured-envelope result. Returning
    a ``CallToolResult`` here makes the low-level server pass it through
    verbatim — bypassing the default ``"Error executing tool"`` string. Only
    ``ToolError`` is caught; control-flow signals (e.g. elicitation) propagate.
    """
    try:
        return await mcp.call_tool(name, arguments)
    except ToolError as exc:
        return _error_call_result(_resolve_error(exc))
