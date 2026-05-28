"""Tests for `astrodynamics_mcp.server`."""

from __future__ import annotations

import json
from typing import Any

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult, TextContent

from astrodynamics_mcp.errors import (
    AstrodynamicsMCPError,
    DataSourceError,
    InvalidInputError,
    UpstreamError,
)
from astrodynamics_mcp.server import (
    _envelope_to_tool_error,
    _resolve_error,
    _validation_to_invalid_input,
    _wrap_unexpected,
    mcp,
    register_tool,
)


async def _wire_call(name: str, arguments: dict[str, Any]) -> CallToolResult:
    """Invoke a tool through the real low-level ``call_tool`` handler.

    This is the wire path an MCP client drives — distinct from the
    ``mcp.call_tool`` *method* (which still raises ``ToolError``). Tools are
    registered on the module singleton at import, so we use it directly.
    """
    import astrodynamics_mcp.tools  # noqa: F401  # registration side effect

    handler = mcp._mcp_server.request_handlers[CallToolRequest]
    request = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name=name, arguments=arguments),
    )
    result = await handler(request)
    assert isinstance(result.root, CallToolResult)
    return result.root


def _envelope_from_tool_error(excinfo: pytest.ExceptionInfo[ToolError]) -> dict[str, Any]:
    """Extract our JSON envelope from a FastMCP-wrapped ToolError.

    FastMCP wraps ``ToolError("X")`` into ``"Error executing tool <name>: X"``
    when surfaced by ``call_tool``. We strip the canonical prefix so the
    JSON envelope can be parsed.
    """
    raw = str(excinfo.value)
    prefix_end = raw.find(": ")
    assert prefix_end != -1, f"unexpected ToolError shape: {raw!r}"
    body = raw[prefix_end + 2 :]
    return json.loads(body)  # type: ignore[no-any-return]


class TestServerInstance:
    async def test_module_singleton_is_fastmcp(self) -> None:
        assert isinstance(mcp, FastMCP)

    async def test_v01_surface_contains_registered_tools(self) -> None:
        """The real module-level instance carries every tool registered on import."""
        tools = await mcp.list_tools()
        tool_names = {t.name for t in tools}
        assert "tle_lookup" in tool_names


class TestEnvelopeSerialisation:
    def test_invalid_input_error_envelope(self) -> None:
        err = InvalidInputError("bad arg", "invalid_input.unknown_unit", data={"got": "furlong"})
        tool_err = _envelope_to_tool_error(err)
        assert isinstance(tool_err, ToolError)
        envelope = json.loads(str(tool_err))
        assert envelope == {
            "code": "invalid_input.unknown_unit",
            "message": "bad arg",
            "data": {"got": "furlong"},
        }

    def test_upstream_error_envelope_round_trips(self) -> None:
        original = ValueError("upstream said no")
        err = UpstreamError(
            "wrapped",
            "upstream.failure",
            original_exception=original,
        )
        envelope = json.loads(str(_envelope_to_tool_error(err)))
        assert envelope["code"] == "upstream.failure"
        assert envelope["data"]["original_exception_type"] == "ValueError"

    def test_wrap_unexpected_uses_canonical_code(self) -> None:
        wrapped = _wrap_unexpected(RuntimeError("boom"))
        assert isinstance(wrapped, UpstreamError)
        assert wrapped.code == "upstream.unexpected_exception"
        assert wrapped.data["original_exception_type"] == "RuntimeError"


class TestRegisterToolAsync:
    @pytest.fixture(autouse=True)
    def _isolated_mcp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Swap in a fresh FastMCP per test so register_tool doesn't pollute the singleton."""
        fresh = FastMCP("test-async")
        monkeypatch.setattr("astrodynamics_mcp.server.mcp", fresh)

    async def test_happy_path_returns_value(self) -> None:
        @register_tool(name="add", description="Add two ints. e.g. add(1, 2) → 3")
        async def add(a: int, b: int) -> int:
            return a + b

        from astrodynamics_mcp import server as server_module

        # call_tool returns (content_list, structured_dict); for a primitive
        # return type the structured dict has the value under `result`.
        result = await server_module.mcp.call_tool("add", {"a": 1, "b": 2})
        assert isinstance(result, tuple)
        _content, structured = result
        assert isinstance(structured, dict)
        assert structured == {"result": 3}

    async def test_astrodynamics_mcp_error_translated(self) -> None:
        @register_tool(name="bad_input", description="Always errors. e.g. invalid_input.foo.")
        async def bad_input() -> int:
            raise InvalidInputError("missing arg", "invalid_input.missing_arg")

        from astrodynamics_mcp import server as server_module

        with pytest.raises(ToolError) as excinfo:
            await server_module.mcp.call_tool("bad_input", {})
        envelope = _envelope_from_tool_error(excinfo)
        assert envelope["code"] == "invalid_input.missing_arg"
        assert envelope["message"] == "missing arg"

    async def test_unexpected_exception_wrapped_as_upstream(self) -> None:
        @register_tool(name="unexpected", description="Raises a raw exception. e.g. KeyError.")
        async def unexpected() -> int:
            raise KeyError("missing key")

        from astrodynamics_mcp import server as server_module

        with pytest.raises(ToolError) as excinfo:
            await server_module.mcp.call_tool("unexpected", {})
        envelope = _envelope_from_tool_error(excinfo)
        assert envelope["code"] == "upstream.unexpected_exception"
        assert envelope["data"]["original_exception_type"] == "KeyError"

    async def test_data_source_error_translated(self) -> None:
        @register_tool(
            name="bad_source",
            description="Always fails the upstream. e.g. CelesTrak unreachable.",
        )
        async def bad_source() -> int:
            raise DataSourceError(
                "celestrak down",
                "data_source.celestrak_unreachable",
                source="celestrak",
            )

        from astrodynamics_mcp import server as server_module

        with pytest.raises(ToolError) as excinfo:
            await server_module.mcp.call_tool("bad_source", {})
        envelope = _envelope_from_tool_error(excinfo)
        assert envelope["code"] == "data_source.celestrak_unreachable"
        assert envelope["data"]["source"] == "celestrak"


class TestRegisterToolSync:
    """register_tool must handle plain (non-async) callables too."""

    @pytest.fixture(autouse=True)
    def _isolated_mcp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fresh = FastMCP("test-sync")
        monkeypatch.setattr("astrodynamics_mcp.server.mcp", fresh)

    async def test_sync_happy_path(self) -> None:
        @register_tool(name="double", description="Double an int. e.g. double(3) → 6.")
        def double(x: int) -> int:
            return x * 2

        from astrodynamics_mcp import server as server_module

        result = await server_module.mcp.call_tool("double", {"x": 21})
        assert isinstance(result, tuple)
        _content, structured = result
        assert isinstance(structured, dict)
        assert structured == {"result": 42}

    async def test_sync_astrodynamics_mcp_error_translated(self) -> None:
        @register_tool(name="sync_bad", description="Sync error. e.g. invalid_input.bad.")
        def sync_bad() -> int:
            raise InvalidInputError("nope", "invalid_input.bad")

        from astrodynamics_mcp import server as server_module

        with pytest.raises(ToolError) as excinfo:
            await server_module.mcp.call_tool("sync_bad", {})
        envelope = _envelope_from_tool_error(excinfo)
        assert envelope["code"] == "invalid_input.bad"

    async def test_sync_unexpected_wrapped(self) -> None:
        @register_tool(name="sync_boom", description="Sync KeyError. e.g. missing key.")
        def sync_boom() -> int:
            raise KeyError("missing")

        from astrodynamics_mcp import server as server_module

        with pytest.raises(ToolError) as excinfo:
            await server_module.mcp.call_tool("sync_boom", {})
        envelope = _envelope_from_tool_error(excinfo)
        assert envelope["code"] == "upstream.unexpected_exception"


class TestRoundTrip:
    """The envelope produced by register_tool's wrapper must JSON-parse cleanly."""

    @pytest.fixture(autouse=True)
    def _isolated_mcp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("astrodynamics_mcp.server.mcp", FastMCP("round-trip"))

    async def test_envelope_is_compact_json(self) -> None:
        @register_tool(name="errnow", description="Always errors. e.g. invalid_input.test.")
        async def errnow() -> int:
            raise InvalidInputError("test message", "invalid_input.test", data={"k": [1, 2]})

        from astrodynamics_mcp import server as server_module

        with pytest.raises(ToolError) as excinfo:
            await server_module.mcp.call_tool("errnow", {})
        envelope = _envelope_from_tool_error(excinfo)
        assert envelope["data"]["k"] == [1, 2]
        # The raw JSON inside the wrapped message must be compact (no spaces
        # between separators) — that's our serializer's discipline.
        raw = str(excinfo.value)
        body = raw[raw.find(": ") + 2 :]
        assert ", " not in body and ": " not in body


class TestSubclassChainBaseError:
    """A bare AstrodynamicsMCPError (not a subclass) still translates correctly."""

    @pytest.fixture(autouse=True)
    def _isolated_mcp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("astrodynamics_mcp.server.mcp", FastMCP("base-err"))

    async def test_root_error_translated(self) -> None:
        @register_tool(name="root_err", description="Raise root error. e.g. freeform.code.")
        async def root_err() -> int:
            raise AstrodynamicsMCPError("freeform failure", "freeform.code")

        from astrodynamics_mcp import server as server_module

        with pytest.raises(ToolError) as excinfo:
            await server_module.mcp.call_tool("root_err", {})
        envelope = _envelope_from_tool_error(excinfo)
        assert envelope["code"] == "freeform.code"


class TestResolveError:
    """Unit tests for the __cause__-chain resolver behind the wire handler."""

    def test_resolves_typed_error_through_wrapped_toolerror(self) -> None:
        typed = InvalidInputError("nope", "invalid_input.foo")
        wrapped = ToolError("Error executing tool x: ...")
        wrapped.__cause__ = typed
        assert _resolve_error(wrapped) is typed

    def test_resolves_typed_error_through_two_layers(self) -> None:
        # register_tool produces ToolError(json) from the typed error, then
        # Tool.run wraps that again — the resolver must reach two levels deep.
        typed = UpstreamError("boom", "upstream.failure")
        inner = ToolError("envelope-json")
        inner.__cause__ = typed
        outer = ToolError("Error executing tool x: envelope-json")
        outer.__cause__ = inner
        assert _resolve_error(outer) is typed

    def test_pydantic_validation_error_maps_to_schema_validation(self) -> None:
        from pydantic import BaseModel

        class _M(BaseModel):
            a: int

        try:
            _M(a="not-an-int")  # type: ignore[arg-type]
        except Exception as exc:  # pydantic.ValidationError
            wrapped = ToolError("Error executing tool x: ...")
            wrapped.__cause__ = exc
            resolved = _resolve_error(wrapped)
        assert resolved.code == "invalid_input.schema_validation"
        assert resolved.data["error_count"] >= 1

    def test_unknown_cause_becomes_unexpected_upstream(self) -> None:
        original = KeyError("missing")
        wrapped = ToolError("Error executing tool x: 'missing'")
        wrapped.__cause__ = original
        resolved = _resolve_error(wrapped)
        assert resolved.code == "upstream.unexpected_exception"
        assert resolved.data["original_exception_type"] == "KeyError"

    def test_validation_message_names_the_field(self) -> None:
        from pydantic import BaseModel

        class _M(BaseModel):
            epoch: int

        try:
            _M()  # type: ignore[call-arg]
        except Exception as exc:
            invalid = _validation_to_invalid_input(exc)  # type: ignore[arg-type]
        assert invalid.code == "invalid_input.schema_validation"
        assert "epoch" in invalid.message


class TestWireErrorContract:
    """End-to-end: the structured envelope reaches the wire via the low-level handler.

    These exercise the actual ``call_tool`` request handler (what stdio / HTTP
    clients drive), guarding the contract that issue #102 broke: a typed code
    must arrive in ``structuredContent`` with no ``Error executing tool`` prefix.
    """

    async def test_argument_validation_error_carries_typed_code(self) -> None:
        # Bare-date epoch fails inside a pydantic validator — historically this
        # lost the code entirely. It must now arrive on the wire.
        result = await _wire_call(
            "frame_transform",
            {
                "state": {
                    "r": [{"value": 7000.0, "unit": "km"}] * 3,
                    "v": [{"value": 1.0, "unit": "km/s"}] * 3,
                    "frame": "GCRS",
                    "epoch": "2026-01-01T00:00:00Z",
                },
                "to_frame": "ITRS",
                "epoch": "2026-01-01",  # bare date — invalid
            },
        )
        assert result.isError is True
        assert result.structuredContent is not None
        assert result.structuredContent["code"] == "invalid_input.epoch_missing_time_component"
        # The text block is the clean JSON envelope — no "Error executing tool" prefix.
        text_block = result.content[0]
        assert isinstance(text_block, TextContent)
        assert not text_block.text.startswith("Error executing tool")
        parsed = json.loads(text_block.text)
        assert parsed["code"] == "invalid_input.epoch_missing_time_component"

    async def test_missing_argument_maps_to_schema_validation(self) -> None:
        result = await _wire_call("frame_transform", {"to_frame": "ITRS"})
        assert result.isError is True
        assert result.structuredContent is not None
        assert result.structuredContent["code"] == "invalid_input.schema_validation"

    async def test_tool_body_error_carries_typed_code(self) -> None:
        # Degenerate Lambert geometry (r1 == r2) raises UpstreamError in the body.
        result = await _wire_call(
            "lambert_solve",
            {"r1": [7000.0, 0.0, 0.0], "r2": [7000.0, 0.0, 0.0], "tof": 3600.0, "mu": "earth"},
        )
        assert result.isError is True
        assert result.structuredContent is not None
        assert result.structuredContent["code"] == "upstream.lambert_no_solution"
        text_block = result.content[0]
        assert isinstance(text_block, TextContent)
        assert not text_block.text.startswith("Error executing tool")

    async def test_happy_path_is_not_an_error(self) -> None:
        result = await _wire_call(
            "time_convert",
            {"value": "2026-05-23T12:00:00", "from_scale": "UTC", "to_scale": "TAI"},
        )
        assert result.isError is not True
