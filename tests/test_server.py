"""Tests for `astrodynamics_mcp.server`."""

from __future__ import annotations

import json
from typing import Any

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from astrodynamics_mcp.errors import (
    AstrodynamicsMCPError,
    DataSourceError,
    InvalidInputError,
    UpstreamError,
)
from astrodynamics_mcp.server import (
    _envelope_to_tool_error,
    _wrap_unexpected,
    mcp,
    register_tool,
)


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
