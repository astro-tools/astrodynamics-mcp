"""Transport-equivalence smoke test — stdio and HTTP yield identical tool payloads.

The MCP client SDK exposes the same ``call_tool`` surface over both
transports; this test spins up the real CLI subprocess in each mode and
checks that the structured content of one deterministic tool call is
byte-identical modulo session metadata.

Gated behind the ``integration`` marker — subprocess + socket bind on
HTTP, subprocess + pipe on stdio. CI's ``-m 'integration or not
integration'`` selector runs it; the default ``uv run pytest`` skips it.

Stdio is the trickier transport: the SIGINT-to-subprocess-reading-stdin
lifecycle is inconsistent across kernels (per the long note in
``test_cli_http_smoke.py``). We rely on the MCP SDK's ``stdio_client``
context manager to drive shutdown via stdin close + subprocess termination,
which is robust on WSL and on Windows; the Ubuntu-CI hang the smoke test
warns about is specific to SIGINT, not stdin EOF.
"""

from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
import time
from typing import Any

import pytest


def _free_port() -> int:
    """Reserve a random free port, release it, and return the number."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def _wait_for_bind(host: str, port: int, *, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                s.connect((host, port))
            except OSError:
                time.sleep(0.1)
                continue
            return
    raise TimeoutError(f"server failed to bind {host}:{port} within {timeout_s}s")


# A deterministic, offline tool call. Choosing sgp4_propagate so the
# transport equivalence test doesn't depend on a mocked network adapter
# (CelesTrak / Horizons) which would need separate per-subprocess wiring.
_TEST_TOOL = "sgp4_propagate"
_TEST_ARGS: dict[str, Any] = {
    "tle": {
        "line1": "1 25544U 98067A   24001.50000000  .00010000  00000-0  18000-3 0  9995",
        "line2": "2 25544  51.6400  90.0000 0001000  90.0000 270.0000 15.50000000    07",
    },
    "epochs": ["2024-01-01T12:00:00Z", "2024-01-01T12:10:00Z"],
    "frame": "TEME",
}


async def _call_via_http(port: int) -> dict[str, Any]:
    """Drive `tools/call` against `astrodynamics-mcp http` on *port*."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    url = f"http://127.0.0.1:{port}/mcp"
    async with (
        streamable_http_client(url) as (read, write, _get_session_id),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        result = await session.call_tool(_TEST_TOOL, _TEST_ARGS)
        return result.model_dump()


async def _call_via_stdio() -> dict[str, Any]:
    """Drive `tools/call` against `astrodynamics-mcp stdio` over a pipe."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "astrodynamics_mcp.cli", "stdio"],
    )
    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        result = await session.call_tool(_TEST_TOOL, _TEST_ARGS)
        return result.model_dump()


def _strip_session_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop fields that are legitimately allowed to vary between transports.

    The MCP client wraps the JSON-RPC ID and session-id when present;
    neither is part of the tool payload contract and both can vary
    between subprocess invocations. Tool output (``structuredContent``,
    ``content``, ``isError``) is what we care about.
    """
    return {
        "structuredContent": payload.get("structuredContent"),
        "content": payload.get("content"),
        "isError": payload.get("isError"),
    }


@pytest.mark.integration
class TestTransportEquivalence:
    def test_stdio_and_http_produce_identical_tool_payloads(self) -> None:
        port = _free_port()
        http_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "astrodynamics_mcp.cli",
                "http",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            _wait_for_bind("127.0.0.1", port)
            http_payload = asyncio.run(_call_via_http(port))
        finally:
            http_proc.terminate()
            try:
                http_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                http_proc.kill()
                http_proc.wait(timeout=5)

        stdio_payload = asyncio.run(_call_via_stdio())

        http_core = _strip_session_fields(http_payload)
        stdio_core = _strip_session_fields(stdio_payload)
        assert http_core == stdio_core, (
            "stdio and HTTP transports produced divergent tool payloads — "
            f"http={http_core!r}, stdio={stdio_core!r}"
        )

        # Sanity that the tool actually executed; we don't want both
        # transports to "agree" on returning an error.
        assert http_core.get("isError") is not True
        structured = http_core.get("structuredContent")
        assert isinstance(structured, dict)
        assert "states" in structured

    def test_tools_list_is_transport_independent(self) -> None:
        """tools/list returns the same registered tool surface over both transports."""
        port = _free_port()
        http_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "astrodynamics_mcp.cli",
                "http",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        async def _list_via_http() -> set[str]:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client

            url = f"http://127.0.0.1:{port}/mcp"
            async with (
                streamable_http_client(url) as (read, write, _get_session_id),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                result = await session.list_tools()
                return {tool.name for tool in result.tools}

        async def _list_via_stdio() -> set[str]:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "astrodynamics_mcp.cli", "stdio"],
            )
            async with (
                stdio_client(params) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                result = await session.list_tools()
                return {tool.name for tool in result.tools}

        try:
            _wait_for_bind("127.0.0.1", port)
            http_tools = asyncio.run(_list_via_http())
        finally:
            http_proc.terminate()
            try:
                http_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                http_proc.kill()
                http_proc.wait(timeout=5)

        stdio_tools = asyncio.run(_list_via_stdio())

        assert http_tools == stdio_tools, (
            f"tools/list diverges across transports — "
            f"only-in-http={http_tools - stdio_tools}, "
            f"only-in-stdio={stdio_tools - http_tools}"
        )
        # Sanity: the v0.1 surface is non-empty.
        assert http_tools, "expected at least one registered tool"
