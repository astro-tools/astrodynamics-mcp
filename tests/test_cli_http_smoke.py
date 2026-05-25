"""End-to-end CLI smoke test: spawn `astrodynamics-mcp http`, hit it with an MCP client, SIGINT.

Gated behind the `integration` marker — real subprocess + socket bind. CI
runs it via the `integration or not integration` selector; local default
`pytest` skips it.

On POSIX we send SIGINT and assert exit code 130 (the SIGINT convention).
On Windows `signal.SIGINT` to a subprocess group isn't reliably honoured
by uvicorn, so we fall back to `terminate()` and accept any clean exit.
"""

from __future__ import annotations

import asyncio
import os
import signal
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
    """Poll until *host:port* accepts a TCP connection or *timeout_s* elapses."""
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


async def _mcp_tools_list(url: str) -> dict[str, Any]:
    """Drive an MCP `tools/list` request via the official Python client SDK."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with (
        streamable_http_client(url) as (read, write, _get_session_id),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        result = await session.list_tools()
        return result.model_dump()


@pytest.mark.integration
class TestHttpServerEndToEnd:
    def test_http_serves_tools_list_and_exits_cleanly(self) -> None:
        port = _free_port()
        proc = subprocess.Popen(
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

            # Drive tools/list via the real MCP client SDK over Streamable HTTP.
            url = f"http://127.0.0.1:{port}/mcp"
            result = asyncio.run(_mcp_tools_list(url))

            # v0.1 surface is empty; the structural shape is what we assert.
            assert "tools" in result
            assert isinstance(result["tools"], list)
        finally:
            if sys.platform == "win32":
                proc.terminate()
            else:
                proc.send_signal(signal.SIGINT)
            try:
                rc = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
                raise

        if sys.platform != "win32":
            # 130 = 128 + SIGINT(2); the conventional clean-shutdown code for Ctrl-C.
            assert rc == 130, f"expected exit 130 on SIGINT, got {rc}"

    def test_stdio_exits_cleanly_on_sigint(self) -> None:
        """`astrodynamics-mcp stdio` blocks on stdin; SIGINT exits 130 (POSIX)."""
        if sys.platform == "win32":
            pytest.skip("SIGINT-to-subprocess unreliable on Windows; covered by unit test")

        proc = subprocess.Popen(
            [sys.executable, "-m", "astrodynamics_mcp.cli", "stdio"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
        )
        try:
            # Give the process a moment to enter mcp.run().
            time.sleep(1.5)
            assert proc.poll() is None, "stdio subcommand exited before SIGINT"
        finally:
            os.kill(proc.pid, signal.SIGINT)
            try:
                rc = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
                raise
        assert rc == 130, f"expected exit 130 on SIGINT, got {rc}"
