"""Shared pytest fixtures for the astrodynamics-mcp test suite."""

from __future__ import annotations

import pytest
from mcp.server.fastmcp import FastMCP

from tests._gmat_helpers import make_fresh_mcp


@pytest.fixture
def gmat_mcp(monkeypatch: pytest.MonkeyPatch) -> FastMCP:
    """Per-test fresh FastMCP with the five GMAT tools registered.

    Replaces the per-file ``_fresh_mcp(monkeypatch)`` helper for new
    tests; the existing call sites continue to delegate through
    :func:`tests._gmat_helpers.make_fresh_mcp`.
    """
    return make_fresh_mcp("gmat-test", monkeypatch)


@pytest.fixture
def gmat_mcp_bare(monkeypatch: pytest.MonkeyPatch) -> FastMCP:
    """Per-test fresh FastMCP with the GMAT guard flipped, no tools registered.

    Use for tests that exercise the registration helpers themselves --
    e.g. asserting that ``_register_gmat_tools()`` is idempotent, or
    that ``_register_gmat_resources()`` raises on a missing skeleton
    file.
    """
    return make_fresh_mcp("gmat-test-bare", monkeypatch, register_tools=False)
