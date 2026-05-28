"""Tests for the eval runner wiring (eval/tasks.py)."""

from __future__ import annotations

from typing import Any

import pytest
from eval import tasks


def test_server_spawned_with_full_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: the stdio server must inherit the eval job's environment.

    Without an explicit env, the MCP stdio client scrubs to a minimal
    allowlist that drops GMAT_ROOT (GMAT tools can't locate the install)
    and the ASTRODYNAMICS_MCP_* credential vars. Assert tasks.py passes the
    process environment through so both survive into the subprocess.
    """
    captured: dict[str, Any] = {}

    def fake_mcp_server_stdio(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(tasks, "mcp_server_stdio", fake_mcp_server_stdio)
    monkeypatch.setenv("GMAT_ROOT", "/sentinel/gmat")
    monkeypatch.setenv("ASTRODYNAMICS_MCP_DISCOSWEB_TOKEN", "sentinel-token")

    tasks.per_sample_react_solver()

    assert "env" in captured and captured["env"] is not None, (
        "mcp_server_stdio was called without an explicit env"
    )
    env = captured["env"]
    assert env.get("GMAT_ROOT") == "/sentinel/gmat"
    assert env.get("ASTRODYNAMICS_MCP_DISCOSWEB_TOKEN") == "sentinel-token"
