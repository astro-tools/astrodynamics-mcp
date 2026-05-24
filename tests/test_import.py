"""Smoke test: the package imports, exposes a non-empty version string, and the CLI runs."""

from __future__ import annotations

import pytest

import astrodynamics_mcp
from astrodynamics_mcp.cli import main


def test_import() -> None:
    assert isinstance(astrodynamics_mcp.__version__, str)
    assert astrodynamics_mcp.__version__


def test_cli_help_resolves(capsys: pytest.CaptureFixture[str]) -> None:
    """`astrodynamics-mcp --help` must resolve to the stub CLI per issue #1 acceptance."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "astrodynamics-mcp" in out
    assert "stdio" in out
    assert "http" in out


def test_cli_no_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Bare `astrodynamics-mcp` (no subcommand) prints help and exits 0."""
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "stdio" in out
    assert "http" in out


def test_cli_stdio_stub_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    """The stdio subcommand is a placeholder until #8 — exits non-zero with a clear message."""
    rc = main(["stdio"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "not yet implemented" in err


def test_cli_http_stub_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    """The http subcommand is a placeholder until #8 — exits non-zero with a clear message."""
    rc = main(["http", "--port", "9999"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "not yet implemented" in err
