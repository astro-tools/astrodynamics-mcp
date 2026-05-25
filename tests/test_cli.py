"""Tests for `astrodynamics_mcp.cli` — argparse wiring and subcommand dispatch."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from astrodynamics_mcp.cli import main


class TestArgparseWiring:
    def test_help_resolves(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["--help"])
        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        assert "astrodynamics-mcp" in out
        assert "stdio" in out
        assert "http" in out

    def test_version_resolves(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["--version"])
        assert excinfo.value.code == 0
        from astrodynamics_mcp import __version__

        assert __version__ in capsys.readouterr().out

    def test_no_args_prints_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main([])
        assert rc == 0
        assert "stdio" in capsys.readouterr().out


class TestStdioSubcommand:
    def test_invokes_mcp_run_with_stdio_transport(self) -> None:
        with patch("astrodynamics_mcp.server.mcp") as fake_mcp:
            rc = main(["stdio"])
        assert rc == 0
        fake_mcp.run.assert_called_once_with(transport="stdio")

    def test_keyboard_interrupt_returns_130(self) -> None:
        with patch("astrodynamics_mcp.server.mcp") as fake_mcp:
            fake_mcp.run.side_effect = KeyboardInterrupt()
            rc = main(["stdio"])
        assert rc == 130


class TestHttpSubcommand:
    def test_invokes_mcp_run_with_streamable_http(self) -> None:
        fake_settings: Any = MagicMock()
        with patch("astrodynamics_mcp.server.mcp") as fake_mcp:
            fake_mcp.settings = fake_settings
            rc = main(["http"])
        assert rc == 0
        # Settings updated to the defaults.
        assert fake_settings.host == "127.0.0.1"
        assert fake_settings.port == 8000
        fake_mcp.run.assert_called_once_with(transport="streamable-http")

    def test_custom_host_and_port_land_in_settings(self) -> None:
        fake_settings: Any = MagicMock()
        with patch("astrodynamics_mcp.server.mcp") as fake_mcp:
            fake_mcp.settings = fake_settings
            rc = main(["http", "--host", "0.0.0.0", "--port", "8765"])
        assert rc == 0
        assert fake_settings.host == "0.0.0.0"
        assert fake_settings.port == 8765

    def test_port_already_in_use_returns_exit_2_with_stderr_message(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with patch("astrodynamics_mcp.server.mcp") as fake_mcp:
            fake_mcp.settings = MagicMock()
            fake_mcp.run.side_effect = OSError("[Errno 98] Address already in use")
            rc = main(["http", "--port", "8000"])
        assert rc == 2
        err = capsys.readouterr().err
        assert "failed to bind" in err
        assert "8000" in err
        assert "Address already in use" in err

    def test_keyboard_interrupt_returns_130(self) -> None:
        with patch("astrodynamics_mcp.server.mcp") as fake_mcp:
            fake_mcp.settings = MagicMock()
            fake_mcp.run.side_effect = KeyboardInterrupt()
            rc = main(["http"])
        assert rc == 130


class TestLogLevel:
    @pytest.fixture(autouse=True)
    def _reset_logging(self) -> None:
        # basicConfig is a no-op if the root logger already has handlers, so
        # each test needs a fresh slate to assert level changes take effect.
        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)

    def test_default_is_warning(self) -> None:
        with patch("astrodynamics_mcp.server.mcp"):
            main(["stdio"])
        assert logging.getLogger().level == logging.WARNING

    def test_debug_flag_lowers_level(self) -> None:
        with patch("astrodynamics_mcp.server.mcp"):
            main(["--log-level", "debug", "stdio"])
        assert logging.getLogger().level == logging.DEBUG

    @pytest.mark.parametrize("level_name", ["debug", "info", "warning", "error"])
    def test_each_level_accepted(self, level_name: str) -> None:
        with patch("astrodynamics_mcp.server.mcp"):
            main(["--log-level", level_name, "stdio"])
        assert logging.getLogger().level == getattr(logging, level_name.upper())

    def test_invalid_level_rejected(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            main(["--log-level", "garbage", "stdio"])
        assert "invalid choice" in capsys.readouterr().err
