"""Smoke test: the package imports and exposes a non-empty version string.

CLI tests live in test_cli.py; this file is just the import-time invariant.
"""

from __future__ import annotations

import astrodynamics_mcp


def test_import() -> None:
    assert isinstance(astrodynamics_mcp.__version__, str)
    assert astrodynamics_mcp.__version__
