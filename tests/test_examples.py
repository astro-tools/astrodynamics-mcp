"""CI smoke test for `examples/run_example_NN.py` scripts.

Each script drives an in-process MCP server with a fixed prompt
sequence and asserts the numerical output is within tolerance. This
test launches each script as a subprocess so the example scripts stay
first-class user-runnable artefacts; the test verifies they run
end-to-end and exit cleanly.

Gated behind the `integration` marker — the existing CI workflow runs
the full ``integration or not integration`` matrix on every PR, so the
gate fires per-PR without a separate workflow.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples"

_EXAMPLE_SCRIPTS = [
    "run_example_01_hohmann.py",
    "run_example_02_hubble_passes.py",
    "run_example_03_mars_launch_window.py",
]


@pytest.mark.integration
@pytest.mark.parametrize("script_name", _EXAMPLE_SCRIPTS)
def test_example_script_exits_cleanly(script_name: str) -> None:
    script_path = _EXAMPLES_DIR / script_name
    assert script_path.exists(), f"missing example script: {script_path}"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
        cwd=_REPO_ROOT,
    )

    if result.returncode != 0:
        pytest.fail(
            f"{script_name} exited {result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
