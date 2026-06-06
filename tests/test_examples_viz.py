"""CI smoke test for the `[viz]`-gated example scripts (05 / 06).

Mirrors :mod:`tests.test_examples` but for the visualisation sessions,
which need the `[viz]` extra (matplotlib for the static plot, gmat-czml
for the CZML export). The module skips cleanly when the extra is absent:
the standard test job installs no extras, so these skip there; the
``[viz] extra install`` CI job installs ``.[viz]`` and runs them
end-to-end, the analog of how the SPICE / GMAT prompts skip without their
extras.

Each script drives the in-process MCP server with a fixed prompt sequence
and asserts the visualisation attachment came back (an ImageContent PNG
for the ground track, an EmbeddedResource CZML for the export). Launched
as a subprocess so the scripts stay first-class user-runnable artefacts.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib", reason="viz example scripts need the [viz] extra")
pytest.importorskip("gmat_czml", reason="viz example scripts need the [viz] extra")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples"

_VIZ_EXAMPLE_SCRIPTS = [
    "run_example_05_ground_track.py",
    "run_example_06_czml_export.py",
]


@pytest.mark.integration
@pytest.mark.parametrize("script_name", _VIZ_EXAMPLE_SCRIPTS)
def test_viz_example_script_exits_cleanly(script_name: str) -> None:
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
