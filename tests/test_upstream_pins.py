"""Lock the upstream-library pins in pyproject to an explicit "last golden regen" anchor.

The reference-output goldens in ``tests/data/golden/`` are diffs against
numerical results produced by specific upstream-library versions. When
``pyproject.toml``'s floor for any of those libraries changes, the goldens
may diverge in ways that look like a regression but are really a
deliberate upstream bump. To force that bump to ride alongside the
golden regen, this module carries an explicit
:data:`PINS_AT_LAST_GOLDEN_REGEN` dict — the version of each library
pyproject pinned the last time the goldens were reviewed.

A pyproject change that does not also update :data:`PINS_AT_LAST_GOLDEN_REGEN`
fails this test loudly, with a message instructing the contributor to
regenerate the goldens and bump the anchor in the same PR.

The issue body also names ``interplanetary-porkchop`` but that is not a
dependency — porkchop is composed in-repo from ``lamberthub`` and the
Horizons adapter (see ``CONTRIBUTING.md``), so it has no upstream version
to pin.
"""

from __future__ import annotations

import importlib.metadata
import re
from pathlib import Path

import pytest

# The floor declared in pyproject.toml at the time the goldens under
# tests/data/golden/ were last regenerated and reviewed. Bump these in
# lockstep with the corresponding pyproject change AND a deliberate
# regeneration of the affected goldens.
PINS_AT_LAST_GOLDEN_REGEN: dict[str, str] = {
    "sgp4": "2.25",
    "lamberthub": "1.0",
    "skyfield": "1.54",
    "astropy": "6.1",
}


_PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"
_DEP_LINE_RE = re.compile(r'"(?P<pkg>[a-zA-Z0-9_.-]+)>=(?P<floor>\d+\.\d+(?:\.\d+)?)"')


def _read_pyproject_floors() -> dict[str, str]:
    """Return ``{package: floor}`` for every ``>=`` constraint in pyproject's deps."""
    text = _PYPROJECT_PATH.read_text(encoding="utf-8")
    return {m.group("pkg"): m.group("floor") for m in _DEP_LINE_RE.finditer(text)}


def _major_minor(version: str) -> str:
    parts = version.split(".")
    if len(parts) < 2:
        return version
    return f"{parts[0]}.{parts[1]}"


@pytest.mark.parametrize("package, expected_floor", sorted(PINS_AT_LAST_GOLDEN_REGEN.items()))
def test_pyproject_floor_matches_anchor(package: str, expected_floor: str) -> None:
    """pyproject's floor for *package* must equal the regen-anchor recorded here.

    Failure means pyproject was bumped without also bumping
    :data:`PINS_AT_LAST_GOLDEN_REGEN` and (almost certainly) regenerating
    the goldens. Fix by:

    1. Run the golden-regeneration script against the new environment.
    2. Review the diff under ``tests/data/golden/``.
    3. Update :data:`PINS_AT_LAST_GOLDEN_REGEN` in this file to the new floor.
    """
    floors = _read_pyproject_floors()
    actual_floor = floors.get(package)
    assert actual_floor is not None, (
        f"{package!r} is in PINS_AT_LAST_GOLDEN_REGEN but not in pyproject.toml; "
        "either restore the dependency entry or remove the anchor"
    )
    assert _major_minor(actual_floor) == _major_minor(expected_floor), (
        f"{package}: pyproject floor is {actual_floor} but the goldens were "
        f"last regenerated against {expected_floor}. Regenerate the goldens "
        "and bump PINS_AT_LAST_GOLDEN_REGEN in tests/test_upstream_pins.py."
    )


@pytest.mark.parametrize("package, expected_floor", sorted(PINS_AT_LAST_GOLDEN_REGEN.items()))
def test_installed_version_satisfies_floor(package: str, expected_floor: str) -> None:
    """The installed major.minor for *package* must be ≥ the regen-anchor floor.

    Catches the case where the lockfile was rolled back to a version
    older than the floor the goldens were generated against — those
    goldens would no longer reproduce. uv's per-Python split resolution
    means the installed minor on Python 3.10 may differ from 3.11+; the
    anchor encodes the lowest minor that must be available everywhere.
    """
    installed = importlib.metadata.version(package)
    installed_tuple = tuple(int(x) for x in installed.split(".")[:2])
    expected_tuple = tuple(int(x) for x in expected_floor.split(".")[:2])
    assert installed_tuple >= expected_tuple, (
        f"{package} installed is {installed} but the goldens require ≥ "
        f"{expected_floor}; refresh the environment with `uv sync`"
    )
