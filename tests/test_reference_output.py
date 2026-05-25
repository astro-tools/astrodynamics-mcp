"""Reference-output regression test — diff each tool's response against a committed golden.

For each registered v0.1 tool, call it with the fixed sample input from
:data:`tests._sample_calls.SAMPLE_CALLS` and assert the response matches
the committed golden under ``tests/data/golden/<tool>.json`` to within a
generous floating-point tolerance.

When an upstream pin bump or a deliberate tool-output change shifts the
numerics, regenerate the goldens with
``uv run python tests/_regenerate_goldens.py`` and review the diff before
committing. The :mod:`tests.test_upstream_pins` anchor must move in
lockstep with any such regen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests._regenerate_goldens import _mask_volatile_fields
from tests._sample_calls import SAMPLE_CALLS, SampleCall

_GOLDEN_DIR = Path(__file__).resolve().parent / "data" / "golden"

# Cross-platform numerical tolerance for golden comparisons. Wide enough
# to absorb libm / erfa rounding drift between Linux and Windows on the
# astropy-backed paths (frame transforms, time conversions) without
# letting algorithm-level regressions slip through. Per-tool overrides
# below tighten the band where the math is exact.
_DEFAULT_ABS_TOL = 1e-6
_PER_TOOL_ABS_TOL: dict[str, float] = {
    # tle_lookup is pure JSON pass-through — no math, exact match.
    "tle_lookup": 0.0,
    # time_convert UTC → TAI is integer leap-seconds; ISO string match.
    "time_convert": 0.0,
    # frame_transform TEME → ICRF goes through astropy; allow libm drift.
    "frame_transform": 1e-3,
    # porkchop fan-out: Lambert numerics + ASCII grid; allow drift.
    "porkchop": 1e-3,
    # sgp4: pure C extension, deterministic across platforms.
    "sgp4_propagate": 1e-9,
    # lamberthub: deterministic numerics.
    "lambert_solve": 1e-6,
    # access_windows: skyfield numerics + epoch-string equality.
    "access_windows": 1e-3,
    # bplane: pure numpy.
    "bplane_target": 1e-9,
}


def _assert_equal_with_tolerance(
    actual: Any, expected: Any, *, abs_tol: float, path: str = ""
) -> None:
    """Recursively diff two JSON-shaped values with a per-leaf float tolerance."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path or '<root>'}: expected dict, got {type(actual)}"
        assert actual.keys() == expected.keys(), (
            f"{path or '<root>'}: key mismatch — "
            f"actual_only={set(actual) - set(expected)}, "
            f"expected_only={set(expected) - set(actual)}"
        )
        for key, expected_value in expected.items():
            _assert_equal_with_tolerance(
                actual[key],
                expected_value,
                abs_tol=abs_tol,
                path=f"{path}.{key}" if path else key,
            )
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected list, got {type(actual)}"
        assert len(actual) == len(expected), (
            f"{path}: list length mismatch — actual {len(actual)}, expected {len(expected)}"
        )
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _assert_equal_with_tolerance(a, e, abs_tol=abs_tol, path=f"{path}[{i}]")
    elif isinstance(expected, float):
        # bool is a subclass of int — guard so JSON True/False booleans
        # don't accidentally match against floats via pytest.approx.
        if isinstance(actual, bool):
            assert actual == expected, f"{path}: bool {actual} mismatch float {expected}"
        else:
            assert actual == pytest.approx(expected, abs=abs_tol), (
                f"{path}: float {actual} differs from {expected} (abs_tol={abs_tol})"
            )
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


@pytest.mark.parametrize("sample", SAMPLE_CALLS, ids=lambda s: s.tool_name)
async def test_tool_output_matches_golden(sample: SampleCall) -> None:
    """Live tool output must match the committed golden within tolerance."""
    golden_path = _GOLDEN_DIR / f"{sample.tool_name}.json"
    assert golden_path.is_file(), (
        f"missing reference golden {golden_path}; regenerate with "
        "`uv run python tests/_regenerate_goldens.py`"
    )

    with sample.setup():
        response = await sample.invoke()
    actual = _mask_volatile_fields(sample.tool_name, response.model_dump(mode="json"))
    expected = json.loads(golden_path.read_text())

    abs_tol = _PER_TOOL_ABS_TOL.get(sample.tool_name, _DEFAULT_ABS_TOL)
    _assert_equal_with_tolerance(actual, expected, abs_tol=abs_tol)


def test_every_sample_call_has_a_golden() -> None:
    """Every entry in SAMPLE_CALLS must have a corresponding golden file."""
    missing = [
        s.tool_name for s in SAMPLE_CALLS if not (_GOLDEN_DIR / f"{s.tool_name}.json").is_file()
    ]
    assert not missing, (
        f"missing goldens for {missing}; regenerate with "
        "`uv run python tests/_regenerate_goldens.py`"
    )
