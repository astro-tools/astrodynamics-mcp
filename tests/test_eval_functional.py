"""Unit tests for the eval-suite functional-answer engine."""

from __future__ import annotations

from typing import Any

import pytest
from eval._functional import FunctionalSpecError, evaluate_checks, validate_check


class TestValidate:
    @pytest.mark.parametrize(
        "check",
        [
            {"path": "$", "equals": 1},
            {"path": "$.foo", "equals": "bar"},
            {"path": "$.foo[0]", "in_range": [0, 100]},
            {"path": "$.foo", "l2_in_range": [0, 100]},
            {"path": "$.foo[*]", "all_equal": "km"},
            {"path": "$.foo", "length": 3},
            {"path": "$.foo", "length": {"min": 1}},
            {"path": "$.foo", "present": True},
            {"path": "$.foo", "case_insensitive_contains": "ISS"},
            {"path": "$.foo", "starts_with": "2026-"},
            {
                "path": "$.foo",
                "numeric_tolerance": {"expected": [1.0, 2.0], "abs": 0.01},
            },
        ],
    )
    def test_accepts(self, check: dict[str, Any]) -> None:
        validate_check(check)

    @pytest.mark.parametrize(
        ("check", "fragment"),
        [
            ({"path": "foo"}, "must start with '\\$'"),
            ({"path": "$.foo"}, "exactly one predicate key"),
            ({"equals": 1}, "missing required key 'path'"),
            ({"path": "$.foo", "unknown": 1}, "unknown functional predicate"),
            ({"path": "$.foo", "in_range": [1]}, "two-element"),
            ({"path": "$.foo", "l2_in_range": [1]}, "two-element"),
            ({"path": "$.foo", "length": "huge"}, "requires an int"),
            ({"path": "$.foo", "present": "yes"}, "requires a boolean"),
            ({"path": "$.foo", "starts_with": 1}, "requires a string"),
            (
                {"path": "$.foo", "numeric_tolerance": {"expected": 1.0}},
                "requires at least one of",
            ),
        ],
    )
    def test_rejects(self, check: Any, fragment: str) -> None:
        with pytest.raises(FunctionalSpecError, match=fragment):
            validate_check(check)


class TestEvaluate:
    def test_equals_pass(self) -> None:
        passed, reasons = evaluate_checks(
            {"frame": "ICRF"}, [{"path": "$.frame", "equals": "ICRF"}]
        )
        assert passed is True
        assert reasons == []

    def test_equals_fail(self) -> None:
        passed, reasons = evaluate_checks(
            {"frame": "ITRS"}, [{"path": "$.frame", "equals": "ICRF"}]
        )
        assert passed is False
        assert any("expected equals='ICRF'" in r for r in reasons)

    def test_path_not_found(self) -> None:
        passed, reasons = evaluate_checks({"foo": 1}, [{"path": "$.bar", "equals": 1}])
        assert passed is False
        assert any("not found" in r for r in reasons)

    def test_in_range(self) -> None:
        passed, _ = evaluate_checks(
            {"r_magnitude": 6700.0},
            [{"path": "$.r_magnitude", "in_range": [6500, 7500]}],
        )
        assert passed is True
        passed, _ = evaluate_checks(
            {"r_magnitude": 6000.0},
            [{"path": "$.r_magnitude", "in_range": [6500, 7500]}],
        )
        assert passed is False

    def test_l2_in_range_pass(self) -> None:
        passed, _ = evaluate_checks(
            {"r": [6700.0, 0.0, 0.0]},
            [{"path": "$.r", "l2_in_range": [6500, 7500]}],
        )
        assert passed is True

    def test_l2_in_range_fail(self) -> None:
        passed, reasons = evaluate_checks(
            {"r": [1000.0, 0.0, 0.0]},
            [{"path": "$.r", "l2_in_range": [6500, 7500]}],
        )
        assert passed is False
        assert any("|v|=1000.0" in r for r in reasons)

    def test_l2_in_range_requires_vector(self) -> None:
        passed, reasons = evaluate_checks(
            {"r": 6700.0},
            [{"path": "$.r", "l2_in_range": [6500, 7500]}],
        )
        assert passed is False
        assert any("requires a vector value" in r for r in reasons)

    def test_length_on_array(self) -> None:
        passed, _ = evaluate_checks({"states": [1, 2, 3]}, [{"path": "$.states", "length": 3}])
        assert passed is True

    def test_present(self) -> None:
        passed, _ = evaluate_checks(
            {"ut1_utc_seconds": 0.123}, [{"path": "$.ut1_utc_seconds", "present": True}]
        )
        assert passed is True
        passed, _ = evaluate_checks({}, [{"path": "$.ut1_utc_seconds", "present": True}])
        assert passed is False
        # Asserting absence works too:
        passed, _ = evaluate_checks({}, [{"path": "$.ut1_utc_seconds", "present": False}])
        assert passed is True

    def test_starts_with(self) -> None:
        passed, _ = evaluate_checks(
            {"value": "2026-05-23T12:00:37.000"},
            [{"path": "$.value", "starts_with": "2026-05-23T12:00:37"}],
        )
        assert passed is True

    def test_case_insensitive_contains(self) -> None:
        passed, _ = evaluate_checks(
            {"name": "ISS (ZARYA)"},
            [{"path": "$.name", "case_insensitive_contains": "iss"}],
        )
        assert passed is True

    def test_numeric_tolerance_vector(self) -> None:
        passed, _ = evaluate_checks(
            {"r": [7000.001, 0.0, 0.0]},
            [
                {
                    "path": "$.r",
                    "numeric_tolerance": {"expected": [7000.0, 0.0, 0.0], "abs": 0.01},
                }
            ],
        )
        assert passed is True

    def test_array_indexing(self) -> None:
        passed, _ = evaluate_checks(
            {"results": [{"norad_id": "25544"}]},
            [{"path": "$.results[0].norad_id", "equals": "25544"}],
        )
        assert passed is True

    def test_flat_all_equal(self) -> None:
        passed, _ = evaluate_checks(
            {"states": [{"r": {"unit": "km"}}, {"r": {"unit": "km"}}]},
            [{"path": "$.states[*].r.unit", "all_equal": "km"}],
        )
        assert passed is True
        passed, _ = evaluate_checks(
            {"states": [{"r": {"unit": "km"}}, {"r": {"unit": "m"}}]},
            [{"path": "$.states[*].r.unit", "all_equal": "km"}],
        )
        assert passed is False

    def test_flat_length(self) -> None:
        passed, _ = evaluate_checks(
            {"results": [1, 2, 3, 4]},
            [{"path": "$.results[*]", "length": {"min": 2}}],
        )
        assert passed is True

    def test_collects_every_failure(self) -> None:
        passed, reasons = evaluate_checks(
            {"a": 1, "b": 2},
            [
                {"path": "$.a", "equals": 99},
                {"path": "$.b", "equals": 99},
            ],
        )
        assert passed is False
        assert len(reasons) == 2
