"""Unit tests for the eval-suite argument-constraint matcher."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from eval._constraints import ConstraintSpecError, match_args, validate_constraint


class TestValidate:
    """validate_constraint catches malformed predicate vocabulary eagerly."""

    @pytest.mark.parametrize(
        "constraint",
        [
            {"equals": 42},
            {"one_of": [1, 2, None]},
            {"case_insensitive_equals": "Madrid"},
            {"case_insensitive_contains": "HUBBLE"},
            {"length": 3},
            {"length": {"min": 1, "max": 5}},
            {"length": {"min": 1}},
            {"numeric_tolerance": {"expected": 1.0, "abs": 0.01}},
            {"numeric_tolerance": {"expected": [1.0, 2.0], "rel": 1e-3}},
            {"field_constraints": {"name": {"equals": "madrid"}}},
            {"has_fields": ["lat", "lon", "alt"]},
        ],
    )
    def test_well_formed(self, constraint: Mapping[str, Any]) -> None:
        validate_constraint(constraint)

    @pytest.mark.parametrize(
        ("constraint", "fragment"),
        [
            ({"unknown_predicate": 1}, "unknown constraint predicate"),
            ({}, "exactly one predicate key"),
            ({"equals": 1, "one_of": [1]}, "exactly one predicate key"),
            ({"one_of": "not-a-list"}, "requires a list"),
            ({"case_insensitive_equals": 42}, "requires a string"),
            ({"numeric_tolerance": {"expected": 1.0}}, "requires at least one of"),
            ({"numeric_tolerance": {"abs": 0.1}}, "missing 'expected'"),
            ({"length": {"foo": 1}}, "min'/'max"),
            ({"length": "three"}, "requires an int or a mapping"),
            ({"has_fields": "lat"}, "requires a list"),
            ({"has_fields": [1, 2]}, "entries must be strings"),
            ({"field_constraints": "nope"}, "requires a mapping"),
        ],
    )
    def test_rejects(self, constraint: Any, fragment: str) -> None:
        with pytest.raises(ConstraintSpecError, match=fragment):
            validate_constraint(constraint)


class TestMatchArgs:
    """match_args applies the vocabulary against captured tool-call arguments."""

    def test_equals_pass(self) -> None:
        passed, reasons = match_args({"query": "25544"}, {"query": {"equals": "25544"}})
        assert passed is True
        assert reasons == []

    def test_equals_fail(self) -> None:
        passed, reasons = match_args({"query": "WRONG"}, {"query": {"equals": "25544"}})
        assert passed is False
        assert any("expected equals='25544'" in r for r in reasons)
        assert any("WRONG" in r for r in reasons)

    def test_equals_deep_list(self) -> None:
        passed, _ = match_args(
            {"r1": [5000.0, 10000.0, 2100.0]},
            {"r1": {"equals": [5000.0, 10000.0, 2100.0]}},
        )
        assert passed is True

    def test_one_of_pass(self) -> None:
        passed, _ = match_args({"mu": "earth"}, {"mu": {"one_of": ["earth", 398600.4418]}})
        assert passed is True

    def test_one_of_allows_omission_when_null_present(self) -> None:
        passed, _ = match_args({}, {"frame": {"one_of": ["TEME", None]}})
        assert passed is True

    def test_one_of_rejects_omission_without_null(self) -> None:
        passed, reasons = match_args({}, {"frame": {"one_of": ["TEME"]}})
        assert passed is False
        assert any("omitted but allowed values" in r for r in reasons)

    def test_one_of_rejects_unmatched_value(self) -> None:
        passed, reasons = match_args({"mu": "venus"}, {"mu": {"one_of": ["earth"]}})
        assert passed is False
        assert any("got 'venus'" in r for r in reasons)

    def test_case_insensitive_equals(self) -> None:
        assert match_args({"name": "Madrid"}, {"name": {"case_insensitive_equals": "MADRID"}})[0]
        assert not match_args(
            {"name": "Goldstone"}, {"name": {"case_insensitive_equals": "MADRID"}}
        )[0]

    def test_case_insensitive_contains(self) -> None:
        assert match_args(
            {"query": "Hubble Space Telescope"},
            {"query": {"case_insensitive_contains": "HUBBLE"}},
        )[0]
        assert not match_args(
            {"query": "ISS (ZARYA)"},
            {"query": {"case_insensitive_contains": "HUBBLE"}},
        )[0]

    def test_numeric_tolerance_scalar_abs(self) -> None:
        passed, _ = match_args(
            {"tof": 3600.5},
            {"tof": {"numeric_tolerance": {"expected": 3600.0, "abs": 1.0}}},
        )
        assert passed is True

    def test_numeric_tolerance_scalar_rel(self) -> None:
        passed, _ = match_args(
            {"tof": 3600.0},
            {"tof": {"numeric_tolerance": {"expected": 3601.0, "rel": 1e-3}}},
        )
        assert passed is True

    def test_numeric_tolerance_vector_fail_on_one_element(self) -> None:
        passed, reasons = match_args(
            {"r1": [1.0, 2.0, 99.0]},
            {"r1": {"numeric_tolerance": {"expected": [1.0, 2.0, 3.0], "abs": 0.1}}},
        )
        assert passed is False
        assert any("r1[2]" in r for r in reasons)

    def test_length_scalar(self) -> None:
        assert match_args({"epochs": [1, 2, 3]}, {"epochs": {"length": 3}})[0]
        assert not match_args({"epochs": [1, 2]}, {"epochs": {"length": 3}})[0]

    def test_length_range(self) -> None:
        assert match_args({"results": [1, 2, 3]}, {"results": {"length": {"min": 2, "max": 4}}})[0]
        assert not match_args({"results": [1]}, {"results": {"length": {"min": 2}}})[0]

    def test_field_constraints_recursive(self) -> None:
        passed, _ = match_args(
            {"observer": {"name": "madrid"}},
            {"observer": {"field_constraints": {"name": {"case_insensitive_equals": "MADRID"}}}},
        )
        assert passed is True

    def test_field_constraints_reports_nested_path(self) -> None:
        passed, reasons = match_args(
            {"observer": {"name": "goldstone"}},
            {"observer": {"field_constraints": {"name": {"case_insensitive_equals": "MADRID"}}}},
        )
        assert passed is False
        assert any("observer.name" in r for r in reasons)

    def test_has_fields(self) -> None:
        passed, _ = match_args(
            {"observer": {"lat": {}, "lon": {}, "alt": {}}},
            {"observer": {"has_fields": ["lat", "lon", "alt"]}},
        )
        assert passed is True
        passed, reasons = match_args(
            {"observer": {"lat": {}, "lon": {}}},
            {"observer": {"has_fields": ["lat", "lon", "alt"]}},
        )
        assert passed is False
        assert any("missing required fields" in r for r in reasons)

    def test_collects_every_failure(self) -> None:
        passed, reasons = match_args(
            {"a": 1, "b": 2},
            {"a": {"equals": 99}, "b": {"equals": 99}},
        )
        assert passed is False
        # Both failures recorded, not just the first.
        assert len(reasons) == 2
