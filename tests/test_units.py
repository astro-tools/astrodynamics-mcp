"""Tests for `astrodynamics_mcp.units` helpers and pydantic wrappers."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from astrodynamics_mcp.errors import InvalidInputError
from astrodynamics_mcp.units import (
    ALLOWED_UNITS,
    Quantity,
    QuantityVector,
    is_finite_number,
    quantity,
    quantity_vector,
)


class TestQuantityHelper:
    def test_happy_path(self) -> None:
        assert quantity(7000.0, "km") == {"value": 7000.0, "unit": "km"}

    def test_int_input_is_coerced_to_float(self) -> None:
        result = quantity(7000, "km")
        assert result == {"value": 7000.0, "unit": "km"}
        assert isinstance(result["value"], float)

    @pytest.mark.parametrize("special", [math.nan, math.inf, -math.inf])
    def test_nan_and_infinity_are_allowed(self, special: float) -> None:
        # Tools use NaN to flag failed / degenerate sub-computations; the
        # discipline layer does not police that.
        result = quantity(special, "km")
        assert result["unit"] == "km"
        assert math.isnan(result["value"]) if math.isnan(special) else result["value"] == special

    def test_unknown_unit_raises_typed_error(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            quantity(1.0, "furlong")
        assert excinfo.value.code == "invalid_input.unknown_unit"
        assert "furlong" in excinfo.value.message

    @pytest.mark.parametrize("bad_value", ["7000", None, [1.0], True, False])
    def test_non_numeric_value_rejected(self, bad_value: object) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            quantity(bad_value, "km")  # type: ignore[arg-type]
        assert excinfo.value.code == "invalid_input.value_not_a_number"

    def test_non_string_unit_rejected(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            quantity(1.0, 7)  # type: ignore[arg-type]
        assert excinfo.value.code == "invalid_input.unit_not_a_string"


class TestQuantityVectorHelper:
    def test_happy_path(self) -> None:
        assert quantity_vector([7000.0, 0.0, 0.0], "km") == {
            "value": [7000.0, 0.0, 0.0],
            "unit": "km",
        }

    def test_mixed_int_float_input_coerces(self) -> None:
        result = quantity_vector([1, 2.5, 3], "km")
        assert result["value"] == [1.0, 2.5, 3.0]
        assert all(isinstance(x, float) for x in result["value"])

    def test_empty_vector_is_allowed(self) -> None:
        # Fixed-length is the tool schema's responsibility, not the helper's.
        assert quantity_vector([], "km") == {"value": [], "unit": "km"}

    def test_non_numeric_element_rejected(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            quantity_vector([1.0, "two", 3.0], "km")  # type: ignore[list-item]
        assert excinfo.value.code == "invalid_input.value_not_a_number"
        assert "values[1]" in excinfo.value.message

    def test_string_input_rejected_even_though_iterable(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            quantity_vector("abc", "km")  # type: ignore[arg-type]
        assert excinfo.value.code == "invalid_input.values_not_a_sequence"

    def test_bytes_input_rejected(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            quantity_vector(b"abc", "km")
        assert excinfo.value.code == "invalid_input.values_not_a_sequence"

    def test_unknown_unit_raises_typed_error(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            quantity_vector([1.0, 2.0, 3.0], "furlong")
        assert excinfo.value.code == "invalid_input.unknown_unit"


class TestAllowedUnitsRegistry:
    @pytest.mark.parametrize(
        "unit",
        [
            "km",
            "m",
            "AU",
            "km/s",
            "m/s",
            "deg",
            "rad",
            "s",
            "min",
            "hours",
            "days",
            "km^2/s^2",
            "km^3/s^2",
            "kg",
            "K",
            "1",
        ],
    )
    def test_v01_units_present(self, unit: str) -> None:
        assert unit in ALLOWED_UNITS

    def test_registry_is_immutable(self) -> None:
        assert isinstance(ALLOWED_UNITS, frozenset)


class TestPydanticQuantity:
    def test_valid_quantity_round_trips_through_json(self) -> None:
        q = Quantity(value=7000.0, unit="km")
        as_json = q.model_dump_json()
        restored = Quantity.model_validate_json(as_json)
        assert restored == q

    def test_unknown_unit_raises_typed_input_error(self) -> None:
        # Field validators raise InvalidInputError directly so the wire
        # contract is the same whether construction goes via the helper
        # or via direct Quantity instantiation.
        with pytest.raises(InvalidInputError) as excinfo:
            Quantity(value=1.0, unit="furlong")
        assert excinfo.value.code == "invalid_input.unknown_unit"

    def test_extra_keys_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            Quantity.model_validate({"value": 1.0, "unit": "km", "extra": "no"})

    def test_int_coerces_to_float(self) -> None:
        q = Quantity(value=7000, unit="km")
        assert isinstance(q.value, float)


class TestPydanticQuantityVector:
    def test_valid_vector_round_trips(self) -> None:
        qv = QuantityVector(value=[7000.0, 0.0, 0.0], unit="km")
        assert qv.model_dump() == {"value": [7000.0, 0.0, 0.0], "unit": "km"}

    def test_empty_vector_allowed(self) -> None:
        QuantityVector(value=[], unit="km")

    def test_unknown_unit_raises_typed_input_error(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            QuantityVector(value=[1.0, 2.0, 3.0], unit="furlong")
        assert excinfo.value.code == "invalid_input.unknown_unit"

    def test_extra_keys_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            QuantityVector.model_validate({"value": [1.0], "unit": "km", "extra": "no"})


class TestIsFiniteNumber:
    @pytest.mark.parametrize("value", [0, 1, -1, 1.5, -1.5, 0.0])
    def test_finite_numbers_pass(self, value: object) -> None:
        assert is_finite_number(value) is True

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_nan_and_inf_fail(self, value: float) -> None:
        assert is_finite_number(value) is False

    @pytest.mark.parametrize("value", ["1", None, [], True, False])
    def test_non_numeric_fails(self, value: object) -> None:
        assert is_finite_number(value) is False
