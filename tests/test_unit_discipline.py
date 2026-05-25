"""Cross-tool unit-discipline meta-test.

Walks every tool's pydantic output schema and asserts every numeric field is
wrapped in the canonical ``Quantity`` / ``QuantityVector`` shape. Adding a
tool that bypasses unit discipline fails this test, by design.

The registry :data:`OUTPUT_SCHEMAS_TO_CHECK` is empty at v0.1 — no tools
register output schemas yet. Each tool issue appends its output models here
as it lands; the test then enforces the rule on the new surface.

Two self-test fixtures exercise the walker itself (one compliant, one
non-compliant) so the meta-test's invariant survives even when the registry
is empty.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from astrodynamics_mcp.schemas.base import (
    Interval,
    KeplerianElements,
    ObserverCoordinates,
    StateVector,
)
from astrodynamics_mcp.tools.lambert import LambertSolveResponse
from astrodynamics_mcp.tools.propagation import Sgp4PropagateResponse
from astrodynamics_mcp.tools.tle import TleLookupResponse
from astrodynamics_mcp.units import (
    Quantity,
    QuantityVector,
    find_unit_discipline_violations,
)

# Registry of tool output schemas the meta-test should police. Each tool
# issue appends its top-level output model here.
OUTPUT_SCHEMAS_TO_CHECK: list[type[BaseModel]] = [
    # Base schemas that compose quantities directly — registering them here
    # verifies the cross-tool lint runs cleanly against real composed shapes,
    # not just the self-test fixtures below.
    StateVector,
    Interval,
    ObserverCoordinates,
    KeplerianElements,
    # Tool output schemas.
    TleLookupResponse,
    Sgp4PropagateResponse,
    LambertSolveResponse,
]


class _GoodSchema(BaseModel):
    """Self-test fixture: a schema that satisfies unit discipline."""

    model_config = ConfigDict(extra="forbid")

    altitude: Quantity
    velocity: QuantityVector
    name: str  # strings are not numeric → fine
    epoch: str  # ISO 8601 timestamp; not a numeric field


class _BadSchema(BaseModel):
    """Self-test fixture: a schema with a bare numeric field."""

    model_config = ConfigDict(extra="forbid")

    altitude_km: float  # ← bare numeric, no unit wrapper. Violation.


class _NestedBadSchema(BaseModel):
    """Self-test fixture: a bare numeric field nested inside an object."""

    model_config = ConfigDict(extra="forbid")

    inner: _BadSchema


class _NestedGoodSchema(BaseModel):
    """Self-test fixture: a quantity-wrapped field nested inside an object."""

    model_config = ConfigDict(extra="forbid")

    inner: _GoodSchema


class TestWalkerSelfChecks:
    def test_compliant_schema_yields_no_violations(self) -> None:
        violations = find_unit_discipline_violations(
            _GoodSchema.model_json_schema(),
            schema_name="_GoodSchema",
        )
        assert violations == []

    def test_bare_numeric_field_is_flagged(self) -> None:
        violations = find_unit_discipline_violations(
            _BadSchema.model_json_schema(),
            schema_name="_BadSchema",
        )
        assert len(violations) == 1
        violation = violations[0]
        assert violation.schema_name == "_BadSchema"
        assert violation.field_path == "altitude_km"
        assert violation.reason == "bare_numeric_field"

    def test_nested_bare_numeric_field_is_flagged(self) -> None:
        violations = find_unit_discipline_violations(
            _NestedBadSchema.model_json_schema(),
            schema_name="_NestedBadSchema",
        )
        assert len(violations) == 1
        assert violations[0].field_path.endswith("altitude_km")

    def test_nested_quantity_wrapped_fields_are_clean(self) -> None:
        violations = find_unit_discipline_violations(
            _NestedGoodSchema.model_json_schema(),
            schema_name="_NestedGoodSchema",
        )
        assert violations == []


class TestRegisteredOutputSchemas:
    def test_every_registered_schema_satisfies_unit_discipline(self) -> None:
        all_violations = []
        for model_cls in OUTPUT_SCHEMAS_TO_CHECK:
            all_violations.extend(
                find_unit_discipline_violations(
                    model_cls.model_json_schema(),
                    schema_name=model_cls.__name__,
                )
            )
        assert all_violations == [], (
            "tool output schemas violate the {value, unit} discipline:\n"
            + "\n".join(f"  - {v.schema_name}.{v.field_path}: {v.reason}" for v in all_violations)
        )

    def test_registry_is_a_list(self) -> None:
        # Sanity that future tool issues can extend it in place.
        assert isinstance(OUTPUT_SCHEMAS_TO_CHECK, list)


@pytest.mark.parametrize(
    "model_cls",
    [Quantity, QuantityVector],
    ids=["Quantity", "QuantityVector"],
)
def test_canonical_wrappers_are_their_own_quantity_shape(model_cls: type[BaseModel]) -> None:
    """The Quantity / QuantityVector models must themselves be valid wrappers.

    A regression here would mean the walker's recognition logic and the
    canonical models have drifted apart — both rooted in the same `value`
    and `unit` keys with extras forbidden.
    """
    violations = find_unit_discipline_violations(
        model_cls.model_json_schema(),
        schema_name=model_cls.__name__,
    )
    assert violations == []


class TestWalkerCornerCases:
    """Exercise the walker's defensive paths against hand-crafted JSON-Schema.

    Pydantic does not emit every JSON-Schema shape the walker has to handle
    correctly (multi-type fields, malformed ``$ref``, boolean-as-schema
    sub-nodes); these tests cover the paths directly.
    """

    def test_bare_array_of_numbers_is_flagged(self) -> None:
        """An array-of-numbers field that isn't a QuantityVector is a violation."""
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "altitudes": {"type": "array", "items": {"type": "number"}},
            },
        }
        violations = find_unit_discipline_violations(schema, schema_name="BareList")
        assert len(violations) == 1
        assert violations[0].field_path == "altitudes[]"

    def test_anyof_branch_with_bare_numeric_is_flagged(self) -> None:
        """A union including a bare number drops a violation inside anyOf[i]."""
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "altitude_or_string": {
                    "anyOf": [{"type": "number"}, {"type": "string"}],
                },
            },
        }
        violations = find_unit_discipline_violations(schema, schema_name="UnionLeak")
        assert len(violations) == 1
        assert violations[0].field_path == "altitude_or_string.anyOf[0]"

    def test_multi_type_node_with_number_is_flagged(self) -> None:
        """JSON Schema's ``type: [...]`` list form is recognised as numeric."""
        # Pydantic doesn't emit this shape, but it's a valid JSON Schema
        # construct (e.g. nullable numbers as ``type: ["number", "null"]``).
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "altitude_or_null": {"type": ["number", "null"]},
            },
        }
        violations = find_unit_discipline_violations(schema, schema_name="MultiType")
        assert len(violations) == 1

    def test_non_dict_property_value_short_circuits(self) -> None:
        """JSON Schema 2019-09 allows ``true`` / ``false`` as a schema; ignore."""
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "anything": True,  # boolean-as-schema → not walked further
            },
        }
        violations = find_unit_discipline_violations(schema, schema_name="BoolSchema")
        assert violations == []

    def test_non_local_ref_is_rejected(self) -> None:
        """Cross-document refs are out of scope; the walker refuses, not guesses."""
        from astrodynamics_mcp.errors import InvalidInputError

        schema = {
            "type": "object",
            "properties": {"foo": {"$ref": "https://example.com/schema.json"}},
        }
        with pytest.raises(InvalidInputError) as excinfo:
            find_unit_discipline_violations(schema, schema_name="RemoteRef")
        assert excinfo.value.code == "invalid_input.unsupported_ref"

    def test_unresolved_local_ref_is_rejected(self) -> None:
        from astrodynamics_mcp.errors import InvalidInputError

        schema = {
            "type": "object",
            "properties": {"foo": {"$ref": "#/$defs/DoesNotExist"}},
            "$defs": {"SomethingElse": {"type": "object"}},
        }
        with pytest.raises(InvalidInputError) as excinfo:
            find_unit_discipline_violations(schema, schema_name="DanglingRef")
        assert excinfo.value.code == "invalid_input.unresolved_ref"

    def test_ref_resolving_to_non_object_is_rejected(self) -> None:
        from astrodynamics_mcp.errors import InvalidInputError

        schema = {
            "type": "object",
            "properties": {"foo": {"$ref": "#/$defs/StringValue"}},
            "$defs": {"StringValue": "not an object"},
        }
        with pytest.raises(InvalidInputError) as excinfo:
            find_unit_discipline_violations(schema, schema_name="BadRef")
        assert excinfo.value.code == "invalid_input.unresolved_ref"
