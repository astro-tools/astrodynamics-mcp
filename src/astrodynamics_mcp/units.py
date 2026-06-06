"""`{value, unit}` discipline helpers and the allowed-unit registry.

Every numeric value crossing the MCP wire goes through one of:

- :func:`quantity` — scalars wrapped as ``{"value": <number>, "unit": <str>}``
- :func:`quantity_vector` — vectors wrapped as ``{"value": [<number>, ...], "unit": <str>}``

The polymorphic ``value`` key (number or list-of-numbers) keeps the envelope
consistent across scalar and vector outputs; each tool's pydantic output
schema fixes the field's type per slot.

The :data:`ALLOWED_UNITS` registry is the closed set tools may emit. Unknown
strings fail fast with :class:`InvalidInputError` so a typo at the tool layer
surfaces as a typed error rather than as a silently malformed payload.

:class:`Quantity` and :class:`QuantityVector` are the pydantic models tool
output schemas compose from; the JSON-schema produced from each is what the
cross-tool unit-discipline meta-test checks every tool against.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from astrodynamics_mcp.errors import InvalidInputError

# Closed registry of unit strings the tool surface may emit. Tool issues
# extend this when they introduce a new physical quantity (the meta-test
# would otherwise reject the new field, forcing the addition to be
# deliberate). Keep grouped by physical dimension; comments are the
# meaningful axis, not import order.
ALLOWED_UNITS: frozenset[str] = frozenset(
    {
        # Dimensionless.
        "1",
        # Length.
        "m",
        "km",
        "AU",
        # Velocity.
        "m/s",
        "km/s",
        # Angle.
        "rad",
        "deg",
        # Angular rate — body-orientation polynomial coefficients (POLE_RA /
        # POLE_DEC are per Julian century, PM is per day; squared forms cover a
        # quadratic term where a body's model carries one).
        "deg/day",
        "deg/century",
        "deg/day^2",
        "deg/century^2",
        # Time.
        "s",
        "min",
        "hours",
        "days",
        # Ephemeris time as a coordinate, not a duration: SPICE ET is seconds
        # past the J2000 TDB epoch (the leap-second-kernel-defined zero), so it
        # carries its anchor in the unit string to distinguish it from a plain
        # `s` interval. Emitted only by spice_time_convert's ET output.
        "s past J2000 TDB",
        # Area / specific energy (C3 is canonically reported in km^2/s^2).
        "km^2/s^2",
        # Gravitational parameter (mu).
        "km^3/s^2",
        # Mass.
        "kg",
        # Temperature.
        "K",
    }
)


def _validate_unit(unit: str) -> None:
    if not isinstance(unit, str):
        raise InvalidInputError(
            f"unit must be a string, got {type(unit).__name__}",
            code="invalid_input.unit_not_a_string",
        )
    if unit not in ALLOWED_UNITS:
        raise InvalidInputError(
            f"unknown unit {unit!r}; allowed units are {sorted(ALLOWED_UNITS)}",
            code="invalid_input.unknown_unit",
        )


def _validate_number(value: object, *, where: str) -> float:
    """Coerce *value* to a finite ``float`` or raise ``InvalidInputError``.

    Rejects ``bool`` (Python's ``isinstance(True, int)`` is true but a quantity
    value must be numeric), strings, ``None``, and anything else. Also rejects
    ``NaN`` and infinities: JSON has no representation for them (``json.dumps``
    emits the non-standard ``NaN`` / ``Infinity`` tokens, which strict MCP
    clients reject), so a non-finite value must never cross the wire. A tool
    whose computation degenerates should detect that and raise the appropriate
    typed error rather than wrapping a non-finite result in a ``Quantity``;
    this guard is the boundary backstop.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInputError(
            f"{where} must be a number (int or float), got {type(value).__name__}",
            code="invalid_input.value_not_a_number",
        )
    if not math.isfinite(value):
        raise InvalidInputError(
            f"{where} must be a finite number, got {value}",
            code="invalid_input.non_finite_value",
        )
    return float(value)


def quantity(value: float | int, unit: str) -> dict[str, Any]:
    """Wrap a scalar numeric value as a ``{value, unit}`` dict."""
    _validate_unit(unit)
    numeric = _validate_number(value, where="value")
    return {"value": numeric, "unit": unit}


def quantity_vector(values: Sequence[float | int], unit: str) -> dict[str, Any]:
    """Wrap a sequence of numeric values as a ``{value: [...], unit}`` dict.

    The shape uses the same ``value`` key as :func:`quantity`; tools choose
    scalar or vector per output field via their pydantic schema.
    """
    _validate_unit(unit)
    if isinstance(values, (str, bytes)):
        raise InvalidInputError(
            f"values must be a sequence of numbers, got {type(values).__name__}",
            code="invalid_input.values_not_a_sequence",
        )
    coerced = [_validate_number(v, where=f"values[{i}]") for i, v in enumerate(values)]
    return {"value": coerced, "unit": unit}


class Quantity(BaseModel):
    """Pydantic model for a scalar ``{value, unit}`` payload.

    Used as the field type in every tool's pydantic output schema where the
    field carries a scalar physical quantity. The JSON-schema export is what
    the unit-discipline meta-test recognises as "correctly wrapped".
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float = Field(..., description="Numeric value of the quantity.")
    unit: str = Field(..., description="Unit string from the allowed-units registry.")

    @field_validator("unit")
    @classmethod
    def _unit_in_registry(cls, v: str) -> str:
        _validate_unit(v)
        return v


class QuantityVector(BaseModel):
    """Pydantic model for a vector ``{value: [...], unit}`` payload.

    The JSON-schema export uses ``type: array, items: {type: number}`` for
    the ``value`` field — the unit-discipline meta-test treats this as the
    canonical vector-quantity shape.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: list[float] = Field(..., description="Numeric values of the quantity vector.")
    unit: str = Field(..., description="Unit string from the allowed-units registry.")

    @field_validator("unit")
    @classmethod
    def _unit_in_registry(cls, v: str) -> str:
        _validate_unit(v)
        return v


# ---------------------------------------------------------------------------
# Schema-walking primitives for the unit-discipline meta-test
# ---------------------------------------------------------------------------


# JSON-Schema types treated as "numeric" by the meta-test. A tool field with
# any of these types — that is not nested inside a Quantity / QuantityVector
# wrapper — is a unit-discipline violation.
_NUMERIC_JSON_TYPES: frozenset[str] = frozenset({"number", "integer"})


class _UnitDisciplineViolation(BaseModel):
    """Diagnostic record returned by :func:`find_unit_discipline_violations`."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    schema_name: str
    field_path: str
    reason: Literal["bare_numeric_field"]


def _resolve_ref(ref: str, root_schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve a local JSON-Schema ``$ref`` (``#/$defs/X``) to its target."""
    if not ref.startswith("#/"):
        # Non-local refs are out of scope for the v0.1 meta-test; refuse rather
        # than guess. No tool output schema needs cross-document refs.
        raise InvalidInputError(
            f"only local $refs are supported in the unit-discipline walker, got {ref!r}",
            code="invalid_input.unsupported_ref",
        )
    cursor: Any = root_schema
    for part in ref[2:].split("/"):
        if not isinstance(cursor, dict) or part not in cursor:
            raise InvalidInputError(
                f"unresolved $ref {ref!r} in schema",
                code="invalid_input.unresolved_ref",
            )
        cursor = cursor[part]
    if not isinstance(cursor, dict):
        raise InvalidInputError(
            f"$ref {ref!r} resolves to {type(cursor).__name__}, not an object",
            code="invalid_input.unresolved_ref",
        )
    return cursor


def _is_quantity_shape(node: dict[str, Any]) -> bool:
    """Return whether *node* describes a ``Quantity`` or ``QuantityVector``.

    Caller's responsibility to pass a ``$ref``-resolved node — the walker
    handles ``$ref`` before invoking this. A node counts as a quantity wrapper
    when it declares an object with the keys ``value`` and ``unit`` and
    disallows extras (pydantic's ``extra="forbid"`` surfaces as
    ``additionalProperties: false``).
    """
    if node.get("type") != "object":
        return False
    if node.get("additionalProperties") is not False:
        return False
    properties = node.get("properties", {})
    return "value" in properties and "unit" in properties


def _iter_violations(
    node: Any,
    *,
    path: str,
    root_schema: dict[str, Any],
    schema_name: str,
    exempt_field_paths: frozenset[str],
) -> Iterator[_UnitDisciplineViolation]:
    """Recursively walk a JSON-Schema node, yielding bare-numeric-field violations.

    A field is a violation when its declared type intersects ``number`` /
    ``integer`` and the node itself is not a quantity-shape wrapper. We
    descend into ``properties``, ``items`` (array element schemas),
    ``anyOf`` / ``oneOf`` / ``allOf`` unions, and ``$defs`` referenced via
    ``$ref``.

    A bare-numeric field whose exact ``path`` is in *exempt_field_paths* is not
    flagged — the single relaxation point for attachment-bearing outputs whose
    summary carries a non-physical cardinality (a pixel count, a packet count)
    that has no place in the ``{value, unit}`` envelope. The exemption is
    per-path, so it can never silently excuse a genuine physical quantity
    elsewhere in the schema.
    """
    if not isinstance(node, dict):
        return

    if "$ref" in node:
        yield from _iter_violations(
            _resolve_ref(node["$ref"], root_schema),
            path=path,
            root_schema=root_schema,
            schema_name=schema_name,
            exempt_field_paths=exempt_field_paths,
        )
        return

    node_type = node.get("type")
    types_set: set[str] = set()
    if isinstance(node_type, str):
        types_set = {node_type}
    elif isinstance(node_type, list):
        types_set = {t for t in node_type if isinstance(t, str)}

    if types_set & _NUMERIC_JSON_TYPES and not _is_quantity_shape(node):
        if path not in exempt_field_paths:
            yield _UnitDisciplineViolation(
                schema_name=schema_name,
                field_path=path,
                reason="bare_numeric_field",
            )
        return

    if _is_quantity_shape(node):
        # The Quantity / QuantityVector wrapper terminates the walk — by
        # construction its inner `value` is numeric, that's the whole point.
        return

    for key in ("anyOf", "oneOf", "allOf"):
        for i, sub in enumerate(node.get(key, []) or []):
            yield from _iter_violations(
                sub,
                path=f"{path}.{key}[{i}]",
                root_schema=root_schema,
                schema_name=schema_name,
                exempt_field_paths=exempt_field_paths,
            )

    properties = node.get("properties")
    if isinstance(properties, dict):
        for prop_name, prop_schema in properties.items():
            yield from _iter_violations(
                prop_schema,
                path=f"{path}.{prop_name}" if path else prop_name,
                root_schema=root_schema,
                schema_name=schema_name,
                exempt_field_paths=exempt_field_paths,
            )

    items = node.get("items")
    if isinstance(items, dict):
        yield from _iter_violations(
            items,
            path=f"{path}[]",
            root_schema=root_schema,
            schema_name=schema_name,
            exempt_field_paths=exempt_field_paths,
        )


def find_unit_discipline_violations(
    schema: dict[str, Any],
    *,
    schema_name: str,
    exempt_field_paths: frozenset[str] = frozenset(),
) -> list[_UnitDisciplineViolation]:
    """Return every bare-numeric-field violation in *schema*.

    The schema is a JSON-Schema dict (e.g. the output of
    ``MyModel.model_json_schema()``). The check is recursive across nested
    objects, arrays, unions, and ``$defs``-style references.

    An empty list means the schema satisfies the unit-discipline rule: every
    numeric field is wrapped in a :class:`Quantity` or :class:`QuantityVector`.

    *exempt_field_paths* relaxes the rule for the listed field paths only —
    the mechanism attachment-bearing tool outputs use to carry a non-physical
    cardinality (a pixel count, a packet count) that does not fit the
    ``{value, unit}`` envelope. The numeric tool surface passes no exemptions,
    so the rule stays strict for every physical quantity. Each entry must match
    a field's exact dotted path (e.g. ``"image.width_px"``); a path that
    matches nothing is simply inert.
    """
    return list(
        _iter_violations(
            schema,
            path="",
            root_schema=schema,
            schema_name=schema_name,
            exempt_field_paths=exempt_field_paths,
        )
    )


# NaN/inf-checking convenience surfaced for tool authors; not used by the
# canonical helpers above but exposed because most arg validators want it.
def is_finite_number(value: object) -> bool:
    """Return whether *value* is a real number with no NaN / inf."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))
