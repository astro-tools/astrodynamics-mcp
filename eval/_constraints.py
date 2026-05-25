"""Argument-constraint matcher for the ``permitted_traces`` half of the scorer.

Each entry under a prompt YAML's ``arg_constraints`` block is a one-key
dict whose key names the predicate and whose value parameterises it. This
module is the closed vocabulary of those predicates plus the matcher that
applies them to actual arguments captured from the LLM's tool-call trace.

The vocabulary is intentionally small at v0.1 — only what the drafted
prompts in the issue tracker need. Extending it is a deliberate act:
add a predicate handler here, document it in ``eval/README.md``, and add
a test in ``tests/test_eval_constraints.py``.

Special cases:

- ``one_of: [..., null]`` lets a constraint pass when the LLM omits the
  argument entirely. This is the canonical "either omitted or one of
  these values" pattern (e.g. ``frame: {one_of: ["TEME", null]}`` where
  TEME is the tool's default).
- For any *other* predicate, a missing argument is a failure — the prompt
  drafter is asserting the argument should have been present.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# Sentinel used as the "no argument was present" marker. We can't use
# ``None`` because ``None`` is a legitimate JSON value an LLM may pass.
_MISSING: Any = object()


class ConstraintSpecError(ValueError):
    """Raised when a constraint dict in a YAML prompt has an unknown predicate.

    Surfaces eagerly during prompt loading so that broken YAML fails fast
    rather than producing confusing scorer output at eval time.
    """


_KNOWN_PREDICATES: frozenset[str] = frozenset(
    {
        "equals",
        "one_of",
        "case_insensitive_equals",
        "case_insensitive_contains",
        "numeric_tolerance",
        "length",
        "field_constraints",
        "has_fields",
    }
)


def validate_constraint(constraint: Any, *, path: str = "") -> None:
    """Validate a single ``arg_constraints`` entry against the closed vocabulary.

    Raises :class:`ConstraintSpecError` with a path-qualified message when
    the dict's key is not in :data:`_KNOWN_PREDICATES` or the parameter
    shape is malformed. Used by ``eval/_prompts.py`` to fail YAML loads
    eagerly.
    """
    if not isinstance(constraint, Mapping):
        raise ConstraintSpecError(
            f"constraint at {path or '<root>'} must be a mapping, got {type(constraint).__name__}"
        )
    if len(constraint) != 1:
        raise ConstraintSpecError(
            f"constraint at {path or '<root>'} must have exactly one predicate key, "
            f"got {sorted(constraint.keys())}"
        )
    (predicate,) = constraint.keys()
    if predicate not in _KNOWN_PREDICATES:
        raise ConstraintSpecError(
            f"unknown constraint predicate {predicate!r} at {path or '<root>'}; "
            f"vocabulary is {sorted(_KNOWN_PREDICATES)}"
        )
    value = constraint[predicate]
    _validate_predicate_value(predicate, value, path=path)


def _validate_predicate_value(predicate: str, value: Any, *, path: str) -> None:
    if predicate == "equals":
        return  # any JSON value is allowed
    if predicate == "one_of":
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ConstraintSpecError(
                f"one_of at {path} requires a list of values, got {type(value).__name__}"
            )
        return
    if predicate in ("case_insensitive_equals", "case_insensitive_contains"):
        if not isinstance(value, str):
            raise ConstraintSpecError(
                f"{predicate} at {path} requires a string, got {type(value).__name__}"
            )
        return
    if predicate == "numeric_tolerance":
        if not isinstance(value, Mapping):
            raise ConstraintSpecError(
                f"numeric_tolerance at {path} requires a mapping with 'expected' "
                f"and at least one of {{abs, rel}}"
            )
        if "expected" not in value:
            raise ConstraintSpecError(f"numeric_tolerance at {path} is missing 'expected'")
        if "abs" not in value and "rel" not in value:
            raise ConstraintSpecError(
                f"numeric_tolerance at {path} requires at least one of 'abs' or 'rel'"
            )
        return
    if predicate == "length":
        if isinstance(value, int):
            return
        if isinstance(value, Mapping) and set(value.keys()) <= {"min", "max"} and value:
            return
        raise ConstraintSpecError(
            f"length at {path} requires an int or a mapping with 'min'/'max' keys"
        )
    if predicate == "field_constraints":
        if not isinstance(value, Mapping):
            raise ConstraintSpecError(
                f"field_constraints at {path} requires a mapping of field name to constraint"
            )
        for field_name, sub_constraint in value.items():
            validate_constraint(sub_constraint, path=f"{path}.{field_name}")
        return
    if predicate == "has_fields":
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ConstraintSpecError(
                f"has_fields at {path} requires a list of field names, got {type(value).__name__}"
            )
        for field_name in value:
            if not isinstance(field_name, str):
                raise ConstraintSpecError(
                    f"has_fields at {path} entries must be strings, got {field_name!r}"
                )
        return
    # Unreachable given the membership check in validate_constraint, but
    # mypy needs the exhaustiveness statement.
    raise ConstraintSpecError(f"unhandled predicate {predicate!r} at {path}")


def match_arg(actual: Any, constraint: Mapping[str, Any], *, path: str = "") -> str | None:
    """Apply *constraint* to *actual*; return ``None`` on pass or a reason string.

    *actual* is :data:`_MISSING` when the LLM did not supply the argument.
    The reason string is intended for surfacing in the scorer's
    ``explanation`` so PR comments can show why a trace failed.
    """
    (predicate,) = constraint.keys()
    value = constraint[predicate]

    if predicate == "equals":
        if actual is _MISSING:
            return f"{path}: expected equals={value!r}, but argument was omitted"
        if actual == value:
            return None
        return f"{path}: expected equals={value!r}, got {actual!r}"

    if predicate == "one_of":
        # `null` in the list is the sentinel for "omitted is allowed".
        if actual is _MISSING:
            if None in value:
                return None
            return f"{path}: argument omitted but allowed values are {value!r}"
        if actual in value:
            return None
        return f"{path}: expected one_of={value!r}, got {actual!r}"

    if predicate == "case_insensitive_equals":
        if actual is _MISSING:
            return f"{path}: expected case_insensitive_equals={value!r}, but argument was omitted"
        if not isinstance(actual, str):
            return f"{path}: case_insensitive_equals requires a string, got {type(actual).__name__}"
        if actual.casefold() == value.casefold():
            return None
        return f"{path}: expected case_insensitive_equals={value!r}, got {actual!r}"

    if predicate == "case_insensitive_contains":
        if actual is _MISSING:
            return f"{path}: expected case_insensitive_contains={value!r}, but argument was omitted"
        if not isinstance(actual, str):
            return (
                f"{path}: case_insensitive_contains requires a string, got {type(actual).__name__}"
            )
        if value.casefold() in actual.casefold():
            return None
        return f"{path}: expected case_insensitive_contains={value!r}, got {actual!r}"

    if predicate == "numeric_tolerance":
        if actual is _MISSING:
            return f"{path}: expected numeric_tolerance match, but argument was omitted"
        return _check_numeric_tolerance(actual, value, path=path)

    if predicate == "length":
        if actual is _MISSING:
            return f"{path}: expected length constraint, but argument was omitted"
        if not hasattr(actual, "__len__"):
            return f"{path}: length requires a sized value, got {type(actual).__name__}"
        actual_len = len(actual)
        if isinstance(value, int):
            if actual_len == value:
                return None
            return f"{path}: expected length={value}, got {actual_len}"
        # Mapping with min/max keys (validate_constraint already enforced shape).
        lo = value.get("min")
        hi = value.get("max")
        if lo is not None and actual_len < lo:
            return f"{path}: expected length>={lo}, got {actual_len}"
        if hi is not None and actual_len > hi:
            return f"{path}: expected length<={hi}, got {actual_len}"
        return None

    if predicate == "field_constraints":
        if actual is _MISSING:
            return f"{path}: expected nested field_constraints, but argument was omitted"
        if not isinstance(actual, Mapping):
            return f"{path}: field_constraints requires a mapping, got {type(actual).__name__}"
        for field_name, sub_constraint in value.items():
            sub_actual = actual.get(field_name, _MISSING)
            sub_path = f"{path}.{field_name}" if path else field_name
            sub_reason = match_arg(sub_actual, sub_constraint, path=sub_path)
            if sub_reason is not None:
                return sub_reason
        return None

    if predicate == "has_fields":
        if actual is _MISSING:
            return f"{path}: expected has_fields={value!r}, but argument was omitted"
        if not isinstance(actual, Mapping):
            return f"{path}: has_fields requires a mapping, got {type(actual).__name__}"
        missing = [name for name in value if name not in actual]
        if missing:
            return f"{path}: missing required fields {missing!r}"
        return None

    # Unreachable: validate_constraint guarantees the predicate is known.
    return f"{path}: unhandled predicate {predicate!r}"


def _check_numeric_tolerance(actual: Any, spec: Mapping[str, Any], *, path: str) -> str | None:
    expected = spec["expected"]
    abs_tol = spec.get("abs")
    rel_tol = spec.get("rel")

    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)):
            return f"{path}: numeric_tolerance expected a list, got {type(actual).__name__}"
        if len(actual) != len(expected):
            return (
                f"{path}: numeric_tolerance length mismatch (expected {len(expected)} "
                f"elements, got {len(actual)})"
            )
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            reason = _check_scalar_tolerance(a, e, abs_tol, rel_tol, path=f"{path}[{i}]")
            if reason is not None:
                return reason
        return None

    return _check_scalar_tolerance(actual, expected, abs_tol, rel_tol, path=path)


def _check_scalar_tolerance(
    actual: Any,
    expected: Any,
    abs_tol: Any,
    rel_tol: Any,
    *,
    path: str,
) -> str | None:
    try:
        a = float(actual)
        e = float(expected)
    except (TypeError, ValueError):
        return f"{path}: numeric_tolerance requires numeric values, got actual={actual!r}"
    diff = abs(a - e)
    tolerated = False
    if abs_tol is not None and diff <= float(abs_tol):
        tolerated = True
    if not tolerated and rel_tol is not None and e != 0 and diff / abs(e) <= float(rel_tol):
        tolerated = True
    if tolerated:
        return None
    return (
        f"{path}: numeric_tolerance failed — expected {e}, got {a} "
        f"(diff={diff}, abs_tol={abs_tol}, rel_tol={rel_tol})"
    )


def match_args(
    call_args: Mapping[str, Any],
    constraints: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, list[str]]:
    """Apply every constraint in *constraints* to the matching key in *call_args*.

    Returns ``(passed, failure_reasons)``. ``passed`` is True iff every
    constraint produced no reason; ``failure_reasons`` lists every failure
    that occurred (all constraints are checked, even after the first failure,
    so PR comments show the complete picture).
    """
    reasons: list[str] = []
    for arg_name, constraint in constraints.items():
        actual = call_args.get(arg_name, _MISSING)
        reason = match_arg(actual, constraint, path=arg_name)
        if reason is not None:
            reasons.append(reason)
    return (not reasons, reasons)
