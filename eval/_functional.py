"""Functional-answer predicates over the final tool response's JSON.

Each entry under a prompt YAML's ``functional_answer`` list pairs a
JSON-path-lite accessor with one of the predicates defined here. The
matcher walks the response, extracts the value(s) at the path, and runs
the predicate.

JSON-path-lite supports the small subset of JSON Path syntax we actually
need from the drafted prompts — no recursive descent, no filters, no
expressions — and is implemented in-line to avoid pulling another
dependency for ~50 lines of parsing.

Supported path tokens:

- ``$`` — the root response object.
- ``.field`` — descend into a mapping key.
- ``[N]`` — index into a list (negative indices supported).
- ``[*]`` — flatten the list; subsequent tokens apply to each element.
  Predicate decides what "match all elements" means
  (``equals`` / ``all_equal`` require every element; ``length`` operates
  on the flattened result set; ``in_range`` requires every element).

The flattening semantics are deliberate: most "every result has a TLE
line / every window has peak elevation above N / every result has unit
'km'" checks read naturally with ``[*]`` plus an element predicate.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

# Sentinel returned by _resolve_path when the path doesn't exist (vs. a
# legitimate ``None`` in the response).
_MISSING: Any = object()


# Sentinel returned by _resolve_path when the path's last token was ``[*]``
# and the matched value should be treated as a *set* of values rather than
# one. Wraps the list so callers can distinguish from a legitimate list-
# valued response field.
class _FlatResult:
    """List wrapper marker — distinguishes ``[*]`` results from list-valued fields."""

    __slots__ = ("values",)

    def __init__(self, values: list[Any]) -> None:
        self.values = values


class FunctionalSpecError(ValueError):
    """Raised when a functional-check dict has an unknown predicate or bad path."""


_KNOWN_PREDICATES: frozenset[str] = frozenset(
    {
        "equals",
        "in_range",
        "length",
        "present",
        "case_insensitive_contains",
        "starts_with",
        "all_equal",
        "numeric_tolerance",
    }
)


_TOKEN_RE = re.compile(r"\.([A-Za-z_][A-Za-z_0-9]*)|\[(-?\d+|\*)\]")


def _tokenise(path: str) -> list[str]:
    """Split ``$.field.sub[0][*]`` into ``['.field', '.sub', '[0]', '[*]']``."""
    if not path.startswith("$"):
        raise FunctionalSpecError(f"path must start with '$', got {path!r}")
    remainder = path[1:]
    tokens: list[str] = []
    pos = 0
    while pos < len(remainder):
        match = _TOKEN_RE.match(remainder, pos)
        if match is None:
            raise FunctionalSpecError(f"unparseable token in path {path!r} at position {pos + 1}")
        tokens.append(match.group(0))
        pos = match.end()
    return tokens


def validate_check(check: Any, *, index: int = 0) -> None:
    """Validate a single ``functional_answer`` entry; raise on malformed input.

    Called eagerly by the prompt loader so broken YAML fails before any
    LLM request is made.
    """
    if not isinstance(check, Mapping):
        raise FunctionalSpecError(
            f"functional_answer[{index}] must be a mapping, got {type(check).__name__}"
        )
    if "path" not in check:
        raise FunctionalSpecError(f"functional_answer[{index}] missing required key 'path'")
    if not isinstance(check["path"], str):
        raise FunctionalSpecError(
            f"functional_answer[{index}].path must be a string, got {type(check['path']).__name__}"
        )
    _tokenise(check["path"])
    predicate_keys = [k for k in check if k != "path"]
    if len(predicate_keys) != 1:
        raise FunctionalSpecError(
            f"functional_answer[{index}] must have exactly one predicate key besides "
            f"'path', got {predicate_keys}"
        )
    (predicate,) = predicate_keys
    if predicate not in _KNOWN_PREDICATES:
        raise FunctionalSpecError(
            f"unknown functional predicate {predicate!r} at functional_answer[{index}]; "
            f"vocabulary is {sorted(_KNOWN_PREDICATES)}"
        )
    _validate_predicate_value(predicate, check[predicate], path=check["path"])


def _validate_predicate_value(predicate: str, value: Any, *, path: str) -> None:
    if predicate == "in_range":
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
            raise FunctionalSpecError(f"in_range at {path} requires a two-element [min, max] list")
        return
    if predicate == "length":
        if isinstance(value, int):
            return
        if isinstance(value, Mapping) and set(value.keys()) <= {"min", "max"} and value:
            return
        raise FunctionalSpecError(
            f"length at {path} requires an int or a mapping with 'min'/'max' keys"
        )
    if predicate == "present":
        if not isinstance(value, bool):
            raise FunctionalSpecError(f"present at {path} requires a boolean, got {value!r}")
        return
    if predicate in ("case_insensitive_contains", "starts_with"):
        if not isinstance(value, str):
            raise FunctionalSpecError(
                f"{predicate} at {path} requires a string, got {type(value).__name__}"
            )
        return
    if predicate == "numeric_tolerance":
        if not isinstance(value, Mapping):
            raise FunctionalSpecError(
                f"numeric_tolerance at {path} requires a mapping with 'expected' "
                f"and at least one of {{abs, rel}}"
            )
        if "expected" not in value:
            raise FunctionalSpecError(f"numeric_tolerance at {path} missing 'expected'")
        if "abs" not in value and "rel" not in value:
            raise FunctionalSpecError(
                f"numeric_tolerance at {path} requires at least one of 'abs' or 'rel'"
            )
        return
    # equals and all_equal accept any JSON value
    return


def _resolve_path(response: Any, path: str) -> Any:
    """Walk *response* by *path*; return the value, :data:`_MISSING`, or a flat-result."""
    current: Any = response
    flat_mode = False
    flat_values: list[Any] = []
    for token in _tokenise(path):
        if token.startswith("."):
            key = token[1:]
            if flat_mode:
                next_values: list[Any] = []
                for item in flat_values:
                    if isinstance(item, Mapping) and key in item:
                        next_values.append(item[key])
                    else:
                        next_values.append(_MISSING)
                flat_values = next_values
                continue
            if not isinstance(current, Mapping) or key not in current:
                return _MISSING
            current = current[key]
            continue
        # Bracketed token: [N] or [*]
        inner = token[1:-1]
        if inner == "*":
            if flat_mode:
                # Nested [*][*] flattens.
                flat_values = [item for sub in flat_values if isinstance(sub, list) for item in sub]
            else:
                if not isinstance(current, list):
                    return _MISSING
                flat_values = list(current)
                flat_mode = True
            continue
        index = int(inner)
        if flat_mode:
            next_values = []
            for item in flat_values:
                if isinstance(item, list):
                    try:
                        next_values.append(item[index])
                    except IndexError:
                        next_values.append(_MISSING)
                else:
                    next_values.append(_MISSING)
            flat_values = next_values
            continue
        if not isinstance(current, list):
            return _MISSING
        try:
            current = current[index]
        except IndexError:
            return _MISSING
    if flat_mode:
        return _FlatResult(flat_values)
    return current


def _evaluate(check: Mapping[str, Any], response: Any) -> str | None:
    path = check["path"]
    (predicate,) = (k for k in check if k != "path")
    spec = check[predicate]
    value = _resolve_path(response, path)

    if predicate == "present":
        if isinstance(value, _FlatResult):
            is_present = bool(value.values) and all(v is not _MISSING for v in value.values)
        else:
            is_present = value is not _MISSING
        if is_present == spec:
            return None
        return f"{path}: expected present={spec}, got present={is_present}"

    if value is _MISSING:
        return f"{path}: not found in response"

    if isinstance(value, _FlatResult):
        return _evaluate_over_flat(value.values, predicate, spec, path)

    return _evaluate_scalar(predicate, spec, value, path=path)


def _evaluate_over_flat(
    values: list[Any],
    predicate: str,
    spec: Any,
    path: str,
) -> str | None:
    if predicate == "length":
        actual_len = len(values)
        if isinstance(spec, int):
            if actual_len == spec:
                return None
            return f"{path}: expected length={spec}, got {actual_len}"
        lo = spec.get("min")
        hi = spec.get("max")
        if lo is not None and actual_len < lo:
            return f"{path}: expected length>={lo}, got {actual_len}"
        if hi is not None and actual_len > hi:
            return f"{path}: expected length<={hi}, got {actual_len}"
        return None
    if predicate == "all_equal":
        # All flattened values must be equal to spec.
        bad = [v for v in values if v != spec]
        if bad:
            return f"{path}: all_equal={spec!r} failed; mismatches: {bad[:3]!r}"
        return None
    if predicate == "equals":
        # equals over a [*] result compares element-by-element to the spec list.
        if not isinstance(spec, Sequence) or isinstance(spec, (str, bytes)):
            return f"{path}: equals over a flat result requires a list spec, got {spec!r}"
        if list(values) == list(spec):
            return None
        return f"{path}: expected equals={spec!r}, got {values!r}"
    # Element-wise predicates (in_range, case_insensitive_contains,
    # starts_with, numeric_tolerance): apply to every value; report the first failure.
    for i, v in enumerate(values):
        sub_path = f"{path}[{i}]"
        if v is _MISSING:
            return f"{sub_path}: not found in response"
        reason = _evaluate_scalar(predicate, spec, v, path=sub_path)
        if reason is not None:
            return reason
    return None


def _evaluate_scalar(predicate: str, spec: Any, value: Any, *, path: str) -> str | None:
    if predicate == "equals":
        if value == spec:
            return None
        return f"{path}: expected equals={spec!r}, got {value!r}"
    if predicate == "in_range":
        lo, hi = spec
        try:
            v = float(value)
        except (TypeError, ValueError):
            return f"{path}: in_range requires a numeric value, got {value!r}"
        if lo <= v <= hi:
            return None
        return f"{path}: expected in_range=[{lo}, {hi}], got {v}"
    if predicate == "length":
        if not hasattr(value, "__len__"):
            return f"{path}: length requires a sized value, got {type(value).__name__}"
        actual_len = len(value)
        if isinstance(spec, int):
            if actual_len == spec:
                return None
            return f"{path}: expected length={spec}, got {actual_len}"
        lo = spec.get("min")
        hi = spec.get("max")
        if lo is not None and actual_len < lo:
            return f"{path}: expected length>={lo}, got {actual_len}"
        if hi is not None and actual_len > hi:
            return f"{path}: expected length<={hi}, got {actual_len}"
        return None
    if predicate == "case_insensitive_contains":
        if not isinstance(value, str):
            kind = type(value).__name__
            return f"{path}: case_insensitive_contains requires a string, got {kind}"
        if spec.casefold() in value.casefold():
            return None
        return f"{path}: expected case_insensitive_contains={spec!r}, got {value!r}"
    if predicate == "starts_with":
        if not isinstance(value, str):
            return f"{path}: starts_with requires a string, got {type(value).__name__}"
        if value.startswith(spec):
            return None
        return f"{path}: expected starts_with={spec!r}, got {value!r}"
    if predicate == "all_equal":
        if value == spec:
            return None
        return f"{path}: expected all_equal={spec!r}, got {value!r}"
    if predicate == "numeric_tolerance":
        return _check_numeric_tolerance(value, spec, path=path)
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
                f"{path}: numeric_tolerance length mismatch (expected {len(expected)}, "
                f"got {len(actual)})"
            )
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            reason = _check_scalar_numeric(a, e, abs_tol, rel_tol, path=f"{path}[{i}]")
            if reason is not None:
                return reason
        return None
    return _check_scalar_numeric(actual, expected, abs_tol, rel_tol, path=path)


def _check_scalar_numeric(
    actual: Any, expected: Any, abs_tol: Any, rel_tol: Any, *, path: str
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


def evaluate_checks(response: Any, checks: Sequence[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    """Apply every check in *checks* to *response*; return ``(passed, reasons)``.

    All checks are evaluated even after the first failure so PR comments
    can show the complete picture rather than just the first stumbling
    block.
    """
    reasons: list[str] = []
    for check in checks:
        reason = _evaluate(check, response)
        if reason is not None:
            reasons.append(reason)
    return (not reasons, reasons)
