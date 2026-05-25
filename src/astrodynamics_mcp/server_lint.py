"""Static description-quality lint for registered tools.

The eval suite is the *runtime* check on tool descriptions — it measures
how well frontier LLMs call our tools under prompt variation. This module
is the cheap *static* check that runs in CI's ``pytest`` and catches the
obvious failure modes before they ever reach an LLM:

- A description too short to be informative (one-liner stubs).
- A description with no worked example values — the LLM has to guess
  realistic inputs from prose.
- A description that omits a known-finicky argument's common-mistake
  warning (e.g. an ``epoch`` arg with no mention of ISO 8601).

The numeric-arg unit discipline (every numeric input wrapped in
``Quantity`` / ``QuantityVector``) is *not* checked here — that's the
unit-discipline meta-test's job, and it operates on the JSON schema
rather than the natural-language description.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from mcp.types import Tool
from pydantic import BaseModel, ConfigDict

# Substring patterns the description must contain when the tool's input
# schema mentions one of these argument names. The pattern is case-
# insensitive; multiple alternatives in a tuple mean "any one suffices".
# Tool issues extend this registry as new finicky arg names appear.
COMMON_MISTAKE_HINTS: dict[str, tuple[str, ...]] = {
    "epoch": ("ISO 8601",),
    "epochs": ("ISO 8601",),
    "frame": ("TEME", "ICRF", "GCRS"),
    "from_frame": ("TEME", "ICRF", "GCRS"),
    "to_frame": ("TEME", "ICRF", "GCRS"),
}


# Heuristics for "this description has a worked example". Tool authors
# whose docstrings naturally use any of these phrases satisfy the rule;
# anyone writing fresh prose can drop one in deliberately.
_EXAMPLE_MARKERS: tuple[str, ...] = ("example", "e.g.", "for instance")


# Below this length the description can't carry the example + units + warning
# discipline; we treat it as a stub by definition.
_MIN_DESCRIPTION_CHARS = 50


class DescriptionViolation(BaseModel):
    """A single failed-rule record returned by :func:`check_tool_descriptions`."""

    model_config = ConfigDict(frozen=True)

    tool_name: str
    rule: Literal["too_short", "missing_example", "missing_common_mistake_warning"]
    detail: str


def _input_arg_names(tool: Tool) -> set[str]:
    """Extract the input-schema's top-level property names for *tool*."""
    schema: Any = tool.inputSchema
    if not isinstance(schema, dict):
        return set()
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return set()
    return set(properties.keys())


def _description_text(tool: Tool) -> str:
    return tool.description or ""


def _check_one(tool: Tool) -> list[DescriptionViolation]:
    violations: list[DescriptionViolation] = []
    description = _description_text(tool)

    # Rule 1 — minimum length.
    if len(description) < _MIN_DESCRIPTION_CHARS:
        violations.append(
            DescriptionViolation(
                tool_name=tool.name,
                rule="too_short",
                detail=(
                    f"description is {len(description)} characters; needs at least "
                    f"{_MIN_DESCRIPTION_CHARS} to carry example + unit + warning discipline"
                ),
            )
        )

    # Rule 2 — must include at least one worked-example marker.
    description_lc = description.lower()
    if not any(marker in description_lc for marker in _EXAMPLE_MARKERS):
        violations.append(
            DescriptionViolation(
                tool_name=tool.name,
                rule="missing_example",
                detail=(
                    "description has no worked example marker — include one of "
                    f"{list(_EXAMPLE_MARKERS)} so the LLM sees a realistic call shape"
                ),
            )
        )

    # Rule 3 — common-mistake warnings for known finicky arg names.
    arg_names = _input_arg_names(tool)
    for arg_name, required_alternatives in COMMON_MISTAKE_HINTS.items():
        if arg_name not in arg_names:
            continue
        if not any(alt.lower() in description_lc for alt in required_alternatives):
            violations.append(
                DescriptionViolation(
                    tool_name=tool.name,
                    rule="missing_common_mistake_warning",
                    detail=(
                        f"input arg {arg_name!r} requires the description to mention one of "
                        f"{list(required_alternatives)} (the canonical common-mistake warning "
                        "for this arg type)"
                    ),
                )
            )

    return violations


def check_tool_descriptions(tools: Iterable[Tool]) -> list[DescriptionViolation]:
    """Return every description-rule violation across *tools*.

    An empty list means the surface satisfies the v0.1 description
    discipline. The check is intentionally cheap — every rule is a
    substring/length test — so it can run in every PR's ``pytest`` without
    slowing the suite down.
    """
    return [v for tool in tools for v in _check_one(tool)]
