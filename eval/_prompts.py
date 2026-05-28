"""Pydantic schema and YAML loader for ``eval/prompts/*.yaml``.

Every prompt is its own YAML file under :data:`PROMPTS_DIR` (excluding
files whose name begins with an underscore, which are reserved for
fixtures and helpers). :class:`PromptSpec` is the typed in-memory form;
:func:`load_prompts` walks the directory and validates everything eagerly.

Constraint and functional-predicate vocabularies are validated inline by
calling into :mod:`eval._constraints` and :mod:`eval._functional`, so a
malformed predicate fails at load time rather than the first eval run.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from astrodynamics_mcp.credentials import SOURCES
from eval._constraints import validate_constraint
from eval._functional import validate_check

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

Tier = Literal["single_tool", "sequential", "planning"]

# The env-var prefix every credential field shares. Mirrors the wire
# convention in astrodynamics_mcp.credentials (ASTRODYNAMICS_MCP_<SOURCE>_<FIELD>)
# so the runner can decide skip-vs-run without importing that module's internals.
_ENV_PREFIX = "ASTRODYNAMICS_MCP_"


class ToolCallSpec(BaseModel):
    """One step in a permitted trace — tool name plus argument constraints."""

    model_config = ConfigDict(extra="forbid")

    tool: str = Field(..., description="Registered tool name (e.g. 'tle_lookup').")
    arg_constraints: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Per-argument constraint dicts. Each value is a single-key dict from "
            "the constraint vocabulary; see eval/_constraints.py."
        ),
    )
    expect_error: str | None = Field(
        default=None,
        description=(
            "When set, the matched call must have produced a typed error envelope "
            "whose `code` equals this string (e.g. 'credential_required.spacetrack'). "
            "A successful call, or an error with a different code, does not match — "
            "this is how error-path prompts assert the tool raised the right typed "
            "failure instead of silently returning an empty success. When unset, a "
            "candidate call matches only if it did *not* error."
        ),
    )

    @field_validator("arg_constraints")
    @classmethod
    def _validate_constraints(cls, value: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        for arg_name, constraint in value.items():
            validate_constraint(constraint, path=arg_name)
        return value

    @field_validator("expect_error")
    @classmethod
    def _expect_error_non_empty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("expect_error must be a non-empty error code string when set")
        return value


class PromptSpec(BaseModel):
    """One eval prompt — natural-language ask plus the trace and answer the LLM should produce."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Slug identifying the prompt (defaults to the file stem).")
    prompt: str = Field(..., description="The natural-language user message handed to the LLM.")
    tier: Tier = Field(..., description="Difficulty tier — gates coverage counts.")
    tools_required: list[str] = Field(
        ...,
        description="Tools that must be registered for the prompt to run.",
    )
    permitted_traces: list[list[ToolCallSpec]] = Field(
        ...,
        description=(
            "List of alternative permitted tool-call sequences. The scorer "
            "passes the prompt if any one matches."
        ),
    )
    functional_answer: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "List of predicate dicts applied to the final tool response's JSON. "
            "Each entry is `{path: <jsonpath-lite>, <predicate>: <value>}`; see "
            "eval/_functional.py for the predicate vocabulary."
        ),
    )
    requires_credential: list[str] = Field(
        default_factory=list,
        description=(
            "Credential source names (see astrodynamics_mcp.credentials.SOURCES) whose "
            "environment variables must all be present for the prompt to run. When a "
            "required credential is absent the runner *skips* the prompt — reported as "
            "skipped, not failed — rather than exercising a guaranteed "
            "credential_required error. Used by the credentialed happy-path prompts."
        ),
    )
    requires_gmat: bool = Field(
        default=False,
        description=(
            "When true the prompt drives a GMAT-backed tool and only runs where a GMAT "
            "install is locatable (gmat_run importable and locate_gmat() succeeds). "
            "Skipped, not failed, when GMAT is absent."
        ),
    )
    notes: str | None = Field(
        default=None,
        description=(
            "Free-form human-only note: known model quirks, related prompts, why a "
            "particular arg constraint is loose. Not consumed by the scorer."
        ),
    )

    @field_validator("requires_credential")
    @classmethod
    def _known_credentials(cls, value: list[str]) -> list[str]:
        unknown = [s for s in value if s not in SOURCES]
        if unknown:
            raise ValueError(
                f"requires_credential lists unknown source(s) {unknown!r}; "
                f"known sources are {sorted(SOURCES)}"
            )
        return value

    @field_validator("tools_required")
    @classmethod
    def _non_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("tools_required must list at least one tool")
        return value

    @field_validator("permitted_traces")
    @classmethod
    def _at_least_one_trace(cls, value: list[list[ToolCallSpec]]) -> list[list[ToolCallSpec]]:
        if not value:
            raise ValueError("permitted_traces must contain at least one trace")
        for i, trace in enumerate(value):
            if not trace:
                raise ValueError(f"permitted_traces[{i}] must contain at least one tool-call step")
        return value

    @field_validator("functional_answer")
    @classmethod
    def _validate_functional(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for i, check in enumerate(value):
            validate_check(check, index=i)
        return value

    @model_validator(mode="after")
    def _tools_required_covers_traces(self) -> PromptSpec:
        required = set(self.tools_required)
        for i, trace in enumerate(self.permitted_traces):
            for j, step in enumerate(trace):
                if step.tool not in required:
                    raise ValueError(
                        f"permitted_traces[{i}][{j}].tool={step.tool!r} is not in "
                        f"tools_required={sorted(required)}"
                    )
        return self


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if loaded is None:
        raise ValueError(f"{path}: file is empty")
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{path}: top-level YAML must be a mapping, got {type(loaded).__name__}")
    return dict(loaded)


def load_prompt_from_yaml(path: Path) -> PromptSpec:
    """Load and validate a single prompt YAML.

    The prompt's ``id`` defaults to the file stem when not provided in YAML —
    the canonical pattern for the prompt files in ``eval/prompts/``.
    """
    data = _load_yaml(path)
    data.setdefault("id", path.stem)
    return PromptSpec.model_validate(data)


def load_prompts(directory: Path | None = None) -> list[PromptSpec]:
    """Walk *directory* (default: :data:`PROMPTS_DIR`) and load every ``*.yaml``.

    Files whose stem begins with ``_`` are skipped — that prefix is
    reserved for fixtures and helpers consumed by the test suite, not
    real eval prompts.
    """
    root = directory if directory is not None else PROMPTS_DIR
    if not root.is_dir():
        return []
    prompts: list[PromptSpec] = []
    for path in sorted(root.glob("*.yaml")):
        if path.stem.startswith("_"):
            continue
        prompts.append(load_prompt_from_yaml(path))
    return prompts


def _gmat_available() -> bool:
    """True when a GMAT install is locatable in the current process.

    Mirrors the ``gmat_installed`` pytest marker's gate: the GMAT-backed
    tools need ``gmat_run`` importable *and* a real install resolvable via
    ``locate_gmat()`` (``GMAT_ROOT`` or a platform default). Any failure —
    missing extra, missing binary — means the GMAT prompts are skipped.
    """
    try:
        from gmat_run.install import locate_gmat

        locate_gmat()
    except Exception:
        return False
    return True


def _credential_available(source: str, env: Mapping[str, str]) -> bool:
    spec = SOURCES.get(source)
    if spec is None:
        return False
    return all(env.get(f"{_ENV_PREFIX}{source.upper()}_{field.upper()}") for field in spec.fields)


def unmet_requirements(prompt: PromptSpec, env: Mapping[str, str] | None = None) -> list[str]:
    """Return the requirement tokens *prompt* does not satisfy in *env*.

    Tokens are human-readable for the CI report: ``"gmat"`` and
    ``"credential:<source>"``. An empty list means the prompt can run.
    """
    resolved_env = os.environ if env is None else env
    unmet: list[str] = []
    if prompt.requires_gmat and not _gmat_available():
        unmet.append("gmat")
    for source in prompt.requires_credential:
        if not _credential_available(source, resolved_env):
            unmet.append(f"credential:{source}")
    return unmet


def requirements_met(prompt: PromptSpec, env: Mapping[str, str] | None = None) -> bool:
    """True when every declared requirement of *prompt* is satisfiable in *env*."""
    return not unmet_requirements(prompt, env)
