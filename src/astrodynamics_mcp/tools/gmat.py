"""GMAT tool slots — registered only when the ``gmat-run`` driver is importable.

The four tools are placeholders for the v0.2 GMAT integration: each per-tool
issue (``gmat_run_mission``, ``gmat_sweep``, ``gmat_execute_script``,
``gmat_validate_script``) fills in its own body and refines its own schema.
This module owns only the conditional-registration mechanism and a uniform
description discipline so subsequent issues drop in without churning the lint
or the transport-equivalence contract.

The guard is intentionally a single ``try: import gmat_run``: ``gmat-sweep``
declares ``gmat-run`` as a dependency, so resolving the ``[gmat]`` extra
gives us both. Per the kickoff design decision (#66), there is no
transport-specific gating — the slots register identically on stdio and
Streamable HTTP, leaving the trust boundary to the operator.
"""

from __future__ import annotations

from typing import Annotated

from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from astrodynamics_mcp.server import register_tool

try:
    import gmat_run  # noqa: F401  # availability probe; the symbol itself isn't used yet

    _GMAT_RUN_AVAILABLE = True
except ImportError:
    _GMAT_RUN_AVAILABLE = False


class GmatPlaceholderResponse(BaseModel):
    """Stub response shape so FastMCP can derive a non-empty ``outputSchema``.

    Each per-tool issue replaces this with its real response model. The
    placeholder body raises before constructing one of these, so the only
    consumer is the schema generator.
    """

    model_config = ConfigDict(frozen=True)

    detail: str = Field(description="Human-readable placeholder marker; never returned at runtime.")


_RUN_MISSION_DESCRIPTION = (
    "Run a single GMAT mission script end-to-end and return structured results "
    "(report tables, ephemeris pointers, convergence flags). Placeholder slot — "
    "the body lands in a follow-up issue. e.g. gmat_run_mission(script='/path/to/hohmann.script')."
)

_SWEEP_DESCRIPTION = (
    "Run a parameter sweep or Monte Carlo over a GMAT mission script via the "
    "gmat-sweep backend. Tagged-union input (grid / samples / perturb) selects "
    "the sweep mode. Placeholder slot — the body lands in a follow-up issue. "
    "e.g. gmat_sweep(script='/path/to/hohmann.script')."
)

_EXECUTE_SCRIPT_DESCRIPTION = (
    "Escape-hatch raw GMAT script executor — minimal validation, raw output. "
    "Prefer gmat_run_mission for curated, schema-checked results. Placeholder "
    "slot — the body lands in a follow-up issue. "
    "e.g. gmat_execute_script(script='Create Spacecraft Sat;...')."
)

_VALIDATE_SCRIPT_DESCRIPTION = (
    "Parse a GMAT script without running the mission sequence; returns parse "
    "errors, unknown resources or fields, and the declared resource list. "
    "Intended for a self-correction loop before gmat_run_mission. Placeholder "
    "slot — the body lands in a follow-up issue. "
    "e.g. gmat_validate_script(script='Create Spacecraft Sat;...')."
)

_PLACEHOLDER_ANNOTATIONS = ToolAnnotations(readOnlyHint=True, openWorldHint=False)


def _register_gmat_tools() -> None:
    """Attach the four GMAT placeholder tools to ``astrodynamics_mcp.server.mcp``.

    Factored out of module top-level so unit tests can drive registration
    against a fresh :class:`~mcp.server.fastmcp.FastMCP` instance (via the
    monkeypatch-the-singleton pattern used elsewhere in the test suite)
    without relying on import-time side effects.
    """

    @register_tool(
        name="gmat_run_mission",
        description=_RUN_MISSION_DESCRIPTION,
        annotations=_PLACEHOLDER_ANNOTATIONS,
    )
    async def gmat_run_mission(
        script: Annotated[
            str,
            Field(
                description=(
                    "Path to a GMAT .script file, or the full inline script text. "
                    "Placeholder accepts the argument but does not execute it."
                ),
            ),
        ],
    ) -> GmatPlaceholderResponse:
        raise NotImplementedError("gmat_run_mission body lands in a follow-up issue")

    @register_tool(
        name="gmat_sweep",
        description=_SWEEP_DESCRIPTION,
        annotations=_PLACEHOLDER_ANNOTATIONS,
    )
    async def gmat_sweep(
        script: Annotated[
            str,
            Field(
                description=(
                    "Path to a GMAT .script file, or the full inline script text — the "
                    "sweep base mission. Placeholder accepts the argument but does not execute it."
                ),
            ),
        ],
    ) -> GmatPlaceholderResponse:
        raise NotImplementedError("gmat_sweep body lands in a follow-up issue")

    @register_tool(
        name="gmat_execute_script",
        description=_EXECUTE_SCRIPT_DESCRIPTION,
        annotations=_PLACEHOLDER_ANNOTATIONS,
    )
    async def gmat_execute_script(
        script: Annotated[
            str,
            Field(
                description=(
                    "Full GMAT script text to execute as-is. Placeholder accepts the "
                    "argument but does not execute it."
                ),
            ),
        ],
    ) -> GmatPlaceholderResponse:
        raise NotImplementedError("gmat_execute_script body lands in a follow-up issue")

    @register_tool(
        name="gmat_validate_script",
        description=_VALIDATE_SCRIPT_DESCRIPTION,
        annotations=_PLACEHOLDER_ANNOTATIONS,
    )
    async def gmat_validate_script(
        script: Annotated[
            str,
            Field(
                description=(
                    "Full GMAT script text to parse-validate without running the mission "
                    "sequence. Placeholder accepts the argument but does not execute it."
                ),
            ),
        ],
    ) -> GmatPlaceholderResponse:
        raise NotImplementedError("gmat_validate_script body lands in a follow-up issue")


if _GMAT_RUN_AVAILABLE:
    _register_gmat_tools()
