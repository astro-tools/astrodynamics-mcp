"""GMAT tool slots — registered only when the ``gmat-run`` driver is importable.

The four tools (``gmat_run_mission``, ``gmat_sweep``, ``gmat_execute_script``,
``gmat_validate_script``) cover the GMAT mission-analysis surface. This
module owns the conditional-registration mechanism, the shared description
discipline, and the real ``gmat_run_mission`` body; ``gmat_sweep``,
``gmat_execute_script``, and ``gmat_validate_script`` are still placeholder
slots whose bodies land in their own follow-up issues.

The guard is intentionally a single ``try: import gmat_run``: ``gmat-sweep``
declares ``gmat-run`` as a dependency, so resolving the ``[gmat]`` extra
gives us both. Per the kickoff design decision (#66), there is no
transport-specific gating — the slots register identically on stdio and
Streamable HTTP, leaving the trust boundary to the operator.
"""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path
from typing import Annotated, Any

from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from astrodynamics_mcp.errors import InvalidInputError, UpstreamError
from astrodynamics_mcp.server import register_tool
from astrodynamics_mcp.units import Quantity

try:
    import gmat_run  # noqa: F401  # availability probe; the symbol itself isn't used yet

    _GMAT_RUN_AVAILABLE = True
except ImportError:
    _GMAT_RUN_AVAILABLE = False


# Threshold below which a ReportFile is returned fully inline. Above it the
# response carries head + tail + a `truncated` marker; the LLM sees the
# trajectory's endpoints without paying the full payload cost. The cap is
# generous enough that a typical "few-impulse mission with a handful of
# report rows" returns whole.
_REPORT_INLINE_ROW_THRESHOLD = 20

# Head / tail size for truncated reports. Five rows on each end is enough to
# characterise a trajectory's start and end while staying well under the
# small-model input cap once tool-overhead bytes are accounted for.
_REPORT_HEAD_TAIL_ROWS = 5


class GmatPlaceholderResponse(BaseModel):
    """Stub response shape so FastMCP can derive a non-empty ``outputSchema``.

    Used by the three slots whose bodies still land in follow-up issues
    (``gmat_sweep``, ``gmat_execute_script``, ``gmat_validate_script``);
    each placeholder raises before constructing one of these, so the only
    consumer is the schema generator.
    """

    model_config = ConfigDict(frozen=True)

    detail: str = Field(description="Human-readable placeholder marker; never returned at runtime.")


class ResourceGroupView(BaseModel):
    """One resource category in :class:`MissionSummaryView`.

    Mirrors :class:`gmat_run.summary.ResourceGroup`. Names appear in the
    declaration order GMAT returned them from the configuration walk.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    category: str = Field(
        ...,
        description=(
            "Resource type — e.g. 'Spacecraft', 'ForceModel', 'Propagator', 'ReportFile'."
        ),
    )
    names: list[str] = Field(
        ...,
        description="Resource names in script-declaration order.",
    )


class ChildCommandView(BaseModel):
    """One direct child of a branch command in :class:`CommandView`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type_name: str = Field(
        ...,
        description="GMAT command type — e.g. 'Propagate', 'Maneuver', 'Vary'.",
    )
    summary: str = Field(
        ...,
        description=(
            "First non-blank line of the command's generating string, truncated. "
            "Empty when GMAT returned nothing useful."
        ),
    )


class CommandView(BaseModel):
    """One top-level mission-sequence command in :class:`MissionSummaryView`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type_name: str = Field(
        ...,
        description="GMAT command type — e.g. 'Propagate', 'Target', 'If', 'Maneuver'.",
    )
    summary: str = Field(
        ...,
        description=(
            "First non-blank line of the command's generating string, truncated. "
            "Empty when GMAT returned nothing useful."
        ),
    )
    children: list[ChildCommandView] = Field(
        default_factory=list,
        description=(
            "Direct children of a branch command (Target / Optimize / If / For / While). "
            "Empty for non-branch commands."
        ),
    )
    has_deeper_nesting: bool = Field(
        default=False,
        description=(
            "True when this branch command nests commands beyond the direct children "
            "exposed in `children`. Use this to decide whether to drill into the script."
        ),
    )


class MissionSummaryView(BaseModel):
    """Structured snapshot of the loaded mission.

    Mirrors :class:`gmat_run.summary.MissionSummary`: per-category resource
    groups plus a top-level mission-sequence outline. Field values are not
    materialised — pass overrides at the tool boundary to inspect specific
    fields, or call the GMAT API directly for advanced surfaces.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    script_name: str = Field(
        ...,
        description=(
            "File name of the loaded script (``Path.name`` of the path Mission.load resolved). "
            "When the caller passes inline script text, this is the temp file's name."
        ),
    )
    resource_groups: list[ResourceGroupView] = Field(
        ...,
        description=(
            "Resources grouped by category in display order — Spacecraft, ForceModel, "
            "Propagator, CoordinateSystem, ImpulsiveBurn, FiniteBurn, ReportFile, "
            "EphemerisFile, ContactLocator, Solver, Subscriber, Other. Empty categories "
            "are omitted."
        ),
    )
    commands: list[CommandView] = Field(
        ...,
        description=(
            "Top-level mission-sequence commands in declaration order. Branch commands "
            "(Target, Optimize, If, For, While) carry their direct children; deeper "
            "nesting is flagged via `has_deeper_nesting`."
        ),
    )


class ReportFileShape(BaseModel):
    """Shape-disciplined inline view of one ReportFile output.

    Small reports (``row_count <= 20``) come back fully inline via ``rows``;
    larger reports populate ``head`` (first five rows) and ``tail`` (last
    five rows) so the response fits small-model input caps regardless of how
    long the run was. ``truncated`` distinguishes the two modes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(
        ...,
        description="ReportFile resource name as declared in the .script.",
    )
    path: str = Field(
        ...,
        description=(
            "Absolute path the report landed at. Lives under a temp directory created "
            "for this run — the path is informational; the inline rows below carry "
            "the data the LLM can actually read."
        ),
    )
    columns: list[str] = Field(
        ...,
        description=(
            "Column headers in the order GMAT wrote them — e.g. "
            "['Sat.UTCGregorian', 'Sat.X', 'Sat.Y', 'Sat.Z']. The unit per column is "
            "carried in the GMAT parameter name itself."
        ),
    )
    row_count: Quantity = Field(
        ...,
        description=(
            "Total row count in the report (dimensionless count, unit '1'). When "
            "`truncated` is False this equals `len(rows)`; when True it equals "
            "`len(head) + len(tail)` plus the dropped middle."
        ),
        examples=[{"value": 1440.0, "unit": "1"}],
    )
    rows: list[dict[str, str | float]] = Field(
        default_factory=list,
        description=(
            "Full report content when row_count <= 20. One dict per row keyed by "
            "column name; string for non-numeric columns (e.g. UTCGregorian), float "
            "otherwise. Empty when `truncated` is True."
        ),
    )
    head: list[dict[str, str | float]] = Field(
        default_factory=list,
        description=(
            "First five rows when `truncated` is True. Same row shape as `rows`. "
            "Empty when the report fit fully inline."
        ),
    )
    tail: list[dict[str, str | float]] = Field(
        default_factory=list,
        description=(
            "Last five rows when `truncated` is True. Same row shape as `rows`. "
            "Empty when the report fit fully inline."
        ),
    )
    truncated: bool = Field(
        ...,
        description=(
            "True when the report had more than 20 rows and the response carries "
            "head + tail rather than the full data. The LLM should note the dropped "
            "middle window when interpreting trajectory shape."
        ),
    )


class OutputPointer(BaseModel):
    """Name + path for an EphemerisFile or ContactLocator output.

    Ephemerides and contact reports are intrinsically too large to inline —
    a 24-hour OEM file is several MB. The pointer carries the resource name
    so an LLM can describe the artefact, plus the absolute path for users
    who want to inspect the file outside the conversation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(
        ...,
        description="Resource name as declared in the .script (e.g. 'EphemerisFile1').",
    )
    path: str = Field(
        ...,
        description=(
            "Absolute path the file landed at. Lives under a temp directory created "
            "for this run; treat as informational, since the directory is cleaned up "
            "after the tool returns."
        ),
    )


class GmatRunMissionResponse(BaseModel):
    """Response from :func:`gmat_run_mission`.

    Carries a structured mission snapshot, the wall-clock cost of the run,
    shape-disciplined inline content for each selected ReportFile, pointers
    for the larger ephemeris and contact artefacts, and the per-solver
    convergence flags GMAT reported.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: MissionSummaryView = Field(
        ...,
        description=(
            "Resource and mission-sequence snapshot of the loaded script. Derived "
            "from gmat_run.Mission.summary() so the same structure backs both the "
            "Python API's __repr__ and this response."
        ),
    )
    wall_clock: Quantity = Field(
        ...,
        description=(
            "Wall-clock duration of mission.run() in seconds (unit 's'). Does not "
            "include script-load time."
        ),
        examples=[{"value": 0.42, "unit": "s"}],
    )
    reports: list[ReportFileShape] = Field(
        ...,
        description=(
            "One :class:`ReportFileShape` per selected ReportFile output. Filtered "
            "by the caller's `select_outputs` if provided; otherwise every ReportFile "
            "the run wrote is included."
        ),
    )
    ephemerides: list[OutputPointer] = Field(
        ...,
        description=(
            "One pointer per selected EphemerisFile output. No inline data — "
            "ephemerides are too large to fit a single tool response."
        ),
    )
    contacts: list[OutputPointer] = Field(
        ...,
        description=(
            "One pointer per selected ContactLocator output. No inline data — the "
            "contact report formats vary widely and are best inspected on disk."
        ),
    )
    converged: dict[str, bool] = Field(
        ...,
        description=(
            "Per-solver convergence flag keyed by solver resource name (e.g. {'DC': "
            "True}). Empty when the mission declares no solvers."
        ),
    )


_RUN_MISSION_DESCRIPTION = (
    "Run a single GMAT mission script end-to-end and return structured results "
    "(report tables, ephemeris pointers, convergence flags). e.g. "
    "gmat_run_mission(script='/abs/path/to/Ex_HohmannTransfer.script') runs a "
    "stock GMAT Hohmann transfer sample and returns the ReportFile data inline "
    "plus the DifferentialCorrector's converged flag. `script` must be either an "
    "absolute path to a .script file or the full inline script text (auto-detected "
    "by leading '%' / 'Create' markers); do not pass a Python Mission object. "
    "`overrides` apply dotted-path field writes (e.g. {'Sat.SMA': 7000.0}) using "
    "the same grammar as the GMAT script — Sat.SMA not Sat['SMA'], and the value's "
    "Python type must match the field's GMAT type. `select_outputs` filters which "
    "ReportFile / EphemerisFile / ContactLocator outputs appear in the response; "
    "leave it null to return every output. Engine failures (script parse errors, "
    "RunScript errors) surface as upstream.gmat_run_* error codes; invalid override "
    "paths surface as invalid_input.gmat_override_*."
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


def _looks_like_inline_script(text: str) -> bool:
    """Heuristic: True if ``text`` is GMAT script content rather than a path.

    GMAT scripts always either begin with a ``%`` comment block or with a
    ``Create`` resource line, and most carry newlines. A bare path is a
    single line that mentions neither. The auto-detect is conservative —
    when it guesses "inline" but the string was meant as a path, the tool
    body writes a temp file the user did not intend, but the run still
    succeeds against that file. When it guesses "path" but the string was
    inline script, the tool fails fast with a path-not-found error so the
    caller can correct the call.
    """
    if "\n" in text:
        return True
    stripped = text.lstrip()
    return stripped.startswith("%") or stripped.startswith("Create ")


def _resolve_script_input(script: str) -> tuple[Path, Path | None]:
    """Return ``(script_path, cleanup_path)`` for the caller's ``script`` argument.

    Path input: ``script_path`` is the supplied path and ``cleanup_path`` is
    ``None``. Inline text: ``script_path`` is the path of a freshly-written
    temp file and ``cleanup_path`` is that same path so the caller can
    unlink it once :meth:`Mission.load` is done. The temp file is created
    with ``delete=False`` so Windows GMAT can re-open it under its own
    handle.
    """
    if _looks_like_inline_script(script):
        # ``delete=False`` so Windows GMAT can re-open the path; the caller
        # is responsible for cleaning the file up via the returned path.
        fd, tmp_name = tempfile.mkstemp(suffix=".script", text=True)
        try:
            with open(fd, "w", encoding="utf-8") as f:
                f.write(script)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        tmp_path = Path(tmp_name)
        return tmp_path, tmp_path
    path = Path(script)
    if not path.is_absolute():
        raise InvalidInputError(
            f"script path must be absolute, got {script!r}; pass the full path "
            "or the inline script text",
            code="invalid_input.script_path_not_absolute",
        )
    if not path.is_file():
        raise InvalidInputError(
            f"script path {script!r} does not exist or is not a regular file",
            code="invalid_input.script_path_not_found",
        )
    return path, None


def _build_mission_summary_view(mission: Any) -> MissionSummaryView:
    """Render :class:`gmat_run.summary.MissionSummary` into the response shape."""
    summary = mission.summary()
    resource_groups = [
        ResourceGroupView(category=g.category, names=list(g.names)) for g in summary.resource_groups
    ]
    commands = [
        CommandView(
            type_name=cmd.type_name,
            summary=cmd.summary,
            children=[
                ChildCommandView(type_name=c.type_name, summary=c.summary) for c in cmd.children
            ],
            has_deeper_nesting=cmd.nested_count > 0,
        )
        for cmd in summary.commands
    ]
    return MissionSummaryView(
        script_name=summary.script_name,
        resource_groups=resource_groups,
        commands=commands,
    )


def _cell_value(value: Any) -> str | float:
    """Coerce a DataFrame cell to a JSON-safe scalar.

    NaN floats round-trip badly through JSON (``NaN`` is not valid JSON
    per RFC 8259) so they collapse to the string ``"NaN"`` — preserving
    the information without breaking strict-JSON consumers downstream.
    """
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return str(value)
        return f
    return str(value)


def _row_to_dict(columns: list[str], row: Any) -> dict[str, str | float]:
    """Convert one DataFrame row (positional) to a column-keyed dict."""
    return {col: _cell_value(row[i]) for i, col in enumerate(columns)}


def _shape_report(name: str, path: Path, frame: Any) -> ReportFileShape:
    """Build a :class:`ReportFileShape` from a parsed ReportFile DataFrame."""
    columns = [str(c) for c in frame.columns]
    row_count = len(frame.index)
    values = frame.to_numpy(dtype=object)
    if row_count <= _REPORT_INLINE_ROW_THRESHOLD:
        rows = [_row_to_dict(columns, values[i]) for i in range(row_count)]
        head: list[dict[str, str | float]] = []
        tail: list[dict[str, str | float]] = []
        truncated = False
    else:
        rows = []
        head = [_row_to_dict(columns, values[i]) for i in range(_REPORT_HEAD_TAIL_ROWS)]
        tail = [
            _row_to_dict(columns, values[i])
            for i in range(row_count - _REPORT_HEAD_TAIL_ROWS, row_count)
        ]
        truncated = True
    return ReportFileShape(
        name=name,
        path=str(path),
        columns=columns,
        row_count=Quantity(value=float(row_count), unit="1"),
        rows=rows,
        head=head,
        tail=tail,
        truncated=truncated,
    )


def _select_keys(all_keys: list[str], selection: list[str] | None) -> tuple[list[str], list[str]]:
    """Return ``(kept, unknown)`` after applying ``selection`` to ``all_keys``.

    ``None`` means "everything"; the unknown list is empty. With a
    selection, ``kept`` preserves the caller's order so a deliberate
    ordering is honoured, and ``unknown`` carries names that didn't match
    any output for a typed error downstream.
    """
    if selection is None:
        return list(all_keys), []
    known = set(all_keys)
    kept = [name for name in selection if name in known]
    unknown = [name for name in selection if name not in known]
    return kept, unknown


def _apply_overrides(mission: Any, overrides: dict[str, Any]) -> None:
    """Apply each override via ``mission[key] = value``, wrapping errors.

    gmat-run raises ``GmatFieldError`` for an unknown resource, an unknown
    field, or a type-coercion failure. We surface those as typed
    ``InvalidInputError`` so the LLM consumer sees a stable code matching
    the failure category.
    """
    if not overrides:
        return
    from gmat_run.errors import GmatFieldError

    for dotted, value in overrides.items():
        try:
            mission[dotted] = value
        except GmatFieldError as exc:
            raise InvalidInputError(
                f"override write to {dotted!r} failed: {exc}",
                code="invalid_input.gmat_override_failed",
                data={"path": dotted, "original_exception_message": str(exc)},
            ) from exc


def _register_gmat_tools() -> None:
    """Attach the four GMAT tools to ``astrodynamics_mcp.server.mcp``.

    Factored out of module top-level so unit tests can drive registration
    against a fresh :class:`~mcp.server.fastmcp.FastMCP` instance (via the
    monkeypatch-the-singleton pattern used elsewhere in the test suite)
    without relying on import-time side effects.
    """

    @register_tool(
        name="gmat_run_mission",
        description=_RUN_MISSION_DESCRIPTION,
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    )
    async def gmat_run_mission(
        script: Annotated[
            str,
            Field(
                description=(
                    "Either the absolute path to a GMAT .script file (e.g. "
                    "'/abs/path/to/Ex_HohmannTransfer.script') or the full inline "
                    "script text starting with '%' comments or 'Create' resource "
                    "declarations. Auto-detected by content: a string with newlines "
                    "or a leading '%' / 'Create ' is inline, anything else is treated "
                    "as a path. Do not pass a Python Mission object."
                ),
            ),
        ],
        overrides: Annotated[
            dict[str, Any] | None,
            Field(
                description=(
                    "Dotted-path field writes applied to the loaded mission before "
                    "running. Keys use GMAT's script grammar (e.g. 'Sat.SMA', "
                    "'FM.Drag.AtmosphereModel', 'Var1.Value'), not subscript form. "
                    "Values must match the field's GMAT type — numbers for real / "
                    "integer fields, booleans for bool fields, strings for "
                    "filename / enumeration fields. Leave null to skip overrides."
                ),
            ),
        ] = None,
        select_outputs: Annotated[
            list[str] | None,
            Field(
                description=(
                    "Optional list of output resource names to include in the "
                    "response (e.g. ['ReportFile1', 'EphemerisFile1']). "
                    "Leave null to return every ReportFile / EphemerisFile / "
                    "ContactLocator the run produced; supply an explicit list to "
                    "trim the response for a small-context LLM."
                ),
            ),
        ] = None,
    ) -> GmatRunMissionResponse:
        from gmat_run import Mission
        from gmat_run.errors import GmatError, GmatLoadError, GmatRunError

        script_path, cleanup_path = _resolve_script_input(script)
        try:
            try:
                mission = Mission.load(script_path)
            except GmatLoadError as exc:
                raise UpstreamError(
                    f"GMAT could not load script: {exc}",
                    code="upstream.gmat_run_load_failed",
                    original_exception=exc,
                ) from exc
            except GmatError as exc:
                raise UpstreamError(
                    f"GMAT discovery / bootstrap failed: {exc}",
                    code="upstream.gmat_run_bootstrap_failed",
                    original_exception=exc,
                ) from exc

            _apply_overrides(mission, overrides or {})

            t0 = time.perf_counter()
            try:
                result = mission.run()
            except GmatRunError as exc:
                raise UpstreamError(
                    f"GMAT mission run failed: {exc}",
                    code="upstream.gmat_run_failed",
                    original_exception=exc,
                ) from exc
            wall_clock_s = time.perf_counter() - t0

            return _build_response(
                mission=mission,
                result=result,
                wall_clock_s=wall_clock_s,
                select_outputs=select_outputs,
            )
        finally:
            if cleanup_path is not None:
                cleanup_path.unlink(missing_ok=True)

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


def _build_response(
    *,
    mission: Any,
    result: Any,
    wall_clock_s: float,
    select_outputs: list[str] | None,
) -> GmatRunMissionResponse:
    """Assemble a :class:`GmatRunMissionResponse` from a finished run.

    Pulled out of the tool body so the unit tests can exercise the
    shaping logic directly against a fake ``result`` without round-
    tripping through ``Mission.load`` / ``mission.run``.
    """
    summary_view = _build_mission_summary_view(mission)

    all_output_names = [
        *result.report_paths.keys(),
        *result.ephemeris_paths.keys(),
        *result.contact_paths.keys(),
    ]
    _, unknown = _select_keys(all_output_names, select_outputs)
    if unknown:
        raise InvalidInputError(
            f"select_outputs contains unknown resource names: {unknown}; "
            f"available outputs are {sorted(all_output_names)}",
            code="invalid_input.unknown_output_selection",
            data={"unknown": unknown, "available": sorted(all_output_names)},
        )

    report_names, _ = _select_keys(list(result.report_paths.keys()), select_outputs)
    ephemeris_names, _ = _select_keys(list(result.ephemeris_paths.keys()), select_outputs)
    contact_names, _ = _select_keys(list(result.contact_paths.keys()), select_outputs)

    reports: list[ReportFileShape] = []
    for name in report_names:
        frame = result.reports[name]
        reports.append(_shape_report(name, result.report_paths[name], frame))

    ephemerides = [
        OutputPointer(name=name, path=str(result.ephemeris_paths[name])) for name in ephemeris_names
    ]
    contacts = [
        OutputPointer(name=name, path=str(result.contact_paths[name])) for name in contact_names
    ]

    return GmatRunMissionResponse(
        summary=summary_view,
        wall_clock=Quantity(value=float(wall_clock_s), unit="s"),
        reports=reports,
        ephemerides=ephemerides,
        contacts=contacts,
        converged=dict(result.converged),
    )


if _GMAT_RUN_AVAILABLE:
    _register_gmat_tools()
