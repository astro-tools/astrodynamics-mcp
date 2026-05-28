"""GMAT tool slots — registered only when the ``gmat-run`` driver is importable.

The four tools (``gmat_run_mission``, ``gmat_sweep``, ``gmat_execute_script``,
``gmat_validate_script``) cover the GMAT mission-analysis surface. This
module owns the conditional-registration mechanism, the shared description
discipline, and the real bodies for every slot.

The guard is intentionally a single ``try: import gmat_run``: ``gmat-sweep``
declares ``gmat-run`` as a dependency, so resolving the ``[gmat]`` extra
gives us both. Per the kickoff design decision, there is no
transport-specific gating — the slots register identically on stdio and
Streamable HTTP, leaving the trust boundary to the operator.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from importlib import resources
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from astrodynamics_mcp import server as _server_module
from astrodynamics_mcp.errors import InvalidInputError, UpstreamError
from astrodynamics_mcp.runs import default_registry
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

# Line-based equivalents for the raw-text shape gmat_execute_script returns.
# A ReportFile is one row per line plus a header; 60 lines covers the
# common "short solver / Hohmann transfer" case end-to-end while keeping
# the response under the ~2 KB small-model target.
_RAW_REPORT_INLINE_LINE_THRESHOLD = 60
_RAW_REPORT_HEAD_TAIL_LINES = 20

# How many bytes gmat_read_run_artefact sniffs from the head of a file
# to decide text-vs-binary. 8 KB is the grep / git default; large enough
# to span any plausible text-format header without paying full-file
# read cost on a multi-hundred-MB SPK ephemeris.
_BINARY_SNIFF_BYTES = 8192


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

    In the default ``output="summary"`` mode, small reports (``row_count
    <= 20``) come back fully inline via ``rows`` while larger reports
    populate ``head`` (first five rows) and ``tail`` (last five rows) so
    the response fits small-model input caps regardless of how long the
    run was. In ``output="full"`` mode the response always carries every
    row regardless of size; ``head`` / ``tail`` are unused there.
    ``truncated`` distinguishes the two cases at read time.
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

    run_id: str = Field(
        ...,
        description=(
            "UUID4 hex identifying this run in the server-process registry. Pass to "
            "gmat_read_run_artefact along with an output's resource name to read the "
            "raw bytes in a later tool call — useful when an ephemeris or contact "
            "report was too large to inline here. The registry retains the last N "
            "runs per process (configurable via ASTRODYNAMICS_MCP_RUN_REGISTRY_LIMIT, "
            "default 50); once evicted the id resolves to invalid_input.unknown_run_id."
        ),
    )
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


class RawReportContent(BaseModel):
    """Raw text view of one ReportFile output for :func:`gmat_execute_script`.

    The escape hatch returns the report verbatim — header line, units row,
    data — instead of parsing into a DataFrame the way
    :class:`ReportFileShape` does. In ``output="summary"`` mode a short
    report (``line_count <= 60``) lands inline via ``content``; a longer
    report populates ``head`` (first 20 lines) and ``tail`` (last 20
    lines) so the response stays under small-model input caps regardless
    of how long the run was. In ``output="full"`` mode ``content`` always
    carries the entire file. ``truncated`` distinguishes the two cases.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(
        ...,
        description="ReportFile resource name as declared in the .script.",
    )
    path: str = Field(
        ...,
        description=(
            "Absolute path the report landed at. Lives under a temp directory "
            "created for this run — informational only; the file is cleaned up "
            "when the tool returns, so the raw text below is what the caller "
            "can actually read."
        ),
    )
    content: str = Field(
        default="",
        description=(
            "Full report text when `truncated` is False, joined with the file's "
            "original newlines. Empty when `truncated` is True — read `head` "
            "and `tail` instead."
        ),
    )
    head: str = Field(
        default="",
        description=(
            "First 20 lines joined with newlines when `truncated` is True. "
            "Empty when the report fit fully inline via `content`."
        ),
    )
    tail: str = Field(
        default="",
        description=(
            "Last 20 lines joined with newlines when `truncated` is True. "
            "Empty when the report fit fully inline via `content`."
        ),
    )
    line_count: Quantity = Field(
        ...,
        description=(
            "Total line count of the file (dimensionless count, unit '1'). "
            "Trailing-newline-only files count their data lines, not an "
            "empty terminal line."
        ),
        examples=[{"value": 1440.0, "unit": "1"}],
    )
    byte_count: Quantity = Field(
        ...,
        description=(
            "Total size of the file on disk in bytes (dimensionless count, "
            "unit '1'). Captured before any UTF-8 decoding."
        ),
        examples=[{"value": 87432.0, "unit": "1"}],
    )
    truncated: bool = Field(
        ...,
        description=(
            "True when the report had more than 60 lines under output='summary' "
            "and the response carries head + tail rather than the full text. "
            "False under output='full' or when the report fit fully inline."
        ),
    )


class GmatExecuteScriptResponse(BaseModel):
    """Response from :func:`gmat_execute_script`.

    Minimal-validation escape hatch: carries the success / failure status,
    GMAT's captured log output, raw text of every ReportFile the run
    wrote, and a pointer-only list of every other artefact that landed in
    the run's output directory. ``ok=False`` surfaces engine failures as
    data — the caller inspects ``stderr`` rather than catching an
    exception — so an LLM can introspect a failing script the same way a
    human reads the GMAT log.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(
        ...,
        description=(
            "UUID4 hex identifying this run in the server-process registry. Pass to "
            "gmat_read_run_artefact along with an artefact's resource name or "
            "basename to read the raw bytes in a later tool call. Populated on both "
            "ok=True and ok=False so a follow-up read can inspect partial outputs "
            "even after an engine failure; for failures the registry usually carries "
            "only the GMAT log."
        ),
    )
    ok: bool = Field(
        ...,
        description=(
            "True when the mission sequence completed without raising a "
            "GmatRunError. False on a GMAT engine failure mid-run — read "
            "`stderr` for GMAT's diagnostic; `reports` and `artefacts` are "
            "empty in that case. Pre-run input failures (path-not-found, "
            "GMAT install missing) still raise typed errors rather than "
            "returning ok=False."
        ),
    )
    stderr: str = Field(
        ...,
        description=(
            "GMAT's captured stdout / stderr log from the run, verbatim. "
            "Always populated: on a successful run it carries warnings and "
            "the solver iteration trace; on a failed run it carries the "
            "engine error and is the primary signal for the caller. Empty "
            "string when GMAT wrote nothing to stderr (rare)."
        ),
    )
    wall_clock: Quantity = Field(
        ...,
        description=(
            "Wall-clock duration of mission.run() in seconds (unit 's'). "
            "Captured even on failure so the caller can tell a fast crash "
            "apart from a long-running run that diverged."
        ),
        examples=[{"value": 0.42, "unit": "s"}],
    )
    reports: list[RawReportContent] = Field(
        default_factory=list,
        description=(
            "One :class:`RawReportContent` per ReportFile the run wrote, in "
            "the order GMAT declared them. Empty when ok=False, since the "
            "run did not reach the point of writing reports."
        ),
    )
    artefacts: list[OutputPointer] = Field(
        default_factory=list,
        description=(
            "Every regular file the run wrote under its output directory, "
            "deduplicated and sorted by path. ReportFile / EphemerisFile / "
            "ContactLocator / Solver outputs carry their declared resource "
            "name; stray files (e.g. the GMAT log if it landed on disk) "
            "fall back to their basename. Pointer-only — read `reports` for "
            "ReportFile content inline, or copy the path yourself before "
            "the tool returns."
        ),
    )


class SweepColumnStats(BaseModel):
    """Per-column summary statistics over ``ok`` rows of a sweep result frame."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    column: str = Field(
        ...,
        description=(
            "Result-frame column name as written by the per-run ReportFile — e.g. "
            "'Sat.X', 'Sat.SMA'. The unit per column is carried in the GMAT "
            "parameter name itself."
        ),
    )
    count: Quantity = Field(
        ...,
        description=(
            "Number of finite values used to compute the stats (dimensionless count, "
            "unit '1'). Rows from failed / skipped runs and any NaN cells are excluded."
        ),
        examples=[{"value": 60.0, "unit": "1"}],
    )
    mean: float = Field(..., description="Arithmetic mean over finite values.")
    std: float = Field(
        ...,
        description=(
            "Sample standard deviation (ddof=1) over finite values. NaN when "
            "`count` < 2 — single-row sweeps have no spread to report."
        ),
    )
    min: float = Field(..., description="Minimum finite value.")
    max: float = Field(..., description="Maximum finite value.")


class SweepStatusCounts(BaseModel):
    """Run-status tally derived from the result frame's ``__status`` column."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: Quantity = Field(
        ...,
        description="Number of runs that completed successfully (count, unit '1').",
        examples=[{"value": 10.0, "unit": "1"}],
    )
    failed: Quantity = Field(
        ...,
        description=(
            "Number of runs the worker reported as failed (count, unit '1'). "
            "Each contributes one NaN-filled row to the result frame."
        ),
        examples=[{"value": 0.0, "unit": "1"}],
    )
    skipped: Quantity = Field(
        ...,
        description=(
            "Number of runs the worker reported as skipped (count, unit '1'). "
            "Same single-row NaN representation as failed runs."
        ),
        examples=[{"value": 0.0, "unit": "1"}],
    )


class GmatSweepResponse(BaseModel):
    """Response from :func:`gmat_sweep`.

    Carries the ``mode`` echo, sweep-level metadata, per-column summary
    statistics computed over ``ok`` rows, a status tally, the head + tail
    of the ``(run_id, time)``-MultiIndexed result frame, and pointers to
    the on-disk manifest and output directory so a follow-up tool call
    can re-load the full sweep.

    In the default ``output="summary"`` mode ``rows`` is empty and the
    response carries only ``head`` + ``tail`` (first / last five rows
    each); ``output="full"`` populates ``rows`` with every result row.
    ``truncated`` distinguishes the two cases at read time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(
        ...,
        description=(
            "UUID4 hex identifying this sweep in the server-process registry. Pass "
            "to gmat_read_run_artefact along with 'manifest.jsonl' or a per-run "
            "artefact basename to read the raw bytes in a later tool call. The same "
            "retention cap (ASTRODYNAMICS_MCP_RUN_REGISTRY_LIMIT, default 50) covers "
            "single-mission runs and sweeps."
        ),
    )
    mode: Literal["grid", "samples", "monte_carlo", "latin_hypercube"] = Field(
        ...,
        description=(
            "Echo of the sweep backend that ran — matches the `mode` input arg. "
            "Stable across the response so a caller can switch on it without "
            "re-tracking the request."
        ),
    )
    script_name: str = Field(
        ...,
        description=(
            "File name of the loaded script (``Path.name`` of the input). When "
            "the caller passed inline script text, this is the temp file's name."
        ),
    )
    run_count: Quantity = Field(
        ...,
        description=(
            "Total runs the sweep dispatched (count, unit '1'). Equals "
            "ok + failed + skipped from `status_counts`."
        ),
        examples=[{"value": 10.0, "unit": "1"}],
    )
    wall_clock: Quantity = Field(
        ...,
        description=(
            "Wall-clock duration of the sweep dispatch in seconds (unit 's'). "
            "Includes per-run worker overhead but excludes the time to write "
            "the response."
        ),
        examples=[{"value": 12.4, "unit": "s"}],
    )
    columns: list[str] = Field(
        ...,
        description=(
            "Non-status data columns of the result frame, in the order GMAT wrote "
            "them — e.g. ['Sat.X', 'Sat.Y', 'Sat.Z', 'Sat.SMA']. The `__status` "
            "column is excluded; it is summarised in `status_counts` instead."
        ),
    )
    status_counts: SweepStatusCounts = Field(
        ...,
        description=(
            "Per-status run tally (ok / failed / skipped). Failed and skipped "
            "runs land as one NaN-filled row apiece in the result frame and are "
            "excluded from `summary_stats`."
        ),
    )
    summary_stats: list[SweepColumnStats] = Field(
        ...,
        description=(
            "One :class:`SweepColumnStats` per numeric column in `columns`, computed "
            "over the finite cells of ``ok`` rows only. Non-numeric columns (string "
            "epochs, categorical fields) are omitted — they appear in `head` / "
            "`tail` / `rows` but not here."
        ),
    )
    head: list[dict[str, str | float]] = Field(
        ...,
        description=(
            "First five rows of the ``(run_id, time)``-indexed result frame, sorted "
            "by (run_id, time). Each dict carries 'run_id', 'time', and one entry "
            "per column in `columns`; NaN cells become the string 'nan'. Always "
            "populated regardless of `output` mode."
        ),
    )
    tail: list[dict[str, str | float]] = Field(
        ...,
        description=(
            "Last five rows of the result frame, sorted the same way. Same row "
            "shape as `head`. Always populated regardless of `output` mode; will "
            "overlap `head` when the frame has ≤ 10 rows."
        ),
    )
    rows: list[dict[str, str | float]] = Field(
        default_factory=list,
        description=(
            "Every row of the result frame when ``output='full'``. Same row shape "
            "as `head` / `tail`. Empty in the default ``output='summary'`` mode — "
            "use `head` + `tail` there."
        ),
    )
    truncated: bool = Field(
        ...,
        description=(
            "True when ``output='summary'`` and the result frame has more than "
            "ten rows so `rows` is intentionally empty. False when the full frame "
            "fits in `head` + `tail` or when ``output='full'`` populated `rows`."
        ),
    )
    manifest_path: str = Field(
        ...,
        description=(
            "Absolute path to the JSON Lines manifest the sweep wrote. A follow-up "
            "call can re-load it via :func:`gmat_sweep.Manifest.load` to walk the "
            "per-run outputs that this response summarises."
        ),
    )
    output_dir: str = Field(
        ...,
        description=(
            "Absolute path to the sweep's output directory (parent of "
            "`manifest_path` and of every per-run Parquet). Lives for the server "
            "process's lifetime; safe to read after the tool call returns."
        ),
    )


class ParseDiagnostic(BaseModel):
    """One error or warning extracted from GMAT's load-time log.

    Shared shape across :attr:`GmatValidateScriptResponse.errors` and
    :attr:`GmatValidateScriptResponse.warnings` — both surfaces carry the
    same ``{line, message, raw}`` triple, classified by which GMAT marker
    produced them (``**** ERROR **** Interpreter Exception:`` → error;
    ``*** WARNING ***`` → warning).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    line: int | None = Field(
        ...,
        description=(
            "Line number in the source script GMAT attributed the diagnostic to. "
            'Populated when GMAT\'s message carries an ``in line: "<N>: ..."`` '
            "trailing context; null for cross-resource reference errors and "
            "warnings that don't pin a single line."
        ),
    )
    message: str = Field(
        ...,
        description=(
            "Cleaned diagnostic text — the substantive portion of GMAT's message "
            "with the script-path prefix and the ``in line:`` continuation stripped. "
            'Action-targetable; e.g. \'The field name "WidgetCount" on object '
            '"Sat" is not permitted\'.'
        ),
    )
    raw: str = Field(
        ...,
        description=(
            "Original log line GMAT emitted, verbatim. Fallback when the scraper "
            "trims information the caller needs (e.g. the full multi-marker line "
            "for non-ASCII errors)."
        ),
    )


class GmatValidateScriptResponse(BaseModel):
    """Response from :func:`gmat_validate_script`.

    Carries GMAT's view of the script: a parse-success boolean, error and
    warning lists scraped from the engine's log, the resource and command
    inventory the parser built on success, and the captured log verbatim
    as a fallback. No numeric fields — the tool is structural, not
    physical, so the response is deliberately exempt from the cross-tool
    ``{value, unit}`` unit discipline.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool = Field(
        ...,
        description=(
            "True when GMAT's interpreter loaded the script (LoadScript returned "
            "True) and the post-load Spacecraft initialisation didn't raise. "
            "False on any parse-time error. Known false-negative: GMAT's "
            "interpreter silently accepts missing statement terminators "
            "(semicolons), so ok=True means the engine built an object graph, "
            "not that the script is syntactically pristine."
        ),
    )
    errors: list[ParseDiagnostic] = Field(
        ...,
        description=(
            "Parse-time errors GMAT surfaced — ``**** ERROR **** Interpreter "
            "Exception:`` lines plus any Spacecraft-initialisation APIException. "
            "Empty when ok=True."
        ),
    )
    warnings: list[ParseDiagnostic] = Field(
        ...,
        description=(
            "Parse-time warnings GMAT surfaced — ``*** WARNING ***`` lines. "
            "Independent of `ok`: GMAT can warn (e.g. missing BeginMissionSequence) "
            "while still loading the script successfully."
        ),
    )
    summary: MissionSummaryView | None = Field(
        ...,
        description=(
            "Structured snapshot of what GMAT parsed — resource categories and "
            "the mission-sequence command outline. Populated on ok=True so the "
            "caller can confirm the parser saw what it intended to declare. Null "
            "on ok=False, since the moderator's state after a failed load is "
            "indeterminate."
        ),
    )
    raw_log: str = Field(
        ...,
        description=(
            "Verbatim GMAT log captured across the LoadScript call. Includes "
            "build-date headers, ephemeris-source notices, and the error / "
            "warning lines the structured fields are scraped from. Read this "
            "when the structured fields miss something — the scraper trades "
            "robustness for tidiness."
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
    "leave it null to return every output. `output` controls row shaping for "
    "ReportFile data: the default 'summary' inlines small reports and trims large "
    "ones to first/last five rows so the response fits small-model input caps; "
    "'full' returns every row of every selected report. Engine failures (script "
    "parse errors, RunScript errors) surface as upstream.gmat_run_* error codes; "
    "invalid override paths surface as invalid_input.gmat_override_*."
)

_SWEEP_DESCRIPTION = (
    "Run a parameter sweep, Monte Carlo, or Latin hypercube over a GMAT mission "
    "script via the gmat-sweep backend. `mode` selects which family runs and which "
    "payload fields are read. e.g. gmat_sweep(script='/abs/path/to/hohmann.script', "
    "mode='grid', grid={'Sat.SMA': [7000, 7100, 7200], 'Sat.INC': [28.5, 51.6]}) "
    "runs a 6-point full factorial; gmat_sweep(script=..., mode='monte_carlo', "
    "perturb={'Sat.SMA': ['normal', 7000, 5.0]}, n=20, seed=42) runs 20 normally-"
    "dispersed runs; mode='latin_hypercube' takes the same perturb / n / seed and "
    "uses a stratified design. `script` is the same shape as gmat_run_mission: an "
    "absolute .script path or full inline script text. Perturb values are JSON "
    "lists of the form [distribution_name, *params] — ['normal', mu, sigma], "
    "['uniform', lo, hi], or ['lognormal', mu, sigma]; do not pass plain numbers "
    "or scipy distributions. For Monte Carlo / Latin hypercube, `seed` is required "
    "for reproducibility — omitting it falls back to OS entropy and two calls with "
    "the same arguments will give different draws. `max_workers` defaults to 1 to "
    "keep the cost ceiling tight; raise it explicitly to parallelise across cores. "
    "Output shaping: the default output='summary' returns per-column mean / std / "
    "min / max plus the head + tail five rows of the result frame so the response "
    "fits small-model input caps. 'full' adds every row in `rows`. `manifest_path` "
    "and `output_dir` point at the on-disk sweep artefacts for a follow-up re-load. "
    "Engine failures surface as upstream.gmat_sweep_failed; config / payload "
    "violations surface as invalid_input.gmat_sweep_*."
)

_EXECUTE_SCRIPT_DESCRIPTION = (
    "Escape-hatch raw GMAT script executor — minimal validation, raw text output. "
    "Prefer gmat_run_mission whenever your goal is 'run this mission and tell me "
    "what happened': it returns a structured snapshot, parsed report rows, and "
    "typed convergence flags, all shape-disciplined for small-model contexts. "
    "Reach for gmat_execute_script only when the curated surface doesn't fit — "
    "e.g. you need the raw ReportFile text verbatim (header line, units row, "
    "formatting) instead of parsed rows, or you want to run a script with "
    "side-effect-only commands that don't surface through the structured "
    "response. e.g. gmat_execute_script(script='/abs/path/to/custom.script') "
    "runs the script and returns each ReportFile's raw text plus a list of every "
    "other artefact GMAT wrote (ephemerides, contact reports, solver logs). "
    "`script` must be either an absolute path to a .script file or the full "
    "inline script text (auto-detected by leading '%' / 'Create' markers); do "
    "not pass a Python Mission object. Failures-as-data contract: a GMAT engine "
    "failure mid-run returns ok=False with the engine's stderr in `stderr` "
    "rather than raising — read the log to diagnose. Pre-run input failures "
    "(bad path, GMAT install missing) still raise typed invalid_input.* / "
    "upstream.* errors. `output` controls line shaping for ReportFile text: "
    "the default 'summary' inlines short reports (<=60 lines) whole and trims "
    "longer ones to first/last 20 lines so the response fits small-model input "
    "caps; 'full' returns every line of every report. Non-ReportFile artefacts "
    "(ephemerides, contact reports, solver logs) are always pointer-only "
    "regardless of mode — read the curated tool's response shape or copy the "
    "paths before the tool returns; the run's temp directory is cleaned up at "
    "function exit."
)

_VALIDATE_SCRIPT_DESCRIPTION = (
    "Parse-validate a GMAT mission script without running the mission sequence: "
    "load it through GMAT's interpreter, capture any errors and warnings GMAT "
    "itself surfaces, and return them alongside the parsed resource and command "
    "inventory. Intended for a self-correction loop where an LLM iterates on its "
    "script — call gmat_validate_script, fix what GMAT flags, then call "
    "gmat_run_mission. e.g. gmat_validate_script(script='/abs/path/to/"
    "Ex_HohmannTransfer.script') returns ok=True with a summary listing the "
    "declared Spacecraft, ForceModel, Propagator, ReportFile resources and the "
    "mission-sequence commands. `script` must be either an absolute path to a "
    ".script file or the full inline script text (auto-detected by leading '%' "
    "/ 'Create' markers); do not pass a Python Mission object. Common-mistake "
    "notes: validate confirms the script parses, not that the mission runs "
    "end-to-end — solver convergence and runtime errors still need "
    "gmat_run_mission. GMAT is case-sensitive: 'Spacecraft sat' and 'Spacecraft "
    "Sat' are different objects. Missing statement terminators (semicolons) are "
    "a known false-negative — GMAT's interpreter accepts them silently, so "
    "ok=True does not guarantee a syntactically pristine script, only that the "
    "interpreter could build the object graph. Output fields carry no physical "
    "units (the tool is structural, not numeric)."
)

_READ_RUN_ARTEFACT_DESCRIPTION = (
    "Read the raw text of one file written by a prior gmat_run_mission, "
    "gmat_sweep, or gmat_execute_script call, keyed by the `run_id` that "
    "producer returned. The producer tools shape their inline responses for "
    "small-model input caps; reach for this tool when you need the verbatim "
    "bytes of an output that was too large to inline (a long EphemerisFile, a "
    "ContactLocator report, the GMAT log) or when you want a ReportFile's "
    "header / units rows preserved exactly. e.g. after gmat_run_mission "
    "returns run_id='abc...' with an OutputPointer for 'EphemerisFile1', call "
    "gmat_read_run_artefact(run_id='abc...', name='EphemerisFile1', "
    "output='summary') to inspect the first and last 20 lines. `name` "
    "resolves first against the run's declared resource names (ReportFile / "
    "EphemerisFile / ContactLocator / Solver), then against plain file "
    "basenames directly under the run's output directory (so 'GMAT.log' and "
    "solver '.data' files are reachable). `output` controls line shaping: "
    "the default 'summary' inlines short files (<=60 lines) whole and trims "
    "longer ones to first/last 20 lines; 'full' returns every line. "
    "Read-only — this tool does not run GMAT, mutate files, or extend the "
    "run's retention. The registry retains the last N runs per server "
    "process (configurable via ASTRODYNAMICS_MCP_RUN_REGISTRY_LIMIT, default "
    "50); evicted runs return invalid_input.unknown_run_id, and an unknown "
    "name within a known run returns invalid_input.unknown_artefact_name "
    "with the available set in `data`. After a server restart the index is "
    "best-effort: if the run's temp directory still exists the read "
    "succeeds, otherwise the tool returns invalid_input.artefact_evicted. "
    "Text outputs only — GMAT's text formats (ReportFile, OEM / CCSDS-OEM / "
    "CCSDS-AEM / STK ephemerides, ContactLocator, solver .data, GMAT.log, "
    "sweep manifest.jsonl) all flow through; binary outputs (SPK and "
    "GMAT Code-500 ephemerides, sweep .parquet) are rejected with "
    "invalid_input.binary_artefact rather than returned as decoded gibberish."
)


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


@contextmanager
def _resolved_script(script: str) -> Iterator[Path]:
    """Resolve ``script`` to an on-disk path and clean up any inline-text temp file.

    Wraps :func:`_resolve_script_input` + the per-tool ``try/finally
    cleanup_path.unlink(missing_ok=True)`` dance. The yielded path is
    valid for the lifetime of the ``with`` block; once the block exits
    (success or failure), any temp file created from inline text is
    unlinked, while a caller-supplied path is left alone.
    """
    script_path, cleanup_path = _resolve_script_input(script)
    try:
        yield script_path
    finally:
        if cleanup_path is not None:
            cleanup_path.unlink(missing_ok=True)


@contextmanager
def _owned_workspace(prefix: str) -> Iterator[tuple[Path, Callable[[], None]]]:
    """Own a ``tempfile.mkdtemp`` workspace until ownership is explicitly handed off.

    Yields ``(workspace_path, mark_handed_off)``. The producer calls
    ``mark_handed_off()`` after :meth:`RunRegistry.register` accepts the
    workspace; from that point the registry owns reclamation. If the
    ``with`` block exits (success or failure) **without** the hand-off
    callback firing, :func:`shutil.rmtree` reaps the directory so a
    failure between ``mkdtemp`` and ``registry.register`` does not leak.
    """
    workspace = Path(tempfile.mkdtemp(prefix=prefix))
    handed_off = False

    def mark_handed_off() -> None:
        nonlocal handed_off
        handed_off = True

    try:
        yield workspace, mark_handed_off
    finally:
        if not handed_off:
            shutil.rmtree(workspace, ignore_errors=True)


def _load_mission(script_path: Path) -> Any:
    """Call :meth:`gmat_run.Mission.load` and wrap typed errors.

    Shared between :func:`gmat_run_mission` and :func:`gmat_execute_script`;
    both need the identical ``GmatLoadError`` → ``upstream.gmat_run_load_failed``
    and ``GmatError`` → ``upstream.gmat_run_bootstrap_failed`` mapping.
    """
    from gmat_run import Mission
    from gmat_run.errors import GmatError, GmatLoadError

    try:
        return Mission.load(script_path)
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


_T = TypeVar("_T")


def _truncate(
    items: Sequence[_T],
    *,
    threshold: int,
    head_tail: int,
    output: Literal["summary", "full"],
) -> tuple[list[_T], list[_T], list[_T], bool]:
    """Apply the inline-vs-truncate decision shared by every shaper.

    Returns ``(full, head, tail, truncated)``: in ``output='full'`` mode
    or when ``len(items) <= threshold`` the response carries every item
    in ``full`` and ``truncated`` is ``False``; otherwise ``full`` is
    empty, ``head`` / ``tail`` carry the first / last ``head_tail``
    items, and ``truncated`` is ``True``.
    """
    total = len(items)
    if output == "full" or total <= threshold:
        return list(items), [], [], False
    return [], list(items[:head_tail]), list(items[-head_tail:]), True


def _render_mission_summary(summary: Any) -> MissionSummaryView:
    """Reshape a :class:`gmat_run.summary.MissionSummary` into the response model.

    Shared between :func:`gmat_run_mission` (where the summary is fetched
    via ``mission.summary()``) and :func:`gmat_validate_script` (where the
    tool drives ``build_mission_summary`` directly, having bypassed the
    ``Mission`` wrapper to capture the load-time log).
    """
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


def _build_mission_summary_view(mission: Any) -> MissionSummaryView:
    """Render the loaded mission's structured snapshot."""
    return _render_mission_summary(mission.summary())


# ---------------------------------------------------------------------------
# GMAT load-time log scraper
# ---------------------------------------------------------------------------
#
# GMAT's LoadScript returns only a bool; the actual error / warning text lives
# in the engine's log file. The patterns below were derived from R2026a output
# captured against handcrafted bad scripts (unknown fields, unknown resource
# types, undeclared references, missing BeginMissionSequence). Three line
# shapes recognised:
#
#   <seqno>: <path>: **** ERROR **** Interpreter Exception: <msg> [in line:\n   "<N>: <text>"]
#   Interpreter Exception: [<path>: ]<msg>
#   *** WARNING ***  <msg>
#
# Wording can shift between GMAT versions — `raw_log` on the response is the
# escape hatch for callers when the scraper misses something. Upstream
# astro-tools/gmat-run#153 promotes this scrape into `Mission.validate()`, at
# which point this helper collapses into a thin wrapper.

_WARNING_LINE_RE = re.compile(r"\*+\s*WARNING\s*\*+\s+(?P<msg>.+?)\s*$")
_ERROR_SEQ_RE = re.compile(
    r"^\d+:\s+\S+:\s+\*+\s*ERROR\s*\*+\s+Interpreter\s+Exception:\s+(?P<msg>.+?)\s*$"
)
_ERROR_BARE_RE = re.compile(r"^Interpreter\s+Exception:\s+(?P<msg>.+?)\s*$")
_LINE_CONTEXT_RE = re.compile(r'^"\s*(?P<line>\d+):\s*[^"]*"\s*$')
_PATH_PREFIX_RE = re.compile(r"^[^:\n]+\.script:\s*")
_IN_LINE_SUFFIX_RE = re.compile(r"\s+in\s+line:\s*$")


def _parse_gmat_log(raw_log: str) -> tuple[list[ParseDiagnostic], list[ParseDiagnostic]]:
    """Scrape errors and warnings from a captured GMAT load-time log."""
    errors: list[ParseDiagnostic] = []
    warnings: list[ParseDiagnostic] = []
    raw_lines = raw_log.splitlines()
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i].rstrip()
        if not line:
            i += 1
            continue
        m_warn = _WARNING_LINE_RE.search(line)
        if m_warn:
            warnings.append(
                ParseDiagnostic(
                    line=None,
                    message=m_warn.group("msg").strip(),
                    raw=line.strip(),
                )
            )
            i += 1
            continue
        m_err = _ERROR_SEQ_RE.match(line) or _ERROR_BARE_RE.match(line)
        if m_err:
            msg = m_err.group("msg").strip()
            msg = _PATH_PREFIX_RE.sub("", msg).strip()
            line_no: int | None = None
            if _IN_LINE_SUFFIX_RE.search(msg):
                msg = _IN_LINE_SUFFIX_RE.sub("", msg).strip()
                # Peek the next non-blank line for the `   "<N>: ..."` context.
                j = i + 1
                while j < len(raw_lines) and not raw_lines[j].strip():
                    j += 1
                if j < len(raw_lines):
                    m_ctx = _LINE_CONTEXT_RE.match(raw_lines[j].strip())
                    if m_ctx:
                        line_no = int(m_ctx.group("line"))
                        i = j
            errors.append(
                ParseDiagnostic(
                    line=line_no,
                    message=msg,
                    raw=line.strip(),
                )
            )
            i += 1
            continue
        i += 1
    return errors, warnings


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


def _shape_report(
    name: str, path: Path, frame: Any, *, output: Literal["summary", "full"]
) -> ReportFileShape:
    """Build a :class:`ReportFileShape` from a parsed ReportFile DataFrame.

    In ``output="summary"`` mode the response trims rows past the inline
    threshold to head + tail; in ``output="full"`` mode every row is
    inlined regardless of size.
    """
    columns = [str(c) for c in frame.columns]
    row_count = len(frame.index)
    values = frame.to_numpy(dtype=object)
    inline_full = output == "full" or row_count <= _REPORT_INLINE_ROW_THRESHOLD
    if inline_full:
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


def _shape_raw_report(
    name: str, path: Path, *, output: Literal["summary", "full"]
) -> RawReportContent:
    """Read a ReportFile from disk into a :class:`RawReportContent`.

    Reads bytes once for the byte_count, decodes UTF-8 (with ``replace``
    so a stray binary byte doesn't abort the tool), and splits on
    universal newlines. The file is closed before this function returns;
    the caller's responsibility is to read it while the temp directory is
    still alive.
    """
    raw_bytes = path.read_bytes()
    byte_count = len(raw_bytes)
    text = raw_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()
    line_count = len(lines)
    inline_full = output == "full" or line_count <= _RAW_REPORT_INLINE_LINE_THRESHOLD
    if inline_full:
        content = text
        head = ""
        tail = ""
        truncated = False
    else:
        content = ""
        head = "\n".join(lines[:_RAW_REPORT_HEAD_TAIL_LINES])
        tail = "\n".join(lines[-_RAW_REPORT_HEAD_TAIL_LINES:])
        truncated = True
    return RawReportContent(
        name=name,
        path=str(path),
        content=content,
        head=head,
        tail=tail,
        line_count=Quantity(value=float(line_count), unit="1"),
        byte_count=Quantity(value=float(byte_count), unit="1"),
        truncated=truncated,
    )


def _walk_artefacts(result: Any) -> list[OutputPointer]:
    """Enumerate every regular file the run wrote under ``result.output_dir``.

    Resolves each file's GMAT resource name when one of the four
    ``*_paths`` mappings on the result claims it; falls back to the file
    basename otherwise (e.g. for a stray GMAT log that landed on disk).
    Output is sorted by absolute path so a caller can diff two runs
    without re-sorting.
    """
    path_to_name: dict[Path, str] = {}
    for paths in (
        getattr(result, "report_paths", {}),
        getattr(result, "ephemeris_paths", {}),
        getattr(result, "contact_paths", {}),
        getattr(result, "solver_paths", {}),
    ):
        for resource_name, path in paths.items():
            path_to_name[Path(path).resolve()] = resource_name

    output_dir = Path(result.output_dir)
    artefacts: list[OutputPointer] = []
    for candidate in sorted(output_dir.rglob("*")):
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        name = path_to_name.get(resolved, candidate.name)
        artefacts.append(OutputPointer(name=name, path=str(candidate)))
    return artefacts


def _collect_artefact_map(result: Any) -> dict[str, Path]:
    """Return the GMAT resource-name → path map a producer registers.

    Unions the four ``*_paths`` mappings on the run result so a follow-up
    ``gmat_read_run_artefact(run_id, name)`` call can resolve any declared
    output (``ReportFile1``, ``EphemerisFile1``, ``ContactLocator1``,
    ``DC`` for solver logs, …) directly by its script-side resource name.
    Stray files (e.g. the GMAT log) are resolved at read time via a
    basename fallback under ``output_dir`` — they are deliberately not
    pre-enumerated here so a sweep with thousands of nested files doesn't
    bloat the registry's JSON index.
    """
    collected: dict[str, Path] = {}
    for paths in (
        getattr(result, "report_paths", {}),
        getattr(result, "ephemeris_paths", {}),
        getattr(result, "contact_paths", {}),
        getattr(result, "solver_paths", {}),
    ):
        for resource_name, path in paths.items():
            collected[str(resource_name)] = Path(path)
    return collected


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


_SWEEP_HEAD_TAIL_ROWS = 5

_SWEEP_INLINE_ROW_THRESHOLD = 2 * _SWEEP_HEAD_TAIL_ROWS

_SWEEP_VALID_MODES: tuple[str, ...] = ("grid", "samples", "monte_carlo", "latin_hypercube")

_DIST_TAGS: tuple[str, ...] = ("normal", "uniform", "lognormal")


def _coerce_perturb(perturb: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    """Validate a JSON ``perturb`` payload and tuple-ify each entry.

    The MCP wire carries shorthand specs as JSON lists (e.g. ``["normal",
    0.0, 1.0]``); ``gmat_sweep.DistSpec`` is tuple-typed, so we tuple-ify
    at the boundary and reject anything that is not a list whose first
    element is a known shorthand tag. Pre-frozen scipy ``rv_frozen``
    distributions are not supported across this boundary — they are not
    JSON-serialisable and the LLM has no realistic way to construct one.
    """
    if not isinstance(perturb, dict) or not perturb:
        raise InvalidInputError(
            f"perturb must be a non-empty mapping of dotted-path to "
            f"[distribution_name, *params] list, got {perturb!r}",
            code="invalid_input.gmat_sweep_perturb_empty",
        )
    coerced: dict[str, tuple[Any, ...]] = {}
    for key, value in perturb.items():
        if not isinstance(value, list) or not value:
            raise InvalidInputError(
                f"perturb[{key!r}] must be a list of the form [distribution_name, *params], "
                f"got {value!r}",
                code="invalid_input.gmat_sweep_perturb_shape",
            )
        tag = value[0]
        if not isinstance(tag, str) or tag not in _DIST_TAGS:
            raise InvalidInputError(
                f"perturb[{key!r}] first element must be one of {list(_DIST_TAGS)}, got {tag!r}",
                code="invalid_input.gmat_sweep_perturb_tag",
            )
        coerced[key] = tuple(value)
    return coerced


def _samples_to_dataframe(samples: list[dict[str, Any]]) -> Any:
    """Materialise a list-of-dict samples payload into a pandas DataFrame.

    Column order is taken from the first row; subsequent rows must carry
    exactly the same key set. Any deviation raises an
    :class:`InvalidInputError` with a typed code so the LLM consumer can
    see which row drifted from the schema.
    """
    if not isinstance(samples, list) or not samples:
        raise InvalidInputError(
            f"samples must be a non-empty list of dotted-path → value dicts, got {samples!r}",
            code="invalid_input.gmat_sweep_samples_empty",
        )
    import pandas as pd

    first = samples[0]
    if not isinstance(first, dict) or not first:
        raise InvalidInputError(
            f"samples[0] must be a non-empty dict of dotted-path → value, got {first!r}",
            code="invalid_input.gmat_sweep_samples_row_shape",
        )
    columns = list(first.keys())
    expected = set(columns)
    rows: list[list[Any]] = []
    for i, row in enumerate(samples):
        if not isinstance(row, dict):
            raise InvalidInputError(
                f"samples[{i}] must be a dict, got {row!r}",
                code="invalid_input.gmat_sweep_samples_row_shape",
            )
        if set(row.keys()) != expected:
            raise InvalidInputError(
                f"samples[{i}] keys {sorted(row.keys())!r} differ from samples[0] keys "
                f"{sorted(expected)!r}; every row must share the same columns",
                code="invalid_input.gmat_sweep_samples_row_drift",
            )
        rows.append([row[c] for c in columns])
    return pd.DataFrame(rows, columns=columns)


def _validate_sweep_payload(
    *,
    mode: str,
    grid: dict[str, list[Any]] | None,
    samples: list[dict[str, Any]] | None,
    perturb: dict[str, Any] | None,
    n: int | None,
    seed: int | None,
) -> None:
    """Reject payloads that don't match the chosen ``mode`` discriminator.

    Each mode owns a distinct set of required and forbidden fields. The
    grid / samples modes refuse perturb / n / seed; the Monte Carlo and
    Latin hypercube modes require perturb and n, and we require an
    explicit seed too — without it the run is irreproducible and the LLM
    user cannot replay it from the response.
    """
    if mode == "grid":
        if grid is None:
            raise InvalidInputError(
                "mode='grid' requires the `grid` argument",
                code="invalid_input.gmat_sweep_grid_required",
            )
        if samples is not None or perturb is not None or n is not None or seed is not None:
            raise InvalidInputError(
                "mode='grid' rejects `samples`, `perturb`, `n`, `seed`",
                code="invalid_input.gmat_sweep_mode_payload_conflict",
            )
    elif mode == "samples":
        if samples is None:
            raise InvalidInputError(
                "mode='samples' requires the `samples` argument",
                code="invalid_input.gmat_sweep_samples_required",
            )
        if grid is not None or perturb is not None or n is not None or seed is not None:
            raise InvalidInputError(
                "mode='samples' rejects `grid`, `perturb`, `n`, `seed`",
                code="invalid_input.gmat_sweep_mode_payload_conflict",
            )
    elif mode in ("monte_carlo", "latin_hypercube"):
        if perturb is None:
            raise InvalidInputError(
                f"mode={mode!r} requires the `perturb` argument",
                code="invalid_input.gmat_sweep_perturb_required",
            )
        if n is None or n < 1:
            raise InvalidInputError(
                f"mode={mode!r} requires `n` >= 1, got {n!r}",
                code="invalid_input.gmat_sweep_n_required",
            )
        if seed is None:
            raise InvalidInputError(
                f"mode={mode!r} requires an integer `seed` for reproducibility; "
                "omit only when an irreproducible run is acceptable, in which case "
                "call gmat-sweep directly",
                code="invalid_input.gmat_sweep_seed_required",
            )
        if grid is not None or samples is not None:
            raise InvalidInputError(
                f"mode={mode!r} rejects `grid` and `samples`",
                code="invalid_input.gmat_sweep_mode_payload_conflict",
            )
    else:  # defensive; the Literal annotation should prevent this
        raise InvalidInputError(
            f"unknown mode {mode!r}; expected one of {list(_SWEEP_VALID_MODES)}",
            code="invalid_input.gmat_sweep_unknown_mode",
        )


def _status_counts(frame: Any) -> tuple[int, int, int]:
    """Return ``(ok, failed, skipped)`` from the result frame's ``__status`` column.

    Older gmat-sweep frames omit ``__status`` when every run succeeded;
    in that case the row count is the ok count and the other two are
    zero. The lookup is positional via ``in frame.columns`` rather than
    catching ``KeyError`` so we can tell the two paths apart cheaply.
    """
    if "__status" not in frame.columns:
        return len(frame.index), 0, 0
    status_col = frame["__status"]
    ok = int((status_col == "ok").sum())
    failed = int((status_col == "failed").sum())
    skipped = int((status_col == "skipped").sum())
    return ok, failed, skipped


def _numeric_column_stats(frame: Any) -> list[SweepColumnStats]:
    """Compute per-column mean / std / min / max over ``ok`` rows.

    Failed and skipped rows carry NaN cells across every data column;
    restricting to ``ok`` first and then dropping the NaNs gives the
    canonical "stats over good runs only" reading. Non-numeric columns
    are skipped silently — they appear in `head` / `tail` instead.
    """
    import numpy as np

    ok_frame = frame[frame["__status"] == "ok"] if "__status" in frame.columns else frame
    stats: list[SweepColumnStats] = []
    for column in frame.columns:
        if column == "__status":
            continue
        series = ok_frame[column]
        try:
            numeric = series.astype(float)
        except (TypeError, ValueError):
            continue
        finite_mask = np.isfinite(numeric.to_numpy())
        finite_values = numeric.to_numpy()[finite_mask]
        count = int(finite_values.size)
        if count == 0:
            continue
        std = float(finite_values.std(ddof=1)) if count >= 2 else float("nan")
        stats.append(
            SweepColumnStats(
                column=str(column),
                count=Quantity(value=float(count), unit="1"),
                mean=float(finite_values.mean()),
                std=std,
                min=float(finite_values.min()),
                max=float(finite_values.max()),
            )
        )
    return stats


def _frame_row_to_dict(
    columns: list[str],
    index_tuple: tuple[Any, Any],
    row_values: Any,
) -> dict[str, str | float]:
    """Render one MultiIndex row into a flat dict carrying run_id, time, columns."""
    run_id, time_val = index_tuple
    out: dict[str, str | float] = {
        "run_id": _cell_value(run_id),
        "time": _cell_value(time_val),
    }
    for i, column in enumerate(columns):
        out[column] = _cell_value(row_values[i])
    return out


def _frame_rows(frame: Any) -> list[dict[str, str | float]]:
    """Convert every row of a ``(run_id, time)``-indexed frame to dicts."""
    data_columns = [c for c in frame.columns if c != "__status"]
    values = frame[data_columns].to_numpy(dtype=object) if data_columns else None
    rows: list[dict[str, str | float]] = []
    for i, idx in enumerate(frame.index):
        run_id, time_val = (idx[0], idx[1]) if isinstance(idx, tuple) else (idx, None)
        row_values = values[i] if values is not None else []
        rows.append(_frame_row_to_dict(data_columns, (run_id, time_val), row_values))
    return rows


def _build_sweep_response(
    *,
    run_id: str,
    mode: Literal["grid", "samples", "monte_carlo", "latin_hypercube"],
    script_name: str,
    frame: Any,
    wall_clock_s: float,
    manifest_path: Path,
    output_dir: Path,
    output: Literal["summary", "full"],
) -> GmatSweepResponse:
    """Assemble a :class:`GmatSweepResponse` from a finished sweep DataFrame.

    Pulled out of the tool body so the unit tests can exercise the
    shaping logic against a fake frame without round-tripping through
    gmat-sweep itself.
    """
    data_columns = [str(c) for c in frame.columns if c != "__status"]
    ok, failed, skipped = _status_counts(frame)
    status_counts = SweepStatusCounts(
        ok=Quantity(value=float(ok), unit="1"),
        failed=Quantity(value=float(failed), unit="1"),
        skipped=Quantity(value=float(skipped), unit="1"),
    )

    all_rows = _frame_rows(frame)
    total = len(all_rows)
    if total <= _SWEEP_INLINE_ROW_THRESHOLD:
        head = list(all_rows)
        tail = list(all_rows)
        truncated_summary = False
    else:
        head = all_rows[:_SWEEP_HEAD_TAIL_ROWS]
        tail = all_rows[-_SWEEP_HEAD_TAIL_ROWS:]
        truncated_summary = True

    if output == "full":
        rows = list(all_rows)
        truncated = False
    else:
        rows = []
        truncated = truncated_summary

    run_count_total = ok + failed + skipped

    return GmatSweepResponse(
        run_id=run_id,
        mode=mode,
        script_name=script_name,
        run_count=Quantity(value=float(run_count_total), unit="1"),
        wall_clock=Quantity(value=float(wall_clock_s), unit="s"),
        columns=data_columns,
        status_counts=status_counts,
        summary_stats=_numeric_column_stats(frame),
        head=head,
        tail=tail,
        rows=rows,
        truncated=truncated,
        manifest_path=str(manifest_path),
        output_dir=str(output_dir),
    )


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
        output: Annotated[
            Literal["summary", "full"],
            Field(
                description=(
                    "Row-shaping mode for ReportFile outputs. The default 'summary' "
                    "inlines small reports (<=20 rows) whole and trims larger ones "
                    "to first/last five rows so the response fits small-model input "
                    "caps. 'full' returns every row of every selected report — pass "
                    "only when downstream consumers need the dense data and can "
                    "absorb the bytes. Ephemerides and ContactLocators are always "
                    "pointer-only regardless of mode (they are intrinsically too "
                    "large to inline)."
                ),
            ),
        ] = "summary",
    ) -> GmatRunMissionResponse:
        from gmat_run import Mission
        from gmat_run.errors import GmatError, GmatLoadError, GmatRunError

        registry = default_registry()
        run_id = registry.mint()
        # Bring our own workspace so the dir's lifetime is ours, not tied
        # to Results._workspace (which the gmat-run runtime cleans up on
        # GC). The registry owns reclamation now: it ``rmtree``s the dir
        # on eviction, and never before then.
        workspace = Path(tempfile.mkdtemp(prefix="astrodynamics-mcp-run-"))
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
                result = mission.run(working_dir=workspace)
            except GmatRunError as exc:
                raise UpstreamError(
                    f"GMAT mission run failed: {exc}",
                    code="upstream.gmat_run_failed",
                    original_exception=exc,
                ) from exc
            wall_clock_s = time.perf_counter() - t0

            registry.register(
                run_id,
                output_dir=workspace,
                artefacts=_collect_artefact_map(result),
            )

            return _build_response(
                run_id=run_id,
                mission=mission,
                result=result,
                wall_clock_s=wall_clock_s,
                select_outputs=select_outputs,
                output=output,
            )
        finally:
            if cleanup_path is not None:
                cleanup_path.unlink(missing_ok=True)

    @register_tool(
        name="gmat_sweep",
        description=_SWEEP_DESCRIPTION,
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    )
    async def gmat_sweep(
        script: Annotated[
            str,
            Field(
                description=(
                    "Either the absolute path to a GMAT .script file (e.g. "
                    "'/abs/path/to/Ex_HohmannTransfer.script') or the full inline "
                    "script text starting with '%' comments or 'Create' resource "
                    "declarations. Auto-detected by content — same shape as the "
                    "`script` argument of gmat_run_mission."
                ),
            ),
        ],
        mode: Annotated[
            Literal["grid", "samples", "monte_carlo", "latin_hypercube"],
            Field(
                description=(
                    "Which sweep backend to dispatch and which payload fields to read. "
                    "'grid' takes `grid`; 'samples' takes `samples`; 'monte_carlo' and "
                    "'latin_hypercube' both take `perturb`, `n`, `seed`. Passing fields "
                    "not associated with the chosen mode raises "
                    "invalid_input.gmat_sweep_mode_payload_conflict."
                ),
            ),
        ],
        grid: Annotated[
            dict[str, list[Any]] | None,
            Field(
                description=(
                    "Full-factorial sweep parameters, used only when mode='grid'. Keys "
                    "are dotted-path field names (e.g. 'Sat.SMA', 'FM.Drag.AtmosphereModel'); "
                    "values are the list of values to sweep on that axis. The run set is "
                    "the cartesian product across every key. Leave null in other modes."
                ),
            ),
        ] = None,
        samples: Annotated[
            list[dict[str, Any]] | None,
            Field(
                description=(
                    "Explicit-row sweep parameters, used only when mode='samples'. Each "
                    "list element is one run: a dict from dotted-path field name to value. "
                    "Every row must carry the same keys; column order is taken from the "
                    "first row. Leave null in other modes."
                ),
            ),
        ] = None,
        perturb: Annotated[
            dict[str, list[Any]] | None,
            Field(
                description=(
                    "Per-parameter distribution specs, used only when "
                    "mode='monte_carlo' or 'latin_hypercube'. Each value is a list of "
                    "the form [distribution_name, *params] — ['normal', mu, sigma], "
                    "['uniform', lo, hi], or ['lognormal', mu, sigma]; plain numbers "
                    "and scipy distributions are not accepted across the MCP boundary. "
                    "Leave null in other modes."
                ),
            ),
        ] = None,
        n: Annotated[
            int | None,
            Field(
                description=(
                    "Number of stochastic runs, used only when mode='monte_carlo' or "
                    "'latin_hypercube'. Must be >= 1. Leave null in other modes; "
                    "the grid mode's run count is derived from the cartesian product "
                    "of `grid`, and the samples mode uses len(samples)."
                ),
            ),
        ] = None,
        seed: Annotated[
            int | None,
            Field(
                description=(
                    "Integer parent seed, required when mode='monte_carlo' or "
                    "'latin_hypercube' so the per-run draws can be reproduced from "
                    "the response alone. Leave null in other modes."
                ),
            ),
        ] = None,
        max_workers: Annotated[
            int,
            Field(
                description=(
                    "Worker count for the local joblib backend. Defaults to 1 to keep "
                    "the cost ceiling tight; raise it explicitly to parallelise across "
                    "cores. Custom backends (Dask, Ray, MPI) are not exposed at this "
                    "MCP boundary — call gmat-sweep directly for those."
                ),
            ),
        ] = 1,
        output: Annotated[
            Literal["summary", "full"],
            Field(
                description=(
                    "Row-shaping mode for the result frame. The default 'summary' returns "
                    "per-column mean / std / min / max plus the head + tail five rows "
                    "of the result frame so the response fits small-model input caps. "
                    "'full' adds every row in `rows` — pass only when downstream consumers "
                    "need the dense data and can absorb the bytes. `head`, `tail`, and "
                    "`summary_stats` are always populated regardless of mode."
                ),
            ),
        ] = "summary",
    ) -> GmatSweepResponse:
        from gmat_sweep import latin_hypercube, monte_carlo, sweep
        from gmat_sweep.backends.joblib import LocalJoblibPool
        from gmat_sweep.errors import SweepConfigError

        _validate_sweep_payload(
            mode=mode, grid=grid, samples=samples, perturb=perturb, n=n, seed=seed
        )

        registry = default_registry()
        run_id = registry.mint()
        script_path, cleanup_path = _resolve_script_input(script)
        try:
            # The sweep's output_dir must outlive this call so `manifest_path`
            # and `output_dir` remain valid pointers in the response *and* so
            # ``gmat_read_run_artefact`` can re-read the manifest after the
            # response serialises out. Owning the dir here (instead of letting
            # gmat-sweep manage it with ``out=None``) hands its lifetime to
            # the registry, which ``rmtree``s the dir only on eviction.
            sweep_out_dir = Path(tempfile.mkdtemp(prefix="astrodynamics-mcp-sweep-"))
            backend = LocalJoblibPool(max_workers=max_workers)
            t0 = time.perf_counter()
            try:
                if mode == "grid":
                    assert grid is not None
                    frame = sweep(
                        script_path,
                        grid=grid,
                        backend=backend,
                        out=sweep_out_dir,
                        progress=False,
                    )
                elif mode == "samples":
                    assert samples is not None
                    samples_df = _samples_to_dataframe(samples)
                    frame = sweep(
                        script_path,
                        samples=samples_df,
                        backend=backend,
                        out=sweep_out_dir,
                        progress=False,
                    )
                elif mode == "monte_carlo":
                    assert perturb is not None and n is not None
                    frame = monte_carlo(
                        script_path,
                        n=n,
                        perturb=_coerce_perturb(perturb),
                        seed=seed,
                        backend=backend,
                        out=sweep_out_dir,
                        progress=False,
                    )
                else:  # latin_hypercube
                    assert perturb is not None and n is not None
                    frame = latin_hypercube(
                        script_path,
                        n=n,
                        perturb=_coerce_perturb(perturb),
                        seed=seed,
                        backend=backend,
                        out=sweep_out_dir,
                        progress=False,
                    )
            except SweepConfigError as exc:
                raise InvalidInputError(
                    f"gmat-sweep rejected the sweep config: {exc}",
                    code="invalid_input.gmat_sweep_config",
                    data={"original_exception_message": str(exc)},
                ) from exc
            except Exception as exc:
                raise UpstreamError(
                    f"gmat-sweep failed: {exc}",
                    code="upstream.gmat_sweep_failed",
                    original_exception=exc,
                ) from exc
            wall_clock_s = time.perf_counter() - t0

            manifest_path = sweep_out_dir / "manifest.jsonl"
            registry.register(
                run_id,
                output_dir=sweep_out_dir,
                artefacts={"manifest.jsonl": manifest_path} if manifest_path.is_file() else {},
            )

            return _build_sweep_response(
                run_id=run_id,
                mode=mode,
                script_name=script_path.name,
                frame=frame,
                wall_clock_s=wall_clock_s,
                manifest_path=manifest_path,
                output_dir=sweep_out_dir,
                output=output,
            )
        finally:
            if cleanup_path is not None:
                cleanup_path.unlink(missing_ok=True)

    @register_tool(
        name="gmat_execute_script",
        description=_EXECUTE_SCRIPT_DESCRIPTION,
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    )
    async def gmat_execute_script(
        script: Annotated[
            str,
            Field(
                description=(
                    "Either the absolute path to a GMAT .script file (e.g. "
                    "'/abs/path/to/custom.script') or the full inline script text "
                    "starting with '%' comments or 'Create' resource declarations. "
                    "Auto-detected by content: a string with newlines or a leading "
                    "'%' / 'Create ' is inline, anything else is treated as a path. "
                    "Do not pass a Python Mission object."
                ),
            ),
        ],
        output: Annotated[
            Literal["summary", "full"],
            Field(
                description=(
                    "Line-shaping mode for ReportFile text. The default 'summary' "
                    "inlines short reports (<=60 lines) whole and trims longer ones "
                    "to first/last 20 lines so the response fits small-model input "
                    "caps. 'full' returns every line of every report — pass only "
                    "when downstream consumers need the dense data and can absorb "
                    "the bytes. Non-ReportFile artefacts are pointer-only "
                    "regardless of mode."
                ),
            ),
        ] = "summary",
    ) -> GmatExecuteScriptResponse:
        from gmat_run import Mission
        from gmat_run.errors import GmatError, GmatLoadError, GmatRunError

        registry = default_registry()
        run_id = registry.mint()
        workspace = Path(tempfile.mkdtemp(prefix="astrodynamics-mcp-run-"))
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

            t0 = time.perf_counter()
            try:
                result = mission.run(working_dir=workspace)
            except GmatRunError as exc:
                wall_clock_s = time.perf_counter() - t0
                # Register the (likely empty) workspace anyway so the caller
                # can pull the GMAT log via gmat_read_run_artefact if GMAT
                # managed to write one before bailing out.
                registry.register(run_id, output_dir=workspace, artefacts={})
                return GmatExecuteScriptResponse(
                    run_id=run_id,
                    ok=False,
                    stderr=exc.log,
                    wall_clock=Quantity(value=float(wall_clock_s), unit="s"),
                    reports=[],
                    artefacts=[],
                )
            wall_clock_s = time.perf_counter() - t0

            registry.register(
                run_id,
                output_dir=workspace,
                artefacts=_collect_artefact_map(result),
            )

            reports = [
                _shape_raw_report(name, Path(result.report_paths[name]), output=output)
                for name in result.report_paths
            ]
            return GmatExecuteScriptResponse(
                run_id=run_id,
                ok=True,
                stderr=result.log,
                wall_clock=Quantity(value=float(wall_clock_s), unit="s"),
                reports=reports,
                artefacts=_walk_artefacts(result),
            )
        finally:
            if cleanup_path is not None:
                cleanup_path.unlink(missing_ok=True)

    @register_tool(
        name="gmat_validate_script",
        description=_VALIDATE_SCRIPT_DESCRIPTION,
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    )
    async def gmat_validate_script(
        script: Annotated[
            str,
            Field(
                description=(
                    "Either the absolute path to a GMAT .script file (e.g. "
                    "'/abs/path/to/Ex_HohmannTransfer.script') or the full inline "
                    "script text starting with '%' comments or 'Create' resource "
                    "declarations. Auto-detected by content — same shape as the "
                    "`script` argument of gmat_run_mission."
                ),
            ),
        ],
    ) -> GmatValidateScriptResponse:
        # Bypasses gmat_run.Mission.load to capture the GMAT log across the
        # parse step — Mission.load only redirects UseLogFile during run(),
        # so its public surface doesn't expose load-time diagnostics. Reaches
        # into gmat-run private helpers (_get_api_exception,
        # _initialize_spacecraft) intentionally; once upstream
        # astro-tools/gmat-run#153 ships Mission.validate(), this body
        # collapses to a single call into that helper.
        from gmat_run.install import locate_gmat
        from gmat_run.mission import _get_api_exception, _initialize_spacecraft
        from gmat_run.runtime import bootstrap
        from gmat_run.summary import build_mission_summary

        script_path, cleanup_path = _resolve_script_input(script)
        try:
            try:
                install = locate_gmat()
                gmat = bootstrap(install)
            except Exception as exc:
                raise UpstreamError(
                    f"GMAT discovery / bootstrap failed: {exc}",
                    code="upstream.gmat_run_bootstrap_failed",
                    original_exception=exc,
                ) from exc

            with tempfile.TemporaryDirectory(prefix="astrodynamics-mcp-validate-") as tmp:
                log_path = Path(tmp) / "validate.log"
                # Fresh sandbox so a prior load in the same process can't
                # leak resources into the summary inventory below. Some
                # plugins refuse Clear under specific states — suppress
                # rather than abort.
                with suppress(Exception):
                    gmat.Clear()
                gmat.UseLogFile(str(log_path))
                init_error: str | None = None
                load_ok = False
                try:
                    load_ok = bool(gmat.LoadScript(str(script_path)))
                    if load_ok:
                        api_exception = _get_api_exception(gmat)
                        try:
                            _initialize_spacecraft(gmat)
                        except api_exception as exc:
                            init_error = f"{type(exc).__name__}: {exc}"
                finally:
                    # Repoint the log handle off the temp path before the
                    # TemporaryDirectory unlinks itself — GMAT's
                    # MessageInterface holds the file open otherwise (same
                    # Windows-handle issue Mission.run handles after a run).
                    with suppress(Exception):
                        gmat.UseLogFile(os.devnull)
                raw_log = log_path.read_text(encoding="utf-8", errors="replace")

            ok = load_ok and init_error is None
            errors, warnings = _parse_gmat_log(raw_log)
            if init_error is not None:
                errors.append(ParseDiagnostic(line=None, message=init_error, raw=init_error))
            summary_view: MissionSummaryView | None = None
            if ok:
                summary_view = _render_mission_summary(build_mission_summary(gmat, script_path))
            return GmatValidateScriptResponse(
                ok=ok,
                errors=errors,
                warnings=warnings,
                summary=summary_view,
                raw_log=raw_log,
            )
        finally:
            if cleanup_path is not None:
                cleanup_path.unlink(missing_ok=True)

    @register_tool(
        name="gmat_read_run_artefact",
        description=_READ_RUN_ARTEFACT_DESCRIPTION,
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    )
    async def gmat_read_run_artefact(
        run_id: Annotated[
            str,
            Field(
                description=(
                    "UUID4 hex returned by an earlier gmat_run_mission, gmat_sweep, "
                    "or gmat_execute_script call. The producer's response carries "
                    "this in its `run_id` field; pass it back verbatim. Unknown ids "
                    "raise invalid_input.unknown_run_id with the known set in `data`."
                ),
            ),
        ],
        name: Annotated[
            str,
            Field(
                description=(
                    "Artefact selector. Resolves first against the run's declared "
                    "GMAT resource names (e.g. 'ReportFile1', 'EphemerisFile1', "
                    "'ContactLocator1', or a Solver name like 'DC'); falls back to a "
                    "plain file basename directly under the run's output directory "
                    "(e.g. 'GMAT.log', 'manifest.jsonl', or a stray '*.data'). The "
                    "lookup is non-recursive — files in subdirectories (e.g. the "
                    "per-run sweep artefacts) are reachable only via their declared "
                    "names or via the sweep manifest. Unknown names raise "
                    "invalid_input.unknown_artefact_name with the available set in "
                    "`data`."
                ),
            ),
        ],
        output: Annotated[
            Literal["summary", "full"],
            Field(
                description=(
                    "Line-shaping mode for the artefact text. The default 'summary' "
                    "inlines short files (<=60 lines) whole and trims longer ones "
                    "to first/last 20 lines so the response fits small-model input "
                    "caps. 'full' returns every line — pass only when downstream "
                    "consumers need the dense data and can absorb the bytes."
                ),
            ),
        ] = "summary",
    ) -> RawReportContent:
        registry = default_registry()
        entry = registry.get(run_id)
        if entry is None:
            raise InvalidInputError(
                f"unknown run_id {run_id!r}; no run with this id is in the server-process registry",
                code="invalid_input.unknown_run_id",
                data={"known_run_ids": registry.known_run_ids()},
            )

        path = entry.artefacts.get(name)
        if path is None:
            # Basename fallback: a file written directly under output_dir
            # that wasn't part of the producer's declared resource map
            # (the GMAT log, a sweep manifest, stray solver .data).
            candidate = entry.output_dir / name
            if candidate.is_file():
                path = candidate

        if path is None:
            basenames: list[str] = []
            if entry.output_dir.is_dir():
                basenames = sorted(
                    child.name for child in entry.output_dir.iterdir() if child.is_file()
                )
            raise InvalidInputError(
                f"unknown artefact {name!r} for run_id {run_id!r}; resolves "
                f"against neither the declared resource names nor a basename "
                f"under the run's output directory",
                code="invalid_input.unknown_artefact_name",
                data={
                    "available_resource_names": sorted(entry.artefacts.keys()),
                    "available_basenames": basenames,
                },
            )

        path = Path(path)
        if not path.is_file():
            # Eagerly drop the dead entry so it doesn't keep a slot in
            # the LRU cap until the next process restart. Symmetric with
            # capacity-driven eviction: index JSON removed + output_dir
            # rmtree'd, all best-effort.
            registry.drop(run_id)
            raise InvalidInputError(
                f"artefact {name!r} for run_id {run_id!r} was registered but "
                f"is no longer on disk at {path!s}; the temp directory was "
                f"likely reaped (OS cleanup, manual deletion) between calls",
                code="invalid_input.artefact_evicted",
                data={"path": str(path)},
            )

        # Binary sniff. `_shape_raw_report` reads the whole file into
        # memory before deciding what to do with it; bypassing it for
        # binary keeps an SPK ephemeris (hundreds of MB possible) from
        # OOMing the server only to return U+FFFD-laden garbage. The
        # heuristic is the standard one (grep -I / git diff --binary):
        # presence of a NULL byte in the first 8 KB. ASCII text and
        # GMAT's text formats (OEM, CCSDS-OEM, ContactLocator, .data,
        # GMAT.log, manifest.jsonl) never contain NULL; the binary
        # formats (CCSDS SPK, GMAT Code-500, Parquet) do, within their
        # headers.
        byte_count = path.stat().st_size
        with path.open("rb") as fh:
            head_bytes = fh.read(_BINARY_SNIFF_BYTES)
        if b"\x00" in head_bytes:
            raise InvalidInputError(
                f"artefact {name!r} for run_id {run_id!r} is binary "
                f"({byte_count} bytes); gmat_read_run_artefact serves text "
                f"only — binary outputs (SPK / Code-500 ephemerides, "
                f"Parquet) cannot be returned through this tool",
                code="invalid_input.binary_artefact",
                data={"byte_count": byte_count},
            )

        return _shape_raw_report(name, path, output=output)


def _build_response(
    *,
    run_id: str,
    mission: Any,
    result: Any,
    wall_clock_s: float,
    select_outputs: list[str] | None,
    output: Literal["summary", "full"] = "summary",
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
        reports.append(_shape_report(name, result.report_paths[name], frame, output=output))

    ephemerides = [
        OutputPointer(name=name, path=str(result.ephemeris_paths[name])) for name in ephemeris_names
    ]
    contacts = [
        OutputPointer(name=name, path=str(result.contact_paths[name])) for name in contact_names
    ]

    return GmatRunMissionResponse(
        run_id=run_id,
        summary=summary_view,
        wall_clock=Quantity(value=float(wall_clock_s), unit="s"),
        reports=reports,
        ephemerides=ephemerides,
        contacts=contacts,
        converged=dict(result.converged),
    )


# ---------------------------------------------------------------------------
# GMAT script-skeleton MCP resources
# ---------------------------------------------------------------------------
#
# Vetted starter scripts ship as MCP resources keyed by stable
# ``gmat-skeleton://<slug>`` URIs. They live as `.script` files under the
# sibling ``astrodynamics_mcp.skeletons`` package, are read at request time
# (no embedded blobs), and gate on the same ``[gmat]`` import guard as the
# tool slots above — useless without a way to actually run them.

_SKELETON_URI_SCHEME = "gmat-skeleton"
_SKELETON_PACKAGE = "astrodynamics_mcp.skeletons"
_SKELETON_MIME_TYPE = "text/plain"

# (slug, filename) pairs driving registration. Slug is the URI path; filename
# is the package-relative `.script` file. Order is stable so `resources/list`
# returns the catalogue in a predictable order.
_SKELETONS: tuple[tuple[str, str], ...] = (
    ("minimal-leo", "minimal_leo.script"),
    ("multibody-gravity", "multibody_gravity.script"),
    ("force-models-comparison", "force_models_comparison.script"),
    ("hohmann-transfer", "hohmann_transfer.script"),
    ("geo-transfer", "geo_transfer.script"),
    ("lunar-transfer", "lunar_transfer.script"),
    ("lunar-station-keeping", "lunar_station_keeping.script"),
    ("mars-b-plane-targeting", "mars_b_plane_targeting.script"),
    ("leo-station-keeping", "leo_station_keeping.script"),
    ("finite-burn", "finite_burn.script"),
    ("target-finite-burn-apogee-raise", "target_finite_burn_apogee_raise.script"),
    ("electric-propulsion", "electric_propulsion.script"),
    ("yukon-optimization", "yukon_optimization.script"),
    ("l2-design", "l2_design.script"),
    ("station-contact-location", "station_contact_location.script"),
    ("eclipse-location", "eclipse_location.script"),
    ("attitude-nadir-pointing", "attitude_nadir_pointing.script"),
    ("attitude-spinner", "attitude_spinner.script"),
    ("constellation", "constellation.script"),
    ("control-flow", "control_flow.script"),
)


_DESCRIPTION_LINE_RE = re.compile(r"^\s*%\s*Description:\s*(?P<text>.+?)\s*$")


def _extract_description(text: str) -> str:
    """Scrape the ``% Description: <line>`` annotation from a skeleton's head.

    Scans the script's leading non-blank lines and returns the first match.
    Raises :class:`ValueError` when no description is present so a malformed
    skeleton fails at registration time rather than shipping silently. The
    scan stops at the first non-comment, non-blank line — the description
    must live in the header banner, not deep in the mission sequence.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = _DESCRIPTION_LINE_RE.match(line)
        if m:
            return m.group("text")
        if not stripped.startswith("%"):
            break
    raise ValueError("skeleton missing '% Description: <text>' header line")


def _make_skeleton_reader(path: Path) -> Any:
    """Build the resource-handler closure that reads ``path`` at request time."""

    def read_skeleton() -> str:
        return path.read_text(encoding="utf-8")

    return read_skeleton


def _register_gmat_resources() -> None:
    """Attach every entry in :data:`_SKELETONS` as an MCP resource.

    Mirrors :func:`_register_gmat_tools`: factored out of module top-level
    so unit tests can drive registration against a fresh
    :class:`~mcp.server.fastmcp.FastMCP` instance, and reads ``mcp`` off
    the :mod:`astrodynamics_mcp.server` module at call time so a
    monkeypatched singleton resolves correctly.
    """
    skeleton_root = resources.files(_SKELETON_PACKAGE)
    for slug, filename in _SKELETONS:
        # ``importlib.resources.files`` returns a ``Traversable``; the
        # ``Path`` cast is safe for the on-disk source checkout and the
        # installed wheel alike (both resolve to regular files; no
        # zipfile-only namespace packages here).
        path = Path(str(skeleton_root.joinpath(filename)))
        if not path.is_file():
            raise FileNotFoundError(
                f"skeleton {slug!r} expected at {path}, but the file is missing"
            )
        text = path.read_text(encoding="utf-8")
        description = _extract_description(text)
        uri = f"{_SKELETON_URI_SCHEME}://{slug}"
        _server_module.mcp.resource(
            uri,
            name=slug,
            description=description,
            mime_type=_SKELETON_MIME_TYPE,
        )(_make_skeleton_reader(path))


if _GMAT_RUN_AVAILABLE:
    _register_gmat_tools()
    _register_gmat_resources()
