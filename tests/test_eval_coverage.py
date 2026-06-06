"""Coverage gate over ``eval/prompts/*.yaml``.

Asserts the suite stays populated (≥40 prompts) and within the
tier-distribution discipline, that every *core* tool has a single-tool
prompt and a sequential-or-planning prompt, that every *SPICE* tool has
a single-tool prompt and a sequential prompt, that every *viz* tool has a
single-tool prompt (with the suite carrying at least one sequential viz
prompt and every viz prompt declaring an attachment golden), and that each
extended tool the suite covers is referenced by at least one prompt. A
failing assertion here means a prompt got added or removed without
rebalancing — the regression catches it before merge.

The strict per-tool tier matrix applies to the core read-only tools and
the SPICE tools. The extended tools (GMAT + DISCOSweb metadata) are
covered more loosely: GMAT prompts skip where no install exists and the
credentialed prompts skip without secrets, so forcing each onto every
tier would not buy real signal in the default gate. The SPICE prompts
likewise skip without the [spice] extra and cached kernels, but the
acceptance contract still pins both tiers per tool. The viz prompts skip
without the [viz] extra; each viz tool is pinned at the single-tool tier
and the suite carries at least one sequential viz chain.

These also double as the test that the live suite is *populated* — an
empty ``eval/prompts/`` directory would fail loudly here rather than
silently producing a green eval run with zero samples.
"""

from __future__ import annotations

from collections import Counter

from eval._prompts import load_prompts

# Core read-only tool surface — subject to the strict single-tool +
# beyond coverage matrix below.
CORE_TOOLS: frozenset[str] = frozenset(
    {
        "tle_lookup",
        "sgp4_propagate",
        "lambert_solve",
        "access_windows",
        "time_convert",
        "frame_transform",
        "porkchop",
        "bplane_target",
    }
)

# The GMAT + DISCOSweb-metadata tools the eval suite references.
# Recognised so a typo in tools_required still fails, but not held to the
# strict per-tier matrix.
EXTENDED_TOOLS: frozenset[str] = frozenset(
    {
        "gmat_run_mission",
        "gmat_sweep",
        "gmat_execute_script",
        "gmat_read_run_artefact",
        "gmat_validate_script",
        "satellite_metadata",
    }
)

# Extended tools the suite is expected to exercise with at least one prompt.
EXTENDED_COVERED_TOOLS: frozenset[str] = EXTENDED_TOOLS

# The SPICE tools the suite covers behind the requires_spice skip-gate. Held to
# their own per-tier matrix below (single-tool *and* sequential each), per the
# v0.3 acceptance contract, but skipped in the default no-[spice] run.
SPICE_TOOLS: frozenset[str] = frozenset(
    {
        "spice_load_kernel",
        "spice_list_kernels",
        "spice_unload_kernel",
        "spice_state",
        "spice_frame_transform",
        "spice_body_parameters",
        "spice_time_convert",
    }
)

# The visualisation tools the suite covers behind the requires_viz skip-gate.
# Held to their own per-tier matrix below (each has a single-tool prompt; the
# suite carries at least one sequential viz prompt), but skipped in the default
# no-[viz] run. Every viz prompt also declares an expected_attachment golden.
VIZ_TOOLS: frozenset[str] = frozenset(
    {
        "plot_ground_track",
        "plot_trajectory",
        "plot_porkchop",
        "czml_trajectory",
    }
)

ALL_TOOLS: frozenset[str] = CORE_TOOLS | EXTENDED_TOOLS | SPICE_TOOLS | VIZ_TOOLS

MIN_PROMPTS = 40
MIN_SINGLE_TOOL = 20
MIN_SEQUENTIAL = 8
MIN_PLANNING = 2


def test_prompt_directory_populated() -> None:
    prompts = load_prompts()
    assert len(prompts) >= MIN_PROMPTS, (
        f"eval/prompts/ has {len(prompts)} prompts; target ≥{MIN_PROMPTS}"
    )


def test_tier_distribution() -> None:
    prompts = load_prompts()
    counts = Counter(p.tier for p in prompts)
    assert counts["single_tool"] >= MIN_SINGLE_TOOL, (
        f"single_tool count = {counts['single_tool']}, need ≥{MIN_SINGLE_TOOL}"
    )
    assert counts["sequential"] >= MIN_SEQUENTIAL, (
        f"sequential count = {counts['sequential']}, need ≥{MIN_SEQUENTIAL}"
    )
    assert counts["planning"] >= MIN_PLANNING, (
        f"planning count = {counts['planning']}, need ≥{MIN_PLANNING}"
    )


def test_every_core_tool_has_a_single_tool_prompt() -> None:
    prompts = load_prompts()
    covered = {
        tool for prompt in prompts if prompt.tier == "single_tool" for tool in prompt.tools_required
    }
    missing = CORE_TOOLS - covered
    assert not missing, (
        f"core tools without a single_tool prompt: {sorted(missing)}; "
        f"every core tool must be covered at the single-tool tier"
    )


def test_every_core_tool_has_a_sequential_or_planning_prompt() -> None:
    prompts = load_prompts()
    covered = {
        tool
        for prompt in prompts
        if prompt.tier in ("sequential", "planning")
        for tool in prompt.tools_required
    }
    missing = CORE_TOOLS - covered
    assert not missing, (
        f"core tools without a sequential/planning prompt: {sorted(missing)}; "
        f"every core tool must be covered beyond the single-tool tier"
    )


def test_extended_tools_each_have_a_prompt() -> None:
    """Every extended tool the suite claims to cover is referenced by a prompt."""
    prompts = load_prompts()
    covered = {tool for prompt in prompts for tool in prompt.tools_required}
    missing = EXTENDED_COVERED_TOOLS - covered
    assert not missing, (
        f"extended tools without any prompt: {sorted(missing)}; "
        f"each must be exercised by at least one prompt"
    )


def test_extended_tools_span_single_tool_and_sequential() -> None:
    """The extended tools cover both the single-tool and sequential tiers."""
    prompts = load_prompts()
    tiers_by_tool: dict[str, set[str]] = {}
    for prompt in prompts:
        for tool in prompt.tools_required:
            if tool in EXTENDED_COVERED_TOOLS:
                tiers_by_tool.setdefault(tool, set()).add(prompt.tier)
    all_tiers = {tier for tiers in tiers_by_tool.values() for tier in tiers}
    assert {"single_tool", "sequential"} <= all_tiers, (
        f"extended tools cover tiers {sorted(all_tiers)}; "
        f"expected at least single_tool and sequential"
    )


def test_no_unknown_tools_in_tools_required() -> None:
    """Catch typos in YAML files before the eval suite spawns the server."""
    prompts = load_prompts()
    for prompt in prompts:
        unknown = set(prompt.tools_required) - ALL_TOOLS
        assert not unknown, (
            f"prompt {prompt.id!r} declares tools_required={sorted(unknown)} "
            f"not in ALL_TOOLS={sorted(ALL_TOOLS)} — typo or unregistered tool"
        )


def test_credentialed_prompts_declare_requirements() -> None:
    """Prompts that exercise a credentialed source gate on requires_credential.

    A prompt is skipped when its secret is absent, so a credentialed call
    must declare the matching requirement or it would fail (not skip) in the
    default no-secret environment.
    """
    prompts = load_prompts()
    for prompt in prompts:
        uses_spacetrack = any(
            step.arg_constraints.get("source", {}).get("equals") == "space-track"
            for trace in prompt.permitted_traces
            for step in trace
        )
        if "satellite_metadata" in prompt.tools_required:
            assert "discosweb" in prompt.requires_credential, (
                f"{prompt.id!r} calls satellite_metadata but does not require the "
                f"discosweb credential"
            )
        if uses_spacetrack:
            assert "spacetrack" in prompt.requires_credential, (
                f"{prompt.id!r} uses source='space-track' but does not require the "
                f"spacetrack credential"
            )


def test_every_spice_tool_has_a_single_tool_prompt() -> None:
    """Each SPICE tool appears in at least one single-tool-tier prompt."""
    prompts = load_prompts()
    covered = {
        tool for prompt in prompts if prompt.tier == "single_tool" for tool in prompt.tools_required
    }
    missing = SPICE_TOOLS - covered
    assert not missing, (
        f"SPICE tools without a single_tool prompt: {sorted(missing)}; "
        f"every SPICE tool must be covered at the single-tool tier"
    )


def test_every_spice_tool_has_a_sequential_prompt() -> None:
    """Each SPICE tool appears in at least one sequential-tier prompt."""
    prompts = load_prompts()
    covered = {
        tool for prompt in prompts if prompt.tier == "sequential" for tool in prompt.tools_required
    }
    missing = SPICE_TOOLS - covered
    assert not missing, (
        f"SPICE tools without a sequential prompt: {sorted(missing)}; "
        f"every SPICE tool must be covered at the sequential tier"
    )


def test_spice_prompts_declare_requires_spice() -> None:
    prompts = load_prompts()
    for prompt in prompts:
        if SPICE_TOOLS & set(prompt.tools_required):
            assert prompt.requires_spice, (
                f"{prompt.id!r} drives a SPICE tool but does not set requires_spice: true"
            )


def test_gmat_prompts_declare_requires_gmat() -> None:
    prompts = load_prompts()
    gmat_tools = {
        "gmat_run_mission",
        "gmat_sweep",
        "gmat_execute_script",
        "gmat_read_run_artefact",
        "gmat_validate_script",
    }
    for prompt in prompts:
        if gmat_tools & set(prompt.tools_required):
            assert prompt.requires_gmat, (
                f"{prompt.id!r} drives a GMAT tool but does not set requires_gmat: true"
            )


def test_prompt_ids_unique() -> None:
    prompts = load_prompts()
    ids = [p.id for p in prompts]
    duplicates = [i for i, count in Counter(ids).items() if count > 1]
    assert not duplicates, f"duplicate prompt ids: {duplicates}"


def test_every_viz_tool_has_a_single_tool_prompt() -> None:
    """Each viz tool appears in at least one single-tool-tier prompt."""
    prompts = load_prompts()
    covered = {
        tool for prompt in prompts if prompt.tier == "single_tool" for tool in prompt.tools_required
    }
    missing = VIZ_TOOLS - covered
    assert not missing, (
        f"viz tools without a single_tool prompt: {sorted(missing)}; "
        f"every viz tool must be covered at the single-tool tier"
    )


def test_viz_has_a_sequential_prompt() -> None:
    """The viz suite carries at least one sequential prompt (e.g. sgp4 -> plot)."""
    prompts = load_prompts()
    sequential_viz = {
        tool
        for prompt in prompts
        if prompt.tier == "sequential"
        for tool in prompt.tools_required
        if tool in VIZ_TOOLS
    }
    assert sequential_viz, (
        "no sequential-tier prompt drives a viz tool; the suite must exercise at "
        "least one viz tool beyond the single-tool tier (e.g. sgp4_propagate -> "
        "plot_ground_track)"
    )


def test_viz_prompts_declare_requires_viz() -> None:
    prompts = load_prompts()
    for prompt in prompts:
        if VIZ_TOOLS & set(prompt.tools_required):
            assert prompt.requires_viz, (
                f"{prompt.id!r} drives a viz tool but does not set requires_viz: true"
            )


def test_viz_prompts_declare_expected_attachment() -> None:
    """Every viz prompt asserts an attachment golden (presence + declared type)."""
    prompts = load_prompts()
    for prompt in prompts:
        if VIZ_TOOLS & set(prompt.tools_required):
            assert prompt.expected_attachment in ("image", "resource"), (
                f"{prompt.id!r} drives a viz tool but does not declare an "
                f"expected_attachment ('image' or 'resource'); viz goldens must be "
                f"attachment-aware"
            )


def test_non_viz_prompts_omit_expected_attachment() -> None:
    """Only viz prompts carry an attachment golden — guards against stray fields."""
    prompts = load_prompts()
    for prompt in prompts:
        if not (VIZ_TOOLS & set(prompt.tools_required)):
            assert prompt.expected_attachment is None, (
                f"{prompt.id!r} declares expected_attachment={prompt.expected_attachment!r} "
                f"but drives no viz tool"
            )
