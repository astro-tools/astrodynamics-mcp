"""Coverage gate over ``eval/prompts/*.yaml``.

Asserts the suite stays populated (≥40 prompts) and within the
tier-distribution discipline, that every *core* tool has a single-tool
prompt and a sequential-or-planning prompt, and that each extended tool
the suite covers is referenced by at least one prompt. A failing
assertion here means a prompt got added or removed without rebalancing —
the regression catches it before merge.

The strict per-tool tier matrix applies to the core read-only tools
only. The extended tools (GMAT + DISCOSweb metadata) are covered more
loosely: GMAT prompts skip where no install exists and the credentialed
prompts skip without secrets, so forcing each onto every tier would not
buy real signal in the default gate.

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

# Extended tools the suite is expected to exercise with at least one
# prompt (gmat_validate_script has no prompt yet, so it is excluded).
EXTENDED_COVERED_TOOLS: frozenset[str] = EXTENDED_TOOLS - {"gmat_validate_script"}

ALL_TOOLS: frozenset[str] = CORE_TOOLS | EXTENDED_TOOLS

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
    """Happy-path credentialed prompts gate on requires_credential.

    The error-path prompts (which assert credential_required.* fires) must
    *not* gate — they run in the default no-secret env. Distinguish them by
    whether any trace step carries expect_error.
    """
    prompts = load_prompts()
    for prompt in prompts:
        uses_spacetrack = any(
            step.arg_constraints.get("source", {}).get("equals") == "space-track"
            for trace in prompt.permitted_traces
            for step in trace
        )
        is_error_path = any(
            step.expect_error is not None for trace in prompt.permitted_traces for step in trace
        )
        if "satellite_metadata" in prompt.tools_required and not is_error_path:
            assert "discosweb" in prompt.requires_credential, (
                f"{prompt.id!r} calls satellite_metadata on a happy path but does "
                f"not require the discosweb credential"
            )
        if uses_spacetrack and not is_error_path:
            assert "spacetrack" in prompt.requires_credential, (
                f"{prompt.id!r} uses source='space-track' on a happy path but does "
                f"not require the spacetrack credential"
            )


def test_gmat_prompts_declare_requires_gmat() -> None:
    prompts = load_prompts()
    gmat_tools = {"gmat_run_mission", "gmat_sweep", "gmat_execute_script", "gmat_read_run_artefact"}
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
