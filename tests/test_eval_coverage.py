"""Coverage gate over ``eval/prompts/*.yaml``.

Asserts the suite stays within the tier-distribution discipline
(20 single-tool / 8 sequential / 2 planning) and that every registered
tool has at least one single-tool prompt and at least one
sequential-or-planning prompt. A failing assertion here means a prompt
got added or removed without rebalancing — the regression catches it
before merge.

These also double as the test that the live suite is *populated* — an
empty ``eval/prompts/`` directory would fail loudly here rather than
silently producing a green eval run with zero samples.
"""

from __future__ import annotations

from collections import Counter

from eval._prompts import load_prompts

# Canonical tool surface that prompts may reference. Update when a tool
# is added or removed from the MCP server's registration.
ASTRODYNAMICS_TOOLS: frozenset[str] = frozenset(
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

MIN_PROMPTS = 30
MIN_SINGLE_TOOL = 20
MIN_SEQUENTIAL = 8
MIN_PLANNING = 2


def test_prompt_directory_populated() -> None:
    prompts = load_prompts()
    assert len(prompts) >= MIN_PROMPTS, (
        f"eval/prompts/ has {len(prompts)} prompts; target ~{MIN_PROMPTS}"
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


def test_every_tool_has_a_single_tool_prompt() -> None:
    prompts = load_prompts()
    covered = {
        tool for prompt in prompts if prompt.tier == "single_tool" for tool in prompt.tools_required
    }
    missing = ASTRODYNAMICS_TOOLS - covered
    assert not missing, (
        f"tools without a single_tool prompt: {sorted(missing)}; "
        f"every tool must be covered at the single-tool tier"
    )


def test_every_tool_has_a_sequential_or_planning_prompt() -> None:
    prompts = load_prompts()
    covered = {
        tool
        for prompt in prompts
        if prompt.tier in ("sequential", "planning")
        for tool in prompt.tools_required
    }
    missing = ASTRODYNAMICS_TOOLS - covered
    assert not missing, (
        f"tools without a sequential/planning prompt: {sorted(missing)}; "
        f"every tool must be covered beyond the single-tool tier"
    )


def test_no_unknown_tools_in_tools_required() -> None:
    """Catch typos in YAML files before the eval suite spawns the server."""
    prompts = load_prompts()
    for prompt in prompts:
        unknown = set(prompt.tools_required) - ASTRODYNAMICS_TOOLS
        assert not unknown, (
            f"prompt {prompt.id!r} declares tools_required={sorted(unknown)} "
            f"not in ASTRODYNAMICS_TOOLS={sorted(ASTRODYNAMICS_TOOLS)} — typo or unregistered tool"
        )


def test_prompt_ids_unique() -> None:
    prompts = load_prompts()
    ids = [p.id for p in prompts]
    duplicates = [i for i, count in Counter(ids).items() if count > 1]
    assert not duplicates, f"duplicate prompt ids: {duplicates}"
