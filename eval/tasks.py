"""Inspect AI Task definitions for the astrodynamics-mcp eval suite.

Spawns the local ``astrodynamics-mcp stdio`` binary as the MCP server
under test, hands its tools to a :func:`~inspect_ai.agent.react` agent,
and scores each conversation with the hybrid trace + functional-answer
scorer.

Run with::

    uv run inspect eval eval/tasks.py --model anthropic/claude-sonnet-4-6

Any Inspect-AI-supported model provider works locally; the CI gate
configures the GitHub Models provider with sampling parameters locked by
the verification spike (see ``eval/README.md``).
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Inspect AI loads this file by path (e.g. ``inspect eval eval/tasks.py``),
# which puts only ``eval/`` on sys.path — not the repo root. Without the
# repo root, ``from eval._prompts import ...`` below fails with
# ``ModuleNotFoundError: No module named 'eval'``. Pytest covers this via
# ``pythonpath = ["."]`` in pyproject.toml; the runtime CLI path needs its
# own escape hatch.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inspect_ai import Task, task  # noqa: E402
from inspect_ai.agent import as_solver, react  # noqa: E402
from inspect_ai.dataset import MemoryDataset, Sample  # noqa: E402
from inspect_ai.solver import Solver  # noqa: E402
from inspect_ai.tool import mcp_server_stdio  # noqa: E402

from eval._prompts import PromptSpec, load_prompts  # noqa: E402
from eval.scoring import hybrid_scorer  # noqa: E402

# Console-script name installed by [project.scripts] in pyproject.toml.
_SERVER_COMMAND = "astrodynamics-mcp"
_SERVER_ARGS: tuple[str, ...] = ("stdio",)


def _sample_from_prompt(prompt: PromptSpec) -> Sample:
    """Build an Inspect AI :class:`Sample` from a typed :class:`PromptSpec`.

    The full prompt spec is serialised onto ``Sample.metadata`` so the
    scorer can reconstruct it inside its (synchronous) closure. ``target``
    is unused — the hybrid scorer reads everything from metadata.
    """
    return Sample(
        id=prompt.id,
        input=prompt.prompt,
        metadata={
            "tier": prompt.tier,
            "tools_required": list(prompt.tools_required),
            "permitted_traces": [
                [step.model_dump() for step in trace] for trace in prompt.permitted_traces
            ],
            "functional_answer": list(prompt.functional_answer),
            "notes": prompt.notes,
        },
    )


def _build_dataset(prompts: Iterable[PromptSpec]) -> MemoryDataset:
    samples = [_sample_from_prompt(p) for p in prompts]
    return MemoryDataset(samples=samples, name="astrodynamics_mcp_eval")


def _build_solver() -> Solver:
    """Build the react agent wired to the local stdio MCP server.

    The :func:`~inspect_ai.tool.mcp_server_stdio` returns an MCP-server
    handle that :func:`~inspect_ai.agent.react` knows how to keep alive
    via :func:`~inspect_ai.tool.mcp_connection` for the duration of each
    sample.
    """
    server: Any = mcp_server_stdio(
        name="astrodynamics-mcp",
        command=_SERVER_COMMAND,
        args=list(_SERVER_ARGS),
    )
    agent = react(
        name="astrodynamics_mcp_eval_agent",
        description="Agent under test for the astrodynamics-mcp eval suite.",
        tools=[server],
    )
    solver: Solver = as_solver(agent)
    return solver


@task
def astrodynamics_mcp_eval() -> Task:
    """The full eval suite — every prompt under ``eval/prompts/``."""
    prompts = load_prompts()
    return Task(
        dataset=_build_dataset(prompts),
        solver=_build_solver(),
        scorer=hybrid_scorer(),
    )


@task
def astrodynamics_mcp_eval_subset(tier: str = "single_tool") -> Task:
    """Tier-filtered slice of the suite — used by the CI gate when budget is tight.

    The free-tier GitHub Models quota will not in general fit the full
    suite per pull request; the CI workflow invokes this task with the
    tier(s) it can afford and runs the full suite on
    ``workflow_dispatch``.
    """
    prompts = [p for p in load_prompts() if p.tier == tier]
    return Task(
        dataset=_build_dataset(prompts),
        solver=_build_solver(),
        scorer=hybrid_scorer(),
    )
