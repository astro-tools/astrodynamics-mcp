"""Inspect AI Task definitions for the astrodynamics-mcp eval suite.

Spawns the local ``astrodynamics-mcp stdio`` binary as the MCP server
under test, hands a **per-sample subset** of its tools to a
:func:`~inspect_ai.agent.react` agent, and scores each conversation
with the hybrid trace + functional-answer scorer.

The per-sample subset is the load-bearing optimisation: GitHub Models
caps the workflow token at 8000 input tokens per request, while the
full 8-tool MCP surface adds up to ~6100 tokens of schema alone. By
exposing only each prompt's ``tools_required`` tools, single-tool
prompts pay ~200-1500 tokens of schema rather than ~6100, leaving
plenty of budget for the prompt, system message, and turn-by-turn
history.

Run with::

    uv run inspect eval eval/tasks.py@astrodynamics_mcp_eval \\
      --model openai-api/github/openai/gpt-4o \\
      -M strict_tools=false \\
      --temperature 0
"""

from __future__ import annotations

import os
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
from inspect_ai.solver import Generate, Solver, TaskState, solver  # noqa: E402
from inspect_ai.tool import mcp_server_stdio, mcp_tools  # noqa: E402

from eval._prompts import PromptSpec, load_prompts, requirements_met  # noqa: E402
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
            "expected_attachment": prompt.expected_attachment,
            "notes": prompt.notes,
        },
    )


def _build_dataset(prompts: Iterable[PromptSpec]) -> MemoryDataset:
    samples = [_sample_from_prompt(p) for p in prompts]
    return MemoryDataset(samples=samples, name="astrodynamics_mcp_eval")


def _runnable(prompts: Iterable[PromptSpec]) -> list[PromptSpec]:
    """Drop prompts whose declared requirements aren't satisfiable in this env.

    Credentialed and GMAT-backed prompts are *skipped* — not failed — when
    their secrets or GMAT install are absent (see ``PromptSpec.requires_*``).
    Filtering them out of the dataset keeps them from running and scoring
    zero; ``eval/_ci_report.py`` re-derives the skipped set from the same
    env so the report can show them as skipped rather than missing.
    """
    return [p for p in prompts if requirements_met(p)]


@solver
def per_sample_react_solver() -> Solver:
    """A solver that builds a fresh ``react`` agent per sample with subsetted tools.

    Reads ``state.metadata['tools_required']`` and exposes only those
    tools from the MCP server to the model. The server itself is the
    same singleton across samples — :func:`mcp_tools` produces a
    filtered :class:`ToolSource` view rather than re-spawning the
    subprocess.
    """
    # Pass the full environment to the spawned server. Without an explicit
    # env, the MCP stdio client scrubs to a minimal allowlist
    # (HOME/PATH/SHELL/TERM/USER/LOGNAME) — dropping GMAT_ROOT (so the
    # GMAT-backed tools can't locate the install) and the
    # ASTRODYNAMICS_MCP_* credential vars (so credentialed tools never see
    # their secrets). The server is our own trusted binary, so inheriting
    # the eval job's environment is the intended behaviour.
    server: Any = mcp_server_stdio(
        name="astrodynamics-mcp",
        command=_SERVER_COMMAND,
        args=list(_SERVER_ARGS),
        env=dict(os.environ),
    )

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        required = state.metadata.get("tools_required") if state.metadata else None
        if required:
            tools_source: Any = mcp_tools(server, tools=list(required))
        else:
            tools_source = server
        # submit=False removes the react agent's "submit your answer" turn
        # at the end of the conversation. Our scorer reads the final tool
        # response directly off state.messages — the submit turn was wasted
        # latency. Saves one LLM round-trip per sample.
        agent = react(
            name="astrodynamics_mcp_eval_agent",
            description="Agent under test for the astrodynamics-mcp eval suite.",
            tools=[tools_source],
            submit=False,
        )
        agent_solver: Solver = as_solver(agent)
        return await agent_solver(state, generate)

    return solve


@task
def astrodynamics_mcp_eval() -> Task:
    """The full eval suite — every runnable prompt under ``eval/prompts/``.

    Prompts whose ``requires_credential`` / ``requires_gmat`` prerequisites
    are absent in this environment are skipped (filtered out), not failed.
    """
    prompts = _runnable(load_prompts())
    return Task(
        dataset=_build_dataset(prompts),
        solver=per_sample_react_solver(),
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
    prompts = _runnable(p for p in load_prompts() if p.tier == tier)
    return Task(
        dataset=_build_dataset(prompts),
        solver=per_sample_react_solver(),
        scorer=hybrid_scorer(),
    )
