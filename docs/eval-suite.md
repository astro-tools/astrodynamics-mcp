# Eval suite

The eval suite is the live regression contract on tool-description quality.
It runs roughly 30 prompts against the local `astrodynamics-mcp stdio`
binary, scores each one against a hybrid (trace + functional) check, and
posts a pass / fail back to every PR.

For the full reference — prompt schema, scorer vocabulary, provider
matrix, CI-gate authentication — see
[`eval/README.md`](https://github.com/astro-tools/astrodynamics-mcp/blob/main/eval/README.md)
in the source tree.

## What it catches

| Check | Catches |
| --- | --- |
| `mypy` / `ruff` | Static type and lint errors. |
| `pytest` unit tests | Numerical correctness of the upstream-library wrappers. |
| `server_lint` (under `pytest`) | Static description-discipline (length, examples, common-mistake warnings). |
| **Eval suite** | Whether the LLM *actually* calls the right tool with the right arguments under prompt variation, and whether the response shape lets the LLM read the answer back. |

The four layers compose: static checks catch syntactic regressions
cheaply, the eval suite catches the semantic regressions only a real
model call can surface (tool-selection drift, arg-binding errors,
description ambiguity).

## The hybrid scorer

Each prompt scores 1 iff both checks pass:

- **Permitted-trace check.** The recorded tool-call sequence matches one
  of the prompt's `permitted_traces`. Each entry constrains the tool
  name and a subset of its arguments — unconstrained arguments may take
  any value. Catches tool-selection regressions (wrong tool picked) and
  argument-binding regressions (right tool, wrong-shaped arg).
- **Functional-answer check.** A set of predicates over the final tool
  response's JSON — `equals`, `in_range`, `l2_in_range`, `present`,
  `case_insensitive_contains`, `starts_with`, `numeric_tolerance`,
  others. Catches the right-tool-wrong-call and
  wrong-tool-right-number-by-coincidence cases the trace check cannot.

Neither check alone catches what the other catches; both must pass for
the prompt to score 1. The full predicate vocabularies live in
[`eval/_constraints.py`](https://github.com/astro-tools/astrodynamics-mcp/blob/main/eval/_constraints.py)
and
[`eval/_functional.py`](https://github.com/astro-tools/astrodynamics-mcp/blob/main/eval/_functional.py).

## Tier structure

The ~30 prompts split across three tiers:

| Tier | Count | Shape |
| --- | --- | --- |
| `single_tool` | 20 | Direct one-tool questions (e.g. "convert 2026-05-23T12:00:00Z from UTC to TT"). |
| `sequential` | 8 | Chained calls (e.g. "fetch the ISS TLE, then propagate it to ..."). |
| `planning` | 2 | Multi-step questions where the tool sequence is non-obvious from the prompt. |

Every v0.1 tool has at least one single-tool prompt and at least one
sequential-or-planning prompt.

## Run it locally

The suite spawns the local `astrodynamics-mcp stdio` binary as its MCP
server under test:

```bash
uv sync --all-groups

GITHUB_API_KEY="$(gh auth token)" \
GITHUB_BASE_URL="https://models.github.ai/inference" \
  uv run inspect eval eval/tasks.py \
  --model openai-api/github/openai/gpt-4.1-mini \
  --temperature 0
```

The `gh auth token` value carries implicit `models:read` access on
free-tier GitHub accounts. For other providers (Claude direct, OpenAI
direct, Google AI Studio, AWS Bedrock, local Ollama) see
[`eval/README.md`](https://github.com/astro-tools/astrodynamics-mcp/blob/main/eval/README.md#running-against-other-providers).

## How this is validated

Two layers beyond the eval suite itself:

- **Transport equivalence.** The same tool call over stdio and over
  Streamable HTTP returns the same JSON payload byte-for-byte (modulo
  session-id fields). Exercised under `pytest` with the
  `integration` marker.
- **Reference-output regression.** Each tool has a small set of fixed
  inputs whose outputs are diffed against committed golden JSON. The
  goldens are regenerated deliberately when an upstream-library version
  pin changes.

Together with the per-PR eval gate these form the validation triangle:
unit tests check upstream wrapping, reference goldens check
deterministic outputs, and the eval suite checks that an LLM picks the
right tool and reads the right answer back.

## CI gate

The gate is `workflow_dispatch`-only at v0.1 — `.github/workflows/eval.yml`
runs the suite against `openai-api/github/openai/gpt-4.1-mini` and posts
the markdown report to the workflow Summary page. The Inspect AI log
uploads as a workflow artefact (14-day retention) — download it and run
`uv run inspect view logs/` locally to replay any failing prompt's
conversation.

The pass threshold is ≥80% of goldens, calibrated to absorb the expected
single-prompt flake (GH Models' `seed` is best-effort, not byte-deterministic).

## Where this comes from

The hybrid scorer pattern, the prompt tiering, and the GitHub Models
free-tier execution surface are documented in the
[charter](https://github.com/astro-tools/astrodynamics-mcp/blob/main/README.md).
The eval suite as a regression contract on tool-description quality is
the project's main engineering content beyond the upstream-library
wrappers themselves — it's how we tell whether a tool-description tweak
helped or hurt the LLM consumer.
