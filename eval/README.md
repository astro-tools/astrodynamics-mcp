# Eval suite

Inspect AI evaluation suite for `astrodynamics-mcp`. The live regression
contract on tool-selection quality, argument binding, and the JSON shape an
LLM client sees when it calls a tool.

The suite is **dev-only** — it is not packaged or shipped to PyPI. The
`eval/` directory lives at the repo root and is invoked from a developer
checkout.

## What this catches that other checks don't

| Check | Catches |
| --- | --- |
| `mypy` / `ruff` | Static type and lint errors in the Python source. |
| `pytest` unit tests | Numerical correctness of the upstream-library wrappers. |
| `server_lint` (in `pytest`) | Static description-discipline (length, examples, common-mistake warnings). |
| **Eval suite** (this directory) | Does the LLM *actually* call the right tool with the right arguments under prompt variation, and does the response shape let the LLM read the answer back. |

The hybrid scorer combines two checks per prompt:

- **Permitted-trace check.** The model's tool-call sequence must match one of
  the prompt's `permitted_traces`. Catches tool-selection regressions
  (wrong tool picked) and argument-binding regressions (right tool,
  wrong-shaped arg). Each trace entry constrains the tool name and a
  subset of its arguments — unconstrained arguments may take any value.
- **Functional-answer check.** A set of predicates over the final tool
  response's JSON. Catches the right-tool-wrong-call and
  wrong-tool-right-number-by-coincidence cases the trace check cannot.

Neither check alone catches what the other catches; both must pass for
the prompt to score 1.

## Layout

```
eval/
├── README.md             ← this file
├── _constraints.py       ← argument-constraint matcher (permitted_traces)
├── _functional.py        ← functional-answer predicates over response JSON
├── _prompts.py           ← pydantic models + YAML loader for prompts/
├── scoring.py            ← Inspect AI Scorer combining the two checks
├── tasks.py              ← Inspect AI Task wiring stdio server + react() agent
└── prompts/
    └── *.yaml            ← one prompt per file (30 v0.1 prompts: 20 single-tool + 8 sequential + 2 planning)
```

## Running locally

The suite spawns the local `astrodynamics-mcp stdio` binary as its MCP
server under test. Install the package and the dev group first:

```bash
uv sync --all-groups
```

Then run:

```bash
uv run inspect eval eval/tasks.py --model anthropic/claude-sonnet-4-6
```

Any Inspect-AI-supported provider works locally; CI runs against GitHub
Models (see the spike-findings section below).

## Prompt YAML schema

Each `eval/prompts/<slug>.yaml` defines one prompt. Schema (validated by
`eval/_prompts.py` on load — invalid files fail eagerly):

```yaml
prompt: "<the natural-language user message>"
tier: single_tool | sequential | planning
tools_required: [<v0.1 tool names that must be available>]
permitted_traces:
  - - tool: <tool_name>
      arg_constraints:
        <arg_name>: <ArgConstraint>      # see _constraints.py for the vocabulary
        ...
    - tool: <next_tool_in_chain>
      arg_constraints: ...
  - - tool: <alternative_first_step>     # any permitted trace passing is enough
      ...
functional_answer:
  - path: "$.<jsonpath>"                  # JSON-path-lite over the final tool response
    <predicate>: <value>                  # see _functional.py for the vocabulary
  ...
notes: "<optional human-only note: known model quirks, related prompts, etc.>"
```

The constraint vocabulary is defined in `eval/_constraints.py`
(`equals`, `one_of`, `case_insensitive_equals`, `case_insensitive_contains`,
`length`, `numeric_tolerance`, `field_constraints`, `has_fields`).
The functional-predicate vocabulary is defined in `eval/_functional.py`
(`equals`, `in_range`, `l2_in_range`, `length`, `present`,
`case_insensitive_contains`, `starts_with`, `all_equal`,
`numeric_tolerance`). `l2_in_range` interprets the path's value as a
vector and checks the L2 norm lies in the `[min, max]` bracket — used
for "is this r-vector in the LEO altitude band" type checks.

`permitted_traces` is a list of *alternative* traces; at least one must
match. Each trace is a list of consecutive `{tool, arg_constraints}`
entries; the model's recorded tool-call sequence must contain them in
order (extra interleaved calls are tolerated only if the prompt's notes
say so — defaults to strict).

## Regenerating goldens

Reference outputs that drive the `functional_answer` checks live next to
each prompt's YAML when the upstream is deterministic. When the upstream
library version pin changes, regenerate with:

```bash
uv run python eval/_regenerate_goldens.py        # not yet implemented — placeholder
```

Review the diff before committing — golden regeneration is a deliberate
action, never a silent CI fixup.

## Verification spike findings

Each subsection is a dated, source-cited record. The CI workflow under
`.github/workflows/eval.yml` references these by section heading when
explaining its model choice, sampling parameters, and pass threshold.

### Inspect AI ↔ GitHub Models integration (2026-05-25)

GitHub Models has **no native Inspect AI provider**. The catalogue is
reached through Inspect AI's generic `openai-api/<name>/<model>`
adapter, which derives env-var names from `<name>` (uppercased,
hyphens-to-underscores).

Working configuration:

| Setting | Value |
| --- | --- |
| Inspect provider string | `openai-api/github/openai/gpt-4.1-mini` |
| `GITHUB_API_KEY` | `$(gh auth token)` (current OAuth token has implicit `models:read`) |
| `GITHUB_BASE_URL` | `https://models.github.ai/inference` |
| Optional dep | `openai` (Inspect's OpenAI-compatible adapter requires it) |

Verified by running a one-sample no-tool task and observing
`accuracy=1.000`.

**Model choice — important deviation from the charter.** The GH Models
catalogue does **not** carry any Anthropic / Claude models as of this
date (publishers: OpenAI, Meta, Mistral, DeepSeek, Microsoft, Cohere,
AI21, xAI). The charter §3 plan ("Claude Sonnet 4.6 via GitHub Models")
is currently unrealisable; we run on `openai/gpt-4.1-mini`, the
cheapest tool-calling-capable low-tier model on the catalogue. Re-
evaluate at v0.2 / v0.3 if the catalogue changes.

### GitHub Models rate-limit observations (2026-05-25)

Per-response headers from a live `gpt-4.1-mini` call:

```
x-ratelimit-key:                  gpt-4.1-mini
x-ratelimit-limit-requests:       1000
x-ratelimit-limit-tokens:         1000000
x-ratelimit-renewalperiod-requests: 60
x-ratelimit-renewalperiod-tokens:   60
x-ratelimit-remaining-requests:   999
x-ratelimit-remaining-tokens:     999992
```

Interpretation: **1000 requests / minute and 1M tokens / minute**, scoped
per-model-per-key. No visible per-day cap in the headers.

These are far above the published "Copilot Free Low tier" numbers (15
req/min, 150 req/day) referenced in earlier GH Models docs. The bare
OAuth token issued by `gh auth login` evidently lands in a more
permissive tier; whether a workflow-issued `GITHUB_TOKEN` with
`permissions: models: read` sees the same numbers is *not yet
verified* — re-measure on the first CI run.

**Practical implication for the gate:** the full 30-prompt suite (≤
~150 LLM requests assuming 3-5 tool calls per prompt) fits comfortably
inside the per-minute budget. The original concern that per-PR runs
would need subsetting does not apply at the observed limits. The
workflow still exposes the `astrodynamics_mcp_eval_subset(tier=...)`
task for use when the budget tightens in future.

### Fixed-seed determinism observations (2026-05-25)

Five identical chat-completion requests against `gpt-4.1-mini` with
`seed=42, temperature=0, top_p=1.0, max_tokens=200` produced **five
different outputs** byte-for-byte (lengths 392 / 396 / 383 / 395 / 363).
The differences were paraphrase-level (synonyms, reordering inside a
list item), not semantic — but reproducibility cannot be assumed.

OpenAI's documented `seed` semantics are "best-effort", and GH Models
appears to apply the same caveat.

**Regression-gate policy:** the per-PR check uses `temperature=0` (to
keep outputs as close to greedy as possible) but does **not** rely on
seed-based reproducibility. The pass threshold is set below the
charter §4 DoD nominal of ≥90% to absorb expected stochasticity. A
single-prompt flake should not flip the gate; a deterioration in *score
trend* over a sequence of PRs is what we treat as a real regression.

## CI gate behaviour

The per-PR eval workflow runs the suite against
`openai-api/github/openai/gpt-4.1-mini` on every pull request and
posts the score as a PR comment. The full 30-prompt suite runs both on
`pull_request` and on `workflow_dispatch`; subsetting (via the
`astrodynamics_mcp_eval_subset` task) is reserved for the case where
budget tightens later.

The gate is **GitHub Models free-tier only** — no paid Anthropic /
OpenAI keys are used or required, in line with charter §2's
non-paid-API-key non-goal.

**Pass threshold:** ≥80% of goldens. Slightly below the charter §4 DoD
nominal of ≥90%, calibrated against the determinism observations above
so a single-prompt flake doesn't flip the gate. The threshold is
revisited at v0.2 once a multi-run history exists.
