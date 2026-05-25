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
    └── *.yaml            ← one prompt per file (populated in the prompt-authoring phase)
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
(`equals`, `in_range`, `length`, `present`, `case_insensitive_contains`,
`starts_with`, `all_equal`, `numeric_tolerance`).

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

Filled in once the GitHub Models / Inspect-AI integration spikes run.
Each spike updates this section with date-stamped findings the CI gate
can cite by section heading.

### Inspect AI ↔ GitHub Models integration (pending)

### GitHub Models rate-limit observations (pending)

### Fixed-seed determinism observations (pending)

## CI gate behaviour

The per-PR eval workflow runs a calibrated subset of the suite against
GitHub Models on every pull request and posts the score as a PR
comment. The full suite runs on `workflow_dispatch` and on release-cut
PRs. Pass threshold and subset size are calibrated against the spike
findings above.

The gate is **GitHub Models free-tier only** — no paid Anthropic /
OpenAI keys are used or required. If the free-tier quota constrains the
per-PR run below the full suite, the workflow falls back to a tier-
balanced subset that exercises every v0.1 tool at least once.
