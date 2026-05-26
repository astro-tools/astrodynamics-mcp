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
    └── *.yaml            ← one prompt per file (30 prompts: 20 single-tool + 8 sequential + 2 planning)
```

## Running locally

The suite spawns the local `astrodynamics-mcp stdio` binary as its MCP
server under test. Install the package and the dev group first:

```bash
uv sync --all-groups
```

Then run the suite against GitHub Models with the configuration the CI
gate uses (see "Model and provider" below):

```bash
GITHUB_API_KEY="$(gh auth token)" \
GITHUB_BASE_URL="https://models.github.ai/inference" \
  uv run inspect eval eval/tasks.py \
  --model openai-api/github/openai/gpt-4.1-mini \
  --temperature 0
```

CI runs against the GitHub Models configuration documented below. For
other providers, see ["Running against other providers"](#running-against-other-providers).

## Prompt YAML schema

Each `eval/prompts/<slug>.yaml` defines one prompt. Schema (validated by
`eval/_prompts.py` on load — invalid files fail eagerly):

```yaml
prompt: "<the natural-language user message>"
tier: single_tool | sequential | planning
tools_required: [<tool names that must be available>]
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

## Model and provider

The CI gate and the recommended local configuration run against GitHub
Models. GitHub Models has **no native Inspect AI provider** — we reach
it through Inspect AI's generic `openai-api/<name>/<model>` adapter,
which derives env-var names from `<name>` (uppercased,
hyphens-to-underscores).

Working configuration:

| Setting | Value |
| --- | --- |
| Inspect provider string | `openai-api/github/openai/gpt-4.1-mini` |
| `GITHUB_API_KEY` | `$(gh auth token)` (the OAuth token has implicit `models:read` access) |
| `GITHUB_BASE_URL` | `https://models.github.ai/inference` |
| Optional dep | `openai` (Inspect's OpenAI-compatible adapter requires it) |

### Why `openai/gpt-4.1-mini`

Selection criteria, in order:

1. **Tool-calling capability** — non-negotiable; the suite is about
   tool use.
2. **Free-tier accessibility** — must be callable with `gh auth token`
   locally and a personal-owned `MODELS_PAT` in CI. No paid keys, no
   project budget for inference.
3. **Daily-quota fit** — the ~150-request-per-run suite must fit
   within the Free-plan daily cap of the model's rate-limit tier.
4. **Tool-use quality** — frontier-class capability so the score
   measures *our tool descriptions*, not the model's weakness.

GitHub Models gates by per-model rate-limit tier; on the Free plan the
binding constraint is the per-day cap, not the per-minute or token
budget. Source:
[GitHub Models — Rate limits](https://docs.github.com/en/github-models/use-github-models/prototyping-with-ai-models#rate-limits).

| Model | Tier | req/min | req/day | concurrent | Verdict |
| --- | --- | --- | --- | --- | --- |
| `openai/gpt-4o` | High | 10 | 50 | 2 | 50/day caps mid-run; unworkable. |
| `openai/gpt-4.1` | High | 10 | 50 | 2 | Same daily cap; same verdict. |
| `openai/gpt-4o-mini` | Low | 15 | 150 | 5 | Fits one full suite per day. Viable alternative. |
| `openai/gpt-4.1-mini` | Low | 15 | 150 | 5 | **Selected.** Same tier and quota as `gpt-4o-mini`; newer architecture. |
| `openai/gpt-5*`, `openai/o3-mini`, `openai/o4-mini` | — | — | — | — | Catalogue lists them but the endpoint returns `400 unavailable_model` / `403`; not free-tier accessible. |
| `cohere/cohere-command-r-plus-08-2024`, `ai21-labs/ai21-jamba-1.5-large` | — | — | — | — | Catalogue lists them; endpoint returns `400`. |

Two practical implications:

- **Daily-cap fit.** At ~150 requests per run (estimate — count properly
  on the next successful run; see below), the suite fits roughly one run
  per UTC day on Free Low tier. The gate must be dispatched at most once
  per day until the suite shrinks or the plan is upgraded. High-tier
  models cap mid-run and produce no usable score.
- **Concurrency cap.** Inspect AI flags must stay under the
  5-concurrent ceiling. `.github/workflows/eval.yml` uses
  `--max-samples 3 --max-connections 4`, sitting 1–2 slots below the
  cap so multi-turn tool-call bursts within a sample don't push over.
  `--fail-on-error 5` aborts the run once a quota-blown state is
  obvious, preserving the rest of the day's quota.

Response-header rate limits (`x-ratelimit-limit-requests` and the like)
report large bucket sizes that **do not match** the published per-plan
policy. Always size against the published table, not against headers.

### Counting requests

The "~150 requests per run" figure is an estimate from the suite's
30 prompts × variable tool-call turns per prompt; it has not been
measured against a clean successful run. Before the next change to
suite size, concurrency, or model selection, count actual requests
from the Inspect AI log:

```bash
inspect view logs/<latest>.eval --json | jq '[.samples[].messages[] | select(.role == "assistant")] | length'
```

If the real count is materially below 150, there is daily-cap headroom
for retries and the concurrency settings can be relaxed; if it's at or
above 150, the suite needs trimming or per-tier dispatch instead.

### Sampling parameters and determinism

Verified 2026-05-25: five identical chat-completion requests against
`gpt-4o` with `seed=42, temperature=0, top_p=1.0, max_tokens=200`
produce different outputs byte-for-byte (paraphrase-level differences in
step-list outputs). OpenAI documents `seed` as best-effort, and GH
Models appears to inherit that caveat.

CI policy:

- `temperature=0` to stay as close to greedy decoding as possible.
- No reliance on `seed` for reproducibility.
- Pass threshold set below 100% to absorb expected single-prompt flake;
  watch *score trend* across PRs rather than single-run identity.

## CI gate behaviour

`.github/workflows/eval.yml` runs the suite against
`openai-api/github/openai/gpt-4.1-mini` and uploads the Inspect log as a
workflow artefact. The full 30-prompt suite is the default; the
`workflow_dispatch` trigger accepts an optional `tier` filter that
delegates to `astrodynamics_mcp_eval_subset` for tier-scoped runs.

**Trigger:** the workflow is `workflow_dispatch`-only — fire it
manually from the Actions tab. Automatic triggers (push to main, cron)
can be re-enabled later once the gate's run-to-run stability is
established.

**Where the score lives.** After a run completes:

- The full markdown report appears on the run's **Summary** page — the
  panel at the top of the Actions run view. `eval/_ci_report.py` writes
  it to `GITHUB_STEP_SUMMARY` directly.
- The raw Inspect AI log uploads as the `inspect-eval-logs` workflow
  artefact (14-day retention). Download it and run
  `uv run inspect view logs/` locally to replay the conversation
  per sample — useful when investigating a specific failure.

Authentication uses a personal-owned `MODELS_PAT` repo secret as the
primary auth path on Free-plan orgs (see "Provisioning `MODELS_PAT`"
below), falling back to the workflow-issued `GITHUB_TOKEN` (with
`permissions: models: read`) only if `MODELS_PAT` is not set. The
workflow token's compatibility with the inference endpoint is unreliable
on Free orgs — observed both `403` and `429` against `models.github.ai`
without obvious cause — so the PAT path is the recommended default. A
pre-flight probe step calls the selected model with a 2-token PING and
fails fast with the HTTP status if neither token works, surfacing auth
issues clearly rather than burying them in the Inspect AI stack trace.

### Provisioning `MODELS_PAT`

Create a fine-grained PAT at
`https://github.com/settings/personal-access-tokens` with:

- **Resource owner:** your personal account, *not* the org. Org-owned
  PATs return `403 no_access` from the inference endpoint regardless of
  the permissions granted.
- **Repository access:** "Public Repositories (read-only)" — the
  inference endpoint ignores the repo list, but GitHub's UI requires
  a non-empty choice.
- **Account permissions** → **Models: Read.**

Store via `gh secret set MODELS_PAT --repo astro-tools/astrodynamics-mcp`.

### Fallback: workflow `GITHUB_TOKEN`

The workflow `GITHUB_TOKEN` with `permissions: models: read` is the
documented inference auth path in GitHub's quickstart, but on Free orgs
it does not reliably authenticate. If you want to try it, leave
`MODELS_PAT` unset; the probe step will surface the actual HTTP status.
The toggle that's supposed to enable this auth path lives at
`https://github.com/organizations/<org>/settings/models` when present.

**Pass threshold:** ≥80% of goldens. Calibrated against the determinism
observations above so a single-prompt flake doesn't flip the gate. The
threshold is revisited once a multi-run history exists.

The report renderer (`eval/_ci_report.py`) shows the overall accuracy,
lists the first 15 failing prompts with their short failure mode
(trace fail / functional fail / both) plus the first reason from each
side, and links to the workflow artefact for the full Inspect log. The
script's exit code drives the workflow's pass/fail step: 0 = above
threshold, 1 = below, 2 = no log produced (the eval crashed).

## Running against other providers

The suite is model-agnostic — Inspect AI separates the `Task` from the
model, so the same prompts, scorer, and MCP-server wiring work against
any Inspect-AI-supported provider. The
[Inspect AI providers page](https://inspect.aisi.org.uk/providers.html)
is the authoritative list; the entries below are the ones most relevant
to this project.

For each provider, set the listed env var, then pass the `--model`
string to `inspect eval`. The package's `astrodynamics-mcp stdio` binary
is spawned as a subprocess regardless of which model is in use.

### Anthropic (Claude direct)

```bash
ANTHROPIC_API_KEY="sk-ant-..." \
  uv run inspect eval eval/tasks.py \
  --model anthropic/claude-sonnet-4-6 \
  --temperature 0
```

Other Anthropic models: `anthropic/claude-opus-4-7`,
`anthropic/claude-haiku-4-5`, etc. — match the API model id.

### OpenAI (direct, not via GH Models)

```bash
OPENAI_API_KEY="sk-..." \
  uv run inspect eval eval/tasks.py \
  --model openai/gpt-4o \
  --temperature 0
```

### Google AI Studio

```bash
GOOGLE_API_KEY="..." \
  uv run inspect eval eval/tasks.py \
  --model google/gemini-2.5-pro \
  --temperature 0
```

### AWS Bedrock

```bash
AWS_PROFILE=... AWS_REGION=us-west-2 \
  uv run inspect eval eval/tasks.py \
  --model bedrock/anthropic.claude-sonnet-4-6-v1:0 \
  --temperature 0
```

### Local model (Ollama)

Useful for iteration without burning API budget — outputs are weaker
than the frontier models, but the scorer still exercises the MCP wiring
and the trace/functional checks.

```bash
uv run inspect eval eval/tasks.py \
  --model ollama/llama3.3:70b \
  --temperature 0
```

### Pinning to a subset of prompts

Useful when iterating on a single prompt's `permitted_traces` or
`functional_answer`:

```bash
# Just the tle_lookup primary prompt:
uv run inspect eval eval/tasks.py \
  --model anthropic/claude-sonnet-4-6 \
  --sample-id tle_lookup_iss_by_norad_id

# Just the planning tier:
uv run inspect eval eval/tasks.py@astrodynamics_mcp_eval_subset \
  -T tier=planning \
  --model anthropic/claude-sonnet-4-6

# First N samples only:
uv run inspect eval eval/tasks.py \
  --model anthropic/claude-sonnet-4-6 \
  --limit 5
```

### Comparing models

Provide multiple `--model` flags and Inspect AI runs the suite against
each, emitting separate logs:

```bash
uv run inspect eval eval/tasks.py \
  --model anthropic/claude-sonnet-4-6 \
  --model openai/gpt-4o \
  --model openai-api/github/openai/gpt-4o \
  --temperature 0
```

Outputs land under `logs/` (gitignored). `inspect view logs/` opens a
browser-based result viewer.
