# Contributing to astrodynamics-mcp

Thanks for your interest. This page is the one place to learn the workflow.

## Getting set up

```bash
git clone https://github.com/astro-tools/astrodynamics-mcp.git
cd astrodynamics-mcp
uv sync --all-groups
```

No external services are needed for the v0.1 surface — every tool either runs
offline or calls a no-auth data source (CelesTrak, JPL Horizons, IERS) cached
on first use under `~/.cache/astrodynamics-mcp/`.

## Branches and PRs

- One issue per branch. Branch names use a short prefix for type:
  - `feat/<slug>` — new capability, tied to a `type:feature` issue.
  - `fix/<slug>` — bug fix, tied to a `type:bug` issue.
  - `chore/<slug>` — infra / tooling / hygiene.
  - `docs/<slug>` — docs-only change.
- Open a PR against `main`. Put `Closes #<N>` in the PR description so the issue
  auto-closes on merge and the project board advances the card to Done.
- Squash-merge is the only merge method. The PR title becomes the squash commit
  subject — write it as a complete imperative sentence.

## Local checks before pushing

```bash
uv run pytest               # unit tests (integration tests are gated behind a marker)
uv run ruff check           # lint
uv run ruff format --check  # formatting
uv run mypy                 # types
```

CI re-runs all four on Ubuntu and Windows × Python 3.10 / 3.11 / 3.12. macOS
is added in v0.2.

### Coverage thresholds

CI enforces coverage gates on the Ubuntu / Python 3.12 cell:

- Overall coverage must be ≥ 80%.
- Each of `src/astrodynamics_mcp/units.py`, `src/astrodynamics_mcp/schemas/`,
  and `src/astrodynamics_mcp/data/` must be ≥ 95%.

To reproduce locally:

```bash
uv run pytest -m "integration or not integration" --cov
uv run coverage report --fail-under=80
uv run coverage report --include='src/astrodynamics_mcp/units.py' --fail-under=95
uv run coverage report --include='src/astrodynamics_mcp/schemas/*' --fail-under=95
uv run coverage report --include='src/astrodynamics_mcp/data/*' --fail-under=95
```

## Commit messages

Keep them short and imperative. One subject line, optional body.

- "Add typed error hierarchy and units discipline"
- "Fix CelesTrak stale-fallback when cache is empty"

Do not include AI or tool attribution trailers in commits, PR titles, PR descriptions,
or comments — see the repo-level convention.

## Scope discipline

astrodynamics-mcp is an MCP server: it wraps vetted upstream astrodynamics
libraries so LLM clients (Claude Code, Cursor, ChatGPT desktop, custom agents)
can call them as tools instead of hallucinating numerical answers. The project
does not ship propagators, integrators, or coordinate systems of its own.
Before opening a feature issue, check the existing issues and milestones to
make sure the work belongs here.

- **SGP4 / TLE propagation →** [`sgp4`](https://github.com/brandon-rhodes/python-sgp4).
- **Lambert's problem →** [`lamberthub`](https://github.com/jorgepiloto/lamberthub).
- **Ground-station / observer geometry →** [`skyfield`](https://rhodesmill.org/skyfield/).
- **Time scales and coordinate frames →** [`astropy`](https://www.astropy.org/).
- **Porkchop and interplanetary mission design →** [`interplanetary-porkchop`](https://github.com/mlewicki/interplanetary-porkchop).
- **Running a GMAT mission →** v0.2 `[gmat]` extra wraps [`gmat-run`](https://github.com/astro-tools/gmat-run) and [`gmat-sweep`](https://github.com/astro-tools/gmat-sweep).
- **SPICE kernels and ephemerides →** v0.3 `[spice]` extra wraps [`spiceypy`](https://github.com/AndrewAnnex/SpiceyPy).
- **Agent orchestration (LangGraph, AutoGen, CrewAI) →** out of scope; we expose
  tools, your agent consumes them.

## Questions

Open a [discussion](https://github.com/orgs/astro-tools/discussions) rather
than an issue for open-ended questions, usage help, or brainstorming. The
astro-tools org runs a single shared discussions space — there is no
per-repo discussions board.
