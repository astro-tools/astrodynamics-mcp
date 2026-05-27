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

- Overall coverage must be ≥ 90%.
- Each of `src/astrodynamics_mcp/units.py`, `src/astrodynamics_mcp/schemas/`,
  and `src/astrodynamics_mcp/data/` must be ≥ 95%.

To reproduce locally:

```bash
uv run pytest -m "integration or not integration" --cov
uv run coverage report --fail-under=90
uv run coverage report --include='src/astrodynamics_mcp/units.py' --fail-under=95
uv run coverage report --include='src/astrodynamics_mcp/schemas/*' --fail-under=95
uv run coverage report --include='src/astrodynamics_mcp/data/*' --fail-under=95
```

## MCPB bundles

Two MCPB bundles ship on every `v*` tag, packed and uploaded by sister
jobs in `.github/workflows/release.yml`. Both carry a literal
`{{VERSION}}` placeholder which the workflow `sed`-substitutes with the
tag version (`${GITHUB_REF_NAME#v}`) at pack time; never commit a real
version into the placeholder.

### Canonical bundle — `packaging/mcpb/`

The default, modern bundle. `manifest_version: "0.4"`, `server.type: "uv"`.
Consumed by:

- The GitHub Release page (attached as `.mcpb` asset by the `pack-mcpb` →
  `github-release` jobs).
- The Anthropic Claude Desktop directory (manual submission of the
  release asset).
- Any other MCPB-aware client that supports the `uv` server type.

Local test-build:

```bash
cp -r packaging/mcpb /tmp/mcpb-test
sed -i "s/{{VERSION}}/0.1.2/g" /tmp/mcpb-test/manifest.json /tmp/mcpb-test/pyproject.toml
npx --yes @anthropic-ai/mcpb@2.1.2 validate /tmp/mcpb-test/manifest.json
npx --yes @anthropic-ai/mcpb@2.1.2 pack /tmp/mcpb-test /tmp/astrodynamics-mcp.mcpb
uv run --directory /tmp/mcpb-test server.py stdio   # exercise the install-time launch path
```

### Smithery bundle — `packaging/mcpb-smithery/`

Smithery's CLI does not yet recognise MCPB `server.type: "uv"`. The
Smithery bundle uses `server.type: "python"` with
`mcp_config.command: "uv"` and inline `--with` dependencies — the
pattern Smithery's own MCPB bundling docs document. Consumed only by
the `publish-smithery` job; the GitHub Release asset is the canonical
bundle, not this one.

`packaging/mcpb-smithery/` contains only `manifest.json`; the workflow
copies `server.py` and `.mcpbignore` from `packaging/mcpb/` at pack time
so the launcher stays single-sourced. Local test-build mirrors that:

```bash
mkdir -p /tmp/mcpb-smithery-test
cp packaging/mcpb-smithery/manifest.json /tmp/mcpb-smithery-test/
cp packaging/mcpb/server.py packaging/mcpb/.mcpbignore /tmp/mcpb-smithery-test/
sed -i "s/{{VERSION}}/0.1.2/g" /tmp/mcpb-smithery-test/manifest.json
npx --yes @anthropic-ai/mcpb@2.1.2 validate /tmp/mcpb-smithery-test/manifest.json
npx --yes @anthropic-ai/mcpb@2.1.2 pack /tmp/mcpb-smithery-test /tmp/astrodynamics-mcp-smithery.mcpb
```

There's no local equivalent of `uv run server.py stdio` for the
Smithery bundle — Smithery's hosted runner is what actually executes
the `uv run --with ...` command described in the manifest, and the bundle
ships no `pyproject.toml` to resolve against locally.

### MCP Registry manifest

`packaging/mcp-registry/server.json` is templated the same way and
published by the `publish-mcp-registry` job via the `mcp-publisher` CLI
with GitHub OIDC — no secrets required. The `$schema` field must point
at `https://static.modelcontextprotocol.io/schemas/<date>/server.schema.json`;
the publisher rejects any other URL form (including the raw GitHub URL
the schema file's own `$id` claims).

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
- **Porkchop and interplanetary mission design →** composed in-repo from [`lamberthub`](https://github.com/jorgepiloto/lamberthub) and the JPL Horizons adapter; no single upstream.
- **Running a GMAT mission →** v0.2 `[gmat]` extra wraps [`gmat-run`](https://github.com/astro-tools/gmat-run) and [`gmat-sweep`](https://github.com/astro-tools/gmat-sweep).
- **SPICE kernels and ephemerides →** v0.3 `[spice]` extra wraps [`spiceypy`](https://github.com/AndrewAnnex/SpiceyPy).
- **Agent orchestration (LangGraph, AutoGen, CrewAI) →** out of scope; we expose
  tools, your agent consumes them.

## Questions

Open a [discussion](https://github.com/orgs/astro-tools/discussions) rather
than an issue for open-ended questions, usage help, or brainstorming. The
astro-tools org runs a single shared discussions space — there is no
per-repo discussions board.
