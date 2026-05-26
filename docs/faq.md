# FAQ

## Stdio vs HTTP — which should I use?

**Stdio** is the default for desktop chat clients. The client launches
`astrodynamics-mcp stdio` as a subprocess and speaks MCP over its
stdin/stdout. No port to manage, no auth to think about, runs as the
user, lives only while the client is up. Pick stdio unless you have a
specific reason otherwise.

**Streamable HTTP** is for remote agents — pipelines, hosted services,
long-running automation that can't or shouldn't co-locate with a chat
client. Start `astrodynamics-mcp http --port 8000`; clients reach the
server at `http://<host>:8000/mcp` using the official MCP SDK's
Streamable HTTP transport.

The tool surface is identical across the two. The Streamable HTTP
transport currently ships no built-in auth — bind to `127.0.0.1` (the
default) and put your own proxy in front when exposing it to the
network.

## What astrodynamics-mcp is *not*

- **Not a general-purpose astrodynamics framework.** We wrap vetted
  upstream libraries (sgp4, lamberthub, skyfield, astropy,
  interplanetary-porkchop, spiceypy). We don't ship propagators,
  integrators, or coordinate systems of our own. No new physics.
- **Not an agent framework.** astrodynamics-mcp exposes MCP tools;
  LangGraph, AutoGen, CrewAI, and the LLM clients themselves consume
  them.
- **Not a SaaS.** There is no `astrodynamics.cloud`, no hosted offering,
  no managed multi-tenant deployment. You run the server locally or in
  your own infrastructure.
- **Not a script-authoring tool.** Mission-script linting and authoring
  belong to a separate project. astrodynamics-mcp computes on demand;
  it doesn't write `.script` files.
- **Not a catalogue or database.** We don't ship a Hubble image
  archive, an asteroid tracker, or a conjunction database. We compute
  state vectors, access windows, and trajectories — what's in the
  catalogues we query lives at the upstream.
- **Not a web UI.** Tool consumption is through MCP clients. There is
  no browser frontend, no desktop app, no notebook widget. Use whatever
  MCP-capable client you already use.

## Where do I get the underlying libraries directly?

If your work doesn't need an LLM in the loop, reach for the libraries
that back each tool:

- **SGP4 / TLE propagation →** [`sgp4`](https://github.com/brandon-rhodes/python-sgp4).
- **Lambert's problem →** [`lamberthub`](https://github.com/jorgepiloto/lamberthub).
- **Ground-station / observer geometry →** [`skyfield`](https://rhodesmill.org/skyfield/).
- **Time scales and coordinate frames →** [`astropy`](https://www.astropy.org/).
- **Porkchop scans (interplanetary) →** composed in-repo from `lamberthub`
  plus the JPL Horizons adapter — no single upstream. For a standalone
  generator see [`interplanetary-porkchop`](https://github.com/mlewicki/interplanetary-porkchop).
- **SPICE kernels and ephemerides →** [`spiceypy`](https://github.com/AndrewAnnex/SpiceyPy).

astrodynamics-mcp is the LLM-callable front-end for these libraries.
Their `pip install` is the recommended fallback when you need surface
area we haven't exposed yet.

## Why no LeoLabs / Space-Track / DISCOS?

[LeoLabs](https://leolabs.space/)'s commercial ToS prohibits
redistribution of derived products, so passthrough credentialling for
individual paid subscribers is deliberately not in any current plan.

[Space-Track](https://www.space-track.org/) and
[DISCOSweb](https://discosweb.esoc.esa.int/) require per-user accounts
and bearer tokens. The current surface is intentionally no-auth so the
quick-start is `pip install astrodynamics-mcp` plus a one-block client
config — no credential dance. A future milestone will add credentialled
passthrough for Space-Track, DISCOSweb, and NASA Earthdata via env-var
loading (stdio) and session-init metadata (HTTP).

## How do I trust the numbers?

Four layers of validation:

- The upstream libraries (sgp4, lamberthub, skyfield, astropy, etc.)
  have their own test suites. We don't re-verify their numerical
  correctness; we delegate.
- Reference-output regression tests pin a small set of fixed inputs to
  committed golden JSON. Pinning the upstream version is a deliberate
  action; goldens regenerate at the same time.
- Frame and unit equivalence tests cross-check that the same physical
  quantity computed through two different tool paths agrees to
  numerical tolerance.
- The [eval suite](eval-suite.md) checks that an LLM, under prompt
  variation, picks the right tool and reads the right answer back. This
  is the regression contract on tool-description quality.

Together these guard against the numerical, schema, and tool-description
regression modes independently. See the
[charter](https://github.com/astro-tools/astrodynamics-mcp/blob/main/README.md)'s
validation section for the full reasoning.

## How are upstream-library versions pinned?

Hard runtime dependencies are pinned to a minimum version in
`pyproject.toml`; every release pins a specific minor version of `sgp4`,
`lamberthub`, `skyfield`, `astropy`, and `interplanetary-porkchop`.
Bumping a pin triggers a re-run of the reference-output regression
suite and the eval suite — golden values move only when the wrapping
contract genuinely changed.

## Will there be a hosted version?

No. astrodynamics-mcp ships as a CLI you run locally or in your own
infrastructure. The cache layer is process-local and per-machine; we do
not store user data, we do not aggregate analytics, we do not run a
hosted MCP server.

## How do I report a bug or request a tool?

- Bug → [open an issue](https://github.com/astro-tools/astrodynamics-mcp/issues/new)
  with the failing tool's input arguments and the typed error code from
  the response envelope. The error code is the fastest path to a
  diagnosis.
- Tool request → open an issue under the "tool request" template
  describing the natural-language question you'd like to ask, and what
  upstream library would back the tool.
- Open-ended question / usage help → a
  [Discussion](https://github.com/orgs/astro-tools/discussions) in the
  astro-tools community space.
