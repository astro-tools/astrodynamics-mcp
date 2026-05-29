# Credentials

A small number of credentialled upstream sources back the tool
surface: ESA DISCOSweb and 18 SDS / USSPACECOM's Space-Track. Both
require per-user accounts; neither permits redistribution of derived
products that would let astrodynamics-mcp ship one shared account on
your behalf. The server therefore *passes through* whatever credential
you give it — once, at server start (stdio) or per session
(Streamable HTTP) — and reaches the upstream as you.

A tool body that needs a credential calls
[`require_credential`](api.md#credentials):

- credential is present → returns a dict of the fields, ready to
  forward to the upstream.
- credential is absent or incomplete → raises a typed
  [`CredentialRequiredError`](api.md#errors) whose `code` is
  `credential_required.<source>`. The LLM consumer sees a stable
  error envelope, not a leaked exception.

The same call works under either transport. The only thing that varies
is where the credential came from.

## Sources

| Source | Required fields | Stable error code |
|--------|-----------------|-------------------|
| `spacetrack` | `username`, `password` | `credential_required.spacetrack` |
| `discosweb` | `token` | `credential_required.discosweb` |

## stdio — environment variables

The convention is `ASTRODYNAMICS_MCP_<SOURCE>_<FIELD>` upper-cased. Set
them in your shell, your `.env`, or your MCP client's server config
before `astrodynamics-mcp stdio` launches.

| Source | Env vars |
|--------|----------|
| Space-Track | `ASTRODYNAMICS_MCP_SPACETRACK_USERNAME`, `ASTRODYNAMICS_MCP_SPACETRACK_PASSWORD` |
| DISCOSweb | `ASTRODYNAMICS_MCP_DISCOSWEB_TOKEN` |

Example MCP client config block (Claude Code):

```json
{
  "mcpServers": {
    "astrodynamics-mcp": {
      "command": "astrodynamics-mcp",
      "args": ["stdio"],
      "env": {
        "ASTRODYNAMICS_MCP_SPACETRACK_USERNAME": "alice@example.org",
        "ASTRODYNAMICS_MCP_SPACETRACK_PASSWORD": "..."
      }
    }
  }
}
```

A credential is "present" only if **every** required field for the
source has a non-empty value. Setting `ASTRODYNAMICS_MCP_SPACETRACK_USERNAME`
without `_PASSWORD` is treated as if neither were set.

## Streamable HTTP — `_meta` on `initialize`

For remote agents and custom clients over the HTTP transport, the
credential travels in the MCP `initialize` request's `_meta` block, under
the namespaced key `astrodynamics_mcp/credentials`. Per-session: the
server stashes whatever the client sent for the lifetime of that
session and forgets it when the session ends.

```jsonc
{
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-11-25",
    "capabilities": { },
    "clientInfo": { "name": "my-agent", "version": "1.0" },
    "_meta": {
      "astrodynamics_mcp/credentials": {
        "spacetrack": {
          "username": "alice@example.org",
          "password": "..."
        },
        "discosweb": {
          "token": "..."
        }
      }
    }
  }
}
```

If the client also has env vars set in the server's process environment,
**session metadata wins** — HTTP overrides stdio in mixed scenarios.

## Security expectations

- Credentials live in your process or your client's config; the server
  never writes them to disk, never logs them, never echoes them into a
  tool response, and never forwards them to the LLM. The LLM sees the
  *result* of the upstream call, not the credential that authorised it.
- The error envelope on a missing credential carries the source name
  and the list of fields that were not satisfied, so the LLM can ask
  the user to set the right variable. It does not carry any portion of
  the value that *was* set.
- The on-disk cache layer described in
  [Data sources](data-sources.md#on-disk-cache) caches *responses*
  keyed by request shape; the credential never appears in any cache key
  or stored payload.
- The server reads credentials at startup (stdio) or per session
  (HTTP) and never writes them back. There is no refresh, no rotation,
  and no persistence beyond the lifetime of the server process or HTTP
  session.

If a credential is mis-configured — wrong password, expired token —
the upstream's authentication failure surfaces as a
[`DataSourceError`](api.md#errors) from the data adapter layer, not as
a `CredentialRequiredError`. The two error codes distinguish
"you didn't give me a credential" from "the credential you gave me
isn't valid."
