# Data sources

The tool surface reaches several external data sources. Each sits behind
an adapter that caches responses on disk so repeated tool calls don't
hammer the upstream, and each handles network failure by falling back to
a cached value flagged `stale=true` rather than silently returning
nothing.

For credentialled sources the credential passthrough itself is documented
separately — see [Credentials](credentials.md).

## CelesTrak — `tle_lookup`

[CelesTrak](https://celestrak.org) is the canonical no-auth source for
current TLEs. `tle_lookup` calls the `gp.php` endpoint with the
caller-supplied query routed to the right CelesTrak parameter — NORAD
catalogue ID (`CATNR=`), satellite name (`NAME=`), or one of the
recognised group / category keywords (`GROUP=`). The recognised groups
are listed in the [tool's description](tool-reference.md#tle_lookup).

CelesTrak honours `If-Modified-Since` for most catalogue endpoints. The
adapter respects the upstream's soft per-IP cap (~100 MB/day) by caching
parsed OMM JSON for six hours by default. A cached response is served
without contacting the upstream; the `fetched_at` field carries the
timestamp of the original fetch.

When the upstream is unreachable:

- A cached value within or beyond its TTL is returned with `stale=True`
  and the original `fetched_at`. Treat the TLE as best-effort.
- No cached value → the call fails with a
  [`DataSourceError`](api.md#errors) carrying the code
  `data_source.celestrak_unreachable`.

The User-Agent header carries the package version and a URL pointing back
to this project so CelesTrak's analytics can distinguish well-behaved
traffic from anonymous scraping.

## Space-Track — `tle_lookup` (with credentials)

[Space-Track](https://www.space-track.org) is 18 SDS / USSPACECOM's
authoritative catalogue. `tle_lookup(source='space-track')` queries the
`/basicspacedata/query/class/gp/` endpoint over a cookie-authenticated
session. Reach for it when CelesTrak is missing a recently-launched
object, when you need deeper historical GP records than CelesTrak's
working window, or as a fallback when CelesTrak is unreachable.

Authentication is username + password POSTed to `/ajaxauth/login`. The
returned session cookie is reused across tool calls for the lifetime of
the server process so we do not re-authenticate per query — Space-Track's
sessions silently expire after roughly two hours of inactivity, at which
point the adapter detects the 401, transparently re-logs in once, and
retries the original query. A missing credential surfaces *before* any
network call as `credential_required.spacetrack` — see
[Credentials](credentials.md) for how to provide one.

Query dispatch matches the CelesTrak adapter's shape: a numeric query
becomes a `NORAD_CAT_ID` lookup, anything else becomes an `OBJECT_NAME`
substring search. Space-Track has no group/category concept, so the
CelesTrak group keywords (`stations`, `weather`, …) have no meaning here
— passing one falls through to a name search and typically returns
nothing useful.

Responses cache on disk under the same XDG layer as CelesTrak with a
6h TTL. Space-Track's
[API Rules of Behaviour](https://www.space-track.org/documentation#/api)
explicitly require this — "make a single query for the data and save it
locally; do not query for the same data repeatedly." Cookies are never
written to disk; the cache stores only the GP-class response payload,
keyed by query shape.

When the upstream is unreachable:

- A cached value within or beyond its TTL is returned with `stale=True`
  and the original `fetched_at`.
- No cached value → `DataSourceError` with code
  `data_source.spacetrack_unreachable`. A refused credential (Space-Track
  returns 200 + `{"Login":"Failed"}` rather than a 401) surfaces as
  `data_source.spacetrack_auth_failed` so the LLM consumer can
  distinguish "credential rejected" from "network unreachable."

Space-Track enforces per-account rate limits — roughly 30 requests per
minute and 300 per hour as of writing. Abusive use can get an account
suspended; the on-disk cache plus session-cookie reuse are how the
adapter stays within that envelope.

## ESA DISCOSweb — `satellite_metadata`

[DISCOSweb](https://discosweb.esoc.esa.int) is ESA's Database and
Information System Characterising Objects in Space — the catalogue of
persistent satellite metadata that the OMM payload from `tle_lookup`
does not carry. `satellite_metadata(norad_id=…)` queries
`/api/objects?filter=eq(satno,<id>)&include=launch,launch.site,operators,reentry`
and returns a single record with mass, bounding-box dimensions, launch
date and site, operator, mission type (Payload / Rocket Body / Debris /
Unknown), and decay status.

NORAD catalogue ID is the cross-reference key between `satellite_metadata`
and `tle_lookup`: the same value is returned by `tle_lookup` and consumed
here. Names and group keywords are not accepted — look up the NORAD ID
first if you only have a name.

### Getting an ESA Space Debris User Account

DISCOSweb sits behind an [ESA Space Debris User Account](https://discosweb.esoc.esa.int/auth/signup).
The signup form asks for a research / institutional purpose; approval is
manual and typically takes a few business days. Once approved, generate a
personal API token from the *Account → API Access* page in the DISCOSweb
UI. The token is the only credential the adapter needs.

Provide the token to the server in one of two ways, per
[Credentials](credentials.md):

- **Stdio transport:** set `ASTRODYNAMICS_MCP_DISCOSWEB_TOKEN=<token>`
  in the environment of the process that launches `astrodynamics-mcp stdio`.
- **HTTP transport:** include
  `{"astrodynamics_mcp/credentials": {"discosweb": {"token": "<token>"}}}`
  in the `_meta` block of the MCP `initialize` request. Session metadata
  wins over the environment variable when both are present.

A missing token surfaces *before* any network call as
`credential_required.discosweb` so the LLM consumer can prompt the user
without spending a network round-trip.

### Caching and rate limits

Responses cache on disk under the same XDG layer as CelesTrak and
Space-Track with a **24h TTL**. DISCOSweb metadata changes slowly —
decay events at most weekly per object, mass and dimensions essentially
never — so a long TTL is appropriate. The cache also conserves the free
DISCOSweb tier's per-account daily quota (in the low hundreds of
requests as of writing); a busy session that re-asks for the same NORAD
ID many times spends one request per day per object, not per call.

Cookies and tokens are never written to the cache; only the JSON
response payload is stored, keyed by NORAD ID under the `discosweb`
source directory.

### Failure modes

When the upstream is unreachable:

- A cached value within or beyond its TTL is returned with `stale=True`
  and the original `fetched_at`.
- No cached value → `DataSourceError` with code
  `data_source.discosweb_unreachable`.

A refused token (HTTP 401 or 403) surfaces as
`data_source.discosweb_auth_failed` so the LLM consumer can distinguish
"credential rejected" from "network unreachable". Authentication
failures do *not* fall through to a stale cache hit — a refused
credential is a permanent state, not a transient outage.

DISCOSweb returns an empty `data` array (not a 404) when no record
matches the NORAD ID. The tool surfaces this as
`data_source.discosweb_norad_not_found` so very recent launches not yet
in DISCOSweb's catalogue have an actionable error code rather than a
silent empty response.

A malformed envelope or record surfaces as
`upstream.discosweb_unexpected_shape` (top-level shape changed) or
`upstream.discosweb_invalid_record` (primary attributes missing).
Malformed responses are never written to the cache.

## JPL Horizons — `porkchop`, `bplane_target`

[JPL Horizons](https://ssd.jpl.nasa.gov/horizons/) supplies planetary
ephemerides for the porkchop and B-plane tools. The adapter fetches
vector ephemerides for the requested bodies over the requested time
window and caches the response under a request hash.

Default TTL is seven days — planetary ephemerides drift on geological
scales, and even the highest-precision missions tolerate week-old
ephemerides without measurable error for these tools' use cases. Horizons
is rate-limited and tolerates one in-flight request per client; the
adapter does not parallelise calls to the upstream.

Failure modes mirror CelesTrak: a stale cached value is preferred over
total failure; no cache → `data_source.horizons_unreachable`. Horizons
returns HTTP 200 with an in-band error for an invalid body, center, or
time window; the adapter surfaces that as `upstream.horizons_error` and
never caches it.

## IERS — `time_convert`, `frame_transform`

Earth-orientation parameters (UT1-UTC, polar motion) and leap-second
tables come from the [IERS data center](https://datacenter.iers.org/).
The adapter reuses `astropy.coordinates`' own IERS cache rather than
maintaining a second copy — `astropy` already polls IERS Bulletin A on a
schedule that matches its Thursday ~20:00 UTC refresh cycle.

Default adapter TTL is 24 hours so the cached value is re-validated
within one upstream cycle. When Bulletin A is unreachable and the
on-disk copy is older than the adapter's TTL, the tool emits a warning
through the response's `stale` field (or, for `time_convert`, a
data-source warning in the response payload) while still using the most
recent cached values for the conversion. The conversion result is still
correct up to the residual UT1-UTC error introduced by the staleness,
typically below a millisecond.

## On-disk cache

All three adapters write to the same XDG-aware cache directory:

- Linux: `~/.cache/astrodynamics-mcp/`
- macOS: `~/Library/Caches/astrodynamics-mcp/`
- Windows: `%LOCALAPPDATA%\astrodynamics-mcp\Cache\`

The cache is one JSON file per `(source, key)` entry, written via
atomic rename so concurrent `astrodynamics-mcp stdio` / `... http`
processes can share the same directory safely. Two writers racing on the
same key both create their own tempfile; whichever `os.replace` lands
second wins, neither write is torn, and readers never see an
intermediate state.

To disable the cache (for tests or pristine CI cells), set
`ASTRODYNAMICS_MCP_CACHE_DIR=""` (empty string). With that override
`get` always misses, `put` is a no-op, and every tool call goes
straight to the upstream.

To redirect the cache to a custom directory, set
`ASTRODYNAMICS_MCP_CACHE_DIR=/path/to/cache` to any writable directory.

The XDG cache layer and its API are documented under
[`astrodynamics_mcp.cache`](api.md#cache).

## `stale=true` semantics

Tools that return cache-backed payloads carry an explicit `stale`
boolean (typically inside each result element — see
[`tle_lookup`](tool-reference.md#tle_lookup)). The contract is:

| `stale` | Meaning |
| --- | --- |
| `false` | The upstream was reached this call (or within the TTL window). The result reflects the latest upstream state. |
| `true`  | The upstream was unreachable on this call; the result is the most recently cached value. The `fetched_at` field carries the timestamp of the original fetch — usually minutes to hours, occasionally days, before the call. |

LLM clients should treat `stale=true` as a signal to mention the
staleness in the human-facing reply ("using a TLE from N hours ago
because CelesTrak was unreachable just now") rather than quoting the
result as live.

## Unit discipline

Every numeric value on the wire is wrapped in `{value, unit}` (or
`{value: [...], unit}` for vectors). Unit strings come from a closed
registry — `"km"`, `"km/s"`, `"deg"`, `"s"`, `"min"`, `"hours"`, `"days"`,
`"AU"`, `"rad"`, `"m"`, `"m/s"`, `"km^2/s^2"`, `"km^3/s^2"`, `"kg"`,
`"K"`, and `"1"` for dimensionless quantities. Adding a new tool with a
bare `number` field fails the static unit-discipline check; the
allowed-units set is extended deliberately as new physical dimensions
enter the tool surface.

See [`astrodynamics_mcp.units`](api.md#units) for the registry and
helpers, and the
[unit-discipline test source](https://github.com/astro-tools/astrodynamics-mcp/blob/main/tests/test_unit_discipline.py)
for the static check.
