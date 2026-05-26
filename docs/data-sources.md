# Data sources

The v0.1 tool surface reaches three external no-auth data sources. Each
sits behind an adapter that caches responses on disk so repeated tool
calls don't hammer the upstream, and each handles network failure by
falling back to a cached value flagged `stale=true` rather than
silently returning nothing.

## CelesTrak — `tle_lookup`

[CelesTrak](https://celestrak.org) is the canonical no-auth source for
current TLEs. `tle_lookup` calls the `gp.php` endpoint with the
caller-supplied query routed to the right CelesTrak parameter — NORAD
catalogue ID (`CATNR=`), satellite name (`NAME=`), or one of the
recognised group / category keywords (`GROUP=`). The recognised groups at
v0.1 are listed in the [tool's description](tool-reference.md#tle_lookup).

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
total failure; no cache → `data_source.horizons_unreachable`.

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
