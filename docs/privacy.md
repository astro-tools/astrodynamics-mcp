# Privacy

`astrodynamics-mcp` runs entirely on your own machine (or your own
infrastructure). The project itself collects nothing: there is no
telemetry, no analytics, no usage reporting, and no account system. This
page describes the only data that leaves your machine — the requests the
tools make to the external data sources they wrap — so you can decide
what you are comfortable enabling.

## What leaves your machine

Tools send the **query parameters you provide** to the relevant data
source over HTTPS, only when a tool that needs that source is invoked.
Nothing is sent in the background and nothing is sent at install time.

| Data source | When | What is sent | Authentication |
| ----------- | ---- | ------------ | -------------- |
| **CelesTrak** | `tle_lookup` (default) | NORAD ID, satellite name, or group keyword | None |
| **JPL Horizons** | `porkchop`, `bplane_target` | Body names, epochs, time windows | None |
| **IERS** | `time_convert`, `frame_transform` (UT1/EOP) | None beyond the bulletin fetch | None |
| **Space-Track** | `tle_lookup` with `source="space-track"` | NORAD ID or object-name query | Your Space-Track username + password |
| **ESA DISCOSweb** | `satellite_metadata` | NORAD ID | Your DISCOSweb bearer token |

The no-auth sources (CelesTrak, JPL Horizons, IERS) are used by the base
install. The credentialed sources (Space-Track, ESA DISCOSweb) are used
**only if you configure their credentials** — otherwise the relevant
tools return a typed `CredentialRequiredError` and make no network call.

## Credentials

Credentials are read from local environment variables (stdio transport)
or from the session-init `_meta` block (HTTP transport). They are sent
only to their own service, over HTTPS, to authenticate your own requests.
The project does not log them, does not transmit them anywhere else, and
does not persist them beyond the lifetime of the process that holds them.

## GMAT tools

The optional GMAT tools (`gmat_run_mission`, `gmat_sweep`,
`gmat_execute_script`, `gmat_validate_script`, `gmat_read_run_artefact`)
run a local GMAT install in a subprocess. Your script content and its
outputs stay on your machine; the project makes no external call on their
behalf beyond whatever the mission you supply configures itself.

## Caching

Responses from the data sources are cached locally under an
XDG-compliant cache directory (`~/.cache/astrodynamics-mcp/` on Linux, the
platform equivalent elsewhere) with per-source TTLs. The cache is
process-local and per-machine; there is no cloud sync. Setting
`ASTRODYNAMICS_MCP_CACHE_DIR=""` disables on-disk caching.

## Upstream service policies

Requests to the external services are subject to each service's own
terms and privacy practices:

- CelesTrak — <https://celestrak.org/>
- JPL Horizons — <https://ssd-api.jpl.nasa.gov/>
- IERS — <https://www.iers.org/>
- Space-Track — <https://www.space-track.org/documentation#/user_agree>
- ESA DISCOSweb — <https://discosweb.esoc.esa.int/>

## Questions

Open an issue at
<https://github.com/astro-tools/astrodynamics-mcp/issues>.
