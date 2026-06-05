# SPICE integration

[SPICE](https://naif.jpl.nasa.gov/naif/) — NASA NAIF's observation-geometry
toolkit — is the de-facto standard for ephemerides, reference frames, time
systems, and body parameters in professional mission analysis. The
astrodynamics-mcp SPICE surface lets an LLM client furnish kernels, query
the state of one body relative to another, rotate vectors between
kernel-defined frames, read body constants, and convert between SPICE time
systems — all backed by [`spiceypy`](https://github.com/AndrewAnnex/SpiceyPy),
the Python binding to CSPICE.

The SPICE tools live behind an optional extra. Install it and they appear in
`tools/list`; skip it and the rest of the tool surface is unaffected.

```bash
uv tool install 'astrodynamics-mcp[spice]'
# or
pipx install 'astrodynamics-mcp[spice]'
```

SPICE is stateful in a way the rest of the tool surface is not: you *furnish*
kernels into a pool, and later queries read whatever that pool holds. CSPICE
keeps that pool in process-global C state and is not thread-safe. Those two
facts — statefulness and thread-unsafety — drive the design decisions below.
This page records those decisions; it is the contract the individual SPICE
tools are built against.

## Concurrency and the kernel pool

CSPICE is not thread-safe, and its kernel pool is a single block of
process-global state. The server, by contrast, is asyncio-based and serves
many tool calls. To reconcile the two, **every SPICE tool call runs on one
dedicated worker thread** — a single-worker executor that the async tool
bodies hand their CSPICE work to. CSPICE is therefore only ever entered from
that one thread; calls are serialised, never concurrent; the kernel pool
persists in-process across calls so a kernel furnished by one call is visible
to the next; and the asyncio event loop never blocks on a CSPICE call.

This is the opposite of the choice the GMAT tools make. Those run each
mission in a fresh, isolated interpreter, because a GMAT run is
self-contained — load a script, run it, read the results, discard the
interpreter. SPICE is the reverse: its whole model is *furnish once, query
many times*, and a fresh process per call would throw away the kernel pool
that statefulness depends on. So SPICE keeps a single long-lived worker and
serialises onto it, rather than isolating per call.

## Kernel-state scope across transports

Because the kernel pool is process-global, it is shared by every client the
server process is talking to. On the stdio transport that is exactly right —
one client, one process, one pool.

The SPICE tools register on both stdio and Streamable HTTP, with no
transport-specific gating, the same as the rest of the tool surface.
Operators own their HTTP deployment's trust boundary — auth proxy, network
controls, who can reach the port. One consequence is specific to SPICE and
worth stating plainly: kernel state is shared across HTTP callers. A kernel
furnished by one caller is visible to — and queryable by — any other caller
reaching the same process, and `spice_list_kernels` reports the whole pool
regardless of who furnished each entry. An HTTP deployment of the SPICE
tools is therefore a single-trust-domain deployment; bind to `127.0.0.1` or
front the server with your own auth unless every caller is already trusted to
share a kernel pool.

## Loading kernels from a URL

`spice_load_kernel` accepts a local filesystem path or a URL. A URL is an
SSRF and path-traversal surface if taken at face value, so URL loads are
constrained:

- The scheme must be `https`.
- The host must be on a NAIF allowlist (`naif.jpl.nasa.gov`). A URL on any
  other host — and any redirect that leaves the allowlist — is refused with a
  typed error rather than fetched.
- The download is routed through the existing on-disk cache layer, keyed by a
  hash of the URL and written with the cache's atomic-rename discipline into
  the XDG cache directory. The kernel is then furnished from that cached local
  path; the URL never names a destination path directly, so there is no
  path-traversal vector and no second fetch on a repeat load.
- Oversized downloads are capped.

A local path is furnished as-is — the caller already has filesystem access, so
no allowlist applies there.

## SPICE time and frame tools vs. the astropy tools

SPICE `et2utc` / `str2et` and `pxform` / `sxform` overlap, on their face, the
existing astropy-backed `time_convert` and `frame_transform`. They ship as
separate `spice_*` tools rather than extending those two, and the dividing
line is one question: **does the operation require furnished kernels?**

- `time_convert` and `frame_transform` stay astropy-only and kernel-free.
  They need no furnished state and work the moment the server starts.
- `spice_time_convert` owns the kernel-defined time systems — ET (TDB seconds
  past J2000, defined by the loaded leap-second kernel) and spacecraft clock
  (SCLK), which is meaningless without a spacecraft-clock kernel.
- `spice_frame_transform` owns the frames that need a furnished FK or PCK —
  in particular the non-Earth body-fixed frames. This is a gap the astropy
  tool already declares: `frame_transform` rejects `IAU_MARS`, `IAU_MOON`,
  and `TIRS` today, because astropy has no rotation for them; the SPICE tool
  is where those become available, once the relevant kernels are loaded.

Keeping the split on the kernel boundary gives each tool a clean contract: the
astropy tools never raise a missing-kernel error, and the SPICE tools
uniformly require their kernels up front. Each tool's description points at
the other so a client reaches for the kernel-free path when it is enough.
