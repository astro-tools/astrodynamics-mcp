"""Space-Track TLE-fetch adapter.

Wraps Space-Track's REST query interface (`/basicspacedata/query/class/gp/`)
with cookie-based session authentication. The session cookie is reused
across calls via a module-level :class:`httpx.AsyncClient` singleton so we
do not re-authenticate on every tool invocation — Space-Track's API Rules
of Behaviour explicitly require this kind of stewardship.

Failure modes mirror :mod:`astrodynamics_mcp.data.celestrak` so the LLM
consumer sees the same vocabulary across both ``tle_lookup`` sources:

- Live upstream unreachable + cached value present → return the cached
  value with ``stale=True`` and the original ``fetched_at``.
- Live upstream unreachable + no cached value → raise
  :class:`~astrodynamics_mcp.errors.DataSourceError` with code
  ``data_source.spacetrack_unreachable``.
- HTTP 4xx (other than the 401/403 re-login retry) / 5xx + no cached
  value → same.

Authentication failures (bad credential, expired session that cannot be
refreshed) surface as :class:`DataSourceError` with code
``data_source.spacetrack_auth_failed``. A missing credential is the
caller's concern — :func:`astrodynamics_mcp.tools.tle.tle_lookup` raises
:class:`~astrodynamics_mcp.errors.CredentialRequiredError` before reaching
this module.

Caching is opt-in via the same XDG layer CelesTrak uses, with a 6h TTL.
Space-Track's own guidance is "make a single query for the data and save
it locally; do not query for the same data repeatedly," so caching is
aligned with the upstream's preferences, not at odds with them.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from astrodynamics_mcp import __version__
from astrodynamics_mcp.cache import DEFAULT_TTLS, Cache, default_cache
from astrodynamics_mcp.data.celestrak import OmmRecord, _omm_to_tle_lines
from astrodynamics_mcp.errors import DataSourceError, UpstreamError
from astrodynamics_mcp.schemas.base import TleLines

_logger = logging.getLogger(__name__)

_BASE_URL = "https://www.space-track.org"
_LOGIN_PATH = "/ajaxauth/login"
_QUERY_PATH_TEMPLATE = "/basicspacedata/query/class/gp/{filter}/format/json"
_SESSION_COOKIE_NAME = "chocolatechip"
_SOURCE = "spacetrack"
_HTTP_TIMEOUT = 30.0

_USER_AGENT = f"astrodynamics-mcp/{__version__} (+https://github.com/astro-tools/astrodynamics-mcp)"
_HTTP_HEADERS: dict[str, str] = {"User-Agent": _USER_AGENT}


class SpacetrackResponse(BaseModel):
    """The wire-format response from :func:`fetch_tle`.

    Shape mirrors :class:`astrodynamics_mcp.data.celestrak.CelestrakResponse`
    so the :func:`~astrodynamics_mcp.tools.tle.tle_lookup` body can compose
    a uniform :class:`~astrodynamics_mcp.tools.tle.TleLookupResponse`
    regardless of which adapter served the request.
    """

    model_config = ConfigDict(extra="forbid")

    results: list[OmmRecord]
    tle_lines: list[TleLines]
    fetched_at: datetime
    stale: bool = False


_singleton_client: httpx.AsyncClient | None = None
_singleton_loop: asyncio.AbstractEventLoop | None = None
_login_lock = asyncio.Lock()


def _build_singleton_client() -> httpx.AsyncClient:
    """Construct the module-level :class:`httpx.AsyncClient` used in production.

    Tests inject their own client (typically backed by
    :class:`httpx.MockTransport`) and never touch this path.
    """
    return httpx.AsyncClient(
        base_url=_BASE_URL,
        timeout=_HTTP_TIMEOUT,
        headers=_HTTP_HEADERS,
    )


def _get_singleton_client() -> httpx.AsyncClient:
    """Lazy-build and return the module-level singleton client.

    Called only on the production code path (``client=None`` in
    :func:`fetch_tle`). The singleton is intentionally long-lived so the
    Space-Track session cookie persists across many tool calls.

    An :class:`httpx.AsyncClient` binds to whichever event loop first drives
    it; reusing it from a *different* loop raises ``... is bound to a
    different event loop``. We stash the owning loop alongside the client and
    rebuild on mismatch. We do not :meth:`~httpx.AsyncClient.aclose` the stale
    client here — it belongs to the other (typically already-closed) loop,
    whose teardown has already released its sockets. In a normal long-lived
    ``stdio`` / ``http`` process there is only ever one loop, so this branch
    fires only under test harnesses that spin a fresh loop per test.

    This function is pure-sync (no ``await``), so on a single event loop it
    runs atomically with respect to other coroutines — concurrent first
    callers cannot double-build the singleton.
    """
    global _singleton_client, _singleton_loop
    running = asyncio.get_running_loop()
    if _singleton_client is not None and _singleton_loop is not running:
        _logger.debug("Space-Track client bound to a stale event loop; rebuilding")
        _singleton_client = None
        _singleton_loop = None
    if _singleton_client is None:
        _singleton_client = _build_singleton_client()
        _singleton_loop = running
    return _singleton_client


async def aclose() -> None:
    """Close the module-level singleton client, if one was built.

    Wired into the server's FastMCP lifespan so the long-lived client's
    sockets / SSL contexts are released on shutdown instead of leaking. Must
    run inside the still-open event loop that owns the client (lifespan
    teardown does), since :meth:`~httpx.AsyncClient.aclose` is a coroutine —
    an ``atexit`` hook firing after the loop closes could not await it.
    Idempotent: a no-op when no client was ever built.
    """
    global _singleton_client, _singleton_loop
    if _singleton_client is not None:
        await _singleton_client.aclose()
        _singleton_client = None
        _singleton_loop = None


def _has_session_cookie(client: httpx.AsyncClient) -> bool:
    """True iff *client*'s cookie jar carries a Space-Track session cookie."""
    return _SESSION_COOKIE_NAME in {cookie.name for cookie in client.cookies.jar}


def _clear_session_cookie(client: httpx.AsyncClient) -> None:
    """Drop the Space-Track session cookie so the next call re-logs in."""
    client.cookies.delete(_SESSION_COOKIE_NAME)


async def _login(client: httpx.AsyncClient, credential: dict[str, str]) -> None:
    """POST credentials to ``/ajaxauth/login`` and stash the session cookie.

    Raises :class:`DataSourceError` with code ``data_source.spacetrack_auth_failed``
    on a refused credential (Space-Track returns 200 + ``{"Login":"Failed"}``
    on bad password, not a 401) or any transport error during the login
    request itself.
    """
    form = {"identity": credential["username"], "password": credential["password"]}
    try:
        response = await client.post(_LOGIN_PATH, data=form)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise DataSourceError(
            f"Space-Track login request failed: {exc}",
            code="data_source.spacetrack_unreachable",
            source=_SOURCE,
        ) from exc
    # Space-Track returns 200 with a JSON body on bad credentials rather than
    # a 401, so the only reliable success signal is the session cookie.
    if not _has_session_cookie(client):
        raise DataSourceError(
            "Space-Track login returned no session cookie; credential is likely invalid",
            code="data_source.spacetrack_auth_failed",
            source=_SOURCE,
        )


async def _ensure_logged_in(client: httpx.AsyncClient, credential: dict[str, str]) -> None:
    """Log in if the client does not already carry a session cookie.

    Concurrent callers serialize on :data:`_login_lock`; the double-check
    inside the lock prevents a re-login when another coroutine already
    logged in while we were waiting.
    """
    if _has_session_cookie(client):
        return
    async with _login_lock:
        if _has_session_cookie(client):
            return
        await _login(client, credential)


def _query_filter(query: str) -> str:
    """Build the Space-Track predicate path for a v0.2 query.

    Space-Track has no group/category concept — the CelesTrak group
    keywords (``stations``, ``weather``, …) have no upstream equivalent
    here. We pass them through as a substring name search rather than
    raising; the caller asked for what they asked for and an empty result
    is more honest than a synthesised error.
    """
    if query.isdigit():
        return f"NORAD_CAT_ID/{query}"
    # Percent-encode the name *value* only. The ``OBJECT_NAME/`` separator and
    # the ``~~ ~~`` substring operators are Space-Track predicate syntax and
    # must stay literal, but a raw ``/`` inside a name (e.g. ``COSMOS
    # 2251/DEB``) would otherwise inject extra path segments and corrupt the
    # ``class/gp/.../format/json`` predicate. ``safe=""`` encodes ``/`` to
    # ``%2F``; httpx preserves the encoding in the wire path without
    # double-encoding the ``%``.
    return f"OBJECT_NAME/~~{quote(query, safe='')}~~"


def _cache_key(query: str) -> str:
    """Cache key for a Space-Track query.

    Prefix mirrors :func:`astrodynamics_mcp.data.celestrak._cache_key` so
    an operator grepping the cache directory sees a consistent vocabulary
    across sources. No ``group:`` prefix — Space-Track has no group queries.
    """
    if query.isdigit():
        return f"catnr:{query}"
    return f"name:{query}"


def _build_response(
    raw: list[dict[str, Any]],
    fetched_at: datetime,
    *,
    stale: bool,
) -> SpacetrackResponse:
    """Construct a :class:`SpacetrackResponse` from a raw GP-class payload.

    Validation failures inside individual records surface as
    :class:`UpstreamError` with code ``upstream.spacetrack_invalid_record``
    so a malformed entry has a stable wire-format code. The whole response
    fails on the first bad record — partial success would silently drop
    entries the caller asked for.
    """
    records: list[OmmRecord] = []
    for i, item in enumerate(raw):
        try:
            records.append(OmmRecord.model_validate(item))
        except ValidationError as exc:
            norad_cat_id = item.get("NORAD_CAT_ID") if isinstance(item, dict) else None
            raise UpstreamError(
                f"Space-Track record at index {i} is malformed: {exc}",
                code="upstream.spacetrack_invalid_record",
                original_exception=exc,
                data={"record_index": i, "norad_cat_id": norad_cat_id},
            ) from exc
    return SpacetrackResponse(
        results=records,
        tle_lines=[_omm_to_tle_lines(r) for r in records],
        fetched_at=fetched_at,
        stale=stale,
    )


async def _do_query(client: httpx.AsyncClient, query: str) -> httpx.Response:
    """Issue one GP-class GET, returning the raw response for the caller to handle."""
    return await client.get(_QUERY_PATH_TEMPLATE.format(filter=_query_filter(query)))


async def fetch_tle(
    query: str,
    credential: dict[str, str],
    *,
    client: httpx.AsyncClient | None = None,
    cache: Cache | None = None,
) -> SpacetrackResponse:
    """Fetch TLEs from Space-Track for *query* (NORAD ID or satellite name).

    Cache → network → stale fallback, matching the CelesTrak contract.
    Production callers (``client=None``) reuse the module-level singleton
    client so the Space-Track session cookie survives across tool calls.
    A 401/403 from the query path triggers exactly one re-login + retry to
    recover from a silently-expired session.

    Args:
        query: A NORAD catalogue ID (pure digits) or a name substring.
        credential: A dict with non-empty ``username`` / ``password`` —
            obtained via :func:`astrodynamics_mcp.credentials.require_credential`.
        client: Optional injected :class:`httpx.AsyncClient`. Tests pass a
            :class:`~httpx.MockTransport`-backed client; production callers
            pass ``None`` to use the long-lived singleton.
        cache: Optional :class:`Cache` override. Tests pass a tmp-rooted
            cache; production callers fall through to the module singleton.
    """
    if cache is None:
        cache = default_cache()
    key = _cache_key(query)

    hit = cache.get(_SOURCE, key, ttl_s=DEFAULT_TTLS[_SOURCE])
    if hit is not None:
        return _build_response(hit.value, hit.fetched_at, stale=False)

    active_client = client if client is not None else _get_singleton_client()

    try:
        await _ensure_logged_in(active_client, credential)
        response = await _do_query(active_client, query)
        if response.status_code in (401, 403):
            # Session expired silently — Space-Track sessions die after
            # ~2h of inactivity. Drop the stale cookie, re-login, retry once.
            _clear_session_cookie(active_client)
            await _ensure_logged_in(active_client, credential)
            response = await _do_query(active_client, query)
        response.raise_for_status()
        payload = response.json()
    except DataSourceError:
        # _login already raises with a typed code; do not double-wrap.
        raise
    except (httpx.HTTPError, ValueError) as exc:
        stale_hit = cache.get_stale(_SOURCE, key)
        if stale_hit is not None:
            try:
                return _build_response(stale_hit.value, stale_hit.fetched_at, stale=True)
            except Exception as rebuild_exc:
                # The cached payload no longer rebuilds — schema tightened
                # since it was written, or sgp4's omm.initialize chokes on a
                # field. "Outage beats hard error" only holds if we can serve
                # the stale value; when we can't, fall through to the typed
                # unreachable error rather than raising a confusing
                # parse/validation failure during an outage.
                _logger.warning(
                    "Space-Track stale cache entry for %r is unusable: %s",
                    query,
                    rebuild_exc,
                )
        raise DataSourceError(
            f"Space-Track unreachable for query {query!r}: {exc}",
            code="data_source.spacetrack_unreachable",
            source=_SOURCE,
        ) from exc

    if not isinstance(payload, list):
        # Space-Track normally returns a JSON list on a successful GP query.
        # An object (e.g. an error envelope) means the upstream shape shifted
        # — refuse to cache it.
        raise UpstreamError(
            f"Space-Track returned non-list payload of type {type(payload).__name__} "
            f"for query {query!r}",
            code="upstream.spacetrack_unexpected_shape",
        )

    # Validate every record before caching so a malformed entry surfaces as
    # ``upstream.spacetrack_invalid_record`` without poisoning the cache.
    _build_response(payload, datetime.now().astimezone(), stale=False)

    cache.put(_SOURCE, key, payload)
    hit_after = cache.get(_SOURCE, key, ttl_s=DEFAULT_TTLS[_SOURCE])
    fetched_at = hit_after.fetched_at if hit_after is not None else datetime.now().astimezone()
    return _build_response(payload, fetched_at, stale=False)
