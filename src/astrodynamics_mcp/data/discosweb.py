"""ESA DISCOSweb satellite-metadata adapter.

Wraps DISCOSweb's REST API (``/api/objects``) with bearer-token
authentication. The token sits in the ``Authorization`` header on every
request; the module-level :class:`httpx.AsyncClient` singleton shares
one connection pool across calls so we are polite to DISCOSweb's
per-account rate limit.

Failure modes mirror :mod:`astrodynamics_mcp.data.spacetrack` so the LLM
consumer sees the same vocabulary across credentialled sources:

- Live upstream unreachable + cached value present → return the cached
  value with ``stale=True`` and the original ``fetched_at``.
- Live upstream unreachable + no cached value → raise
  :class:`~astrodynamics_mcp.errors.DataSourceError` with code
  ``data_source.discosweb_unreachable``.
- HTTP 4xx (other than auth) / 5xx + no cached value → same.
- HTTP 401/403 (bad or expired token) → raise
  :class:`DataSourceError` with code
  ``data_source.discosweb_auth_failed``. Authentication failures never
  fall through to a stale cache hit — a refused credential is a
  permanent state, not a transient outage.

DISCOSweb's filter endpoint returns an empty ``data`` array for unknown
NORAD IDs (not a 404), so "no record" is a successful response and
surfaces here as :class:`DiscoswebResponse` with ``record=None``.

Caching uses the same XDG layer as the other adapters with a 24h TTL.
DISCOSweb metadata changes slowly (decay events at most weekly; mass
and dimensions almost never), so a long TTL is appropriate and helps
stay within the free-tier per-account quota.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from astrodynamics_mcp import __version__
from astrodynamics_mcp.cache import DEFAULT_TTLS, Cache, default_cache
from astrodynamics_mcp.errors import DataSourceError, UpstreamError

_BASE_URL = "https://discosweb.esoc.esa.int"
_OBJECTS_PATH = "/api/objects"
_SOURCE = "discosweb"
_HTTP_TIMEOUT = 30.0

# Inline related resources so one request covers the full record. DISCOSweb's
# JSON-API ``include`` parameter follows dotted paths for nested relations.
_INCLUDE = "launch,launch.site,operators,reentry"

_USER_AGENT = f"astrodynamics-mcp/{__version__} (+https://github.com/astro-tools/astrodynamics-mcp)"
_HTTP_HEADERS: dict[str, str] = {
    "User-Agent": _USER_AGENT,
    "Accept": "application/vnd.api+json",
}


class DiscoswebRecord(BaseModel):
    """The flat, normalised metadata record returned by :func:`fetch_metadata`.

    Composed from a DISCOSweb JSON-API ``Object`` resource plus its
    inlined ``launch``, ``launch.site``, ``operators``, and ``reentry``
    relationships. Tool bodies consume this directly; the upstream's
    JSON-API envelope is not part of the tool wire format.

    Nullable fields reflect what DISCOSweb actually carries — mass and
    bounding-box dimensions are unknown for the long tail of debris
    objects; launch/operator/site fields can be empty for very old
    records that pre-date the related catalogues.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    cospar_id: str | None
    norad_id: int
    mass_kg: float | None
    width_m: float | None
    height_m: float | None
    depth_m: float | None
    object_class: str | None
    launch_date: str | None
    launch_site_name: str | None
    operator_names: list[str]
    decay_date: str | None


class DiscoswebResponse(BaseModel):
    """The wire-format response from :func:`fetch_metadata`.

    ``record`` is :data:`None` when DISCOSweb has no entry for the
    requested NORAD ID (the filter endpoint returns an empty array,
    not a 404). The tool layer turns that into a typed error code;
    the adapter itself just reports the empty result.
    """

    model_config = ConfigDict(extra="forbid")

    record: DiscoswebRecord | None
    fetched_at: datetime
    stale: bool = False


_singleton_client: httpx.AsyncClient | None = None


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
    :func:`fetch_metadata`). The singleton is intentionally long-lived
    so the HTTPS connection pool stays warm across many tool calls.
    """
    global _singleton_client
    if _singleton_client is None:
        _singleton_client = _build_singleton_client()
    return _singleton_client


def _cache_key(norad_id: str) -> str:
    """Cache key for a DISCOSweb lookup, prefixed for grep-ability."""
    return f"norad:{norad_id}"


def _index_included(included: Any) -> dict[tuple[str, str], dict[str, Any]]:
    """Build a ``(type, id) -> resource`` lookup for an ``included`` list.

    JSON-API responses carry related resources in a flat ``included``
    array; relationships reference them by ``{"type": ..., "id": ...}``.
    Indexing once avoids a linear scan per relationship.
    """
    index: dict[tuple[str, str], dict[str, Any]] = {}
    if not isinstance(included, list):
        return index
    for item in included:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        item_id = item.get("id")
        if isinstance(item_type, str) and isinstance(item_id, str):
            index[(item_type, item_id)] = item
    return index


def _resolve_to_one(
    relationships: Any,
    name: str,
    index: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    """Resolve a JSON-API to-one relationship link to its included resource."""
    if not isinstance(relationships, dict):
        return None
    rel = relationships.get(name)
    if not isinstance(rel, dict):
        return None
    data = rel.get("data")
    if not isinstance(data, dict):
        return None
    rtype = data.get("type")
    rid = data.get("id")
    if not isinstance(rtype, str) or not isinstance(rid, str):
        return None
    return index.get((rtype, rid))


def _resolve_to_many(
    relationships: Any,
    name: str,
    index: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve a JSON-API to-many relationship link to its included resources."""
    if not isinstance(relationships, dict):
        return []
    rel = relationships.get(name)
    if not isinstance(rel, dict):
        return []
    data = rel.get("data")
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    for ref in data:
        if not isinstance(ref, dict):
            continue
        rtype = ref.get("type")
        rid = ref.get("id")
        if not isinstance(rtype, str) or not isinstance(rid, str):
            continue
        resource = index.get((rtype, rid))
        if resource is not None:
            out.append(resource)
    return out


def _str_or_none(value: Any) -> str | None:
    """Return *value* iff it is a non-empty string; ``None`` otherwise."""
    if isinstance(value, str) and value:
        return value
    return None


def _float_or_none(value: Any) -> float | None:
    """Coerce a numeric value to ``float``; return ``None`` for missing / non-numeric."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _parse_payload(payload: Any) -> DiscoswebRecord | None:
    """Translate a DISCOSweb JSON-API response into a flat record.

    The query uses ``filter=eq(satno,X)``, so ``data`` is a list with at
    most one entry. An empty list means DISCOSweb has no record for the
    NORAD ID and the caller gets ``None``.

    Malformed envelopes (non-dict root, non-list ``data``, missing primary
    attributes) surface as :class:`UpstreamError` with code
    ``upstream.discosweb_unexpected_shape`` so the cache is never poisoned
    with junk.
    """
    if not isinstance(payload, dict):
        raise UpstreamError(
            f"DISCOSweb returned non-object payload of type {type(payload).__name__}",
            code="upstream.discosweb_unexpected_shape",
        )
    data = payload.get("data")
    if not isinstance(data, list):
        raise UpstreamError(
            f"DISCOSweb 'data' field is not a list (got {type(data).__name__})",
            code="upstream.discosweb_unexpected_shape",
        )
    if not data:
        return None
    primary = data[0]
    if not isinstance(primary, dict):
        raise UpstreamError(
            f"DISCOSweb 'data[0]' is not an object (got {type(primary).__name__})",
            code="upstream.discosweb_unexpected_shape",
        )
    attrs = primary.get("attributes")
    if not isinstance(attrs, dict):
        raise UpstreamError(
            "DISCOSweb 'data[0].attributes' is missing or not an object",
            code="upstream.discosweb_invalid_record",
            data={"resource_id": primary.get("id")},
        )

    index = _index_included(payload.get("included"))
    relationships = primary.get("relationships")

    launch = _resolve_to_one(relationships, "launch", index)
    launch_attrs = launch.get("attributes") if isinstance(launch, dict) else None
    launch_date = (
        _str_or_none(launch_attrs.get("epoch")) if isinstance(launch_attrs, dict) else None
    )

    site = (
        _resolve_to_one(launch.get("relationships"), "site", index)
        if isinstance(launch, dict)
        else None
    )
    site_attrs = site.get("attributes") if isinstance(site, dict) else None
    site_name = _str_or_none(site_attrs.get("name")) if isinstance(site_attrs, dict) else None

    operator_names: list[str] = []
    for operator in _resolve_to_many(relationships, "operators", index):
        op_attrs = operator.get("attributes")
        if isinstance(op_attrs, dict):
            name = _str_or_none(op_attrs.get("name"))
            if name is not None:
                operator_names.append(name)

    reentry = _resolve_to_one(relationships, "reentry", index)
    reentry_attrs = reentry.get("attributes") if isinstance(reentry, dict) else None
    decay_date = (
        _str_or_none(reentry_attrs.get("epoch")) if isinstance(reentry_attrs, dict) else None
    )

    satno = attrs.get("satno")
    if not isinstance(satno, int) or isinstance(satno, bool):
        raise UpstreamError(
            f"DISCOSweb record is missing a numeric 'satno' (got {type(satno).__name__})",
            code="upstream.discosweb_invalid_record",
            data={"resource_id": primary.get("id"), "cosparId": attrs.get("cosparId")},
        )

    try:
        return DiscoswebRecord(
            name=_str_or_none(attrs.get("name")) or "",
            cospar_id=_str_or_none(attrs.get("cosparId")),
            norad_id=satno,
            mass_kg=_float_or_none(attrs.get("mass")),
            width_m=_float_or_none(attrs.get("width")),
            height_m=_float_or_none(attrs.get("height")),
            depth_m=_float_or_none(attrs.get("depth")),
            object_class=_str_or_none(attrs.get("objectClass")),
            launch_date=launch_date,
            launch_site_name=site_name,
            operator_names=operator_names,
            decay_date=decay_date,
        )
    except ValidationError as exc:
        raise UpstreamError(
            f"DISCOSweb record for NORAD {satno} failed validation: {exc}",
            code="upstream.discosweb_invalid_record",
            original_exception=exc,
            data={"norad_id": satno, "cosparId": attrs.get("cosparId")},
        ) from exc


def _build_response(
    payload: Any,
    fetched_at: datetime,
    *,
    stale: bool,
) -> DiscoswebResponse:
    record = _parse_payload(payload)
    return DiscoswebResponse(record=record, fetched_at=fetched_at, stale=stale)


def _auth_headers(credential: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {credential['token']}"}


def _query_params(norad_id: str) -> dict[str, str]:
    return {
        "filter": f"eq(satno,{norad_id})",
        "include": _INCLUDE,
    }


async def fetch_metadata(
    norad_id: str,
    credential: dict[str, str],
    *,
    client: httpx.AsyncClient | None = None,
    cache: Cache | None = None,
) -> DiscoswebResponse:
    """Fetch satellite metadata from DISCOSweb for *norad_id*.

    Cache → network → stale fallback. Production callers (``client=None``)
    reuse the module-level singleton client so the HTTPS connection pool
    survives across tool calls.

    Args:
        norad_id: A NORAD catalogue ID as a string of digits. The tool
            layer is responsible for input validation; the adapter trusts
            the caller and passes the value straight into the upstream
            ``filter=eq(satno,...)`` predicate.
        credential: A dict with a non-empty ``token`` field — obtained
            via :func:`astrodynamics_mcp.credentials.require_credential`.
        client: Optional injected :class:`httpx.AsyncClient`. Tests pass
            a :class:`~httpx.MockTransport`-backed client; production
            callers pass ``None`` to use the long-lived singleton.
        cache: Optional :class:`Cache` override. Tests pass a tmp-rooted
            cache; production callers fall through to the module
            singleton.
    """
    if cache is None:
        cache = default_cache()
    key = _cache_key(norad_id)

    hit = cache.get(_SOURCE, key, ttl_s=DEFAULT_TTLS[_SOURCE])
    if hit is not None:
        return _build_response(hit.value, hit.fetched_at, stale=False)

    active_client = client if client is not None else _get_singleton_client()

    try:
        response = await active_client.get(
            _OBJECTS_PATH,
            params=_query_params(norad_id),
            headers=_auth_headers(credential),
        )
        if response.status_code in (401, 403):
            raise DataSourceError(
                f"DISCOSweb refused the bearer token (HTTP {response.status_code})",
                code="data_source.discosweb_auth_failed",
                source=_SOURCE,
            )
        response.raise_for_status()
        payload = response.json()
    except DataSourceError:
        # Auth failures are permanent; do not fall through to a stale hit.
        raise
    except (httpx.HTTPError, ValueError) as exc:
        stale_hit = cache.get_stale(_SOURCE, key)
        if stale_hit is not None:
            return _build_response(stale_hit.value, stale_hit.fetched_at, stale=True)
        raise DataSourceError(
            f"DISCOSweb unreachable for NORAD {norad_id!r}: {exc}",
            code="data_source.discosweb_unreachable",
            source=_SOURCE,
        ) from exc

    # Validate every record before caching so a malformed envelope surfaces as
    # ``upstream.discosweb_*`` without poisoning the on-disk cache.
    _build_response(payload, datetime.now().astimezone(), stale=False)

    cache.put(_SOURCE, key, payload)
    hit_after = cache.get(_SOURCE, key, ttl_s=DEFAULT_TTLS[_SOURCE])
    fetched_at = hit_after.fetched_at if hit_after is not None else datetime.now().astimezone()
    return _build_response(payload, fetched_at, stale=False)
