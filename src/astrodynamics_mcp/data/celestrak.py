"""CelesTrak TLE-fetch adapter.

Wraps CelesTrak's ``gp.php`` endpoint with the on-disk cache in front. A
successful fetch caches the parsed OMM JSON for the 6h CelesTrak TTL; the
raw TLE ``line1`` / ``line2`` are derived from the OMM via
``sgp4.exporter.export_tle`` so we make exactly one network request per
satellite, not two.

Failure modes match the data-adapter contract described in the issue:

- Live upstream unreachable + cached value present → return the cached value
  with ``stale=True`` and the original ``fetched_at``.
- Live upstream unreachable + no cached value → raise
  :class:`~astrodynamics_mcp.errors.DataSourceError` with code
  ``data_source.celestrak_unreachable``.
- HTTP 4xx / 5xx + no cached value → same.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError
from sgp4 import omm
from sgp4.api import Satrec
from sgp4.exporter import export_tle

from astrodynamics_mcp.cache import DEFAULT_TTLS, Cache, default_cache
from astrodynamics_mcp.errors import DataSourceError, UpstreamError
from astrodynamics_mcp.schemas.base import TleLines

_CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"
_SOURCE = "celestrak"
_HTTP_TIMEOUT = 30.0

# CelesTrak group / category names supported by the v0.1 surface. A query
# matching one of these (case-insensitive) routes to ``gp.php?GROUP=<name>``;
# anything else falls through to NAME lookup. Whitelisting keeps a passing
# query like "weather" from being misread as a satellite name and silently
# returning an empty result set.
_GROUP_NAMES: frozenset[str] = frozenset(
    {
        "active",
        "stations",
        "weather",
        "visual",
        "science",
        "geo",
        "gnss",
        "military",
        "last-30-days",
        "starlink",
        "oneweb",
    }
)


class OmmRecord(BaseModel):
    """A single OMM record returned by CelesTrak.

    Field set splits into two tiers. The orbital-mechanics-critical fields
    (identifiers, six mean elements, three drag/rate terms) are required —
    without them no valid TLE can be reconstructed. The CCSDS-metadata
    fields carry CCSDS-spec defaults so a thin CelesTrak record (occasionally
    seen on decay-tracking entries) still parses and round-trips to a
    syntactically valid TLE rather than failing at the wire boundary.

    ``extra="allow"`` keeps the model forward-compatible with new CelesTrak
    fields without forcing a release on every upstream change.
    """

    model_config = ConfigDict(extra="allow")

    OBJECT_NAME: str
    EPOCH: str
    MEAN_MOTION: float
    ECCENTRICITY: float
    INCLINATION: float
    RA_OF_ASC_NODE: float
    ARG_OF_PERICENTER: float
    MEAN_ANOMALY: float
    NORAD_CAT_ID: int
    BSTAR: float
    MEAN_MOTION_DOT: float
    MEAN_MOTION_DDOT: float

    # CCSDS OMM marks these required, but CelesTrak occasionally omits them
    # for decay-tracking entries. Defaults are CCSDS-spec values where
    # defined (``"U"`` unclassified, ephemeris type 0) and zero/empty
    # elsewhere — TLE export tolerates these so the line stays well-formed.
    OBJECT_ID: str = ""
    EPHEMERIS_TYPE: int = 0
    CLASSIFICATION_TYPE: str = "U"
    ELEMENT_SET_NO: int = 0
    REV_AT_EPOCH: int = 0


class CelestrakResponse(BaseModel):
    """The wire-format response from :func:`fetch_tle`."""

    model_config = ConfigDict(extra="forbid")

    results: list[OmmRecord]
    tle_lines: list[TleLines]
    fetched_at: datetime
    stale: bool = False


def _cache_key(query: str) -> str:
    """Cache key for a CelesTrak query.

    Numeric queries are catalogue IDs, whitelisted strings are group/category
    bulk fetches, everything else is a name search. The key prefix makes the
    cache dir human-readable when an operator grep-s for a satellite they're
    debugging.
    """
    if query.isdigit():
        return f"catnr:{query}"
    if query.lower() in _GROUP_NAMES:
        return f"group:{query.lower()}"
    return f"name:{query}"


def _query_params(query: str) -> dict[str, str]:
    """CelesTrak `gp.php` parameter dict for a v0.1 query.

    CelesTrak's API exposes catalog-id / name / group / international-designator
    as mutually exclusive query keys — you can't combine "find satellite X
    within group Y". The v0.1 adapter signature takes only ``query`` and
    dispatches on shape: pure-digit → CATNR, whitelisted group keyword →
    GROUP, anything else → NAME.
    """
    params: dict[str, str] = {"FORMAT": "json"}
    if query.isdigit():
        params["CATNR"] = query
    elif query.lower() in _GROUP_NAMES:
        params["GROUP"] = query.lower()
    else:
        params["NAME"] = query
    return params


def _omm_to_tle_lines(record: OmmRecord) -> TleLines:
    """Derive raw TLE ``line1`` / ``line2`` from an OMM record.

    Saves a second HTTP request to CelesTrak's TLE-format endpoint: the
    sgp4 library carries an OMM-to-Satrec initialiser plus a TLE exporter,
    so we round-trip JSON → Satrec → text purely in memory.
    """
    satrec = Satrec()
    omm.initialize(satrec, record.model_dump())
    line1, line2 = export_tle(satrec)
    return TleLines(line1=line1, line2=line2)


def _build_response(
    raw: list[dict[str, Any]],
    fetched_at: datetime,
    *,
    stale: bool,
) -> CelestrakResponse:
    """Construct a :class:`CelestrakResponse` from a raw OMM payload.

    Validation failures inside individual records are wrapped in a typed
    :class:`UpstreamError` so a malformed entry surfaces on the MCP wire
    as ``upstream.celestrak_invalid_record`` rather than as a raw pydantic
    traceback. The whole response fails on the first bad record — partial
    success would silently drop entries the caller asked for.
    """
    records: list[OmmRecord] = []
    for i, item in enumerate(raw):
        try:
            records.append(OmmRecord.model_validate(item))
        except ValidationError as exc:
            norad_cat_id = item.get("NORAD_CAT_ID") if isinstance(item, dict) else None
            raise UpstreamError(
                f"CelesTrak record at index {i} is malformed: {exc}",
                code="upstream.celestrak_invalid_record",
                original_exception=exc,
                data={"record_index": i, "norad_cat_id": norad_cat_id},
            ) from exc
    return CelestrakResponse(
        results=records,
        tle_lines=[_omm_to_tle_lines(r) for r in records],
        fetched_at=fetched_at,
        stale=stale,
    )


async def fetch_tle(
    query: str,
    *,
    client: httpx.AsyncClient | None = None,
    cache: Cache | None = None,
) -> CelestrakResponse:
    """Fetch TLEs from CelesTrak for *query* (NORAD ID or satellite name).

    Cache → network → stale fallback. The optional ``client`` / ``cache``
    parameters let tests inject a :class:`~httpx.MockTransport`-backed
    client and a ``tmp_path``-rooted cache; production callers pass
    neither and the module-level singletons are used.
    """
    if cache is None:
        cache = default_cache()
    key = _cache_key(query)

    hit = cache.get(_SOURCE, key, ttl_s=DEFAULT_TTLS[_SOURCE])
    if hit is not None:
        return _build_response(hit.value, hit.fetched_at, stale=False)

    try:
        if client is None:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as owned_client:
                response = await owned_client.get(_CELESTRAK_GP_URL, params=_query_params(query))
                response.raise_for_status()
                payload = response.json()
        else:
            response = await client.get(_CELESTRAK_GP_URL, params=_query_params(query))
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        # Stale-fallback path: an outage downstream beats a hard error if
        # the operator has *any* recent value in the cache.
        stale_hit = cache.get_stale(_SOURCE, key)
        if stale_hit is not None:
            return _build_response(stale_hit.value, stale_hit.fetched_at, stale=True)
        raise DataSourceError(
            f"CelesTrak unreachable for query {query!r}: {exc}",
            code="data_source.celestrak_unreachable",
            source=_SOURCE,
        ) from exc

    if not isinstance(payload, list):
        # CelesTrak returns a JSON array on success; anything else is a sign
        # the endpoint shape changed and the cache shouldn't poison itself.
        raise UpstreamError(
            f"CelesTrak returned non-list payload of type {type(payload).__name__} "
            f"for query {query!r}",
            code="upstream.celestrak_unexpected_shape",
        )

    # Validate every record before caching so a malformed entry surfaces as
    # ``upstream.celestrak_invalid_record`` without poisoning the on-disk
    # cache. The validated response is rebuilt below with the cache's own
    # ``fetched_at`` anchor once the write lands.
    _build_response(payload, datetime.now().astimezone(), stale=False)

    cache.put(_SOURCE, key, payload)
    hit_after = cache.get(_SOURCE, key, ttl_s=DEFAULT_TTLS[_SOURCE])
    # `hit_after is None` would mean the cache write succeeded but the read
    # missed — only possible on a disabled cache. In that case, fall back to
    # `datetime.now(UTC)` so the response still carries a timestamp.
    fetched_at = hit_after.fetched_at if hit_after is not None else datetime.now().astimezone()
    return _build_response(payload, fetched_at, stale=False)
