"""JPL Horizons ephemeris-fetch adapter.

Wraps the Horizons HTTP API (``ssd.jpl.nasa.gov/api/horizons.api``) with the
on-disk cache in front. Horizons responses are large text blobs — parsing
into state vectors belongs to the consuming tool (porkchop / B-plane); the
adapter just memoises the response so the second porkchop query against the
same (target, center, window, step) is local.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from astrodynamics_mcp.cache import DEFAULT_TTLS, Cache, default_cache
from astrodynamics_mcp.errors import DataSourceError, UpstreamError

_HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
_SOURCE = "horizons"
_HTTP_TIMEOUT = 60.0  # Horizons is slower than CelesTrak; queries take 5-30s.


class HorizonsResponse(BaseModel):
    """The wire-format response from :func:`fetch_ephemeris`."""

    model_config = ConfigDict(extra="forbid")

    signature: dict[str, Any]
    """Request fingerprint (target, center, start, stop, step) — lets the
    consumer verify the cached payload matches what they asked for."""

    result: str
    """The raw text block Horizons returns under its ``"result"`` JSON key.
    Parsing into state vectors is the consuming tool's responsibility."""

    fetched_at: datetime
    stale: bool = False


def _signature(target: str, center: str, start: str, stop: str, step: str) -> dict[str, str]:
    return {
        "target": target,
        "center": center,
        "start": start,
        "stop": stop,
        "step": step,
    }


def _cache_key(signature: dict[str, str]) -> str:
    """SHA256 of the canonical signature; opaque but stable across processes."""
    canonical = "|".join(f"{k}={signature[k]}" for k in sorted(signature))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _query_params(target: str, center: str, start: str, stop: str, step: str) -> dict[str, str]:
    """Horizons API parameter dict for a state-vector ephemeris query."""
    return {
        "format": "json",
        "COMMAND": f"'{target}'",
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "VECTORS",
        "CENTER": f"'{center}'",
        "START_TIME": f"'{start}'",
        "STOP_TIME": f"'{stop}'",
        "STEP_SIZE": f"'{step}'",
        "OUT_UNITS": "KM-S",
        "REF_PLANE": "ECLIPTIC",
        "REF_SYSTEM": "ICRF",
        "VEC_TABLE": "2",
    }


def _build_response(
    signature: dict[str, str],
    raw: dict[str, Any],
    fetched_at: datetime,
    *,
    stale: bool,
) -> HorizonsResponse:
    return HorizonsResponse(
        signature=signature,
        result=raw["result"],
        fetched_at=fetched_at,
        stale=stale,
    )


async def fetch_ephemeris(
    target: str,
    center: str,
    start: str,
    stop: str,
    step: str,
    *,
    client: httpx.AsyncClient | None = None,
    cache: Cache | None = None,
) -> HorizonsResponse:
    """Fetch a state-vector ephemeris from JPL Horizons.

    ``target`` and ``center`` follow Horizons' naming (e.g. ``"499"`` for
    Mars, ``"@sun"`` for the heliocentre); ``start`` and ``stop`` are
    Horizons-compatible date strings (ISO 8601 works); ``step`` is a
    Horizons step-size string (``"1d"``, ``"6h"``).

    Cache → network → stale fallback. The cache key is sha256 of the
    canonical request signature so two requests with the same parameters
    in different orders hit the same entry.
    """
    if cache is None:
        cache = default_cache()
    sig = _signature(target, center, start, stop, step)
    key = _cache_key(sig)

    hit = cache.get(_SOURCE, key, ttl_s=DEFAULT_TTLS[_SOURCE])
    if hit is not None:
        return _build_response(sig, hit.value, hit.fetched_at, stale=False)

    try:
        if client is None:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as owned_client:
                response = await owned_client.get(
                    _HORIZONS_URL,
                    params=_query_params(target, center, start, stop, step),
                )
                response.raise_for_status()
                payload = response.json()
        else:
            response = await client.get(
                _HORIZONS_URL,
                params=_query_params(target, center, start, stop, step),
            )
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        stale_hit = cache.get_stale(_SOURCE, key)
        if stale_hit is not None:
            return _build_response(sig, stale_hit.value, stale_hit.fetched_at, stale=True)
        raise DataSourceError(
            f"Horizons unreachable for {target!r}@{center!r}: {exc}",
            code="data_source.horizons_unreachable",
            source=_SOURCE,
        ) from exc

    if not isinstance(payload, dict) or "result" not in payload:
        raise UpstreamError(
            f"Horizons returned unexpected shape (no 'result' key) for {target!r}@{center!r}",
            code="upstream.horizons_unexpected_shape",
        )

    cache.put(_SOURCE, key, payload)
    hit_after = cache.get(_SOURCE, key, ttl_s=DEFAULT_TTLS[_SOURCE])
    fetched_at = hit_after.fetched_at if hit_after is not None else datetime.now().astimezone()
    return _build_response(sig, payload, fetched_at, stale=False)
