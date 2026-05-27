"""`tle_lookup` tool — fetch current TLEs from CelesTrak.

Wraps :func:`astrodynamics_mcp.data.celestrak.fetch_tle` with a
pydantic-typed response shape and registers it against the module-level
:data:`astrodynamics_mcp.server.mcp` singleton on import.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from astrodynamics_mcp.data.celestrak import fetch_tle
from astrodynamics_mcp.schemas.base import Epoch
from astrodynamics_mcp.server import register_tool


class TleResult(BaseModel):
    """A single TLE-lookup result.

    Carries the satellite name, NORAD catalogue ID (as a string — NORAD IDs
    are categorical identifiers, not measurements), the raw two-line element
    strings, the parsed OMM JSON, and per-result freshness flags propagated
    from the upstream-data adapter.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description="Satellite name as carried in OMM.OBJECT_NAME, e.g. 'ISS (ZARYA)'.",
        examples=["ISS (ZARYA)", "HST"],
    )
    norad_id: str = Field(
        ...,
        description=(
            "NORAD catalogue ID as a string. Stable per-satellite identifier; "
            "leading zeros preserved. e.g. '25544' for the ISS."
        ),
        examples=["25544", "20580"],
    )
    tle_line1: str = Field(
        ...,
        description="First TLE line, exactly 69 characters (starts with '1 ').",
        examples=["1 25544U 98067A   24001.50000000  .00010000  00000-0  18000-3 0  9990"],
    )
    tle_line2: str = Field(
        ...,
        description="Second TLE line, exactly 69 characters (starts with '2 ').",
        examples=["2 25544  51.6400  90.0000 0001000  90.0000 270.0000 15.50000000000000"],
    )
    omm: dict[str, Any] = Field(
        ...,
        description=(
            "Parsed OMM JSON (CCSDS standard fields like CCSDS_OMM_VERS, EPOCH, "
            "MEAN_MOTION, …) returned by CelesTrak. CelesTrak-metadata fields that "
            "the upstream occasionally omits are backfilled with CCSDS-spec defaults."
        ),
    )
    fetched_at: Epoch = Field(
        ...,
        description=(
            "UTC ISO 8601 timestamp when this TLE was retrieved from the upstream "
            "(or originally cached, when stale=true)."
        ),
    )
    stale: bool = Field(
        ...,
        description=(
            "True when CelesTrak was unreachable and the cached value was returned "
            "as a stale fallback. Operators should treat the TLE as best-effort."
        ),
    )


class TleLookupResponse(BaseModel):
    """Top-level response from :func:`tle_lookup`.

    A list of :class:`TleResult`. Single-satellite queries return a single
    element; group/category queries return one element per satellite in the
    catalogue.
    """

    model_config = ConfigDict(extra="forbid")

    results: list[TleResult] = Field(
        ...,
        description="List of matching TLE results, one per satellite returned by CelesTrak.",
    )


_DESCRIPTION = (
    "Fetch current two-line element sets (TLEs) from CelesTrak by NORAD "
    "catalogue ID, satellite name, or a CelesTrak group/category name. "
    "Returns parsed OMM JSON plus the raw two-line strings. "
    "e.g. tle_lookup('25544') for the ISS, tle_lookup('HUBBLE') for the "
    "Hubble Space Telescope, or tle_lookup('weather') for the multi-satellite "
    "weather category. Names are case-insensitive but CelesTrak prefers the "
    "exact catalog spelling — 'ISS (ZARYA)', not 'iss'. If a name lookup "
    "returns no results, fall back to the NORAD ID. Supported group keywords: "
    "'active', 'stations', 'weather', 'visual', 'science', 'geo', 'gnss', "
    "'military', 'last-30-days', 'starlink', 'oneweb'."
)


@register_tool(
    name="tle_lookup",
    description=_DESCRIPTION,
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
)
async def tle_lookup(
    query: Annotated[
        str,
        Field(
            description=(
                "What to look up: a NORAD catalogue ID like '25544', a satellite "
                "name (case-insensitive, prefers exact catalog spelling like "
                "'ISS (ZARYA)' or 'HUBBLE'), or a CelesTrak group keyword "
                "('active', 'stations', 'weather', 'visual', 'science', 'geo', "
                "'gnss', 'military', 'last-30-days', 'starlink', 'oneweb'). "
                "Name lookups returning zero results should fall back to the NORAD ID."
            ),
        ),
    ],
    source: Annotated[
        Literal["celestrak"],
        Field(
            description=(
                "Upstream catalogue to query. Only 'celestrak' is supported at v0.1; "
                "Space-Track lands in v0.2."
            ),
        ),
    ] = "celestrak",
) -> TleLookupResponse:
    del source  # only "celestrak" is supported at v0.1; Space-Track lands in v0.2.
    response = await fetch_tle(query)
    fetched_at_iso = response.fetched_at.isoformat()
    results = [
        TleResult(
            name=record.OBJECT_NAME,
            norad_id=str(record.NORAD_CAT_ID),
            tle_line1=tle_lines.line1,
            tle_line2=tle_lines.line2,
            omm=record.model_dump(),
            fetched_at=fetched_at_iso,
            stale=response.stale,
        )
        for record, tle_lines in zip(response.results, response.tle_lines, strict=True)
    ]
    return TleLookupResponse(results=results)
