"""`satellite_metadata` tool — persistent satellite metadata from ESA DISCOSweb.

Wraps :func:`astrodynamics_mcp.data.discosweb.fetch_metadata` behind a
pydantic-typed response shape. NORAD ID is the cross-reference key
between this tool and :func:`~astrodynamics_mcp.tools.tle.tle_lookup`:
the OMM payload from CelesTrak / Space-Track does not carry mass,
dimensions, launch site, owner, mission type, or decay status, so a
follow-up call to ``satellite_metadata`` fills in those gaps.

Registered against the module-level
:data:`astrodynamics_mcp.server.mcp` singleton on import.
"""

from __future__ import annotations

from typing import Annotated, Literal

from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from astrodynamics_mcp.credentials import require_credential
from astrodynamics_mcp.data.discosweb import DiscoswebRecord, fetch_metadata
from astrodynamics_mcp.errors import DataSourceError, InvalidInputError
from astrodynamics_mcp.schemas.base import Epoch
from astrodynamics_mcp.server import register_tool
from astrodynamics_mcp.units import Quantity


class Dimensions(BaseModel):
    """Bounding-box dimensions of a satellite, when all three axes are known.

    DISCOSweb tracks width, height, and depth as separate nullable fields.
    The tool only emits a :class:`Dimensions` value when all three are
    populated — a partial set would force the LLM to guess which axis is
    missing, so the schema prefers "no dimensions" over "two of three".
    """

    model_config = ConfigDict(extra="forbid")

    x: Quantity = Field(
        ...,
        description="Width along the satellite's body-x axis. Unit must be a length (m / km).",
        examples=[{"value": 73.0, "unit": "m"}],
    )
    y: Quantity = Field(
        ...,
        description="Height along the satellite's body-y axis. Unit must be a length (m / km).",
        examples=[{"value": 45.0, "unit": "m"}],
    )
    z: Quantity = Field(
        ...,
        description="Depth along the satellite's body-z axis. Unit must be a length (m / km).",
        examples=[{"value": 27.5, "unit": "m"}],
    )


class SatelliteMetadataResponse(BaseModel):
    """Top-level response from :func:`satellite_metadata`.

    Carries identifiers (name, COSPAR ID, NORAD ID), physical properties
    (mass, dimensions — both nullable because DISCOSweb's records vary),
    provenance (launch date and site, operator), classification, decay
    status, and cache-freshness flags.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description="Object name as carried by DISCOSweb, e.g. 'ISS (ZARYA)'.",
        examples=["ISS (ZARYA)", "HST"],
    )
    cospar_id: str | None = Field(
        ...,
        description=(
            "COSPAR / International Designator (e.g. '1998-067A'). Stable "
            "per-object identifier issued at launch. Nullable for objects "
            "that lack one in DISCOSweb's records."
        ),
        examples=["1998-067A", "1990-037B"],
    )
    norad_id: str = Field(
        ...,
        description=(
            "NORAD catalogue ID as a string. The same identifier "
            "tle_lookup returns, allowing cross-reference between the two tools."
        ),
        examples=["25544", "20580"],
    )
    mass_kg: Quantity | None = Field(
        ...,
        description=(
            "Object mass. Nullable: DISCOSweb does not carry a mass for "
            "every object (long tail of debris). When populated the value "
            "is the launch / dry mass as recorded by ESA."
        ),
        examples=[{"value": 420000.0, "unit": "kg"}],
    )
    dimensions_m: Dimensions | None = Field(
        ...,
        description=(
            "Bounding-box dimensions (width, height, depth) in metres. "
            "Nullable: only populated when all three axes are known. "
            "DISCOSweb supplies these for tracked payloads but not for the "
            "long tail of fragmentation debris."
        ),
    )
    launch_date: str | None = Field(
        ...,
        description=(
            "Launch epoch as an ISO 8601 string when known. Nullable for "
            "very old objects whose launch records pre-date DISCOSweb's catalogue."
        ),
        examples=["1998-11-20T06:40:00Z"],
    )
    launch_site: str | None = Field(
        ...,
        description="Name of the launch site (e.g. 'Baikonur Cosmodrome'). Nullable.",
        examples=["Baikonur Cosmodrome", "Kennedy Space Center"],
    )
    owner: str | None = Field(
        ...,
        description=(
            "Operator(s) recorded by DISCOSweb. Multiple operators are "
            "joined with ', '. Nullable when no operator is recorded."
        ),
        examples=["NASA", "ESA, JAXA"],
    )
    mission_type: str | None = Field(
        ...,
        description=(
            "DISCOSweb object classification — typically one of 'Payload', "
            "'Rocket Body', 'Debris', 'Unknown'. Free-form string because "
            "the upstream taxonomy can extend."
        ),
        examples=["Payload", "Rocket Body", "Debris"],
    )
    decay_status: Literal["active", "decayed", "unknown"] = Field(
        ...,
        description=(
            "Whether the object is still in orbit. 'decayed' when DISCOSweb "
            "carries a reentry epoch, 'active' otherwise. 'unknown' is "
            "reserved for future cases where DISCOSweb explicitly signals "
            "an indeterminate state; the current adapter never emits it."
        ),
        examples=["active", "decayed"],
    )
    decay_date: str | None = Field(
        ...,
        description=(
            "Reentry epoch as an ISO 8601 string. Populated iff "
            "decay_status='decayed'; null otherwise."
        ),
        examples=["2001-03-23T05:59:24Z"],
    )
    source: Literal["discosweb"] = Field(
        "discosweb",
        description=(
            "Always 'discosweb' for this tool — declared on the wire for "
            "symmetry with multi-source tools."
        ),
    )
    fetched_at: Epoch = Field(
        ...,
        description=(
            "UTC ISO 8601 timestamp when this record was retrieved from "
            "DISCOSweb (or originally cached, when stale=true)."
        ),
    )
    stale: bool = Field(
        ...,
        description=(
            "True when DISCOSweb was unreachable and the cached value was "
            "returned as a stale fallback. Operators should treat the "
            "record as best-effort in that case."
        ),
    )


_DESCRIPTION = (
    "Look up persistent satellite metadata from the ESA DISCOSweb catalogue "
    "by NORAD catalogue ID. Returns information the OMM payload from "
    "tle_lookup does not carry: mass, bounding-box dimensions, launch date "
    "and site, owner / operator, mission type (Payload / Rocket Body / "
    "Debris / Unknown), and decay status. "
    "Use this to cross-reference results from tle_lookup (NORAD ID is the "
    "shared key between the two tools), to filter by physical properties "
    "(e.g. distinguish a CubeSat from an ISS-class platform), to sanity-check "
    "an OMM (e.g. confirm a launch epoch is consistent with the EPOCH field), "
    "or to verify an object is still in orbit before relying on its TLE. "
    "Requires an ESA Space Debris User Account bearer token; without "
    "credentials the call raises credential_required.discosweb without "
    "contacting the upstream. DISCOSweb has no record for very recent "
    "launches that have not yet been catalogued — those surface as "
    "data_source.discosweb_norad_not_found."
)


def _validate_norad_id(value: str) -> str:
    if not value.isdigit():
        raise InvalidInputError(
            f"norad_id must be a string of digits, got {value!r}",
            code="invalid_input.norad_id_not_digits",
        )
    return value


def _to_dimensions(record: DiscoswebRecord) -> Dimensions | None:
    if record.width_m is None or record.height_m is None or record.depth_m is None:
        return None
    return Dimensions(
        x=Quantity(value=record.width_m, unit="m"),
        y=Quantity(value=record.height_m, unit="m"),
        z=Quantity(value=record.depth_m, unit="m"),
    )


@register_tool(
    name="satellite_metadata",
    description=_DESCRIPTION,
    annotations=ToolAnnotations(title="Satellite Metadata", readOnlyHint=True, openWorldHint=True),
)
async def satellite_metadata(
    norad_id: Annotated[
        str,
        Field(
            description=(
                "NORAD catalogue ID as a string of digits. The same value "
                "tle_lookup returns in its norad_id field; use it to "
                "cross-reference. Example: '25544' for the ISS, '20580' "
                "for Hubble. Names and group keywords are not accepted — "
                "look up the NORAD ID via tle_lookup first if you only "
                "have a name."
            ),
            examples=["25544", "20580"],
        ),
    ],
) -> SatelliteMetadataResponse:
    norad_id = _validate_norad_id(norad_id)
    credential = require_credential("discosweb")
    response = await fetch_metadata(norad_id, credential=credential)
    if response.record is None:
        raise DataSourceError(
            f"DISCOSweb has no record for NORAD ID {norad_id!r}",
            code="data_source.discosweb_norad_not_found",
            source="discosweb",
            data={"norad_id": norad_id},
        )
    record = response.record
    mass = Quantity(value=record.mass_kg, unit="kg") if record.mass_kg is not None else None
    owner = ", ".join(record.operator_names) if record.operator_names else None
    decay_status: Literal["active", "decayed", "unknown"] = (
        "decayed" if record.decay_date is not None else "active"
    )
    return SatelliteMetadataResponse(
        name=record.name,
        cospar_id=record.cospar_id,
        norad_id=str(record.norad_id),
        mass_kg=mass,
        dimensions_m=_to_dimensions(record),
        launch_date=record.launch_date,
        launch_site=record.launch_site_name,
        owner=owner,
        mission_type=record.object_class,
        decay_status=decay_status,
        decay_date=record.decay_date,
        source="discosweb",
        fetched_at=response.fetched_at.isoformat(),
        stale=response.stale,
    )
