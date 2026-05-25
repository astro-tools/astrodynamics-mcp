"""IERS Bulletin A shim over astropy's auto-fetch.

Tools that need Earth-orientation data (time-scale conversions involving
UT1, frame transforms touching ITRS/TIRS) go through this shim rather
than importing :mod:`astropy.utils.iers` directly — so the policy of "use
astropy's IERS_Auto, never re-poll Bulletin A ourselves" lives in one
place.

astropy and skyfield both populate the same on-disk Bulletin A cache via
astropy's machinery; we deliberately do not own a parallel cache for it.
The :class:`Cache` layer is for *our* upstream responses (CelesTrak,
Horizons), not for re-implementing what astropy already does correctly.
"""

from __future__ import annotations

from astropy.time import Time
from astropy.utils import iers
from pydantic import BaseModel, ConfigDict


class IersStatus(BaseModel):
    """Freshness snapshot of the loaded IERS Bulletin A table.

    Tools surface this so the LLM (and downstream user) can see how recent
    the Earth-orientation data backing a time / frame conversion was.
    """

    model_config = ConfigDict(extra="forbid")

    last_updated: str
    """Latest table epoch as a UTC ISO 8601 string. May include predicted
    (not yet observed) rows — astropy's IERS_Auto extends Bulletin A with
    short-range predictions so EOP corrections stay available across the
    near-future window most tools query."""

    rows: int
    """Total EOP rows currently loaded (observed + predicted)."""

    cache_path: str | None
    """astropy's on-disk cache file backing this load, if known."""


def load_iers() -> IersStatus:
    """Load (and auto-fetch if needed) the IERS Bulletin A table; return status.

    Idempotent: astropy caches the table in-process, so repeat calls are
    cheap. The first call after a fresh interpreter may take a few seconds
    while astropy contacts its CDN.

    The shim never raises on a stale or partially-loaded table — astropy's
    own fallbacks (use the bundled older Bulletin A; allow extrapolation)
    keep the surface usable. Tools that need a strict-freshness guarantee
    should inspect :attr:`IersStatus.last_updated` themselves.
    """
    bulletin = iers.IERS_Auto.open()

    # ``bulletin['MJD']`` is an astropy Quantity column in days; the last
    # entry is the most recent epoch (which may be a Bulletin A prediction
    # extending past the latest IERS observation). Convert to ISO 8601 UTC.
    last_mjd_quantity = bulletin["MJD"][-1]
    last_mjd = float(last_mjd_quantity.value)
    last_updated_time = Time(last_mjd, format="mjd", scale="utc")
    last_updated = f"{last_updated_time.isot}Z"
    rows = len(bulletin)

    # astropy's IERS_Auto carries the URL it loaded from in `meta['data_url']`
    # when present; surface it best-effort so an operator debugging EOP
    # staleness has a concrete file (or URL) to inspect.
    cache_path: str | None = None
    meta = getattr(bulletin, "meta", None)
    if isinstance(meta, dict):
        raw_path = meta.get("data_url") or meta.get("url")
        if isinstance(raw_path, str):
            cache_path = raw_path

    return IersStatus(
        last_updated=last_updated,
        rows=rows,
        cache_path=cache_path,
    )
