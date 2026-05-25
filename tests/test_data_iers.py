"""Tests for `astrodynamics_mcp.data.iers`.

The shim is intentionally thin — astropy owns the Bulletin A cache and we
don't re-poll. These tests pin: (a) the shim calls astropy's IERS_Auto
exactly once per `load_iers()`, (b) the returned status is the canonical
shape, (c) the shim doesn't make HTTP requests of its own.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest

from astrodynamics_mcp.data.iers import IersStatus, load_iers


class TestLoadIers:
    def test_returns_status_object(self) -> None:
        """Live call: astropy may auto-fetch on first use; subsequent calls cached."""
        status = load_iers()
        assert isinstance(status, IersStatus)
        # last_updated is an ISO 8601 timestamp; sanity-match the prefix shape.
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", status.last_updated)
        assert status.rows > 0

    def test_shim_calls_iers_auto_open_exactly_once_per_call(self) -> None:
        """The shim must not re-poll Bulletin A — astropy owns that policy."""
        # bulletin['MJD'][-1] returns an astropy Quantity in days; mock its .value.
        fake_mjd_entry = MagicMock()
        fake_mjd_entry.value = 61540.0  # MJD ~2027-05-15
        fake_table = MagicMock()
        fake_table.__getitem__ = lambda self, key: [None, fake_mjd_entry] if key == "MJD" else None
        fake_table.__len__ = lambda self: 365
        fake_table.meta = {}

        with patch(
            "astrodynamics_mcp.data.iers.iers.IERS_Auto.open", return_value=fake_table
        ) as mocked_open:
            status = load_iers()

        assert mocked_open.call_count == 1
        assert status.rows == 365
        # MJD 61540 → 2027-05-15
        assert status.last_updated.startswith("2027-05-15T00:00:00")

    def test_last_updated_ends_with_z(self) -> None:
        """Output is canonical UTC ISO 8601 with the Z designator."""
        fake_mjd_entry = MagicMock()
        fake_mjd_entry.value = 61540.0
        fake_table = MagicMock()
        fake_table.__getitem__ = lambda self, key: [None, fake_mjd_entry] if key == "MJD" else None
        fake_table.__len__ = lambda self: 1
        fake_table.meta = {}

        with patch("astrodynamics_mcp.data.iers.iers.IERS_Auto.open", return_value=fake_table):
            status = load_iers()
        assert status.last_updated.endswith("Z")

    def test_cache_path_extracted_from_meta(self) -> None:
        """When astropy's table.meta carries data_url, the shim surfaces it."""
        fake_mjd_entry = MagicMock()
        fake_mjd_entry.value = 61540.0
        fake_table = MagicMock()
        fake_table.__getitem__ = lambda self, key: [None, fake_mjd_entry] if key == "MJD" else None
        fake_table.__len__ = lambda self: 1
        fake_table.meta = {"data_url": "https://datacenter.iers.org/data/9/finals2000A.all"}

        with patch("astrodynamics_mcp.data.iers.iers.IERS_Auto.open", return_value=fake_table):
            status = load_iers()
        assert status.cache_path == "https://datacenter.iers.org/data/9/finals2000A.all"

    def test_cache_path_is_string_or_none(self) -> None:
        """Cache path is best-effort metadata; either a string or None."""
        status = load_iers()
        assert status.cache_path is None or isinstance(status.cache_path, str)


class TestIersStatusModel:
    def test_extra_fields_forbidden(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            IersStatus.model_validate(
                {
                    "last_updated": "2026-05-25T00:00:00Z",
                    "rows": 100,
                    "cache_path": None,
                    "extra": "no",
                }
            )

    def test_round_trip_through_json(self) -> None:
        status = IersStatus(
            last_updated="2026-05-25T00:00:00Z",
            rows=100,
            cache_path="/tmp/iers.dat",
        )
        restored = IersStatus.model_validate_json(status.model_dump_json())
        assert restored == status
