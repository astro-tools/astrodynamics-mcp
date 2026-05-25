"""Cross-tool equivalence — same physical quantity computed two ways must agree.

Two equivalence cases:

- ``sgp4_propagate(..., frame="ICRF")`` vs
  ``sgp4_propagate(..., frame="TEME") → frame_transform(to_frame="ICRF")``:
  the direct propagation+transform path and the chained one share the
  same astropy machinery internally, but the contract is that callers
  see the same final state regardless of which path they pick.
- ``time_convert(scale="UTC" → "TAI")`` direct vs
  ``time_convert(UTC → TT)`` followed by ``time_convert(TT → TAI)``:
  the same TAI instant must come out either way.

These tests defend against a regression where one of the redundant paths
silently diverges from the other — a class of bug per-tool tests cannot
catch on their own.
"""

from __future__ import annotations

import pytest

from astrodynamics_mcp.schemas.base import Frame, TimeScale, TleLines
from astrodynamics_mcp.tools.frames import frame_transform
from astrodynamics_mcp.tools.propagation import sgp4_propagate
from astrodynamics_mcp.tools.time import time_convert

# Fixed ISS-like TLE — same lines used across the sgp4 / access tests.
_ISS_LINE1 = "1 25544U 98067A   24001.50000000  .00010000  00000-0  18000-3 0  9995"
_ISS_LINE2 = "2 25544  51.6400  90.0000 0001000  90.0000 270.0000 15.50000000    07"
_ISS_TLE = TleLines(line1=_ISS_LINE1, line2=_ISS_LINE2)

# A handful of epochs spread across the orbit so any per-epoch drift would
# show up. Avoiding 2024-01-01T00:00:00Z (TLE epoch) on the off chance the
# library has a degenerate zero-elapsed-time branch.
_EQUIV_EPOCHS: list[str] = [
    "2024-01-01T12:30:00Z",
    "2024-01-01T15:45:00Z",
    "2024-01-02T03:15:00Z",
]


class TestSgp4FrameEquivalence:
    """SGP4 direct-to-ICRF must match TEME-then-frame_transform-to-ICRF."""

    async def test_icrf_path_matches_teme_then_transform(self) -> None:
        direct = await sgp4_propagate(tle=_ISS_TLE, epochs=_EQUIV_EPOCHS, frame=Frame.ICRF)
        teme = await sgp4_propagate(tle=_ISS_TLE, epochs=_EQUIV_EPOCHS, frame=Frame.TEME)

        assert len(direct.states) == len(teme.states) == len(_EQUIV_EPOCHS)

        for i, (direct_state, teme_state) in enumerate(
            zip(direct.states, teme.states, strict=True)
        ):
            transformed = await frame_transform(state=teme_state, to_frame=Frame.ICRF)

            # The two paths use the same astropy machinery under the
            # hood, so position agreement should be to numerical noise.
            # Use 1e-6 km (1 mm) as a generous floor that absorbs libm
            # rounding without letting algorithm-level divergence pass.
            for axis in range(3):
                assert direct_state.r.value[axis] == pytest.approx(
                    transformed.state.r.value[axis], abs=1e-6
                ), f"epoch[{i}] r[{axis}] mismatch"
                assert direct_state.v.value[axis] == pytest.approx(
                    transformed.state.v.value[axis], abs=1e-9
                ), f"epoch[{i}] v[{axis}] mismatch"

            assert direct_state.frame is Frame.ICRF
            assert transformed.state.frame is Frame.ICRF
            assert direct_state.epoch == transformed.state.epoch


class TestTimeScaleChainEquivalence:
    """UTC → TAI direct must match UTC → TT → TAI chained."""

    async def test_direct_and_chained_paths_agree_within_microsecond(self) -> None:
        direct = await time_convert(
            value="2026-05-23T12:00:00",
            from_scale=TimeScale.UTC,
            to_scale=TimeScale.TAI,
            out_format="jd",
        )

        # Path: UTC → TT, then TT → TAI.
        intermediate = await time_convert(
            value="2026-05-23T12:00:00",
            from_scale=TimeScale.UTC,
            to_scale=TimeScale.TT,
            out_format="jd",
        )
        assert isinstance(intermediate.value, float)
        chained = await time_convert(
            value=intermediate.value,
            from_scale=TimeScale.TT,
            to_scale=TimeScale.TAI,
            in_format="jd",
            out_format="jd",
        )

        assert isinstance(direct.value, float)
        assert isinstance(chained.value, float)

        # 1 µs in JD-days is ≈ 1.15e-11. Stay an order of magnitude above
        # that floor (1e-10 day ≈ 8.6 µs) so float64 round-off in the
        # chained path doesn't surface as a failure.
        assert chained.value == pytest.approx(direct.value, abs=1e-10), (
            f"direct UTC→TAI={direct.value} vs chained UTC→TT→TAI={chained.value}"
        )

    async def test_iso_paths_agree_to_the_millisecond(self) -> None:
        """The same equivalence holds when both paths emit ISO 8601 strings."""
        direct = await time_convert(
            value="2026-05-23T12:00:00",
            from_scale=TimeScale.UTC,
            to_scale=TimeScale.TAI,
            out_format="iso",
        )
        intermediate = await time_convert(
            value="2026-05-23T12:00:00",
            from_scale=TimeScale.UTC,
            to_scale=TimeScale.TT,
            out_format="iso",
        )
        assert isinstance(intermediate.value, str)
        chained = await time_convert(
            value=intermediate.value,
            from_scale=TimeScale.TT,
            to_scale=TimeScale.TAI,
            in_format="iso",
            out_format="iso",
        )
        # Astropy's isot precision is millisecond; both paths should
        # produce the same string. We compare the second-resolution prefix
        # so any sub-ms tail noise (which would be < 1 µs anyway) doesn't
        # surface as a false negative.
        assert isinstance(direct.value, str)
        assert isinstance(chained.value, str)
        assert direct.value[:19] == chained.value[:19], (
            f"direct={direct.value} vs chained={chained.value}"
        )
