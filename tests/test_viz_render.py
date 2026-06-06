"""Tests for the deterministic matplotlib-to-PNG renderer.

matplotlib ships only with the ``[viz]`` extra, which the standard test
environment does not install, so this module self-skips there. The byte-for-
byte determinism guarantee is also asserted in CI's ``[viz]`` extra-install
smoke job, where matplotlib is present; this test is the local counterpart and
the documented contract.
"""

from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib", reason="[viz] extra not installed")

from matplotlib.figure import Figure  # noqa: E402

from astrodynamics_mcp.viz_render import render_png, use_agg_backend  # noqa: E402

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _build_figure() -> Figure:
    """Construct a fixed, non-trivial figure with no time- or env-dependent state."""
    figure = Figure(figsize=(4.0, 3.0))
    axes = figure.add_subplot(111)
    axes.plot([0.0, 1.0, 2.0, 3.0], [3.0, 1.0, 2.0, 0.0])
    axes.set_title("determinism")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    return figure


class TestRenderPng:
    def test_output_is_a_png(self) -> None:
        use_agg_backend()
        png = render_png(_build_figure())
        assert png.startswith(_PNG_MAGIC)
        assert len(png) > len(_PNG_MAGIC)

    def test_two_identical_figures_render_byte_identically(self) -> None:
        """The transport-equivalence contract: equivalent renders → identical bytes.

        Two figures built from the same instructions must produce byte-for-byte
        identical PNGs, so the same tool call returns the same payload whether it
        ran in the stdio process or the HTTP process.
        """
        use_agg_backend()
        first = render_png(_build_figure())
        second = render_png(_build_figure())
        assert first == second

    def test_repeated_render_of_one_figure_is_stable(self) -> None:
        """Rendering the same figure object twice is also stable."""
        use_agg_backend()
        figure = _build_figure()
        assert render_png(figure) == render_png(figure)
