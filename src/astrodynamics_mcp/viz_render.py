"""Deterministic matplotlib-to-PNG rendering for the static-plot tools.

The transport-equivalence contract requires a tool call to return a
byte-identical payload over stdio and Streamable HTTP. For an image-bearing
tool that means the same figure must render to the same PNG bytes every time —
otherwise two transports running the same call in two processes would disagree.

matplotlib is not deterministic out of the box: the default PNG writer stamps
the matplotlib version into a ``Software`` text chunk, and an interactive
backend can pull in environment-dependent state. :func:`render_png` pins both
down — the headless **Agg** backend, a fixed DPI, and a metadata dict that
suppresses the version/timestamp chunks — so repeated renders of the same
figure are byte-for-byte identical within a matplotlib version.

matplotlib ships only behind the ``[viz]`` extra, so it is imported lazily
inside the function: importing this module on a base install is free and never
pulls matplotlib in.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Fixed render resolution. Pinning the DPI keeps the rasterised pixel grid
# stable across calls; the static-plot tools render at this one resolution.
_RENDER_DPI = 100

# PNG text chunks matplotlib would otherwise populate with non-reproducible
# values: ``Software`` carries the matplotlib version string, and ``Creation
# Time`` (when enabled) a wall-clock timestamp. Mapping each to None tells the
# Agg PNG writer to omit the chunk entirely, so the output depends only on the
# figure content.
_REPRODUCIBLE_PNG_METADATA: dict[str, str | None] = {
    "Software": None,
    "Creation Time": None,
}


def render_png(figure: Figure) -> bytes:
    """Render *figure* to deterministic PNG bytes.

    Uses a fixed DPI and strips the non-reproducible PNG metadata chunks, so
    two calls with an equivalent figure return identical bytes. The caller owns
    the figure's lifecycle (typically closing it after this returns); this
    helper only reads it.
    """
    buffer = io.BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=_RENDER_DPI,
        metadata=_REPRODUCIBLE_PNG_METADATA,
    )
    return buffer.getvalue()


def use_agg_backend() -> None:
    """Select matplotlib's headless **Agg** backend for reproducible rendering.

    Call this once before constructing figures in a tool body. Agg is the
    non-interactive raster backend — it needs no display, behaves identically
    on a server and a workstation, and is what the deterministic PNG output
    relies on. Imported lazily so a base install never loads matplotlib.
    """
    import matplotlib

    matplotlib.use("Agg")
