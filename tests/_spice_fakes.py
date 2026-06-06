"""In-memory ``spiceypy`` stand-in for the SPICE tests.

The test environment does not install ``spiceypy`` (it ships only with the
``[spice]`` extra), so the tool bodies and the :mod:`astrodynamics_mcp.spice_runtime`
pool primitives — which ``import spiceypy`` lazily — are exercised against this
fake, injected via ``sys.modules`` the way the GMAT tests inject a fake
``gmat_run``. The fake maintains a real in-process pool so furnish / list /
unload behave end-to-end, including meta-kernel fan-out and CSPICE's
silently-additive furnish.

The underscore prefix keeps pytest from collecting this module as a test file.
"""

from __future__ import annotations

from pathlib import PurePath
from typing import Any


class FakeSpiceyError(Exception):
    """Stand-in for ``spiceypy``'s ``SpiceyError`` — what CSPICE raises on failure."""


# File-extension → CSPICE category, used when a test does not pin a furnish plan
# explicitly. Binary kernels get a non-zero handle; text / meta kernels get 0,
# matching CSPICE (text kernels load into the pool, not as DAF/DAS files).
_EXT_TYPE: dict[str, str] = {
    ".bsp": "SPK",
    ".bc": "CK",
    ".bpc": "PCK",
    ".tpc": "PCK",
    ".bes": "EK",
    ".bds": "DSK",
    ".tm": "META",
    ".tls": "TEXT",
    ".tf": "TEXT",
    ".tsc": "TEXT",
    ".ti": "TEXT",
}
_BINARY_TYPES = frozenset({"SPK", "CK", "DSK", "EK"})


def _infer_type(path: str) -> str:
    return _EXT_TYPE.get(PurePath(path).suffix.lower(), "TEXT")


class FakeSpice:
    """A behaviour-compatible subset of the ``spiceypy`` module surface.

    Implements only what the SPICE surface calls: ``furnsh`` / ``unload`` /
    ``ktotal`` / ``kdata`` for kernel management; ``str2et`` / ``spkezr`` for
    state queries; ``pxform`` / ``sxform`` for frame rotations; ``bods2c`` /
    ``bodvcd`` for body constants; ``et2utc`` / ``sce2s`` / ``scs2e`` for time
    conversions; plus the three error-handling setters (``erract`` / ``errdev``
    / ``errprt``). Tests pin behaviour with :meth:`plan_furnish` (e.g. a
    meta-kernel fanning out to several kernels), :meth:`fail_furnsh` (a corrupt
    kernel), :meth:`plan_state`, :meth:`plan_rotation`, :meth:`plan_body_code` /
    :meth:`plan_body_constant`, :meth:`plan_time`, and :meth:`plan_sclk`. To
    mirror CSPICE's kernel dependence, ``str2et`` / ``et2utc`` raise unless a
    leap-second (TEXT) kernel is in the pool, ``spkezr`` raises unless an SPK is,
    and ``sce2s`` / ``scs2e`` raise unless a spacecraft-clock kernel (``.tsc``)
    is.
    """

    SpiceyError = FakeSpiceyError

    def __init__(self) -> None:
        self._pool: list[dict[str, Any]] = []
        self._furnish_plan: dict[str, list[dict[str, Any]]] = {}
        self._furnsh_error: dict[str, BaseException] = {}
        self._state_plan: dict[tuple[str, str], tuple[list[float], float]] = {}
        self._rotation_plan: dict[
            tuple[str, str], tuple[list[list[float]], list[list[float]] | None, str | None]
        ] = {}
        self._body_code_plan: dict[str, int] = {}
        self._body_constant_plan: dict[tuple[int, str], tuple[list[float], str | None]] = {}
        self._str2et_plan: dict[str, float] = {}
        self._et2utc_plan: dict[float, str] = {}
        self._scs2e_plan: dict[tuple[int, str], float] = {}
        self._sce2s_plan: dict[tuple[int, float], str] = {}
        self._next_handle = 1
        self.calls: dict[str, list[Any]] = {
            "erract": [],
            "errdev": [],
            "errprt": [],
            "furnsh": [],
            "unload": [],
            "str2et": [],
            "spkezr": [],
            "pxform": [],
            "sxform": [],
            "bods2c": [],
            "bodvcd": [],
            "et2utc": [],
            "sce2s": [],
            "scs2e": [],
        }

    # -- test configuration --------------------------------------------------

    def plan_furnish(self, path: str, rows: list[dict[str, Any]]) -> None:
        """Pin the exact pool rows a ``furnsh(path)`` should add (meta-kernel fan-out)."""
        self._furnish_plan[path] = rows

    def fail_furnsh(self, path: str, exc: BaseException | None = None) -> None:
        """Make ``furnsh(path)`` raise — a corrupt / unreadable kernel."""
        self._furnsh_error[path] = exc or FakeSpiceyError(f"could not load kernel {path!r}")

    def plan_state(self, target: str, observer: str, state: list[float], light_time: float) -> None:
        """Pin the ``(state, light_time)`` a ``spkezr(target, …, observer)`` returns.

        ``state`` is the six-element [x, y, z, vx, vy, vz] vector in km / km/s.
        Keyed case-insensitively on (target, observer) so a test can mimic a real
        reference state without modelling CSPICE's ephemeris math.
        """
        self._state_plan[(target.upper(), observer.upper())] = (list(state), float(light_time))

    def plan_rotation(
        self,
        from_frame: str,
        to_frame: str,
        rotation: list[list[float]],
        state_transform: list[list[float]] | None = None,
        requires: str | None = None,
    ) -> None:
        """Pin the matrices a ``pxform`` / ``sxform`` for this frame pair returns.

        ``rotation`` is the 3x3 pxform matrix; ``state_transform`` the optional
        6x6 sxform matrix — an ``sxform`` with none pinned raises, mirroring a
        request for a state transform a frame cannot provide. ``requires`` names
        a kernel category (e.g. ``"PCK"``) that must be in the pool for either
        call to succeed, mirroring CSPICE's dependence on a furnished FK / PCK
        for a body-fixed frame; ``None`` needs no kernel (a built-in inertial
        frame). Keyed case-insensitively on (from_frame, to_frame).
        """
        self._rotation_plan[(from_frame.upper(), to_frame.upper())] = (
            rotation,
            state_transform,
            requires,
        )

    def plan_body_code(self, name: str, code: int) -> None:
        """Pin the NAIF code ``bods2c`` resolves a body *name* to (case-insensitive)."""
        self._body_code_plan[name.upper()] = code

    def plan_body_constant(
        self, code: int, item: str, values: list[float], requires: str | None = None
    ) -> None:
        """Pin the values ``bodvcd`` returns for ``BODY<code>_<item>``.

        ``requires`` names a kernel category (e.g. ``"PCK"``) that must be in the
        pool, mirroring CSPICE reading the constant from a furnished kernel; the
        constant is otherwise absent, so a missing-constant query raises
        ``SPICE(KERNELVARNOTFOUND)``. Keyed case-insensitively on the item.
        """
        self._body_constant_plan[(code, item.upper())] = (list(values), requires)

    # -- error-handling setters (recorded, no behaviour) ---------------------

    def erract(self, op: str, action: str | None = None) -> str:
        self.calls["erract"].append((op, action))
        return action or "RETURN"

    def errdev(self, op: str, device: str | None = None) -> str:
        self.calls["errdev"].append((op, device))
        return device or "NULL"

    def errprt(self, op: str, value: str | None = None) -> str:
        self.calls["errprt"].append((op, value))
        return value or "NONE"

    # -- kernel pool ---------------------------------------------------------

    def _default_row(self, path: str) -> dict[str, Any]:
        ktype = _infer_type(path)
        handle = 0
        if ktype in _BINARY_TYPES:
            handle = self._next_handle
            self._next_handle += 1
        return {"name": path, "type": ktype, "source": "", "handle": handle}

    def furnsh(self, path: str) -> None:
        self.calls["furnsh"].append(path)
        if path in self._furnsh_error:
            raise self._furnsh_error[path]
        rows = self._furnish_plan.get(path)
        if rows is None:
            rows = [self._default_row(path)]
        for row in rows:
            if not any(entry["name"] == row["name"] for entry in self._pool):
                self._pool.append(dict(row))

    def unload(self, path: str) -> None:
        self.calls["unload"].append(path)
        self._pool = [entry for entry in self._pool if entry["name"] != path]

    def _filtered(self, kind: str | None) -> list[dict[str, Any]]:
        if not kind or kind == "ALL":
            return list(self._pool)
        wanted = set(kind.split())
        return [entry for entry in self._pool if entry["type"] in wanted]

    def ktotal(self, kind: str) -> int:
        return len(self._filtered(kind))

    def kdata(
        self,
        which: int,
        kind: str,
        fillen: int = 256,
        typlen: int = 33,
        srclen: int = 256,
    ) -> tuple[str, str, str, int, bool]:
        rows = self._filtered(kind)
        if which < 0 or which >= len(rows):
            return ("", "", "", 0, False)
        entry = rows[which]
        return (entry["name"], entry["type"], entry["source"], int(entry["handle"]), True)

    # -- state queries -------------------------------------------------------

    def _has_type(self, ktype: str) -> bool:
        return any(entry["type"] == ktype for entry in self._pool)

    def str2et(self, time: str) -> float:
        """Resolve a UTC string to ephemeris time; needs a leap-second kernel.

        CSPICE ``str2et`` reads the loaded LSK to apply leap seconds, so without
        one furnished it raises ``SPICE(NOLEAPSECONDS)``. The returned value is a
        deterministic stand-in (the call is recorded so a test can assert the
        offset-stripped UTC string the tool passed).
        """
        self.calls["str2et"].append(time)
        if not self._has_type("TEXT"):
            raise FakeSpiceyError("SPICE(NOLEAPSECONDS): no leapseconds kernel has been loaded")
        return self._str2et_plan.get(time, 0.0)

    def spkezr(
        self, targ: str, et: float, ref: str, abcorr: str, obs: str
    ) -> tuple[list[float], float]:
        """Return the pinned state of *targ* relative to *obs*; needs an SPK.

        Without an SPK furnished CSPICE raises ``SPICE(SPKINSUFFDATA)``; we mimic
        that. With one loaded, a state pinned via :meth:`plan_state` is returned,
        else a deterministic default so unplanned calls still round-trip.
        """
        self.calls["spkezr"].append((targ, et, ref, abcorr, obs))
        if not self._has_type("SPK"):
            raise FakeSpiceyError(
                "SPICE(SPKINSUFFDATA): insufficient ephemeris data has been loaded "
                f"to compute the state of {targ!r} relative to {obs!r}"
            )
        planned = self._state_plan.get((targ.upper(), obs.upper()))
        if planned is not None:
            state, light_time = planned
            return (list(state), light_time)
        return ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1.234)

    # -- frame transforms ----------------------------------------------------

    def pxform(self, frm: str, to: str, et: float) -> list[list[float]]:
        """Return the pinned 3x3 rotation; mirror CSPICE's kernel / frame errors.

        Without a rotation pinned for the pair, CSPICE raises
        ``SPICE(NOFRAMECONNECT)`` (an unknown or unconnectable frame); when the
        plan names a required kernel category that is not furnished, it raises a
        no-data error (the FK / PCK defining the frame has not been loaded). We
        mimic both so the tool's typed-error paths are exercised through real
        pool state.
        """
        self.calls["pxform"].append((frm, to, et))
        plan = self._rotation_plan.get((frm.upper(), to.upper()))
        if plan is None:
            raise FakeSpiceyError(
                f"SPICE(NOFRAMECONNECT): no connection between frames {frm!r} and {to!r}"
            )
        rotation, _state_transform, requires = plan
        if requires is not None and not self._has_type(requires):
            raise FakeSpiceyError(
                f"SPICE(NOFRAMEDATA): the kernel data defining frame {to!r} has not been loaded"
            )
        return [list(row) for row in rotation]

    def sxform(self, frm: str, to: str, et: float) -> list[list[float]]:
        """Return the pinned 6x6 state transform; mirror CSPICE's kernel / frame errors."""
        self.calls["sxform"].append((frm, to, et))
        plan = self._rotation_plan.get((frm.upper(), to.upper()))
        if plan is None:
            raise FakeSpiceyError(
                f"SPICE(NOFRAMECONNECT): no connection between frames {frm!r} and {to!r}"
            )
        _rotation, state_transform, requires = plan
        if state_transform is None:
            raise FakeSpiceyError(
                f"SPICE(NOFRAMECONNECT): no state transform between frames {frm!r} and {to!r}"
            )
        if requires is not None and not self._has_type(requires):
            raise FakeSpiceyError(
                f"SPICE(NOFRAMEDATA): the kernel data defining frame {to!r} has not been loaded"
            )
        return [list(row) for row in state_transform]

    # -- body constants ------------------------------------------------------

    def bods2c(self, name: str) -> tuple[int, bool]:
        """Resolve a body name / ID string to a NAIF code, with a found flag.

        Digit strings resolve to their integer (CSPICE's built-in code mapping);
        names resolve via :meth:`plan_body_code`; anything else is not found —
        the unknown-body path.
        """
        self.calls["bods2c"].append(name)
        key = name.strip().upper()
        if key in self._body_code_plan:
            return (self._body_code_plan[key], True)
        if key.lstrip("-").isdigit():
            return (int(key), True)
        return (0, False)

    def bodvcd(self, bodyid: int, item: str, maxn: int) -> tuple[int, list[float]]:
        """Return the pinned values for ``BODY<bodyid>_<item>``; mirror CSPICE's errors.

        Without a value pinned for the pair — or with its required kernel category
        absent from the pool — CSPICE raises ``SPICE(KERNELVARNOTFOUND)``; we mimic
        that so the missing-constant path is exercised through real pool state.
        """
        self.calls["bodvcd"].append((bodyid, item, maxn))
        plan = self._body_constant_plan.get((bodyid, item.upper()))
        if plan is None:
            raise FakeSpiceyError(
                f"SPICE(KERNELVARNOTFOUND): the variable BODY{bodyid}_{item} could not be "
                "found in the kernel pool"
            )
        values, requires = plan
        if requires is not None and not self._has_type(requires):
            raise FakeSpiceyError(
                f"SPICE(KERNELVARNOTFOUND): the variable BODY{bodyid}_{item} could not be "
                "found in the kernel pool"
            )
        return (len(values), list(values))

    # -- time conversions ----------------------------------------------------

    def plan_time(self, utc: str, et: float) -> None:
        """Pin a bidirectional UTC<->ET pair: ``str2et(utc)`` and ``et2utc(et)``.

        *utc* is the offset-free calendar string CSPICE sees — the form the tool
        layer hands ``str2et`` (a trailing ``Z`` / zone offset already stripped),
        which is also what ``et2utc`` returns for the inverse. Pinning both
        directions makes a UTC<->ET round-trip exact without modelling CSPICE's
        leap-second math.
        """
        self._str2et_plan[utc] = float(et)
        self._et2utc_plan[float(et)] = utc

    def plan_sclk(self, sc: int, sclk: str, et: float) -> None:
        """Pin a bidirectional SCLK<->ET pair for spacecraft *sc*.

        ``scs2e(sc, sclk)`` returns *et* and ``sce2s(sc, et)`` returns *sclk*, so
        an SCLK round-trip is exact. Keyed on the integer NAIF spacecraft ID the
        tool resolves the request's `spacecraft` to.
        """
        self._scs2e_plan[(sc, sclk)] = float(et)
        self._sce2s_plan[(sc, float(et))] = sclk

    def _has_sclk(self) -> bool:
        """Whether a spacecraft-clock kernel (``.tsc``) is furnished.

        CSPICE reports an SCLK kernel as ``TEXT`` like an LSK, so the category
        alone cannot tell them apart; the fake keys SCLK availability on the
        ``.tsc`` suffix of a furnished path, mirroring "an SCLK kernel must be
        loaded" precisely enough to exercise the tool's missing-SCLK error path.
        """
        return any(entry["name"].endswith(".tsc") for entry in self._pool)

    def et2utc(self, et: float, format_str: str, prec: int) -> str:
        """Render ephemeris time as a UTC calendar string; needs a leap-second kernel.

        Without an LSK furnished CSPICE raises ``SPICE(NOLEAPSECONDS)``; we mimic
        that. With one loaded, a value pinned via :meth:`plan_time` is returned,
        else a deterministic stand-in so unplanned calls still produce a string.
        """
        self.calls["et2utc"].append((et, format_str, prec))
        if not self._has_type("TEXT"):
            raise FakeSpiceyError("SPICE(NOLEAPSECONDS): no leapseconds kernel has been loaded")
        planned = self._et2utc_plan.get(et)
        if planned is not None:
            return planned
        return f"ET{et}"

    def sce2s(self, sc: int, et: float) -> str:
        """Encode ephemeris time as a spacecraft-clock string; needs an SCLK kernel.

        Without an SCLK kernel furnished for *sc* CSPICE raises an SCLK error; we
        mimic that with :meth:`_has_sclk`. With one loaded, the value pinned via
        :meth:`plan_sclk` is returned; an unpinned (sc, et) raises rather than
        fabricating a clock reading.
        """
        self.calls["sce2s"].append((sc, et))
        if not self._has_sclk():
            raise FakeSpiceyError(
                f"SPICE(NOSCLKFILEFOUND): no spacecraft-clock kernel loaded for spacecraft {sc}"
            )
        planned = self._sce2s_plan.get((sc, et))
        if planned is None:
            raise FakeSpiceyError(
                f"SPICE(SPKINSUFFDATA): no SCLK mapping for spacecraft {sc} at et {et}"
            )
        return planned

    def scs2e(self, sc: int, sclkch: str) -> float:
        """Decode a spacecraft-clock string to ephemeris time; needs an SCLK kernel.

        Without an SCLK kernel furnished for *sc* CSPICE raises an SCLK error; we
        mimic that. An SCLK string with no pinned mapping raises
        ``SPICE(INVALIDSCLKSTRING)``, exercising the tool's unparseable-SCLK path.
        """
        self.calls["scs2e"].append((sc, sclkch))
        if not self._has_sclk():
            raise FakeSpiceyError(
                f"SPICE(NOSCLKFILEFOUND): no spacecraft-clock kernel loaded for spacecraft {sc}"
            )
        planned = self._scs2e_plan.get((sc, sclkch))
        if planned is None:
            raise FakeSpiceyError(
                f"SPICE(INVALIDSCLKSTRING): cannot parse SCLK string {sclkch!r} for spacecraft {sc}"
            )
        return planned
