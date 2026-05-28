"""`lambert_solve` tool — Lambert's problem against `lamberthub`.

Wraps four `lamberthub` solvers (Izzo 2015, Gooding 1990, Battin 1984) under
a uniform tool surface. Multi-revolution solutions are enumerated via
``M=0..revs`` with both `low_path` branches for ``M ≥ 1``; the primary
``v1/v2`` echo the user-requested ``(revs, low_path=True)`` solution and
``all_solutions`` lists every feasible alternative.

Note that "requested" means ``M = revs`` — the *highest* rev count asked
for. With the default ``revs=0`` the headline is the direct transfer, but
for ``revs > 0`` the headline ``v1/v2/transfer_elements/dv`` describe the
highest-rev arc (usually a much larger Δv); the direct transfer is the
``M=0`` entry in ``all_solutions``.

dv is the two-impulse sum ``|v1 - depart_velocity| + |v2 - arrive_velocity|``
when both boundary velocities are supplied — the cost of dropping a
spacecraft moving at ``depart_velocity`` onto the transfer arc and then
matching to ``arrive_velocity`` at the other end.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal

import numpy as np
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from astrodynamics_mcp.errors import InvalidInputError, UpstreamError
from astrodynamics_mcp.schemas.base import KeplerianElements
from astrodynamics_mcp.server import register_tool
from astrodynamics_mcp.units import Quantity, QuantityVector

# `lamberthub` pulls scipy.special at import time (~4 s on a cold WSL box).
# The package-level side-effect chain re-imports this module on every
# subprocess spawn — eager `import lamberthub` is enough to blow past the
# multi-process cache test's 10 s timeout. Defer the solver lookup until a
# tool call actually needs it (see ``_algorithms`` below).

# Gravitational parameters (km³/s²) for v0.1 named bodies. Values from
# JPL ssd.jpl.nasa.gov; barycentre μ used for the Sun rather than the
# heliocentric μ, since Lambert solves in the inertial frame of the chosen
# body.
_BODY_MU: dict[str, float] = {
    "sun": 1.32712440018e11,
    "mercury": 2.2032e4,
    "venus": 3.24858592e5,
    "earth": 3.986004418e5,
    "moon": 4.9028e3,
    "mars": 4.282837e4,
    "jupiter": 1.26686534e8,
    "saturn": 3.7931187e7,
    "uranus": 5.793939e6,
    "neptune": 6.836529e6,
}

# Algorithm-name → lamberthub callable, populated lazily on first call.
# "izzo" is Izzo's 2015 paper; "izzo_revisited" is the same algorithm under
# its colloquial name — kept as a separate enum value so the LLM-facing
# string surface stays stable across the wider literature.
_ALGORITHMS: dict[str, Callable[..., Any]] = {}


def _algorithms() -> dict[str, Callable[..., Any]]:
    """Return the algorithm-name → solver map, initialising on first call.

    Imports of `lamberthub` are deferred until a tool call actually needs
    a solver — keeps package-level import cheap (the cache test's spawned
    subprocesses must boot inside 10 s) without affecting the wire surface.
    """
    if _ALGORITHMS:
        return _ALGORITHMS
    import lamberthub

    _ALGORITHMS.update(
        {
            "izzo": lamberthub.izzo2015,
            "izzo_revisited": lamberthub.izzo2015,
            "gooding": lamberthub.gooding1990,
            "battin": lamberthub.battin1984,
        }
    )
    return _ALGORITHMS


# Direction → prograde flag for lamberthub.
_DIRECTION_PROGRADE: dict[str, bool] = {"prograde": True, "retrograde": False}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class LambertSolution(BaseModel):
    """One Lambert solution row inside :class:`LambertSolveResponse.all_solutions`."""

    model_config = ConfigDict(extra="forbid")

    v1: QuantityVector = Field(
        ..., description="Initial velocity vector on the transfer arc (km/s)."
    )
    v2: QuantityVector = Field(..., description="Final velocity vector on the transfer arc (km/s).")
    transfer_elements: KeplerianElements = Field(
        ..., description="Classical orbital elements of the transfer arc."
    )
    revs: Quantity = Field(
        ...,
        description="Revolution count M, dimensionless. e.g. {value: 2, unit: '1'}.",
        examples=[{"value": 0, "unit": "1"}, {"value": 2, "unit": "1"}],
    )
    low_path: bool = Field(
        ...,
        description=(
            "Which of the two multi-rev branches this solution is. True for the "
            "low-path branch (always True for the degenerate M=0 case)."
        ),
    )


class LambertSolveResponse(BaseModel):
    """Response from :func:`lambert_solve`.

    Top-level ``v1`` / ``v2`` / ``transfer_elements`` / ``dv`` echo the
    user-requested ``(revs, low_path=True)`` solution for ergonomics —
    i.e. the ``M = revs`` arc. With the default ``revs=0`` that is the direct
    transfer; for ``revs > 0`` it is the highest-rev arc (usually a much
    larger Δv), and the direct transfer is the ``M=0`` entry in
    ``all_solutions``, which lists every feasible (M, low_path) pair from M=0
    up to the requested revolution count.
    """

    model_config = ConfigDict(extra="forbid")

    v1: QuantityVector = Field(..., description="Initial velocity for the primary solution (km/s).")
    v2: QuantityVector = Field(..., description="Final velocity for the primary solution (km/s).")
    transfer_elements: KeplerianElements = Field(
        ..., description="Orbital elements of the primary solution's transfer arc."
    )
    dv: Quantity | None = Field(
        None,
        description=(
            "Two-impulse Δv (km/s) when both `depart_velocity` and `arrive_velocity` "
            "were supplied. None otherwise."
        ),
    )
    all_solutions: list[LambertSolution] = Field(
        ...,
        description=(
            "Every feasible (M, low_path) Lambert solution from M=0 to the "
            "requested revolution count. Infeasible alternatives are skipped, "
            "not surfaced as errors."
        ),
    )


# ---------------------------------------------------------------------------
# Tool description (subject to server_lint)
# ---------------------------------------------------------------------------

_DESCRIPTION = (
    "Solve Lambert's problem — find the orbital transfer between two position "
    "vectors r1 and r2 in a given time-of-flight tof. e.g. lambert_solve("
    "r1=[5000, 10000, 2100], r2=[-14600, 2500, 7000], tof=3600, mu='earth') "
    "returns the initial and final inertial velocities on the transfer arc "
    "plus the arc's classical orbital elements. `mu` selects the central "
    "body and is REQUIRED — there is no default, because the wrong choice "
    "silently produces garbage. `r1` and `r2` are heliocentric km when "
    "mu='sun'; geocentric km when mu='earth'; planetocentric km for other "
    "named bodies. If your problem is interplanetary (e.g. Earth-to-Mars "
    "transfer), use mu='sun' and pull body positions via the porkchop tool. "
    "Algorithm: `izzo` is fast and robust for the common case; switch to "
    "`gooding` if `izzo` fails on near-degenerate geometries. For revs > 0 "
    "the response enumerates the multi-rev solutions in all_solutions "
    "(low + high path per rev count), and the top-level v1/v2/dv then describe "
    "the highest-rev arc you asked for, not the direct transfer — read the M=0 "
    "entry in all_solutions for the direct transfer. Supply depart_velocity AND arrive_velocity "
    "together to get the two-impulse Δv. Degenerate geometries (r1 == r2, "
    "infeasible tof, no convergence) surface as `upstream.lambert_no_solution`."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_mu(mu: str | float) -> float:
    if isinstance(mu, str):
        key = mu.lower()
        if key not in _BODY_MU:
            raise InvalidInputError(
                f"unknown body {mu!r}; supported: {sorted(_BODY_MU)}",
                code="invalid_input.unknown_body",
            )
        return _BODY_MU[key]
    # bool is a subclass of int — guard explicitly so True/False don't pose as μ.
    if isinstance(mu, bool) or not isinstance(mu, (int, float)):
        raise InvalidInputError(
            f"mu must be a body name or a numeric km^3/s^2 value, got {type(mu).__name__}",
            code="invalid_input.wrong_mu_type",
        )
    if mu <= 0 or not np.isfinite(mu):
        raise InvalidInputError(
            f"mu must be a positive finite number (km^3/s^2), got {mu}",
            code="invalid_input.wrong_mu_value",
        )
    return float(mu)


def _validate_position(vec: list[float], field: str) -> np.ndarray:
    if len(vec) != 3:
        raise InvalidInputError(
            f"{field} must have exactly 3 components, got {len(vec)}",
            code="invalid_input.wrong_vector_length",
        )
    for i, x in enumerate(vec):
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x):
            raise InvalidInputError(
                f"{field}[{i}] must be a finite number, got {x!r}",
                code="invalid_input.value_not_a_number",
            )
    return np.asarray(vec, dtype=float)


def _classical_elements(r: np.ndarray, v: np.ndarray, mu: float) -> KeplerianElements:
    """Standard r,v -> (a, e, i, RAAN, argp, nu) conversion (km, km/s, deg)."""
    eps = 1e-10

    r_mag = float(np.linalg.norm(r))
    v_mag = float(np.linalg.norm(v))
    h = np.cross(r, v)
    h_mag = float(np.linalg.norm(h))
    n = np.cross([0.0, 0.0, 1.0], h)
    n_mag = float(np.linalg.norm(n))

    e_vec = ((v_mag**2 - mu / r_mag) * r - float(np.dot(r, v)) * v) / mu
    e = float(np.linalg.norm(e_vec))

    energy = v_mag**2 / 2 - mu / r_mag
    a = float(-mu / (2 * energy))  # negative for hyperbolic; OK by design

    i_rad = float(np.arccos(np.clip(h[2] / h_mag, -1.0, 1.0)))

    if n_mag > eps:
        raan_rad = float(np.arccos(np.clip(n[0] / n_mag, -1.0, 1.0)))
        if n[1] < 0:
            raan_rad = 2 * np.pi - raan_rad
    else:
        raan_rad = 0.0

    if e > eps and n_mag > eps:
        argp_rad = float(np.arccos(np.clip(float(np.dot(n, e_vec)) / (n_mag * e), -1.0, 1.0)))
        if e_vec[2] < 0:
            argp_rad = 2 * np.pi - argp_rad
    else:
        argp_rad = 0.0

    if e > eps:
        nu_rad = float(np.arccos(np.clip(float(np.dot(e_vec, r)) / (e * r_mag), -1.0, 1.0)))
        if float(np.dot(r, v)) < 0:
            nu_rad = 2 * np.pi - nu_rad
    else:
        nu_rad = 0.0

    return KeplerianElements(
        a=Quantity(value=a, unit="km"),
        e=Quantity(value=e, unit="1"),
        i=Quantity(value=float(np.degrees(i_rad)), unit="deg"),
        raan=Quantity(value=float(np.degrees(raan_rad)), unit="deg"),
        argp=Quantity(value=float(np.degrees(argp_rad)), unit="deg"),
        nu=Quantity(value=float(np.degrees(nu_rad)), unit="deg"),
    )


def _solve_one(
    algorithm: Callable[..., Any],
    mu: float,
    r1: np.ndarray,
    r2: np.ndarray,
    tof: float,
    M: int,
    prograde: bool,
    low_path: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Call lamberthub once. Returns (v1, v2) or None on infeasible solution."""
    try:
        result = algorithm(mu, r1, r2, tof, M=M, prograde=prograde, low_path=low_path)
    except (AssertionError, ValueError):
        return None
    # lamberthub returns either (v1, v2) or (v1, v2, numiter) depending on solver.
    v1 = np.asarray(result[0], dtype=float)
    v2 = np.asarray(result[1], dtype=float)
    return v1, v2


def _build_solution(
    v1: np.ndarray,
    v2: np.ndarray,
    r1: np.ndarray,
    mu: float,
    M: int,
    low_path: bool,
) -> LambertSolution:
    return LambertSolution(
        v1=QuantityVector(value=[float(v1[0]), float(v1[1]), float(v1[2])], unit="km/s"),
        v2=QuantityVector(value=[float(v2[0]), float(v2[1]), float(v2[2])], unit="km/s"),
        transfer_elements=_classical_elements(r1, v1, mu),
        revs=Quantity(value=float(M), unit="1"),
        low_path=low_path,
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@register_tool(
    name="lambert_solve",
    description=_DESCRIPTION,
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
)
async def lambert_solve(
    r1: Annotated[
        list[float],
        Field(
            description=(
                "Departure position vector (km), 3-component [x, y, z] in the "
                "inertial frame of the central body identified by `mu`."
            ),
        ),
    ],
    r2: Annotated[
        list[float],
        Field(
            description=(
                "Arrival position vector (km), same frame and units as `r1`. The "
                "solver finds the transfer arc connecting `r1` to `r2` in `tof` "
                "seconds."
            ),
        ),
    ],
    tof: Annotated[
        float,
        Field(
            description=(
                "Time of flight from `r1` to `r2`, in seconds. Must be strictly "
                "positive and finite."
            ),
        ),
    ],
    mu: Annotated[
        str | float,
        Field(
            description=(
                "Gravitational parameter of the central body — REQUIRED, no default. "
                "Pass a body name ('sun', 'mercury', 'venus', 'earth', 'moon', "
                "'mars', 'jupiter', 'saturn', 'uranus', 'neptune') to use the JPL-"
                "published μ, or a raw number in km³/s² for a custom value. Must "
                "match the frame `r1`/`r2` are expressed in: heliocentric → 'sun', "
                "geocentric → 'earth', planetocentric → that planet."
            ),
        ),
    ],
    direction: Annotated[
        Literal["prograde", "retrograde"],
        Field(
            description=(
                "Sense of motion along the transfer arc. 'prograde' is the "
                "common case (eastward in Earth orbit, counter-clockwise viewed "
                "from the +Z pole); 'retrograde' flips the direction."
            ),
        ),
    ] = "prograde",
    revs: Annotated[
        int,
        Field(
            description=(
                "Maximum number of complete revolutions to enumerate. 0 (default) "
                "returns only the zero-rev / direct-transfer solution. For revs ≥ 1 "
                "both low_path branches are enumerated; the primary v1/v2 echo the "
                "(revs, low_path=True) solution — i.e. the highest-rev arc, NOT the "
                "direct transfer — and all_solutions lists every feasible alternative "
                "(the direct transfer is the M=0 entry there)."
            ),
        ),
    ] = 0,
    algorithm: Annotated[
        Literal["izzo", "izzo_revisited", "gooding", "battin"],
        Field(
            description=(
                "Lambert solver. 'izzo' (default) has the broadest convergence "
                "basin; 'izzo_revisited', 'gooding', and 'battin' are alternative "
                "implementations from `lamberthub` exposed for cross-validation."
            ),
        ),
    ] = "izzo",
    depart_velocity: Annotated[
        list[float] | None,
        Field(
            description=(
                "Optional departure-state velocity (km/s), 3-component, in the "
                "same frame as `r1`. When supplied alongside `arrive_velocity`, "
                "the tool also returns the two-impulse Δv "
                "|v1 - depart_velocity| + |v2 - arrive_velocity|. Pass both or neither."
            ),
        ),
    ] = None,
    arrive_velocity: Annotated[
        list[float] | None,
        Field(
            description=(
                "Optional arrival-state velocity (km/s), 3-component, in the same "
                "frame as `r2`. Required together with `depart_velocity` to "
                "compute the two-impulse Δv; omit both to skip Δv computation."
            ),
        ),
    ] = None,
) -> LambertSolveResponse:
    # Input validation.
    r1_arr = _validate_position(r1, "r1")
    r2_arr = _validate_position(r2, "r2")
    if isinstance(tof, bool) or not isinstance(tof, (int, float)) or not np.isfinite(tof):
        raise InvalidInputError(
            f"tof must be a finite number (s), got {tof!r}",
            code="invalid_input.value_not_a_number",
        )
    if tof <= 0:
        raise InvalidInputError(
            f"tof must be strictly positive seconds, got {tof}",
            code="invalid_input.tof_not_positive",
        )
    if isinstance(revs, bool) or not isinstance(revs, int) or revs < 0:
        raise InvalidInputError(
            f"revs must be a non-negative integer, got {revs!r}",
            code="invalid_input.revs_not_a_non_negative_int",
        )
    if (depart_velocity is None) ^ (arrive_velocity is None):
        raise InvalidInputError(
            "depart_velocity and arrive_velocity must both be supplied to compute dv, "
            "or both omitted to skip dv computation.",
            code="invalid_input.lambert_partial_dv",
        )
    mu_value = _resolve_mu(mu)
    solver = _algorithms()[algorithm]
    prograde = _DIRECTION_PROGRADE[direction]

    # Solve the primary (requested-revs, low_path=True) case first; failure
    # here is a hard error since the caller explicitly asked for this combo.
    primary = _solve_one(solver, mu_value, r1_arr, r2_arr, tof, revs, prograde, low_path=True)
    if primary is None:
        raise UpstreamError(
            f"Lambert solver {algorithm!r} found no solution for the requested "
            f"(revs={revs}, low_path=True) configuration",
            code="upstream.lambert_no_solution",
            data={
                "algorithm": algorithm,
                "revs": revs,
                "low_path": True,
                "direction": direction,
            },
        )
    primary_v1, primary_v2 = primary
    all_solutions: list[LambertSolution] = []

    # Enumerate every (M, low_path) pair from M=0 to revs. M=0 has a single
    # branch — low_path / high_path coincide — so we emit only the
    # low_path=True row to avoid a duplicate.
    for m_value in range(revs + 1):
        branches = [True] if m_value == 0 else [True, False]
        for branch_low_path in branches:
            if m_value == revs and branch_low_path is True:
                # Reuse the already-computed primary so the solver is hit
                # once per unique problem.
                all_solutions.append(
                    _build_solution(
                        primary_v1, primary_v2, r1_arr, mu_value, m_value, branch_low_path
                    )
                )
                continue
            alt = _solve_one(
                solver, mu_value, r1_arr, r2_arr, tof, m_value, prograde, branch_low_path
            )
            if alt is None:
                continue
            v1_alt, v2_alt = alt
            all_solutions.append(
                _build_solution(v1_alt, v2_alt, r1_arr, mu_value, m_value, branch_low_path)
            )

    # Optional two-impulse Δv.
    dv: Quantity | None = None
    if depart_velocity is not None and arrive_velocity is not None:
        dep = _validate_position(depart_velocity, "depart_velocity")
        arr = _validate_position(arrive_velocity, "arrive_velocity")
        dv_value = float(np.linalg.norm(primary_v1 - dep) + np.linalg.norm(arr - primary_v2))
        dv = Quantity(value=dv_value, unit="km/s")

    primary_elements = _classical_elements(r1_arr, primary_v1, mu_value)

    return LambertSolveResponse(
        v1=QuantityVector(
            value=[float(primary_v1[0]), float(primary_v1[1]), float(primary_v1[2])],
            unit="km/s",
        ),
        v2=QuantityVector(
            value=[float(primary_v2[0]), float(primary_v2[1]), float(primary_v2[2])],
            unit="km/s",
        ),
        transfer_elements=primary_elements,
        dv=dv,
        all_solutions=all_solutions,
    )
