"""Final (or steady) state of a run: one entry point for transient and steady solves.

Solver names (``ode_solver``, extensions of Hydrograd.jl's names):

* ``SteadyNewton()``: steady Newton solve with IFT gradients;
* ``BackwardEuler()`` / ``CrankNicolson()``: implicit theta-scheme (theta = 1 / 0.5) with
  fixed dt = ``time_settings.dt`` and per-step IFT adjoints;
* anything else: explicit time integration with diffrax.

Steady and implicit solves support reverse-mode gradients only.
"""

from __future__ import annotations

from typing import Any

import diffrax as dfx
import jax.numpy as jnp
from jax import Array

from hydrograd.physics.model import SWEModel
from hydrograd.physics.rhs import rhs
from hydrograd.solvers.implicit import ImplicitSettings, solve_implicit
from hydrograd.solvers.ode import ODESettings, solve_swe
from hydrograd.solvers.steady import NewtonSettings, steady_state

__all__ = ["IMPLICIT", "STEADY", "final_state", "implicit_settings", "is_implicit", "is_steady",
           "newton_settings", "reverse_only"]

STEADY = "SteadyNewton()"
STEADY_TOL = 1.0e-8   # max|dU/dt| accepted as converged when checking the result


def is_steady(s: ODESettings) -> bool:
    """True if the settings request the steady Newton solve."""
    return s.solver == STEADY


IMPLICIT = {"BackwardEuler()": 1.0, "CrankNicolson()": 0.5}


def is_implicit(s: ODESettings) -> bool:
    """True if the settings request the implicit theta-scheme."""
    return s.solver in IMPLICIT


def reverse_only(s: ODESettings) -> bool:
    """Steady and implicit solves provide reverse-mode (IFT) gradients only."""
    return is_steady(s) or is_implicit(s)


def implicit_settings(s: ODESettings) -> ImplicitSettings:
    """Theta-scheme settings from the ODE settings (fixed dt = s.dt, same save frames)."""
    return ImplicitSettings(s.t0, s.t1, s.dt, theta=IMPLICIT[s.solver], n_save=s.n_save)


def newton_settings(s: ODESettings) -> NewtonSettings:
    """Newton settings derived from the ODE settings (dtau0 = 100 dt)."""
    return NewtonSettings(dtau0=100.0 * s.dt)


def final_state(m: SWEModel, theta: dict[str, Any], U0: Array, s: ODESettings) -> tuple[Array, Array]:
    """(U_final [3, n_cells], ok flag) for a transient or steady run (differentiable)."""
    if is_steady(s):
        U = steady_state(m, theta, U0, newton_settings(s))
        ok = jnp.max(jnp.abs(rhs(m, theta, U))) < STEADY_TOL
        return U, ok
    if is_implicit(s):
        _, ys = solve_implicit(m, theta, U0, implicit_settings(s))
        return ys[-1], jnp.all(jnp.isfinite(ys[-1]))
    sol = solve_swe(m, theta, U0, s)
    assert sol.ys is not None
    return sol.ys[-1], jnp.asarray(sol.result == dfx.RESULTS.successful)
