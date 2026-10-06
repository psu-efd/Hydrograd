"""Final (or steady) state of a run: one entry point for transient and steady solves.

``ode_solver = "SteadyNewton()"`` (extension of the run_control.json solver names)
selects the steady Newton solve with IFT gradients; any other name integrates in
time with diffrax and returns the last saved frame.
"""

from __future__ import annotations

from typing import Any

import diffrax as dfx
import jax.numpy as jnp
from jax import Array

from hydrograd.physics.model import SWEModel
from hydrograd.physics.rhs import rhs
from hydrograd.solvers.ode import ODESettings, solve_swe
from hydrograd.solvers.steady import NewtonSettings, steady_state

__all__ = ["STEADY", "final_state", "is_steady", "newton_settings"]

STEADY = "SteadyNewton()"
STEADY_TOL = 1.0e-8   # max|dU/dt| accepted as converged when checking the result


def is_steady(s: ODESettings) -> bool:
    """True if the settings request the steady Newton solve."""
    return s.solver == STEADY


def newton_settings(s: ODESettings) -> NewtonSettings:
    """Newton settings derived from the ODE settings (dtau0 = 100 dt)."""
    return NewtonSettings(dtau0=100.0 * s.dt)


def final_state(m: SWEModel, theta: dict[str, Any], U0: Array, s: ODESettings) -> tuple[Array, Array]:
    """(U_final [3, n_cells], ok flag) for a transient or steady run (differentiable)."""
    if is_steady(s):
        U = steady_state(m, theta, U0, newton_settings(s))
        ok = jnp.max(jnp.abs(rhs(m, theta, U))) < STEADY_TOL
        return U, ok
    sol = solve_swe(m, theta, U0, s)
    assert sol.ys is not None
    return sol.ys[-1], jnp.asarray(sol.result == dfx.RESULTS.successful)
