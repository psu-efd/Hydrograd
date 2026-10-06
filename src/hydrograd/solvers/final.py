"""Final (or steady) state of a run: one entry point for transient and steady solves.

Inversion and UDE losses compare the final state with observations; this module picks
the solver from ``ODESettings.solver`` (``ode_solver`` in ``run_control.json``; names
extend Hydrograd.jl's):

* ``SteadyNewton()``: steady Newton solve with IFT gradients (``solvers.steady``);
* ``BackwardEuler()`` / ``CrankNicolson()``: implicit theta-scheme (theta = 1 / 0.5) with
  fixed dt = ``time_settings.dt`` and per-step IFT adjoints (``solvers.implicit``);
* anything else: explicit time integration with diffrax (``solvers.ode``).

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

# Solver name of the steady Newton solve.
STEADY = "SteadyNewton()"
STEADY_TOL = 1.0e-8   # max|dU/dt| accepted as converged when checking the result

# Implicit solver names and their implicitness theta.
IMPLICIT = {"BackwardEuler()": 1.0, "CrankNicolson()": 0.5}


def is_steady(s: ODESettings) -> bool:
    """True if the settings request the steady Newton solve.

    Args:
        s: ODE settings.

    Returns:
        Whether ``s.solver`` is ``SteadyNewton()``.
    """
    return s.solver == STEADY


def is_implicit(s: ODESettings) -> bool:
    """True if the settings request the implicit theta-scheme.

    Args:
        s: ODE settings.

    Returns:
        Whether ``s.solver`` is ``BackwardEuler()`` or ``CrankNicolson()``.
    """
    return s.solver in IMPLICIT


def reverse_only(s: ODESettings) -> bool:
    """Steady and implicit solves provide reverse-mode (IFT) gradients only.

    Args:
        s: ODE settings.

    Returns:
        True if forward-mode (sensitivity) gradients are unavailable for this solver.
    """
    return is_steady(s) or is_implicit(s)


def implicit_settings(s: ODESettings) -> ImplicitSettings:
    """Theta-scheme settings from the ODE settings (fixed dt = s.dt, same save frames).

    Args:
        s: ODE settings with an implicit solver name.

    Returns:
        The matching ``ImplicitSettings``.
    """
    return ImplicitSettings(s.t0, s.t1, s.dt, theta=IMPLICIT[s.solver], n_save=s.n_save)


def newton_settings(s: ODESettings) -> NewtonSettings:
    """Newton settings derived from the ODE settings (dtau0 = 100 dt).

    Args:
        s: ODE settings.

    Returns:
        ``NewtonSettings`` with initial pseudo-time step 100 * s.dt (s).
    """
    return NewtonSettings(dtau0=100.0 * s.dt)


def final_state(
    m: SWEModel, theta: dict[str, Any], U0: Array, s: ODESettings,
) -> tuple[Array, Array]:
    """Final state of a transient or steady run (differentiable w.r.t. ``theta``).

    Args:
        m: the model.
        theta: differentiable parameters.
        U0: initial state (or initial guess for the steady solve), shape [3, n_cells].
        s: ODE settings; ``s.solver`` selects the solver.

    Returns:
        (U_final, ok): the final state, shape [3, n_cells], and a boolean scalar that is
        True if the solve succeeded (steady: max|dU/dt| < STEADY_TOL; implicit: finite
        state; explicit: diffrax reported success).
    """
    # Steady Newton solve; converged if the residual is below STEADY_TOL.
    if is_steady(s):
        U = steady_state(m, theta, U0, newton_settings(s))
        ok = jnp.max(jnp.abs(rhs(m, theta, U))) < STEADY_TOL
        return U, ok

    # Implicit theta-scheme: last saved frame; ok if finite.
    if is_implicit(s):
        _, ys = solve_implicit(m, theta, U0, implicit_settings(s))
        return ys[-1], jnp.all(jnp.isfinite(ys[-1]))

    # Explicit diffrax integration: last saved frame; ok if diffrax succeeded.
    sol = solve_swe(m, theta, U0, s)
    assert sol.ys is not None
    return sol.ys[-1], jnp.asarray(sol.result == dfx.RESULTS.successful)
