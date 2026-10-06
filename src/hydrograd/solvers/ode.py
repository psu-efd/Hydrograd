"""Time integration of the semi-discrete SWEs with diffrax.

Mirrors the SciML settings of Hydrograd.jl (``swe_2D_forward_simulation.jl``):
``Tsit5()`` adaptive with ``abstol = 1e-6``, ``reltol = 1e-3`` and initial
``dt``; ``Euler()``/``RK4()`` with a fixed ``dt``. Saving at
``t0 : (t1 - t0)/nSave : t1`` (nSave + 1 frames).

Reverse-mode gradients use diffrax's ``RecursiveCheckpointAdjoint`` (memory
grows only logarithmically with the number of steps); forward mode uses
``ForwardMode``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hydrograd.physics.model import SWEModel
from hydrograd.physics.rhs import rhs

logger = logging.getLogger(__name__)

__all__ = ["ODESettings", "n_checkpoints", "solve_swe"]

_FIXED_ONLY = ("Euler()", "RK4()")
_SOLVERS_EXTRA = ("SSPRK3()",)
_SOLVERS = {"Tsit5()": dfx.Tsit5, "Euler()": dfx.Euler, "RK4()": None, "Heun()": dfx.Heun,
            "Dopri5()": dfx.Dopri5, "Bosh3()": dfx.Bosh3}


@dataclass(frozen=True)
class ODESettings:
    """Time-integration settings (times in s).

    Attributes:
        t0, t1: time span.
        dt: initial (adaptive) or fixed step.
        n_save: number of save intervals (nSave + 1 frames incl. t0).
        solver: Julia-style name, e.g. ``Tsit5()``.
        adaptive: adaptive step control.
        rtol, atol: tolerances for adaptive stepping.
        max_steps: step cap (static, sets the checkpoint buffer).
        adjoint: ``checkpoint`` (reverse), ``forward`` (forward mode), or ``direct``.
        cfl: Courant number for ``SSPRK3()``, which always uses CFL-limited steps
            dt = cfl * min(A/P) / max(|u| + sqrt(g h)). Default (None): 0.5 at first
            order and 0.25 at second order, the positivity limits of the scheme.
        checkpoints: reverse-mode checkpoints: a number (default 64), or ``"auto"``.
            Memory is about 3 * checkpoints states. diffrax's own default (``None``) scales
            with sqrt(max_steps): ~4,200 states at max_steps = 1e6, which runs out of memory
            on large meshes (52 GB at 512k cells). Fewer checkpoints than time steps means
            recomputation in the backward pass (gradient ~5-8x a forward run at 64).
            ``"auto"``: as many as fit in ``checkpoint_memory_gb``, i.e.
            floor(budget / (3 * state bytes)), capped at max_steps. Small meshes then store
            every step (no recomputation); large meshes get as many as memory allows.
        checkpoint_memory_gb: memory budget (GB) for ``checkpoints = "auto"``.
    """

    t0: float
    t1: float
    dt: float
    n_save: int = 100
    solver: str = "Tsit5()"
    adaptive: bool = True
    rtol: float = 1.0e-3
    atol: float = 1.0e-6
    max_steps: int = 1_000_000
    adjoint: str = "checkpoint"
    checkpoints: int | str = 64
    checkpoint_memory_gb: float = 2.0
    cfl: float | None = None   # None: 0.5 for first order, 0.25 for second order


class CFLStepSize(dfx.AbstractStepSizeController):
    """Step size dt = cfl * dx / s_max(U) from the current state (not differentiated)."""

    cfl: float
    dx: Any
    dt_max: float
    speed: Callable = eqx.field(static=True)

    def wrap(self, direction):
        return self

    def _dt(self, y, args):
        s = jax.lax.stop_gradient(self.speed(y, args))
        return jnp.minimum(self.cfl * jax.lax.stop_gradient(self.dx) / jnp.maximum(s, 1e-12), self.dt_max)

    def init(self, terms, t0, t1, y0, dt0, args, func, error_order):
        return t0 + self._dt(y0, args), None

    def adapt_step_size(self, t0, t1, y0, y1_candidate, args, y_error, error_order, controller_state):
        return True, t1, t1 + self._dt(y1_candidate, args), False, None, dfx.RESULTS.successful


class SSPRK3PointImplicit(dfx.AbstractSolver):
    """Shu-Osher SSP-RK3 with point-implicit friction in every forward-Euler sub-step.

    Sub-step: W = U + dt F(U) (fluxes and bed terms), then q_W <- q_W / (1 + dt c(U)),
    with friction = -c q. Unconditionally stable for friction (explicit friction needs
    dt c < 2, which fails on thin films: c ~ 24 /s for h = 1 mm, |u| = 1 m/s, n = 0.035),
    and friction can never reverse the flow. Stages: U1 = E(U); U2 = 3/4 U + 1/4 E(U1);
    U3 = 1/3 U + 2/3 E(U2), E being the point-implicit Euler sub-step.
    """

    split: Callable = eqx.field(static=True)       # (t, U, args) -> (F, c)
    term_structure: ClassVar = dfx.AbstractTerm
    interpolation_cls: ClassVar = dfx.LocalLinearInterpolation

    def order(self, terms):
        return 3

    def init(self, terms, t0, t1, y0, args):
        return None

    def _euler(self, t, U, args, dt):
        F, c = self.split(t, U, args)
        W = U + dt * F
        return W.at[1:].set(W[1:] / (1.0 + dt * c))

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        dt = t1 - t0
        U1 = self._euler(t0, y0, args, dt)
        U2 = 0.75 * y0 + 0.25 * self._euler(t1, U1, args, dt)
        U3 = y0 / 3.0 + 2.0 / 3.0 * self._euler(t0 + 0.5 * dt, U2, args, dt)
        return U3, None, dict(y0=y0, y1=U3), None, dfx.RESULTS.successful

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)


def _ssprk3() -> dfx.AbstractSolver:
    """Shu-Osher SSP-RK3 (as an explicit RK tableau; identical updates)."""
    tab = dfx.ButcherTableau(
        c=np.array([1.0, 0.5]),
        b_sol=np.array([1 / 6, 1 / 6, 2 / 3]),
        b_error=np.zeros(3),
        a_lower=(np.array([1.0]), np.array([0.25, 0.25])))

    class _SSPRK3(dfx.AbstractERK):
        tableau = tab
        interpolation_cls = dfx.ThirdOrderHermitePolynomialInterpolation.from_k

        def order(self, terms: Any) -> int:
            return 3
    return _SSPRK3()


def _solver(name: str) -> dfx.AbstractSolver:
    if name == "SSPRK3()":
        return _ssprk3()
    if name == "RK4()":
        # Classical RK4 tableau (diffrax has no named RK4).
        tab = dfx.ButcherTableau(
            c=np.array([0.5, 0.5, 1.0]),
            b_sol=np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6]),
            b_error=np.zeros(4),
            a_lower=(np.array([0.5]), np.array([0.0, 0.5]), np.array([0.0, 0.0, 1.0])))

        class _RK4(dfx.AbstractERK):
            tableau = tab
            interpolation_cls = dfx.ThirdOrderHermitePolynomialInterpolation.from_k

            def order(self, terms: Any) -> int:
                return 4
        return _RK4()
    if name not in _SOLVERS:
        logger.warning("ODE solver %s not available; using Tsit5()", name)
        name = "Tsit5()"
    return _SOLVERS[name]()


def n_checkpoints(s: ODESettings, U0: Array) -> int:
    """Number of reverse-mode checkpoints for these settings and state size."""
    if s.checkpoints != "auto":
        return int(s.checkpoints)
    state_bytes = U0.size * U0.dtype.itemsize
    n = int(s.checkpoint_memory_gb * 2 ** 30 // (3 * state_bytes))
    return max(16, min(n, s.max_steps))


def solve_swe(m: SWEModel, theta: dict[str, Any], U0: Array, s: ODESettings,
              ts: Array | None = None) -> dfx.Solution:
    """Integrate dU/dt = rhs(U; theta) from ``s.t0`` to ``s.t1``.

    Args:
        m: static model; theta: parameters (differentiable pytree).
        U0: initial state [3, n_cells].
        s: settings; ts: save times (default: nSave + 1 uniform frames).

    Returns:
        diffrax Solution; ``sol.ys`` has shape [n_frames, 3, n_cells].
    """
    if ts is None:
        ts = jnp.linspace(s.t0, s.t1, s.n_save + 1)
    term = dfx.ODETerm(lambda t, y, args: rhs(m, args, y, None, t))
    if s.solver == "SSPRK3()":
        from hydrograd.physics.rhs_hr import max_wave_speed
        perim = (jax.ops.segment_sum(m.fi_L, m.fi_o, m.n_cells) + jax.ops.segment_sum(m.fi_L, m.fi_r, m.n_cells)
                 + jax.ops.segment_sum(m.fb_L, m.fb_c, m.n_cells))
        cfl = s.cfl if s.cfl is not None else (0.25 if m.order == 2 and m.scheme == "hr_hllc" else 0.5)
        ctrl = CFLStepSize(cfl=cfl, dx=jnp.min(m.area / perim), dt_max=s.t1 - s.t0,
                           speed=lambda y, a: max_wave_speed(m, a, y))
    elif s.adaptive and s.solver in _FIXED_ONLY:
        logger.warning("%s has no error estimator; using fixed steps dt = %g s", s.solver, s.dt)
    if s.solver == "SSPRK3()":
        pass
    elif s.adaptive and s.solver not in _FIXED_ONLY:
        ctrl = dfx.PIDController(rtol=s.rtol, atol=s.atol)
    else:
        ctrl = dfx.ConstantStepSize()
    adjoint = {"checkpoint": dfx.RecursiveCheckpointAdjoint(checkpoints=n_checkpoints(s, U0)),
               "forward": dfx.ForwardMode(),
               "direct": dfx.DirectAdjoint()}[s.adjoint]
    solver = _solver(s.solver)
    if s.solver == "SSPRK3()" and m.scheme == "hr_hllc":
        from hydrograd.physics.rhs import boundary_values_at
        from hydrograd.physics.rhs_hr import rhs_hr
        solver = SSPRK3PointImplicit(
            split=lambda t, y, a: rhs_hr(m, boundary_values_at(m, a, t), y, None, split=True))
    return dfx.diffeqsolve(term, solver, s.t0, s.t1, s.dt, U0, args=theta,
                           saveat=dfx.SaveAt(ts=ts), stepsize_controller=ctrl,
                           max_steps=s.max_steps, adjoint=adjoint, throw=False)
