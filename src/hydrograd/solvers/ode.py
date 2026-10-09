"""Explicit time integration of the semi-discrete SWEs with diffrax.

The finite-volume discretization (``physics.rhs``) turns the SWEs into the ODE system

    dU/dt = R(U, t; theta),   U = [xi, q_x, q_y], shape [3, n_cells],

which this module integrates from t0 to t1 (times in s).

Solvers (Hydrograd.jl names)
----------------------------
Mirrors the SciML settings of Hydrograd.jl (``swe_2D_forward_simulation.jl``):

- ``Tsit5()`` (default), ``Heun()``, ``Dopri5()``, ``Bosh3()``: adaptive steps with a
  PID controller, ``abstol = 1e-6``, ``reltol = 1e-3`` and initial ``dt``.
- ``Euler()``, ``RK4()``: fixed step ``dt`` (no error estimator).
- ``SSPRK3()``: Shu-Osher strong-stability-preserving RK3 with CFL-limited steps

      dt = cfl * min_i(A_i / P_i) / max_i(|u_i| + sqrt(g h_i)),

  with A_i the cell area (m^2) and P_i its perimeter (m). With the default ``hr_hllc``
  scheme the friction is treated point-implicitly in every stage (see
  ``SSPRK3PointImplicit``), which keeps thin films stable.

Unknown names fall back to ``Tsit5()`` with a warning.

Output is saved at ``t0 : (t1 - t0)/nSave : t1`` (nSave + 1 frames).

Gradients
---------
Reverse-mode gradients use diffrax's ``RecursiveCheckpointAdjoint`` with a fixed number
of checkpoints (``ODESettings.checkpoints``): the backward pass recomputes the steps
between checkpoints, so memory is bounded by the checkpoint count. Forward mode uses
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

# Solvers without an error estimator: always run with fixed steps.
_FIXED_ONLY = ("Euler()", "RK4()")
# Solvers handled outside ``_SOLVERS``.
_SOLVERS_EXTRA = ("SSPRK3()",)
# Julia-style name -> diffrax solver class (RK4 is built from its tableau in ``_solver``).
_SOLVERS = {
    "Tsit5()": dfx.Tsit5,
    "Euler()": dfx.Euler,
    "RK4()": None,
    "Heun()": dfx.Heun,
    "Dopri5()": dfx.Dopri5,
    "Bosh3()": dfx.Bosh3,
}


@dataclass(frozen=True)
class ODESettings:
    """Time-integration settings (times in s).

    Attributes:
        t0, t1: time span (s).
        dt: initial (adaptive) or fixed step (s).
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
    """Step size dt = cfl * dx / s_max(U) from the current state (not differentiated).

    Every step is accepted; the next step size follows from the new state.

    Attributes:
        cfl: Courant number (dimensionless).
        dx: length scale min(A/P) of the mesh (m), scalar.
        dt_max: upper bound on the step (s).
        speed: function (U, theta) -> largest wave speed |u| + sqrt(g h) (m/s), scalar.
    """

    cfl: float
    dx: Any
    dt_max: float
    speed: Callable = eqx.field(static=True)

    def wrap(self, direction):
        """No direction-dependent state; return the controller unchanged."""
        return self

    def _dt(self, y, args):
        """CFL step for state ``y`` and parameters ``args`` (s), scalar; gradients stopped."""
        s = jax.lax.stop_gradient(self.speed(y, args))
        dx = jax.lax.stop_gradient(self.dx)
        dt_cfl = self.cfl * dx / jnp.maximum(s, 1e-12)
        return jnp.minimum(dt_cfl, self.dt_max)

    def init(self, terms, t0, t1, y0, dt0, args, func, error_order):
        """First step end time t0 + dt(y0); no controller state."""
        return t0 + self._dt(y0, args), None

    def adapt_step_size(
        self, t0, t1, y0, y1_candidate, args, y_error, error_order, controller_state,
    ):
        """Accept the step and choose the next one from the new state.

        Returns:
            (accept, next t0, next t1, made_jump, controller_state, result), the diffrax
            step-controller protocol.
        """
        next_t1 = t1 + self._dt(y1_candidate, args)
        return True, t1, next_t1, False, None, dfx.RESULTS.successful


class SSPRK3PointImplicit(dfx.AbstractSolver):
    """Shu-Osher SSP-RK3 with point-implicit friction in every forward-Euler sub-step.

    Sub-step E (a point-implicit Euler step):

        W = U + dt F(U)                  (fluxes, bed and production terms)
        W <- W / (1 + dt c(U))           (linear sinks -c U, c in 1/s per row: friction
                                          on q, dissipation on h k and h epsilon)

    Unconditionally stable for friction (explicit friction needs dt c < 2, which fails on
    thin films: c ~ 24 /s for h = 1 mm, |u| = 1 m/s, n = 0.035), and friction can never
    reverse the flow. Stages:

        U1 = E(U)
        U2 = 3/4 U + 1/4 E(U1)
        U3 = 1/3 U + 2/3 E(U2)

    Attributes:
        split: function (t, U, theta) -> (F, c): dU/dt without the sinks, shape
            [n_rows, n_cells], and the sink rate of every row c (1/s), shape
            [n_rows, n_cells] (0 for xi).
    """

    split: Callable = eqx.field(static=True)       # (t, U, args) -> (F, c)
    term_structure: ClassVar = dfx.AbstractTerm
    interpolation_cls: ClassVar = dfx.LocalLinearInterpolation

    def order(self, terms):
        """Order of accuracy: 3."""
        return 3

    def init(self, terms, t0, t1, y0, args):
        """No solver state (returns None)."""
        return

    def _euler(self, t, U, args, dt):
        """Point-implicit Euler sub-step E(U) at time t (see the class docstring)."""
        F, c = self.split(t, U, args)
        W = U + dt * F
        return W / (1.0 + dt * c)

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        """One SSP-RK3 step from t0 to t1.

        Returns:
            (y1, error estimate None, dense info, solver state None, result).
        """
        dt = t1 - t0
        U1 = self._euler(t0, y0, args, dt)
        U2 = 0.75 * y0 + 0.25 * self._euler(t1, U1, args, dt)
        U3 = y0 / 3.0 + 2.0 / 3.0 * self._euler(t0 + 0.5 * dt, U2, args, dt)

        dense_info = {"y0": y0, "y1": U3}
        return U3, None, dense_info, None, dfx.RESULTS.successful

    def func(self, terms, t0, y0, args):
        """Vector field dU/dt of the term (used by diffrax for the initial step)."""
        return terms.vf(t0, y0, args)


def _ssprk3() -> dfx.AbstractSolver:
    """Shu-Osher SSP-RK3 (as an explicit RK tableau; identical updates).

    Returns:
        A diffrax explicit RK solver (no error estimate; use fixed or CFL steps).
    """
    tab = dfx.ButcherTableau(
        c=np.array([1.0, 0.5]),
        b_sol=np.array([1 / 6, 1 / 6, 2 / 3]),
        b_error=np.zeros(3),
        a_lower=(np.array([1.0]), np.array([0.25, 0.25])))

    class _SSPRK3(dfx.AbstractERK):
        """SSP-RK3 from the tableau above."""

        tableau = tab
        interpolation_cls = dfx.ThirdOrderHermitePolynomialInterpolation.from_k

        def order(self, terms: Any) -> int:
            """Order of accuracy: 3."""
            return 3

    return _SSPRK3()


def _solver(name: str) -> dfx.AbstractSolver:
    """diffrax solver for a Julia-style solver name.

    Args:
        name: e.g. ``"Tsit5()"``, ``"RK4()"``, ``"SSPRK3()"``.

    Returns:
        A diffrax solver instance (``Tsit5`` for unknown names).
    """
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
            """Classical RK4 from the tableau above."""

            tableau = tab
            interpolation_cls = dfx.ThirdOrderHermitePolynomialInterpolation.from_k

            def order(self, terms: Any) -> int:
                """Order of accuracy: 4."""
                return 4

        return _RK4()

    if name not in _SOLVERS:
        logger.warning("ODE solver %s not available; using Tsit5()", name)
        name = "Tsit5()"

    cls = _SOLVERS[name]
    assert cls is not None
    return cls()


def n_checkpoints(s: ODESettings, U0: Array) -> int:
    """Number of reverse-mode checkpoints for these settings and state size.

    Args:
        s: settings (``checkpoints``, ``checkpoint_memory_gb``, ``max_steps``).
        U0: a state, shape [3, n_cells] (gives the bytes per state).

    Returns:
        ``s.checkpoints`` if it is a number; for ``"auto"``,
        floor(budget / (3 * state bytes)) clipped to [16, max_steps].
    """
    if s.checkpoints != "auto":
        return int(s.checkpoints)

    state_bytes = U0.size * U0.dtype.itemsize
    n = int(s.checkpoint_memory_gb * 2 ** 30 // (3 * state_bytes))
    return max(16, min(n, s.max_steps))


def solve_swe(m: SWEModel, theta: dict[str, Any], U0: Array, s: ODESettings,
              ts: Array | None = None) -> dfx.Solution:
    """Integrate dU/dt = rhs(U; theta) from ``s.t0`` to ``s.t1``.

    Args:
        m: static model.
        theta: parameters (differentiable pytree).
        U0: initial state [3, n_cells].
        s: settings.
        ts: save times (s) (default: nSave + 1 uniform frames).

    Returns:
        diffrax Solution; ``sol.ys`` has shape [n_frames, 3, n_cells].
    """
    if ts is None:
        ts = jnp.linspace(s.t0, s.t1, s.n_save + 1)

    # ------------------------------------------------------------------
    # 1. Vector field dU/dt = R(U, t; theta); theta is passed as diffrax's ``args``.
    # ------------------------------------------------------------------
    def vector_field(t, y, args):
        """R(y, t; args), shape [3, n_cells]."""
        return rhs(m, args, y, None, t)

    term = dfx.ODETerm(vector_field)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # 2. Step-size control
    # ------------------------------------------------------------------
    ctrl: dfx.AbstractStepSizeController
    if s.solver == "SSPRK3()":
        # CFL-limited steps: length scale min(A/P), with P the cell perimeter summed over
        # interior faces (owner and neighbour side) and boundary faces.
        from hydrograd.physics.rhs_hr import max_eddy_viscosity, max_wave_speed

        perim_owner = jax.ops.segment_sum(m.fi_L, m.fi_o, m.n_cells)
        perim_neighbour = jax.ops.segment_sum(m.fi_L, m.fi_r, m.n_cells)
        perim_boundary = jax.ops.segment_sum(m.fb_L, m.fb_c, m.n_cells)
        perim = perim_owner + perim_neighbour + perim_boundary

        # Courant number: user value, or the positivity limit of the scheme.
        if s.cfl is not None:
            cfl = s.cfl
        elif m.order == 2 and m.scheme == "hr_hllc":
            cfl = 0.25
        else:
            cfl = 0.5

        length_scale = jnp.min(m.area / perim)
        diffusion_limited = m.turbulence.mode in ("mixing_length", "k_epsilon")

        def wave_speed(y, a):
            """Largest wave speed (m/s) of state y with parameters a.

            With the mixing-length or k-epsilon eddy viscosity (which can grow large in
            shear layers) the diffusion adds the speed 2 nu_max / dx, so that the step
            also satisfies the explicit diffusion limit dt <= cfl dx^2 / (2 nu_max).
            """
            speed = max_wave_speed(m, a, y)
            if diffusion_limited:
                speed = speed + 2.0 * max_eddy_viscosity(m, a, y) / length_scale
            return speed

        ctrl = CFLStepSize(cfl=cfl, dx=length_scale, dt_max=s.t1 - s.t0, speed=wave_speed)
    elif s.adaptive and s.solver in _FIXED_ONLY:
        logger.warning("%s has no error estimator; using fixed steps dt = %g s", s.solver, s.dt)

    if s.solver == "SSPRK3()":
        pass   # controller set above
    elif s.adaptive and s.solver not in _FIXED_ONLY:
        ctrl = dfx.PIDController(rtol=s.rtol, atol=s.atol)
    else:
        ctrl = dfx.ConstantStepSize()

    # ------------------------------------------------------------------
    # 3. Adjoint (differentiation) method
    # ------------------------------------------------------------------
    adjoints = {
        "checkpoint": dfx.RecursiveCheckpointAdjoint(checkpoints=n_checkpoints(s, U0)),
        "forward": dfx.ForwardMode(),
        "direct": dfx.DirectAdjoint(),
    }
    adjoint = adjoints[s.adjoint]

    # ------------------------------------------------------------------
    # 4. Solver; SSPRK3 with the HR-HLLC scheme treats friction point-implicitly
    # ------------------------------------------------------------------
    solver = _solver(s.solver)
    if s.solver == "SSPRK3()" and m.scheme == "hr_hllc":
        from hydrograd.physics.rhs import boundary_values_at
        from hydrograd.physics.rhs_hr import rhs_hr

        def split_rhs(t, y, a):
            """(flux-and-bed dU/dt, friction rate c) at time t with time-varying BCs."""
            return rhs_hr(m, boundary_values_at(m, a, t), y, None, split=True)

        solver = SSPRK3PointImplicit(split=split_rhs)

    # ------------------------------------------------------------------
    # 5. Integrate (no exception on failure: check ``sol.result``)
    # ------------------------------------------------------------------
    return dfx.diffeqsolve(term, solver, s.t0, s.t1, s.dt, U0, args=theta,
                           saveat=dfx.SaveAt(ts=ts), stepsize_controller=ctrl,
                           max_steps=s.max_steps, adjoint=adjoint, throw=False)
