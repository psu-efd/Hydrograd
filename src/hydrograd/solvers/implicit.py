"""Implicit theta-scheme time stepping with Newton and per-step IFT adjoints.

Selected with ``ode_solver = "BackwardEuler()"`` (theta = 1) or ``"CrankNicolson()"``
(theta = 1/2); see ``solvers.final``.

Theta-scheme residual
---------------------
With R(U, t) = dU/dt the semi-discrete residual, each step solves, for U = U^{n+1},

    G(U) = U - U^n - dt [theta R(U, t^{n+1}) + (1 - theta) R(U^n, t^n)] = 0.

G has the units of U (m for xi, m^2/s for q_x, q_y).

- theta = 1 (backward Euler, default): first order in time, unconditionally stable,
  damps fast waves; suited to slowly varying floods with dt of tens of seconds.
- theta = 1/2 (Crank-Nicolson): second order, less damping, can oscillate at bores.

Newton iteration
----------------
G(U) = 0 is solved by damped Newton with the exact sparse Jacobian

    dG/dU = I - theta dt (J_local + A C^T)

(:mod:`hydrograd.solvers.jacobian`; Woodbury + sparse LU on the host). Each iteration
solves (dG/dU) dU = -G(U) and takes the update fraction alpha in {1, 1/2, 1/4, 1/8}
with the smallest max|G|. The Jacobian is reused while the residual falls fast enough
(chord-type reuse, ``refresh_ratio``) and re-formed otherwise. (Modified Newton with the
Jacobian frozen at U^n for the whole step diverged on a rising flood with floodplain
wetting at dt = 10 s.)

Unlike the steady Newton solver, this can wet dry cells: the identity term keeps
their rows non-singular and inflow from wet neighbours enters G directly. Depths can
optionally be clipped to >= 0 after each Newton update (``clip_depth``).

Gradients: implicit function theorem per step
---------------------------------------------
``implicit_step`` is a ``jax.custom_vjp``. Since G(U^{n+1}; U^n, theta) = 0, with mu
solving (dG/dU)^T mu = g (g = dL/dU^{n+1}):

    dL/dU^n   = mu + (1 - theta) dt J_R(U^n)^T mu,
    dL/dtheta = dt [theta dR(U^{n+1})/dtheta + (1 - theta) dR(U^n)/dtheta]^T mu,

one transposed solve and two VJPs per step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import Array

from hydrograd.physics.model import SWEModel
from hydrograd.physics.rhs import rhs, still_depth
from hydrograd.solvers.jacobian import jacobian_pattern, linearize
from hydrograd.solvers.steady import _woodbury_solve

logger = logging.getLogger(__name__)

__all__ = ["ImplicitSettings", "implicit_step", "solve_implicit"]

# Differentiable parameters, e.g. {"n_zone": ..., "zb": ..., "Q": ..., "nn": ...}.
Theta = dict[str, Any]


@dataclass(frozen=True)
class ImplicitSettings:
    """Theta-scheme settings.

    Attributes:
        t0, t1: time span (s); dt: fixed step (s), adjusted down to divide (t1 - t0) evenly.
        theta: implicitness (1 = backward Euler, 0.5 = Crank-Nicolson).
        newton_iter: maximum damped-Newton iterations per step.
        tol: Newton tolerance on max|G| (m and m^2/s per step).
        accept_tol, max_halvings: a step whose Newton residual stays above ``accept_tol`` is
            redone as two half steps, recursively at most ``max_halvings`` times (default 0:
            fixed steps). Halving makes forward runs robust at advancing wet fronts, but the
            accept/subdivide decision depends on the parameters, so the loss becomes
            discontinuous: keep 0 for gradient-based work.
        refresh_ratio: the Newton Jacobian is formed at U^n and re-formed only after an
            iteration whose residual reduction max|G|_new / max|G|_old exceeds this ratio
            (chord-type reuse; 0 = re-form every iteration). The factorization of a reused
            Jacobian is cached on the host. The adjoint always uses the exact Jacobian at U^{n+1}.
        clip_depth: clip h >= 0 after each Newton update. Off by default: a clipped state no
            longer solves G = 0, which breaks mass conservation and the IFT adjoint; the
            residual already treats h = max(xi + h_still, 0), so small negative states are inert.
        n_save: number of save intervals (n_save + 1 frames incl. t0).
    """

    t0: float
    t1: float
    dt: float
    theta: float = 1.0
    newton_iter: int = 30
    tol: float = 1.0e-8
    accept_tol: float = 1.0e-6
    refresh_ratio: float = 0.3
    clip_depth: bool = False
    max_halvings: int = 0
    n_save: int = 1

    def steps(self) -> tuple[int, float, int]:
        """Step count and step size that divide every save interval evenly.

        Returns:
            (total steps, dt (s), steps per save interval).
        """
        # Steps per save interval: enough that the step does not exceed the requested dt.
        save_interval = (self.t1 - self.t0) / self.n_save
        per_save = max(1, int(np.ceil(save_interval / self.dt)))

        n = per_save * self.n_save
        return n, (self.t1 - self.t0) / n, per_save


# ----------------------------------------------------------------------
# Host linear solve with (I - a J)
# ----------------------------------------------------------------------
def _host_solve(m: SWEModel, comp: np.ndarray, A: np.ndarray, C: np.ndarray, b: np.ndarray,
                a: float, transpose: bool) -> np.ndarray:
    """Solve (I - a J) x = b, or its transpose; J = J_local + A C^T.

    Args:
        m: the model (selects the cached Jacobian pattern).
        comp: compressed JVPs of J_local, shape [3 n_colors, 3, n].
        A: dR/dK, shape [3n, n_in].
        C: (dK/dU)^T, shape [3n, n_in].
        b: right-hand side, shape [3, n].
        a: theta dt (s).
        transpose: solve (I - a J)^T x = b (adjoint).

    Returns:
        Solution x, float64, same shape as ``b``.
    """
    # M = I - a J_local (sparse part of dG/dU).
    Jl = jacobian_pattern(m).assemble(np.asarray(comp))
    M = sp.identity(Jl.shape[0], format="csc") - float(a) * Jl
    b2 = np.asarray(b).reshape(Jl.shape[0], -1)
    A = np.asarray(A)
    C = np.asarray(C)

    # Low-rank inlet term -a A C^T (or its transpose) by the Woodbury identity.
    if transpose:   # (I - a J_l^T - a C A^T) x = b
        x = _woodbury_solve(M.T, -float(a) * C, A, b2)
    else:           # (I - a J_l - a A C^T) x = b
        x = _woodbury_solve(M, -float(a) * A, C, b2)

    return x.reshape(np.shape(b)).astype(np.float64)


def _solve(m: SWEModel, lin, b: Array, a: Array, transpose: bool) -> Array:
    """JAX wrapper of ``_host_solve`` (host callback; traceable, not differentiable).

    Args:
        m: the model.
        lin: linearization (comp, A, C) from ``solvers.jacobian.linearize``.
        b: right-hand side, shape [3, n].
        a: theta dt (s), scalar.
        transpose: solve the transposed system.

    Returns:
        Solution x, float64, same shape as ``b``.
    """
    out = jax.ShapeDtypeStruct(b.shape, jnp.float64)

    def host_callback(c, A, C, v, s):
        """Host-side call with NumPy arguments (comp, A, C, b, a)."""
        return _host_solve(m, c, A, C, v, s, transpose)

    return jax.pure_callback(host_callback, out, *lin, b, a, vmap_method="sequential")


# ----------------------------------------------------------------------
# One step: damped Newton, wrapped in an IFT custom VJP
# ----------------------------------------------------------------------
def _newton_step(m: SWEModel, s: ImplicitSettings, theta: Theta, U0: Array, t0: Array,
                 dt: Array) -> tuple[Array, Array]:
    """U^{n+1} from U^n by damped Newton (Jacobian reused while convergence is fast).

    Damping: the update fraction alpha in {1, 1/2, 1/4, 1/8} with the smallest max|G| is
    taken. Stops when max|G| < tol or after ``newton_iter`` iterations.

    Args:
        m: the model.
        s: theta-scheme settings.
        theta: differentiable parameters.
        U0: state U^n, shape [3, n_cells].
        t0: time t^n (s), scalar.
        dt: step size (s), scalar.

    Returns:
        (U^{n+1}, final max|G|): the state, shape [3, n_cells], and the Newton residual
        norm (m or m^2/s, scalar).
    """
    # Explicit part of G, fixed during the step.
    R0 = rhs(m, theta, U0, None, t0)
    t1 = t0 + dt

    xi_min = -still_depth(m, theta)                          # h >= 0
    a = s.theta * dt
    alphas = jnp.array([1.0, 0.5, 0.25, 0.125])

    def G(U):
        """Theta-scheme residual G(U), shape [3, n_cells] (units of U)."""
        return U - U0 - dt * (s.theta * rhs(m, theta, U, None, t1) + (1.0 - s.theta) * R0)

    def clip(U):
        """Optionally clip xi so that h >= 0 (``s.clip_depth``)."""
        if U.shape[0] > 3:
            U = U.at[3:].set(jnp.maximum(U[3:], 0.0))      # h k, h epsilon >= 0
        if not s.clip_depth:
            return U
        return U.at[0].set(jnp.maximum(U[0], xi_min))

    def cond(c):
        """Continue while not converged and below the iteration cap."""
        _, it, gn, _, _ = c
        return (gn > s.tol) & (it < s.newton_iter)

    def body(c):
        """One damped Newton iteration.

        Carry: (U, iteration, max|G(U)|, linearization (comp, A, C), refresh flag).
        """
        U, it, gn, lin, refresh = c

        # 1. Re-form the Jacobian at U if the previous iteration asked for it.
        def relinearize(_):
            """Linearization of R at the current U and t^{n+1}."""
            return linearize(m, theta, U, t1)

        def keep(_):
            """Reuse the previous linearization."""
            return lin

        lin = jax.lax.cond(refresh, relinearize, keep, None)

        # 2. Newton direction: (I - theta dt J) dU = -G(U).
        dU = _solve(m, lin, -G(U), a, transpose=False)

        # 3. Damping: try four update fractions and keep the one with the smallest max|G|.
        def candidate(al):
            """Updated state U + alpha dU (clipped if requested)."""
            return clip(U + al * dU)

        def residual_norm(V):
            """max|G(V)| (scalar)."""
            return jnp.max(jnp.abs(G(V)))

        cands = jax.vmap(candidate)(alphas)
        norms = jax.vmap(residual_norm)(cands)
        norms = jnp.where(jnp.isfinite(norms), norms, jnp.inf)
        k = jnp.argmin(norms)

        # 4. Chord-type reuse: refresh the Jacobian only when the residual fell by less than
        #    ``refresh_ratio`` (always when s.refresh_ratio >= 1).
        refresh_next = norms[k] > s.refresh_ratio * gn

        return cands[k], it + 1, norms[k], lin, refresh_next

    # Initial linearization at U^n (time t^{n+1}), then iterate.
    lin0 = linearize(m, theta, U0, t1)
    init = (U0, jnp.asarray(0), jnp.max(jnp.abs(G(U0))), lin0, jnp.asarray(False))
    U, _, gn, _, _ = jax.lax.while_loop(cond, body, init)

    return U, gn


@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def implicit_step(m: SWEModel, s: ImplicitSettings, theta: Theta, U0: Array, t0: Array,
                  dt: Array) -> tuple[Array, Array]:
    """One theta-scheme step: (U^{n+1}, Newton residual), differentiable by the IFT.

    Args:
        m: the model (static).
        s: theta-scheme settings (static).
        theta: differentiable parameters.
        U0: state U^n, shape [3, n_cells].
        t0: time t^n (s), scalar (zero gradient).
        dt: step size (s), scalar (zero gradient).

    Returns:
        (U^{n+1}, final max|G|); the residual norm carries no gradient.
    """
    return _newton_step(m, s, theta, U0, t0, dt)


def _fwd(m, s, theta, U0, t0, dt):
    """Forward pass of ``implicit_step``: Newton solve, save (theta, U^n, U^{n+1}, t^n, dt)."""
    U1, gn = _newton_step(m, s, theta, U0, t0, dt)
    return (U1, gn), (theta, U0, U1, t0, dt)


def _bwd(m, s, res, cot):
    """Backward pass of ``implicit_step`` (IFT adjoint, see the module docstring).

    Args:
        m: the model.
        s: theta-scheme settings.
        res: residuals (theta, U^n, U^{n+1}, t^n, dt) saved by ``_fwd``.
        cot: cotangents (dL/dU^{n+1}, dL/d residual norm); the second is ignored.

    Returns:
        (dL/dtheta, dL/dU^n, 0 for t^n, 0 for dt).
    """
    g, _ = cot                     # the residual norm output carries no gradient
    theta, U0, U1, t0, dt = res
    t1 = t0 + dt

    # Adjoint solve (dG/dU)^T mu = g at U^{n+1}.
    mu = _solve(m, linearize(m, theta, U1, t1), g, s.theta * dt, transpose=True)

    # VJPs of R at the new state (w.r.t. theta) and at the old state (w.r.t. theta, U^n).
    def residual_new(th):
        """R(U^{n+1}, t^{n+1}; th)."""
        return rhs(m, th, U1, None, t1)

    def residual_old(th, U):
        """R(U, t^n; th)."""
        return rhs(m, th, U, None, t0)

    _, vjp1 = jax.vjp(residual_new, theta)
    _, vjp0 = jax.vjp(residual_old, theta, U0)
    (g_th1,) = vjp1(mu)
    g_th0, g_U0 = vjp0(mu)

    # Weights theta dt (new state) and (1 - theta) dt (old state).
    w1 = s.theta * dt
    w0 = (1.0 - s.theta) * dt

    def combine(a, b):
        """w1 a + w0 b for one parameter leaf."""
        return w1 * a + w0 * b

    g_theta = jax.tree.map(combine, g_th1, g_th0)

    return g_theta, mu + w0 * g_U0, jnp.zeros_like(t0), jnp.zeros_like(dt)


implicit_step.defvjp(_fwd, _bwd)


# ----------------------------------------------------------------------
# Step subdivision and the time loop
# ----------------------------------------------------------------------
def adaptive_step(m: SWEModel, s: ImplicitSettings, theta: Theta, U0: Array, t0: Array, dt: Array,
                  level: int = 0) -> Array:
    """One step of dt, subdivided where Newton does not converge.

    If Newton does not reach ``accept_tol``, redo the step as two half steps,
    recursively up to ``max_halvings`` levels (dt / 2^max_halvings). Linear trace size.

    Args:
        m: the model.
        s: theta-scheme settings.
        theta: differentiable parameters.
        U0: state at t0, shape [3, n_cells].
        t0: start time (s), scalar.
        dt: step size (s), scalar.
        level: current halving depth (static Python int).

    Returns:
        State at t0 + dt, shape [3, n_cells].
    """
    U1, gn = implicit_step(m, s, theta, U0, t0, dt)
    if level >= s.max_halvings:
        return U1

    def refine(_):
        """Two half steps, each one subdivided further if needed."""
        half = 0.5 * dt

        def half_step(i, U):
            """Half step number i (0 or 1)."""
            return adaptive_step(m, s, theta, U, t0 + i * half, half, level + 1)

        return jax.lax.fori_loop(0, 2, half_step, U0)

    def accept(_):
        """Keep the full step."""
        return U1

    return jax.lax.cond(gn <= s.accept_tol, accept, refine, None)


def solve_implicit(
    m: SWEModel, theta: Theta, U0: Array, s: ImplicitSettings,
) -> tuple[Array, Array]:
    """Integrate with the theta-scheme from ``s.t0`` to ``s.t1``.

    Each save interval is one ``lax.scan`` over its steps, rematerialized
    (``jax.checkpoint``) in the backward pass, so reverse-mode memory scales with the
    number of saved frames rather than the number of steps.

    Args:
        m: the model.
        theta: differentiable parameters.
        U0: initial state, shape [3, n_cells].
        s: theta-scheme settings.

    Returns:
        (save times (s), shape [n_save + 1]; states, shape [n_save + 1, 3, n_cells]).
    """
    _n_steps, dt, per_save = s.steps()
    dt_a = jnp.asarray(dt)

    def inner(U, k):
        """Time step number k (global index)."""
        t = s.t0 + k * dt_a
        return adaptive_step(m, s, theta, U, t, dt_a), None

    def outer(U, j):
        """Save interval number j: ``per_save`` steps; the end state is saved."""
        ks = j * per_save + jnp.arange(per_save)
        U, _ = jax.lax.scan(jax.checkpoint(inner), U, ks)
        return U, U

    _, ys = jax.lax.scan(outer, U0, jnp.arange(s.n_save))

    # Save times and frames, the initial state included.
    ts = s.t0 + dt * per_save * jnp.arange(s.n_save + 1)
    return ts, jnp.concatenate([U0[None], ys], axis=0)
