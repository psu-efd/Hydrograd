"""Implicit theta-scheme time stepping with Newton and per-step IFT adjoints.

Each step solves, for U = U^{n+1},

    G(U) = U - U^n - dt [theta R(U, t^{n+1}) + (1 - theta) R(U^n, t^n)] = 0

by damped Newton with the exact sparse Jacobian dG/dU = I - theta dt (J_local + A C^T)
(:mod:`hydrograd.solvers.jacobian`; Woodbury + sparse LU on the host), re-formed every
iteration. (Modified Newton with the Jacobian frozen at U^n diverged on a rising flood with
floodplain wetting at dt = 10 s.)

theta = 1 (backward Euler, default): first order in time, unconditionally stable,
damps fast waves; suited to slowly varying floods with dt of tens of seconds.
theta = 1/2 (Crank-Nicolson): second order, less damping, can oscillate at bores.

Unlike the steady Newton solver, this can wet dry cells: the identity term keeps
their rows non-singular and inflow from wet neighbours enters G directly. Depths
are kept >= 0 after each Newton update.

Gradients (``jax.custom_vjp`` per step): with mu solving (dG/dU)^T mu = g,

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

Theta = dict[str, Any]


@dataclass(frozen=True)
class ImplicitSettings:
    """Theta-scheme settings.

    Attributes:
        t0, t1: time span (s); dt: fixed step (s), adjusted down to divide (t1 - t0) evenly.
        theta: implicitness (1 = backward Euler, 0.5 = Crank-Nicolson).
        newton_iter: maximum damped-Newton iterations per step.
        tol: Newton tolerance on max|G| (m and m^2/s per step).
        refresh_ratio: the Newton Jacobian is formed at U^n and re-formed only after an
            iteration whose residual reduction max|G|_new / max|G|_old exceeds this ratio
            (chord-type reuse; 0 = re-form every iteration). The factorization of a reused
            Jacobian is cached on the host. The adjoint always uses the exact Jacobian at U^{n+1}.
        accept_tol, max_halvings: a step whose Newton residual stays above ``accept_tol`` is
            redone as two half steps, recursively at most ``max_halvings`` times (default 0:
            fixed steps). Halving makes forward runs robust at advancing wet fronts, but the
            accept/subdivide decision depends on the parameters, so the loss becomes
            discontinuous: keep 0 for gradient-based work.
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
        """(total steps, dt (s), steps per save interval)."""
        per_save = max(1, int(np.ceil((self.t1 - self.t0) / self.n_save / self.dt)))
        n = per_save * self.n_save
        return n, (self.t1 - self.t0) / n, per_save


def _host_solve(m: SWEModel, comp: np.ndarray, A: np.ndarray, C: np.ndarray, b: np.ndarray,
                a: float, transpose: bool) -> np.ndarray:
    """Solve (I - a J) x = b, or its transpose; J = J_local + A C^T."""
    Jl = jacobian_pattern(m).assemble(np.asarray(comp))
    M = sp.identity(Jl.shape[0], format="csc") - float(a) * Jl
    b2 = np.asarray(b).reshape(Jl.shape[0], -1)
    A, C = np.asarray(A), np.asarray(C)
    if transpose:   # (I - a J_l^T - a C A^T) x = b
        x = _woodbury_solve(M.T, -float(a) * C, A, b2)
    else:           # (I - a J_l - a A C^T) x = b
        x = _woodbury_solve(M, -float(a) * A, C, b2)
    return x.reshape(np.shape(b)).astype(np.float64)


def _solve(m: SWEModel, lin, b: Array, a: Array, transpose: bool) -> Array:
    out = jax.ShapeDtypeStruct(b.shape, jnp.float64)
    return jax.pure_callback(lambda c, A, C, v, s: _host_solve(m, c, A, C, v, s, transpose),
                             out, *lin, b, a, vmap_method="sequential")


def _newton_step(m: SWEModel, s: ImplicitSettings, theta: Theta, U0: Array, t0: Array,
                 dt: Array) -> tuple[Array, Array]:
    """U^{n+1} from U^n by damped Newton (Jacobian re-formed every iteration).

    Damping: the update fraction alpha in {1, 1/2, 1/4, 1/8} with the smallest max|G| is
    taken. Stops when max|G| < tol or after ``newton_iter`` iterations.

    Returns:
        (U^{n+1}, final max|G|).
    """
    R0 = rhs(m, theta, U0, None, t0)
    t1 = t0 + dt
    xi_min = -still_depth(m, theta)                          # h >= 0
    a = s.theta * dt
    alphas = jnp.array([1.0, 0.5, 0.25, 0.125])

    def G(U):
        return U - U0 - dt * (s.theta * rhs(m, theta, U, None, t1) + (1.0 - s.theta) * R0)

    def clip(U):
        if not s.clip_depth:
            return U
        return U.at[0].set(jnp.maximum(U[0], xi_min))

    def cond(c):
        _, it, gn, _, _ = c
        return (gn > s.tol) & (it < s.newton_iter)

    def body(c):
        U, it, gn, lin, refresh = c
        lin = jax.lax.cond(refresh, lambda _: linearize(m, theta, U, t1), lambda _: lin, None)
        dU = _solve(m, lin, -G(U), a, transpose=False)
        cands = jax.vmap(lambda al: clip(U + al * dU))(alphas)
        norms = jax.vmap(lambda V: jnp.max(jnp.abs(G(V))))(cands)
        norms = jnp.where(jnp.isfinite(norms), norms, jnp.inf)
        k = jnp.argmin(norms)
        # Chord-type reuse: refresh the Jacobian only when the residual fell by less than
        # ``refresh_ratio`` (always when s.refresh_ratio >= 1).
        refresh_next = norms[k] > s.refresh_ratio * gn
        return cands[k], it + 1, norms[k], lin, refresh_next

    lin0 = linearize(m, theta, U0, t1)
    U, _, gn, _, _ = jax.lax.while_loop(
        cond, body, (U0, jnp.asarray(0), jnp.max(jnp.abs(G(U0))), lin0, jnp.asarray(False)))
    return U, gn


@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def implicit_step(m: SWEModel, s: ImplicitSettings, theta: Theta, U0: Array, t0: Array,
                  dt: Array) -> tuple[Array, Array]:
    """One theta-scheme step: (U^{n+1}, Newton residual), differentiable by the IFT."""
    return _newton_step(m, s, theta, U0, t0, dt)


def _fwd(m, s, theta, U0, t0, dt):
    U1, gn = _newton_step(m, s, theta, U0, t0, dt)
    return (U1, gn), (theta, U0, U1, t0, dt)


def _bwd(m, s, res, cot):
    g, _ = cot                     # the residual norm output carries no gradient
    theta, U0, U1, t0, dt = res
    t1 = t0 + dt
    mu = _solve(m, linearize(m, theta, U1, t1), g, s.theta * dt, transpose=True)
    _, vjp1 = jax.vjp(lambda th: rhs(m, th, U1, None, t1), theta)
    _, vjp0 = jax.vjp(lambda th, U: rhs(m, th, U, None, t0), theta, U0)
    (g_th1,) = vjp1(mu)
    g_th0, g_U0 = vjp0(mu)
    w1, w0 = s.theta * dt, (1.0 - s.theta) * dt
    g_theta = jax.tree.map(lambda a, b: w1 * a + w0 * b, g_th1, g_th0)
    return g_theta, mu + w0 * g_U0, jnp.zeros_like(t0), jnp.zeros_like(dt)


implicit_step.defvjp(_fwd, _bwd)


def adaptive_step(m: SWEModel, s: ImplicitSettings, theta: Theta, U0: Array, t0: Array, dt: Array,
                  level: int = 0) -> Array:
    """One step of dt; if Newton does not reach ``accept_tol``, redo it as two half steps,
    recursively up to ``max_halvings`` levels (dt / 2^max_halvings). Linear trace size."""
    U1, gn = implicit_step(m, s, theta, U0, t0, dt)
    if level >= s.max_halvings:
        return U1

    def refine(_):
        half = 0.5 * dt
        return jax.lax.fori_loop(
            0, 2, lambda i, U: adaptive_step(m, s, theta, U, t0 + i * half, half, level + 1), U0)

    return jax.lax.cond(gn <= s.accept_tol, lambda _: U1, refine, None)


def solve_implicit(m: SWEModel, theta: Theta, U0: Array, s: ImplicitSettings) -> tuple[Array, Array]:
    """Integrate with the theta-scheme; returns (save times [n_save+1], states [n_save+1, 3, n])."""
    n, dt, per_save = s.steps()
    dt_a = jnp.asarray(dt)

    def inner(U, k):
        t = s.t0 + k * dt_a
        return adaptive_step(m, s, theta, U, t, dt_a), None

    def outer(U, j):
        ks = j * per_save + jnp.arange(per_save)
        U, _ = jax.lax.scan(jax.checkpoint(inner), U, ks)
        return U, U

    _, ys = jax.lax.scan(outer, U0, jnp.arange(s.n_save))
    ts = s.t0 + dt * per_save * jnp.arange(s.n_save + 1)
    return ts, jnp.concatenate([U0[None], ys], axis=0)
