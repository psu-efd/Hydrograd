"""Steady states R(U; theta) = 0 by Newton, with implicit-function-theorem gradients.

Forward: pseudo-transient continuation (switched evolution relaxation),

    (I/dtau - J) dU = R(U),   dtau_k = dtau_0 (|R_0| / |R_k|)^p,

with the exact sparse Jacobian J = dR/dU (coloring + JVPs). The Newton loop
runs in JAX (``lax.while_loop``); only the sparse LU solve is done on the host
(SciPy, through ``jax.pure_callback``), so the whole solve can sit inside a
jitted loss.

Gradient (``jax.custom_vjp``): at the converged state, dU*/dtheta = -J^{-1} R_theta,
so for a cotangent g = dL/dU*:

    J^T lambda = g,     dL/dtheta = -lambda^T dR/dtheta     (one transposed solve).

Only reverse mode is supported (the IFT adjoint costs one solve for any number of
parameters). The closure must be local (no whole-array LayerNorm), so that J is sparse.
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
import scipy.sparse.linalg as spla
from jax import Array

from hydrograd.physics.model import SWEModel
from hydrograd.physics.rhs import rhs
from hydrograd.solvers.jacobian import jacobian_pattern, linearize

logger = logging.getLogger(__name__)

__all__ = ["NewtonSettings", "newton_solve", "steady_state"]

Theta = dict[str, Any]


@dataclass(frozen=True)
class NewtonSettings:
    """Pseudo-transient Newton settings.

    Attributes:
        dtau0: initial pseudo-time step (s).
        dtau_max: cap on the pseudo-time step (s); large -> pure Newton.
        power: SER exponent p.
        tol: convergence on max|dU/dt| (m/s and m^2/s^2).
        max_iter: iteration cap.
        max_growth: a step is rejected if it raises max|R| by more than this factor
            (or makes it non-finite); the pseudo-time step is then multiplied by ``shrink``.
    """

    dtau0: float = 1.0
    dtau_max: float = 1.0e12
    power: float = 1.5
    tol: float = 1.0e-10
    max_iter: int = 300
    max_growth: float = 10.0
    shrink: float = 0.1


try:  # optional multithreaded direct solver (Intel MKL PARDISO)
    import pypardiso as _pardiso
except ImportError:  # pragma: no cover - optional dependency
    _pardiso = None

PARDISO_MIN_ROWS = 30_000  # below this SciPy's SuperLU is as fast


_FACTOR_CACHE: "dict[bytes, Any]" = {}
_FACTOR_CACHE_SIZE = 4


def _factorized(M: sp.spmatrix):
    """Cached SuperLU factorization of M (keyed by a hash of its values and pattern)."""
    import hashlib

    Mc = M.tocsc()
    h = hashlib.blake2b(Mc.data.tobytes(), digest_size=16)
    h.update(Mc.indices.tobytes())
    key = h.digest()
    lu = _FACTOR_CACHE.get(key)
    if lu is None:
        lu = spla.splu(Mc)
        if len(_FACTOR_CACHE) >= _FACTOR_CACHE_SIZE:
            _FACTOR_CACHE.pop(next(iter(_FACTOR_CACHE)))
        _FACTOR_CACHE[key] = lu
    return lu


def _sparse_solve(M: sp.spmatrix, B: np.ndarray) -> np.ndarray:
    """Solve M X = B (B: [N] or [N, k]) with one factorization (PARDISO if available).

    A singular or non-finite system returns NaN instead of raising: the error then
    propagates as a non-finite state/loss (which optimizers and step control can reject)
    rather than aborting the whole traced computation inside a host callback.
    """
    if not (np.all(np.isfinite(M.data)) and np.all(np.isfinite(B))):
        return np.full(np.shape(B), np.nan)
    try:
        if _pardiso is not None and M.shape[0] >= PARDISO_MIN_ROWS:
            return _pardiso.spsolve(M.tocsr(), np.ascontiguousarray(B))
        return _factorized(M).solve(B)
    except (RuntimeError, ValueError) as e:
        logger.warning("sparse solve failed (%s); returning NaN", e)
        return np.full(np.shape(B), np.nan)


def _woodbury_solve(M: sp.spmatrix, Uw: np.ndarray, Vw: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve (M + Uw Vw^T) x = b with one sparse factorization of M (Woodbury identity)."""
    if not np.any(Uw):
        return _sparse_solve(M, b)
    k = b.shape[1]
    X = _sparse_solve(M, np.hstack([b, Uw]))            # [M^{-1} b, M^{-1} Uw]
    x, MiU = X[:, :k], X[:, k:]
    small = np.eye(Uw.shape[1]) + Vw.T @ MiU             # [r, r]
    return x - MiU @ np.linalg.solve(small, Vw.T @ x)


def _host_solve(m: SWEModel, comp: np.ndarray, A: np.ndarray, C: np.ndarray, rhs_vec: np.ndarray,
                shift: float, transpose: bool) -> np.ndarray:
    """Solve (shift I - J) x = b, or J^T x = b if ``transpose`` (host, SciPy).

    J = J_local + A C^T (see ``solvers.jacobian``); b may be a vector or a matrix.
    """
    Jl = jacobian_pattern(m).assemble(np.asarray(comp))
    A, C = np.asarray(A), np.asarray(C)
    b = np.asarray(rhs_vec)
    b2 = b.reshape(Jl.shape[0], -1)
    if transpose:   # (J_l^T + C A^T) x = b
        x = _woodbury_solve(Jl.T, C, A, b2)
    else:           # (shift I - J_l - A C^T) x = b
        M = sp.identity(Jl.shape[0], format="csc") * float(shift) - Jl
        x = _woodbury_solve(M, -A, C, b2)
    return x.reshape(b.shape).astype(np.float64)


def _solve(m: SWEModel, lin: tuple[Array, Array, Array], b: Array, shift: Array, transpose: bool) -> Array:
    out = jax.ShapeDtypeStruct(b.shape, jnp.float64)
    return jax.pure_callback(lambda c, a, cc, v, s: _host_solve(m, c, a, cc, v, s, transpose),
                             out, *lin, b, shift, vmap_method="sequential")


def newton_solve(m: SWEModel, theta: Theta, U0: Array, s: NewtonSettings) -> tuple[Array, Array, Array]:
    """Run the Newton loop (no custom gradient).

    Returns:
        (U*, iterations, final max|R|).
    """
    f = lambda U: rhs(m, theta, U)
    hfloor = m.h_small if m.scheme == "julia_xi_roe" else 0.0     # depth floor of the scheme
    hs = m.hstill if "zb" not in theta else m.wstill - theta["zb"]
    xi_min = hfloor - hs

    def cond(c):
        _, it, rn, _, _ = c
        return (rn > s.tol) & (it < s.max_iter)

    def body(c):
        U, it, rn, r0, fac = c
        R = f(U)
        lin = linearize(m, theta, U)
        dtau = jnp.minimum(fac * s.dtau0 * (r0 / jnp.maximum(rn, 1e-300)) ** s.power, s.dtau_max)
        dU = _solve(m, lin, R, 1.0 / dtau, transpose=False)
        U_new = U + dU
        U_new = U_new.at[0].set(jnp.maximum(U_new[0], xi_min))   # keep h >= floor
        rn_new = jnp.max(jnp.abs(f(U_new)))
        # Safeguard: reject steps that blow up the residual and shrink the pseudo-time step.
        ok = jnp.isfinite(rn_new) & (rn_new < s.max_growth * rn)
        U = jnp.where(ok, U_new, U)
        rn = jnp.where(ok, rn_new, rn)
        fac = jnp.where(ok, jnp.minimum(2.0 * fac, 1.0), fac * s.shrink)   # recover after accepted steps
        return U, it + 1, rn, r0, fac

    r0 = jnp.max(jnp.abs(f(U0)))
    U, it, rn, _, _ = jax.lax.while_loop(cond, body, (U0, jnp.asarray(0), r0, r0, jnp.asarray(1.0)))
    return U, it, rn


@partial(jax.custom_vjp, nondiff_argnums=(0, 3))
def steady_state(m: SWEModel, theta: Theta, U0: Array, s: NewtonSettings) -> Array:
    """Steady state U*(theta) with IFT gradients w.r.t. ``theta`` (U0 is a guess)."""
    return newton_solve(m, theta, U0, s)[0]


def _fwd(m: SWEModel, theta: Theta, U0: Array, s: NewtonSettings):
    U = newton_solve(m, theta, U0, s)[0]
    return U, (theta, U, U0)


def _bwd(m: SWEModel, s: NewtonSettings, res, g: Array):
    theta, U, U0 = res
    lam = _solve(m, linearize(m, theta, U), g, jnp.asarray(0.0), transpose=True)
    _, vjp_theta = jax.vjp(lambda th: rhs(m, th, U), theta)
    (g_theta,) = vjp_theta(-lam)
    return g_theta, jnp.zeros_like(U0)


steady_state.defvjp(_fwd, _bwd)


def steady_jacobian(m: SWEModel, theta: Theta, U: Array) -> np.ndarray:
    """Sensitivities dU*/dtheta = -J^{-1} dR/dtheta at a steady state (host, one LU).

    Args:
        m: model; theta: parameters (a dict with one flat array, e.g. {"n_zone": p});
        U: converged steady state [3, n_cells].

    Returns:
        [3 n_cells, n_params] array (flat state index k * n_cells + i).
    """
    (key, p), = theta.items()
    comp, A, C = jax.jit(lambda U: linearize(m, theta, U))(U)
    R_theta = np.asarray(jax.jacfwd(lambda q: rhs(m, {key: q}, U).reshape(-1))(p))
    # J x = -R_theta  <=>  (0 I - J) x = R_theta
    return _host_solve(m, comp, A, C, R_theta, 0.0, transpose=False)
