"""Steady states R(U; theta) = 0 by Newton, with implicit-function-theorem gradients.

Here R(U; theta) = dU/dt is the semi-discrete residual (``physics.rhs``) and U the state
[xi, q_x, q_y], shape [3, n_cells]. A steady state is a root U* of R.

Forward solve: pseudo-transient continuation
--------------------------------------------
Plain Newton (J dU = -R) diverges from a poor initial guess, so each iteration solves an
implicit pseudo-time step of size dtau (switched evolution relaxation, SER):

    (I/dtau - J) dU = R(U),     U <- U + dU,
    dtau_k = min(f_k dtau_0 (|R_0| / |R_k|)^p, dtau_max),

with J = dR/dU the exact sparse Jacobian (graph coloring + JVPs, ``solvers.jacobian``),
|R| = max|R| the residual norm, p the SER exponent and f_k <= 1 a safeguard factor. As
the residual falls, dtau grows and the iteration becomes pure Newton (quadratic
convergence). A step that makes the residual non-finite or raises it by more than a
factor ``max_growth`` is rejected and f_k is cut by ``shrink``; f_k doubles back towards
1 after accepted steps.

The Newton loop runs in JAX (``lax.while_loop``); only the sparse LU solve is done on the
host (SciPy, through ``jax.pure_callback``), so the whole solve can sit inside a jitted
loss.

Inlet coupling: Woodbury identity
---------------------------------
The Jacobian is J = J_local + A C^T (sparse plus rank n_in, see ``solvers.jacobian``).
For a sparse M and thin dense U_w, V_w,

    (M + U_w V_w^T)^{-1} b = x - M^{-1} U_w (I + V_w^T M^{-1} U_w)^{-1} V_w^T x,
    x = M^{-1} b,

so one sparse factorization of M (applied to [b, U_w]) plus a small r x r dense solve
gives the exact solution.

Gradient: implicit function theorem
-----------------------------------
``steady_state`` is a ``jax.custom_vjp``. At the converged state R(U*, theta) = 0, so
dU*/dtheta = -J^{-1} R_theta. For a cotangent g = dL/dU*:

    J^T lambda = g,     dL/dtheta = -lambda^T dR/dtheta     (one transposed solve).

Only reverse mode is supported (the IFT adjoint costs one solve for any number of
parameters). The closure must be local (no whole-array LayerNorm), so that J is sparse.

Units: dtau in s; R in m/s (xi row) and m^2/s^2 (momentum rows).
"""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
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

# Differentiable parameters, e.g. {"n_zone": ..., "zb": ..., "Q": ..., "nn": ...}.
Theta = dict[str, Any]


@dataclass(frozen=True)
class NewtonSettings:
    """Pseudo-transient Newton settings.

    Attributes:
        dtau0: initial pseudo-time step dtau_0 (s).
        dtau_max: cap on the pseudo-time step (s); large -> pure Newton.
        power: SER exponent p (dimensionless).
        tol: convergence on max|dU/dt| (m/s and m^2/s^2).
        max_iter: iteration cap.
        max_growth: a step is rejected if it raises max|R| by more than this factor
            (or makes it non-finite); the pseudo-time step is then multiplied by ``shrink``.
        shrink: factor (< 1) applied to the pseudo-time step after a rejected step.
    """

    dtau0: float = 1.0
    dtau_max: float = 1.0e12
    power: float = 1.5
    tol: float = 1.0e-10
    max_iter: int = 300
    max_growth: float = 10.0
    shrink: float = 0.1


# ----------------------------------------------------------------------
# Host-side sparse linear algebra (SciPy / PARDISO)
# ----------------------------------------------------------------------
try:  # optional multithreaded direct solver (Intel MKL PARDISO)
    import pypardiso as _pardiso
except ImportError:  # pragma: no cover - optional dependency
    _pardiso = None

PARDISO_MIN_ROWS = 30_000  # below this SciPy's SuperLU is as fast


# Cache of SuperLU factorizations, keyed by a hash of the matrix (oldest evicted first).
# A reused (chord) Jacobian in the implicit stepper is then factorized only once.
_FACTOR_CACHE: dict[bytes, Any] = {}
_FACTOR_CACHE_SIZE = 4
# All host factorizations and solves run on this one persistent Python thread. XLA calls
# host callbacks from C++ worker threads, each with a temporary Python thread state, and
# SciPy's SuperLU wrapper tracks its allocations per thread state: a cached SuperLU object
# freed in a later callback is never released (~3 MB leaked per factorization; implicit
# Savannah inversions ran out of 48 GB in 40 min). On a persistent thread the cached
# factorizations are created and freed under one thread state.
_SOLVER_THREAD = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hydrograd-solve")


def _factorized(M: sp.spmatrix) -> Any:
    """Cached SuperLU factorization of M (keyed by a hash of its values and pattern).

    Call only on ``_SOLVER_THREAD`` (via ``_sparse_solve``).

    Args:
        M: square sparse matrix, shape [N, N].

    Returns:
        SciPy ``SuperLU`` object of M (its ``solve`` method applies M^{-1}).
    """
    Mc = M.tocsc()

    # Key: 128-bit hash of the values and the sparsity pattern.
    digest = hashlib.blake2b(Mc.data.tobytes(), digest_size=16)
    digest.update(Mc.indices.tobytes())
    digest.update(Mc.indptr.tobytes())
    key = digest.digest()

    # Factorize on a miss; evict the oldest entry when the cache is full.
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

    Args:
        M: square sparse matrix, shape [N, N].
        B: right-hand side(s), shape [N] or [N, k].

    Returns:
        Solution X, same shape as ``B`` (NaN if the solve failed).
    """
    # Run on the persistent solver thread (see the comment at ``_SOLVER_THREAD``).
    return _SOLVER_THREAD.submit(_sparse_solve_impl, M, B).result()


def _sparse_solve_impl(M: sp.spmatrix, B: np.ndarray) -> np.ndarray:
    """Body of ``_sparse_solve``; runs on ``_SOLVER_THREAD``.

    Args:
        M: square sparse matrix, shape [N, N].
        B: right-hand side(s), shape [N] or [N, k].

    Returns:
        Solution X, same shape as ``B`` (NaN if the inputs are non-finite or the
        solve failed).
    """
    # Non-finite input: return NaN without calling the factorization.
    if not (np.all(np.isfinite(M.data)) and np.all(np.isfinite(B))):
        return np.full(np.shape(B), np.nan)

    # PARDISO for large systems when installed, otherwise cached SuperLU.
    try:
        if _pardiso is not None and M.shape[0] >= PARDISO_MIN_ROWS:
            return _pardiso.spsolve(M.tocsr(), np.ascontiguousarray(B))
        return _factorized(M).solve(B)
    except (RuntimeError, ValueError) as e:
        logger.warning("sparse solve failed (%s); returning NaN", e)
        return np.full(np.shape(B), np.nan)


def _woodbury_solve(M: sp.spmatrix, Uw: np.ndarray, Vw: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve (M + Uw Vw^T) x = b with one sparse factorization of M (Woodbury identity).

    x = y - M^{-1} Uw (I + Vw^T M^{-1} Uw)^{-1} Vw^T y,   with y = M^{-1} b.

    Args:
        M: sparse matrix, shape [N, N].
        Uw, Vw: dense low-rank factors, shape [N, r].
        b: right-hand sides, shape [N, k].

    Returns:
        Solution x, shape [N, k].
    """
    # No low-rank term (e.g. no inlet): a plain sparse solve.
    if not np.any(Uw):
        return _sparse_solve(M, b)

    # One factorization applied to [b, Uw] gives y = M^{-1} b and M^{-1} Uw.
    k = b.shape[1]
    X = _sparse_solve(M, np.hstack([b, Uw]))            # [M^{-1} b, M^{-1} Uw]
    x = X[:, :k]
    MiU = X[:, k:]

    # Small dense capacitance system (I + Vw^T M^{-1} Uw) z = Vw^T y, then correct y.
    small = np.eye(Uw.shape[1]) + Vw.T @ MiU             # [r, r]
    return x - MiU @ np.linalg.solve(small, Vw.T @ x)


def _host_solve(m: SWEModel, comp: np.ndarray, A: np.ndarray, C: np.ndarray, rhs_vec: np.ndarray,
                shift: float, transpose: bool) -> np.ndarray:
    """Solve (shift I - J) x = b, or J^T x = b if ``transpose`` (host, SciPy).

    J = J_local + A C^T (see ``solvers.jacobian``); b may be a vector or a matrix.

    Args:
        m: the model (selects the cached Jacobian pattern).
        comp: compressed JVPs of J_local, shape [3 n_colors, 3, n].
        A: dR/dK, shape [3n, n_in].
        C: (dK/dU)^T, shape [3n, n_in].
        rhs_vec: right-hand side b, shape [3, n] (or any shape with 3n rows when flattened
            row-major, e.g. [3n, k]).
        shift: diagonal shift 1/dtau (1/s); ignored when ``transpose`` is True.
        transpose: solve J^T x = b (adjoint) instead of (shift I - J) x = b.

    Returns:
        Solution x, float64, same shape as ``rhs_vec``.
    """
    Jl = jacobian_pattern(m).assemble(np.asarray(comp))
    A = np.asarray(A)
    C = np.asarray(C)

    # Right-hand side as columns: [3n, k].
    b = np.asarray(rhs_vec)
    b2 = b.reshape(Jl.shape[0], -1)

    if transpose:   # (J_l^T + C A^T) x = b
        x = _woodbury_solve(Jl.T, C, A, b2)
    else:           # (shift I - J_l - A C^T) x = b
        M = sp.identity(Jl.shape[0], format="csc") * float(shift) - Jl
        x = _woodbury_solve(M, -A, C, b2)

    return x.reshape(b.shape).astype(np.float64)


def _solve(
    m: SWEModel, lin: tuple[Array, Array, Array], b: Array, shift: Array, transpose: bool,
) -> Array:
    """JAX wrapper of ``_host_solve`` (host callback; traceable, not differentiable).

    Args:
        m: the model.
        lin: linearization (comp, A, C) from ``solvers.jacobian.linearize``.
        b: right-hand side, shape [3, n].
        shift: diagonal shift 1/dtau (1/s), scalar.
        transpose: solve J^T x = b instead of (shift I - J) x = b.

    Returns:
        Solution x, float64, same shape as ``b``.
    """
    out = jax.ShapeDtypeStruct(b.shape, jnp.float64)

    def host_callback(c, a, cc, v, s):
        """Host-side call with NumPy arguments (comp, A, C, b, shift)."""
        return _host_solve(m, c, a, cc, v, s, transpose)

    return jax.pure_callback(host_callback, out, *lin, b, shift, vmap_method="sequential")


# ----------------------------------------------------------------------
# Newton loop and the IFT custom VJP
# ----------------------------------------------------------------------
def newton_solve(
    m: SWEModel, theta: Theta, U0: Array, s: NewtonSettings,
) -> tuple[Array, Array, Array]:
    """Run the Newton loop (no custom gradient).

    Args:
        m: the model.
        theta: differentiable parameters.
        U0: initial guess, state [xi, q_x, q_y], shape [3, n_cells].
        s: Newton settings.

    Returns:
        (U*, iterations, final max|R|): the state, shape [3, n_cells]; the iteration
        count (int scalar); and the residual norm (m/s or m^2/s^2, scalar).
    """
    def f(U: Array) -> Array:
        """Residual R(U) = dU/dt, shape [3, n_cells]."""
        return rhs(m, theta, U)

    # Lower bound on xi that keeps the depth h = xi + h_still above the scheme's floor.
    hfloor = m.h_small if m.scheme == "julia_xi_roe" else 0.0     # depth floor of the scheme
    hs = m.hstill if "zb" not in theta else m.wstill - theta["zb"]
    xi_min = hfloor - hs

    def cond(c):
        """Continue while not converged and below the iteration cap."""
        _, it, rn, _, _ = c
        return (rn > s.tol) & (it < s.max_iter)

    def body(c):
        """One pseudo-transient Newton iteration.

        Carry: (U, iteration, max|R(U)|, max|R(U0)|, safeguard factor f_k).
        """
        U, it, rn, r0, fac = c

        # 1. Residual and Jacobian at the current state.
        R = f(U)
        lin = linearize(m, theta, U)

        # 2. SER pseudo-time step: grows as the residual falls.
        residual_ratio = r0 / jnp.maximum(rn, 1e-300)
        dtau_ser = fac * s.dtau0 * residual_ratio ** s.power
        dtau = jnp.minimum(dtau_ser, s.dtau_max)

        # 3. Solve (I/dtau - J) dU = R and update; keep h >= floor.
        dU = _solve(m, lin, R, 1.0 / dtau, transpose=False)
        U_new = U + dU
        U_new = U_new.at[0].set(jnp.maximum(U_new[0], xi_min))   # keep h >= floor
        if U_new.shape[0] > 3:
            U_new = U_new.at[3:].set(jnp.maximum(U_new[3:], 0.0))  # keep h k, h eps >= 0
        rn_new = jnp.max(jnp.abs(f(U_new)))

        # 4. Safeguard: reject steps that blow up the residual and shrink the pseudo-time
        #    step; after accepted steps the factor recovers towards 1.
        ok = jnp.isfinite(rn_new) & (rn_new < s.max_growth * rn)
        U = jnp.where(ok, U_new, U)
        rn = jnp.where(ok, rn_new, rn)
        fac = jnp.where(ok, jnp.minimum(2.0 * fac, 1.0), fac * s.shrink)

        return U, it + 1, rn, r0, fac

    # Initial residual norm (also the SER reference |R_0|), then iterate.
    r0 = jnp.max(jnp.abs(f(U0)))
    init = (U0, jnp.asarray(0), r0, r0, jnp.asarray(1.0))
    U, it, rn, _, _ = jax.lax.while_loop(cond, body, init)

    return U, it, rn


@partial(jax.custom_vjp, nondiff_argnums=(0, 3))
def steady_state(m: SWEModel, theta: Theta, U0: Array, s: NewtonSettings) -> Array:
    """Steady state U*(theta) with IFT gradients w.r.t. ``theta`` (U0 is a guess).

    Args:
        m: the model (static).
        theta: differentiable parameters.
        U0: initial guess, shape [3, n_cells]; it gets a zero gradient.
        s: Newton settings (static).

    Returns:
        Steady state U*, shape [3, n_cells].
    """
    return newton_solve(m, theta, U0, s)[0]


def _fwd(m: SWEModel, theta: Theta, U0: Array, s: NewtonSettings):
    """Forward pass of ``steady_state``: solve and save (theta, U*, U0) for the backward pass."""
    U = newton_solve(m, theta, U0, s)[0]
    return U, (theta, U, U0)


def _bwd(m: SWEModel, s: NewtonSettings, res, g: Array):
    """Backward pass of ``steady_state`` (IFT adjoint).

    Solves J^T lambda = g at U*, then dL/dtheta = -lambda^T dR/dtheta by one VJP.

    Args:
        m: the model.
        s: Newton settings (unused).
        res: residuals (theta, U*, U0) saved by ``_fwd``.
        g: cotangent dL/dU*, shape [3, n_cells].

    Returns:
        (dL/dtheta, dL/dU0 = 0).
    """
    theta, U, U0 = res

    # Adjoint solve J^T lambda = g.
    lam = _solve(m, linearize(m, theta, U), g, jnp.asarray(0.0), transpose=True)

    # dL/dtheta = VJP of R(U*; theta) with -lambda.
    def residual_of_theta(th: Theta) -> Array:
        """R(U*; th), shape [3, n_cells]."""
        return rhs(m, th, U)

    _, vjp_theta = jax.vjp(residual_of_theta, theta)
    (g_theta,) = vjp_theta(-lam)

    return g_theta, jnp.zeros_like(U0)


steady_state.defvjp(_fwd, _bwd)


def steady_jacobian(m: SWEModel, theta: Theta, U: Array) -> np.ndarray:
    """Sensitivities dU*/dtheta = -J^{-1} dR/dtheta at a steady state (host, one LU).

    Args:
        m: the model.
        theta: parameters, a dict with one flat array, e.g. {"n_zone": p}.
        U: converged steady state, shape [3, n_cells].

    Returns:
        [3 n_cells, n_params] array (flat state index k * n_cells + i).
    """
    ((key, p),) = theta.items()

    # Linearization (comp, A, C) of R at U.
    def linearize_at(X: Array) -> tuple[Array, Array, Array]:
        """``linearize`` at state X with the given theta."""
        return linearize(m, theta, X)

    comp, A, C = jax.jit(linearize_at)(U)

    # dR/dtheta by forward mode: [3 n_cells, n_params].
    def flat_residual_of_param(q: Array) -> Array:
        """Flat R(U; {key: q}), shape [3 n_cells]."""
        return rhs(m, {key: q}, U).reshape(-1)

    R_theta = np.asarray(jax.jacfwd(flat_residual_of_param)(p))

    # J x = -R_theta  <=>  (0 I - J) x = R_theta
    return _host_solve(m, comp, A, C, R_theta, 0.0, transpose=False)
