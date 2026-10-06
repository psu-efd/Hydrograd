"""Exact sparse Jacobian dR/dU of the semi-discrete residual (coloring + JVPs).

For the first-order scheme, the residual of cell i depends on U_i and on its
face neighbours, plus, for inlet-Q cells, on the inlet's total conveyance K,
which sums over all cells of that inlet. The Jacobian is therefore split as

    dR/dU = J_local + A C^T,   A = dR/dK [3N x n_in],  C = (dK/dU)^T [3N x n_in],

where J_local (K held fixed) is sparse on the face-adjacency pattern (its square for
second-order MUSCL) and is
recovered exactly by a distance-2 coloring with 3 * n_colors forward-mode JVPs
(Curtis-Powell-Reid compression). The color count no longer grows with the
inlet width; linear solves use the Woodbury identity (``solvers.steady``).

State layout: U has shape [3, n_cells]; the flat index of component k of cell i
is ``k * n_cells + i`` (``U.reshape(-1)``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import Array

from hydrograd.physics.model import SWEModel

__all__ = ["JacobianPattern", "full_jacobian", "jacobian_pattern", "linearize"]


def _cell_adjacency(m: SWEModel) -> sp.csr_matrix:
    """Boolean face-adjacency pattern of cells, including the diagonal."""
    n = m.n_cells
    o, r = np.asarray(m.fi_o), np.asarray(m.fi_r)
    rows = np.concatenate([o, r, np.arange(n)])
    cols = np.concatenate([r, o, np.arange(n)])
    A = sp.csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(n, n))
    A.sum_duplicates()
    if getattr(m, "order", 1) == 2:   # MUSCL: R_i depends on neighbours' gradients (two-cell stencil)
        Ai = A.astype(np.int32)
        A = (Ai @ Ai).astype(bool).tocsr()
    return A


def _distance2_coloring(A: sp.csr_matrix) -> np.ndarray:
    """Greedy coloring of A^T A: columns that share a nonzero row get distinct colors."""
    Ai = A.astype(np.int32)
    A2 = (Ai.T @ Ai).tocsr()
    n = A.shape[0]
    color = -np.ones(n, dtype=np.int64)
    for i in range(n):
        nb = A2.indices[A2.indptr[i]:A2.indptr[i + 1]]
        used = set(color[nb][color[nb] >= 0].tolist())
        c = 0
        while c in used:
            c += 1
        color[i] = c
    return color


class JacobianPattern:
    """Sparsity pattern, coloring and seeds for dR/dU of one model."""

    def __init__(self, m: SWEModel):
        self.n = m.n_cells
        A = _cell_adjacency(m).tocoo()
        self.rows, self.cols = A.row.astype(np.int64), A.col.astype(np.int64)
        self.color = _distance2_coloring(A.tocsr())
        self.n_colors = int(self.color.max()) + 1
        seeds = np.zeros((3 * self.n_colors, 3, self.n))
        for c in range(self.n_colors):
            for k in range(3):
                seeds[3 * c + k, k, self.color == c] = 1.0
        self.seeds = seeds  # NumPy: the pattern is cached across traces, so no JAX arrays here
        # Expansion of each cell-pair nonzero into its 3x3 block (row comp kk, col comp k).
        kr, kc = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")   # row / column component
        self.R = (kr[None] * self.n + self.rows[:, None, None]).ravel()
        self.C = (kc[None] * self.n + self.cols[:, None, None]).ravel()
        cj = self.color[self.cols]
        # comp[3 cj + kc, kr, i] is J[(kr, i), (kc, j)]
        self._gather = (3 * cj[:, None, None] + kc[None], kr[None], self.rows[:, None, None])

    def compressed(self, f: Callable[[Array], Array], U: Array) -> Array:
        """JVPs of f at U along all seeds, shape [3 n_colors, 3, n] (JAX-traceable)."""
        return jax.vmap(lambda s: jax.jvp(f, (U,), (s,))[1])(jnp.asarray(self.seeds))

    def assemble(self, comp: np.ndarray) -> sp.csr_matrix:
        """Decompress JVP products into a (3n x 3n) CSR matrix (NumPy/SciPy, host)."""
        vals = np.asarray(comp)[self._gather]  # [nnz, 3(kk), 3(k)]
        N = 3 * self.n
        return sp.csr_matrix((vals.ravel(), (self.R, self.C)), shape=(N, N))


_CACHE: dict[int, tuple[SWEModel, JacobianPattern]] = {}


def jacobian_pattern(m: SWEModel) -> JacobianPattern:
    """Cached pattern per model object.

    The entry keeps a reference to its model: ``id`` values are reused after garbage
    collection, so an id match alone could return another mesh's pattern.
    """
    hit = _CACHE.get(id(m))
    if hit is None or hit[0] is not m:
        hit = _CACHE[id(m)] = (m, JacobianPattern(m))
    return hit[1]


def linearize(m: SWEModel, theta: dict[str, Any], U: Array, t: Any = None) -> tuple[Array, Array, Array]:
    """Linearization of R(U, t) at U (JAX-traceable).

    ``t`` (s) must be the time at which R is evaluated when boundary conditions vary in time
    (the inlet coupling term dR/dK scales with Q(t)); None uses the first series value,
    as in steady solves.

    Returns:
        comp: compressed JVPs of R with K frozen, [3 n_colors, 3, n] (-> J_local);
        A: dR/dK, [3 n, n_in];  C: dK/dU transposed, [3 n, n_in].
    """
    from hydrograd.physics.rhs import inlet_conveyance, rhs

    P = jacobian_pattern(m)
    K0 = jax.lax.stop_gradient(inlet_conveyance(m, theta, U))
    comp = P.compressed(lambda X: rhs(m, theta, X, K0, t), U)
    A = jax.jacfwd(lambda K: rhs(m, theta, U, K, t).reshape(-1))(K0)
    C = jax.jacrev(lambda X: inlet_conveyance(m, theta, X))(U).reshape(K0.shape[0], -1).T
    return comp, A, C


def full_jacobian(m: SWEModel, comp: np.ndarray, A: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Dense J_local + A C^T (tests / small meshes only)."""
    return jacobian_pattern(m).assemble(comp).toarray() + np.asarray(A) @ np.asarray(C).T


def dense_jacobian(m: SWEModel, theta: dict[str, Any], U: Array) -> np.ndarray:
    """Reference dense dR/dU via jacfwd (small meshes / tests only)."""
    from hydrograd.physics.rhs import rhs

    J = jax.jacfwd(lambda X: rhs(m, theta, X.reshape(3, -1)).reshape(-1))(U.reshape(-1))
    return np.asarray(J)
