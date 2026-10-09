"""Exact sparse Jacobian dR/dU of the semi-discrete residual (graph coloring + JVPs).

The steady Newton solver (:mod:`hydrograd.solvers.steady`) and the implicit theta-scheme
(:mod:`hydrograd.solvers.implicit`) need the Jacobian J = dR/dU of the residual
R(U) = dU/dt. A dense Jacobian costs 3N JVPs (N = number of cells); this module recovers
the same matrix exactly with a few dozen JVPs by exploiting its sparsity.

State layout
------------
U has shape [3, n_cells]; the flat index of component k (0: xi, 1: q_x, 2: q_y) of
cell i is ``k * n_cells + i`` (the ordering of ``U.reshape(-1)``).

Splitting off the inlet coupling
--------------------------------
For the first-order scheme the residual of cell i depends on U_i and on its face
neighbours. Inlet-Q cells add one non-local dependence: the inlet discharge is split
over the inlet faces in proportion to a conveyance, whose inlet total K sums over all
cells of that inlet. The Jacobian is therefore written as

    dR/dU = J_local + A C^T,

    J_local = dR/dU with K held fixed                      (sparse)   [3N x 3N]
    A       = dR/dK                                        (dense)    [3N x n_in]
    C       = (dK/dU)^T                                    (dense)    [3N x n_in]

with n_in the number of inlets. Linear solves with this low-rank-plus-sparse matrix use
the Woodbury identity (see ``solvers.steady._woodbury_solve``).

Sparsity pattern
----------------
J_local is nonzero only where cells i and j are face neighbours (or i = j). With
second-order MUSCL the stencil is two cells wide (R_i depends on its neighbours'
gradients), so the pattern is the square of the face adjacency.

Graph coloring (Curtis-Powell-Reid compression)
-----------------------------------------------
Two columns j1, j2 of J_local may share one JVP if no row has a nonzero in both, i.e. if
cells j1 and j2 have no common neighbour. A greedy distance-2 coloring of the cell graph
groups the cells into n_colors such sets. For each color c and component k the seed

    s_{c,k} = sum over cells j of color c of e_{(k, j)}

gives J_local s_{c,k}, whose row (kr, i) equals the single entry J[(kr, i), (k, j)] with
j the (unique) cell of color c that touches i. So 3 * n_colors JVPs recover every
nonzero exactly (no approximation). The color count depends on the mesh connectivity
only, not on the inlet width, because the inlet coupling is carried by A C^T.
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
    """Boolean cell-to-cell dependence pattern of the residual, including the diagonal.

    Args:
        m: the model (uses the interior-face connectivity ``fi_o``, ``fi_r`` and ``order``).

    Returns:
        Boolean CSR matrix, shape [n_cells, n_cells]; entry (i, j) is True if R_i may
        depend on U_j. First order: face neighbours; second order or the mixing-length
        eddy viscosity: neighbours of neighbours.
    """
    n = m.n_cells
    owner = np.asarray(m.fi_o)
    neighbour = np.asarray(m.fi_r)

    # Symmetric face adjacency plus the diagonal (every cell depends on itself).
    rows = np.concatenate([owner, neighbour, np.arange(n)])
    cols = np.concatenate([neighbour, owner, np.arange(n)])
    values = np.ones(len(rows), dtype=bool)
    A = sp.csr_matrix((values, (rows, cols)), shape=(n, n))
    A.sum_duplicates()

    # MUSCL, and the mixing-length eddy viscosity: R_i depends on its neighbours'
    # gradients, so the stencil is two cells wide. The pattern of A @ A is that stencil.
    turbulence = getattr(m, "turbulence", None)
    mixing_length = turbulence is not None and turbulence.mode == "mixing_length"
    if getattr(m, "order", 1) == 2 or mixing_length:
        A_int = A.astype(np.int32)
        A = (A_int @ A_int).astype(bool).tocsr()

    return A


def _distance2_coloring(A: sp.csr_matrix) -> np.ndarray:
    """Greedy coloring of A^T A: columns that share a nonzero row get distinct colors.

    Columns of the same color can then be probed by one JVP without their entries
    overlapping in any row.

    Args:
        A: square sparsity pattern, shape [n, n].

    Returns:
        Color index (0, 1, ...) of each column, int64 array of shape [n].
    """
    # Column-intersection graph: (A^T A)_{ij} != 0 if columns i and j share a row.
    A_int = A.astype(np.int32)
    A2 = (A_int.T @ A_int).tocsr()
    n = A.shape[0]

    # -1 marks a column that has no color yet.
    color = -np.ones(n, dtype=np.int64)

    # Visit the columns in order and give each the smallest color not used by any
    # already-colored column it conflicts with.
    for i in range(n):
        conflicts = A2.indices[A2.indptr[i]:A2.indptr[i + 1]]
        conflict_colors = color[conflicts]
        used = set(conflict_colors[conflict_colors >= 0].tolist())

        c = 0
        while c in used:
            c += 1
        color[i] = c

    return color


class JacobianPattern:
    """Sparsity pattern, coloring and JVP seeds for dR/dU of one model.

    Everything here is NumPy (not JAX): the pattern is cached across traces, and JAX
    arrays created inside one trace must not leak into another.

    Attributes:
        n: number of cells.
        rows, cols: cell indices (i, j) of the nonzero cell pairs of the pattern, int64
            arrays of shape [nnz_cells].
        color: color of each cell (column), int64 array of shape [n].
        n_colors: number of colors.
        seeds: JVP seed directions, shape [3 n_colors, 3, n]; seed 3 c + k is 1 in
            component k of every cell of color c and 0 elsewhere.
        R, C: flat row and column indices (``k * n + i``) of every scalar nonzero, shape
            [9 nnz_cells]; each cell pair expands into a 3 x 3 block.
        _gather: index tuple that picks every nonzero value out of the compressed JVP
            array (see ``assemble``).
    """

    def __init__(self, m: SWEModel):
        """Build the pattern, the coloring and the seeds for model ``m``.

        Args:
            m: the model (mesh connectivity and scheme order).
        """
        self.n = m.n_cells

        # Nonzero cell pairs (i, j) of the dependence pattern.
        A = _cell_adjacency(m).tocoo()
        self.rows = A.row.astype(np.int64)
        self.cols = A.col.astype(np.int64)

        # Distance-2 coloring of the cells.
        self.color = _distance2_coloring(A.tocsr())
        self.n_colors = int(self.color.max()) + 1

        # One seed per (color, component): ones in component k of all cells of color c.
        seeds = np.zeros((3 * self.n_colors, 3, self.n))
        for c in range(self.n_colors):
            for k in range(3):
                seeds[3 * c + k, k, self.color == c] = 1.0
        self.seeds = seeds  # NumPy: the pattern is cached across traces, so no JAX arrays here

        # Expand each cell-pair nonzero (i, j) into its 3x3 block of scalar nonzeros:
        # row component kr, column component kc, flat indices kr * n + i and kc * n + j.
        kr, kc = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")   # row / column component
        self.R = (kr[None] * self.n + self.rows[:, None, None]).ravel()
        self.C = (kc[None] * self.n + self.cols[:, None, None]).ravel()

        # Where each nonzero sits in the compressed JVPs:
        # comp[3 cj + kc, kr, i] is J[(kr, i), (kc, j)], with cj the color of column cell j.
        cj = self.color[self.cols]
        self._gather = (3 * cj[:, None, None] + kc[None], kr[None], self.rows[:, None, None])

    def compressed(self, f: Callable[[Array], Array], U: Array) -> Array:
        """JVPs of f at U along all seeds (JAX-traceable).

        Args:
            f: residual function U -> R(U), both of shape [3, n].
            U: linearization point, shape [3, n].

        Returns:
            Compressed Jacobian (J seed for every seed), shape [3 n_colors, 3, n].
        """
        def jvp_along(seed: Array) -> Array:
            """Directional derivative J seed, shape [3, n]."""
            return jax.jvp(f, (U,), (seed,))[1]

        return jax.vmap(jvp_along)(jnp.asarray(self.seeds))

    def assemble(self, comp: np.ndarray) -> sp.csr_matrix:
        """Decompress the JVP products into the sparse matrix J_local (NumPy/SciPy, host).

        Args:
            comp: compressed JVPs from ``compressed``, shape [3 n_colors, 3, n].

        Returns:
            J_local as a CSR matrix, shape [3n, 3n].
        """
        vals = np.asarray(comp)[self._gather]  # [nnz, 3(kk), 3(k)]
        N = 3 * self.n
        return sp.csr_matrix((vals.ravel(), (self.R, self.C)), shape=(N, N))


# Pattern cache: id(model) -> (model, pattern).
_CACHE: dict[int, tuple[SWEModel, JacobianPattern]] = {}


def jacobian_pattern(m: SWEModel) -> JacobianPattern:
    """Cached pattern per model object.

    The entry keeps a reference to its model: ``id`` values are reused after garbage
    collection, so an id match alone could return another mesh's pattern.

    Args:
        m: the model.

    Returns:
        The ``JacobianPattern`` of ``m`` (built on the first call).
    """
    hit = _CACHE.get(id(m))

    # Build on a miss, or when the cached entry belongs to another (collected) model.
    if hit is None or hit[0] is not m:
        hit = (m, JacobianPattern(m))
        _CACHE[id(m)] = hit

    return hit[1]


def linearize(
    m: SWEModel, theta: dict[str, Any], U: Array, t: Any = None,
) -> tuple[Array, Array, Array]:
    """Linearization dR/dU = J_local + A C^T of R(U, t) at U (JAX-traceable).

    ``t`` (s) must be the time at which R is evaluated when boundary conditions vary in time
    (the inlet coupling term dR/dK scales with Q(t)); None uses the first series value,
    as in steady solves.

    Args:
        m: the model.
        theta: differentiable parameters.
        U: linearization point, state [xi, q_x, q_y], shape [3, n].
        t: time (s), or None.

    Returns:
        (comp, A, C):
        comp: compressed JVPs of R with K frozen, [3 n_colors, 3, n] (-> J_local);
        A: dR/dK, [3 n, n_in];
        C: dK/dU transposed, [3 n, n_in].
    """
    from hydrograd.physics.rhs import inlet_conveyance, rhs

    P = jacobian_pattern(m)

    # Inlet conveyance totals at U, frozen (not differentiated) for J_local.
    K0 = jax.lax.stop_gradient(inlet_conveyance(m, theta, U))

    # J_local: JVPs of R with K held at K0.
    def residual_fixed_K(X: Array) -> Array:
        """R(X) with the inlet conveyance frozen at K0, shape [3, n]."""
        return rhs(m, theta, X, K0, t)

    comp = P.compressed(residual_fixed_K, U)

    # A = dR/dK: forward mode, one column per inlet.
    def residual_of_K(K: Array) -> Array:
        """Flat R(U) as a function of the inlet conveyances K, shape [3n]."""
        return rhs(m, theta, U, K, t).reshape(-1)

    A = jax.jacfwd(residual_of_K)(K0)

    # C = (dK/dU)^T: reverse mode, one row per inlet.
    def conveyance_of_U(X: Array) -> Array:
        """Inlet conveyance totals K(X), shape [n_in]."""
        return inlet_conveyance(m, theta, X)

    C = jax.jacrev(conveyance_of_U)(U).reshape(K0.shape[0], -1).T

    return comp, A, C


def full_jacobian(m: SWEModel, comp: np.ndarray, A: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Dense J_local + A C^T (tests / small meshes only).

    Args:
        m: the model.
        comp: compressed JVPs, shape [3 n_colors, 3, n].
        A: dR/dK, shape [3n, n_in].
        C: (dK/dU)^T, shape [3n, n_in].

    Returns:
        Dense Jacobian dR/dU, shape [3n, 3n].
    """
    return jacobian_pattern(m).assemble(comp).toarray() + np.asarray(A) @ np.asarray(C).T


def dense_jacobian(m: SWEModel, theta: dict[str, Any], U: Array) -> np.ndarray:
    """Reference dense dR/dU via jacfwd (small meshes / tests only).

    Args:
        m: the model.
        theta: differentiable parameters.
        U: state, shape [3, n].

    Returns:
        Dense Jacobian dR/dU, shape [3n, 3n].
    """
    from hydrograd.physics.rhs import rhs

    def flat_residual(X: Array) -> Array:
        """Flat R as a function of the flat state, shape [3n] -> [3n]."""
        return rhs(m, theta, X.reshape(3, -1)).reshape(-1)

    J = jax.jacfwd(flat_residual)(U.reshape(-1))
    return np.asarray(J)
