"""Exact sparse Jacobian dR/dU of the FV residual via graph coloring + JVPs (JAX).

For a first-order FV scheme, R_i depends only on U_i and its face
neighbours, so dR/dU has 3x3 blocks on the cell-adjacency pattern. A
distance-2 coloring of the cell graph lets 3*n_colors forward-mode JVPs
(vmapped) recover every nonzero exactly (Curtis-Powell-Reid compression).
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import scipy.sparse as sp

from mesh import FVMesh


def cell_adjacency(mesh: FVMesh) -> sp.csr_matrix:
    """Boolean cell adjacency including the diagonal."""
    inner = mesh.neigh >= 0
    o, r = mesh.owner[inner], mesh.neigh[inner]
    n = mesh.n_cells
    rows = np.concatenate([o, r, np.arange(n)])
    cols = np.concatenate([r, o, np.arange(n)])
    return sp.csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(n, n))


def distance2_coloring(A: sp.csr_matrix) -> np.ndarray:
    """Greedy coloring of A^2 (cells sharing any row get distinct colors)."""
    A2 = (A.astype(np.int32) @ A.astype(np.int32)).tocsr()
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


class SparseJacobian:
    """Assembles dR/dU (3n x 3n CSR) for a residual R: [n,3] -> [n,3]."""

    def __init__(self, mesh: FVMesh):
        import jax.numpy as jnp

        self.n = mesh.n_cells
        A = cell_adjacency(mesh).tocoo()
        self.rows, self.cols = A.row, A.col
        self.color = distance2_coloring(cell_adjacency(mesh))
        self.n_colors = int(self.color.max()) + 1
        seeds = np.zeros((3 * self.n_colors, self.n, 3))
        for c in range(self.n_colors):
            for k in range(3):
                seeds[3 * c + k, self.color == c, k] = 1.0
        self.seeds = jnp.asarray(seeds)
        # CSR index pattern for the 3x3-block expansion
        bi, bj = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")  # (row comp, col comp)
        self.R = (3 * self.rows[:, None, None] + bi[None]).ravel()
        self.C = (3 * self.cols[:, None, None] + bj[None]).ravel()

    def make(self, resid: Callable) -> Callable:
        """Return jitted f(U, theta) -> compressed JVP products [3*nc, n, 3]."""
        import jax

        seeds = self.seeds

        def compressed(U, theta):
            f = lambda X: resid(theta, X)
            jvp = lambda s: jax.jvp(f, (U,), (s,))[1]
            return jax.vmap(jvp)(seeds)

        return jax.jit(compressed)

    def assemble(self, comp: np.ndarray) -> sp.csr_matrix:
        """Decompress JVP products into a scipy CSR matrix."""
        comp = np.asarray(comp)  # [3*nc, n, 3]: comp[3c+k, i, kk] = dR_i,kk/dU_j*,k
        cj = self.color[self.cols]
        # vals[nnz, kk, k]
        vals = comp[(3 * cj[:, None] + np.arange(3)[None, :]), self.rows[:, None], :]  # [nnz, k, kk]
        vals = np.transpose(vals, (0, 2, 1))
        N = 3 * self.n
        return sp.csr_matrix((vals.ravel(), (self.R, self.C)), shape=(N, N))
