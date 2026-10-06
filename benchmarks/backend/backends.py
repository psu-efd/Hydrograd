"""Thin array-ops shims so physics.py runs unchanged on PyTorch or JAX.

``physics.residual`` is written against a small set of array operations
(``sqrt``, ``where``, ``scatter_add``, ...). ``TorchOps`` and ``JaxOps``
implement that set for each backend, so both backends evaluate exactly the
same finite-volume discretization of the 2D shallow water equations.

The module also provides backend-neutral helpers for the benchmark case:
mesh-to-array conversion, the still-water initial state, a synthetic
Manning's n field, MLP weight initialization, and a CFL time step

    dt = CFL * min_dx / (sqrt(g * h_max) + u_max)   (s).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from mesh import INLET, OUTLET, WALL, FVMesh


class TorchOps:
    """PyTorch implementation of the ops used by physics.residual."""

    sqrt2 = math.sqrt(2.0)

    def __init__(self, device: str = "cpu", dtype: str = "float64"):
        """Store the torch module, device and floating-point dtype.

        Args:
            device: torch device string, e.g. ``"cpu"`` or ``"cuda"``.
            dtype: name of the torch floating dtype, e.g. ``"float64"``.
        """
        import torch

        self.t = torch
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)

    def asarray(self, x: np.ndarray, integer: bool = False, boolean: bool = False):
        """Convert a NumPy array to a backend array.

        Args:
            x: input array, any shape.
            integer: if True, return an integer index array.
            boolean: if True, return a boolean mask (takes precedence).

        Returns:
            Backend array of the same shape (float dtype by default).
        """
        t = self.t
        if boolean:
            return t.as_tensor(x, dtype=t.bool, device=self.device)
        if integer:
            return t.as_tensor(x, dtype=t.int64, device=self.device)
        return t.as_tensor(x, dtype=self.dtype, device=self.device)

    def sqrt(self, x):
        """Elementwise square root of x (any shape)."""
        return self.t.sqrt(x)

    def abs(self, x):
        """Elementwise absolute value of x (any shape)."""
        return self.t.abs(x)

    def tanh(self, x):
        """Elementwise hyperbolic tangent of x (any shape)."""
        return self.t.tanh(x)

    def sigmoid(self, x):
        """Elementwise logistic sigmoid 1/(1 + exp(-x)) (any shape)."""
        return self.t.sigmoid(x)

    def where(self, c, a, b):
        """Select a where the boolean mask c is True, else b (broadcast shapes)."""
        return self.t.where(c, a, b)

    def maximum(self, x, s: float):
        """Elementwise max(x, s) with a scalar floor s (same units as x)."""
        return self.t.clamp(x, min=s)

    def maximum2(self, x, y):
        """Elementwise max(x, y) of two arrays of the same shape."""
        return self.t.maximum(x, y)

    def stack(self, xs, axis: int = 0):
        """Stack a list of equal-shape arrays along a new axis."""
        return self.t.stack(xs, dim=axis)

    def scatter_add(self, vals, idx, n: int):
        """Sum rows of vals into n bins given by idx.

        Args:
            vals: values to accumulate, shape [m, k].
            idx: integer bin of each row, shape [m].
            n: number of bins (cells).

        Returns:
            out[i] = sum of vals[j] over j with idx[j] == i, shape [n, k].
        """
        out = self.t.zeros((n, vals.shape[1]), dtype=vals.dtype, device=vals.device)
        return out.index_add(0, idx, vals)

    def clip_depth(self, U):
        """Clip the depth column to h >= 0 (m), leaving hu, hv unchanged.

        Args:
            U: state [h, hu, hv], shape [n_cells, 3].

        Returns:
            State with column 0 clipped at zero, shape [n_cells, 3].
        """
        return self.t.cat([self.t.clamp(U[:, :1], min=0.0), U[:, 1:]], dim=1)


class JaxOps:
    """JAX implementation of the ops used by physics.residual."""

    sqrt2 = math.sqrt(2.0)

    def __init__(self, dtype: str = "float64"):
        """Import JAX, enable float64 if requested, and store the dtype.

        Args:
            dtype: name of the jax.numpy floating dtype, e.g. ``"float64"``.
        """
        import jax
        import jax.numpy as jnp

        if dtype == "float64":
            jax.config.update("jax_enable_x64", True)
        self.jax = jax
        self.jnp = jnp
        self.dtype = getattr(jnp, dtype)

    def asarray(self, x: np.ndarray, integer: bool = False, boolean: bool = False):
        """Convert a NumPy array to a backend array.

        Args:
            x: input array, any shape.
            integer: if True, return an integer index array.
            boolean: if True, return a boolean mask (takes precedence).

        Returns:
            Backend array of the same shape (float dtype by default).
        """
        jnp = self.jnp
        if boolean:
            return jnp.asarray(x, dtype=bool)
        if integer:
            return jnp.asarray(x, dtype=jnp.int32)
        return jnp.asarray(x, dtype=self.dtype)

    def sqrt(self, x):
        """Elementwise square root of x (any shape)."""
        return self.jnp.sqrt(x)

    def abs(self, x):
        """Elementwise absolute value of x (any shape)."""
        return self.jnp.abs(x)

    def tanh(self, x):
        """Elementwise hyperbolic tangent of x (any shape)."""
        return self.jnp.tanh(x)

    def sigmoid(self, x):
        """Elementwise logistic sigmoid 1/(1 + exp(-x)) (any shape)."""
        return self.jax.nn.sigmoid(x)

    def where(self, c, a, b):
        """Select a where the boolean mask c is True, else b (broadcast shapes)."""
        return self.jnp.where(c, a, b)

    def maximum(self, x, s: float):
        """Elementwise max(x, s) with a scalar floor s (same units as x)."""
        return self.jnp.maximum(x, s)

    def maximum2(self, x, y):
        """Elementwise max(x, y) of two arrays of the same shape."""
        return self.jnp.maximum(x, y)

    def stack(self, xs, axis: int = 0):
        """Stack a list of equal-shape arrays along a new axis."""
        return self.jnp.stack(xs, axis=axis)

    def scatter_add(self, vals, idx, n: int):
        """Sum rows of vals into n bins given by idx.

        Args:
            vals: values to accumulate, shape [m, k].
            idx: integer bin of each row, shape [m].
            n: number of bins (cells).

        Returns:
            out[i] = sum of vals[j] over j with idx[j] == i, shape [n, k].
        """
        return self.jax.ops.segment_sum(vals, idx, num_segments=n)

    def clip_depth(self, U):
        """Clip the depth column to h >= 0 (m), leaving hu, hv unchanged.

        Args:
            U: state [h, hu, hv], shape [n_cells, 3].

        Returns:
            State with column 0 clipped at zero, shape [n_cells, 3].
        """
        return U.at[:, 0].set(self.jnp.maximum(U[:, 0], 0.0))


def mesh_arrays(ops, mesh: FVMesh) -> Dict[str, Any]:
    """Convert an FVMesh to backend arrays split into interior/boundary faces.

    Args:
        ops: backend shim (``TorchOps`` or ``JaxOps``).
        mesh: NumPy face-based mesh.

    Returns:
        Dict of backend arrays: ``area`` (m^2) and ``zb`` (m), shape [n_cells];
        interior-face owner/neighbour indices, unit normals and lengths (m),
        shape [n_inner]; boundary-face owner, normals, lengths (m) and
        wall/inlet/outlet masks, shape [n_bnd]; plus the int ``n_cells``.
    """
    # Interior faces have a right neighbour; boundary faces have neigh = -1
    inner = mesh.neigh >= 0
    bnd = ~inner
    tag = mesh.btag[bnd]

    # fi_*: interior faces (owner o, right cell r); fb_*: boundary faces
    return {
        "n_cells": mesh.n_cells,
        "area": ops.asarray(mesh.area),
        "zb": ops.asarray(mesh.zb),
        "fi_o": ops.asarray(mesh.owner[inner], integer=True),
        "fi_r": ops.asarray(mesh.neigh[inner], integer=True),
        "fi_nx": ops.asarray(mesh.normal[inner, 0]),
        "fi_ny": ops.asarray(mesh.normal[inner, 1]),
        "fi_L": ops.asarray(mesh.length[inner]),
        "fb_o": ops.asarray(mesh.owner[bnd], integer=True),
        "fb_nx": ops.asarray(mesh.normal[bnd, 0]),
        "fb_ny": ops.asarray(mesh.normal[bnd, 1]),
        "fb_L": ops.asarray(mesh.length[bnd]),
        "fb_wall": ops.asarray(tag == WALL, boolean=True),
        "fb_in": ops.asarray(tag == INLET, boolean=True),
        "fb_out": ops.asarray(tag == OUTLET, boolean=True),
    }


def initial_state(mesh: FVMesh, wse0: float = 1.5) -> np.ndarray:
    """Still water at elevation wse0 (m); returns U0 [n_cells, 3].

    Args:
        mesh: face-based mesh (bed ``zb`` in m).
        wse0: still water-surface elevation (m).

    Returns:
        U0 = [h, hu, hv] with h = max(wse0 - zb, 0) (m) and zero discharge
        (m^2/s), shape [n_cells, 3].
    """
    U0 = np.zeros((mesh.n_cells, 3))
    U0[:, 0] = np.maximum(wse0 - mesh.zb, 0.0)
    return U0


def true_n_field(mesh: FVMesh) -> np.ndarray:
    """Synthetic spatially varying Manning's n used to make observations.

    n = 0.03 + 0.01 sin(2 pi x/x_max) cos(pi y/y_max)   (s/m^(1/3)).

    Args:
        mesh: face-based mesh (centroids in m).

    Returns:
        Manning's n per cell (s/m^(1/3)), shape [n_cells].
    """
    # Cell-centroid coordinates (m)
    x = mesh.centroid[:, 0]
    y = mesh.centroid[:, 1]

    # Coordinates scaled to [0, 1] by their maxima (guarded against zero)
    xs = x / max(x.max(), 1e-12)
    ys = y / max(y.max(), 1e-12)
    return 0.03 + 0.01 * np.sin(2 * np.pi * xs) * np.cos(np.pi * ys)


def init_mlp(sizes: List[int], seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Xavier-initialized MLP weights as NumPy arrays.

    Args:
        sizes: layer widths [d_in, hidden..., d_out].
        seed: seed of the NumPy random generator.

    Returns:
        List of (W, b) per layer: W of shape [d_a, d_b] drawn from
        N(0, 2/(d_a + d_b)), b zeros of shape [d_b].
    """
    rng = np.random.default_rng(seed)
    params = []

    # One (weight, bias) pair per consecutive pair of layer widths
    for a, b in zip(sizes[:-1], sizes[1:]):
        weight = rng.normal(0, math.sqrt(2.0 / (a + b)), (a, b))
        params.append((weight, np.zeros(b)))
    return params


def cfl_dt(mesh: FVMesh, cfl: float = 0.4, h_max: float = 2.0, u_max: float = 2.0) -> float:
    """Fixed explicit time step (s) from a CFL bound.

    Args:
        mesh: face-based mesh (``min_dx`` in m).
        cfl: Courant number (-).
        h_max: assumed maximum depth (m).
        u_max: assumed maximum flow speed (m/s).

    Returns:
        dt = cfl * min_dx / (sqrt(9.81 h_max) + u_max) (s).
    """
    return cfl * mesh.min_dx / (math.sqrt(9.81 * h_max) + u_max)
