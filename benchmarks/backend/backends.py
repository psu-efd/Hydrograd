"""Thin array-ops shims so physics.py runs unchanged on PyTorch or JAX."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from mesh import INLET, OUTLET, WALL, FVMesh


class TorchOps:
    """PyTorch implementation of the ops used by physics.residual."""

    sqrt2 = math.sqrt(2.0)

    def __init__(self, device: str = "cpu", dtype: str = "float64"):
        import torch

        self.t = torch
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)

    def asarray(self, x: np.ndarray, integer: bool = False, boolean: bool = False):
        t = self.t
        if boolean:
            return t.as_tensor(x, dtype=t.bool, device=self.device)
        if integer:
            return t.as_tensor(x, dtype=t.int64, device=self.device)
        return t.as_tensor(x, dtype=self.dtype, device=self.device)

    def sqrt(self, x):
        return self.t.sqrt(x)

    def abs(self, x):
        return self.t.abs(x)

    def tanh(self, x):
        return self.t.tanh(x)

    def sigmoid(self, x):
        return self.t.sigmoid(x)

    def where(self, c, a, b):
        return self.t.where(c, a, b)

    def maximum(self, x, s: float):
        return self.t.clamp(x, min=s)

    def maximum2(self, x, y):
        return self.t.maximum(x, y)

    def stack(self, xs, axis: int = 0):
        return self.t.stack(xs, dim=axis)

    def scatter_add(self, vals, idx, n: int):
        out = self.t.zeros((n, vals.shape[1]), dtype=vals.dtype, device=vals.device)
        return out.index_add(0, idx, vals)

    def clip_depth(self, U):
        return self.t.cat([self.t.clamp(U[:, :1], min=0.0), U[:, 1:]], dim=1)


class JaxOps:
    """JAX implementation of the ops used by physics.residual."""

    sqrt2 = math.sqrt(2.0)

    def __init__(self, dtype: str = "float64"):
        import jax
        import jax.numpy as jnp

        if dtype == "float64":
            jax.config.update("jax_enable_x64", True)
        self.jax, self.jnp = jax, jnp
        self.dtype = getattr(jnp, dtype)

    def asarray(self, x: np.ndarray, integer: bool = False, boolean: bool = False):
        jnp = self.jnp
        if boolean:
            return jnp.asarray(x, dtype=bool)
        if integer:
            return jnp.asarray(x, dtype=jnp.int32)
        return jnp.asarray(x, dtype=self.dtype)

    def sqrt(self, x):
        return self.jnp.sqrt(x)

    def abs(self, x):
        return self.jnp.abs(x)

    def tanh(self, x):
        return self.jnp.tanh(x)

    def sigmoid(self, x):
        return self.jax.nn.sigmoid(x)

    def where(self, c, a, b):
        return self.jnp.where(c, a, b)

    def maximum(self, x, s: float):
        return self.jnp.maximum(x, s)

    def maximum2(self, x, y):
        return self.jnp.maximum(x, y)

    def stack(self, xs, axis: int = 0):
        return self.jnp.stack(xs, axis=axis)

    def scatter_add(self, vals, idx, n: int):
        return self.jax.ops.segment_sum(vals, idx, num_segments=n)

    def clip_depth(self, U):
        return U.at[:, 0].set(self.jnp.maximum(U[:, 0], 0.0))


def mesh_arrays(ops, mesh: FVMesh) -> Dict[str, Any]:
    """Convert an FVMesh to backend arrays split into interior/boundary faces."""
    inner = mesh.neigh >= 0
    bnd = ~inner
    tag = mesh.btag[bnd]
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
    """Still water at elevation wse0 (m); returns U0 [n_cells, 3]."""
    U0 = np.zeros((mesh.n_cells, 3))
    U0[:, 0] = np.maximum(wse0 - mesh.zb, 0.0)
    return U0


def true_n_field(mesh: FVMesh) -> np.ndarray:
    """Synthetic spatially varying Manning's n used to make observations."""
    x, y = mesh.centroid[:, 0], mesh.centroid[:, 1]
    xs, ys = x / max(x.max(), 1e-12), y / max(y.max(), 1e-12)
    return 0.03 + 0.01 * np.sin(2 * np.pi * xs) * np.cos(np.pi * ys)


def init_mlp(sizes: List[int], seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Xavier-initialized MLP weights as NumPy arrays."""
    rng = np.random.default_rng(seed)
    params = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        params.append((rng.normal(0, math.sqrt(2.0 / (a + b)), (a, b)), np.zeros(b)))
    return params


def cfl_dt(mesh: FVMesh, cfl: float = 0.4, h_max: float = 2.0, u_max: float = 2.0) -> float:
    """Fixed explicit time step (s) from a CFL bound."""
    return cfl * mesh.min_dx / (math.sqrt(9.81 * h_max) + u_max)
