"""PyTorch time integration + loss/gradient for the explicit SWE benchmark.

Mirror of ``sim_jax``: SSP-RK2 steps of fixed dt (s) in a Python loop, a
depth (m) snapshot every ``k_inner`` steps, and the mean squared depth misfit
(m^2) as loss. Reverse-mode gradients use ``torch.autograd.grad`` with
optional two-level ``torch.utils.checkpoint`` and an optional
``torch.compile`` of the single step.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from backends import TorchOps, init_mlp, mesh_arrays
from mesh import FVMesh
from physics import FlowConfig, ssprk2_step


def build(mesh: FVMesh, cfg: FlowConfig, dt: float, n_save: int, k_inner: int,
          checkpoint: bool, dtype: str = "float64", device: str = "cpu",
          compile_step: bool = False) -> Dict[str, Any]:
    """Return forward/loss/grad callables (same semantics as sim_jax.build).

    Args:
        mesh: face-based mesh.
        cfg: flow configuration.
        dt: fixed time step (s).
        n_save: number of depth snapshots.
        k_inner: steps between snapshots.
        checkpoint: if True, checkpoint every block and every step.
        dtype: floating dtype name.
        device: ``"gpu"`` (CUDA) or ``"cpu"``.
        compile_step: if True, wrap the single step in ``torch.compile``.

    Returns:
        Dict with ``ops``, no-grad ``simulate`` and ``loss``, and ``grad``
        returning (loss, tuple of gradients per parameter leaf).
    """
    import torch
    from torch.utils.checkpoint import checkpoint as torch_ckpt

    ops = TorchOps("cuda" if device == "gpu" else "cpu", dtype)
    m = mesh_arrays(ops, mesh)

    # ---- 1. one SSP-RK2 step (optionally compiled) --------------------------
    def step(U, theta):
        """Advance U [n_cells, 3] by one step of dt (s)."""
        return ssprk2_step(ops, m, cfg, theta, U, dt)

    if compile_step:
        step = torch.compile(step, dynamic=False)

    # ---- 2. a block of k_inner steps (each checkpointed when taping) -------
    def block(U, theta):
        """Advance U [n_cells, 3] by k_inner steps."""
        for _ in range(k_inner):
            if checkpoint and torch.is_grad_enabled():
                U = torch_ckpt(step, U, theta, use_reentrant=False)
            else:
                U = step(U, theta)
        return U

    # ---- 3. full simulation: n_save blocks, depth snapshot after each -------
    def simulate(theta, U0):
        """Depth snapshots (m), shape [n_save, n_cells], from U0 [n_cells, 3]."""
        U = U0
        hs = []
        for _ in range(n_save):
            if checkpoint and torch.is_grad_enabled():
                U = torch_ckpt(block, U, theta, use_reentrant=False)
            else:
                U = block(U, theta)
            hs.append(U[:, 0])
        return torch.stack(hs)

    # ---- 4. loss, parameter leaves and gradient ------------------------------
    def loss(theta, U0, h_obs):
        """Mean squared depth misfit (m^2); h_obs has shape [n_save, n_cells]."""
        return torch.mean((simulate(theta, U0) - h_obs) ** 2)

    def leaves(theta):
        """Flat list of parameter tensors to differentiate with respect to."""
        if "n" in theta:
            return [theta["n"]]
        return [p for Wb in theta["mlp"] for p in Wb]

    def grad(theta, U0, h_obs):
        """Loss (m^2) and its gradient w.r.t. each leaf of theta."""
        L = loss(theta, U0, h_obs)
        g = torch.autograd.grad(L, leaves(theta))
        return L, g

    # ---- 5. no-grad wrappers for timing the forward pass --------------------
    def simulate_nograd(theta, U0):
        """``simulate`` without building an autograd tape."""
        with torch.no_grad():
            return simulate(theta, U0)

    def loss_nograd(theta, U0, h_obs):
        """``loss`` without building an autograd tape."""
        with torch.no_grad():
            return loss(theta, U0, h_obs)

    return {"ops": ops, "simulate": simulate_nograd, "loss": loss_nograd, "grad": grad}


def to_theta(ops, mode: str, n_field: np.ndarray, mlp_sizes=(2, 16, 16, 1),
             seed: int = 0) -> Dict[str, Any]:
    """Parameters requiring grad: a Manning's n field or MLP weights.

    Args:
        ops: ``TorchOps`` shim.
        mode: ``"field"`` (per-cell n) or anything else (MLP closure).
        n_field: Manning's n per cell (s/m^(1/3)), shape [n_cells].
        mlp_sizes: MLP layer widths.
        seed: seed for the MLP weight initialization.

    Returns:
        ``{"n": [n_cells]}`` or ``{"mlp": [(W, b), ...]}`` with requires_grad set.
    """
    # Per-cell Manning's n field
    if mode == "field":
        return {"n": ops.asarray(n_field).requires_grad_(True)}

    # MLP closure weights
    layers = []
    for W, b in init_mlp(list(mlp_sizes), seed):
        weight = ops.asarray(W).requires_grad_(True)
        bias = ops.asarray(b).requires_grad_(True)
        layers.append((weight, bias))
    return {"mlp": layers}


def synchronize(ops) -> None:
    """Wait for queued CUDA kernels (no-op on CPU); ops is a ``TorchOps``."""
    if ops.device.type == "cuda":
        ops.t.cuda.synchronize()


def reset_peak(ops) -> None:
    """Reset the CUDA peak-memory counter (no-op on CPU)."""
    if ops.device.type == "cuda":
        ops.t.cuda.reset_peak_memory_stats()


def peak_memory_bytes(ops) -> int:
    """Peak CUDA memory allocated (bytes) since the last reset; -1 on CPU."""
    if ops.device.type == "cuda":
        return int(ops.t.cuda.max_memory_allocated())
    return -1
