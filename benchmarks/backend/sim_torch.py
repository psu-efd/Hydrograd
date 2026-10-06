"""PyTorch time integration + loss/gradient for the explicit SWE benchmark."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from backends import TorchOps, init_mlp, mesh_arrays
from mesh import FVMesh
from physics import FlowConfig, ssprk2_step


def build(mesh: FVMesh, cfg: FlowConfig, dt: float, n_save: int, k_inner: int,
          checkpoint: bool, dtype: str = "float64", device: str = "cpu",
          compile_step: bool = False) -> Dict[str, Any]:
    """Return forward/loss/grad callables (same semantics as sim_jax.build)."""
    import torch
    from torch.utils.checkpoint import checkpoint as torch_ckpt

    ops = TorchOps("cuda" if device == "gpu" else "cpu", dtype)
    m = mesh_arrays(ops, mesh)

    def step(U, theta):
        return ssprk2_step(ops, m, cfg, theta, U, dt)

    if compile_step:
        step = torch.compile(step, dynamic=False)

    def block(U, theta):
        for _ in range(k_inner):
            if checkpoint and torch.is_grad_enabled():
                U = torch_ckpt(step, U, theta, use_reentrant=False)
            else:
                U = step(U, theta)
        return U

    def simulate(theta, U0):
        U, hs = U0, []
        for _ in range(n_save):
            if checkpoint and torch.is_grad_enabled():
                U = torch_ckpt(block, U, theta, use_reentrant=False)
            else:
                U = block(U, theta)
            hs.append(U[:, 0])
        return torch.stack(hs)

    def loss(theta, U0, h_obs):
        return torch.mean((simulate(theta, U0) - h_obs) ** 2)

    def leaves(theta):
        if "n" in theta:
            return [theta["n"]]
        return [p for Wb in theta["mlp"] for p in Wb]

    def grad(theta, U0, h_obs):
        L = loss(theta, U0, h_obs)
        g = torch.autograd.grad(L, leaves(theta))
        return L, g

    def simulate_nograd(theta, U0):
        with torch.no_grad():
            return simulate(theta, U0)

    def loss_nograd(theta, U0, h_obs):
        with torch.no_grad():
            return loss(theta, U0, h_obs)

    return {"ops": ops, "simulate": simulate_nograd, "loss": loss_nograd, "grad": grad}


def to_theta(ops, mode: str, n_field: np.ndarray, mlp_sizes=(2, 16, 16, 1),
             seed: int = 0) -> Dict[str, Any]:
    """Parameters requiring grad: a Manning's n field or MLP weights."""
    if mode == "field":
        return {"n": ops.asarray(n_field).requires_grad_(True)}
    return {"mlp": [(ops.asarray(W).requires_grad_(True), ops.asarray(b).requires_grad_(True))
                    for W, b in init_mlp(list(mlp_sizes), seed)]}


def synchronize(ops) -> None:
    if ops.device.type == "cuda":
        ops.t.cuda.synchronize()


def reset_peak(ops) -> None:
    if ops.device.type == "cuda":
        ops.t.cuda.reset_peak_memory_stats()


def peak_memory_bytes(ops) -> int:
    if ops.device.type == "cuda":
        return int(ops.t.cuda.max_memory_allocated())
    return -1
