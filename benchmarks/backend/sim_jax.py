"""JAX time integration + loss/gradient for the explicit SWE benchmark."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np

from backends import JaxOps, init_mlp, mesh_arrays
from mesh import FVMesh
from physics import FlowConfig, ssprk2_step


def build(mesh: FVMesh, cfg: FlowConfig, dt: float, n_save: int, k_inner: int,
          checkpoint: bool, dtype: str = "float64", device: str = "cpu") -> Dict[str, Any]:
    """Return jitted forward/loss/grad functions.

    The simulation runs n_save * k_inner SSP-RK2 steps; the depth field is
    compared with observations after every k_inner steps (n_save snapshots).
    With ``checkpoint`` (nested two-level checkpointing) each k_inner block
    and each step inside it are rematerialized on the backward pass, so the
    tape stores ~n_save + k_inner states plus one step's intermediates.
    """
    ops = JaxOps(dtype)
    jax, jnp = ops.jax, ops.jnp
    dev = jax.devices("gpu" if device == "gpu" else "cpu")[0]
    with jax.default_device(dev):
        m = mesh_arrays(ops, mesh)

    def one_step(U, theta):
        return ssprk2_step(ops, m, cfg, theta, U, dt)

    stp = jax.checkpoint(one_step) if checkpoint else one_step

    def block(U, theta):
        def body(U, _):
            return stp(U, theta), None
        U, _ = jax.lax.scan(body, U, None, length=k_inner)
        return U

    blk = jax.checkpoint(block) if checkpoint else block

    def simulate(theta, U0):
        def outer(U, _):
            U = blk(U, theta)
            return U, U[:, 0]
        _, hs = jax.lax.scan(outer, U0, None, length=n_save)
        return hs  # [n_save, n_cells]

    def loss(theta, U0, h_obs):
        hs = simulate(theta, U0)
        return jnp.mean((hs - h_obs) ** 2)

    def fwd_grad(theta, U0, h_obs):
        """Forward-mode gradient (one vmapped JVP per parameter), as Hydrograd.jl."""
        from jax.flatten_util import ravel_pytree
        flat, unravel = ravel_pytree(theta)
        f = lambda x: loss(unravel(x), U0, h_obs)
        basis = jnp.eye(flat.size, dtype=flat.dtype)
        return f(flat), jax.vmap(lambda e: jax.jvp(f, (flat,), (e,))[1])(basis)

    return {
        "fwd_grad": jax.jit(fwd_grad),
        "ops": ops,
        "device": dev,
        "simulate": jax.jit(simulate),
        "loss": jax.jit(loss),
        "grad": jax.jit(jax.value_and_grad(loss)),
    }


def to_theta(ops, mode: str, n_field: np.ndarray, mlp_sizes=(2, 16, 16, 1), seed: int = 0,
             device=None) -> Dict[str, Any]:
    """Parameters as a JAX pytree: a Manning's n field or MLP weights."""
    jax = ops.jax
    with jax.default_device(device):
        if mode == "field":
            return {"n": ops.asarray(n_field)}
        return {"mlp": [(ops.asarray(W), ops.asarray(b)) for W, b in init_mlp(list(mlp_sizes), seed)]}


def block_until_ready(x: Any) -> Any:
    import jax

    return jax.block_until_ready(x)


def peak_memory_bytes(device) -> int:
    """Peak device memory (GPU only; -1 if unavailable)."""
    try:
        stats = device.memory_stats()
        return int(stats.get("peak_bytes_in_use", -1)) if stats else -1
    except (AttributeError, RuntimeError):
        return -1
