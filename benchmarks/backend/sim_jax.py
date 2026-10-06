"""JAX time integration + loss/gradient for the explicit SWE benchmark.

The forward model marches U = [h, hu, hv] with SSP-RK2 steps of fixed dt (s)
using ``jax.lax.scan`` and records the depth h (m) every ``k_inner`` steps.
The loss is the mean squared depth misfit

    L(theta) = mean_{s, i} (h_s,i(theta) - h_obs_s,i)^2   (m^2),

and its gradient is taken by reverse mode (``jax.value_and_grad``), with
optional two-level checkpointing, or by forward mode (one JVP per parameter).
"""

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

    Args:
        mesh: face-based mesh.
        cfg: flow configuration.
        dt: fixed time step (s).
        n_save: number of depth snapshots.
        k_inner: steps between snapshots.
        checkpoint: if True, use nested ``jax.checkpoint``.
        dtype: floating dtype name.
        device: ``"gpu"`` or ``"cpu"``.

    Returns:
        Dict with jitted ``simulate``, ``loss``, ``grad`` (value and reverse
        gradient), ``fwd_grad`` (forward mode), plus ``ops`` and ``device``.
    """
    ops = JaxOps(dtype)
    jax = ops.jax
    jnp = ops.jnp

    # Place the mesh arrays on the requested device
    dev = jax.devices("gpu" if device == "gpu" else "cpu")[0]
    with jax.default_device(dev):
        m = mesh_arrays(ops, mesh)

    # ---- 1. one SSP-RK2 step (optionally rematerialized) --------------------
    def one_step(U, theta):
        """Advance U [n_cells, 3] by one step of dt (s)."""
        return ssprk2_step(ops, m, cfg, theta, U, dt)

    stp = jax.checkpoint(one_step) if checkpoint else one_step

    # ---- 2. a block of k_inner steps (optionally rematerialized) ------------
    def block(U, theta):
        """Advance U [n_cells, 3] by k_inner steps."""

        def body(U, _):
            """scan body: one step, no per-step output."""
            return stp(U, theta), None

        U, _ = jax.lax.scan(body, U, None, length=k_inner)
        return U

    blk = jax.checkpoint(block) if checkpoint else block

    # ---- 3. full simulation: n_save blocks, depth snapshot after each -------
    def simulate(theta, U0):
        """Depth snapshots (m), shape [n_save, n_cells], from U0 [n_cells, 3]."""

        def outer(U, _):
            """scan body: one block, output the depth column."""
            U = blk(U, theta)
            return U, U[:, 0]

        _, hs = jax.lax.scan(outer, U0, None, length=n_save)
        return hs  # [n_save, n_cells]

    # ---- 4. loss and gradients ----------------------------------------------
    def loss(theta, U0, h_obs):
        """Mean squared depth misfit (m^2); h_obs has shape [n_save, n_cells]."""
        hs = simulate(theta, U0)
        return jnp.mean((hs - h_obs) ** 2)

    def fwd_grad(theta, U0, h_obs):
        """Forward-mode gradient (one vmapped JVP per parameter), as Hydrograd.jl.

        Args:
            theta: parameter pytree.
            U0: initial state, shape [n_cells, 3].
            h_obs: observed depths (m), shape [n_save, n_cells].

        Returns:
            Tuple (loss (m^2), gradient of shape [n_params]) w.r.t. the
            flattened parameters.
        """
        from jax.flatten_util import ravel_pytree
        flat, unravel = ravel_pytree(theta)

        def f(x):
            """Loss as a function of the flat parameter vector x [n_params]."""
            return loss(unravel(x), U0, h_obs)

        def directional(e):
            """Directional derivative of f along the basis vector e [n_params]."""
            return jax.jvp(f, (flat,), (e,))[1]

        basis = jnp.eye(flat.size, dtype=flat.dtype)
        return f(flat), jax.vmap(directional)(basis)

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
    """Parameters as a JAX pytree: a Manning's n field or MLP weights.

    Args:
        ops: ``JaxOps`` shim.
        mode: ``"field"`` (per-cell n) or anything else (MLP closure).
        n_field: Manning's n per cell (s/m^(1/3)), shape [n_cells].
        mlp_sizes: MLP layer widths.
        seed: seed for the MLP weight initialization.
        device: JAX device to place the arrays on.

    Returns:
        ``{"n": [n_cells]}`` or ``{"mlp": [(W, b), ...]}``.
    """
    jax = ops.jax
    with jax.default_device(device):
        # Per-cell Manning's n field
        if mode == "field":
            return {"n": ops.asarray(n_field)}

        # MLP closure weights
        layers = init_mlp(list(mlp_sizes), seed)
        return {"mlp": [(ops.asarray(W), ops.asarray(b)) for W, b in layers]}


def block_until_ready(x: Any) -> Any:
    """Wait for asynchronous JAX computation on x (any pytree); return x."""
    import jax

    return jax.block_until_ready(x)


def peak_memory_bytes(device) -> int:
    """Peak device memory (GPU only; -1 if unavailable).

    Args:
        device: JAX device.

    Returns:
        Peak bytes in use, or -1 if the device reports no statistics.
    """
    try:
        stats = device.memory_stats()
        return int(stats.get("peak_bytes_in_use", -1)) if stats else -1
    except (AttributeError, RuntimeError):
        return -1
