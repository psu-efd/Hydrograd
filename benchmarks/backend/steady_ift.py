"""Steady-state gradients: implicit function theorem vs backprop-through-pseudo-time.

The steady state U*(theta) of the semi-discrete shallow water equations solves

    R(U*; theta) = 0,   U = [h, q_x, q_y] per cell, shape [n_cells, 3]  (m, m^2/s, m^2/s)

and the loss is L = mean_cells (h* - h_obs)^2 (m^2). By the implicit function theorem (IFT)

    dL/dtheta = -lambda^T dR/dtheta,   with   (dR/dU)^T lambda = dL/dU,

so one transposed linear solve replaces differentiation through the iterations.

Methods (one per process for clean memory numbers):
  newton_ift    implicit pseudo-transient Newton steady solve (sparse LU) +
                IFT adjoint with the exact sparse Jacobian (one transposed solve)
  newton_gmres  same forward, adjoint via matrix-free GMRES on VJPs (no Jacobian)
  explicit_bptt explicit SSP-RK2 march to steady state + reverse-mode AD
                through every step (two-level checkpointing)
  explicit_fwd  explicit march to steady state only (forward cost reference)

Usage:
    python steady_ift.py --method newton_ift --cells 4000

Prints one JSON line; the truth observations and the gradient vector are cached in
results/steady/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import scipy.sparse.linalg as spla

from backends import JaxOps, cfl_dt, init_mlp, initial_state, mesh_arrays, true_n_field
from bench_explicit import process_peak_mb
from jacobian import SparseJacobian
from mesh import make_channel_mesh
from physics import FlowConfig, residual, ssprk2_step

OUT = os.path.join(os.path.dirname(__file__), "results", "steady")


def newton_steady(
    resid_j, comp_j, SJ, theta, U, dtau0: float, tol: float = 1e-10, max_it: int = 200
):
    """Pseudo-transient continuation Newton (SER) for R(U; theta) = 0.

    Each iteration solves (I / dtau - J) dU = R with sparse LU, where J = dR/dU and the
    pseudo time step grows as residuals fall (switched evolution relaxation):

        dtau = min(dtau0 (r0 / r)^1.5, 1e8),   r = max |R|.

    Depth is clipped at 1e-4 m after each update.

    Args:
        resid_j: Jitted residual R(theta, U), returns shape [n_cells, 3].
        comp_j: Compressed Jacobian evaluator comp_j(U, theta) from ``SparseJacobian.make``.
        SJ: The ``SparseJacobian`` that assembles the compressed Jacobian.
        theta: Parameter pytree (Manning's n field or MLP weights).
        U: Initial state, shape [n_cells, 3] (m, m^2/s, m^2/s); updated out of place.
        dtau0: Initial pseudo time step (s).
        tol: Convergence tolerance on max |R|.
        max_it: Maximum number of Newton iterations.

    Returns:
        Tuple (U, iterations, final max |R|): the steady state, shape [n_cells, 3], the
        number of iterations used, and the final residual norm.
    """
    import scipy.sparse as sp

    r0 = None
    for it in range(max_it):
        # Residual and its max norm; the first norm is the SER reference r0.
        R = np.asarray(resid_j(theta, U))
        rn = float(np.abs(R).max())
        r0 = rn if r0 is None else r0
        if rn < tol:
            return U, it, rn

        # Newton matrix (I / dtau - J) with the SER pseudo time step.
        J = SJ.assemble(comp_j(U, theta))
        dtau = min(dtau0 * (r0 / rn) ** 1.5, 1e8)
        M = (sp.identity(J.shape[0], format="csc") / dtau - J).tocsc()

        # Update and depth clipping.
        dU = spla.splu(M).solve(R.ravel()).reshape(-1, 3)
        U = U + dU
        U[:, 0] = np.maximum(U[:, 0], 1e-4)
    return U, max_it, rn


def main() -> None:
    """Parse the CLI, solve the steady problem, compute the gradient, print one JSON line."""
    # ---------------------------------------------------------------------------------
    # 1. Command-line options
    # ---------------------------------------------------------------------------------
    p = argparse.ArgumentParser()
    p.add_argument(
        "--method",
        required=True,
        choices=["newton_ift", "newton_gmres", "explicit_bptt", "explicit_fwd"],
    )
    p.add_argument("--cells", type=int, default=4000)
    p.add_argument("--length", type=float, default=200.0)
    p.add_argument("--width", type=float, default=20.0)
    p.add_argument("--n_steps", type=int, default=0, help="explicit steps (0 = from file)")
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)

    import jax
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    # ---------------------------------------------------------------------------------
    # 2. Problem setup: channel mesh, flow configuration, residual, MLP parameters
    # ---------------------------------------------------------------------------------
    ops = JaxOps("float64")
    mesh = make_channel_mesh(a.cells, a.length, a.width)
    cfg = FlowConfig(q_in=1.0, wse_out=1.5)
    m = mesh_arrays(ops, mesh)
    dt = cfl_dt(mesh)
    U0 = initial_state(mesh, 1.5)
    tag = f"{mesh.n_cells}"

    def resid(theta, U):
        """Residual R(U; theta), shape [n_cells, 3]."""
        return residual(ops, m, cfg, theta, U)

    resid_j = jax.jit(resid)
    mlp_layers = [(jnp.asarray(W), jnp.asarray(b)) for W, b in init_mlp([2, 16, 16, 1])]
    theta_nn = {"mlp": mlp_layers}
    res = {"method": a.method, "cells": mesh.n_cells, "dt": dt}

    # ---------------------------------------------------------------------------------
    # 3. Truth observations: steady state with the true n field (cached on disk)
    # ---------------------------------------------------------------------------------
    obs_file = os.path.join(OUT, f"obs_{tag}.npz")
    if not os.path.exists(obs_file):
        SJ = SparseJacobian(mesh)
        comp_j = SJ.make(resid)
        theta_true = {"n": jnp.asarray(true_n_field(mesh))}
        Ut, _, _ = newton_steady(resid_j, comp_j, SJ, theta_true, U0.copy(), 10 * dt)
        np.savez(obs_file, h_obs=Ut[:, 0])
    h_obs = jnp.asarray(np.load(obs_file)["h_obs"])

    def loss_U(U):
        """Loss mean_cells (h - h_obs)^2 (m^2) of a state U, shape [n_cells, 3]."""
        return jnp.mean((U[:, 0] - h_obs) ** 2)

    t0 = time.perf_counter()
    if a.method.startswith("newton"):
        # -----------------------------------------------------------------------------
        # 4a. Newton forward solve, then the IFT adjoint
        # -----------------------------------------------------------------------------
        SJ = SparseJacobian(mesh)
        comp_j = SJ.make(resid)
        res["n_colors"] = SJ.n_colors

        # Forward: pseudo-transient Newton from the initial state.
        t1 = time.perf_counter()
        U, iters, rn = newton_steady(resid_j, comp_j, SJ, theta_nn, U0.copy(), 10 * dt)
        res.update(fwd_s=time.perf_counter() - t1, newton_iters=iters, final_res=rn)

        # Adjoint right-hand side dL/dU, flattened to shape [3 n_cells].
        t1 = time.perf_counter()
        Uj = jnp.asarray(U)
        gU = np.array(jax.grad(loss_U)(Uj)).ravel()

        if a.method == "newton_ift":
            # Direct solve J^T lam = dL/dU with the exact sparse Jacobian.
            J = SJ.assemble(comp_j(Uj, theta_nn))
            lam = spla.splu(J.T.tocsc()).solve(gU)
        else:
            # Matrix-free GMRES on v -> J^T v computed by a VJP.
            _, vjp_U = jax.vjp(lambda X: resid(theta_nn, X), Uj)
            vjp_j = jax.jit(lambda v: vjp_U(v)[0])
            count = [0]

            def matvec(v):
                """Product J^T v for a flat vector v, shape [3 n_cells]."""
                v_state = jnp.asarray(v.reshape(-1, 3))
                return np.array(vjp_j(v_state)).ravel()

            def cb(_):
                """GMRES callback: count iterations."""
                return count.__setitem__(0, count[0] + 1)

            op = spla.LinearOperator((gU.size, gU.size), dtype=np.float64, matvec=matvec)
            lam, info = spla.gmres(
                op, gU, rtol=1e-8, restart=200, maxiter=50, callback=cb, callback_type="pr_norm"
            )
            res.update(gmres_iters=count[0], gmres_info=int(info))

        # Parameter gradient dL/dtheta = -lam^T dR/dtheta.
        _, vjp_th = jax.vjp(lambda th: resid(th, Uj), theta_nn)
        g = vjp_th(jnp.asarray(-lam.reshape(-1, 3)))[0]
        res["adjoint_s"] = time.perf_counter() - t1
        L = float(loss_U(Uj))
    else:
        # -----------------------------------------------------------------------------
        # 4b. Explicit SSP-RK2 march with two-level checkpointing: n_out blocks of k steps
        # -----------------------------------------------------------------------------
        n_steps = a.n_steps
        k = max(1, int(math.sqrt(n_steps)))
        n_out = math.ceil(n_steps / k)

        def _ssprk2(U, th):
            """One SSP-RK2 step of size dt (s) of state U, shape [n_cells, 3]."""
            return ssprk2_step(ops, m, cfg, th, U, dt)

        step = jax.checkpoint(_ssprk2)

        def block(U, th):
            """k checkpointed steps (inner checkpoint level).

            Args:
                U: State, shape [n_cells, 3].
                th: Parameter pytree.

            Returns:
                State after k steps, shape [n_cells, 3].
            """
            return jax.lax.scan(lambda U, _: (step(U, th), None), U, None, length=k)[0]

        blk = jax.checkpoint(block)

        def march(th):
            """March n_out blocks from U0 (outer checkpoint level).

            Args:
                th: Parameter pytree.

            Returns:
                Final state, shape [n_cells, 3].
            """
            U = jax.lax.scan(
                lambda U, _: (blk(U, th), None), jnp.asarray(U0), None, length=n_out
            )[0]
            return U

        if a.method == "explicit_fwd":
            # Forward only: time the march and report the remaining steady residual.
            U = jax.block_until_ready(jax.jit(march)(theta_nn))
            res["fwd_s"] = time.perf_counter() - t0
            res["final_res"] = float(jnp.abs(resid_j(theta_nn, U)).max())
            g = None
            L = float(loss_U(U))
        else:
            # Reverse-mode AD through every step.
            L, g = jax.jit(jax.value_and_grad(lambda th: loss_U(march(th))))(theta_nn)
            jax.block_until_ready(g)
            L = float(L)
        res["n_steps"] = n_out * k

    # ---------------------------------------------------------------------------------
    # 5. Output: timings, loss, gradient norm (gradient vector saved to results/steady/)
    # ---------------------------------------------------------------------------------
    res["total_s"] = time.perf_counter() - t0
    res["loss"] = L
    if g is not None:
        gf = np.asarray(ravel_pytree(g)[0])
        grad_file = f"grad_{a.method}_{tag}_{res.get('n_steps', 0)}.npy"
        np.save(os.path.join(OUT, grad_file), gf)
        res["grad_norm"] = float(np.linalg.norm(gf))
    res["proc_peak_mb"] = process_peak_mb()
    print(json.dumps(res))


if __name__ == "__main__":
    main()
