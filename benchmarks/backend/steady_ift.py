"""Steady-state gradients: implicit function theorem vs backprop-through-pseudo-time.

Methods (one per process for clean memory numbers):
  newton_ift    implicit pseudo-transient Newton steady solve (sparse LU) +
                IFT adjoint with the exact sparse Jacobian (one transposed solve)
  newton_gmres  same forward, adjoint via matrix-free GMRES on VJPs (no Jacobian)
  explicit_bptt explicit SSP-RK2 march to steady state + reverse-mode AD
                through every step (two-level checkpointing)
  explicit_fwd  explicit march to steady state only (forward cost reference)

Example: python steady_ift.py --method newton_ift --cells 4000
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


def newton_steady(resid_j, comp_j, SJ, theta, U, dtau0: float, tol: float = 1e-10,
                  max_it: int = 200):
    """Pseudo-transient continuation Newton (SER) for R(U; theta) = 0."""
    import scipy.sparse as sp

    r0 = None
    for it in range(max_it):
        R = np.asarray(resid_j(theta, U))
        rn = float(np.abs(R).max())
        r0 = rn if r0 is None else r0
        if rn < tol:
            return U, it, rn
        J = SJ.assemble(comp_j(U, theta))
        dtau = min(dtau0 * (r0 / rn) ** 1.5, 1e8)
        M = (sp.identity(J.shape[0], format="csc") / dtau - J).tocsc()
        dU = spla.splu(M).solve(R.ravel()).reshape(-1, 3)
        U = U + dU
        U[:, 0] = np.maximum(U[:, 0], 1e-4)
    return U, max_it, rn


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True,
                   choices=["newton_ift", "newton_gmres", "explicit_bptt", "explicit_fwd"])
    p.add_argument("--cells", type=int, default=4000)
    p.add_argument("--length", type=float, default=200.0)
    p.add_argument("--width", type=float, default=20.0)
    p.add_argument("--n_steps", type=int, default=0, help="explicit steps (0 = from file)")
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)

    import jax
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    ops = JaxOps("float64")
    mesh = make_channel_mesh(a.cells, a.length, a.width)
    cfg = FlowConfig(q_in=1.0, wse_out=1.5)
    m = mesh_arrays(ops, mesh)
    dt = cfl_dt(mesh)
    U0 = initial_state(mesh, 1.5)
    tag = f"{mesh.n_cells}"
    resid = lambda theta, U: residual(ops, m, cfg, theta, U)
    resid_j = jax.jit(resid)
    theta_nn = {"mlp": [(jnp.asarray(W), jnp.asarray(b)) for W, b in init_mlp([2, 16, 16, 1])]}
    res = {"method": a.method, "cells": mesh.n_cells, "dt": dt}

    obs_file = os.path.join(OUT, f"obs_{tag}.npz")
    if not os.path.exists(obs_file):  # truth: steady state with the true n field
        SJ = SparseJacobian(mesh)
        comp_j = SJ.make(resid)
        Ut, _, _ = newton_steady(resid_j, comp_j, SJ, {"n": jnp.asarray(true_n_field(mesh))},
                                 U0.copy(), 10 * dt)
        np.savez(obs_file, h_obs=Ut[:, 0])
    h_obs = jnp.asarray(np.load(obs_file)["h_obs"])
    loss_U = lambda U: jnp.mean((U[:, 0] - h_obs) ** 2)

    t0 = time.perf_counter()
    if a.method.startswith("newton"):
        SJ = SparseJacobian(mesh)
        comp_j = SJ.make(resid)
        res["n_colors"] = SJ.n_colors
        t1 = time.perf_counter()
        U, iters, rn = newton_steady(resid_j, comp_j, SJ, theta_nn, U0.copy(), 10 * dt)
        res.update(fwd_s=time.perf_counter() - t1, newton_iters=iters, final_res=rn)
        t1 = time.perf_counter()
        Uj = jnp.asarray(U)
        gU = np.array(jax.grad(loss_U)(Uj)).ravel()
        if a.method == "newton_ift":
            J = SJ.assemble(comp_j(Uj, theta_nn))
            lam = spla.splu(J.T.tocsc()).solve(gU)
        else:
            _, vjp_U = jax.vjp(lambda X: resid(theta_nn, X), Uj)
            vjp_j = jax.jit(lambda v: vjp_U(v)[0])
            count = [0]
            op = spla.LinearOperator((gU.size, gU.size), dtype=np.float64,
                                     matvec=lambda v: np.array(vjp_j(jnp.asarray(v.reshape(-1, 3)))).ravel())
            cb = lambda _: count.__setitem__(0, count[0] + 1)
            lam, info = spla.gmres(op, gU, rtol=1e-8, restart=200, maxiter=50,
                                   callback=cb, callback_type="pr_norm")
            res.update(gmres_iters=count[0], gmres_info=int(info))
        _, vjp_th = jax.vjp(lambda th: resid(th, Uj), theta_nn)
        g = vjp_th(jnp.asarray(-lam.reshape(-1, 3)))[0]
        res["adjoint_s"] = time.perf_counter() - t1
        L = float(loss_U(Uj))
    else:
        n_steps = a.n_steps
        k = max(1, int(math.sqrt(n_steps)))
        n_out = math.ceil(n_steps / k)
        step = jax.checkpoint(lambda U, th: ssprk2_step(ops, m, cfg, th, U, dt))

        def block(U, th):
            return jax.lax.scan(lambda U, _: (step(U, th), None), U, None, length=k)[0]

        blk = jax.checkpoint(block)

        def march(th):
            U = jax.lax.scan(lambda U, _: (blk(U, th), None), jnp.asarray(U0), None,
                             length=n_out)[0]
            return U

        if a.method == "explicit_fwd":
            U = jax.block_until_ready(jax.jit(march)(theta_nn))
            res["fwd_s"] = time.perf_counter() - t0
            res["final_res"] = float(jnp.abs(resid_j(theta_nn, U)).max())
            g, L = None, float(loss_U(U))
        else:
            L, g = jax.jit(jax.value_and_grad(lambda th: loss_U(march(th))))(theta_nn)
            jax.block_until_ready(g)
            L = float(L)
        res["n_steps"] = n_out * k
    res["total_s"] = time.perf_counter() - t0
    res["loss"] = L
    if g is not None:
        gf = np.asarray(ravel_pytree(g)[0])
        np.save(os.path.join(OUT, f"grad_{a.method}_{tag}_{res.get('n_steps', 0)}.npy"), gf)
        res["grad_norm"] = float(np.linalg.norm(gf))
    res["proc_peak_mb"] = process_peak_mb()
    print(json.dumps(res))


if __name__ == "__main__":
    main()
