"""Unsteady flood wave: explicit SSP-RK2 vs implicit theta-scheme (Newton + sparse LU).

Implicit step (theta = 1: backward Euler; 0.5-0.6: Crank-Nicolson-like):
    G(X) = X - U^n - dt [theta R(X) + (1 - theta) R(U^n)] = 0
Gradient of the snapshot loss uses the exact discrete adjoint of each step
(implicit function theorem per step): one transposed sparse solve per
step, so the Newton/linear-solver iterations are never differentiated.

Example: python implicit_unsteady.py --cells 4000 --method implicit --mult 20 --theta 1.0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from backends import JaxOps, cfl_dt, init_mlp, initial_state, mesh_arrays, true_n_field
from bench_explicit import process_peak_mb
from jacobian import SparseJacobian
from mesh import make_channel_mesh
from physics import FlowConfig, residual, ssprk2_step
from steady_ift import newton_steady

OUT = os.path.join(os.path.dirname(__file__), "results", "unsteady")
T_END = 300.0  # simulated time (s)
N_SNAP = 10  # observation snapshots, equally spaced in (0, T_END]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["explicit", "implicit"], required=True)
    p.add_argument("--cells", type=int, default=4000)
    p.add_argument("--mult", type=float, default=20.0, help="implicit dt / explicit CFL dt")
    p.add_argument("--theta", type=float, default=1.0)
    p.add_argument("--no_grad", action="store_true")
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)

    import jax
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    ops = JaxOps("float64")
    mesh = make_channel_mesh(a.cells, 200.0, 20.0)
    m = mesh_arrays(ops, mesh)
    cfg0, cfg = FlowConfig(q_in=1.0), FlowConfig(q_in=2.0)  # inflow steps 1 -> 2 m^2/s
    n_true = {"n": jnp.asarray(true_n_field(mesh))}
    theta_nn = {"mlp": [(jnp.asarray(W), jnp.asarray(b)) for W, b in init_mlp([2, 16, 16, 1])]}
    SJ = SparseJacobian(mesh)
    tag = f"{mesh.n_cells}"

    # initial condition: steady state at q_in = 1 with the true roughness
    ic_file = os.path.join(OUT, f"ic_{tag}.npz")
    if not os.path.exists(ic_file):
        r0 = lambda th, U: residual(ops, m, cfg0, th, U)
        U_ic, _, _ = newton_steady(jax.jit(r0), SJ.make(r0), SJ, n_true,
                                   initial_state(mesh, 1.5), 10 * cfl_dt(mesh))
        np.savez(ic_file, U=U_ic)
    U_ic = np.load(ic_file)["U"]

    resid = lambda th, U: residual(ops, m, cfg, th, U)
    resid_j = jax.jit(resid)
    comp_j = SJ.make(resid)
    dt_cfl = cfl_dt(mesh, h_max=2.5, u_max=2.5)
    res = {"method": a.method, "cells": mesh.n_cells, "dt_cfl": dt_cfl}

    def snapshot_steps(dt):
        n = int(round(T_END / dt))
        per = n // N_SNAP
        return per * N_SNAP, per

    # ---------------- explicit ----------------------------------------
    def run_explicit(th, store_obs=False):
        n, per = snapshot_steps(dt_cfl)
        k = max(1, int(math.sqrt(per)))
        while per % k:
            k -= 1
        step = jax.checkpoint(lambda U, th: ssprk2_step(ops, m, cfg, th, U, dt_cfl))
        blk = jax.checkpoint(lambda U, th: jax.lax.scan(lambda U, _: (step(U, th), None),
                                                       U, None, length=k)[0])

        def snap(U, th):
            U = jax.lax.scan(lambda U, _: (blk(U, th), None), U, None, length=per // k)[0]
            return U, U[:, 0]

        def sim(th):
            return jax.lax.scan(lambda U, _: snap(U, th), jnp.asarray(U_ic), None,
                                length=N_SNAP)[1]
        return sim, n

    # ---------------- implicit ----------------------------------------
    def implicit_forward(th, dt, theta):
        n, per = snapshot_steps(dt)
        Us, hs, newton_its = [U_ic.copy()], [], 0
        U = U_ic.copy()
        I = sp.identity(3 * mesh.n_cells, format="csc")
        for s in range(n):
            Rn = np.asarray(resid_j(th, U))
            X = U.copy()
            for it in range(20):
                G = X - U - dt * (theta * np.asarray(resid_j(th, X)) + (1 - theta) * Rn)
                if np.abs(G).max() < 1e-10:
                    break
                J = SJ.assemble(comp_j(jnp.asarray(X), th))
                X = X - spla.splu((I - theta * dt * J).tocsc()).solve(G.ravel()).reshape(-1, 3)
                X[:, 0] = np.maximum(X[:, 0], 1e-4)
                newton_its += 1
            U = X
            Us.append(U.copy())
            if (s + 1) % per == 0:
                hs.append(U[:, 0].copy())
        return np.array(hs), Us, newton_its, n

    vjp_U = jax.jit(lambda th, U, v: jax.vjp(lambda X: resid(th, X), U)[1](v)[0])
    vjp_th = jax.jit(lambda th, U, v: jax.vjp(lambda t: resid(t, U), th)[1](v)[0])

    def implicit_adjoint(th, Us, hs, h_obs, dt, theta, per):
        n = len(Us) - 1
        I = sp.identity(3 * mesh.n_cells, format="csc")
        flat0, unravel = ravel_pytree(th)
        g = np.zeros_like(np.asarray(flat0))
        lam = np.zeros((mesh.n_cells, 3))
        scale = 2.0 / (N_SNAP * mesh.n_cells)
        for s in range(n, 0, -1):  # lam = dL/dU^s (total)
            if s % per == 0:
                lam[:, 0] += scale * (hs[s // per - 1] - h_obs[s // per - 1])
            Xs, Up = jnp.asarray(Us[s]), jnp.asarray(Us[s - 1])
            J = SJ.assemble(comp_j(Xs, th))
            mu = spla.splu((I - theta * dt * J).T.tocsc()).solve(lam.ravel()).reshape(-1, 3)
            mu_j = jnp.asarray(mu)
            gth = theta * np.asarray(ravel_pytree(vjp_th(th, Xs, mu_j))[0])
            gth += (1 - theta) * np.asarray(ravel_pytree(vjp_th(th, Up, mu_j))[0])
            g += dt * gth
            lam = mu + (1 - theta) * dt * np.asarray(vjp_U(th, Up, mu_j))
        return g

    # truth observations (explicit, true n) and explicit reference solution
    obs_file = os.path.join(OUT, f"obs_{tag}.npz")
    if not os.path.exists(obs_file):
        sim, _ = run_explicit(n_true)
        np.savez(obs_file, h_obs=np.asarray(jax.jit(sim)(n_true)))
    h_obs = np.load(obs_file)["h_obs"]
    ref_file = os.path.join(OUT, f"ref_nn_{tag}.npz")

    t0 = time.perf_counter()
    if a.method == "explicit":
        sim, n = run_explicit(theta_nn)
        sim_j = jax.jit(sim)
        hs = np.asarray(jax.block_until_ready(sim_j(theta_nn)))
        res["fwd_first_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        hs = np.asarray(jax.block_until_ready(sim_j(theta_nn)))
        res["fwd_s"] = time.perf_counter() - t1
        np.savez(ref_file, hs=hs)
        res.update(steps=n, dt=dt_cfl)
        if not a.no_grad:
            lossf = lambda th: jnp.sum((sim(th) - h_obs) ** 2) / (N_SNAP * mesh.n_cells)
            gf = jax.jit(jax.grad(lossf))
            jax.block_until_ready(gf(theta_nn))
            t1 = time.perf_counter()
            g = np.asarray(ravel_pytree(jax.block_until_ready(gf(theta_nn)))[0])
            res["grad_s"] = time.perf_counter() - t1
    else:
        dt = a.mult * dt_cfl
        hs, Us, nits, n = implicit_forward(theta_nn, dt, a.theta)
        res.update(fwd_s=time.perf_counter() - t0, steps=n, dt=dt, newton_its=nits,
                   mult=a.mult, theta=a.theta)
        if os.path.exists(ref_file):
            ref = np.load(ref_file)["hs"]
            res["h_err_max_m"] = float(np.abs(hs - ref).max())
            res["h_err_rms_m"] = float(np.sqrt(np.mean((hs - ref) ** 2)))
        if not a.no_grad:
            t1 = time.perf_counter()
            per = n // N_SNAP
            g = implicit_adjoint(theta_nn, Us, hs, h_obs, dt, a.theta, per)
            res["grad_s"] = time.perf_counter() - t1
    res["loss"] = float(np.sum((hs - h_obs) ** 2) / (N_SNAP * mesh.n_cells))
    if not a.no_grad:
        np.save(os.path.join(OUT, f"grad_{a.method}_{tag}_{a.mult}_{a.theta}.npy"), g)
        res["grad_norm"] = float(np.linalg.norm(g))
    res["proc_peak_mb"] = process_peak_mb()
    print(json.dumps(res))


if __name__ == "__main__":
    main()
