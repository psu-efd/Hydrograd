"""Unsteady flood wave: explicit SSP-RK2 vs implicit theta-scheme (Newton + sparse LU).

Test problem: a 200 m x 20 m channel starts from the steady state at an inflow of
q_in = 1 m^2/s (true roughness) and the inflow steps to 2 m^2/s at t = 0. The depth field h
(m) is observed at N_SNAP equally spaced snapshots in (0, T_END], and the loss is

    L = sum_snapshots sum_cells (h - h_obs)^2 / (N_SNAP n_cells)   (m^2).

The roughness is an MLP closure; the gradient is dL/d(MLP weights).

Implicit step (theta = 1: backward Euler; 0.5-0.6: Crank-Nicolson-like):
    G(X) = X - U^n - dt [theta R(X) + (1 - theta) R(U^n)] = 0
Gradient of the snapshot loss uses the exact discrete adjoint of each step
(implicit function theorem per step): one transposed sparse solve per
step, so the Newton/linear-solver iterations are never differentiated.

Usage:
    python implicit_unsteady.py --cells 4000 --method implicit --mult 20 --theta 1.0

Prints one JSON line. The initial condition, observations, explicit reference solution and
gradient vectors are cached in results/unsteady/.
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
    """Parse the CLI, run the explicit or implicit solver and its gradient, print JSON."""
    # ---------------------------------------------------------------------------------
    # 1. Command-line options
    # ---------------------------------------------------------------------------------
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

    # ---------------------------------------------------------------------------------
    # 2. Problem setup
    # ---------------------------------------------------------------------------------
    ops = JaxOps("float64")
    mesh = make_channel_mesh(a.cells, 200.0, 20.0)
    m = mesh_arrays(ops, mesh)

    # Inflow steps 1 -> 2 m^2/s: cfg0 gives the initial steady state, cfg the run.
    cfg0 = FlowConfig(q_in=1.0)
    cfg = FlowConfig(q_in=2.0)
    n_true = {"n": jnp.asarray(true_n_field(mesh))}
    mlp_layers = [(jnp.asarray(W), jnp.asarray(b)) for W, b in init_mlp([2, 16, 16, 1])]
    theta_nn = {"mlp": mlp_layers}
    SJ = SparseJacobian(mesh)
    tag = f"{mesh.n_cells}"

    # Initial condition: steady state at q_in = 1 with the true roughness (cached on disk).
    ic_file = os.path.join(OUT, f"ic_{tag}.npz")
    if not os.path.exists(ic_file):

        def r0(th, U):
            """Residual at the initial inflow q_in = 1 m^2/s, shape [n_cells, 3]."""
            return residual(ops, m, cfg0, th, U)

        U_ic, _, _ = newton_steady(
            jax.jit(r0), SJ.make(r0), SJ, n_true, initial_state(mesh, 1.5), 10 * cfl_dt(mesh)
        )
        np.savez(ic_file, U=U_ic)
    U_ic = np.load(ic_file)["U"]

    def resid(th, U):
        """Residual R(U; th) at q_in = 2 m^2/s, shape [n_cells, 3]."""
        return residual(ops, m, cfg, th, U)

    resid_j = jax.jit(resid)
    comp_j = SJ.make(resid)

    # Explicit CFL step (s) for depths and speeds up to 2.5 m and 2.5 m/s.
    dt_cfl = cfl_dt(mesh, h_max=2.5, u_max=2.5)
    res = {"method": a.method, "cells": mesh.n_cells, "dt_cfl": dt_cfl}

    def snapshot_steps(dt):
        """Step counts for a run to T_END with N_SNAP snapshots.

        Args:
            dt: Time step (s).

        Returns:
            Tuple (total steps, steps per snapshot); the total is a multiple of N_SNAP.
        """
        n = int(round(T_END / dt))
        per = n // N_SNAP
        return per * N_SNAP, per

    # ---------------------------------------------------------------------------------
    # 3. Explicit SSP-RK2 simulator (scan over snapshots, two-level checkpointing)
    # ---------------------------------------------------------------------------------
    def run_explicit(th, store_obs=False):
        """Build the explicit simulator.

        Args:
            th: Parameter pytree (unused while building; the simulator takes its own).
            store_obs: Unused.

        Returns:
            Tuple (sim, n): ``sim(th)`` returns the snapshot depths, shape
            [N_SNAP, n_cells] (m); n is the total number of steps.
        """
        n, per = snapshot_steps(dt_cfl)

        # Inner block length k: the largest divisor of per that is <= sqrt(per).
        k = max(1, int(math.sqrt(per)))
        while per % k:
            k -= 1

        def _ssprk2(U, th):
            """One SSP-RK2 step of size dt_cfl (s), state shape [n_cells, 3]."""
            return ssprk2_step(ops, m, cfg, th, U, dt_cfl)

        step = jax.checkpoint(_ssprk2)

        def _block(U, th):
            """k checkpointed steps, state shape [n_cells, 3]."""
            return jax.lax.scan(lambda U, _: (step(U, th), None), U, None, length=k)[0]

        blk = jax.checkpoint(_block)

        def snap(U, th):
            """Advance one snapshot interval (per steps).

            Args:
                U: State, shape [n_cells, 3].
                th: Parameter pytree.

            Returns:
                Tuple (new state, its depth h of shape [n_cells] (m)).
            """
            U = jax.lax.scan(lambda U, _: (blk(U, th), None), U, None, length=per // k)[0]
            return U, U[:, 0]

        def sim(th):
            """Run from U_ic to T_END; returns depths, shape [N_SNAP, n_cells] (m)."""
            return jax.lax.scan(
                lambda U, _: snap(U, th), jnp.asarray(U_ic), None, length=N_SNAP
            )[1]

        return sim, n

    # ---------------------------------------------------------------------------------
    # 4. Implicit theta-scheme: Newton with sparse LU at every step
    # ---------------------------------------------------------------------------------
    def implicit_forward(th, dt, theta):
        """Run the implicit theta-scheme from U_ic to T_END.

        Each step solves G(X) = 0 by Newton (at most 20 iterations, tolerance 1e-10 on
        max |G|) with the matrix I - theta dt J, J = dR/dX; depth is clipped at 1e-4 m.

        Args:
            th: Parameter pytree.
            dt: Time step (s).
            theta: Implicitness weight in [0.5, 1].

        Returns:
            Tuple (hs, Us, newton_its, n): snapshot depths, shape [N_SNAP, n_cells] (m);
            list of all n + 1 states, each shape [n_cells, 3]; total Newton iterations;
            number of steps n.
        """
        n, per = snapshot_steps(dt)
        Us = [U_ic.copy()]
        hs = []
        newton_its = 0
        U = U_ic.copy()
        I = sp.identity(3 * mesh.n_cells, format="csc")
        for s in range(n):
            # Explicit part R(U^n), then Newton from X = U^n.
            Rn = np.asarray(resid_j(th, U))
            X = U.copy()
            for it in range(20):
                G = X - U - dt * (theta * np.asarray(resid_j(th, X)) + (1 - theta) * Rn)
                if np.abs(G).max() < 1e-10:
                    break
                J = SJ.assemble(comp_j(jnp.asarray(X), th))
                newton_matrix = (I - theta * dt * J).tocsc()
                X = X - spla.splu(newton_matrix).solve(G.ravel()).reshape(-1, 3)
                X[:, 0] = np.maximum(X[:, 0], 1e-4)
                newton_its += 1

            # Accept the step; store every state and the depth at snapshot times.
            U = X
            Us.append(U.copy())
            if (s + 1) % per == 0:
                hs.append(U[:, 0].copy())
        return np.array(hs), Us, newton_its, n

    def _vjp_U(th, U, v):
        """Product v^T dR/dU at state U; v and the result have shape [n_cells, 3]."""
        return jax.vjp(lambda X: resid(th, X), U)[1](v)[0]

    def _vjp_th(th, U, v):
        """Product v^T dR/dth at state U, v of shape [n_cells, 3]; returns a pytree."""
        return jax.vjp(lambda t: resid(t, U), th)[1](v)[0]

    vjp_U = jax.jit(_vjp_U)
    vjp_th = jax.jit(_vjp_th)

    def implicit_adjoint(th, Us, hs, h_obs, dt, theta, per):
        """Exact discrete adjoint of the implicit theta-scheme (IFT per step).

        Backwards over the steps s = n, ..., 1, with lam = dL/dU^s (total):

            (I - theta dt J_s)^T mu = lam
            dL/dth += dt mu^T [theta dR/dth(U^s) + (1 - theta) dR/dth(U^{s-1})]
            lam    <- mu + (1 - theta) dt mu^T dR/dU(U^{s-1})

        Args:
            th: Parameter pytree.
            Us: All states of the forward run, n + 1 arrays of shape [n_cells, 3].
            hs: Snapshot depths, shape [N_SNAP, n_cells] (m).
            h_obs: Observed depths, shape [N_SNAP, n_cells] (m).
            dt: Time step (s).
            theta: Implicitness weight.
            per: Steps per snapshot.

        Returns:
            Flat gradient dL/dth, shape [n_params].
        """
        n = len(Us) - 1
        I = sp.identity(3 * mesh.n_cells, format="csc")
        flat0, unravel = ravel_pytree(th)
        g = np.zeros_like(np.asarray(flat0))
        lam = np.zeros((mesh.n_cells, 3))
        scale = 2.0 / (N_SNAP * mesh.n_cells)
        for s in range(n, 0, -1):  # lam = dL/dU^s (total)
            # Loss contribution at snapshot steps.
            if s % per == 0:
                lam[:, 0] += scale * (hs[s // per - 1] - h_obs[s // per - 1])

            # Transposed solve with the step's Newton matrix.
            Xs = jnp.asarray(Us[s])
            Up = jnp.asarray(Us[s - 1])
            J = SJ.assemble(comp_j(Xs, th))
            adjoint_matrix = (I - theta * dt * J).T.tocsc()
            mu = spla.splu(adjoint_matrix).solve(lam.ravel()).reshape(-1, 3)
            mu_j = jnp.asarray(mu)

            # Parameter gradient from both residual evaluations of the step.
            gth = theta * np.asarray(ravel_pytree(vjp_th(th, Xs, mu_j))[0])
            gth += (1 - theta) * np.asarray(ravel_pytree(vjp_th(th, Up, mu_j))[0])
            g += dt * gth

            # Propagate the adjoint to the previous state.
            lam = mu + (1 - theta) * dt * np.asarray(vjp_U(th, Up, mu_j))
        return g

    # ---------------------------------------------------------------------------------
    # 5. Truth observations (explicit, true n) and explicit reference solution file
    # ---------------------------------------------------------------------------------
    obs_file = os.path.join(OUT, f"obs_{tag}.npz")
    if not os.path.exists(obs_file):
        sim, _ = run_explicit(n_true)
        np.savez(obs_file, h_obs=np.asarray(jax.jit(sim)(n_true)))
    h_obs = np.load(obs_file)["h_obs"]
    ref_file = os.path.join(OUT, f"ref_nn_{tag}.npz")

    t0 = time.perf_counter()
    if a.method == "explicit":
        # -----------------------------------------------------------------------------
        # 6a. Explicit: forward (first call and timed call), reference file, AD gradient
        # -----------------------------------------------------------------------------
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

            def lossf(th):
                """Snapshot loss (m^2) of the explicit simulator."""
                return jnp.sum((sim(th) - h_obs) ** 2) / (N_SNAP * mesh.n_cells)

            gf = jax.jit(jax.grad(lossf))
            jax.block_until_ready(gf(theta_nn))
            t1 = time.perf_counter()
            g = np.asarray(ravel_pytree(jax.block_until_ready(gf(theta_nn)))[0])
            res["grad_s"] = time.perf_counter() - t1
    else:
        # -----------------------------------------------------------------------------
        # 6b. Implicit: forward, error against the explicit reference, adjoint gradient
        # -----------------------------------------------------------------------------
        dt = a.mult * dt_cfl
        hs, Us, nits, n = implicit_forward(theta_nn, dt, a.theta)
        res.update(
            fwd_s=time.perf_counter() - t0,
            steps=n,
            dt=dt,
            newton_its=nits,
            mult=a.mult,
            theta=a.theta,
        )
        if os.path.exists(ref_file):
            ref = np.load(ref_file)["hs"]
            res["h_err_max_m"] = float(np.abs(hs - ref).max())
            res["h_err_rms_m"] = float(np.sqrt(np.mean((hs - ref) ** 2)))
        if not a.no_grad:
            t1 = time.perf_counter()
            per = n // N_SNAP
            g = implicit_adjoint(theta_nn, Us, hs, h_obs, dt, a.theta, per)
            res["grad_s"] = time.perf_counter() - t1

    # ---------------------------------------------------------------------------------
    # 7. Output: loss, gradient norm (gradient vector saved to results/unsteady/)
    # ---------------------------------------------------------------------------------
    res["loss"] = float(np.sum((hs - h_obs) ** 2) / (N_SNAP * mesh.n_cells))
    if not a.no_grad:
        grad_file = f"grad_{a.method}_{tag}_{a.mult}_{a.theta}.npy"
        np.save(os.path.join(OUT, grad_file), g)
        res["grad_norm"] = float(np.linalg.norm(g))
    res["proc_peak_mb"] = process_peak_mb()
    print(json.dumps(res))


if __name__ == "__main__":
    main()
