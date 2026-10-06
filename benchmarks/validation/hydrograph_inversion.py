"""Unsteady inversion: recover an inflow hydrograph Q(t) from stage-gauge records.

Compound valley (1 km x 200 m, slope 1e-3, 40 m main channel, floodplains rising 3 %), default
second-order wet/dry scheme, SSPRK3 with CFL steps. Truth: a smooth gamma-shaped flood
(40 -> 350 m^3/s, peak at 45 min, 2 h event) that inundates the floodplains. Observations: WSE at 3 gauges on the channel
centreline every 5 min (optionally with Gaussian noise). Unknowns: Q at 15-min knots,
Q = exp(p) (positive); start: base flow everywhere. Gradients: reverse mode through the
implicit backward-Euler model (dt = 60 s, per-step IFT adjoints); optimizer: L-BFGS (optax)
with a zoom line search (max 8 steps). Observations come from the explicit SSPRK3 model.

Usage: python hydrograph_inversion.py OUT_DIR [NOISE_MM] [N_ITER]
"""
import json, sys, time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

import hydrograd  # noqa: F401
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.io.synthetic import write_channel_case
from hydrograd.solvers.implicit import ImplicitSettings, solve_implicit
from hydrograd.solvers.ode import solve_swe

L, W, S0, N_MAN = 1000.0, 200.0, 1.0e-3, 0.035
NX, NY = 50, 10
T_END = 2.0 * 3600.0
GAUGES_X = (200.0, 500.0, 800.0)
DT_IMPLICIT = 60.0   # s, implicit step of the inversion model (observations every 300 s)


def bed(X, Y):
    r = np.abs(Y - 0.5 * W)
    lateral = np.where(r <= 20.0, 0.0, np.where(r <= 25.0, 2.0 * (r - 20.0) / 5.0, 2.0 + 0.03 * (r - 25.0)))
    return S0 * (L - X) + lateral


def q_true(t):
    """Gamma-shaped flood (m^3/s): base 40, peak 350 at t = 45 min."""
    tp = 2700.0
    return 40.0 + 310.0 * (t / tp) ** 3 * np.exp(3.0 * (1.0 - t / tp))


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    noise_mm = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    n_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 40

    # Truth run: hydrograph sampled every 5 min (finer than the inversion knots).
    t_fine = np.arange(0.0, T_END + 1.0, 300.0)
    t_knots = np.arange(0.0, T_END + 1.0, 900.0)
    h_n = 1.03  # base-flow normal depth in the main channel (m) for Q = 40 m^3/s
    common = dict(length=L, width=W, n_manning=N_MAN, zb_fn=bed, wstill=0.0, dt=0.1, t_end=T_END,
                  solver="SSPRK3()", inlet="INLET-Q", outlet="EXIT-H", wse_out=h_n,
                  ic_wse=lambda x, y: S0 * (L - x) + h_n)
    write_channel_case(out / "truth", NX, NY, Q_series=(t_fine, q_true(t_fine)), **common)
    write_channel_case(out / "inv", NX, NY, Q_series=(t_knots, np.full(len(t_knots), 40.0)), **common)
    truth = setup_case(read_control(out / "truth" / "run_control.json"))
    inv = setup_case(read_control(out / "inv" / "run_control.json"))
    t_obs = jnp.arange(0.0, T_END + 1.0, 300.0)
    cx, cy = inv.mesh.centroid[:, 0], inv.mesh.centroid[:, 1]
    gauge = jnp.asarray([int(np.argmin((cx - gx) ** 2 + (cy - W / 2) ** 2)) for gx in GAUGES_X])

    def stages(case, theta):
        """Explicit SSPRK3 (CFL steps): used for the synthetic observations."""
        sol = solve_swe(case.model, theta, case.U0, case.ode, ts=t_obs)
        return sol.ys[:, 0, gauge] + case.model.wstill[gauge]           # [n_obs, n_gauges] WSE (m)

    # Inversion model: implicit backward Euler with a fixed dt (= observation interval / 5).
    # Fixed steps make the loss smooth in Q (CFL step selection does not change with Q) and the
    # IFT adjoint costs ~1.5 forward runs. Different integrator than the truth (no inverse crime).
    si = ImplicitSettings(0.0, T_END, DT_IMPLICIT, theta=1.0, n_save=len(t_obs) - 1, tol=1e-6)

    def stages_inv(theta):
        _, ys = solve_implicit(inv.model, theta, inv.U0, si)
        return ys[:, 0, gauge] + inv.model.wstill[gauge]

    t0 = time.perf_counter()
    obs = np.asarray(jax.jit(lambda: stages(truth, {}))())
    rng = np.random.default_rng(0)
    obs_noisy = jnp.asarray(obs + noise_mm * 1e-3 * rng.normal(size=obs.shape))
    print(f"truth run (incl. compile): {time.perf_counter() - t0:.1f} s; stage rise at gauges "
          f"{np.round(obs.max(0) - obs[0], 2)} m", flush=True)
    t0 = time.perf_counter(); jax.block_until_ready(jax.jit(lambda: stages(truth, {}))())
    print(f"truth run (compiled): {time.perf_counter() - t0:.1f} s on {jax.devices()[0].device_kind}", flush=True)

    m = inv.model

    def loss(p):
        w = stages_inv({"Q": jnp.exp(p)[None, :]})
        L = jnp.mean((w - obs_noisy) ** 2)
        return jnp.where(jnp.isfinite(L), L, 1.0e6)   # failed solves: reject in the line search

    vg = jax.jit(jax.value_and_grad(loss))
    p = jnp.log(jnp.asarray(m.Q)[0])
    # Zoom (strong Wolfe) line search. A backtracking (Armijo) search is cheaper per trial
    # (forward runs only) but failed here: CFL step selection is not differentiated, so the
    # loss is slightly rough at small parameter changes and Armijo probing stalls/increases it.
    opt = optax.lbfgs(linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=8))
    state = opt.init(p)
    hist = []
    t0 = time.perf_counter()
    step = jax.jit(lambda p, state, v, g: opt.update(g, state, p, value=v, grad=g, value_fn=loss))
    for k in range(n_iter):
        v, g = vg(p)
        hist.append(float(v))
        q = np.exp(np.asarray(p))
        print(f"iter {k:3d}: loss {float(v):.3e} (RMS stage misfit {np.sqrt(float(v)) * 1000:.2f} mm), "
              f"Q peak {q.max():.1f} m^3/s, {time.perf_counter() - t0:.0f} s", flush=True)
        (out / f"progress_noise{noise_mm:g}mm.json").write_text(json.dumps(
            {"iteration": k, "loss_history": hist, "q_current": q.tolist(), "elapsed_s": time.perf_counter() - t0}))
        upd, state = step(p, state, v, g)
        p = optax.apply_updates(p, upd)
        if float(optax.tree.norm(g)) < 1e-12:
            break
    q_inv = np.exp(np.asarray(p))
    q_ref = q_true(np.asarray(t_knots))
    err = np.sqrt(np.mean((q_inv - q_ref) ** 2))
    summary = {"noise_mm": noise_mm, "iterations": len(hist), "wall_s": time.perf_counter() - t0,
               "cells": m.n_cells, "final_rms_stage_misfit_mm": float(np.sqrt(hist[-1]) * 1000),
               "q_rms_error_m3s": float(err), "q_peak_true": float(q_ref.max()), "q_peak_inv": float(q_inv.max()),
               "t_knots_h": (np.asarray(t_knots) / 3600).tolist(), "q_inv": q_inv.tolist(), "q_true_knots": q_ref.tolist(),
               "loss_history": hist}
    (out / f"summary_noise{noise_mm:g}mm.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}), flush=True)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    w_fit = np.asarray(jax.jit(lambda p: stages_inv({"Q": jnp.exp(p)[None, :]}))(p))
    w0 = np.asarray(jax.jit(lambda: stages_inv({}))())
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    tt = np.linspace(0, T_END, 400)
    ax[0].plot(tt / 3600, q_true(tt), "k-", lw=2, label="true inflow")
    ax[0].plot(np.asarray(t_knots) / 3600, np.full(len(t_knots), 40.0), "C7--", label="initial guess")
    ax[0].plot(np.asarray(t_knots) / 3600, q_inv, "C3o-", label=f"inverted (15-min knots)")
    ax[0].set_xlabel("t (h)"); ax[0].set_ylabel("Q (m$^3$/s)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    for j, gx in enumerate(GAUGES_X):
        ax[1].plot(np.asarray(t_obs) / 3600, np.asarray(obs_noisy)[:, j], f"C{j}.", ms=4, label=f"gauge x = {gx:.0f} m (obs)")
        ax[1].plot(np.asarray(t_obs) / 3600, w_fit[:, j], f"C{j}-", lw=1.2)
        ax[1].plot(np.asarray(t_obs) / 3600, w0[:, j], f"C{j}:", lw=1)
    ax[1].set_xlabel("t (h)"); ax[1].set_ylabel("WSE (m)"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    ax[1].set_title("dots: observed, solid: fitted, dotted: initial", fontsize=9)
    ax[2].semilogy(hist, "k.-"); ax[2].set_xlabel("L-BFGS iteration"); ax[2].set_ylabel("loss (m$^2$)"); ax[2].grid(alpha=.3)
    fig.suptitle(f"Hydrograph inversion from 3 stage gauges ({m.n_cells} cells, wet/dry, noise {noise_mm:g} mm)")
    fig.tight_layout(); fig.savefig(out / f"hydrograph_inversion_noise{noise_mm:g}mm.png", dpi=140)


if __name__ == "__main__":
    main()
