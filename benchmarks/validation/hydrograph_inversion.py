"""Unsteady inversion: recover an inflow hydrograph Q(t) from stage-gauge records.

Usage: python hydrograph_inversion.py OUT_DIR [NOISE_MM] [N_ITER]   (defaults 0 mm, 40)

Case: compound valley 1 km x 200 m (NX x NY = 50 x 10 cells), bed slope 1e-3, 40 m main
channel, floodplains rising 3 %, Manning n = 0.035; default second-order wet/dry scheme.

Truth: a smooth gamma-shaped flood (40 -> 350 m^3/s, peak at 45 min, 2 h event) that
inundates the floodplains, run with explicit SSPRK3 (CFL steps), hydrograph sampled every
5 min. Observations: WSE at 3 gauges on the channel centreline every 5 min, optionally with
Gaussian noise of standard deviation NOISE_MM (mm).

Inversion: unknowns are Q at 15-min knots, parameterized as Q = exp(p) (positive); start:
base flow (40 m^3/s) everywhere. Loss: mean squared stage misfit (m^2) over gauges and
times. Gradients: reverse mode through the implicit backward-Euler model (dt = 60 s,
per-step IFT adjoints); optimizer: L-BFGS (optax) with a zoom line search (max 8 steps).
Using a different integrator for the truth and the inversion avoids an inverse crime.

Outputs in OUT_DIR: progress and summary JSON files and a three-panel figure.
"""
import json
import sys
import time
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

# Valley length (m), width (m), bed slope (-), Manning n (s/m^(1/3)).
L, W, S0, N_MAN = 1000.0, 200.0, 1.0e-3, 0.035
# Mesh cells along and across the valley.
NX, NY = 50, 10
# Event duration (s).
T_END = 2.0 * 3600.0
# Gauge x positions on the centreline (m).
GAUGES_X = (200.0, 500.0, 800.0)
DT_IMPLICIT = 60.0   # s, implicit step of the inversion model (observations every 300 s)


def bed(X, Y):
    """z_b (m): valley slope + trapezoidal main channel + inclined floodplains.

    Args:
        X: streamwise coordinates (m), any shape.
        Y: cross-valley coordinates (m), same shape as X.

    Returns:
        Bed elevation (m), same shape as X.
    """
    # Distance from the valley centreline (m).
    r = np.abs(Y - 0.5 * W)

    # Lateral profile: flat channel bed, 2 m bank over 5 m, then the 3 % floodplain.
    floodplain = 2.0 + 0.03 * (r - 25.0)
    bank_or_floodplain = np.where(r <= 25.0, 2.0 * (r - 20.0) / 5.0, floodplain)
    lateral = np.where(r <= 20.0, 0.0, bank_or_floodplain)
    return S0 * (L - X) + lateral


def q_true(t):
    """Gamma-shaped flood (m^3/s): base 40, peak 350 at t = 45 min.

    Q(t) = 40 + 310 (t / t_p)^3 exp(3 (1 - t / t_p)),  t_p = 2700 s.

    Args:
        t: time (s), any shape.

    Returns:
        Discharge (m^3/s), same shape as t.
    """
    tp = 2700.0
    return 40.0 + 310.0 * (t / tp) ** 3 * np.exp(3.0 * (1.0 - t / tp))


def main():
    """Make the synthetic observations, invert the hydrograph, write outputs."""
    # 1. Command line.
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    noise_mm = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    n_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 40

    # 2. Cases. Truth run: hydrograph sampled every 5 min (finer than the inversion knots).
    t_fine = np.arange(0.0, T_END + 1.0, 300.0)
    t_knots = np.arange(0.0, T_END + 1.0, 900.0)
    h_n = 1.03  # base-flow normal depth in the main channel (m) for Q = 40 m^3/s

    def initial_wse(x, y):
        """Initial WSE (m): base-flow normal depth h_n above the sloping valley bed."""
        return S0 * (L - x) + h_n

    common = dict(length=L, width=W, n_manning=N_MAN, zb_fn=bed, wstill=0.0, dt=0.1, t_end=T_END,
                  solver="SSPRK3()", inlet="INLET-Q", outlet="EXIT-H", wse_out=h_n,
                  ic_wse=initial_wse)
    write_channel_case(out / "truth", NX, NY, Q_series=(t_fine, q_true(t_fine)), **common)
    write_channel_case(out / "inv", NX, NY, Q_series=(t_knots, np.full(len(t_knots), 40.0)),
                       **common)
    truth = setup_case(read_control(out / "truth" / "run_control.json"))
    inv = setup_case(read_control(out / "inv" / "run_control.json"))

    # 3. Observation times (s) and gauge cells (nearest centroid to each centreline point).
    t_obs = jnp.arange(0.0, T_END + 1.0, 300.0)
    cx = inv.mesh.centroid[:, 0]
    cy = inv.mesh.centroid[:, 1]
    gauge = jnp.asarray([int(np.argmin((cx - gx) ** 2 + (cy - W / 2) ** 2)) for gx in GAUGES_X])

    def stages(case, theta):
        """Explicit SSPRK3 (CFL steps): used for the synthetic observations.

        Args:
            case: set-up case (model, initial state, ODE settings).
            theta: parameter overrides (dict, e.g. {"Q": ...}); {} for the case values.

        Returns:
            WSE at the gauges (m), shape [n_obs, n_gauges].
        """
        sol = solve_swe(case.model, theta, case.U0, case.ode, ts=t_obs)
        return sol.ys[:, 0, gauge] + case.model.wstill[gauge]           # [n_obs, n_gauges] WSE (m)

    # Inversion model: implicit backward Euler with a fixed dt (= observation interval / 5).
    # Fixed steps make the loss smooth in Q (CFL step selection does not change with Q) and the
    # IFT adjoint costs ~1.5 forward runs. Different integrator than the truth (no inverse crime).
    si = ImplicitSettings(0.0, T_END, DT_IMPLICIT, theta=1.0, n_save=len(t_obs) - 1, tol=1e-6)

    def stages_inv(theta):
        """Gauge WSE (m), shape [n_obs, n_gauges], from the implicit inversion model.

        Args:
            theta: parameter overrides, e.g. {"Q": hydrograph (m^3/s), shape [1, n_knots]}.
        """
        _, ys = solve_implicit(inv.model, theta, inv.U0, si)
        return ys[:, 0, gauge] + inv.model.wstill[gauge]

    # 4. Synthetic observations (truth run), plus optional Gaussian noise (seed 0).
    t0 = time.perf_counter()
    obs = np.asarray(jax.jit(lambda: stages(truth, {}))())
    rng = np.random.default_rng(0)
    obs_noisy = jnp.asarray(obs + noise_mm * 1e-3 * rng.normal(size=obs.shape))
    print(f"truth run (incl. compile): {time.perf_counter() - t0:.1f} s; stage rise at gauges "
          f"{np.round(obs.max(0) - obs[0], 2)} m", flush=True)
    t0 = time.perf_counter()
    jax.block_until_ready(jax.jit(lambda: stages(truth, {}))())
    print(f"truth run (compiled): {time.perf_counter() - t0:.1f} s on "
          f"{jax.devices()[0].device_kind}", flush=True)

    m = inv.model

    # 5. Loss and its gradient.
    def loss(p):
        """Mean squared stage misfit (m^2) for log-discharges p (shape [n_knots])."""
        w = stages_inv({"Q": jnp.exp(p)[None, :]})
        misfit = jnp.mean((w - obs_noisy) ** 2)
        return jnp.where(jnp.isfinite(misfit), misfit, 1.0e6)   # failed solves: reject in the line search

    vg = jax.jit(jax.value_and_grad(loss))
    p = jnp.log(jnp.asarray(m.Q)[0])

    # 6. L-BFGS. Zoom (strong Wolfe) line search. A backtracking (Armijo) search is cheaper per
    # trial (forward runs only) but failed here: CFL step selection is not differentiated, so the
    # loss is slightly rough at small parameter changes and Armijo probing stalls/increases it.
    opt = optax.lbfgs(linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=8))
    state = opt.init(p)
    hist = []
    t0 = time.perf_counter()

    def lbfgs_update(p, state, v, g):
        """One L-BFGS update: returns (parameter increment, new optimizer state)."""
        return opt.update(g, state, p, value=v, grad=g, value_fn=loss)

    step = jax.jit(lbfgs_update)
    for k in range(n_iter):
        # Loss, gradient and progress report.
        v, g = vg(p)
        hist.append(float(v))
        q = np.exp(np.asarray(p))
        print(f"iter {k:3d}: loss {float(v):.3e} (RMS stage misfit {np.sqrt(float(v)) * 1000:.2f} mm), "
              f"Q peak {q.max():.1f} m^3/s, {time.perf_counter() - t0:.0f} s", flush=True)
        progress = {"iteration": k, "loss_history": hist, "q_current": q.tolist(),
                    "elapsed_s": time.perf_counter() - t0}
        (out / f"progress_noise{noise_mm:g}mm.json").write_text(json.dumps(progress))

        # Parameter update; stop on a vanishing gradient.
        upd, state = step(p, state, v, g)
        p = optax.apply_updates(p, upd)
        if float(optax.tree.norm(g)) < 1e-12:
            break

    # 7. Summary: hydrograph error at the knots (m^3/s).
    q_inv = np.exp(np.asarray(p))
    q_ref = q_true(np.asarray(t_knots))
    err = np.sqrt(np.mean((q_inv - q_ref) ** 2))
    t_knots_h = (np.asarray(t_knots) / 3600).tolist()
    summary = {"noise_mm": noise_mm, "iterations": len(hist), "wall_s": time.perf_counter() - t0,
               "cells": m.n_cells, "final_rms_stage_misfit_mm": float(np.sqrt(hist[-1]) * 1000),
               "q_rms_error_m3s": float(err), "q_peak_true": float(q_ref.max()),
               "q_peak_inv": float(q_inv.max()),
               "t_knots_h": t_knots_h, "q_inv": q_inv.tolist(), "q_true_knots": q_ref.tolist(),
               "loss_history": hist}
    (out / f"summary_noise{noise_mm:g}mm.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}), flush=True)

    # 8. Figure: hydrographs, gauge stages, loss history.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def fitted_stages(p):
        """Gauge WSE (m) of the inversion model for log-discharges p."""
        return stages_inv({"Q": jnp.exp(p)[None, :]})

    def initial_stages():
        """Gauge WSE (m) of the inversion model with the initial (base-flow) hydrograph."""
        return stages_inv({})

    w_fit = np.asarray(jax.jit(fitted_stages)(p))
    w0 = np.asarray(jax.jit(initial_stages)())
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    t_obs_h = np.asarray(t_obs) / 3600
    t_knots_hours = np.asarray(t_knots) / 3600

    # Panel 1: true, initial and inverted hydrographs.
    tt = np.linspace(0, T_END, 400)
    ax[0].plot(tt / 3600, q_true(tt), "k-", lw=2, label="true inflow")
    ax[0].plot(t_knots_hours, np.full(len(t_knots), 40.0), "C7--", label="initial guess")
    ax[0].plot(t_knots_hours, q_inv, "C3o-", label="inverted (15-min knots)")
    ax[0].set_xlabel("t (h)")
    ax[0].set_ylabel("Q (m$^3$/s)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)

    # Panel 2: observed, fitted and initial gauge stages.
    for j, gx in enumerate(GAUGES_X):
        ax[1].plot(t_obs_h, np.asarray(obs_noisy)[:, j], f"C{j}.", ms=4,
                   label=f"gauge x = {gx:.0f} m (obs)")
        ax[1].plot(t_obs_h, w_fit[:, j], f"C{j}-", lw=1.2)
        ax[1].plot(t_obs_h, w0[:, j], f"C{j}:", lw=1)
    ax[1].set_xlabel("t (h)")
    ax[1].set_ylabel("WSE (m)")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=.3)
    ax[1].set_title("dots: observed, solid: fitted, dotted: initial", fontsize=9)

    # Panel 3: loss history.
    ax[2].semilogy(hist, "k.-")
    ax[2].set_xlabel("L-BFGS iteration")
    ax[2].set_ylabel("loss (m$^2$)")
    ax[2].grid(alpha=.3)
    fig.suptitle(f"Hydrograph inversion from 3 stage gauges ({m.n_cells} cells, wet/dry, "
                 f"noise {noise_mm:g} mm)")
    fig.tight_layout()
    fig.savefig(out / f"hydrograph_inversion_noise{noise_mm:g}mm.png", dpi=140)


if __name__ == "__main__":
    main()
