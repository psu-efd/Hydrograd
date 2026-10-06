"""Unsteady roughness calibration on the Savannah River mesh from stage-gauge records.

Real SRH-2D mesh (1,306 cells, 5 Manning zones), IC = SRH-2D steady state at 187.4 m^3/s.
A flood hydrograph (187.4 -> 450 -> 187.4 m^3/s over 6 h) enters at the inlet; the exit WSE
is held at 29.3 m. Truth: explicit second-order SSPRK3 with the true zone n. Observations:
WSE at N_GAUGES cells along the reach every 10 min (optionally noisy). Unknowns: n of zones
1-5 (zone 0 has no cells), start 0.03. Inversion model: implicit backward Euler (dt = 30 s,
per-step IFT adjoints); optimizer: L-BFGS (optax, zoom line search).

Usage: python savannah_unsteady_inversion.py JL_CASE_DIR OUT_DIR [NOISE_MM] [N_ITER] [GAUGES] [DT]
(JL_CASE_DIR: Hydrograd.jl examples/SWE_2D/forward_simulation/Savannah_River; it is copied.)
GAUGES: "bins" (default; deepest cell in each of N_GAUGES streamwise bins, which puts 5 of 6
gauges in the main-channel zone 3) or "zones" (GAUGES_PER_ZONE cells of median depth per zone,
spread along x, so every roughness zone is observed). DT: implicit step (s), default 30.
"""
import json, shutil, sys, time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

import hydrograd  # noqa: F401
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.solvers.implicit import ImplicitSettings, solve_implicit
from hydrograd.solvers.ode import solve_swe

T_END = 6.0 * 3600.0
DT_OBS = 600.0
DT_IMPLICIT = 30.0
N_GAUGES = 6
GAUGES_PER_ZONE = 2
T_HYD = np.array([0.0, 0.5, 2.0, 4.5, 6.0]) * 3600.0
Q_HYD = np.array([187.4, 187.4, 450.0, 187.4, 187.4])


def prepare(src: Path, d: Path) -> None:
    """Copy the Julia case; switch to an unsteady run with the hydrograph and SSPRK3."""
    shutil.copytree(src, d, ignore=shutil.ignore_patterns("*.vtk", "*.gif", "SRH_2D", "*results*"))
    rows = [f"{t / 3600:.17g} {q:.17g}" for t, q in zip(T_HYD, Q_HYD)]
    (d / "flood.xys").write_text('XY 1 5 0 0 0 0 0 0 "flood"\n' + "\n".join(rows) + "\n")
    hydro = (d / "savana_SI.srhhydro").read_text().replace("IQParams 1 187.4 SI CONVEYANCE",
                                                          "IQParams 1 flood.xys SI CONVEYANCE")
    (d / "savana_SI.srhhydro").write_text(hydro)
    c = json.loads((d / "run_control.json").read_text())
    c["time_settings"].update(tspan=[0.0, T_END], dt=0.1)
    c["forward_simulation_options"].update(forward_simulation_ode_solver="SSPRK3()",
                                           forward_simulation_nSave=int(T_END / DT_OBS))
    (d / "run_control.json").write_text(json.dumps(c, indent=1))


def pick_gauges(h0: np.ndarray, x: np.ndarray, zone: np.ndarray, layout: str) -> list[int]:
    """Gauge cell indices.

    Args:
        h0: initial depth per cell (m); x: streamwise centroid coordinate (m);
        zone: material id per cell; layout: "bins" or "zones" (see module docstring).
    """
    if layout == "bins":
        edges = np.quantile(x, np.linspace(0.05, 0.95, N_GAUGES + 1))
        return [int(np.flatnonzero((x >= a) & (x < b))[np.argmax(h0[(x >= a) & (x < b)])])
                for a, b in zip(edges[:-1], edges[1:])]
    out = []
    for k in np.unique(zone):
        cells = np.flatnonzero(zone == k)
        for part in np.array_split(cells[np.argsort(x[cells])], GAUGES_PER_ZONE):  # spread along x
            out.append(int(part[np.argmin(np.abs(h0[part] - np.median(h0[part])))]))
    return out


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    noise_mm = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    n_iter = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    layout = sys.argv[5] if len(sys.argv) > 5 else "bins"
    dt_impl = float(sys.argv[6]) if len(sys.argv) > 6 else DT_IMPLICIT
    out.mkdir(parents=True, exist_ok=True)
    d = out / "case"
    if not d.exists():
        prepare(src, d)
    case = setup_case(read_control(d / "run_control.json"))
    m = case.model
    n_true = np.asarray(m.n_zone)
    t_obs = jnp.arange(0.0, T_END + 1.0, DT_OBS)

    h0 = np.asarray(case.U0[0] + m.hstill)
    gauge = jnp.asarray(pick_gauges(h0, case.mesh.centroid[:, 0], np.asarray(m.mat_id), layout))

    t0 = time.perf_counter()
    sol = jax.jit(lambda: solve_swe(m, {}, case.U0, case.ode, ts=t_obs))()
    obs = np.asarray(sol.ys[:, 0, gauge] + m.wstill[gauge])
    print(f"truth (explicit SSPRK3, order {m.order}): {time.perf_counter() - t0:.1f} s, "
          f"{int(sol.stats['num_steps'])} steps; stage rise {np.round(obs.max(0) - obs[0], 2)} m", flush=True)
    obs = jnp.asarray(obs + noise_mm * 1e-3 * np.random.default_rng(0).normal(size=obs.shape))

    si = ImplicitSettings(0.0, T_END, dt_impl, theta=1.0, n_save=len(t_obs) - 1, tol=1e-8)
    zones = jnp.arange(1, len(n_true))                    # zone 0 has no cells

    def stages(n_zone):
        _, ys = solve_implicit(m, {"n_zone": n_zone}, case.U0, si)
        return ys[:, 0, gauge] + m.wstill[gauge]

    def loss(p):
        n_zone = jnp.asarray(n_true).at[zones].set(jnp.exp(p))
        L = jnp.mean((stages(n_zone) - obs) ** 2)
        return jnp.where(jnp.isfinite(L), L, 1.0e6)

    p = jnp.log(jnp.full(len(zones), 0.03))
    vg = jax.jit(jax.value_and_grad(loss))
    opt = optax.lbfgs(linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=8))
    state = opt.init(p)
    step = jax.jit(lambda p, st, v, g: opt.update(g, st, p, value=v, grad=g, value_fn=loss))
    hist, t0 = [], time.perf_counter()
    for k in range(n_iter):
        v, g = vg(p)
        hist.append(float(v))
        n_k = np.exp(np.asarray(p))
        print(f"iter {k:3d}: RMS stage misfit {np.sqrt(float(v)) * 1000:7.2f} mm, n = {np.round(n_k, 4)}, "
              f"{time.perf_counter() - t0:.0f} s", flush=True)
        (out / f"progress_noise{noise_mm:g}mm.json").write_text(json.dumps(
            {"iteration": k, "loss_history": hist, "n_zones": n_k.tolist()}))
        upd, state = step(p, state, v, g)
        p = optax.apply_updates(p, upd)
        if float(optax.tree.norm(g)) < 1e-12:
            break
    n_inv = np.exp(np.asarray(p))
    summary = {"noise_mm": noise_mm, "iterations": len(hist), "wall_s": time.perf_counter() - t0,
               "n_true": n_true[1:].tolist(), "n_inverted": n_inv.tolist(),
               "n_rel_error": (np.abs(n_inv - n_true[1:]) / n_true[1:]).tolist(),
               "final_rms_stage_misfit_mm": float(np.sqrt(hist[-1]) * 1000), "loss_history": hist,
               "gauge_cells": np.asarray(gauge).tolist(), "gauge_layout": layout,
               "dt_implicit_s": dt_impl}
    (out / f"summary_noise{noise_mm:g}mm.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "loss_history"}), flush=True)


if __name__ == "__main__":
    main()
