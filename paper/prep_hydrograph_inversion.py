"""Data for Figure 7: inflow-hydrograph inversion from stage gauges (phase 2 report, section 4h).

The inversions themselves (``benchmarks/validation/hydrograph_inversion.py``, 25 L-BFGS
iterations on an A100, about 10 min each) stored only the recovered hydrograph and the loss
history. This script rebuilds the same case and recomputes, on the CPU, the quantities the
figure needs:

- the mesh, bed elevation z_b (m) and the initial depth h (m) of the compound valley;
- the observations: water-surface elevation (WSE, m) at the 3 centreline gauges every 5 min
  from the explicit second-order SSPRK3 truth run, plus the 5 mm Gaussian noise of the
  original run (same generator and seed);
- the gauge WSE (m) of the implicit backward-Euler inversion model (Delta t = 60 s) for the
  initial hydrograph (base flow 40 m^3/s) and for the two recovered hydrographs.

Case: valley 1 km x 200 m (50 x 10 cells of 20 m), bed slope 1e-3, 40 m main channel 2 m
deep, floodplains rising 3 %, Manning n = 0.035 s/m^(1/3). True inflow: gamma-shaped flood
Q(t) = 40 + 310 (t/t_p)^3 exp(3 (1 - t/t_p)) m^3/s with t_p = 45 min, over 2 h.

Usage: ``JAX_PLATFORMS=cpu python paper/prep_hydrograph_inversion.py [WORK_DIR]``
(a few minutes on 8 CPU cores).

Outputs in ``paper/data/hydrograph_inversion/``: ``hydrograph_inversion.npz`` and copies of
the two inversion summaries (``summary_noise{0,5}mm.json``).
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks" / "validation"))

import hydrograd  # noqa: F401
import hydrograph_inversion as hinv
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.io.synthetic import write_channel_case
from hydrograd.solvers.implicit import ImplicitSettings, solve_implicit
from hydrograd.solvers.ode import solve_swe

# Base-flow normal depth in the main channel for Q = 40 m^3/s (m), as in the original script.
H_NORMAL = 1.03
# Observation interval (s), truth hydrograph sampling (s) and inversion knot spacing (s).
DT_OBS = 300.0
DT_TRUTH_SERIES = 300.0
DT_KNOTS = 900.0
# Gauge noise levels of the two inversion runs (mm).
NOISE_LEVELS_MM = (0.0, 5.0)

SOURCE_DIR = REPO / "temp" / "hydrograph_inversion"
OUT_DIR = REPO / "paper" / "data" / "hydrograph_inversion"


def initial_wse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Initial WSE (m): base-flow normal depth above the sloping valley bed.

    Args:
        x: streamwise coordinates (m), any shape.
        y: cross-valley coordinates (m), same shape (unused).

    Returns:
        WSE (m), same shape as x.
    """
    return hinv.S0 * (hinv.L - x) + H_NORMAL


def padded_cell_nodes(cell_nodes: list[np.ndarray]) -> np.ndarray:
    """Cell-to-node table as a rectangular array, padded with -1.

    Args:
        cell_nodes: node indices of each cell (list of n_cells integer arrays).

    Returns:
        Integer array, shape [n_cells, max_nodes_per_cell].
    """
    width = max(len(c) for c in cell_nodes)
    table = np.full((len(cell_nodes), width), -1, dtype=np.int32)
    for i, nodes in enumerate(cell_nodes):
        table[i, : len(nodes)] = nodes
    return table


def main() -> None:
    """Rebuild the cases, recompute observations and fitted stages, write the figure data."""
    # 1. Folders.
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "temp" / "paper_hydrograph_inv"
    work.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Truth case (hydrograph sampled every 5 min) and inversion case (15-min knots).
    t_fine = np.arange(0.0, hinv.T_END + 1.0, DT_TRUTH_SERIES)
    t_knots = np.arange(0.0, hinv.T_END + 1.0, DT_KNOTS)
    common = dict(length=hinv.L, width=hinv.W, n_manning=hinv.N_MAN, zb_fn=hinv.bed, wstill=0.0,
                  dt=0.1, t_end=hinv.T_END, solver="SSPRK3()", inlet="INLET-Q",
                  outlet="EXIT-H", wse_out=H_NORMAL, ic_wse=initial_wse)
    write_channel_case(work / "truth", hinv.NX, hinv.NY,
                       Q_series=(t_fine, hinv.q_true(t_fine)), **common)
    write_channel_case(work / "inv", hinv.NX, hinv.NY,
                       Q_series=(t_knots, np.full(len(t_knots), 40.0)), **common)
    truth = setup_case(read_control(work / "truth" / "run_control.json"))
    inv = setup_case(read_control(work / "inv" / "run_control.json"))

    # 3. Gauge cells: nearest centroid to each centreline point.
    centroid = np.asarray(inv.mesh.centroid)
    gauges = []
    for gauge_x in hinv.GAUGES_X:
        distance2 = (centroid[:, 0] - gauge_x) ** 2 + (centroid[:, 1] - hinv.W / 2) ** 2
        gauges.append(int(np.argmin(distance2)))
    gauge = jnp.asarray(gauges)
    t_obs = jnp.arange(0.0, hinv.T_END + 1.0, DT_OBS)

    # 4. Observations: explicit SSPRK3 truth run; noise as in the original runs (seed 0).
    t0 = time.perf_counter()
    sol = solve_swe(truth.model, {}, truth.U0, truth.ode, ts=t_obs)
    obs_clean = np.asarray(sol.ys[:, 0, gauge] + truth.model.wstill[gauge])
    print(f"truth run: {time.perf_counter() - t0:.1f} s", flush=True)
    observations = {}
    for noise_mm in NOISE_LEVELS_MM:
        rng = np.random.default_rng(0)
        observations[noise_mm] = obs_clean + noise_mm * 1e-3 * rng.normal(size=obs_clean.shape)

    # 5. Implicit inversion model (backward Euler, fixed Delta t = 60 s).
    settings = ImplicitSettings(0.0, hinv.T_END, hinv.DT_IMPLICIT, theta=1.0,
                                n_save=len(t_obs) - 1, tol=1e-6)

    def stages_implicit(q_knots: jax.Array) -> jax.Array:
        """Gauge WSE (m), shape [n_obs, n_gauges], for the knot discharges (m^3/s), [n_knots]."""
        _, ys = solve_implicit(inv.model, {"Q": q_knots[None, :]}, inv.U0, settings)
        return ys[:, 0, gauge] + inv.model.wstill[gauge]

    stages_jit = jax.jit(stages_implicit)
    w_initial = np.asarray(stages_jit(jnp.full(len(t_knots), 40.0)))

    # 6. Fitted stages of the two recovered hydrographs; recomputed RMS misfits (mm).
    fitted = {}
    misfit_mm = {}
    for noise_mm in NOISE_LEVELS_MM:
        name = f"summary_noise{noise_mm:g}mm.json"
        source = SOURCE_DIR / f"noise{noise_mm:g}mm" / name
        shutil.copy(source, OUT_DIR / name)
        summary = json.loads(source.read_text())
        t0 = time.perf_counter()
        fitted[noise_mm] = np.asarray(stages_jit(jnp.asarray(summary["q_inv"])))
        residual = fitted[noise_mm] - observations[noise_mm]
        misfit_mm[noise_mm] = float(np.sqrt(np.mean(residual ** 2)) * 1e3)
        print(f"noise {noise_mm:g} mm: fitted run {time.perf_counter() - t0:.1f} s, "
              f"RMS misfit {misfit_mm[noise_mm]:.2f} mm "
              f"(logged before the last update: {summary['final_rms_stage_misfit_mm']:.2f})",
              flush=True)

    # 7. Store everything the figure needs.
    model = inv.model
    np.savez_compressed(
        OUT_DIR / "hydrograph_inversion.npz",
        nodes_xy=np.asarray(inv.mesh.nodes[:, :2]),
        cell_nodes=padded_cell_nodes(inv.mesh.cell_nodes),
        zb=np.asarray(model.zb),
        h_initial=np.asarray(inv.U0[0] + model.hstill),
        gauge_cells=np.asarray(gauges),
        gauge_xy=centroid[gauges],
        t_obs_s=np.asarray(t_obs),
        t_knots_s=t_knots,
        obs_clean_m=obs_clean,
        obs_noise0mm_m=observations[0.0],
        obs_noise5mm_m=observations[5.0],
        wse_initial_m=w_initial,
        wse_fit_noise0mm_m=fitted[0.0],
        wse_fit_noise5mm_m=fitted[5.0],
        rms_misfit_recomputed_mm=np.array([misfit_mm[0.0], misfit_mm[5.0]]),
    )
    print(f"wrote {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
