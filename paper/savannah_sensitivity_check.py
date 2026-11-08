"""Check of the finite-difference sensitivity matrix of Fig. 8e against the adjoint.

The identifiability analysis of Sect. 4.5 uses the sensitivity matrix S of the gauge stages
with respect to log n of the five roughness zones, built by one-sided finite differences
(step 0.02 in log n) through backward Euler with Delta t = 30 s, because the implicit solver
provides reverse-mode derivatives only. This script rebuilds S the same way and checks it
against the adjoint: for a singular direction u_k of S (left singular vector), one
reverse-mode gradient of the loss u_k . W(log n) gives S^T u_k exactly, which must equal
s_k v_k. The check is done for the strongest (k = 1) and the weakest (k = 5) direction of
both gauge layouts (six gauges in bins, ten gauges with two per zone).

Inputs: ``paper/cases/savannah_river_unsteady`` (the inversion case at the true n).
Output: ``paper/results/savannah_sensitivity_check.json``. CPU, about 45 min.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

PAPER_DIR = Path(__file__).resolve().parent
REPO = PAPER_DIR.parent
sys.path.insert(0, str(REPO / "benchmarks" / "validation"))

import savannah_unsteady_inversion as savannah

from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.solvers.implicit import ImplicitSettings, solve_implicit

CASE_CONTROL = PAPER_DIR / "cases" / "savannah_river_unsteady" / "run_control.json"
RESULTS_FILE = PAPER_DIR / "results" / "savannah_sensitivity_check.json"
FD_STEP_LOG_N = 0.02      # one-sided step in log n, as in the identifiability analysis
DT_IMPLICIT = 30.0        # s, backward Euler step of the Savannah unsteady inversion
NEWTON_TOL = 1.0e-8
LAYOUTS = ("bins", "zones")
CHECKED_DIRECTIONS = (0, 4)   # strongest and weakest singular direction (0-based)


def main() -> None:
    """Build S by finite differences, check S^T u_k against the adjoint, write the results."""
    case = setup_case(read_control(CASE_CONTROL))
    model = case.model
    n_true = np.asarray(model.n_zone)          # zone 0 has no cells; zones 1..5 are inverted
    zones = jnp.arange(1, len(n_true))
    t_obs = np.arange(0.0, savannah.T_END + 1.0, savannah.DT_OBS)
    settings = ImplicitSettings(0.0, savannah.T_END, DT_IMPLICIT, theta=1.0,
                                n_save=len(t_obs) - 1, tol=NEWTON_TOL)

    # Gauge cells of both layouts, chosen as in the inversion script.
    h0 = np.asarray(case.U0[0] + model.hstill)
    x = np.asarray(case.mesh.centroid[:, 0])
    zone_of_cell = np.asarray(model.mat_id)
    gauges = {layout: np.asarray(savannah.pick_gauges(h0, x, zone_of_cell, layout))
              for layout in LAYOUTS}

    def wse_all_cells(log_n: jax.Array) -> jax.Array:
        """WSE (m) at every cell and observation time, shape [n_obs + 1, n_cells]."""
        n_zone = jnp.asarray(n_true).at[zones].set(jnp.exp(log_n))
        states = solve_implicit(model, {"n_zone": n_zone}, case.U0, settings)[1]
        return states[:, 0, :] + model.wstill

    wse_jit = jax.jit(wse_all_cells)
    log_n_true = jnp.log(jnp.asarray(n_true[1:]))

    # 1. Finite-difference sensitivity D[time, cell, zone] = d WSE / d log n (m), one-sided.
    t0 = time.perf_counter()
    base = np.asarray(wse_jit(log_n_true))
    print(f"base run {time.perf_counter() - t0:.0f} s (incl. compile)", flush=True)
    columns = []
    for k in range(len(n_true) - 1):
        perturbed = log_n_true.at[k].add(FD_STEP_LOG_N)
        columns.append((np.asarray(wse_jit(perturbed)) - base) / FD_STEP_LOG_N)
        print(f"zone {k + 1} finite difference done", flush=True)
    sensitivity = np.stack(columns, axis=-1)

    # 2. For each layout: SVD of S (rows = observations after t = 0), then the adjoint check.
    results: dict[str, Any] = {"fd_step_log_n": FD_STEP_LOG_N, "dt_s": DT_IMPLICIT,
                               "newton_tol": NEWTON_TOL, "layouts": {}}
    for layout in LAYOUTS:
        cells = gauges[layout]
        S = sensitivity[1:, cells, :].reshape(-1, sensitivity.shape[-1])      # [N_obs, 5]
        U, s, Vt = np.linalg.svd(S, full_matrices=False)
        n_obs = S.shape[0]
        checks = []
        for k in CHECKED_DIRECTIONS:
            u = jnp.asarray(U[:, k])

            def projected_stage(log_n: jax.Array, u: jax.Array = u,
                                cells: np.ndarray = cells) -> jax.Array:
                """u . W(log n) over the gauge observations after t = 0 (m)."""
                wse = wse_all_cells(log_n)[1:, cells].reshape(-1)
                return jnp.dot(u, wse)

            t0 = time.perf_counter()
            adjoint = np.asarray(jax.grad(projected_stage)(log_n_true))      # S^T u_k, exact
            seconds = time.perf_counter() - t0
            fd = s[k] * Vt[k]                                                  # S_FD^T u_k
            rel = float(np.linalg.norm(adjoint - fd) / np.linalg.norm(fd))
            checks.append({"direction": k + 1, "singular_value_fd_m": float(s[k]),
                           "singular_value_adjoint_m": float(np.dot(adjoint, Vt[k])),
                           "S_T_u_adjoint": adjoint.tolist(), "S_T_u_fd": fd.tolist(),
                           "relative_difference": rel, "adjoint_seconds": seconds})
            print(f"{layout}: direction {k + 1}: s_fd {s[k]:.4e} m, s_adj {np.dot(adjoint, Vt[k]):.4e} m, "
                  f"rel. diff {rel:.2e}, {seconds:.0f} s", flush=True)
        results["layouts"][layout] = {
            "n_gauges": len(cells), "n_obs": int(n_obs),
            "singular_values_m": s.tolist(), "condition_number": float(s[0] / s[-1]),
            "rms_stage_change_per_unit_log_n_mm": (1e3 * s / np.sqrt(n_obs)).tolist(),
            "checks": checks}

    RESULTS_FILE.write_text(json.dumps(results, indent=1))
    print("wrote", RESULTS_FILE)


if __name__ == "__main__":
    main()
