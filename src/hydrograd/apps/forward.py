"""Forward simulation (port of ``swe_2D_forward_simulation.jl`` + post-processing).

Outputs in the case directory, named as in Hydrograd.jl:

* ``forward_simulation_results_%04d.vtk`` for each saved frame (1-based);
* the truth JSON (``forward_simulation_save_solution_truth_file_name``) with the
  same keys as Julia, from the last frame;
* ``total_water_volume.csv`` (sum h A per frame, m^3);
* ``forward_simulation_solution.npz`` (all frames; replaces Julia's .jld2).

Difference from Julia: ``friction_x/y_truth`` use the *actual* (possibly
variable) n; Julia always uses the constant zone n there.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np

from hydrograd.apps.setup import Case
from hydrograd.io.output import write_csv, write_json, write_vtk
from hydrograd.params.manning import manning_law
from hydrograd.physics.model import SWEModel
from hydrograd.physics.rhs import (
    _safe_speed,
    clipped_state,
    friction,
    manning_n_cells,
    still_depth,
)
from hydrograd.solvers.final import STEADY_TOL, is_steady, newton_settings
from hydrograd.solvers.ode import solve_swe
from hydrograd.solvers.steady import newton_solve

logger = logging.getLogger(__name__)

__all__ = ["derived_fields", "run_forward"]


def derived_fields(m: SWEModel, theta: dict[str, Any], U: np.ndarray, wstill: np.ndarray) -> dict[str, np.ndarray]:
    """Per-cell diagnostic fields of one state (Julia post-processing conventions).

    Velocities are ``q/(h + h_small)``; ``h = xi + h_still`` *without* clipping.
    """
    U = jnp.asarray(U)
    hs = still_depth(m, theta)
    xi, hc, qxc, qyc = clipped_state(m, U, hs)
    h = np.asarray(U[0] + hs)
    n = manning_n_cells(m, theta, hc, qxc, qyc)
    fx, fy = friction(m, theta, hc, qxc, qyc, n)
    out = {"xi": np.asarray(U[0]), "wstill": wstill, "hstill": np.asarray(hs), "h": h,
           "wse": np.asarray(U[0]) + wstill, "u": np.asarray(U[1]) / (h + m.h_small),
           "v": np.asarray(U[2]) / (h + m.h_small), "q_x": np.asarray(U[1]), "q_y": np.asarray(U[2]),
           "zb": np.asarray(theta.get("zb", m.zb)), "ManningN": np.asarray(n),
           "friction_x": np.asarray(fx), "friction_y": np.asarray(fy)}
    if m.resistance.mode == "law":
        _, h_ks, f, re = manning_law(m.resistance.law, hc, _safe_speed(qxc, qyc, hc), m.ks)
        out.update(ks=np.asarray(m.ks), h_ks=np.asarray(h_ks), friction_factor=np.asarray(f), Re=np.asarray(re))
    return out


def _truth_dict(case: Case, fields: dict[str, np.ndarray]) -> dict[str, Any]:
    m = case.model
    n = m.n_cells
    z = np.zeros(n)
    return {"xi_truth": fields["xi"], "wstill_truth": fields["wstill"], "hstill_truth": fields["hstill"],
            "wse_truth": fields["wse"], "h_truth": fields["h"], "u_truth": fields["u"], "v_truth": fields["v"],
            "zb_cell_truth": fields["zb"], "ManningN_cells_truth": fields["ManningN"],
            "h_ks_cells_truth": fields.get("h_ks", z), "friction_factor_cells_truth": fields.get("friction_factor", z),
            "Re_cells_truth": fields.get("Re", z), "friction_x_truth": fields["friction_x"],
            "friction_y_truth": fields["friction_y"], "S0_cells_truth": np.asarray(m.S0),
            "ManningN_zone_values_truth": np.asarray(m.n_zone), "inlet_discharges_truth": np.asarray(m.Q)}


def write_frame_vtk(case: Case, path: str, fields: dict[str, np.ndarray]) -> None:
    """VTK with the main cell fields of one frame."""
    names = ["xi", "wstill", "hstill", "h", "q_x", "q_y", "ManningN", "zb", "wse",
             "friction_x", "friction_y", "ks", "h_ks", "friction_factor", "Re"]
    sc = {k if k != "wse" else "WSE": fields[k] for k in names if k in fields}
    write_vtk(case.ctrl.case_dir / path, case.mesh, sc,
              {"U": np.stack([fields["u"], fields["v"]], 1), "slope": np.asarray(case.model.S0)})


def run_forward(case: Case) -> dict[str, Any]:
    """Run the forward simulation and write all outputs.

    Returns:
        dict with ``ts`` (s), ``ys`` [n_frames, 3, n_cells] and the last-frame fields.
    """
    m, s = case.model, case.ode
    theta: dict[str, Any] = {}
    t0 = time.perf_counter()
    if is_steady(s):
        U, it, rn = jax.jit(lambda U0: newton_solve(m, theta, U0, newton_settings(s)))(case.U0)
        wall = time.perf_counter() - t0
        if not float(rn) < STEADY_TOL:
            raise RuntimeError(f"steady Newton solve did not converge (max|R| = {float(rn):.2e} "
                               f"after {int(it)} iterations); no results written")
        logger.info("steady state: %d Newton iterations, max|R| %.1e, %.2f s incl. compile",
                    int(it), float(rn), wall)
        ys = np.stack([np.asarray(case.U0), np.asarray(U)])
        ts = np.array([s.t0, s.t1])  # IC and steady state
    else:
        sol = jax.jit(lambda U0: solve_swe(m, theta, U0, s))(case.U0)
        ys, ts = np.asarray(sol.ys), np.asarray(sol.ts)
        wall = time.perf_counter() - t0
        if not bool(sol.result == dfx.RESULTS.successful):
            raise RuntimeError(f"ODE solve failed ({sol.result}); no results written. "
                               "Check dt, tspan and the max_steps limit.")
        logger.info("forward simulation: %d steps (%d rejected), %.2f s incl. compile",
                    int(sol.stats["num_steps"]), int(sol.stats["num_rejected_steps"]), wall)

    vol = []
    for k, U in enumerate(ys):
        fields = derived_fields(m, theta, U, case.wstill)
        write_frame_vtk(case, f"forward_simulation_results_{k + 1:04d}.vtk", fields)
        vol.append((float(ts[k]), float(np.sum(fields["h"] * np.asarray(m.area)))))
    d = case.ctrl.case_dir
    write_csv(d / "total_water_volume.csv", ["t", "total_water_volume"], vol)
    np.savez_compressed(d / "forward_simulation_solution.npz", t=ts, u=ys)
    write_json(d / case.ctrl.opt("forward_simulation_save_solution_truth_file_name"), _truth_dict(case, fields))
    return {"ts": ts, "ys": ys, "fields": fields, "wall_s": wall}
