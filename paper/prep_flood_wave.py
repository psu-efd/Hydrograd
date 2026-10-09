"""Data for Supplementary Figure S1: flood wave through a compound channel (wet/dry floodplains).

Re-runs the case of ``benchmarks/validation/flood_wave_demo.py`` (phase 2 report, section 4b)
with fine output, and stores the small arrays that ``figS1_flood_wave.py`` plots.

Case
----
Valley L x W = 2 km x 200 m, bed slope S0 = 1e-3; trapezoidal main channel 40 m wide and
2 m deep; floodplains rising 3 % towards the valley sides; Manning n = 0.035 s/m^(1/3)
everywhere; 400 x 40 = 16,000 quadrilateral cells of 5 m. Inflow hydrograph
Q_in(t) = 40 -> 400 -> 40 m^3/s (piecewise linear); exit stage = normal-flow stage of the
compound section for the current discharge. Initial condition: base flow after a 2 h
spin-up at Q = 40 m^3/s. Numerics: package defaults (hydrostatic reconstruction + HLLC,
second-order MUSCL, SSPRK3 with CFL-limited steps).

Diagnostics
-----------
- Storage V(t) = sum_i h_i A_i (m^3), with h (m) the depth and A (m^2) the cell area.
- Outflow from the mass balance, Q_out = Q_in - dV/dt (m^3/s); exact because the scheme
  conserves mass and the only open boundaries are the inlet and the exit. Frames every
  60 s, central differences.
- Inundated area: sum of A_i over cells with h_i > 1 cm (m^2).

Usage (A100 recommended; see ``hpc/roar/paper_prep.sbatch``)
-----
    python paper/prep_flood_wave.py [WORK_DIR]

Outputs: ``paper/data/flood_wave/flood_wave.npz`` and ``run_info.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks" / "validation"))

import flood_wave_demo as demo
import hydrograd  # noqa: F401
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.io.synthetic import write_channel_case
from hydrograd.solvers.ode import solve_swe

# Mesh cells along (x) and across (y) the valley.
NX = 400
NY = 40
# Simulated time of the flood wave (s) and of the base-flow spin-up (s).
T_END = 6.0 * 3600.0
T_SPIN = 2.0 * 3600.0
# Output interval for the time series (s) and for the stored depth maps (s).
DT_SERIES = 60.0
DT_MAPS = 900.0
# Depth threshold for "inundated" (m).
H_WET = 0.01

OUT_DIR = REPO / "paper" / "data" / "flood_wave"


def padded_cell_nodes(cell_nodes: list[np.ndarray]) -> np.ndarray:
    """Cell-to-node table as a rectangular array, padded with -1.

    Args:
        cell_nodes: node indices of each cell (list of n_cells integer arrays).

    Returns:
        Integer array, shape [n_cells, max_nodes_per_cell], -1 where a cell has fewer nodes.
    """
    width = max(len(c) for c in cell_nodes)
    table = np.full((len(cell_nodes), width), -1, dtype=np.int32)
    for i, nodes in enumerate(cell_nodes):
        table[i, : len(nodes)] = nodes
    return table


def main() -> None:
    """Spin up the base flow, run the flood wave, write the figure data."""
    # 1. Work folder (case files) and boundary stages (m).
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "temp" / "paper_flood_wave"
    work.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stage = np.array([demo.normal_stage(q, NY) for q in demo.Q_HYD])
    bed_centre = demo.bed(np.array([demo.L]), np.array([demo.W / 2]))[0]
    h_base = demo.normal_stage(demo.Q_HYD[0], NY) - bed_centre
    device = jax.devices()[0].device_kind
    print(f"device: {device}", flush=True)

    # 2. Base flow: transient spin-up at constant Q from a normal-depth guess.
    common = dict(length=demo.L, width=demo.W, n_manning=demo.N_MAN, zb_fn=demo.bed,
                  inlet="INLET-Q", outlet="EXIT-H", wstill=0.0, dt=0.1)

    def base_guess_wse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Initial WSE (m): base-flow normal depth above the sloping valley bed."""
        return demo.S0 * (demo.L - x) + h_base

    base_dir = work / "base_flow"
    write_channel_case(base_dir, NX, NY, Q=demo.Q_HYD[0], wse_out=stage[0], solver="SSPRK3()",
                       t_end=T_SPIN, ic_wse=base_guess_wse, **common)
    base = setup_case(read_control(base_dir / "run_control.json"))
    t0 = time.perf_counter()
    base_sol = solve_swe(base.model, {}, base.U0, base.ode, ts=jnp.array([0.0, T_SPIN]))
    U_base = np.asarray(base_sol.ys[-1])
    print(f"spin-up (2 h): {time.perf_counter() - t0:.1f} s", flush=True)

    # 3. Flood-wave case starting from the base flow.
    wave_dir = work / "flood_wave"
    wave_dir.mkdir(parents=True, exist_ok=True)
    wstill = np.asarray(base.wstill)
    ic = {"wse": (U_base[0] + wstill).tolist(), "wstill": wstill.tolist(),
          "q_x": U_base[1].tolist(), "q_y": U_base[2].tolist()}
    (wave_dir / "ic_base_flow.json").write_text(json.dumps(ic))
    write_channel_case(wave_dir, NX, NY, Q_series=(demo.T_HYD, demo.Q_HYD),
                       WSE_series=(demo.T_HYD, stage), solver="SSPRK3()", t_end=T_END,
                       n_save=int(T_END / DT_MAPS), ic_file="ic_base_flow.json", **common)
    case = setup_case(read_control(wave_dir / "run_control.json"))
    model = case.model
    ts = jnp.arange(0.0, T_END + 1.0, DT_SERIES)

    def run() -> tuple[jax.Array, jax.Array]:
        """Flood-wave solution and the number of time steps.

        Returns:
            (ys, num_steps): ys has shape [n_frames, 3, n_cells] (xi in m, q_x and q_y in
            m^2/s); num_steps is the number of accepted SSPRK3 steps (scalar).
        """
        sol = solve_swe(model, {}, case.U0, case.ode, ts=ts)
        return sol.ys, sol.stats["num_steps"]

    # 4. Run twice: the first call includes compilation, the second gives the run time.
    run_jit = jax.jit(run)
    t0 = time.perf_counter()
    jax.block_until_ready(run_jit())
    wall_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    ys, num_steps = jax.block_until_ready(run_jit())
    wall = time.perf_counter() - t0
    n_steps = int(num_steps)
    print(f"flood wave: {model.n_cells} cells, 6 h in {wall:.1f} s "
          f"(first call incl. compile {wall_first:.1f} s)", flush=True)

    # 5. Depth (m), storage (m^3), inundated area (m^2) and the mass-balance outflow (m^3/s).
    ys = np.asarray(ys)
    t_s = np.asarray(ts)
    hstill = np.asarray(model.hstill)
    area = np.asarray(model.area)
    depth = ys[:, 0, :] + hstill[None, :]
    storage = depth @ area
    wet_area = (depth > H_WET).astype(float) @ area
    q_in = np.interp(t_s, demo.T_HYD, demo.Q_HYD)
    q_out = q_in - np.gradient(storage, t_s)

    # 6. Cross-check: exit discharge from the unit discharge q_x of the last cell column
    #    (cell-centre value, half a cell upstream of the exit face).
    centroid = np.asarray(case.mesh.centroid)
    last_column = centroid[:, 0] > demo.L - demo.L / NX
    dy = demo.W / NY
    q_out_column = ys[:, 1, last_column].sum(axis=1) * dy

    # 7. Store the mesh, bed, depth maps every 15 min and the time series.
    every = int(round(DT_MAPS / DT_SERIES))
    np.savez_compressed(
        OUT_DIR / "flood_wave.npz",
        nodes_xy=np.asarray(case.mesh.nodes[:, :2]),
        cell_nodes=padded_cell_nodes(case.mesh.cell_nodes),
        zb=np.asarray(model.zb).astype(np.float32),
        t_maps_s=t_s[::every],
        depth_maps_m=depth[::every].astype(np.float32),
        t_s=t_s, q_in_m3s=q_in, q_out_m3s=q_out, q_out_column_m3s=q_out_column,
        storage_m3=storage, wet_area_m2=wet_area, min_depth_m=depth.min(axis=1),
    )
    info = {"device": device, "cells": int(model.n_cells), "order": int(model.order),
            "wall_s": wall, "wall_first_call_incl_compile_s": wall_first,
            "simulated_h": T_END / 3600.0, "dt_series_s": DT_SERIES,
            "valley_area_m2": demo.L * demo.W, "bankfull_m3s_approx": 120.0,
            "solver": "SSPRK3() CFL-limited", "n_steps": n_steps}
    (OUT_DIR / "run_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
