"""Data for Figure 8: Savannah River unsteady roughness calibration (phase 2 report, section 4k).

Collects, from the saved outputs of the calibration study, the small arrays and tables that
``fig08_savannah_unsteady.py`` plots. Nothing is re-simulated; only the case is set up
(mesh, zones, initial depth) to pick the gauge cells.

Sources
-------
- Calibration runs of ``benchmarks/validation/savannah_unsteady_inversion.py`` (L-BFGS on
  log n, implicit backward-Euler model with Delta t = 30 s, 30 iterations), whose Slurm logs
  hold one line per iteration, ``iter k: RMS stage misfit x mm, n = [n1 .. n5], t s``:
    run 1  : 6 "bins" gauges, start n = 0.03 (A100)     hpc/roar/hg-savinv.56108444.out
    run 2a : 10 gauges (2 per zone), start 0.03 (CPU)   temp/savinv_zones_n0/
    run 2b : 10 gauges, start 0.045 (CPU)               temp/savinv_zones_start0.045/
    run 2c : 10 gauges, start 0.06 (CPU)                temp/savinv_zones_start0.06/
- Identifiability at the true n (``temp/modelfloor/ident.py``): singular values (m) of the
  finite-difference sensitivity J = d WSE / d log n over all (time, gauge) observations.
- Model-error floor (``temp/modelfloor/floor.py``): explicit SSPRK3 truth and implicit
  (Delta t = 30 s) gauge stages at the true n for the 6-gauge layout (m).

Usage: ``JAX_PLATFORMS=cpu python paper/prep_savannah_unsteady.py`` (seconds).

Outputs in ``paper/data/savannah_unsteady/``: ``savannah_unsteady.npz``, ``runs.json``,
``identifiability.json``.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks" / "validation"))

import savannah_unsteady_inversion as sav

import hydrograd  # noqa: F401
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control

OUT_DIR = REPO / "paper" / "data" / "savannah_unsteady"
CASE_CONTROL = REPO / "paper" / "cases" / "savannah_river_unsteady" / "run_control.json"
MODEL_FLOOR_DIR = REPO / "temp" / "modelfloor"

# Calibration runs: label -> (Slurm log, summary JSON, hardware).
RUNS = {
    "1": (REPO / "hpc/roar/hg-savinv.56108444.out",
          REPO / "temp/savannah_inversion/summary_noise0mm.json", "NVIDIA A100-PCIE-40GB"),
    "2a": (REPO / "temp/savinv_zones_n0/hg-savinv.56114390.out",
           REPO / "temp/savinv_zones_n0/summary_noise0mm.json", "CPU, 16 cores"),
    "2b": (REPO / "temp/savinv_zones_start0.045/hg-savinv.56116904.out",
           REPO / "temp/savinv_zones_start0.045/summary_noise0mm.json", "CPU, 16 cores"),
    "2c": (REPO / "temp/savinv_zones_start0.06/hg-savinv.56116905.out",
           REPO / "temp/savinv_zones_start0.06/summary_noise0mm.json", "CPU, 16 cores"),
}

ITER_LINE = re.compile(
    r"iter\s+(\d+): RMS stage misfit\s+([\d.]+) mm, n = \[([^\]]+)\], (\d+) s")


def read_log(path: Path) -> dict[str, list]:
    """Per-iteration misfit, Manning's n and elapsed time from a calibration log.

    Args:
        path: Slurm log of one calibration run.

    Returns:
        Dict with "misfit_mm" [n_iter] (RMS stage misfit, mm), "n" [n_iter][5]
        (s/m^(1/3), printed to 4 decimals) and "elapsed_s" [n_iter] (s).
    """
    misfit = []
    n_values = []
    elapsed = []
    for match in ITER_LINE.finditer(path.read_text()):
        misfit.append(float(match.group(2)))
        n_values.append([float(value) for value in match.group(3).split()])
        elapsed.append(float(match.group(4)))
    return {"misfit_mm": misfit, "n": n_values, "elapsed_s": elapsed}


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
    """Collect the mesh, gauges, run histories and identifiability results."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Mesh, zones, bed and initial depth (m) of the calibration case.
    case = setup_case(read_control(CASE_CONTROL))
    model = case.model
    mat_id = np.asarray(model.mat_id)
    h0 = np.asarray(case.U0[0] + model.hstill)
    x = np.asarray(case.mesh.centroid[:, 0])

    # 2. Gauge cells of the two layouts (same rule as the calibration script).
    gauges_bins = np.asarray(sav.pick_gauges(h0, x, mat_id, "bins"))
    gauges_zones = np.asarray(sav.pick_gauges(h0, x, mat_id, "zones"))

    # 3. Run histories and summaries; check that the gauges match the runs' own records.
    runs = {}
    for label, (log, summary_path, hardware) in RUNS.items():
        summary = json.loads(summary_path.read_text())
        history = read_log(log)
        layout = summary.get("gauge_layout", "bins")
        expected = gauges_bins if layout == "bins" else gauges_zones
        if list(expected) != list(summary["gauge_cells"]):
            raise RuntimeError(f"run {label}: gauge cells differ from the recorded ones")
        runs[label] = {
            "gauge_layout": layout,
            "n_gauges": len(summary["gauge_cells"]),
            "n_start": summary.get("n0", 0.03),
            "hardware": hardware,
            "wall_s": summary["wall_s"],
            "iterations": summary["iterations"],
            "n_true": summary["n_true"],
            "n_final": summary["n_inverted"],
            "n_rel_error": summary["n_rel_error"],
            "final_rms_misfit_mm": summary["final_rms_stage_misfit_mm"],
            "history": history,
        }
    (OUT_DIR / "runs.json").write_text(json.dumps(runs, indent=1))

    # 4. Identifiability results and the model-error floor (m).
    shutil.copy(MODEL_FLOOR_DIR / "identifiability.json", OUT_DIR / "identifiability.json")
    stage_explicit = np.load(MODEL_FLOOR_DIR / "obs.npy")
    stage_implicit = np.load(MODEL_FLOOR_DIR / "impl_dt30.npy")

    # 5. Arrays for the maps and the hydrograph.
    np.savez_compressed(
        OUT_DIR / "savannah_unsteady.npz",
        nodes_xy=np.asarray(case.mesh.nodes[:, :2]),
        cell_nodes=padded_cell_nodes(case.mesh.cell_nodes),
        mat_id=mat_id,
        zb=np.asarray(model.zb),
        h_initial=h0,
        gauges_bins=gauges_bins,
        gauges_zones=gauges_zones,
        centroid=np.asarray(case.mesh.centroid),
        t_hydrograph_s=sav.T_HYD,
        q_hydrograph_m3s=sav.Q_HYD,
        t_end_s=sav.T_END,
        dt_obs_s=sav.DT_OBS,
        stage_explicit_true_n_m=stage_explicit,
        stage_implicit_true_n_m=stage_implicit,
    )
    print(f"wrote {OUT_DIR}; cells {model.n_cells}, cells per zone {np.bincount(mat_id)}")


if __name__ == "__main__":
    main()
