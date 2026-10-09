"""Figure 4: steady Savannah River flow, Hydrograd against SRH-2D.

Case
----
The Hydrograd.jl example ``Savannah_River``: a 1 km reach of the Savannah River on an
SRH-2D mesh of 1306 cells, five Manning zones (n = 0.040, 0.050, 0.030, 0.045, 0.050
s m^(-1/3)), inlet discharge Q = 187.4 m^3/s split by conveyance, and exit water-surface
elevation 29.3 m. The example's initial-condition file
``forward_simulation_initial_condition.json`` holds the SRH-2D steady solution (water-surface
elevation WSE (m) and unit discharges q_x, q_y (m^2/s) per cell), which is the reference
(see ``benchmarks/validation/savannah_vs_srh2d.py``). SRH-2D used its parabolic eddy
viscosity nu_t = C_t u_* h with C_t = 0.7.

Hydrograd solve
---------------
The steady state R(U*) = 0 of the semi-discrete SWEs (default scheme: hydrostatic
reconstruction + HLLC, second-order MUSCL with the Venkatakrishnan limiter) is found by
pseudo-transient Newton (``SteadyNewton()``) on the exact sparse Jacobian, starting from the
SRH-2D state. Two physics settings are solved:

- ``default``: no horizontal eddy viscosity (the package default); numbers only;
- ``parabolic``: SRH-2D's parabolic eddy viscosity with C_t = 0.7, the same physics as the
  reference; shown in the figure;
- ``default_order1``: as ``default`` with the first-order scheme (numbers only).

Compared quantities on wet cells (h > 0.01 m in both models):

    WSE = h + z_b (m),     |u| = sqrt(q_x^2 + q_y^2) / max(h, 1e-3)   (m/s),

with the difference Hydrograd - SRH-2D, its RMS and maximum.

Panels: rows WSE (m) and |u| (m/s); columns SRH-2D | Hydrograd | difference, all drawn on
the true polygon cells. Numbers go to ``paper/results/savannah_srh2d.json``; the fields to
``paper/data/savannah_srh2d.npz``.

Usage
-----
    python paper/fig04_savannah_srh2d.py              # compute if needed, then plot
    python paper/fig04_savannah_srh2d.py --compute    # recompute (timings) and plot
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import jax
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from figstyle import (
    DIFFERENCE_CMAP,
    FULL_WIDTH,
    MAGNITUDE_CMAP,
    apply_style,
    colorbar,
    panel_label,
    save,
)
from savannah_common import (
    DATA_DIR,
    FORWARD_CASE,
    copy_case,
    machine_info,
    mesh_polygons,
    poly_field,
    read_json_with_commas,
    write_control,
    write_results,
)

import hydrograd  # noqa: F401  (enables float64)
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.solvers.final import newton_settings
from hydrograd.solvers.steady import newton_solve

# Output data file of the fields (cell values) and the number of timing repetitions.
DATA_FILE = DATA_DIR / "savannah_srh2d.npz"
N_TIMING_REPEATS = 5

# Wet-cell threshold (m) and the depth floor of the velocity (m).
WET_DEPTH = 0.01
VELOCITY_DEPTH_FLOOR = 1.0e-3

# Physics setting drawn in the figure: SRH-2D's own eddy-viscosity model (like for like).
FIGURE_SETTING = "parabolic"

# Physics settings: name -> optional "turbulence" block of the run control and the order.
# Each setting: (turbulence block or None, spatial order).
SETTINGS = {
    "default": (None, 2),
    "parabolic": ({"model": "parabolic", "coefficient": 0.7}, 2),
    "default_order1": (None, 1),
}


def _solve(case_dir: Path, control: dict[str, Any], turbulence: dict | None,
           order: int) -> dict[str, Any]:
    """Steady Newton solve of one physics setting, with timings.

    Args:
        case_dir: copied Savannah case folder.
        control: the example's run control (forward mode).
        turbulence: ``"turbulence"`` block, or None for no eddy viscosity.
        order: spatial order of the scheme (1 or 2).

    Returns:
        Dictionary with the steady depth h (m), WSE (m), q_x, q_y (m^2/s) per cell
        [n_cells], the iteration count, the final max|R|, the compile-plus-solve time and
        the warm solve times (s), and the case (mesh, bed).
    """
    # 1. Run control: steady Newton, spatial order, optional eddy viscosity.
    control = json.loads(json.dumps(control))
    control["forward_simulation_options"]["forward_simulation_ode_solver"] = "SteadyNewton()"
    control["numerics"] = {"spatial_order": order}
    control.pop("turbulence", None)
    if turbulence is not None:
        control["turbulence"] = turbulence
    write_control(case_dir, control)
    case = setup_case(read_control(case_dir / "run_control.json"))
    model = case.model
    settings = newton_settings(case.ode)

    # 2. Jitted Newton loop; the first call includes compilation.
    def solve(U0: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Newton solve from U0 [3, n_cells]; returns (U*, iterations, max|R|)."""
        return newton_solve(model, {}, U0, settings)

    solve_jit = jax.jit(solve)
    start = time.perf_counter()
    U, iterations, residual = jax.block_until_ready(solve_jit(case.U0))
    first_call = time.perf_counter() - start

    # 3. Warm timings (compiled), repeated.
    warm = []
    for _ in range(N_TIMING_REPEATS):
        start = time.perf_counter()
        jax.block_until_ready(solve_jit(case.U0))
        warm.append(time.perf_counter() - start)

    # 4. Fields: depth, WSE and unit discharges.
    depth = np.asarray(U[0] + model.hstill)
    wse = np.asarray(U[0] + model.wstill)
    return {
        "case": case,
        "h": depth,
        "wse": wse,
        "q_x": np.asarray(U[1]),
        "q_y": np.asarray(U[2]),
        "iterations": int(iterations),
        "residual": float(residual),
        "first_call_s": first_call,
        "warm_s": warm,
    }


def _speed(q_x: np.ndarray, q_y: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Depth-averaged speed |q| / max(h, floor) (m/s), shape [n_cells].

    Args:
        q_x, q_y: unit discharges (m^2/s), shape [n_cells].
        depth: water depth h (m), shape [n_cells].

    Returns:
        Speed (m/s), shape [n_cells].
    """
    return np.hypot(q_x, q_y) / np.maximum(depth, VELOCITY_DEPTH_FLOOR)


def compute() -> None:
    """Solve both physics settings, compare with SRH-2D and write the data and results."""
    # 1. Copy the example and read the SRH-2D reference fields.
    case_dir = copy_case(FORWARD_CASE, "Savannah_River")
    control = read_json_with_commas(case_dir / "run_control.json")
    reference = read_json_with_commas(case_dir / "forward_simulation_initial_condition.json")
    wse_srh = np.asarray(reference["wse"])

    results: dict[str, Any] = {
        "case": "Savannah River, Hydrograd.jl example forward_simulation/Savannah_River",
        "reference": ("SRH-2D steady solution stored in forward_simulation_initial_condition.json "
                      "(SRH-2D parabolic eddy viscosity C_t = 0.7)"),
        "scheme": "hr_hllc, MUSCL (Venkatakrishnan) unless spatial_order 1, SteadyNewton()",
        "initial_guess": "SRH-2D steady state",
        "wet_depth_threshold_m": WET_DEPTH,
        "machine": machine_info(),
        "figure_setting": FIGURE_SETTING,
        "settings": {},
    }
    fields: dict[str, np.ndarray] = {}

    for name, (turbulence, order) in SETTINGS.items():
        solution = _solve(case_dir, control, turbulence, order)
        zb = np.asarray(solution["case"].model.zb)

        # 2. SRH-2D depth and speed from the reference WSE and unit discharges.
        depth_srh = wse_srh - zb
        speed_srh = _speed(np.asarray(reference["q_x"]), np.asarray(reference["q_y"]), depth_srh)
        speed = _speed(solution["q_x"], solution["q_y"], solution["h"])

        # 3. Differences on wet cells.
        wet = (solution["h"] > WET_DEPTH) & (depth_srh > WET_DEPTH)
        wse_error = solution["wse"][wet] - wse_srh[wet]
        speed_error = speed[wet] - speed_srh[wet]
        results["settings"][name] = {
            "turbulence": turbulence,
            "spatial_order": order,
            "newton_iterations": solution["iterations"],
            "final_max_abs_residual": solution["residual"],
            "solve_time_first_call_incl_compile_s": solution["first_call_s"],
            "solve_time_warm_median_s": float(np.median(solution["warm_s"])),
            "solve_time_warm_all_s": solution["warm_s"],
            "n_wet_cells": int(wet.sum()),
            "wse_rms_difference_m": float(np.sqrt(np.mean(wse_error ** 2))),
            "wse_max_abs_difference_m": float(np.max(np.abs(wse_error))),
            "wse_mean_difference_m": float(np.mean(wse_error)),
            "speed_rms_difference_m_s": float(np.sqrt(np.mean(speed_error ** 2))),
            "speed_max_abs_difference_m_s": float(np.max(np.abs(speed_error))),
            "speed_mean_difference_m_s": float(np.mean(speed_error)),
            "srh2d_speed_range_m_s": [float(speed_srh[wet].min()), float(speed_srh[wet].max())],
            "srh2d_wse_range_m": [float(wse_srh[wet].min()), float(wse_srh[wet].max())],
        }

        # 4. Keep the fields for the plot.
        fields[f"wse_{name}"] = solution["wse"]
        fields[f"speed_{name}"] = speed
        fields[f"wet_{name}"] = wet
        fields["wse_srh"] = wse_srh
        fields["speed_srh"] = speed_srh

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(DATA_FILE, **fields)
    write_results("savannah_srh2d", results)
    print(json.dumps(results["settings"], indent=1))


def _blank_colorbar_space(ax: plt.Axes) -> None:
    """Reserve the colour-bar space of a map that has no own colour bar (equal panel widths).

    Args:
        ax: the map axes.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.08)
    cax.set_axis_off()


def plot() -> Path:
    """Draw the 2 x 3 comparison (SRH-2D | Hydrograd | difference) and save the figure.

    Returns:
        Path of the PDF file.
    """
    apply_style()
    data = np.load(DATA_FILE)

    # Mesh polygons from a fresh copy of the example (geometry only).
    case_dir = copy_case(FORWARD_CASE, "Savannah_River")
    mesh = setup_case(read_control(case_dir / "run_control.json")).mesh
    polygons = mesh_polygons(mesh)

    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, 2.7), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.30, "hspace": 0.10})

    # Rows: (field name, symbol and unit, difference limit).
    rows = [
        ("wse", r"WSE", "(m)", 0.025),
        ("speed", r"$|\mathbf{u}|$", r"(m\,s$^{-1}$)", 0.2),
    ]
    labels = iter(["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"])

    for row, (key, symbol, unit, diff_limit) in enumerate(rows):
        srh = data[f"{key}_srh"]
        model = data[f"{key}_{FIGURE_SETTING}"]
        difference = model - srh

        # Shared colour limits of the two solutions.
        low = float(min(srh.min(), model.min()))
        high = float(max(srh.max(), model.max()))
        norm = Normalize(low, high)

        # SRH-2D and Hydrograd maps.
        poly_field(axes[row, 0], polygons, srh, cmap=MAGNITUDE_CMAP, norm=norm)
        _blank_colorbar_space(axes[row, 0])
        mappable = poly_field(axes[row, 1], polygons, model, cmap=MAGNITUDE_CMAP, norm=norm)
        colorbar(axes[row, 1], mappable, unit)

        # Difference map, symmetric limits.
        diff_norm = Normalize(-diff_limit, diff_limit)
        diff_map = poly_field(axes[row, 2], polygons, difference, cmap=DIFFERENCE_CMAP,
                              norm=diff_norm)
        colorbar(axes[row, 2], diff_map, unit)

        # Row label on the left, panel labels and column titles.
        axes[row, 0].set_ylabel(r"$y$ (m)")
        axes[row, 0].text(0.03, 0.06, symbol, transform=axes[row, 0].transAxes, fontsize=10)
        for column in range(3):
            panel_label(axes[row, column], next(labels), x=-0.02, y=1.02)

    axes[0, 0].set_title("SRH-2D", fontsize=10, pad=14)
    axes[0, 1].set_title("Hydrograd", fontsize=10, pad=14)
    axes[0, 2].set_title("Hydrograd $-$ SRH-2D", fontsize=10, pad=14)
    for column in range(3):
        axes[1, column].set_xlabel(r"$x$ (m)")

    return save(fig, "fig04_savannah_srh2d")


def main() -> None:
    """Compute (if asked or if no data exist) and draw Figure 4."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--compute", action="store_true", help="recompute the steady solves")
    parser.add_argument("--no-plot", action="store_true", help="compute only")
    args = parser.parse_args()

    if args.compute or not DATA_FILE.exists():
        compute()
    if not args.no_plot:
        print(f"wrote {plot()}")


if __name__ == "__main__":
    main()
