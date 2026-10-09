"""Figure 7: inflow-hydrograph inversion from stage gauges with the implicit per-step adjoint.

Case (phase 2 report, section 4h): compound valley 1 km x 200 m (500 quadrilateral cells of
20 m), bed slope 1e-3, a 40 m wide, 2 m deep main channel and floodplains rising 3 %;
Manning n = 0.035 s/m^(1/3). The true inflow is a gamma-shaped flood
Q(t) = 40 + 310 (t/t_p)^3 exp(3 (1 - t/t_p)) m^3/s, t_p = 45 min, that inundates the
floodplains. Observations: water-surface elevation (WSE, m) at three centreline gauges every
5 min from the explicit second-order SSPRK3 model, without noise and with 5 mm Gaussian noise.

Inversion: unknown Q at 15-min knots (9 values, Q = exp(p), linear between knots), start at
base flow 40 m^3/s; loss = mean squared stage misfit (m^2). The inversion model is implicit
backward Euler with Delta t = 60 s, whose gradient comes from per-step implicit-function-
theorem adjoints; L-BFGS with a zoom line search, 25 iterations, on an NVIDIA A100.

Panels
------
(a) mesh coloured by bed elevation z_b (m), with the gauges, the inflow and the exit;
(b) true, initial and recovered hydrographs Q(t) (m^3/s) for 0 mm and 5 mm noise;
(c) stage at the gauges G1-G3 (markers: observations with 5 mm noise; lines: fitted stages of
    both runs and the stages of the initial guess);
(d) loss history (m^2) for both runs.

Data: ``paper/data/hydrograph_inversion/`` (made by ``prep_hydrograph_inversion.py``).
Numbers: ``paper/results/hydrograph_inversion.json``.
Usage: ``python paper/fig07_hydrograph_inversion.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from figstyle import (
    FULL_WIDTH,
    MAGNITUDE_CMAP,
    MODEL,
    MODEL_2,
    REFERENCE,
    apply_style,
    colorbar,
    panel_label,
    save,
)

PAPER = Path(__file__).resolve().parent
DATA = PAPER / "data" / "hydrograph_inversion"
RESULTS = PAPER / "results" / "hydrograph_inversion.json"

# True-flood parameters: base and peak discharge (m^3/s), time to peak (s).
Q_BASE = 40.0
Q_PEAK = 350.0
T_PEAK = 2700.0
# Hardware of the inversion runs (from their Slurm logs, jobs 56108432 and 56108433).
HARDWARE = "NVIDIA A100-PCIE-40GB (Roar)"
# Marker of each gauge in panel (c).
GAUGE_MARKERS = ("o", "s", "^")
# Initial-guess style: grey dotted.
INITIAL = {"color": "0.55", "linestyle": ":", "linewidth": 1.2}


def q_true(t: np.ndarray) -> np.ndarray:
    """True inflow (m^3/s): Q = 40 + 310 (t/t_p)^3 exp(3 (1 - t/t_p)).

    Args:
        t: time (s), any shape.

    Returns:
        Discharge (m^3/s), same shape.
    """
    ratio = t / T_PEAK
    return Q_BASE + (Q_PEAK - Q_BASE) * ratio ** 3 * np.exp(3.0 * (1.0 - ratio))


def cell_polygons(nodes_xy: np.ndarray, cell_nodes: np.ndarray) -> list[np.ndarray]:
    """Polygon vertices of every cell.

    Args:
        nodes_xy: node coordinates (m), shape [n_nodes, 2].
        cell_nodes: node indices per cell padded with -1, shape [n_cells, max_nodes].

    Returns:
        List of n_cells arrays of shape [k_i, 2] (m).
    """
    polygons = []
    for row in cell_nodes:
        polygons.append(nodes_xy[row[row >= 0]])
    return polygons


def plot_domain(ax: plt.Axes, data: np.lib.npyio.NpzFile) -> None:
    """Panel (a): mesh coloured by bed elevation, gauges, inflow and exit.

    Args:
        ax: target axes.
        data: arrays from ``hydrograph_inversion.npz``.
    """
    # Cells coloured by z_b (m) with their edges, so the mesh is visible.
    polygons = cell_polygons(data["nodes_xy"], data["cell_nodes"])
    cells = PolyCollection(polygons, cmap=MAGNITUDE_CMAP, edgecolors="w", linewidths=0.3)
    cells.set_array(data["zb"])
    ax.add_collection(cells)
    colorbar(ax, cells, "(m)").set_label(r"$z_b$", fontsize=9)

    # Gauges G1-G3 on the centreline.
    for j, (x, y) in enumerate(data["gauge_xy"]):
        ax.plot(x, y, GAUGE_MARKERS[j], ms=6, mfc="w", mec="k", mew=1.0)
        ax.text(x, y + 55.0, rf"G{j + 1}", ha="center", va="bottom", fontsize=9, color="k")

    # Inflow and exit annotations.
    ax.annotate(r"$Q(t)$", xy=(10.0, 100.0), xytext=(-75.0, 100.0), fontsize=9,
                va="center", arrowprops={"arrowstyle": "->", "lw": 1.0})
    ax.text(1030.0, 100.0, "stage BC", fontsize=8, va="center", ha="center", rotation=90)

    # Axes.
    ax.set_xlim(-90.0, 1065.0)
    ax.set_ylim(0.0, 200.0)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.set_xticks(np.arange(0, 1001, 200))


def plot_hydrographs(ax: plt.Axes, summaries: dict, t_knots_h: np.ndarray) -> None:
    """Panel (b): true, initial and recovered hydrographs.

    Args:
        ax: target axes.
        summaries: inversion summaries keyed by noise level (mm).
        t_knots_h: knot times (h), shape [n_knots].
    """
    t_fine = np.linspace(0.0, 7200.0, 400)
    ax.plot(t_fine / 3600.0, q_true(t_fine), label="true", **REFERENCE)
    ax.plot(t_knots_h, np.full(len(t_knots_h), Q_BASE), label="initial", **INITIAL)
    ax.plot(t_knots_h, summaries[0.0]["q_inv"], marker="o", mfc="none", ms=4,
            label="recovered, 0 mm", **MODEL)
    ax.plot(t_knots_h, summaries[5.0]["q_inv"], marker="s", mfc="none", ms=4,
            label="recovered, 5 mm", **MODEL_2)
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 540.0)
    ax.set_xlabel(r"$t$ (h)")
    ax.set_ylabel(r"$Q$ (m$^3$\,s$^{-1}$)")
    ax.legend(loc="upper right", fontsize=7, handlelength=2.0, labelspacing=0.25)


def plot_stages(ax: plt.Axes, data: np.lib.npyio.NpzFile) -> None:
    """Panel (c): observed (5 mm noise), fitted and initial stages at the gauges.

    Args:
        ax: target axes.
        data: arrays from ``hydrograph_inversion.npz``.
    """
    t_h = data["t_obs_s"] / 3600.0
    for j in range(data["obs_noise5mm_m"].shape[1]):
        # Observations (one marker per gauge), fitted and initial model stages.
        first = j == 0
        ax.plot(t_h, data["obs_noise5mm_m"][:, j], GAUGE_MARKERS[j], ms=3.2, mfc="none",
                mec="k", mew=0.7, label="observed" if first else None)
        ax.plot(t_h, data["wse_fit_noise5mm_m"][:, j], label="fitted, 5 mm" if first else None,
                **MODEL_2)
        ax.plot(t_h, data["wse_fit_noise0mm_m"][:, j], label="fitted, 0 mm" if first else None,
                **MODEL)
        ax.plot(t_h, data["wse_initial_m"][:, j], label="initial" if first else None, **INITIAL)

        # Gauge name just above the fitted peak stage.
        peak = int(np.argmax(data["wse_fit_noise5mm_m"][:, j]))
        ax.text(t_h[peak], data["wse_fit_noise5mm_m"][peak, j] + 0.07, f"G{j + 1}",
                fontsize=8, va="bottom", ha="center")
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(1.1, 4.7)
    ax.set_xlabel(r"$t$ (h)")
    ax.set_ylabel("WSE (m)")
    ax.legend(loc="upper center", fontsize=7, handlelength=1.3, ncol=2, labelspacing=0.25,
              columnspacing=0.4, handletextpad=0.3)


def plot_loss(ax: plt.Axes, summaries: dict) -> None:
    """Panel (d): loss (m^2) against the L-BFGS iteration.

    Args:
        ax: target axes.
        summaries: inversion summaries keyed by noise level (mm).
    """
    for noise_mm, style, marker in ((0.0, MODEL, "o"), (5.0, MODEL_2, "s")):
        loss = np.asarray(summaries[noise_mm]["loss_history"])
        ax.semilogy(np.arange(len(loss)), loss, marker=marker, ms=2.5, mfc="none",
                    label=f"{noise_mm:g} mm noise", **style)

    # Loss of the noise alone, sigma^2 = (5 mm)^2.
    ax.axhline(0.005 ** 2, color="0.5", ls=":", lw=1.0)
    ax.text(23.0, 0.005 ** 2 * 1.3, r"$\sigma^2$, 5 mm", fontsize=8, color="0.35", ha="right",
            va="bottom")
    ax.set_xlim(0, 24)
    ax.set_ylim(1.0e-5, 3.0)
    ax.set_xlabel("L-BFGS iteration")
    ax.set_ylabel(r"loss (m$^2$)")
    ax.legend(loc="upper right", fontsize=7.5)


def main() -> None:
    """Make Figure 7 and write the numbers."""
    apply_style()

    # 1. Data.
    data = np.load(DATA / "hydrograph_inversion.npz")
    summaries = {}
    for noise_mm in (0.0, 5.0):
        path = DATA / f"summary_noise{noise_mm:g}mm.json"
        summaries[noise_mm] = json.loads(path.read_text())
    t_knots_h = data["t_knots_s"] / 3600.0

    # 2. Layout: (a) the map across the full width; (b)-(d) in one row below.
    fig = plt.figure(figsize=(FULL_WIDTH, 4.6))
    grid = fig.add_gridspec(2, 1, height_ratios=[0.8, 1.25], hspace=0.38)
    bottom = grid[1].subgridspec(1, 3, width_ratios=[1.0, 1.0, 0.85], wspace=0.42)
    ax_map = fig.add_subplot(grid[0])
    ax_q = fig.add_subplot(bottom[0])
    ax_w = fig.add_subplot(bottom[1])
    ax_loss = fig.add_subplot(bottom[2])

    # 3. Panels.
    plot_domain(ax_map, data)
    plot_hydrographs(ax_q, summaries, t_knots_h)
    plot_stages(ax_w, data)
    plot_loss(ax_loss, summaries)
    panel_label(ax_map, "(a)", x=-0.09, y=1.06)
    panel_label(ax_q, "(b)", x=-0.3)
    panel_label(ax_w, "(c)", x=-0.3)
    panel_label(ax_loss, "(d)", x=-0.36)
    save(fig, "fig07_hydrograph_inversion")

    # 4. Numbers for the text.
    obs = data["obs_clean_m"]
    results = {
        "cells": summaries[0.0]["cells"],
        "domain_m": [1000.0, 200.0],
        "true_peak_m3s": Q_PEAK,
        "true_time_to_peak_min": T_PEAK / 60.0,
        "gauges": 3,
        "stage_rise_at_gauges_m": (obs.max(axis=0) - obs[0]).tolist(),
        "observation_interval_min": 5.0,
        "knots": len(t_knots_h),
        "knot_spacing_min": 15.0,
        "inversion_model": "implicit backward Euler, dt = 60 s, per-step IFT adjoint",
        "truth_model": "explicit SSPRK3, second order, CFL steps",
        "optimizer": "L-BFGS (optax, zoom line search)",
        "hardware": HARDWARE,
        "runs": {},
    }
    for index, noise_mm in enumerate((0.0, 5.0)):
        summary = summaries[noise_mm]
        q_inv = np.asarray(summary["q_inv"])
        results["runs"][f"noise_{noise_mm:g}mm"] = {
            "iterations": summary["iterations"],
            "wall_s": summary["wall_s"],
            "q_rms_error_m3s": summary["q_rms_error_m3s"],
            "q_peak_recovered_m3s": summary["q_peak_inv"],
            "peak_error_pct": 100.0 * (summary["q_peak_inv"] - Q_PEAK) / Q_PEAK,
            "time_of_recovered_peak_min": float(t_knots_h[np.argmax(q_inv)] * 60.0),
            "loss_initial_m2": summary["loss_history"][0],
            "loss_final_m2": summary["loss_history"][-1],
            "rms_stage_misfit_last_iterate_mm": summary["final_rms_stage_misfit_mm"],
            "rms_stage_misfit_final_q_recomputed_mm": float(
                data["rms_misfit_recomputed_mm"][index]),
        }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
