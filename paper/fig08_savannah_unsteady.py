"""Figure 8: unsteady Manning's-n calibration on the Savannah River mesh from stage gauges.

Case (phase 2 report, section 4k): real SRH-2D mesh of 1,306 cells with 5 roughness zones,
true Manning's n = (0.040, 0.050, 0.030, 0.045, 0.050) s/m^(1/3) (zone 3 is the main
channel). A flood hydrograph 187.4 -> 450 -> 187.4 m^3/s over 6 h enters at the inlet; the
exit water-surface elevation (WSE) is held at 29.3 m. Synthetic observations: WSE every
10 min from the explicit second-order SSPRK3 model at the true n. Inversion model: implicit
backward Euler (Delta t = 30 s, 720 steps) with per-step implicit-function-theorem adjoints;
L-BFGS on p = log n with a zoom line search; loss = mean squared stage misfit (m^2).

Panels
------
(a) mesh coloured by roughness zone, with the two gauge layouts: 6 gauges at the deepest
    cell of 6 streamwise bins ("bins"; 5 fall in the main channel) and 10 gauges, two per
    zone;
(b) inflow hydrograph Q_in(t) (m^3/s);
(c) Manning's n of each zone against the L-BFGS iteration for the four runs, truth as
    black lines;
(d) RMS stage misfit (mm) against the iteration (line styles as in (c)), with the
    model-error floor (implicit Delta t = 30 s against the explicit truth at the true n);
(e) identifiability at the true n: singular values s_k of J = d WSE / d log n over all
    (time, gauge) observations, scaled to the RMS stage change per unit log n,
    s_k / sqrt(N_obs) (mm), for the two layouts.

Data: ``paper/data/savannah_unsteady/`` (made by ``prep_savannah_unsteady.py``).
Numbers: ``paper/results/savannah_unsteady.json``.
Usage: ``python paper/fig08_savannah_unsteady.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figstyle import (
    FULL_WIDTH,
    MODEL,
    MODEL_2,
    MODEL_3,
    REFERENCE,
    apply_style,
    panel_label,
    save,
)

PAPER = Path(__file__).resolve().parent
DATA = PAPER / "data" / "savannah_unsteady"
RESULTS = PAPER / "results" / "savannah_unsteady.json"

# Zone colours (colour-vision-safe Okabe-Ito tones) and names, zones 1-5.
ZONE_COLOURS = ["#E69F00", "#56B4E9", "#9ecae1", "#009E73", "#F0E442"]
ZONE_COLOURS[2] = "#0072B2"
ZONE_NAMES = ["zone 1", "zone 2", "zone 3 (channel)", "zone 4", "zone 5"]

# Line style and legend label of each run.
RUN_STYLE = {
    "1": {**MODEL_2, "label": r"6 gauges, $n_0{=}0.03$"},
    "2a": {**MODEL, "label": r"10 gauges, $n_0{=}0.03$"},
    "2b": {**MODEL_3, "label": r"10 gauges, $n_0{=}0.045$"},
    "2c": {"color": "tab:purple", "linestyle": (0, (5, 1, 1, 1)), "linewidth": 1.2,
           "label": r"10 gauges, $n_0{=}0.06$"},
}


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


def plot_mesh_zones(ax: plt.Axes, data: np.lib.npyio.NpzFile) -> None:
    """Panel (a): cells coloured by zone, with both gauge layouts.

    Args:
        ax: target axes.
        data: arrays from ``savannah_unsteady.npz``.
    """
    # Cells filled with the zone colour, thin grey edges to show the unstructured mesh.
    polygons = cell_polygons(data["nodes_xy"], data["cell_nodes"])
    zone_index = data["mat_id"] - 1
    cmap = ListedColormap(ZONE_COLOURS)
    cells = PolyCollection(polygons, cmap=cmap, edgecolors="0.35", linewidths=0.15)
    cells.set_array(zone_index.astype(float))
    cells.set_clim(-0.5, 4.5)
    ax.add_collection(cells)

    # Gauges: 6-gauge layout as open triangles, 10-gauge layout as filled circles.
    centroid = data["centroid"]
    bins = centroid[data["gauges_bins"]]
    zones = centroid[data["gauges_zones"]]
    ax.plot(bins[:, 0], bins[:, 1], "^", ms=6.5, mfc="none", mec="k", mew=1.1)
    ax.plot(zones[:, 0], zones[:, 1], "o", ms=3.8, mfc="tab:red", mec="k", mew=0.5)

    # Axes and equal scale; zone legend at the lower left, gauge legend at the upper right
    # (both in parts of the frame that hold no cells).
    ax.set_xlim(0.0, 1000.0)
    ax.set_ylim(0.0, 580.0)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    zone_handles = []
    for colour, name in zip(ZONE_COLOURS, ZONE_NAMES, strict=True):
        zone_handles.append(Patch(facecolor=colour, edgecolor="0.35", linewidth=0.4, label=name))
    zone_legend = ax.legend(handles=zone_handles, loc="lower left", fontsize=7, handlelength=1.0,
                            labelspacing=0.2, borderaxespad=0.3)
    ax.add_artist(zone_legend)
    gauge_handles = [
        Line2D([], [], marker="^", ls="none", mfc="none", mec="k", ms=6, label="6 gauges (along channel)"),
        Line2D([], [], marker="o", ls="none", mfc="tab:red", mec="k", ms=4,
               label="10 gauges (2 per zone)"),
    ]
    ax.legend(handles=gauge_handles, loc="upper right", fontsize=7, handlelength=1.0,
              labelspacing=0.2, borderaxespad=0.3)


def plot_hydrograph(ax: plt.Axes, data: np.lib.npyio.NpzFile) -> None:
    """Panel (b): prescribed inflow hydrograph and observation times.

    Args:
        ax: target axes.
        data: arrays from ``savannah_unsteady.npz``.
    """
    t_h = data["t_hydrograph_s"] / 3600.0
    ax.plot(t_h, data["q_hydrograph_m3s"], **REFERENCE)

    # Observation times (every 10 min) as ticks along the bottom axis.
    t_obs_h = np.arange(0.0, float(data["t_end_s"]) + 1.0, float(data["dt_obs_s"])) / 3600.0
    ax.plot(t_obs_h, np.full(len(t_obs_h), 160.0), "|", color="0.4", ms=4)
    ax.text(3.0, 168.0, "observations", fontsize=8, ha="center", va="bottom", color="0.3")
    ax.set_xlim(0.0, 6.0)
    ax.set_ylim(150.0, 480.0)
    ax.set_xticks(np.arange(0, 7, 1))
    ax.set_xlabel(r"$t$ (h)")
    ax.set_ylabel(r"$Q_\mathrm{in}$ (m$^3$\,s$^{-1}$)")


def plot_n_histories(axes: list[plt.Axes], runs: dict) -> None:
    """Panel (c): n of each zone against the L-BFGS iteration, truth as black lines.

    Args:
        axes: five axes, one per zone.
        runs: run records from ``runs.json``.
    """
    n_true = runs["2a"]["n_true"]
    for zone, ax in enumerate(axes):
        # Truth, then every run's iterates.
        ax.axhline(n_true[zone], **REFERENCE)
        for label, run in runs.items():
            n_history = np.asarray(run["history"]["n"])
            style = dict(RUN_STYLE[label])
            style.pop("label")
            ax.plot(np.arange(len(n_history)), n_history[:, zone], **style)
        ax.set_title(ZONE_NAMES[zone], fontsize=9, pad=3)
        ax.set_xlim(0, 29)
        ax.set_xticks([0, 10, 20])
        ax.set_ylim(0.025, 0.063)
        ax.set_xlabel("iteration")
        if zone > 0:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel(r"$n$ (s\,m$^{-1/3}$)")


def plot_misfit(ax: plt.Axes, runs: dict, floor_mm: float) -> None:
    """Panel (d): RMS stage misfit against the iteration, with the model-error floor.

    Args:
        ax: target axes.
        runs: run records from ``runs.json``.
        floor_mm: model-error floor (mm).
    """
    for label, run in runs.items():
        misfit = np.asarray(run["history"]["misfit_mm"])
        ax.semilogy(np.arange(len(misfit)), misfit, **RUN_STYLE[label])
    ax.axhline(floor_mm, color="0.5", ls=":", lw=1.0)
    ax.text(0.5, floor_mm * 1.15, "model-error floor", fontsize=8, color="0.35", va="bottom")
    ax.set_xlim(0, 29)
    ax.set_ylim(0.08, 300.0)
    ax.set_xlabel("L-BFGS iteration")
    ax.set_ylabel("RMS stage misfit (mm)")


def plot_singular_values(ax: plt.Axes, ident: dict, n_obs: dict) -> dict:
    """Panel (e): scaled singular values of the sensitivity matrix for both layouts.

    Args:
        ax: target axes.
        ident: contents of ``identifiability.json`` (singular values in m).
        n_obs: number of (time, gauge) observations per layout.

    Returns:
        Scaled singular values (mm per unit log n) per layout.
    """
    scaled = {}
    styles = {"bins": (MODEL_2, "^", "6 gauges (along channel)"),
              "zones": (MODEL, "o", "10 gauges (2 per zone)")}
    k = np.arange(1, 6)
    for layout, (style, marker, label) in styles.items():
        values_mm = np.asarray(ident[layout]["singular_values_m"]) / np.sqrt(n_obs[layout]) * 1e3
        scaled[layout] = values_mm.tolist()
        condition = ident[layout]["condition"]
        ax.semilogy(k, values_mm, marker=marker, mfc="none", ms=5,
                    label=rf"{label}, $\kappa = {condition:.0f}$", **style)
    ax.set_xticks(k)
    ax.set_xlim(0.7, 5.3)
    ax.set_ylim(0.8, 500.0)
    ax.set_xlabel(r"singular direction $k$")
    ax.set_ylabel(r"$s_k/\sqrt{N_\mathrm{obs}}$ (mm)")
    ax.legend(loc="upper right", fontsize=7.5, handlelength=2.6)
    return scaled


def main() -> None:
    """Make Figure 8 and write the numbers."""
    apply_style()

    # 1. Data.
    data = np.load(DATA / "savannah_unsteady.npz")
    runs = json.loads((DATA / "runs.json").read_text())
    ident = json.loads((DATA / "identifiability.json").read_text())
    floor_m = data["stage_implicit_true_n_m"] - data["stage_explicit_true_n_m"]
    floor_mm = float(np.sqrt(np.mean(floor_m ** 2)) * 1e3)
    n_times = int(round(float(data["t_end_s"]) / float(data["dt_obs_s"])))  # t = 0 dropped
    n_obs = {"bins": n_times * len(data["gauges_bins"]),
             "zones": n_times * len(data["gauges_zones"])}

    # 2. Layout: (a) map + (b) hydrograph; (c) five n panels; (d) misfit + (e) singular values.
    fig = plt.figure(figsize=(FULL_WIDTH, 7.9))
    grid = fig.add_gridspec(3, 1, height_ratios=[1.25, 0.85, 1.0], hspace=0.55)
    top = grid[0].subgridspec(1, 2, width_ratios=[1.85, 1.0], wspace=0.28)
    middle = grid[1].subgridspec(1, 5, wspace=0.08)
    bottom = grid[2].subgridspec(1, 2, wspace=0.3)
    ax_map = fig.add_subplot(top[0])
    ax_q = fig.add_subplot(top[1])
    ax_n = [fig.add_subplot(middle[i]) for i in range(5)]
    ax_misfit = fig.add_subplot(bottom[0])
    ax_sv = fig.add_subplot(bottom[1])

    # 3. Panels.
    plot_mesh_zones(ax_map, data)
    plot_hydrograph(ax_q, data)
    plot_n_histories(ax_n, runs)
    plot_misfit(ax_misfit, runs, floor_mm)
    scaled = plot_singular_values(ax_sv, ident, n_obs)
    # Match the hydrograph panel's height to the equal-aspect map.
    fig.canvas.draw()
    map_box = ax_map.get_position()
    q_box = ax_q.get_position()
    ax_q.set_position([q_box.x0, map_box.y0, q_box.width, map_box.height])

    # Run legend above the n panels (shared by (c) and (d)), truth included.
    run_handles = [Line2D([], [], label=r"true $n$", **REFERENCE)]
    for style in RUN_STYLE.values():
        run_handles.append(Line2D([], [], **style))
    n_box = ax_n[0].get_position()
    fig.legend(handles=run_handles, loc="lower center", ncol=5, fontsize=7,
               handlelength=2.0, handletextpad=0.4, columnspacing=0.9,
               bbox_to_anchor=(0.55, n_box.y1 + 0.028))

    panel_label(ax_map, "(a)", x=-0.1)
    panel_label(ax_q, "(b)", x=-0.25)
    panel_label(ax_n[0], "(c)", x=-0.5, y=1.1)
    panel_label(ax_misfit, "(d)", x=-0.24)
    panel_label(ax_sv, "(e)", x=-0.18)
    save(fig, "fig08_savannah_unsteady")

    # 4. Numbers for the text.
    results = {
        "cells": len(data["mat_id"]),
        "zones": 5,
        "n_true_s_per_m13": runs["2a"]["n_true"],
        "hydrograph_m3s": {"base": 187.4, "peak": 450.0, "duration_h": 6.0},
        "exit_wse_m": 29.3,
        "observation_interval_min": float(data["dt_obs_s"]) / 60.0,
        "inversion_model": "implicit backward Euler, dt = 30 s (720 steps), per-step IFT adjoint",
        "optimizer": "L-BFGS (optax, zoom line search), p = log n, 30 iterations",
        "model_error_floor_rms_mm": floor_mm,
        "model_error_floor_max_mm": float(np.abs(floor_m).max() * 1e3),
        "runs": {},
        "identifiability": {},
    }
    for label, run in runs.items():
        results["runs"][label] = {
            key: run[key] for key in ("gauge_layout", "n_gauges", "n_start", "hardware",
                                      "iterations", "n_final", "final_rms_misfit_mm")
        }
        results["runs"][label]["wall_h"] = run["wall_s"] / 3600.0
        results["runs"][label]["n_rel_error_pct"] = [100 * e for e in run["n_rel_error"]]
        results["runs"][label]["max_n_rel_error_pct"] = 100 * max(run["n_rel_error"])
        results["runs"][label]["initial_rms_misfit_mm"] = run["history"]["misfit_mm"][0]
    for layout in ("bins", "zones"):
        # 1-sigma uncertainty in log n along each singular direction for 5 mm stage noise:
        # sigma_k = sigma_noise / s_k (linearised least squares, s_k in m).
        singular_values_m = np.asarray(ident[layout]["singular_values_m"])
        sigma_log_n = 0.005 / singular_values_m
        results["identifiability"][layout] = {
            "n_gauges": len(data[f"gauges_{layout}"]),
            "rms_stage_change_per_unit_log_n_mm": scaled[layout],
            "condition_number": ident[layout]["condition"],
            "sigma_log_n_at_5mm_noise": sigma_log_n.tolist(),
            "strongest_direction_dlogn": ident[layout]["strongest_direction_dlogn"],
            "weakest_direction_dlogn": ident[layout]["weakest_direction_dlogn"],
        }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=1))
    print(json.dumps(results["runs"], indent=1))


if __name__ == "__main__":
    main()
