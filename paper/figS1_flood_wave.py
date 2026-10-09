"""Supplementary Figure S1: flood wave through a compound channel with wetting and drying.

Case (phase 2 report, section 4b): valley 2 km x 200 m with bed slope 1e-3, a trapezoidal
main channel 40 m wide and 2 m deep, and floodplains rising 3 % towards the valley sides;
Manning n = 0.035 s/m^(1/3); 400 x 40 = 16,000 quadrilateral cells of 5 m. The inflow
hydrograph rises from 40 to 400 m^3/s (bankfull about 120 m^3/s) and falls back; the exit
stage is the normal-flow stage of the compound section for the current discharge. The
floodplains wet on the rising limb and dry on the falling limb. Numerics: hydrostatic
reconstruction + HLLC, second-order MUSCL, SSPRK3 with CFL-limited steps.

Panels
------
(a) water depth h (m) on the mesh at four times (dry cells, h < 1 cm, left white);
(b) inflow Q_in(t) and outflow Q_out(t) = Q_in - dV/dt (m^3/s), V the stored volume (m^3);
(c) inundated area (h > 1 cm) as a percentage of the valley area.

Data: ``paper/data/flood_wave/`` (made by ``prep_flood_wave.py`` on an A100).
Numbers: ``paper/results/flood_wave.json``.
Usage: ``python paper/figS1_flood_wave.py``.
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
    REFERENCE,
    apply_style,
    panel_label,
    save,
)

PAPER = Path(__file__).resolve().parent
DATA = PAPER / "data" / "flood_wave"
RESULTS = PAPER / "results" / "flood_wave.json"

# Depth threshold for "wet" (m) and the snapshot times (h): base flow, rising limb, inflow
# peak (close to the largest inundation), falling limb.
H_WET = 0.01
SNAPSHOT_TIMES_H = (0.0, 1.0, 1.5, 3.0)
# Upper limit of the depth colour scale (m).
DEPTH_MAX = 4.0


def cell_polygons(nodes_xy: np.ndarray, cell_nodes: np.ndarray) -> np.ndarray:
    """Polygon vertices of every cell (all cells have the same number of nodes here).

    Args:
        nodes_xy: node coordinates (m), shape [n_nodes, 2].
        cell_nodes: node indices per cell, shape [n_cells, 4] (no padding on this mesh).

    Returns:
        Vertices (m), shape [n_cells, 4, 2].
    """
    return nodes_xy[cell_nodes]


def peak_time_and_value(t_s: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    """Time and value of a series maximum, refined by a parabola through three samples.

    Args:
        t_s: uniformly spaced times (s), shape [n].
        q: series values, shape [n].

    Returns:
        (time of the maximum (s), maximum value).
    """
    k = int(np.argmax(q))
    dt = t_s[1] - t_s[0]

    # Vertex of the parabola through (k-1, k, k+1).
    left = q[k - 1]
    centre = q[k]
    right = q[k + 1]
    curvature = left - 2.0 * centre + right
    shift = 0.5 * (left - right) / curvature
    value = centre - 0.25 * (left - right) * shift
    return float(t_s[k] + shift * dt), float(value)


def plot_snapshots(axes: list[plt.Axes], data: np.lib.npyio.NpzFile, frames: list[int]):
    """Panel (a): depth maps at the selected frames.

    Args:
        axes: one axes per snapshot.
        data: arrays from ``flood_wave.npz``.
        frames: indices into the 15-min map frames, one per axes.

    Returns:
        The last PolyCollection (for the colour bar).
    """
    polygons = cell_polygons(data["nodes_xy"], data["cell_nodes"])
    t_maps_h = data["t_maps_s"] / 3600.0
    q_in_maps = np.interp(data["t_maps_s"], data["t_s"], data["q_in_m3s"])
    cells = None
    for ax, frame in zip(axes, frames, strict=True):
        # Wet cells only, coloured by depth; dry cells stay white.
        depth = data["depth_maps_m"][frame]
        wet = depth > H_WET
        cells = PolyCollection(polygons[wet], cmap=MAGNITUDE_CMAP, edgecolors="face",
                               linewidths=0.1, rasterized=True)
        cells.set_array(depth[wet])
        cells.set_clim(0.0, DEPTH_MAX)
        ax.add_collection(cells)

        # Valley outline, axes and the time / inflow label.
        ax.plot([0, 2000, 2000, 0, 0], [0, 0, 200, 200, 0], color="0.3", lw=0.6)
        ax.set_xlim(0.0, 2000.0)
        ax.set_ylim(0.0, 200.0)
        ax.set_aspect("equal")
        ax.set_yticks([0, 100, 200])
        ax.set_ylabel(r"$y$ (m)", fontsize=9)
        label = (rf"$t = {t_maps_h[frame]:.1f}$ h, "
                 + rf"$Q_\mathrm{{in}} = {q_in_maps[frame]:.0f}$ m$^3$\,s$^{{-1}}$")
        ax.text(0.008, 0.95, label, transform=ax.transAxes, fontsize=8, va="top", ha="left",
                bbox={"facecolor": "w", "edgecolor": "none", "pad": 1.0, "alpha": 0.85})
        if ax is not axes[-1]:
            ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel(r"$x$ (m)")
    return cells


def main() -> None:
    """Make Figure S1 and write the numbers."""
    apply_style()

    # 1. Data and derived series.
    data = np.load(DATA / "flood_wave.npz")
    info = json.loads((DATA / "run_info.json").read_text())
    t_s = data["t_s"]
    t_h = t_s / 3600.0
    wet_pct = 100.0 * data["wet_area_m2"] / info["valley_area_m2"]

    # 2. Snapshot frames (nearest 15-min map to each requested time).
    t_maps_s = data["t_maps_s"]
    frames = []
    for time_h in SNAPSHOT_TIMES_H:
        frames.append(int(np.argmin(np.abs(t_maps_s - time_h * 3600.0))))

    # 3. Layout: four stacked maps on top; hydrographs and inundated area below.
    fig = plt.figure(figsize=(FULL_WIDTH, 6.4))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.55, 1.0], hspace=0.3)
    maps = grid[0].subgridspec(4, 1, hspace=0.12)
    lower = grid[1].subgridspec(1, 2, wspace=0.32)
    ax_maps = [fig.add_subplot(maps[i]) for i in range(4)]
    ax_q = fig.add_subplot(lower[0])
    ax_area = fig.add_subplot(lower[1])

    # 4. (a) depth maps with one shared colour bar to the right of the stack.
    cells = plot_snapshots(ax_maps, data, frames)
    fig.canvas.draw()
    top_box = ax_maps[0].get_position()
    bottom_box = ax_maps[-1].get_position()
    cax = fig.add_axes([0.93, bottom_box.y0, 0.012, top_box.y1 - bottom_box.y0])
    bar = fig.colorbar(cells, cax=cax)
    bar.ax.set_title("(m)", fontsize=9, pad=4)
    bar.set_label(r"$h$", fontsize=9)
    bar.ax.tick_params(labelsize=8)

    # 5. (b) inflow and outflow hydrographs, bankfull discharge and snapshot times.
    ax_q.plot(t_h, data["q_in_m3s"], label=r"$Q_\mathrm{in}$ (prescribed)", **REFERENCE)
    ax_q.plot(t_h, data["q_out_m3s"], label=r"$Q_\mathrm{out} = Q_\mathrm{in} - dV/dt$",
              **MODEL)
    ax_q.axhline(info["bankfull_m3s_approx"], color="0.5", ls=":", lw=1.0)
    ax_q.text(5.9, info["bankfull_m3s_approx"] + 8.0, "bankfull", fontsize=8, color="0.35",
              ha="right", va="bottom")
    ax_q.set_xlim(0.0, 6.0)
    ax_q.set_ylim(0.0, 450.0)
    ax_q.set_xticks(np.arange(0, 7, 1))
    ax_q.set_xlabel(r"$t$ (h)")
    ax_q.set_ylabel(r"$Q$ (m$^3$\,s$^{-1}$)")
    ax_q.legend(loc="upper right", fontsize=7.5)

    # 6. (c) inundated area.
    ax_area.plot(t_h, wet_pct, **REFERENCE)
    ax_area.set_xlim(0.0, 6.0)
    ax_area.set_ylim(0.0, 80.0)
    ax_area.set_xticks(np.arange(0, 7, 1))
    ax_area.set_xlabel(r"$t$ (h)")
    ax_area.set_ylabel(r"inundated area (\%)")

    # Snapshot times as short ticks at the top of (b) and (c).
    for frame in frames:
        for ax in (ax_q, ax_area):
            ax.axvline(t_maps_s[frame] / 3600.0, ymin=0.94, ymax=1.0, color="0.3", lw=1.0)
    panel_label(ax_maps[0], "(a)", x=-0.075, y=1.08)
    panel_label(ax_q, "(b)", x=-0.2)
    panel_label(ax_area, "(c)", x=-0.2)
    save(fig, "figS1_flood_wave")

    # 7. Numbers: peaks, attenuation, lag, inundation, mass balance, cost.
    # The inflow is piecewise linear: its peak is the largest sample (no parabola).
    k_in = int(np.argmax(data["q_in_m3s"]))
    t_in = float(t_s[k_in])
    q_in_peak = float(data["q_in_m3s"][k_in])
    t_out, q_out_peak = peak_time_and_value(t_s, data["q_out_m3s"])
    t_col, q_col_peak = peak_time_and_value(t_s, data["q_out_column_m3s"])
    k_wet = int(np.argmax(wet_pct))
    results = {
        "cells": info["cells"],
        "spatial_order": info["order"],
        "solver": info["solver"],
        "time_steps": info["n_steps"],
        "simulated_h": info["simulated_h"],
        "hardware": info["device"],
        "wall_s": info["wall_s"],
        "wall_first_call_incl_compile_s": info["wall_first_call_incl_compile_s"],
        "wall_s_report_4b_first_order_cfl05": 88.4,
        "q_in_peak_m3s": q_in_peak,
        "q_in_peak_time_h": t_in / 3600.0,
        "q_out_peak_m3s": q_out_peak,
        "q_out_peak_time_h": t_out / 3600.0,
        "peak_attenuation_m3s": q_in_peak - q_out_peak,
        "peak_attenuation_pct": 100.0 * (q_in_peak - q_out_peak) / q_in_peak,
        "peak_lag_min": (t_out - t_in) / 60.0,
        "q_out_peak_from_exit_column_m3s": q_col_peak,
        "q_out_peak_time_from_exit_column_h": t_col / 3600.0,
        "inundated_pct_initial": float(wet_pct[0]),
        "inundated_pct_max": float(wet_pct[k_wet]),
        "inundated_max_time_h": float(t_h[k_wet]),
        "inundated_pct_final": float(wet_pct[-1]),
        "min_depth_m": float(data["min_depth_m"].min()),
        "storage_change_m3": float(data["storage_m3"][-1] - data["storage_m3"][0]),
        "snapshot_times_h": [float(t_maps_s[f] / 3600.0) for f in frames],
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
