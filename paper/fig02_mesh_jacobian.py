"""Figure 2: Savannah River mesh, distance-2 graph colouring and sparsity of dR/dU.

The Savannah River case (Hydrograd.jl example, SRH-2D mesh) is a 1 km reach discretised
by 1306 mixed quadrilateral and triangular cells with five Manning roughness zones.

Panels
------
(a) The unstructured mesh coloured by material (Manning roughness) zone, with an inset
    that zooms into the transition region where triangles and quadrilaterals mix.
(b) The distance-2 colouring of the cell graph used to compress the exact sparse Jacobian
    J = dR/dU of the second-order residual R(U) (``hydrograd.solvers.jacobian``): two cells
    share a colour only if no residual depends on both, so one Jacobian-vector product per
    colour and state component recovers every nonzero (Curtis-Powell-Reid compression).
(c) The sparsity pattern of dR/dU = J_local + A C^T for the state U = [xi, q_x, q_y]
    (3 x 3 blocks of size n_cells), second-order MUSCL stencil. J_local is the sparse
    two-ring stencil; A C^T is the rank-one coupling of the inlet cells through the
    conveyance-weighted inlet discharge split, which is handled with the Woodbury identity
    and is shown in a second colour.
(d) A zoom of (c) into the xi-xi block around the inlet cells.

The state U has xi (m) and q_x, q_y (m^2/s); R = dU/dt has the units of U per second.

Numbers written to ``paper/results/mesh_jacobian.json``: n_cells, n_faces, cell types,
number of colours at first and second order, and the nonzero counts.

Usage
-----
    python paper/fig02_mesh_jacobian.py
"""

from __future__ import annotations

import collections
import dataclasses
import sys
from pathlib import Path
from typing import Any

import jax
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))

from figstyle import FULL_WIDTH, apply_style, panel_label, save
from savannah_common import (
    FORWARD_CASE,
    copy_case,
    mesh_polygons,
    poly_field,
    write_results,
)

import hydrograd  # noqa: F401  (enables float64)
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.solvers.jacobian import JacobianPattern, jacobian_pattern, linearize

# Zoom window of the inset (m): the region where the mesh mixes triangles and quads.
ZOOM_X = (515.0, 600.0)
ZOOM_Y = (120.0, 220.0)

# Colour-vision-safe categorical colours for the material zones (Okabe-Ito palette).
ZONE_COLOURS = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#CC79A7", "#0072B2"]


def compute() -> dict[str, Any]:
    """Build the Savannah model and its Jacobian pattern, colouring and nonzeros.

    Returns:
        Dictionary with the mesh, the material ID per cell [n_cells], the zone names and
        Manning's n values, the colouring at second order [n_cells], and the scalar
        nonzero coordinates of J_local and of the inlet term A C^T (row, column arrays).
    """
    # 1. Copy the example (Hydrograd.jl is never written to) and set up the case.
    case_dir = copy_case(FORWARD_CASE, "Savannah_River")
    case = setup_case(read_control(case_dir / "run_control.json"))
    model = case.model
    mesh = case.mesh

    # 2. Colourings at first and second order (the default scheme is second order).
    model_order1 = dataclasses.replace(model, order=1)
    model_order2 = dataclasses.replace(model, order=2)
    pattern1 = JacobianPattern(model_order1)
    pattern2 = jacobian_pattern(model_order2)

    # 3. Exact Jacobian at the SRH-2D steady state (the case's initial condition).
    compressed, A, C = jax.jit(lambda U: linearize(model_order2, {}, U))(case.U0)
    J_local = pattern2.assemble(np.asarray(compressed)).tocoo()

    # 4. Nonzeros of the rank-n_in inlet term A C^T (n_in = number of inlets).
    low_rank = np.asarray(A) @ np.asarray(C).T
    lr_rows, lr_cols = np.nonzero(low_rank)

    # Entries of A C^T that are not already in the J_local pattern.
    local_keys = set(zip(pattern2.R.tolist(), pattern2.C.tolist()))
    extra = [(r, c) for r, c in zip(lr_rows.tolist(), lr_cols.tolist()) if (r, c) not in local_keys]

    return {
        "mesh": mesh,
        "mat_id": np.asarray(model.mat_id),
        "zone_names": case.srh.mat.names,
        "n_zone": np.asarray(model.n_zone),
        "pattern1": pattern1,
        "pattern2": pattern2,
        "J_rows": J_local.row,
        "J_cols": J_local.col,
        "J_values": J_local.data,
        "lr_rows": lr_rows,
        "lr_cols": lr_cols,
        "n_extra": len(extra),
        "n_inlets": int(np.asarray(A).shape[1]),
    }


def record(data: dict[str, Any]) -> dict[str, Any]:
    """Collect the mesh and Jacobian statistics and write ``mesh_jacobian.json``.

    Args:
        data: output of ``compute``.

    Returns:
        The results dictionary.
    """
    mesh = data["mesh"]
    pattern1 = data["pattern1"]
    pattern2 = data["pattern2"]

    # Cell types by vertex count.
    counts = collections.Counter(len(nodes) for nodes in mesh.cell_nodes)
    names = {3: "triangles", 4: "quadrilaterals"}
    cell_types = {names.get(k, f"{k}-gons"): int(v) for k, v in sorted(counts.items())}

    # Cells per material zone (zone 0 is SRH-2D's default zone).
    cells_per_zone = np.bincount(data["mat_id"], minlength=len(data["n_zone"]))
    zones = []
    for zone_id, n_value in enumerate(data["n_zone"]):
        zones.append({
            "id": zone_id,
            "name": data["zone_names"].get(zone_id, ""),
            "manning_n": float(n_value),
            "n_cells": int(cells_per_zone[zone_id]),
        })

    n_unknowns = 3 * mesh.n_cells
    nnz_structural = len(pattern2.R)
    nnz_numerical = int(np.count_nonzero(data["J_values"]))
    results = {
        "case": "Savannah River (Hydrograd.jl example, SRH-2D mesh savana_SI.srhgeom)",
        "n_cells": int(mesh.n_cells),
        "n_faces": int(mesh.n_faces),
        "n_interior_faces": len(mesh.interior),
        "n_nodes": len(mesh.nodes),
        "cell_types": cell_types,
        "zones": zones,
        "cell_size_sqrt_area_m": [float(np.sqrt(mesh.area).min()),
                                  float(np.sqrt(mesh.area).max())],
        "boundaries": {str(k): {"type": v, "n_faces": len(mesh.boundary_faces[k])}
                       for k, v in mesh.bc_types.items()},
        "n_colours_order1": int(pattern1.n_colors),
        "n_colours_order2": int(pattern2.n_colors),
        "n_jvps_order2": int(3 * pattern2.n_colors),
        "n_unknowns": n_unknowns,
        "nnz_structural_order1": len(pattern1.R),
        "nnz_structural_order2": nnz_structural,
        "nnz_numerical_order2_at_srh2d_state": nnz_numerical,
        "nnz_per_row_order2": nnz_structural / n_unknowns,
        "density_order2": nnz_structural / n_unknowns ** 2,
        "n_inlets": data["n_inlets"],
        "nnz_low_rank_inlet_term": len(data["lr_rows"]),
        "nnz_low_rank_outside_local_pattern": int(data["n_extra"]),
        "note": ("Jacobian evaluated at the SRH-2D steady state with the default scheme "
                 "hr_hllc, second-order MUSCL (Venkatakrishnan limiter). dR/dU = J_local + "
                 "A C^T with A C^T the inlet conveyance coupling (rank n_inlets)."),
    }
    write_results("mesh_jacobian", results)
    return results


def _spy(ax: plt.Axes, data: dict[str, Any], marker_local: float, marker_inlet: float,
         label_local: str = "", label_inlet: str = "") -> None:
    """Scatter the nonzeros of J_local (black) and of the inlet term A C^T (red).

    Args:
        ax: the axes.
        data: output of ``compute`` (scalar row and column indices of the nonzeros).
        marker_local: marker size of the J_local nonzeros (pt).
        marker_inlet: marker size of the A C^T nonzeros (pt).
        label_local: legend label of J_local ("" for none).
        label_inlet: legend label of A C^T ("" for none).
    """
    ax.plot(data["J_cols"], data["J_rows"], linestyle="none", marker="s",
            markersize=marker_local, markeredgewidth=0.0, color="k", rasterized=True,
            label=label_local or None)
    ax.plot(data["lr_cols"], data["lr_rows"], linestyle="none", marker="s",
            markersize=marker_inlet, markeredgewidth=0.0, color="tab:red", rasterized=True,
            label=label_inlet or None)


def plot(data: dict[str, Any], results: dict[str, Any]) -> Path:
    """Draw the four panels and save ``figures/fig02_mesh_jacobian``.

    Args:
        data: output of ``compute``.
        results: output of ``record``.

    Returns:
        Path of the PDF file.
    """
    apply_style()
    mesh = data["mesh"]
    polygons = mesh_polygons(mesh)
    n = mesh.n_cells

    # Layout: the two maps (a), (b) on top; the full spy plot (c) and its zoom (d) below.
    fig = plt.figure(figsize=(FULL_WIDTH, 5.2))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.45], hspace=0.34)
    top = outer[0].subgridspec(1, 2, wspace=0.08)
    bottom = outer[1].subgridspec(1, 2, wspace=0.30)
    ax_zone = fig.add_subplot(top[0, 0])
    ax_colour = fig.add_subplot(top[0, 1])
    ax_spy = fig.add_subplot(bottom[0, 0])
    ax_zoom = fig.add_subplot(bottom[0, 1])

    # ------------------------------------------------------------------
    # (a) Material zones on the mesh, with an inset of the mixed-cell region
    # ------------------------------------------------------------------
    zone_ids = np.unique(data["mat_id"])
    zone_cmap = ListedColormap(ZONE_COLOURS[:len(data["n_zone"])])
    zone_norm = BoundaryNorm(np.arange(len(data["n_zone"]) + 1) - 0.5, zone_cmap.N)
    poly_field(ax_zone, polygons, data["mat_id"], edge_width=0.1, cmap=zone_cmap, norm=zone_norm)

    # Legend: one patch per zone with its Manning's n (s m^-1/3), in the empty lower left.
    handles = []
    for zone_id in zone_ids:
        n_value = data["n_zone"][zone_id]
        label = rf"{zone_id}: ${n_value:.3f}$"
        handles.append(Patch(facecolor=zone_cmap(zone_id), edgecolor="0.25", linewidth=0.3,
                             label=label))
    ax_zone.legend(handles=handles, loc="lower left", fontsize=7, handlelength=0.9,
                   borderaxespad=0.2, labelspacing=0.2, handletextpad=0.4,
                   title=r"zone: $n$", title_fontsize=7)

    # Inset (upper middle, empty part of the frame): the triangle/quadrilateral transition.
    inset = ax_zone.inset_axes([400.0, 255.0, 230.0, 270.0], transform=ax_zone.transData)
    poly_field(inset, polygons, data["mat_id"], edge_width=0.4, cmap=zone_cmap, norm=zone_norm)
    inset.set_xlim(*ZOOM_X)
    inset.set_ylim(*ZOOM_Y)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.tick_params(top=False, right=False)
    for spine in inset.spines.values():
        spine.set_linewidth(0.6)
    ax_zone.indicate_inset_zoom(inset, edgecolor="k", linewidth=0.5)

    ax_zone.set_xlim(-20.0, 1010.0)
    ax_zone.set_ylim(-30.0, 530.0)
    ax_zone.set_xlabel(r"$x$ (m)")
    ax_zone.set_ylabel(r"$y$ (m)")
    panel_label(ax_zone, "(a)", x=-0.16)

    # ------------------------------------------------------------------
    # (b) Distance-2 colouring (second order)
    # ------------------------------------------------------------------
    colours = data["pattern2"].color
    n_colours = int(results["n_colours_order2"])
    palette = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    colour_cmap = ListedColormap(palette[:n_colours])
    colour_norm = BoundaryNorm(np.arange(n_colours + 1) - 0.5, n_colours)
    poly_field(ax_colour, polygons, colours, edge_width=0.1, cmap=colour_cmap, norm=colour_norm)
    colour_text = rf"{n_colours} colours (first order: {results['n_colours_order1']})"
    ax_colour.text(0.02, 0.03, colour_text, transform=ax_colour.transAxes, va="bottom",
                   fontsize=8)
    ax_colour.set_xlim(-20.0, 1010.0)
    ax_colour.set_ylim(-30.0, 530.0)
    ax_colour.set_xlabel(r"$x$ (m)")
    ax_colour.tick_params(labelleft=False)
    panel_label(ax_colour, "(b)", x=-0.04)

    # ------------------------------------------------------------------
    # (c) Sparsity pattern of dR/dU (J_local and the inlet term A C^T)
    # ------------------------------------------------------------------
    nnz_text = f"{results['nnz_structural_order2']:,}".replace(",", r"\,")
    _spy(ax_spy, data, 0.12, 0.8)

    # Block boundaries and labels of the state components.
    for k in (1, 2):
        ax_spy.axhline(k * n, color="0.6", linewidth=0.5)
        ax_spy.axvline(k * n, color="0.6", linewidth=0.5)
    block_labels = [r"$\xi$", r"$q_x$", r"$q_y$"]
    centres = [(k + 0.5) * n for k in range(3)]
    ax_spy.set_xticks(centres, block_labels)
    ax_spy.set_yticks(centres, block_labels)
    ax_spy.tick_params(length=0, top=False, right=False)
    ax_spy.set_xlim(-0.5, 3 * n - 0.5)
    ax_spy.set_ylim(3 * n - 0.5, -0.5)
    ax_spy.set_aspect("equal")
    ax_spy.set_xlabel(r"state $U$ (column)")
    ax_spy.set_ylabel(r"residual $R$ (row)")
    ax_spy.set_title(rf"$3N = {3 * n}$ unknowns, {nnz_text} nonzeros", fontsize=9)
    panel_label(ax_spy, "(c)", x=-0.20)

    # Zoom window around the inlet cells in the xi-xi block, shown in (c) by a box.
    inlet_cells = np.unique(data["lr_cols"])
    centre = 0.5 * (inlet_cells.min() + inlet_cells.max())
    window = (centre - 60.0, centre + 60.0)
    ax_spy.add_patch(Rectangle((window[0], window[0]), window[1] - window[0],
                               window[1] - window[0], fill=False, edgecolor="tab:blue",
                               linewidth=0.8))

    # ------------------------------------------------------------------
    # (d) Zoom: two-ring stencil and the dense inlet block of A C^T
    # ------------------------------------------------------------------
    _spy(ax_zoom, data, 1.6, 1.6, label_local=r"$J_{\mathrm{local}}$ (two-ring stencil)",
         label_inlet=r"inlet term $A C^{\mathsf{T}}$")
    ax_zoom.set_xlim(window[0], window[1])
    ax_zoom.set_ylim(window[1], window[0])
    ax_zoom.set_aspect("equal")
    ax_zoom.set_xlabel(r"column index $j$ ($\xi$ block)")
    ax_zoom.set_ylabel(r"row index $i$ ($\xi$ block)")
    for spine in ax_zoom.spines.values():
        spine.set_edgecolor("tab:blue")
    ax_zoom.legend(loc="lower left", fontsize=7, markerscale=2.5, handletextpad=0.3,
                   borderaxespad=0.3, frameon=True, framealpha=0.9, edgecolor="none")
    panel_label(ax_zoom, "(d)", x=-0.22)

    return save(fig, "fig02_mesh_jacobian")


def main() -> None:
    """Compute the mesh and Jacobian data, record the numbers and draw the figure."""
    data = compute()
    results = record(data)
    pdf = plot(data, results)
    print(f"wrote {pdf}")
    print({k: results[k] for k in ("n_cells", "n_faces", "cell_types", "n_colours_order1",
                                   "n_colours_order2", "nnz_structural_order2",
                                   "nnz_low_rank_inlet_term",
                                   "nnz_low_rank_outside_local_pattern")})


if __name__ == "__main__":
    main()
