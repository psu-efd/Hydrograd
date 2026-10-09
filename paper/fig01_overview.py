"""Figure 1: overview of Hydrograd v1.0 and its gradient methods (schematic).

Left: inputs (SRH-2D unstructured mesh, materials, boundary conditions, run control) and
the model parameters theta. Centre: the discrete residual R(U; theta) on the
unstructured mesh, with its closures. Right: the three solution paths and their
gradient methods with typical cost and memory. Bottom: the loss and the optimiser,
which update theta (calibration and closure learning).

Output: ``paper/figures/fig01_overview.pdf`` (and ``.png``).

Usage
-----
    python fig01_overview.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from figstyle import FULL_WIDTH, apply_style, save
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

# Box colours (edge, fill): inputs, model core, solvers, learning loop.
INPUT = ("#4C72B0", "#EAF0F8")
CORE = ("#55A868", "#EAF5EC")
SOLVER = ("#8172B2", "#F0EEF7")
LEARN = ("#C44E52", "#F9EBEB")


def box(axis: plt.Axes, x: float, y: float, width: float, height: float,
        colours: tuple[str, str], title: str, lines: list[str]) -> None:
    """Draw a rounded box with a bold title and lines of text.

    Args:
        axis: axes in unit-square coordinates.
        x, y: lower-left corner.
        width, height: box size.
        colours: (edge colour, fill colour).
        title: first line (bold).
        lines: further lines (LaTeX allowed).
    """
    patch = FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.004,rounding_size=0.015",
        linewidth=1.2, edgecolor=colours[0], facecolor=colours[1],
    )
    axis.add_patch(patch)
    axis.text(x + 0.01, y + height - 0.02, r"\textbf{" + title + "}", fontsize=9.5,
              va="top", ha="left")
    axis.text(x + 0.01, y + height - 0.085, "\n".join(lines), fontsize=8.2, va="top",
              ha="left", linespacing=1.3)


def arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float],
          colour: str = "0.3", curve: float = 0.0, label: str = "",
          label_xy: tuple[float, float] | None = None) -> None:
    """Draw an arrow with an optional label.

    Args:
        axis: axes.
        start, end: tail and head (unit-square coordinates).
        colour: line colour.
        curve: arc curvature.
        label: text next to the arrow.
        label_xy: label position (default: midpoint).
    """
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, color=colour,
                            linewidth=1.0, connectionstyle=f"arc3,rad={curve}")
    axis.add_patch(patch)
    if label:
        position = label_xy or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        axis.text(*position, label, fontsize=7, ha="center", va="center", color="0.2",
                  bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none"})


def mesh_sketch(axis: plt.Axes, x0: float, y0: float, size: float) -> None:
    """Draw a small unstructured mesh of mixed triangles and quadrilaterals.

    Args:
        axis: axes.
        x0, y0: lower-left corner of the sketch.
        size: width of the sketch (unit-square coordinates).
    """
    rng = np.random.default_rng(3)
    nodes_x = np.linspace(0, 1, 5)
    nodes_y = np.linspace(0, 1, 4)
    grid_x, grid_y = np.meshgrid(nodes_x, nodes_y)
    grid_x = grid_x + 0.06 * rng.standard_normal(grid_x.shape) * (grid_x > 0) * (grid_x < 1)
    grid_y = grid_y + 0.06 * rng.standard_normal(grid_y.shape) * (grid_y > 0) * (grid_y < 1)
    scale_y = 0.62 * size

    # Quadrilaterals on the left half, triangles on the right half.
    for i in range(len(nodes_y) - 1):
        for j in range(len(nodes_x) - 1):
            corners = [(grid_x[i, j], grid_y[i, j]), (grid_x[i, j + 1], grid_y[i, j + 1]),
                       (grid_x[i + 1, j + 1], grid_y[i + 1, j + 1]),
                       (grid_x[i + 1, j], grid_y[i + 1, j])]
            corners = [(x0 + size * cx, y0 + scale_y * cy) for cx, cy in corners]
            if j < 2:
                cells = [corners]
            else:
                cells = [[corners[0], corners[1], corners[2]], [corners[0], corners[2], corners[3]]]
            for cell in cells:
                axis.add_patch(Polygon(cell, closed=True, fill=True, facecolor="#D5E8D4",
                                       edgecolor="#2F6B3A", linewidth=0.5))


def main() -> None:
    """Draw the overview and save it."""
    apply_style()
    figure = plt.figure(figsize=(FULL_WIDTH, 0.5 * FULL_WIDTH))
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    # Inputs and parameters (left column).
    box(axis, 0.01, 0.52, 0.21, 0.44, INPUT, "Inputs",
        [r"SRH-2D unstructured mesh", r"(triangles, quadrilaterals,", r"polygons)",
         r"materials (roughness zones)", r"boundary conditions:", r"$Q(t)$, WSE$(t)$",
         r"run control (JSON)"])
    box(axis, 0.01, 0.04, 0.21, 0.42, LEARN, r"Parameters $\boldsymbol{\theta}$",
        [r"Manning's $n$ (zones, laws)", r"NN closure weights $\mathbf{w}$",
         r"bed $z_b$, inflow $Q(t)$", r"vegetation drag $C_D$",
         r"$C_t$, $C_m$, $k_{s,w}$, $\beta$"])

    # Model core and learning loop (centre column).
    box(axis, 0.27, 0.32, 0.25, 0.64, CORE, r"Residual $\mathbf{R}(\mathbf{U};\boldsymbol{\theta})$",
        [r"finite volumes on the mesh:", r"gather $\rightarrow$ face flux $\rightarrow$ add",
         r"hydrostatic reconstruction", r"+ HLLC, 1st/2nd order,", r"wetting and drying",
         r"closures: friction, NN,", r"vegetation, turbulence", r"exact sparse Jacobian",
         r"$\mathbf{J}=\partial\mathbf{R}/\partial\mathbf{U}$ (colouring)"])
    mesh_sketch(axis, 0.32, 0.345, 0.15)
    box(axis, 0.27, 0.04, 0.25, 0.22, LEARN, r"Loss $\mathcal{L}$, optimiser",
        [r"WSE, velocity, forces", r"Adam, L-BFGS"])

    # Solvers and their gradient methods (right column).
    solver_boxes = [
        (0.70, "Steady flow",
         [r"Newton: $\mathbf{R}(\mathbf{U}^*;\boldsymbol{\theta})=\mathbf{0}$",
          r"gradient: implicit function theorem,",
          r"$\mathbf{J}^{\mathrm{T}}\boldsymbol{\lambda}=\partial\mathcal{L}/\partial\mathbf{U}$"
          r" (one linear solve)"]),
        (0.37, "Unsteady flow, explicit",
         [r"SSP-RK3, point-implicit friction", r"gradient: reverse-mode AD with",
          r"checkpointing ($\approx$4--8 runs, fixed memory)"]),
        (0.04, "Unsteady flow, implicit",
         [r"$\vartheta$-scheme, Newton per time step",
          r"gradient: one adjoint solve per step", r"($\approx$2--2.6 runs, smooth loss)"]),
    ]
    for y, title, lines in solver_boxes:
        box(axis, 0.60, y, 0.39, 0.26, SOLVER, title, lines)

    # Arrows: inputs and parameters into the residual.
    arrow(axis, (0.22, 0.74), (0.27, 0.74))
    arrow(axis, (0.22, 0.36), (0.27, 0.46), curve=-0.15)

    # Residual into the three solvers.
    arrow(axis, (0.52, 0.86), (0.60, 0.86))
    arrow(axis, (0.52, 0.55), (0.60, 0.53))
    arrow(axis, (0.52, 0.36), (0.60, 0.22), curve=0.15)

    # Gradients of the three solvers collected into the loss.
    collector_x = 0.565
    for y in (0.74, 0.41, 0.08):
        axis.plot([0.60, collector_x], [y, y], color=LEARN[0], lw=1.0)
    axis.plot([collector_x, collector_x], [0.74, 0.08], color=LEARN[0], lw=1.0)
    arrow(axis, (collector_x, 0.15), (0.52, 0.15), colour=LEARN[0])
    axis.text(collector_x + 0.004, 0.30,
              r"$\displaystyle\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\boldsymbol{\theta}}$",
              fontsize=9, color=LEARN[0], ha="left", va="center")

    # Optimiser updates the parameters.
    arrow(axis, (0.27, 0.13), (0.22, 0.13), colour=LEARN[0], label=r"update",
          label_xy=(0.245, 0.09))

    save(figure, "fig01_overview")


if __name__ == "__main__":
    main()
