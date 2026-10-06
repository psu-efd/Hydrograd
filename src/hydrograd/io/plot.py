"""Quick-look plots of cell fields on unstructured meshes (matplotlib).

Cells are drawn as filled polygons, so any SRH-2D mesh works; large meshes are
rasterized. For interactive inspection, load the ``.vtk`` output in ParaView.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from hydrograd.mesh import Mesh

__all__ = ["plot_cell_fields"]


def _polygons(mesh: Mesh, sel: np.ndarray) -> list:
    xy = mesh.nodes[:, :2]
    return [xy[mesh.cell_nodes[i]] for i in sel]


def plot_cell_fields(mesh: Mesh, fields: Mapping[str, tuple[np.ndarray, str]], path: str,
                     xlim: Sequence[float] | None = None, ylim: Sequence[float] | None = None,
                     title: str = "", quiver: tuple[np.ndarray, np.ndarray] | None = None,
                     panel_height: float = 2.2, width: float = 11.0, cmaps: Mapping[str, str] | None = None) -> None:
    """Stacked panels, one per field, saved to ``path`` (PNG).

    Args:
        mesh: the mesh.
        fields: label -> (per-cell values, colorbar label with units).
        path: output image file.
        xlim, ylim: optional zoom window (m); only cells inside are drawn.
        title: figure title.
        quiver: optional (u, v) per cell (m/s), drawn as sparse arrows on the last panel.
        panel_height, width: figure size (in).
        cmaps: label -> matplotlib colormap name.
    """
    c = mesh.centroid
    sel = np.ones(mesh.n_cells, dtype=bool)
    if xlim is not None:
        sel &= (c[:, 0] >= xlim[0]) & (c[:, 0] <= xlim[1])
    if ylim is not None:
        sel &= (c[:, 1] >= ylim[0]) & (c[:, 1] <= ylim[1])
    idx = np.flatnonzero(sel)
    polys = _polygons(mesh, idx)
    n = len(fields)
    fig, axes = plt.subplots(n, 1, figsize=(width, panel_height * n + 0.6), squeeze=False)
    x0, x1 = (c[idx, 0].min(), c[idx, 0].max()) if xlim is None else xlim
    y0, y1 = (c[idx, 1].min(), c[idx, 1].max()) if ylim is None else ylim
    for ax, (label, (vals, unit)) in zip(axes[:, 0], fields.items()):
        pc = PolyCollection(polys, array=np.asarray(vals)[idx], cmap=(cmaps or {}).get(label, "viridis"),
                            edgecolors="none", rasterized=True)
        ax.add_collection(pc)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=10, loc="left")
        ax.set_ylabel("y (m)")
        fig.colorbar(pc, ax=ax, label=unit, fraction=0.025, pad=0.01)
    if quiver is not None:
        u, v = quiver
        stride = max(1, len(idx) // 400)
        q = idx[::stride]
        axes[-1, 0].quiver(c[q, 0], c[q, 1], np.asarray(u)[q], np.asarray(v)[q], color="w", width=0.0015)
    axes[-1, 0].set_xlabel("x (m)")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
