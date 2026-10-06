"""Bed elevation interpolation and bed slope (port of ``process_bed_2D.jl``).

* ``zb_cell``  = mean of the cell's node elevations;
* ``zb_face``  = mean of the two adjacent cells (interior), the cell value (boundary);
* ``S0_cell``  = -grad(zb), Green-Gauss from face values:
  ``S0 = -(1/A) sum_f zb_f n_f L_f`` with outward normals.

The face/slope functions are written with ``jax.numpy`` so that the bed can be
an inversion parameter.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hydrograd.mesh import Mesh

__all__ = ["BedGeometry", "bed_slope", "zb_nodes_to_cells"]


def zb_nodes_to_cells(mesh: Mesh) -> np.ndarray:
    """Cell bed elevation (m) as the arithmetic mean of its node elevations."""
    return np.array([mesh.nodes[cn, 2].mean() for cn in mesh.cell_nodes])


class BedGeometry:
    """Index arrays needed to evaluate face values and Green-Gauss slopes in JAX."""

    def __init__(self, mesh: Mesh):
        inner = mesh.neigh >= 0
        self.n_cells = mesh.n_cells
        self.fi_o = jnp.asarray(mesh.owner[inner])
        self.fi_r = jnp.asarray(mesh.neigh[inner])
        self.fi_nL = jnp.asarray(mesh.normal[inner] * mesh.length[inner, None])
        self.fb_c = jnp.asarray(mesh.owner[~inner])
        self.fb_nL = jnp.asarray(mesh.normal[~inner] * mesh.length[~inner, None])
        self.area = jnp.asarray(mesh.area)


def bed_slope(geo: BedGeometry, zb_cells: Array) -> Array:
    """Cell bed slope S0 = -grad(zb) (dimensionless), shape [n_cells, 2]."""
    zf = 0.5 * (zb_cells[geo.fi_o] + zb_cells[geo.fi_r])
    acc = jax.ops.segment_sum(zf[:, None] * geo.fi_nL, geo.fi_o, geo.n_cells)
    acc = acc - jax.ops.segment_sum(zf[:, None] * geo.fi_nL, geo.fi_r, geo.n_cells)
    acc = acc + jax.ops.segment_sum(zb_cells[geo.fb_c][:, None] * geo.fb_nL, geo.fb_c, geo.n_cells)
    return -acc / geo.area[:, None]


def bed_and_slope(geo: BedGeometry, zb_cells: Array) -> tuple[Array, Array]:
    """Convenience: (zb_cells, S0_cells)."""
    return zb_cells, bed_slope(geo, zb_cells)
