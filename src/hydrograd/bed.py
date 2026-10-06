"""Bed elevation interpolation and bed slope (port of ``process_bed_2D.jl``).

Definitions
-----------
- ``zb_cell``  = mean of the cell's node elevations (m);
- ``zb_face``  = mean of the two adjacent cells (interior), the cell value (boundary) (m);
- ``S0_cell``  = -grad(zb) (dimensionless), Green-Gauss from face values:

      S0 = -(1/A) sum_f zb_f n_f L_f

  with A the cell area (m^2), n_f the outward unit normal and L_f the face length (m).

The face/slope functions are written with ``jax.numpy`` so that the bed can be an
inversion parameter.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hydrograd.mesh import Mesh

__all__ = ["BedGeometry", "bed_slope", "zb_nodes_to_cells"]


def zb_nodes_to_cells(mesh: Mesh) -> np.ndarray:
    """Cell bed elevation (m) as the arithmetic mean of its node elevations.

    Args:
        mesh: the mesh (column 2 of ``mesh.nodes`` is the node bed elevation, m).

    Returns:
        Cell bed elevation (m), shape [n_cells].
    """
    return np.array([mesh.nodes[cn, 2].mean() for cn in mesh.cell_nodes])


class BedGeometry:
    """Index arrays needed to evaluate face values and Green-Gauss slopes in JAX.

    Attributes:
        n_cells: number of cells.
        fi_o: owner cell of each interior face, shape [n_interior].
        fi_r: neighbour cell of each interior face, shape [n_interior].
        fi_nL: unit normal times length of each interior face (m), shape [n_interior, 2].
        fb_c: cell of each boundary face, shape [n_boundary].
        fb_nL: outward unit normal times length of each boundary face (m),
            shape [n_boundary, 2].
        area: cell areas (m^2), shape [n_cells].
    """

    def __init__(self, mesh: Mesh):
        """Extract the interior and boundary face arrays from the mesh.

        Args:
            mesh: the mesh.
        """
        inner = mesh.neigh >= 0

        self.n_cells = mesh.n_cells

        # Interior faces: owner, neighbour, and n L (normal points owner -> neighbour).
        self.fi_o = jnp.asarray(mesh.owner[inner])
        self.fi_r = jnp.asarray(mesh.neigh[inner])
        self.fi_nL = jnp.asarray(mesh.normal[inner] * mesh.length[inner, None])

        # Boundary faces: the cell inside, and the outward n L.
        self.fb_c = jnp.asarray(mesh.owner[~inner])
        self.fb_nL = jnp.asarray(mesh.normal[~inner] * mesh.length[~inner, None])

        self.area = jnp.asarray(mesh.area)


def bed_slope(geo: BedGeometry, zb_cells: Array) -> Array:
    """Cell bed slope S0 = -grad(zb) (dimensionless), shape [n_cells, 2].

    Args:
        geo: face index arrays of the mesh.
        zb_cells: cell bed elevation (m), shape [n_cells].

    Returns:
        Bed slope (S0_x, S0_y) of each cell (dimensionless), shape [n_cells, 2].
    """
    # Interior faces: zb_f is the mean of the two cells. The flux zb_f n L leaves the
    # owner (+) and enters the neighbour (-).
    zf = 0.5 * (zb_cells[geo.fi_o] + zb_cells[geo.fi_r])
    interior_flux = zf[:, None] * geo.fi_nL
    acc = jax.ops.segment_sum(interior_flux, geo.fi_o, geo.n_cells)
    acc = acc - jax.ops.segment_sum(interior_flux, geo.fi_r, geo.n_cells)

    # Boundary faces: zb_f is the value of the cell inside.
    boundary_flux = zb_cells[geo.fb_c][:, None] * geo.fb_nL
    acc = acc + jax.ops.segment_sum(boundary_flux, geo.fb_c, geo.n_cells)

    # Green-Gauss: grad(zb) = (1/A) sum_f zb_f n_f L_f, and S0 = -grad(zb).
    return -acc / geo.area[:, None]


def bed_and_slope(geo: BedGeometry, zb_cells: Array) -> tuple[Array, Array]:
    """Convenience: (zb_cells, S0_cells).

    Args:
        geo: face index arrays of the mesh.
        zb_cells: cell bed elevation (m), shape [n_cells].

    Returns:
        (zb_cells, S0_cells): the input bed (m), shape [n_cells], and the bed slope
        (dimensionless), shape [n_cells, 2].
    """
    return zb_cells, bed_slope(geo, zb_cells)
