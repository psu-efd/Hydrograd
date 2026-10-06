"""Unstructured polygonal finite-volume mesh built from an SRH-2D ``.srhgeom``.

Numbering conventions
---------------------
Mirrors ``src/meshes/mesh_2D.jl`` of Hydrograd.jl:

- faces are numbered by first encounter while looping over cells and their local
  edges (edge i joins local nodes i and i+1, and the last edge wraps to node 0);
- the *owner* of a face is the first cell that encounters it (the lower cell ID); its
  outward normal ``(dy, -dx)/L`` (counter-clockwise cells) is the face normal, so
  ``normal`` points owner -> neighbour;
- boundary faces are grouped per boundary (nodestring) ID in nodestring order;
  boundary faces not on any BC nodestring form a default ``WALL`` boundary.

Geometry
--------
For a cell with counter-clockwise nodes (x_k, y_k), k = 0..K-1 (index K wraps to 0):

    c_k      = x_k y_{k+1} - x_{k+1} y_k
    A        = (1/2) sum_k c_k                                    shoelace area
    x_c      = sum_k (x_k + x_{k+1}) c_k / (6 A)                  centroid
    y_c      = sum_k (y_k + y_{k+1}) c_k / (6 A)

For a face traversed by its owner from node a to node b, with d = (dx, dy) = p_b - p_a:

    L        = |d|                                                face length
    n        = (dy, -dx) / L                                      unit normal

All arrays are NumPy and 0-based. Lengths are in the units of the ``.srhgeom``
(m for SI cases).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from hydrograd.io.srh2d import SRHGeom

logger = logging.getLogger(__name__)

__all__ = ["Mesh", "build_mesh"]


@dataclass(frozen=True)
class Mesh:
    """Face-based unstructured mesh (0-based indices).

    Attributes:
        nodes: node coordinates (x, y, z_bed) (m), shape [n_nodes, 3].
        cell_nodes: node indices of each cell (list of arrays, counter-clockwise).
        area: cell areas (m^2), shape [n_cells].
        centroid: cell centroids (m), shape [n_cells, 2].
        face_nodes: (min, max) node index of each face, shape [n_faces, 2].
        owner: first cell of each face, shape [n_faces].
        neigh: second cell, or -1 for boundary faces, shape [n_faces].
        normal: unit normal pointing owner -> neighbour (outward on boundaries), [n_faces, 2].
        length: face lengths (m), shape [n_faces].
        midpoint: face midpoints (m), shape [n_faces, 2].
        boundary_faces: boundary ID -> face indices in nodestring order.
        bc_types: boundary ID -> BC type (``INLET-Q``, ``EXIT-H``, ``WALL``, ``SYMM``).
    """

    nodes: np.ndarray
    cell_nodes: list[np.ndarray]
    area: np.ndarray
    centroid: np.ndarray
    face_nodes: np.ndarray
    owner: np.ndarray
    neigh: np.ndarray
    normal: np.ndarray
    length: np.ndarray
    midpoint: np.ndarray
    boundary_faces: dict[int, np.ndarray]
    bc_types: dict[int, str]

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return len(self.area)

    @property
    def n_faces(self) -> int:
        """Number of faces (interior and boundary)."""
        return len(self.length)

    @property
    def interior(self) -> np.ndarray:
        """Indices of interior faces, shape [n_interior_faces]."""
        return np.flatnonzero(self.neigh >= 0)

    def boundary_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """All boundary faces and their boundary IDs, concatenated in sorted-ID order.

        Returns:
            (faces, bids), each of shape [n_boundary_faces]: face indices, and the
            boundary ID of each face.
        """
        ids = sorted(self.boundary_faces)

        faces = np.concatenate([self.boundary_faces[i] for i in ids])
        bids = np.concatenate([np.full(len(self.boundary_faces[i]), i) for i in ids])

        return faces, bids

    def min_dx(self) -> float:
        """Smallest 2A/perimeter over cells (m), a CFL length scale.

        Returns:
            min over cells of 2 A / P (m), where P is the cell perimeter (m).
        """
        # Perimeter of each cell: every face adds its length to the owner, and interior
        # faces also to the neighbour.
        perim = np.zeros(self.n_cells)
        np.add.at(perim, self.owner, self.length)
        inner = self.neigh >= 0
        np.add.at(perim, self.neigh[inner], self.length[inner])

        return float(np.min(2.0 * self.area / perim))


def _padded_cells(cells: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Cells as a padded [n_cells, k_max] node array (pad = repeat of the last node) and counts.

    Repeating the last node makes the padded edges zero-length, so they do not change
    the area or the centroid.

    Args:
        cells: node indices of each cell (list of arrays of varying length).

    Returns:
        (pad, counts): padded node indices, shape [n_cells, k_max], dtype int64; and the
        number of nodes of each cell, shape [n_cells], dtype int64.
    """
    counts = np.fromiter((len(c) for c in cells), dtype=np.int64, count=len(cells))
    k_max = int(counts.max())
    pad = np.empty((len(cells), k_max), dtype=np.int64)

    # One vectorized block per polygon size (triangles, quads, ...).
    for k in np.unique(counts):
        idx = np.flatnonzero(counts == k)
        block = np.stack([cells[i] for i in idx])
        pad[idx, :k] = block
        pad[idx, k:] = block[:, -1:]

    return pad, counts


def build_mesh(geom: SRHGeom, bc_types: dict[int, str]) -> Mesh:
    """Build the face connectivity and geometry from a parsed ``.srhgeom`` (vectorized).

    Args:
        geom: parsed mesh.
        bc_types: boundary (nodestring) ID -> BC type, from the ``.srhhydro``
            (monitoring lines excluded).

    Returns:
        Mesh with a default ``WALL`` boundary added if needed.

    Raises:
        ValueError: on clockwise cells or nodestrings that do not follow mesh edges.
    """
    nodes = geom.nodes
    n_cells = len(geom.cells)
    n_nodes = len(nodes)

    # ------------------------------------------------------------------
    # Cell edges as padded arrays: edge (cell, k) runs from n0[cell, k] to n1[cell, k].
    # ------------------------------------------------------------------
    pad, counts = _padded_cells(geom.cells)
    k_max = pad.shape[1]
    local = np.arange(k_max)

    # Real edges (not padding), shape [n_cells, k_max].
    valid = local[None, :] < counts[:, None]

    # Local index of the next node; the last real node wraps to node 0.
    nxt = np.where(local[None, :] + 1 < counts[:, None], local[None, :] + 1, 0)

    n0 = pad
    n1 = np.take_along_axis(pad, nxt, axis=1)

    # ------------------------------------------------------------------
    # Shoelace area and centroid (padded edges have zero length -> no contribution).
    # ------------------------------------------------------------------
    x0 = nodes[n0, 0]
    y0 = nodes[n0, 1]
    x1 = nodes[n1, 0]
    y1 = nodes[n1, 1]

    cross = np.where(valid, x0 * y1 - x1 * y0, 0.0)
    area = 0.5 * cross.sum(axis=1)

    # Counter-clockwise cells have positive signed area.
    bad = np.flatnonzero(area <= 0.0)
    if len(bad):
        raise ValueError(
            f"cell {bad[0] + 1} is clockwise or degenerate (signed area {area[bad[0]]:.3e})"
        )

    centroid_x_sum = ((x0 + x1) * cross).sum(1)
    centroid_y_sum = ((y0 + y1) * cross).sum(1)
    centroid = np.stack([centroid_x_sum, centroid_y_sum], axis=1) / (6.0 * area[:, None])

    # ------------------------------------------------------------------
    # Faces: unique undirected edges, numbered by first encounter (cell order, local order).
    # ------------------------------------------------------------------
    # Flattened list of real edges (row-major: cell by cell, local edge by local edge).
    e_cell = np.repeat(np.arange(n_cells), counts)
    e0 = n0[valid]
    e1 = n1[valid]

    # Undirected edge key (min node, max node) packed into one integer.
    key = np.minimum(e0, e1) * n_nodes + np.maximum(e0, e1)

    # np.unique sorts the keys; renumber them by the position of their first edge.
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))

    # Face index of each flattened cell edge.
    face_of_edge = rank[inv]
    n_faces = len(order)

    # ------------------------------------------------------------------
    # Owner (first encounter) and neighbour (second encounter) of each face.
    # ------------------------------------------------------------------
    first_edge = first[order]
    owner_a = e_cell[first_edge]

    neigh_a = np.full(n_faces, -1, dtype=np.int64)
    second = np.ones(len(key), dtype=bool)
    second[first_edge] = False
    neigh_a[face_of_edge[second]] = e_cell[second]

    if np.bincount(face_of_edge, minlength=n_faces).max() > 2:
        raise ValueError("an edge is shared by more than two cells")

    # ------------------------------------------------------------------
    # Face geometry from the owner's traversal direction d = p_b - p_a.
    # ------------------------------------------------------------------
    d = nodes[e1[first_edge], :2] - nodes[e0[first_edge], :2]
    length = np.hypot(d[:, 0], d[:, 1])

    # (dy, -dx) / L is outward for the owner (counter-clockwise cells).
    normal = np.stack([d[:, 1], -d[:, 0]], axis=1) / length[:, None]

    # Face nodes as (min, max) node index, and the two end points.
    face_min_node = np.minimum(e0, e1)[first_edge]
    face_max_node = np.maximum(e0, e1)[first_edge]
    fn = np.stack([face_min_node, face_max_node], axis=1)
    p0 = nodes[fn[:, 0], :2]
    p1 = nodes[fn[:, 1], :2]

    # Lookup (min node, max node) -> face index, for boundary faces only.
    edge_id: dict[tuple[int, int], int] = {}
    for f, (a, b) in enumerate(fn):
        if neigh_a[f] < 0:
            edge_id[(int(a), int(b))] = f

    # ------------------------------------------------------------------
    # Boundary faces per BC nodestring, in nodestring order.
    # ------------------------------------------------------------------
    boundary_faces: dict[int, np.ndarray] = {}
    used = np.zeros(len(fn), dtype=bool)

    for bid in sorted(bc_types):
        if bid not in geom.nodestrings:
            raise ValueError(f"BC {bid} has no NodeString in the .srhgeom")

        ns = geom.nodestrings[bid]
        faces = []

        # Each pair of consecutive nodestring nodes must be a boundary edge.
        for a_, b_ in zip(ns[:-1], ns[1:]):
            f = edge_id.get((min(a_, b_), max(a_, b_)))
            if f is None:
                raise ValueError(
                    f"NodeString {bid}: nodes {a_ + 1}-{b_ + 1} are not a boundary edge"
                )
            faces.append(f)
            used[f] = True

        boundary_faces[bid] = np.asarray(faces, dtype=np.int64)

    # A face may belong to one BC nodestring only.
    if boundary_faces:
        all_b = np.concatenate([v for v in boundary_faces.values()])
    else:
        all_b = np.zeros(0, int)
    if len(np.unique(all_b)) != len(all_b):
        raise ValueError("a boundary face belongs to more than one BC nodestring")

    # Boundary faces on no BC nodestring form a default wall with the next free ID.
    bc_all = dict(bc_types)
    leftover = np.flatnonzero((neigh_a < 0) & ~used)
    if len(leftover):
        wall_id = max(bc_all, default=0) + 1
        boundary_faces[wall_id] = leftover
        bc_all[wall_id] = "WALL"
        logger.info("Default wall boundary %d: %d faces", wall_id, len(leftover))

    midpoint = 0.5 * (p0 + p1)

    return Mesh(
        nodes,
        list(geom.cells),
        area,
        centroid,
        fn,
        owner_a,
        neigh_a,
        normal,
        length,
        midpoint,
        boundary_faces,
        bc_all,
    )
