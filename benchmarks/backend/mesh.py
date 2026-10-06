"""Backend-agnostic unstructured mesh utilities (NumPy only).

Builds a face-based finite-volume connectivity from a list of polygonal
cells, either generated (jittered Delaunay channel) or read from an
SRH-2D ``.srhgeom`` file. All backends (PyTorch, JAX) consume the same
``FVMesh`` arrays so that benchmarks compare identical discretizations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import Delaunay

# Boundary tags for faces
INTERIOR = 0
WALL = 1
INLET = 2
OUTLET = 3


@dataclass(frozen=True)
class FVMesh:
    """Face-based unstructured mesh.

    Attributes:
        n_cells: number of cells.
        n_faces: number of faces (interior + boundary).
        area: cell areas (m^2), shape [n_cells].
        centroid: cell centroids (m), shape [n_cells, 2].
        owner: left cell of each face, shape [n_faces].
        neigh: right cell of each face (-1 on boundary), shape [n_faces].
        normal: unit normal pointing owner -> neigh, shape [n_faces, 2].
        length: face lengths (m), shape [n_faces].
        btag: boundary tag per face (INTERIOR/WALL/INLET/OUTLET).
        zb: bed elevation at cell centroids (m), shape [n_cells].
    """

    n_cells: int
    n_faces: int
    area: np.ndarray
    centroid: np.ndarray
    owner: np.ndarray
    neigh: np.ndarray
    normal: np.ndarray
    length: np.ndarray
    btag: np.ndarray
    zb: np.ndarray

    @property
    def min_dx(self) -> float:
        """Smallest characteristic cell size (m) used for CFL estimates.

        Returns:
            min over cells of 2 A / P (m), with A the area (m^2) and P the
            perimeter (m); for a triangle this is the inradius times two.
        """
        # Accumulate each face length into its owner and (if interior) neighbour
        perim = np.zeros(self.n_cells)
        np.add.at(perim, self.owner, self.length)
        inner = self.neigh >= 0
        np.add.at(perim, self.neigh[inner], self.length[inner])
        return float(np.min(2.0 * self.area / perim))


def build_fv_mesh(nodes: np.ndarray, cells: List[np.ndarray], zb_nodes: np.ndarray,
                  tag_fn) -> FVMesh:
    """Build face connectivity from polygon cells.

    Args:
        nodes: node coordinates (m), shape [n_nodes, 2].
        cells: list of node-index arrays (any polygon), counter-clockwise or not.
        zb_nodes: bed elevation at nodes (m), shape [n_nodes].
        tag_fn: callable(face_midpoints [nb, 2]) -> boundary tags [nb].

    Returns:
        The face-based ``FVMesh``. Cell area and centroid come from the
        shoelace formulas; cell bed elevation is the mean of its node values.
    """
    n_cells = len(cells)
    area = np.empty(n_cells)
    centroid = np.empty((n_cells, 2))
    zb = np.empty(n_cells)

    # Maps a sorted node pair to [cell, start node, cell, start node, ...]
    edge_map: Dict[Tuple[int, int], List[int]] = {}

    # ---- 1. cell geometry and edge registration ----------------------------
    for c, cn in enumerate(cells):
        # Shoelace terms x_k y_{k+1} - x_{k+1} y_k and signed area (m^2)
        xy = nodes[cn]
        x = xy[:, 0]
        y = xy[:, 1]
        xs = np.roll(x, -1)
        ys = np.roll(y, -1)
        cross = x * ys - xs * y
        a = 0.5 * cross.sum()

        # Enforce counter-clockwise ordering (positive area)
        if a < 0:
            cn = cn[::-1]
            xy = nodes[cn]
            x = xy[:, 0]
            y = xy[:, 1]
            xs = np.roll(x, -1)
            ys = np.roll(y, -1)
            cross = x * ys - xs * y
            a = 0.5 * cross.sum()

        # Area, polygon centroid and mean bed elevation
        area[c] = a
        centroid[c, 0] = ((x + xs) * cross).sum() / (6 * a)
        centroid[c, 1] = ((y + ys) * cross).sum() / (6 * a)
        zb[c] = zb_nodes[cn].mean()

        # Register each edge with this cell and the node it starts from
        for k in range(len(cn)):
            n0 = int(cn[k])
            n1 = int(cn[(k + 1) % len(cn)])
            edge_map.setdefault((min(n0, n1), max(n0, n1)), []).append(c)
            edge_map[(min(n0, n1), max(n0, n1))].append(n0)  # remember orientation

    # ---- 2. faces: owner = first cell seen, neighbour = second (or -1) ----
    owner = []
    neigh = []
    normal = []
    length = []
    mids = []
    for (a_, b_), info in edge_map.items():
        c0 = info[0]
        start0 = info[1]

        # Edge endpoints in the owner's counter-clockwise direction
        if start0 == a_:
            p0 = nodes[a_]
            p1 = nodes[b_]
        else:
            p0 = nodes[b_]
            p1 = nodes[a_]

        # Edge vector, length (m) and right-hand unit normal
        d = p1 - p0
        L = float(np.hypot(*d))
        owner.append(c0)
        neigh.append(info[2] if len(info) > 2 else -1)
        normal.append([d[1] / L, -d[0] / L])  # outward for CCW owner
        length.append(L)
        mids.append(0.5 * (p0 + p1))

    # ---- 3. arrays and boundary tags ---------------------------------------
    owner = np.asarray(owner, dtype=np.int64)
    neigh = np.asarray(neigh, dtype=np.int64)
    mids = np.asarray(mids)
    btag = np.zeros(len(owner), dtype=np.int64)
    bnd = neigh < 0
    btag[bnd] = tag_fn(mids[bnd])
    return FVMesh(n_cells, len(owner), area, centroid, owner, neigh,
                  np.asarray(normal), np.asarray(length), btag, zb)


def channel_bed(x: np.ndarray, y: np.ndarray, length: float) -> np.ndarray:
    """Sloped channel bed with a Gaussian bump in the middle (m).

    zb = S (length - x) + 0.2 exp(-((x - length/2) / (0.05 length))^2),
    with slope S = 1e-3.

    Args:
        x: streamwise coordinate (m), any shape.
        y: transverse coordinate (m), same shape (unused: bed is uniform in y).
        length: channel length (m).

    Returns:
        Bed elevation (m), same shape as x.
    """
    slope = 1.0e-3
    bump = 0.2 * np.exp(-((x - 0.5 * length) / (0.05 * length)) ** 2)
    return slope * (length - x) + bump


def make_channel_mesh(target_cells: int, length: float = 1000.0, width: float = 100.0,
                      seed: int = 0) -> FVMesh:
    """Jittered Delaunay triangulation of a rectangular channel.

    Inlet at x=0, outlet at x=length, walls on y=0 and y=width.

    Args:
        target_cells: approximate number of triangles.
        length: channel length (m).
        width: channel width (m).
        seed: seed of the node-jitter random generator.

    Returns:
        The face-based ``FVMesh`` with the ``channel_bed`` bathymetry.
    """
    rng = np.random.default_rng(seed)

    # Grid resolution: ~2 triangles per point, aspect ratio of the channel
    npts = target_cells / 2.0
    ny = max(3, int(round(np.sqrt(npts * width / length))))
    nx = max(3, int(round(npts / ny)))

    # Structured node grid (m)
    xs = np.linspace(0, length, nx + 1)
    ys = np.linspace(0, width, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    dx = length / nx
    dy = width / ny

    # Jitter interior nodes by up to a quarter cell in each direction
    interior = (X > 0) & (X < length) & (Y > 0) & (Y < width)
    X = X + interior * rng.uniform(-0.25, 0.25, X.shape) * dx
    Y = Y + interior * rng.uniform(-0.25, 0.25, Y.shape) * dy

    # Delaunay triangles and node bed elevations
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    tri = Delaunay(nodes).simplices
    zb_nodes = channel_bed(nodes[:, 0], nodes[:, 1], length)
    tol = 1e-6 * length

    def tag(m: np.ndarray) -> np.ndarray:
        """Boundary tags from face midpoints m (m), shape [nb, 2] -> [nb]."""
        t = np.full(len(m), WALL)
        t[m[:, 0] < tol] = INLET
        t[m[:, 0] > length - tol] = OUTLET
        return t

    return build_fv_mesh(nodes, [np.asarray(c) for c in tri], zb_nodes, tag)


def read_srhgeom(path: str | Path, inlet_ns: int, outlet_ns: int) -> FVMesh:
    """Read an SRH-2D .srhgeom mesh; node-strings mark inlet/outlet faces.

    Args:
        path: path to the ``.srhgeom`` file (coordinates in m).
        inlet_ns: node-string id of the inlet boundary.
        outlet_ns: node-string id of the outlet boundary.

    Returns:
        The face-based ``FVMesh``; boundary faces not on either node string
        are walls.
    """
    nodes = {}
    cells = []
    ns = {}
    cur_ns = None

    # ---- 1. parse Node / Elem / NodeString records -------------------------
    for line in Path(path).read_text().splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "Node":
            nodes[int(tok[1])] = [float(v) for v in tok[2:5]]
            cur_ns = None
        elif tok[0] == "Elem":
            cells.append(np.array([int(v) - 1 for v in tok[2:]]))
            cur_ns = None
        elif tok[0] == "NodeString":
            cur_ns = int(tok[1])
            ns[cur_ns] = [int(v) - 1 for v in tok[2:]]
        elif cur_ns is not None and tok[0].lstrip("-").isdigit():
            # Continuation line of the current node string
            ns[cur_ns] += [int(v) - 1 for v in tok]

    # ---- 2. node coordinates (sorted by id) --------------------------------
    xyz = np.array([nodes[k] for k in sorted(nodes)])
    in_set = set(ns[inlet_ns])
    out_set = set(ns[outlet_ns])
    nodes_xy = xyz[:, :2]

    # ---- 3. map face midpoints back to node strings via a midpoint lookup --
    mid_tag = {}
    for s, t in ((in_set, INLET), (out_set, OUTLET)):
        lst = ns[inlet_ns] if t == INLET else ns[outlet_ns]
        for a, b in zip(lst[:-1], lst[1:]):
            mid_tag[tuple(np.round(0.5 * (nodes_xy[a] + nodes_xy[b]), 6))] = t

    def tag(m: np.ndarray) -> np.ndarray:
        """Boundary tags from face midpoints m (m), shape [nb, 2] -> [nb]."""
        return np.array([mid_tag.get(tuple(np.round(p, 6)), WALL) for p in m])

    return build_fv_mesh(nodes_xy, cells, xyz[:, 2], tag)
