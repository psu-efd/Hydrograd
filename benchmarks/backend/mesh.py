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
INTERIOR, WALL, INLET, OUTLET = 0, 1, 2, 3


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
        """Smallest characteristic cell size (m) used for CFL estimates."""
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
    """
    n_cells = len(cells)
    area = np.empty(n_cells)
    centroid = np.empty((n_cells, 2))
    zb = np.empty(n_cells)
    edge_map: Dict[Tuple[int, int], List[int]] = {}
    for c, cn in enumerate(cells):
        xy = nodes[cn]
        x, y = xy[:, 0], xy[:, 1]
        xs, ys = np.roll(x, -1), np.roll(y, -1)
        cross = x * ys - xs * y
        a = 0.5 * cross.sum()
        if a < 0:  # enforce counter-clockwise ordering
            cn = cn[::-1]
            xy = nodes[cn]
            x, y = xy[:, 0], xy[:, 1]
            xs, ys = np.roll(x, -1), np.roll(y, -1)
            cross = x * ys - xs * y
            a = 0.5 * cross.sum()
        area[c] = a
        centroid[c, 0] = ((x + xs) * cross).sum() / (6 * a)
        centroid[c, 1] = ((y + ys) * cross).sum() / (6 * a)
        zb[c] = zb_nodes[cn].mean()
        for k in range(len(cn)):
            n0, n1 = int(cn[k]), int(cn[(k + 1) % len(cn)])
            edge_map.setdefault((min(n0, n1), max(n0, n1)), []).append(c)
            edge_map[(min(n0, n1), max(n0, n1))].append(n0)  # remember orientation

    owner, neigh, normal, length, mids = [], [], [], [], []
    for (a_, b_), info in edge_map.items():
        c0, start0 = info[0], info[1]
        p0, p1 = (nodes[a_], nodes[b_]) if start0 == a_ else (nodes[b_], nodes[a_])
        d = p1 - p0
        L = float(np.hypot(*d))
        owner.append(c0)
        neigh.append(info[2] if len(info) > 2 else -1)
        normal.append([d[1] / L, -d[0] / L])  # outward for CCW owner
        length.append(L)
        mids.append(0.5 * (p0 + p1))
    owner = np.asarray(owner, dtype=np.int64)
    neigh = np.asarray(neigh, dtype=np.int64)
    mids = np.asarray(mids)
    btag = np.zeros(len(owner), dtype=np.int64)
    bnd = neigh < 0
    btag[bnd] = tag_fn(mids[bnd])
    return FVMesh(n_cells, len(owner), area, centroid, owner, neigh,
                  np.asarray(normal), np.asarray(length), btag, zb)


def channel_bed(x: np.ndarray, y: np.ndarray, length: float) -> np.ndarray:
    """Sloped channel bed with a Gaussian bump in the middle (m)."""
    slope = 1.0e-3
    bump = 0.2 * np.exp(-((x - 0.5 * length) / (0.05 * length)) ** 2)
    return slope * (length - x) + bump


def make_channel_mesh(target_cells: int, length: float = 1000.0, width: float = 100.0,
                      seed: int = 0) -> FVMesh:
    """Jittered Delaunay triangulation of a rectangular channel.

    Inlet at x=0, outlet at x=length, walls on y=0 and y=width.
    """
    rng = np.random.default_rng(seed)
    # ~2 triangles per point
    npts = target_cells / 2.0
    ny = max(3, int(round(np.sqrt(npts * width / length))))
    nx = max(3, int(round(npts / ny)))
    xs, ys = np.linspace(0, length, nx + 1), np.linspace(0, width, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    dx, dy = length / nx, width / ny
    interior = (X > 0) & (X < length) & (Y > 0) & (Y < width)
    X = X + interior * rng.uniform(-0.25, 0.25, X.shape) * dx
    Y = Y + interior * rng.uniform(-0.25, 0.25, Y.shape) * dy
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    tri = Delaunay(nodes).simplices
    zb_nodes = channel_bed(nodes[:, 0], nodes[:, 1], length)
    tol = 1e-6 * length

    def tag(m: np.ndarray) -> np.ndarray:
        t = np.full(len(m), WALL)
        t[m[:, 0] < tol] = INLET
        t[m[:, 0] > length - tol] = OUTLET
        return t

    return build_fv_mesh(nodes, [np.asarray(c) for c in tri], zb_nodes, tag)


def read_srhgeom(path: str | Path, inlet_ns: int, outlet_ns: int) -> FVMesh:
    """Read an SRH-2D .srhgeom mesh; node-strings mark inlet/outlet faces."""
    nodes, cells, ns = {}, [], {}
    cur_ns = None
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
            ns[cur_ns] += [int(v) - 1 for v in tok]
    xyz = np.array([nodes[k] for k in sorted(nodes)])
    in_set, out_set = set(ns[inlet_ns]), set(ns[outlet_ns])
    nodes_xy = xyz[:, :2]
    # map face midpoints back to node strings via a midpoint lookup
    mid_tag = {}
    for s, t in ((in_set, INLET), (out_set, OUTLET)):
        lst = ns[inlet_ns] if t == INLET else ns[outlet_ns]
        for a, b in zip(lst[:-1], lst[1:]):
            mid_tag[tuple(np.round(0.5 * (nodes_xy[a] + nodes_xy[b]), 6))] = t

    def tag(m: np.ndarray) -> np.ndarray:
        return np.array([mid_tag.get(tuple(np.round(p, 6)), WALL) for p in m])

    return build_fv_mesh(nodes_xy, cells, xyz[:, 2], tag)
