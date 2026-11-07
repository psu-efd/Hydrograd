"""Scattered surface velocities (LSPIV, PIV, ADV, CFD probes) mapped onto the mesh cells.

Large-scale particle image velocimetry (LSPIV) and similar methods give velocities at
scattered points (a regular image grid, in general not aligned with the computational
mesh). This module reads such a file, maps the points to model coordinates, converts
surface velocities to depth averages, and averages them over the mesh cells, so that they
can enter a calibration loss (``apps.velocity_data``).

File format
-----------
A text file of columns separated by commas, semicolons, tabs or spaces (in a
space-separated file the column names must not contain spaces). Lines starting
with ``#`` or ``%`` are comments. The column names are taken from the last non-numeric
line before the first data row, so the ASCII export of PIVlab (``x [m],y [m],u [m/s],
v [m/s],...`` after a few title lines) and plain CSV files are both read. Names are
compared without case and without units in brackets: ``x [m]`` matches ``x``. The default
columns are ``x``, ``y``, ``u`` and ``v``; others can be named in ``columns``. Rows with a
missing (NaN) velocity are dropped (masked PIV vectors).

Coordinates and velocities
--------------------------
Points are mapped from the measurement frame (x', y') to the model frame (x, y) by

    (x', y') <- (x', -y')                    if flip_y (image rows point down)
    (x, y)    = origin + scale R(theta) (x', y')
    (u, v)    = velocity_scale R(theta) (u', v')

with R(theta) the rotation by ``rotation_deg`` counter-clockwise. ``scale`` converts the
measurement length unit to model length (e.g. pixels to m), ``velocity_scale`` the
velocity unit (e.g. pixels/frame to m/s); flip_y also flips v'.

Surface to depth average
------------------------
LSPIV measures the surface velocity u_s; the SWEs carry the depth average U. They are
related by the velocity index alpha = U / u_s (``surface_factor``), about 0.85 for a
logarithmic profile in a wide channel; use 1 for data that are already depth averages
(ADV profiles, depth-averaged CFD). alpha is best measured with a few velocity profiles
in the same flow (it varies near vegetation and walls).

Averaging onto cells
--------------------
Each point is assigned to the mesh cell that contains it (points outside the mesh are
dropped). For a cell with N points the observation is the mean of their velocities, with
the uncertainty

    sigma_cell = sqrt(sigma_point^2 / N + sigma_floor^2)          (m/s)

where sigma_point is the uncertainty of a single vector and sigma_floor a floor that
represents the error of comparing a cell average with a few points (and the uncertainty
of alpha). Cells with fewer than ``min_points`` points are not used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from hydrograd.mesh import Mesh

__all__ = [
    "CellVelocities",
    "PointVelocities",
    "bin_to_cells",
    "locate_cells",
    "read_velocity_points",
    "transform_points",
]

# Number of nearest cell centroids searched for the cell that contains a point.
_CANDIDATE_CELLS = 12


@dataclass(frozen=True)
class PointVelocities:
    """Velocities at scattered points.

    Attributes:
        x, y: point coordinates (length unit of the file or the model), shape [n_points].
        u, v: velocity components (velocity unit of the file or m/s), shape [n_points].
    """

    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray


@dataclass(frozen=True)
class CellVelocities:
    """Velocities averaged over the mesh cells that contain observation points.

    Attributes:
        cells: 0-based cell indices with observations, shape [n_obs].
        u, v: mean depth-averaged velocity components (m/s), shape [n_obs].
        sigma: uncertainty of each cell mean (m/s), shape [n_obs].
        count: number of points in each cell, shape [n_obs].
        n_points_used: points inside the mesh.
        n_points_total: points read (after dropping NaN rows).
    """

    cells: np.ndarray
    u: np.ndarray
    v: np.ndarray
    sigma: np.ndarray
    count: np.ndarray
    n_points_used: int
    n_points_total: int


def _normalize_name(name: str) -> str:
    """Column name without units, spaces and case: ``"U [m/s]"`` -> ``"u"``.

    Args:
        name: raw column name.

    Returns:
        The normalized name.
    """
    without_units = re.sub(r"[\[\(].*?[\]\)]", "", name)
    return without_units.strip().lower().replace(" ", "_")


def _split(line: str) -> list[str]:
    """Split a line at commas, else semicolons, else tabs, else whitespace.

    Args:
        line: one line of the file.

    Returns:
        The fields, stripped, without empty fields.
    """
    for delimiter in (",", ";", "\t"):
        if delimiter in line:
            fields = line.split(delimiter)
            return [field.strip() for field in fields if field.strip()]
    return line.split()


def _is_numeric_row(fields: list[str]) -> bool:
    """Whether every field of a row parses as a float (NaN allowed).

    Args:
        fields: fields of one line.

    Returns:
        True for a data row.
    """
    if not fields:
        return False
    try:
        for field in fields:
            float(field)
    except ValueError:
        return False
    return True


def read_velocity_points(
    path: Path | str, columns: dict[str, str] | None = None,
) -> PointVelocities:
    """Read scattered velocities from a CSV or PIVlab ASCII file.

    Args:
        path: the text file (see the module docstring for the format).
        columns: optional map from ``x``, ``y``, ``u``, ``v`` to the column names of the
            file (compared without case and units), e.g. ``{"u": "vx", "v": "vy"}``.

    Returns:
        The points; rows with a NaN coordinate or velocity are dropped.

    Raises:
        ValueError: if the header or a required column is missing, or if there are no data.
    """
    lines = Path(path).read_text().splitlines()

    # Header: the last non-comment, non-numeric line before the first data row.
    header: list[str] | None = None
    rows: list[list[float]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped[0] in "#%":
            continue
        fields = _split(stripped)
        if _is_numeric_row(fields):
            rows.append([float(field) for field in fields])
        elif not rows:
            header = fields

    if header is None:
        raise ValueError(f"{path}: no header line with column names")
    if not rows:
        raise ValueError(f"{path}: no data rows")

    # Column positions of x, y, u and v.
    names = [_normalize_name(name) for name in header]
    wanted = {"x": "x", "y": "y", "u": "u", "v": "v"}
    if columns:
        wanted.update({key: _normalize_name(name) for key, name in columns.items()})
    position = {}
    for key, name in wanted.items():
        if name not in names:
            raise ValueError(f"{path}: column {name!r} not found in {header}")
        position[key] = names.index(name)

    # Data as an array; rows shorter than the header are padded with NaN.
    width = len(names)
    data = np.full((len(rows), width), np.nan)
    for k, row in enumerate(rows):
        data[k, : min(width, len(row))] = row[:width]

    # Drop masked vectors and incomplete rows.
    selected = data[:, [position["x"], position["y"], position["u"], position["v"]]]
    valid = np.all(np.isfinite(selected), axis=1)
    selected = selected[valid]

    return PointVelocities(
        x=selected[:, 0], y=selected[:, 1], u=selected[:, 2], v=selected[:, 3],
    )


def transform_points(
    points: PointVelocities,
    origin: tuple[float, float] = (0.0, 0.0),
    rotation_deg: float = 0.0,
    scale: float = 1.0,
    velocity_scale: float = 1.0,
    flip_y: bool = False,
) -> PointVelocities:
    """Map points and velocities from the measurement frame to the model frame.

    Args:
        points: points in the measurement frame.
        origin: model coordinates of the measurement origin (m).
        rotation_deg: counter-clockwise rotation of the measurement axes (degrees).
        scale: model length per measurement length unit (e.g. m per pixel).
        velocity_scale: model velocity per measurement velocity unit.
        flip_y: negate y and v first (image coordinates with rows pointing down).

    Returns:
        The points in model coordinates (m) with velocities in model units (m/s).
    """
    x = points.x
    y = points.y
    u = points.u
    v = points.v

    # Image axes: rows point down.
    if flip_y:
        y = -y
        v = -v

    # Rotation matrix entries.
    angle = np.deg2rad(rotation_deg)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    # Positions: rotate, scale and shift.
    x_model = origin[0] + scale * (cos_angle * x - sin_angle * y)
    y_model = origin[1] + scale * (sin_angle * x + cos_angle * y)

    # Velocities: rotate and scale.
    u_model = velocity_scale * (cos_angle * u - sin_angle * v)
    v_model = velocity_scale * (sin_angle * u + cos_angle * v)

    return PointVelocities(x=x_model, y=y_model, u=u_model, v=v_model)


def _inside_polygon(px: np.ndarray, py: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Crossing-number test of points against one polygon (boundary counts as inside).

    Args:
        px, py: point coordinates (m), shape [n].
        polygon: polygon vertices (m), shape [n_vertices, 2].

    Returns:
        Boolean array, shape [n].
    """
    inside = np.zeros(px.shape, dtype=bool)
    n_vertices = len(polygon)

    # Toggle for every edge that a horizontal ray to the right of the point crosses.
    for k in range(n_vertices):
        x1, y1 = polygon[k]
        x2, y2 = polygon[(k + 1) % n_vertices]
        spans = (y1 > py) != (y2 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        inside ^= spans & (px < x_cross)

    return inside


def locate_cells(mesh: Mesh, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cell that contains each point, or -1 outside the mesh.

    Args:
        mesh: the mesh (cell polygons and centroids).
        x, y: point coordinates (m), shape [n_points].

    Returns:
        0-based cell index per point (-1 if no cell contains it), shape [n_points].
    """
    cell = np.full(x.shape, -1, dtype=int)
    if x.size == 0:
        return cell

    # Candidate cells: the nearest centroids of each point.
    n_candidates = min(_CANDIDATE_CELLS, mesh.n_cells)
    tree = cKDTree(mesh.centroid)
    _, candidates = tree.query(np.c_[x, y], k=n_candidates)
    candidates = candidates.reshape(len(x), n_candidates)

    # Test the candidates from the nearest outwards; keep the first cell that contains
    # the point.
    for rank in range(n_candidates):
        unresolved = np.flatnonzero(cell < 0)
        if unresolved.size == 0:
            break
        candidate_cells = candidates[unresolved, rank]
        for candidate in np.unique(candidate_cells):
            points = unresolved[candidate_cells == candidate]
            polygon = mesh.nodes[mesh.cell_nodes[candidate], :2]
            hit = _inside_polygon(x[points], y[points], polygon)
            cell[points[hit]] = candidate

    return cell


def bin_to_cells(
    mesh: Mesh,
    points: PointVelocities,
    surface_factor: float = 1.0,
    sigma_point: float = 0.02,
    sigma_floor: float = 0.0,
    min_points: int = 1,
) -> CellVelocities:
    """Average point velocities (model frame) over the cells that contain them.

    Args:
        mesh: the mesh.
        points: points in model coordinates (m) with velocities (m/s).
        surface_factor: velocity index alpha = U / u_s (-); 1 for depth averages.
        sigma_point: uncertainty of one velocity vector (m/s).
        sigma_floor: floor of the cell-mean uncertainty (m/s).
        min_points: minimum number of points for a cell to be used.

    Returns:
        The cell observations (depth-averaged velocities, m/s).
    """
    cell = locate_cells(mesh, points.x, points.y)
    inside = cell >= 0
    cell_inside = cell[inside]

    # Depth-averaged velocities of the points inside the mesh.
    u = surface_factor * points.u[inside]
    v = surface_factor * points.v[inside]

    # Sums per cell.
    count = np.bincount(cell_inside, minlength=mesh.n_cells)
    sum_u = np.bincount(cell_inside, weights=u, minlength=mesh.n_cells)
    sum_v = np.bincount(cell_inside, weights=v, minlength=mesh.n_cells)

    # Cells with enough points; their means and uncertainties.
    observed = np.flatnonzero(count >= max(min_points, 1))
    n_obs = count[observed]
    mean_u = sum_u[observed] / n_obs
    mean_v = sum_v[observed] / n_obs
    sigma = np.sqrt(sigma_point**2 / n_obs + sigma_floor**2)

    return CellVelocities(
        cells=observed,
        u=mean_u,
        v=mean_v,
        sigma=sigma,
        count=n_obs,
        n_points_used=int(inside.sum()),
        n_points_total=int(points.x.size),
    )
