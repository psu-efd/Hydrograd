"""Scattered velocity observations (LSPIV and similar) as calibration data.

The truth-file losses of inversion and UDE (``apps.losses``) need a value in every cell.
Measured velocities cover only part of the domain, at points that do not coincide with the
cells. This module reads such data (``io.lspiv``), averages them over the cells that
contain them, and adds a velocity misfit on those cells to the loss of the inversion and
UDE modes.

Run-control block
-----------------
::

    "velocity_data": {
        "file": "lspiv.csv",          // points, relative to the case folder (io.lspiv)
        "columns": {"u": "vx"},       // optional column names (default x, y, u, v)
        "transform": {                // optional measurement frame -> model frame
            "origin": [0.0, 0.0], "rotation_deg": 0.0, "scale": 1.0,
            "velocity_scale": 1.0, "flip_y": false
        },
        "surface_factor": 0.85,       // depth average / surface velocity (1 for averages)
        "sigma_point": 0.02,          // uncertainty of one vector (m/s)
        "sigma_floor": 0.005,         // floor of a cell-mean uncertainty (m/s)
        "min_points": 1,              // minimum points per cell
        "components": ["u", "v"],     // velocity components used in the loss
        "weight": 1.0,                // factor of the term in the total loss
        "binned_file": "lspiv_cells.csv"  // optional: write the cell observations
    }

Loss term
---------
    L_vel = weight * mean over observed cells and components [ ((U_pred - U_obs) / sigma)^2 ]

with U_pred = q / (h + h_small) (m/s) as in the truth-file losses, and sigma the
uncertainty of each cell mean (``io.lspiv.bin_to_cells``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

from hydrograd.apps.setup import Case
from hydrograd.io.lspiv import bin_to_cells, read_velocity_points, transform_points
from hydrograd.physics.model import SWEModel

logger = logging.getLogger(__name__)

__all__ = ["VelocityData", "load_velocity_data", "velocity_loss"]


@dataclass(frozen=True)
class VelocityData:
    """Depth-averaged velocity observations on a subset of the cells.

    Attributes:
        cells: 0-based indices of the observed cells, shape [n_obs].
        observed: observed (u, v) (m/s), shape [n_obs, 2].
        sigma: uncertainty of each cell mean (m/s), shape [n_obs].
        mask: 1 for the components used in the loss, shape [2] (u, v).
        factor: weight of the term in the total loss.
    """

    cells: Array
    observed: Array
    sigma: Array
    mask: Array
    factor: float


def _write_binned(path: Any, data: VelocityData, case: Case, count: np.ndarray) -> None:
    """Write the cell observations as CSV (cell ID, centroid, u, v, sigma, points).

    Args:
        path: output file.
        data: the cell observations.
        case: the case (cell centroids).
        count: number of points per observed cell.
    """
    cells = np.asarray(data.cells)
    centroid = case.mesh.centroid[cells]
    observed = np.asarray(data.observed)
    sigma = np.asarray(data.sigma)

    lines = ["cell_id,x_m,y_m,u_m_s,v_m_s,sigma_m_s,n_points"]
    for k, cell in enumerate(cells):
        lines.append(
            f"{cell + 1},{centroid[k, 0]:.6f},{centroid[k, 1]:.6f},{observed[k, 0]:.6f},"
            f"{observed[k, 1]:.6f},{sigma[k]:.6f},{count[k]}"
        )
    path.write_text("\n".join(lines) + "\n")


def load_velocity_data(case: Case) -> VelocityData | None:
    """Velocity observations from the ``"velocity_data"`` block, or None if there is none.

    Args:
        case: the case (run control and mesh).

    Returns:
        The cell observations, or None.

    Raises:
        ValueError: if no observation point lies inside the mesh.
    """
    block = case.ctrl.raw.get("velocity_data")
    if not block:
        return None

    # Read the points and map them to model coordinates.
    points = read_velocity_points(case.ctrl.case_dir / block["file"], block.get("columns"))
    frame = block.get("transform", {})
    points = transform_points(
        points,
        origin=tuple(frame.get("origin", (0.0, 0.0))),
        rotation_deg=float(frame.get("rotation_deg", 0.0)),
        scale=float(frame.get("scale", 1.0)),
        velocity_scale=float(frame.get("velocity_scale", 1.0)),
        flip_y=bool(frame.get("flip_y", False)),
    )

    # Average over the cells; surface velocities become depth averages.
    binned = bin_to_cells(
        case.mesh,
        points,
        surface_factor=float(block.get("surface_factor", 1.0)),
        sigma_point=float(block.get("sigma_point", 0.02)),
        sigma_floor=float(block.get("sigma_floor", 0.0)),
        min_points=int(block.get("min_points", 1)),
    )
    if binned.cells.size == 0:
        raise ValueError(f"velocity_data: no point of {block['file']} lies inside the mesh")

    # Components used in the loss.
    components = block.get("components", ["u", "v"])
    mask = np.array([float("u" in components), float("v" in components)])
    if mask.sum() == 0:
        raise ValueError("velocity_data.components must contain 'u', 'v' or both")

    data = VelocityData(
        cells=jnp.asarray(binned.cells),
        observed=jnp.asarray(np.c_[binned.u, binned.v]),
        sigma=jnp.asarray(binned.sigma),
        mask=jnp.asarray(mask),
        factor=float(block.get("weight", 1.0)),
    )

    # Optional record of the cell observations, for checking the mapping.
    if "binned_file" in block:
        _write_binned(case.ctrl.case_dir / block["binned_file"], data, case, binned.count)

    logger.info(
        "velocity data: %d of %d points inside the mesh, %d cells observed",
        binned.n_points_used, binned.n_points_total, binned.cells.size,
    )
    return data


def velocity_loss(m: SWEModel, theta: dict[str, Any], U: Array, data: VelocityData) -> Array:
    """Weighted velocity misfit on the observed cells (dimensionless).

    Args:
        m: the model.
        theta: differentiable parameters (``zb`` if the bed is inverted).
        U: final state [xi, q_x, q_y], shape [3, n_cells].
        data: the velocity observations.

    Returns:
        weight * mean of ((U_pred - U_obs) / sigma)^2 over the observed cells and
        components.
    """
    # Depth on the observed cells, consistent with apps.losses.data_losses.
    h_still = m.wstill - theta["zb"] if "zb" in theta else m.hstill
    h = U[0, data.cells] + h_still[data.cells]
    velocity_eps = m.h_small if m.loss_velocity_eps is None else m.loss_velocity_eps

    # Predicted depth-averaged velocities (m/s).
    u = U[1, data.cells] / (h + velocity_eps)
    v = U[2, data.cells] / (h + velocity_eps)
    predicted = jnp.stack([u, v], axis=1)

    # Normalized squared misfit of the selected components.
    normalized = (predicted - data.observed) / data.sigma[:, None]
    squared = data.mask[None, :] * normalized**2
    n_terms = data.cells.shape[0] * jnp.sum(data.mask)
    return data.factor * jnp.sum(squared) / n_terms
