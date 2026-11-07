"""Observed vegetation drag forces as calibration data (extension of inversion and UDE).

Flow fields (water levels, velocities) constrain the total flow resistance well, but not
how it is distributed among neighbouring vegetation patches, nor inside a patch (as the
3D CFD pilot study of tandem vegetation patches showed). Measured forces on patches or sub-patches (force sensors,
or forces from 3D CFD) supply that information. This module adds them to the loss of the
inversion and UDE modes.

Run-control block
-----------------
::

    "force_data": {
        "file": "forces.json",      // the observations (below), relative to the case folder
        "weight": 1.0,              // factor of the force term in the total loss
        "sigma_N": 0.1,             // force uncertainty (N): the misfit unit
        "rho": 1000.0               // water density (kg/m^3)
    }

Observation file
----------------
::

    {"regions": [
        {"name": "patch1", "materials": [2], "Fx": 5.06},
        {"name": "row3",   "box": [x0, x1, y0, y1], "Fx": 0.88},
        {"name": "gauge",  "cells": [12, 13, 14], "Fx": 1.0, "Fy": 0.1}
    ]}

A region is a set of cells: the cells of the listed material IDs, the cells whose
centroid lies in an axis-aligned box (m), or a list of 1-based cell IDs. ``Fx`` (and
optionally ``Fy``) is the observed streamwise (and lateral) drag force on the region (N).

Loss term
---------
    L_force = weight * mean over observed components [ ((F_pred - F_obs) / sigma)^2 ]

with F_pred the differentiable drag force of the vegetation
(``physics.vegetation.vegetation_force``) plus that of the porous zones
(``physics.porous.porous_force``), whichever the model has.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

from hydrograd.apps.setup import Case
from hydrograd.physics.model import SWEModel
from hydrograd.physics.porous import porous_force
from hydrograd.physics.vegetation import vegetation_force

logger = logging.getLogger(__name__)

__all__ = ["ForceData", "drag_force", "force_loss", "load_force_data"]


@dataclass(frozen=True)
class ForceData:
    """Observed vegetation forces on regions of cells.

    Attributes:
        names: region names.
        weights: cell weights of every region (1 inside, 0 outside), [n_regions, n_cells].
        observed: observed force components (N), [n_regions, 2] (x, y).
        mask: 1 where a component is observed, 0 otherwise, [n_regions, 2].
        sigma: force uncertainty (N).
        rho: water density (kg/m^3).
        factor: weight of the force term in the total loss.
    """

    names: tuple[str, ...]
    weights: Array
    observed: Array
    mask: Array
    sigma: float
    rho: float
    factor: float


def _region_cells(region: dict[str, Any], case: Case) -> np.ndarray:
    """Cell indicator (0/1) of one region from its materials, box or cell list.

    Args:
        region: the region entry of the observation file.
        case: the case (material IDs and cell centroids).

    Returns:
        Indicator of shape [n_cells].
    """
    n_cells = case.model.n_cells

    if "materials" in region:
        material = np.asarray(case.model.mat_id)
        inside = np.isin(material, region["materials"])
    elif "box" in region:
        x0, x1, y0, y1 = (float(value) for value in region["box"])
        x = case.mesh.centroid[:, 0]
        y = case.mesh.centroid[:, 1]
        inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    elif "cells" in region:
        inside = np.zeros(n_cells, dtype=bool)
        inside[np.asarray(region["cells"], dtype=int) - 1] = True
    else:
        raise ValueError(f"force region {region.get('name')!r} needs 'materials', 'box' or 'cells'")

    if not inside.any():
        raise ValueError(f"force region {region.get('name')!r} contains no cells")
    return inside.astype(float)


def load_force_data(case: Case) -> ForceData | None:
    """Force observations from the ``"force_data"`` block, or None if there is none.

    Args:
        case: the case (run control, model and mesh).

    Returns:
        The observations, or None.
    """
    block = case.ctrl.raw.get("force_data")
    if not block:
        return None
    if case.model.vegetation.mode == "none" and case.model.porous.mode == "none":
        raise ValueError("force_data needs a 'vegetation' or a 'porous' block in the run control")

    spec = json.loads((case.ctrl.case_dir / block["file"]).read_text())
    regions = spec["regions"]

    # Region weights and observed components (unobserved components are masked out).
    weights = np.array([_region_cells(region, case) for region in regions])
    observed = np.zeros((len(regions), 2))
    mask = np.zeros((len(regions), 2))
    for k, region in enumerate(regions):
        for component, key in enumerate(("Fx", "Fy")):
            if key in region:
                observed[k, component] = float(region[key])
                mask[k, component] = 1.0

    data = ForceData(
        names=tuple(str(region.get("name", f"region{k}")) for k, region in enumerate(regions)),
        weights=jnp.asarray(weights),
        observed=jnp.asarray(observed),
        mask=jnp.asarray(mask),
        sigma=float(block.get("sigma_N", 0.1)),
        rho=float(block.get("rho", 1000.0)),
        factor=float(block.get("weight", 1.0)),
    )
    logger.info("force data: %d regions (%s), sigma %.3g N", len(regions),
                ", ".join(data.names), data.sigma)
    return data


def drag_force(
    m: SWEModel, theta: dict[str, Any], U: Array, region_weights: Array, rho: float,
) -> Array:
    """Drag force of the vegetation and the porous zones on regions of cells.

    Args:
        m: the model (with vegetation, porous zones, or both).
        theta: differentiable parameters.
        U: state [xi, q_x, q_y], shape [3, n_cells].
        region_weights: cell weights of every region, shape [n_regions, n_cells].
        rho: water density (kg/m^3).

    Returns:
        Force (N) on each region, x and y components, shape [n_regions, 2].
    """
    force = jnp.zeros((region_weights.shape[0], 2))

    # Stem drag (vegetation block).
    if m.vegetation.mode != "none":
        force = force + vegetation_force(m, theta, U, region_weights, rho)

    # Darcy-Forchheimer drag (porous block).
    if m.porous.mode != "none":
        force = force + porous_force(m, theta, U, region_weights, rho)

    return force


def force_loss(m: SWEModel, theta: dict[str, Any], U: Array, data: ForceData) -> Array:
    """Weighted force misfit (dimensionless).

    Args:
        m: the model (with vegetation, porous zones, or both).
        theta: differentiable parameters.
        U: final state [xi, q_x, q_y], shape [3, n_cells].
        data: the force observations.

    Returns:
        weight * mean of ((F_pred - F_obs) / sigma)^2 over the observed components.
    """
    predicted = drag_force(m, theta, U, data.weights, data.rho)
    squared = data.mask * ((predicted - data.observed) / data.sigma) ** 2
    return data.factor * jnp.sum(squared) / jnp.sum(data.mask)
