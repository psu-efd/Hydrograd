"""Drag of rigid emergent vegetation as a friction rate in the depth-averaged SWEs.

Physics
-------
An array of rigid, emergent stems (diameter d, spacing dS) exerts a drag force on the flow.
Averaged over the plan area and divided by the water density rho, the drag per unit plan
area is

    F / rho = 1/2 C_D a h |u| u                                  (m^2/s^2)

where
- a = d / dS^2 is the frontal area per unit volume (1/m) (Nepf 2012); a h is the frontal
  area per unit bed area;
- C_D is the bulk drag coefficient of the stems (-), referred to the depth-averaged
  velocity u (m/s);
- h is the water depth (m).

With the unit discharge q = h u this is a linear term in q,

    F / rho = c_veg q,      c_veg = 1/2 C_D a |u|                (1/s),

of the same form as the Manning bed friction c_bed q. The SWE momentum equations therefore
get the total rate c = c_bed + c_veg. Because the vegetation enters only through this rate,
the point-implicit friction of the explicit SSP-RK stepper, the implicit and steady
Jacobians and all adjoints include it with no further change.

Parameterizations of C_D
------------------------
- ``zone``: one C_D per material zone (constant, or inverted as ``theta["veg_CD"]``).
- ``nn_CD``: a local neural-network closure C_D_i = f(h_i, |u|_i, a_i) with weights in
  ``theta["nn"]``. Each cell's C_D depends only on that cell's state (a local law).

The frontal-area density a is a property of the vegetation (one value per material zone);
it can also be made a parameter, ``theta["veg_a"]``.

Units: the formula is dimensionally consistent, so it holds in SI and in US customary units.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array

from hydrograd.closures.mlp import apply_mlp, normalize
from hydrograd.physics.model import SWEModel
from hydrograd.physics.roe import smooth_sqrt

__all__ = ["vegetation_drag_coefficient", "vegetation_drag_rate"]

# Differentiable parameters, e.g. {"veg_CD": ..., "veg_a": ..., "nn": ...}.
Theta = dict[str, Any]


def vegetation_drag_coefficient(
    m: SWEModel, theta: Theta, h: Array, speed: Array, a: Array,
) -> Array:
    """Drag coefficient C_D of the stems in each cell.

    Args:
        m: the model (its ``vegetation`` selects the parameterization).
        theta: differentiable parameters: ``veg_CD`` (per zone) or ``nn`` (network weights).
        h: water depth (m), shape [n_cells].
        speed: depth-averaged speed |u| (m/s), shape [n_cells].
        a: frontal area per unit volume (1/m), shape [n_cells].

    Returns:
        C_D (-), shape [n_cells].
    """
    veg = m.vegetation

    # Local neural-network closure C_D = f(h, |u|, a), inputs normalized to [-1, 1].
    if veg.mode == "nn_CD":
        h_input = normalize(h, *veg.h_bounds)
        speed_input = normalize(speed, *veg.umag_bounds)
        a_input = normalize(a, *veg.a_bounds)
        features = jnp.stack([h_input, speed_input, a_input], axis=1)
        network_output = apply_mlp(theta["nn"], features, veg.activations, *veg.CD_bounds)
        return network_output[:, 0]

    # Default: one C_D per material zone.
    CD_zone = theta.get("veg_CD", jnp.asarray(veg.CD_zone))
    return CD_zone[m.mat_id]


def vegetation_drag_rate(m: SWEModel, theta: Theta, h: Array, qx: Array, qy: Array) -> Array:
    """Vegetation drag rate c_veg = 1/2 C_D a |u| (1/s) per cell.

    The drag per unit plan area (divided by the density) is c_veg q.

    Args:
        m: the model (``vegetation`` settings and material IDs).
        theta: differentiable parameters (``veg_CD``, ``veg_a`` or ``nn``).
        h: water depth (m), floored at h_small by the caller, shape [n_cells].
        qx, qy: unit discharges (m^2/s), shape [n_cells].

    Returns:
        c_veg (1/s), shape [n_cells]; zero where a = 0.
    """
    veg = m.vegetation

    # Frontal area per unit volume in each cell (1/m), from its material zone.
    a_zone = theta.get("veg_a", jnp.asarray(veg.a_zone))
    a = a_zone[m.mat_id]

    # Depth-averaged speed |u| = |q| / h (m/s); the smoothed |q| keeps the derivative
    # finite at q = 0.
    q_mag = smooth_sqrt(qx * qx + qy * qy)
    speed = q_mag / h

    # Drag coefficient and rate.
    CD = vegetation_drag_coefficient(m, theta, h, speed, a)
    return 0.5 * CD * a * speed
