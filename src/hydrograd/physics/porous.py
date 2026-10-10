"""Darcy-Forchheimer drag of porous zones as a friction rate in the depth-averaged SWEs.

Physics
-------
A porous zone (dense vegetation treated as a porous medium, a log jam, a rock or brush
structure) resists the flow through it with a drag that is linear in the velocity at low
pore Reynolds numbers (Darcy) and quadratic at high ones (Forchheimer). Averaged over the
depth and the plan area and divided by the water density rho, the drag per unit plan area is

    F / rho = (alpha + beta |u|) u h                               (m^2/s^2)

where
- u is the depth-averaged superficial velocity (discharge per unit area, m/s);
- h is the water depth (m);
- alpha (1/s) is the Darcy coefficient and beta (1/m) the Forchheimer coefficient.

This is Approach 3 of the 2D vegetation parameterizations (the porous-media option of
SRH-2D and HEC-RAS). With the unit discharge q = h u it is a linear term in q,

    F / rho = c_df q,      c_df = alpha + beta |u|                 (1/s),

of the same form as the Manning bed friction c_bed q and the stem drag c_veg q of
``physics.vegetation``. The SWE momentum equations get the total rate
c = c_bed + c_veg + c_df, so the point-implicit friction, the implicit and steady
Jacobians and all adjoints include it with no further change.

Ergun coefficients
------------------
For a packed bed of grains or stems of diameter d (m) and porosity epsilon (-), Ergun's
equation (Ergun 1952) gives the pressure gradient per unit volume; multiplied by h it is
the drag per unit plan area above, with

    alpha = 150 nu (1 - epsilon)^2 / (epsilon^3 d^2)              (1/s)
    beta  = 1.75 (1 - epsilon) / (epsilon^3 d)                     (1/m)

and nu the kinematic viscosity of water (m^2/s). :func:`ergun_coefficients` evaluates them;
the run control can give either alpha and beta directly or epsilon and d.

Parameters
----------
alpha and beta are per material zone (``Porous.alpha_zone``, ``Porous.beta_zone``). Either
can be made differentiable as ``theta["df_alpha"]`` or ``theta["df_beta"]`` (one value per
material ID), e.g. to invert them from flow or force data.

Units: the formula is dimensionally consistent, so it holds in SI and in US customary units
(alpha in 1/s, beta in 1/length).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array

from hydrograd.physics.model import SWEModel
from hydrograd.physics.roe import smooth_sqrt

__all__ = ["ergun_coefficients", "porous_drag_rate", "porous_force"]

# Differentiable parameters, e.g. {"df_alpha": ..., "df_beta": ...}.
Theta = dict[str, Any]

# Kinematic viscosity of water at about 20 degrees C (m^2/s).
WATER_VISCOSITY = 1.0e-6


def ergun_coefficients(
    porosity: float, diameter: float, viscosity: float = WATER_VISCOSITY,
) -> tuple[float, float]:
    """Darcy and Forchheimer coefficients of a packed bed from Ergun's equation.

    Args:
        porosity: void fraction epsilon of the porous zone (-), 0 < epsilon <= 1.
        diameter: grain or stem diameter d (m).
        viscosity: kinematic viscosity of water nu (m^2/s).

    Returns:
        ``(alpha, beta)``: alpha = 150 nu (1 - epsilon)^2 / (epsilon^3 d^2) (1/s) and
        beta = 1.75 (1 - epsilon) / (epsilon^3 d) (1/m). Both are 0 for epsilon = 1.
    """
    if not 0.0 < porosity <= 1.0:
        raise ValueError(f"porosity must lie in (0, 1], got {porosity}")
    if diameter <= 0.0:
        raise ValueError(f"diameter must be positive, got {diameter}")

    solid_fraction = 1.0 - porosity
    porosity_cubed = porosity**3

    alpha = 150.0 * viscosity * solid_fraction**2 / (porosity_cubed * diameter**2)
    beta = 1.75 * solid_fraction / (porosity_cubed * diameter)
    return alpha, beta


def porous_drag_rate(m: SWEModel, theta: Theta, h: Array, qx: Array, qy: Array) -> Array:
    """Darcy-Forchheimer drag rate c_df = alpha + beta |u| (1/s) per cell.

    The drag per unit plan area (divided by the density) is c_df q.

    Args:
        m: the model (``porous`` settings and material IDs).
        theta: differentiable parameters (``df_alpha``, ``df_beta``; optional).
        h: water depth (m), floored at h_small by the caller, shape [n_cells].
        qx, qy: unit discharges (m^2/s), shape [n_cells].

    Returns:
        c_df (1/s), shape [n_cells]; zero where alpha = beta = 0.
    """
    porous = m.porous

    # Coefficients of each cell from its material zone (1/s and 1/m).
    alpha_zone = theta.get("df_alpha", jnp.asarray(porous.alpha_zone))
    beta_zone = theta.get("df_beta", jnp.asarray(porous.beta_zone))
    alpha = alpha_zone[m.mat_id]
    beta = beta_zone[m.mat_id]

    # Depth-averaged speed |u| = |q| / h (m/s); the smoothed |q| keeps the derivative
    # finite at q = 0.
    q_mag = smooth_sqrt(qx * qx + qy * qy)
    speed = q_mag / h

    return alpha + beta * speed


def porous_force(
    m: SWEModel,
    theta: Theta,
    U: Array,
    region_weights: Array,
    rho: float = 1000.0,
) -> Array:
    """Darcy-Forchheimer drag force on the flow, integrated over regions of cells.

    Args:
        m: the model (``porous`` settings, cell areas, still-water depth).
        theta: differentiable parameters (``df_alpha``, ``df_beta``, ``zb``).
        U: state [xi, q_x, q_y], shape [3, n_cells] (or more rows; only the first three
            are used).
        region_weights: weight of every cell in every region (1 inside, 0 outside),
            shape [n_regions, n_cells].
        rho: water density (kg/m^3).

    Returns:
        Force (N) on each region, x and y components, shape [n_regions, 2].
    """
    # Depth h = xi + h_still (m), clipped at zero and floored at h_small as in the
    # residual.
    h_still = m.wstill - theta["zb"] if "zb" in theta else m.hstill
    h = jnp.maximum(U[0] + h_still, 0.0)
    h_floor = jnp.maximum(h, m.h_small)
    qx = U[1]
    qy = U[2]

    # Drag per unit plan area divided by the density: c_df q (m^2/s^2), times the area.
    rate = porous_drag_rate(m, theta, h_floor, qx, qy)
    drag_x = rate * qx * m.area
    drag_y = rate * qy * m.area

    # Sum over the cells of each region and convert to newtons.
    force_x = region_weights @ drag_x
    force_y = region_weights @ drag_y
    return rho * jnp.stack([force_x, force_y], axis=1)
