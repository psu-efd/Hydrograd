"""Loss terms of inversion and UDE training (port of ``swe_2D_inversion.jl``).

Definitions
-----------
All terms use the *final* saved frame. With ``eps_s = sqrt(eps(Float64))`` and N the
number of cells:

* ``L_WSE   = mean(((WSE_pred - WSE_t) / (max WSE_t - min WSE_t + eps_s))^2)``
* ``L_uv    = (sum((u - u_t)/V)^2 + sum((v - v_t)/V)^2) / N``, ``V = max |U_t|`` (m/s)
* ``L_bound = sum_p [(max(0, lb - p)/(ub - lb))^2 + (max(0, p - ub)/(ub - lb))^2] / N``
  (divided by the number of *cells*, as in Julia)
* ``L_slope = bound-loss of |grad zb| against [0, slope_ub]`` (zb inversion only)

with ``h = xi + h_still`` (m), ``u = q_x/(h + h_small)`` (m/s), ``WSE = h + zb`` (m).
All terms are dimensionless.

Velocity definition
-------------------
Velocities use the same definition as the truth files (``u = q/(h + h_small)``,
Hydrograd.jl post-processing). Julia's loss uses ``q/(h + eps_s)`` instead, which
leaves a loss floor (~4e-8 on Savannah) and a small bias in the recovered
parameters; ``SWEModel.loss_velocity_eps = EPS_S`` reproduces Julia for parity tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

from hydrograd.bed import bed_slope
from hydrograd.physics.model import SWEModel

__all__ = ["LossWeights", "Truth", "bound_loss", "data_losses", "load_truth", "slope_loss"]

# sqrt of the float64 machine epsilon (Julia's sqrt(eps(Float64))) and the epsilon itself.
EPS_S = float(np.sqrt(np.finfo(np.float64).eps))
EPS = float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class Truth:
    """Observed final-state fields per cell.

    Attributes:
        wse: water-surface elevation (m), shape [n_cells].
        u: x-velocity (m/s), shape [n_cells].
        v: y-velocity (m/s), shape [n_cells].
    """

    wse: Array
    u: Array
    v: Array


def load_truth(d: dict[str, Any]) -> Truth:
    """Truth fields from a truth JSON dict (keys ``wse_truth``, ``u_truth``, ``v_truth``).

    Args:
        d: the parsed truth JSON file (written by a forward run).

    Returns:
        The truth fields as float64 JAX arrays, each of shape [n_cells].
    """
    wse = jnp.asarray(np.asarray(d["wse_truth"], dtype=float))
    u = jnp.asarray(np.asarray(d["u_truth"], dtype=float))
    v = jnp.asarray(np.asarray(d["v_truth"], dtype=float))

    return Truth(wse, u, v)


@dataclass(frozen=True)
class LossWeights:
    """Which loss terms are active (Julia uses on/off switches, i.e. unit weights).

    Attributes:
        wse: include L_WSE.
        uv: include L_uv.
        bound: include L_bound.
        slope: include L_slope (zb inversion only).
        lb: lower parameter bound (units of the parameter).
        ub: upper parameter bound (units of the parameter).
        slope_ub: upper bound of the bed-slope magnitude |grad zb| (-).
    """

    wse: bool = True
    uv: bool = False
    bound: bool = False
    slope: bool = False
    lb: float = -np.inf
    ub: float = np.inf
    slope_ub: float = np.inf


def data_losses(m: SWEModel, U: Array, truth: Truth, zb: Array | None = None) -> dict[str, Array]:
    """WSE and velocity misfits of a final state (see the module docstring).

    Args:
        m: the model.
        U: final state [xi, q_x, q_y], shape [3, n_cells].
        truth: observed final-state fields.
        zb: bed elevation (m), shape [n_cells], when it is the inverted parameter;
            None to use the model bed.

    Returns:
        Dict with the scalar losses ``"wse"`` and ``"uv"`` (dimensionless) and the
        predicted fields, each of shape [n_cells]: ``"WSE_pred"`` (m), ``"h_pred"`` (m),
        ``"u_pred"`` and ``"v_pred"`` (m/s).
    """
    # Still-water depth, consistent with physics.rhs.still_depth (h_still = w_still - zb
    # when zb is inverted).
    hstill = m.hstill if zb is None else m.wstill - zb
    zb = m.zb if zb is None else zb

    # Predicted depth, velocities and water-surface elevation.
    h = U[0] + hstill
    velocity_eps = m.h_small if m.loss_velocity_eps is None else m.loss_velocity_eps
    u = U[1] / (h + velocity_eps)
    v = U[2] / (h + velocity_eps)
    wse = h + zb

    # L_WSE: misfit normalized by the range of the observed WSE.
    span = jnp.max(truth.wse) - jnp.min(truth.wse) + EPS_S
    l_wse = jnp.mean(((wse - truth.wse) / span) ** 2)

    # L_uv: misfit normalized by the largest observed speed.
    vmax = jnp.maximum(jnp.max(jnp.sqrt(truth.u ** 2 + truth.v ** 2)), EPS)
    sum_u = jnp.sum(((u - truth.u) / vmax) ** 2)
    sum_v = jnp.sum(((v - truth.v) / vmax) ** 2)
    l_uv = (sum_u + sum_v) / m.n_cells

    return {"wse": l_wse, "uv": l_uv, "WSE_pred": wse, "h_pred": h, "u_pred": u, "v_pred": v}


def bound_loss(p: Array, lb: float, ub: float, n_cells: int) -> Array:
    """Quadratic penalty outside [lb, ub], normalized by ``(ub - lb)`` and ``n_cells``.

    Args:
        p: parameter values, shape [n_params].
        lb: lower bound (units of ``p``).
        ub: upper bound (units of ``p``).
        n_cells: number of cells (the divisor, as in Julia).

    Returns:
        Scalar penalty (dimensionless).
    """
    width = ub - lb
    below = (jnp.maximum(0.0, lb - p) / width) ** 2
    above = (jnp.maximum(0.0, p - ub) / width) ** 2

    return jnp.sum(below + above) / n_cells


def slope_loss(m: SWEModel, zb: Array, slope_ub: float) -> Array:
    """Penalty on bed slopes steeper than ``slope_ub`` (Green-Gauss gradient).

    Args:
        m: the model (bed geometry for the gradient).
        zb: bed elevation (m), shape [n_cells].
        slope_ub: largest allowed slope magnitude |grad zb| (-).

    Returns:
        Scalar penalty (dimensionless).
    """
    S = bed_slope(m.bed, zb)
    slope_magnitude = jnp.sqrt(S[:, 0] ** 2 + S[:, 1] ** 2 + EPS)

    return bound_loss(slope_magnitude, 0.0, slope_ub, m.n_cells)
