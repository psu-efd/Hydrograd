"""Loss terms of inversion and UDE training (port of ``swe_2D_inversion.jl``).

All terms use the *final* saved frame. With ``eps_s = sqrt(eps(Float64))``:

* ``L_WSE   = mean(((WSE_pred - WSE_t) / (max WSE_t - min WSE_t + eps_s))^2)``
* ``L_uv    = (sum((u - u_t)/V)^2 + sum((v - v_t)/V)^2) / N``, ``V = max |U_t|``
* ``L_bound = sum_p [(max(0, lb - p)/(ub - lb))^2 + (max(0, p - ub)/(ub - lb))^2] / N``
  (divided by the number of *cells*, as in Julia)
* ``L_slope = bound-loss of |grad zb| against [0, slope_ub]`` (zb inversion only)

with ``h = xi + h_still``, ``u = q_x/(h + h_small)``, ``WSE = h + zb``.

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

EPS_S = float(np.sqrt(np.finfo(np.float64).eps))
EPS = float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class Truth:
    """Observed final-state fields (per cell): WSE (m), u, v (m/s)."""

    wse: Array
    u: Array
    v: Array


def load_truth(d: dict[str, Any]) -> Truth:
    """From a truth JSON dict (``wse_truth``, ``u_truth``, ``v_truth``)."""
    return Truth(*(jnp.asarray(np.asarray(d[k], dtype=float)) for k in ("wse_truth", "u_truth", "v_truth")))


@dataclass(frozen=True)
class LossWeights:
    """Which terms are active (Julia uses on/off switches, unit weights)."""

    wse: bool = True
    uv: bool = False
    bound: bool = False
    slope: bool = False
    lb: float = -np.inf
    ub: float = np.inf
    slope_ub: float = np.inf


def data_losses(m: SWEModel, U: Array, truth: Truth, zb: Array | None = None) -> dict[str, Array]:
    """WSE and velocity misfits of a final state ``U`` [3, n_cells]."""
    hstill = m.hstill if zb is None else m.wstill - zb   # consistent with physics.rhs.still_depth
    zb = m.zb if zb is None else zb
    h = U[0] + hstill
    ve = m.h_small if m.loss_velocity_eps is None else m.loss_velocity_eps
    u, v = U[1] / (h + ve), U[2] / (h + ve)
    wse = h + zb
    span = jnp.max(truth.wse) - jnp.min(truth.wse) + EPS_S
    l_wse = jnp.mean(((wse - truth.wse) / span) ** 2)
    vmax = jnp.maximum(jnp.max(jnp.sqrt(truth.u ** 2 + truth.v ** 2)), EPS)
    l_uv = (jnp.sum(((u - truth.u) / vmax) ** 2) + jnp.sum(((v - truth.v) / vmax) ** 2)) / m.n_cells
    return {"wse": l_wse, "uv": l_uv, "WSE_pred": wse, "h_pred": h, "u_pred": u, "v_pred": v}


def bound_loss(p: Array, lb: float, ub: float, n_cells: int) -> Array:
    """Quadratic penalty outside [lb, ub], normalized by ``(ub - lb)`` and ``n_cells``."""
    w = ub - lb
    return jnp.sum((jnp.maximum(0.0, lb - p) / w) ** 2 + (jnp.maximum(0.0, p - ub) / w) ** 2) / n_cells


def slope_loss(m: SWEModel, zb: Array, slope_ub: float) -> Array:
    """Penalty on bed slopes steeper than ``slope_ub`` (Green-Gauss gradient)."""
    S = bed_slope(m.bed, zb)
    smag = jnp.sqrt(S[:, 0] ** 2 + S[:, 1] ** 2 + EPS)
    return bound_loss(smag, 0.0, slope_ub, m.n_cells)
