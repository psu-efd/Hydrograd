"""Limited linear reconstruction for second-order MUSCL on unstructured meshes.

* Gradients: Green-Gauss, face values = mean of the two cells (cell value on boundary faces).
* Limiters (phi_i = min over the cell's faces of phi_f; Delta_f = grad q_i . (x_f - x_i),
  Delta_+ = q_max - q_i or q_min - q_i by the sign of Delta_f, q_min/max over the cell and
  its face neighbours):
  - ``venkatakrishnan`` (default): phi_f = (D+^2 + eps^2 + 2 D+ Df) / (D+^2 + 2 Df^2 + D+ Df + eps^2),
    eps^2 = (K dx)^3 with dx = sqrt(A), K = 1 (Venkatakrishnan 1995). Smooth: inactive in smooth
    flow, so steady residuals converge (Newton / time marching) and gradients are well behaved.
  - ``barth_jespersen``: phi_f = min(1, D+/Df). Strict local bounds, but its switching stalls
    steady convergence (limit cycles; residual ~1e-2 in the bump test).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from hydrograd.physics.model import SWEModel

__all__ = ["green_gauss", "limited_face_values"]

TINY = 1.0e-14


def green_gauss(m: SWEModel, q: Array) -> Array:
    """Cell gradients of a cell field q, shape [n_cells, 2]."""
    qf = 0.5 * (q[m.fi_o] + q[m.fi_r])
    nL = jnp.stack([m.fi_nx * m.fi_L, m.fi_ny * m.fi_L], axis=1)
    bL = jnp.stack([m.fb_nx * m.fb_L, m.fb_ny * m.fb_L], axis=1)
    acc = (jax.ops.segment_sum(qf[:, None] * nL, m.fi_o, m.n_cells)
           - jax.ops.segment_sum(qf[:, None] * nL, m.fi_r, m.n_cells)
           + jax.ops.segment_sum(q[m.fb_c][:, None] * bL, m.fb_c, m.n_cells))
    return acc / m.area[:, None]


def _neighbour_range(m: SWEModel, q: Array) -> tuple[Array, Array]:
    """(min, max) of q over each cell and its face neighbours."""
    n = m.n_cells
    qmax = jnp.maximum(q, jnp.maximum(jax.ops.segment_max(q[m.fi_r], m.fi_o, n),
                                      jax.ops.segment_max(q[m.fi_o], m.fi_r, n)))
    qmin = jnp.minimum(q, jnp.minimum(jax.ops.segment_min(q[m.fi_r], m.fi_o, n),
                                      jax.ops.segment_min(q[m.fi_o], m.fi_r, n)))
    return qmin, qmax


def _ratio(d: Array, qi: Array, qmin: Array, qmax: Array) -> Array:
    big = jnp.ones_like(d)
    pos, neg = d > TINY, d < -TINY
    r = jnp.where(pos, (qmax - qi) / jnp.where(pos, d, 1.0), big)
    r = jnp.where(neg, (qmin - qi) / jnp.where(neg, d, 1.0), r)
    return jnp.minimum(r, 1.0)


def _venkat(d: Array, qi: Array, qmin: Array, qmax: Array, eps2: Array) -> Array:
    dp = jnp.where(d >= 0.0, qmax - qi, qmin - qi)
    num = dp * dp + eps2 + 2.0 * d * dp
    den = dp * dp + 2.0 * d * d + dp * d + eps2
    return jnp.where(den > 0.0, num / jnp.where(den > 0.0, den, 1.0), 1.0)


def limited_face_values(m: SWEModel, q: Array, first_order: Array,
                        limiter: str = "venkatakrishnan", K: float = 1.0) -> tuple[Array, Array]:
    """Limited values of q at interior-face midpoints.

    Args:
        m: model (needs ``fi_ro``, ``fi_rr``: midpoint minus owner/neighbour centroid, m).
        q: cell values [n_cells].
        first_order: weight in [0, 1] per cell multiplying the limiter (0 = first order,
            e.g. next to dry cells), or a bool array (True = first order).
        limiter: ``venkatakrishnan`` or ``barth_jespersen``; K: Venkatakrishnan constant.

    Returns:
        (q at the face from the owner side, q from the neighbour side), each [n_interior_faces].
    """
    assert m.fi_ro is not None and m.fi_rr is not None
    g = green_gauss(m, q)
    qmin, qmax = _neighbour_range(m, q)
    o, r = m.fi_o, m.fi_r
    d_o = jnp.sum(g[o] * m.fi_ro, axis=1)
    d_r = jnp.sum(g[r] * m.fi_rr, axis=1)
    n = m.n_cells
    if limiter == "barth_jespersen":
        f_o, f_r = _ratio(d_o, q[o], qmin[o], qmax[o]), _ratio(d_r, q[r], qmin[r], qmax[r])
    else:
        eps2 = (K * jnp.sqrt(m.area)) ** 3
        f_o = _venkat(d_o, q[o], qmin[o], qmax[o], eps2[o])
        f_r = _venkat(d_r, q[r], qmin[r], qmax[r], eps2[r])
    phi = jnp.minimum(jax.ops.segment_min(f_o, o, n), jax.ops.segment_min(f_r, r, n))
    if first_order.dtype == jnp.bool_:
        phi = jnp.where(first_order, 0.0, jnp.clip(phi, 0.0, 1.0))
    else:
        phi = first_order * jnp.clip(phi, 0.0, 1.0)
    return q[o] + phi[o] * d_o, q[r] + phi[r] * d_r
