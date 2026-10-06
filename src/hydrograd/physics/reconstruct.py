"""Limited linear reconstruction for second-order MUSCL on unstructured meshes.

Used by the ``hr_hllc`` scheme at ``order = 2`` (:mod:`hydrograd.physics.rhs_hr`).

Gradients
---------
Green-Gauss: grad q_i = (1/A_i) sum_f q_f n_f L_f, with the face value q_f the mean of
the two cells on interior faces and the cell value on boundary faces.

Face values
-----------
On an interior face f of cell i, with x_f the face midpoint and x_i the cell centroid,

    q_f = q_i + phi_i Delta_f,    Delta_f = grad q_i . (x_f - x_i),

where phi_i in [0, 1] is the limiter of cell i (0 gives first order).

Limiters
--------
phi_i = min over the cell's faces of phi_f. Delta_+ = q_max - q_i or q_min - q_i by the
sign of Delta_f, with q_min/max over the cell and its face neighbours.

- ``venkatakrishnan`` (default), with D+ = Delta_+ and Df = Delta_f:

      phi_f = (D+^2 + eps^2 + 2 D+ Df) / (D+^2 + 2 Df^2 + D+ Df + eps^2),

  eps^2 = (K dx)^3 with dx = sqrt(A) (m), K = 1 (Venkatakrishnan 1995). Smooth: inactive
  in smooth flow, so steady residuals converge (Newton / time marching) and gradients are
  well behaved.
- ``barth_jespersen``: phi_f = min(1, D+/Df). Strict local bounds, but its switching stalls
  steady convergence (limit cycles; residual ~1e-2 in the bump test).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from hydrograd.physics.model import SWEModel

__all__ = ["green_gauss", "limited_face_values"]

# Face increments |Delta_f| <= TINY are treated as zero (no limiting needed).
TINY = 1.0e-14


def green_gauss(m: SWEModel, q: Array) -> Array:
    """Cell gradients of a cell field q, shape [n_cells, 2].

    Args:
        m: the model (mesh connectivity, face normals and lengths, cell areas).
        q: cell values (any units), shape [n_cells].

    Returns:
        Gradient of q (units of q per m), shape [n_cells, 2].
    """
    # Interior face values (mean of the two cells) and normal-times-length vectors (m).
    qf = 0.5 * (q[m.fi_o] + q[m.fi_r])
    nL = jnp.stack([m.fi_nx * m.fi_L, m.fi_ny * m.fi_L], axis=1)
    bL = jnp.stack([m.fb_nx * m.fb_L, m.fb_ny * m.fb_L], axis=1)

    # sum_f q_f n_f L_f per cell; the normal points out of the owner, into the neighbour.
    surface_sum = (
        jax.ops.segment_sum(qf[:, None] * nL, m.fi_o, m.n_cells)
        - jax.ops.segment_sum(qf[:, None] * nL, m.fi_r, m.n_cells)
        + jax.ops.segment_sum(q[m.fb_c][:, None] * bL, m.fb_c, m.n_cells)
    )
    return surface_sum / m.area[:, None]


def _neighbour_range(m: SWEModel, q: Array) -> tuple[Array, Array]:
    """(min, max) of q over each cell and its face neighbours.

    Args:
        m: the model (interior-face connectivity).
        q: cell values, shape [n_cells].

    Returns:
        (q_min, q_max), each of shape [n_cells].
    """
    n = m.n_cells

    # Largest neighbour value, seen from both face orientations.
    max_over_neighbours = jax.ops.segment_max(q[m.fi_r], m.fi_o, n)
    max_over_owners = jax.ops.segment_max(q[m.fi_o], m.fi_r, n)
    qmax = jnp.maximum(q, jnp.maximum(max_over_neighbours, max_over_owners))

    # Smallest neighbour value, seen from both face orientations.
    min_over_neighbours = jax.ops.segment_min(q[m.fi_r], m.fi_o, n)
    min_over_owners = jax.ops.segment_min(q[m.fi_o], m.fi_r, n)
    qmin = jnp.minimum(q, jnp.minimum(min_over_neighbours, min_over_owners))

    return qmin, qmax


def _ratio(d: Array, qi: Array, qmin: Array, qmax: Array) -> Array:
    """Barth-Jespersen face limiter phi_f = min(1, Delta_+ / Delta_f).

    Args:
        d: unlimited face increment Delta_f, shape [n_faces].
        qi: cell value, shape [n_faces].
        qmin, qmax: local bounds of the cell, shape [n_faces].

    Returns:
        phi_f (-), at most 1 (1 where |Delta_f| <= TINY), shape [n_faces].
    """
    big = jnp.ones_like(d)
    pos = d > TINY
    neg = d < -TINY

    # Increasing face value: limit by the maximum; decreasing: by the minimum.
    r = jnp.where(pos, (qmax - qi) / jnp.where(pos, d, 1.0), big)
    r = jnp.where(neg, (qmin - qi) / jnp.where(neg, d, 1.0), r)

    return jnp.minimum(r, 1.0)


def _venkat(d: Array, qi: Array, qmin: Array, qmax: Array, eps2: Array) -> Array:
    """Venkatakrishnan face limiter (see the module docstring).

    Args:
        d: unlimited face increment Delta_f, shape [n_faces].
        qi: cell value, shape [n_faces].
        qmin, qmax: local bounds of the cell, shape [n_faces].
        eps2: eps^2 = (K dx)^3 of the cell, shape [n_faces].

    Returns:
        phi_f (-), shape [n_faces]; 1 where the denominator vanishes.
    """
    # Allowed increment Delta_+ in the direction of Delta_f.
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
            e.g. next to dry cells), or a bool array (True = first order), shape [n_cells].
        limiter: ``venkatakrishnan`` or ``barth_jespersen``.
        K: Venkatakrishnan constant (-).

    Returns:
        (q at the face from the owner side, q from the neighbour side), each [n_interior_faces].
    """
    assert m.fi_ro is not None and m.fi_rr is not None

    # ------------------------------------------------------------------
    # 1. Unlimited increments Delta_f = grad q . (x_f - x_i) on both sides
    # ------------------------------------------------------------------
    gradient = green_gauss(m, q)
    qmin, qmax = _neighbour_range(m, q)
    owner = m.fi_o
    neighbour = m.fi_r
    d_o = jnp.sum(gradient[owner] * m.fi_ro, axis=1)
    d_r = jnp.sum(gradient[neighbour] * m.fi_rr, axis=1)
    n = m.n_cells

    # ------------------------------------------------------------------
    # 2. Face limiters phi_f on the owner and neighbour side
    # ------------------------------------------------------------------
    if limiter == "barth_jespersen":
        f_o = _ratio(d_o, q[owner], qmin[owner], qmax[owner])
        f_r = _ratio(d_r, q[neighbour], qmin[neighbour], qmax[neighbour])
    else:
        eps2 = (K * jnp.sqrt(m.area)) ** 3
        f_o = _venkat(d_o, q[owner], qmin[owner], qmax[owner], eps2[owner])
        f_r = _venkat(d_r, q[neighbour], qmin[neighbour], qmax[neighbour], eps2[neighbour])

    # ------------------------------------------------------------------
    # 3. Cell limiter: minimum over the cell's faces, then the first-order blend
    # ------------------------------------------------------------------
    phi = jnp.minimum(jax.ops.segment_min(f_o, owner, n), jax.ops.segment_min(f_r, neighbour, n))
    if first_order.dtype == jnp.bool_:
        phi = jnp.where(first_order, 0.0, jnp.clip(phi, 0.0, 1.0))
    else:
        phi = first_order * jnp.clip(phi, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 4. Limited face values
    # ------------------------------------------------------------------
    q_face_owner = q[owner] + phi[owner] * d_o
    q_face_neighbour = q[neighbour] + phi[neighbour] * d_r

    return q_face_owner, q_face_neighbour
