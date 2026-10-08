"""Dispersive momentum flux from sub-cell velocity variations (momentum coefficient beta).

Physics
-------
The 2D SWEs follow from averaging the 3D momentum equations over the depth and, for a
finite-volume cell, over the plan area of the cell. The advective momentum flux averages
to

    < u_i u_j > = beta_ij U_i U_j,

where U is the averaged velocity and beta the momentum (Boussinesq) coefficient. It has
a vertical part (the velocity profile over the depth) and a horizontal part (velocity
variations within the cell). Inside an array of emergent stems the horizontal part is
large where a cell holds about one stem: the pore jets between the stems and the wakes
behind them give beta ~ 1.3 in a dense array (a = 9 1/m, one stem per 3.3 cm cell; RANS
of the pilot flume), while the vertical part is ~ 1.003.

The 2D flux (HLLC) uses beta = 1. The missing part is added as the divergence of the
extra flux tensor (per unit density):

    S_x = - div[(beta - 1) h U U]     = - [d/dx((beta-1) h U^2) + d/dy((beta-1) h U V)]
    S_y = - div[(beta - 1) h U V]     (m^2/s^2)

with h the depth (m) and U = (u, v) the depth-averaged velocity (m/s). Where beta jumps
(the leading edge of a patch) the extra flux rises and the term retards the flow; where
it falls (the trailing edge) it accelerates it, like a pressure jump at each edge.

Discretization
--------------
Finite volume: the extra flux through an interior face is the average of the owner and
neighbour values of (beta - 1) h U (U . n) times the face length; the cell rate is the
net inflow divided by the cell area. Boundary faces carry no extra flux (beta = 1 at
inlets and outlets; U . n = 0 at walls). The term couples face neighbours only, so the
Jacobian sparsity pattern of the scheme is unchanged.

Units: SI (or any consistent system); beta is dimensionless.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from hydrograd.physics.model import SWEModel


def momentum_coefficient(m: SWEModel, theta: dict[str, Any]) -> Array:
    """Momentum coefficient beta per cell.

    Args:
        m: the model (``dispersion.beta_zone`` per material, ``mat_id`` per cell).
        theta: differentiable parameters; ``theta["disp_beta"]`` (one value per material
            ID) overrides ``beta_zone``.

    Returns:
        beta (-), shape [n_cells].
    """
    beta_zone = theta.get("disp_beta", jnp.asarray(m.dispersion.beta_zone))
    return beta_zone[m.mat_id]


def dispersive_flux_source(
    m: SWEModel, theta: dict[str, Any], h: Array, u: Array, v: Array
) -> Array:
    """Rate of change of the state from the extra momentum flux (beta - 1) h U U.

    Args:
        m: the model (mesh connectivity, ``dispersion`` settings).
        theta: differentiable parameters (``disp_beta``).
        h: water depth (m), shape [n_cells].
        u, v: depth-averaged velocities (m/s), shape [n_cells].

    Returns:
        dU/dt contribution [0, S_x, S_y], shape [3, n_cells]; momentum rows in m^2/s^2.
    """
    owner = m.fi_o
    neighbour = m.fi_r
    n_cells = m.n_cells

    # Excess coefficient (beta - 1) h per cell (m).
    excess = (momentum_coefficient(m, theta) - 1.0) * h

    # Velocity normal to each interior face in the owner and the neighbour (m/s).
    normal_owner = u[owner] * m.fi_nx + v[owner] * m.fi_ny
    normal_neighbour = u[neighbour] * m.fi_nx + v[neighbour] * m.fi_ny

    # Extra momentum flux from owner to neighbour (m^4/s^2): face average of
    # (beta - 1) h U (U . n), times the face length.
    flux_x = 0.5 * m.fi_L * (
        excess[owner] * u[owner] * normal_owner
        + excess[neighbour] * u[neighbour] * normal_neighbour
    )
    flux_y = 0.5 * m.fi_L * (
        excess[owner] * v[owner] * normal_owner
        + excess[neighbour] * v[neighbour] * normal_neighbour
    )

    # The owner loses what crosses to the neighbour; divide by the cell area.
    rate_x = (jax.ops.segment_sum(flux_x, neighbour, n_cells)
              - jax.ops.segment_sum(flux_x, owner, n_cells))
    rate_y = (jax.ops.segment_sum(flux_y, neighbour, n_cells)
              - jax.ops.segment_sum(flux_y, owner, n_cells))

    return jnp.stack([jnp.zeros(n_cells), rate_x / m.area, rate_y / m.area])
