"""Horizontal turbulent diffusion of momentum in the depth-averaged SWEs.

Physics
-------
Depth averaging the turbulent stresses gives, in the momentum equations for q = h u, the
diffusion term (the common simplified form used in depth-averaged models)

    d q / dt  +=  div( h nu_t grad u )                         (m^2/s^2)

with the depth-averaged eddy viscosity nu_t (m^2/s). Two models:

- parabolic (SRH-2D):  nu_t = C_t u_* h + nu_0,
  where u_* = sqrt(tau_b / rho) = sqrt(g) n |u| / (k_n h^(1/6)) is the bed friction
  velocity (m/s) from Manning's law, C_t ~ 0.7 (Rodi 1993; SRH-2D default), and nu_0 the
  background viscosity;
- constant:            nu_t = nu_constant + nu_0.

The term spreads momentum across shear layers, e.g. the wake behind a vegetation patch or
a channel-floodplain interface. Without it, depth-averaged wakes do not recover.

Discretization
--------------
Finite volumes on the interior faces, with a two-point normal gradient:

    flux_f = h_f nu_f L_f (u_R - u_L) / (d . n),     d = x_R - x_L,

where h_f = min(h_L, h_R) (no diffusion into dry cells) and nu_f is the mean of the two
cell viscosities. The owner cell gains flux_f and the neighbour loses it, so momentum is
conserved exactly. Boundary faces carry no diffusive flux (free-slip walls; inlet and
exit fluxes are imposed by the boundary conditions). The stencil is the face-neighbour
stencil, so the Jacobian sparsity pattern of the scheme does not change.

The two-point gradient is exact on orthogonal meshes and first order on skewed ones (no
non-orthogonal correction); adequate for the diffusion term, which is small next to the
advective fluxes.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from hydrograd.physics.model import SWEModel

__all__ = ["eddy_viscosity", "turbulent_diffusion"]

# Differentiable parameters, e.g. {"eddy_Ct": ...}.
Theta = dict[str, Any]


def eddy_viscosity(m: SWEModel, theta: Theta, h: Array, u: Array, v: Array,
                   manning_n: Array) -> Array:
    """Depth-averaged eddy viscosity nu_t (m^2/s) per cell.

    Args:
        m: the model (``turbulence`` settings, ``g``, ``k_n``, ``h_small``).
        theta: differentiable parameters (``eddy_Ct`` overrides C_t).
        h: water depth (m), shape [n_cells].
        u, v: depth-averaged velocities (m/s), shape [n_cells].
        manning_n: Manning's n (s/m^(1/3)), shape [n_cells].

    Returns:
        nu_t (m^2/s), shape [n_cells].
    """
    turbulence = m.turbulence

    # Constant eddy viscosity.
    if turbulence.mode == "constant":
        return jnp.full_like(h, turbulence.nu_constant + turbulence.nu0)

    # Parabolic model: nu_t = C_t u_* h + nu_0. The speed is smoothed so that its
    # derivative is finite at u = 0, and the depth is floored at h_small.
    speed = jnp.sqrt(u * u + v * v + 1.0e-12)
    h_floor = jnp.maximum(h, m.h_small)
    u_star = jnp.sqrt(m.g) * manning_n * speed / (m.k_n * h_floor ** (1.0 / 6.0))
    C_t = theta.get("eddy_Ct", turbulence.coefficient)
    return C_t * u_star * h + turbulence.nu0


def turbulent_diffusion(m: SWEModel, theta: Theta, h: Array, u: Array, v: Array,
                        manning_n: Array) -> Array:
    """Rate of change of the state from turbulent diffusion, div(h nu_t grad u).

    Args:
        m: the model (mesh connectivity and turbulence settings).
        theta: differentiable parameters.
        h: water depth (m), shape [n_cells].
        u, v: depth-averaged velocities (m/s), shape [n_cells].
        manning_n: Manning's n (s/m^(1/3)), shape [n_cells].

    Returns:
        dU/dt contribution, shape [3, n_cells]: zero for xi, and the x- and y-momentum
        diffusion (m^2/s^2).
    """
    owner = m.fi_o
    neighbour = m.fi_r
    nu_t = eddy_viscosity(m, theta, h, u, v, manning_n)

    # Centroid-to-centroid distance projected on the face normal (m). The face-to-centroid
    # vectors are set by build_model for every model.
    if m.fi_ro is None or m.fi_rr is None:
        raise ValueError("turbulent diffusion needs the face-to-centroid vectors fi_ro, fi_rr")
    d_x = m.fi_ro[:, 0] - m.fi_rr[:, 0]
    d_y = m.fi_ro[:, 1] - m.fi_rr[:, 1]
    normal_distance = d_x * m.fi_nx + d_y * m.fi_ny

    # Face depth (no diffusion into or out of dry cells) and face viscosity.
    h_face = jnp.minimum(h[owner], h[neighbour])
    nu_face = 0.5 * (nu_t[owner] + nu_t[neighbour])

    # Face transfer coefficient h nu L / (d . n) (m^3/s).
    transfer = h_face * nu_face * m.fi_L / normal_distance

    # Diffusive momentum fluxes from the neighbour into the owner (m^4/s^2).
    flux_x = transfer * (u[neighbour] - u[owner])
    flux_y = transfer * (v[neighbour] - v[owner])

    # The owner gains, the neighbour loses; divide by the cell area.
    n_cells = m.n_cells
    rate_x = (jax.ops.segment_sum(flux_x, owner, n_cells)
              - jax.ops.segment_sum(flux_x, neighbour, n_cells))
    rate_y = (jax.ops.segment_sum(flux_y, owner, n_cells)
              - jax.ops.segment_sum(flux_y, neighbour, n_cells))

    return jnp.stack([jnp.zeros(n_cells), rate_x / m.area, rate_y / m.area])
