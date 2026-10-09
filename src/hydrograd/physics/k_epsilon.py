"""Depth-averaged k-epsilon turbulence model (Rastogi & Rodi 1978).

Equations
---------
With the depth-averaged turbulent kinetic energy k (m^2/s^2) and its dissipation rate
epsilon (m^2/s^3), the model transports the depth-integrated quantities K = h k and
E = h epsilon, which are rows 3 and 4 of the state U = [xi, q_x, q_y, K, E]:

    d(h k)/dt   + div(h U k)   = div(h nu_t / sigma_k grad k)
                                 + h (P_h + P_kV - epsilon)
    d(h eps)/dt + div(h U eps) = div(h nu_t / sigma_e grad eps)
                                 + h (c_1 (eps / k) P_h + P_eV - c_2 eps^2 / k)

with the eddy viscosity nu_t = c_mu k^2 / epsilon (m^2/s), used also in the momentum
diffusion div(h nu_t grad u), and

- P_h  = nu_t |S|^2, |S|^2 = 2 U_x^2 + 2 V_y^2 + (U_y + V_x)^2: production by the
  horizontal shear of the depth-averaged velocity (m^2/s^3);
- P_kV = c_k u_*^3 / h and P_eV = c_e u_*^4 / h^2: production by the bed shear (the
  vertical velocity profile), with u_* the bed friction velocity,
  c_k = 1 / sqrt(c_f), c_e = 3.6 c_2 sqrt(c_mu) / c_f^(3/4), c_f = u_*^2 / |U|^2.

With Manning's law u_* = sqrt(c_f) |U|, c_f = g n^2 / (k_n^2 h^(1/3)), so

    P_kV = c_f |U|^3 / h                  (the work of the bed shear stress),
    P_eV = 3.6 c_2 sqrt(c_mu) c_f^(5/4) |U|^4 / h^2.

In uniform flow (P_h = 0) the local equilibrium is

    epsilon_eq = c_f |U|^3 / h,   k_eq = u_*^2 / (3.6 sqrt(c_mu) c_f^(1/4)),
    nu_t,eq = c_mu k_eq^2 / epsilon_eq = u_* h / 12.96 = 0.077 u_* h,

the classic depth-averaged eddy viscosity of Rastogi & Rodi. Standard constants:
c_mu = 0.09, c_1 = 1.44, c_2 = 1.92, sigma_k = 1.0, sigma_e = 1.3.

Side walls (wall functions)
---------------------------
The bed enters through P_kV and P_eV. Side walls with log-law friction (the
``"wall_friction"`` block, ``physics.walls``) add, in each wall-adjacent cell:

- the work of the wall shear stress as production of k (the side-wall counterpart of
  the bed term c_f |U|^3 = h P_kV):

      h P_kW = sum over wall faces of u_*w^2 |u_t| h L_f / A          (m^3/s^3),

  with u_*w the log-law friction velocity, u_t the tangential velocity, L_f the face
  length and A the cell area; it equals the rate at which the wall stress removes
  mean-flow kinetic energy;
- a relaxation of epsilon towards the log-layer value
  eps_w = c_mu^(3/4) k^(3/2) / (kappa y), y the distance from the centroid to the wall,
  over one turbulence time scale k / epsilon:

      d(h eps)/dt += h (eps_w - eps) eps / k,

  whose -h eps^2 / k part is a sink of E (rate eps / k, point-implicit in SSP-RK3).

Free-slip walls (no ``wall_friction`` block) carry no turbulence sources.

Discretization
--------------
- Advection: first-order upwind with the mass flux of the HLLC scheme at each face, so
  the transport is consistent with continuity (a uniform k stays uniform).
- Boundaries: inflow faces (inlets, or reversed flow at an exit) bring the local
  equilibrium values of the boundary cell; outflow faces carry the cell values; walls
  carry nothing (zero mass flux, no diffusive flux).
- Diffusion: the two-point face formula of ``physics.turbulence``.
- Sources: cell values; the strain rate from Green-Gauss gradients, so the residual
  stencil is two cells wide (``solvers.jacobian`` widens the pattern).
- Positivity: k and epsilon are floored smoothly at k_min and eps_min inside the
  formulas. The sinks -epsilon h and -c_2 h eps^2 / k are linear in K and E with the
  rates epsilon / k and c_2 epsilon / k, which the SSP-RK3 stepper treats
  point-implicitly (as the bed friction), so K and E stay positive in explicit runs.

Units: SI (m, s); any consistent system works (u_*, n and k_n in the same units).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from hydrograd.physics.model import KIND_WALL, SWEModel
from hydrograd.physics.reconstruct import green_gauss
from hydrograd.physics.walls import wall_face_state

__all__ = [
    "bed_friction_coefficient",
    "equilibrium",
    "k_epsilon_rates",
    "turbulence_variables",
    "wall_function_terms",
]

# Differentiable parameters (none specific to k-epsilon yet).
Theta = dict[str, Any]


def _smooth_floor(x: Array, floor: float) -> Array:
    """Smooth positive part of x with a small floor: (x + sqrt(x^2 + floor^2)) / 2.

    Equal to x for x >> floor, floor / 2 at x = 0, and positive for x < 0; differentiable.

    Args:
        x: values (any units), any shape.
        floor: positive floor (same units).

    Returns:
        Positive values, same shape.
    """
    return 0.5 * (x + jnp.sqrt(x * x + floor * floor))


def bed_friction_coefficient(m: SWEModel, h: Array, manning_n: Array) -> Array:
    """Bed friction coefficient c_f = u_*^2 / |U|^2 = g n^2 / (k_n^2 h^(1/3)) (-).

    Args:
        m: the model (``g``, ``k_n``, ``h_small``).
        h: water depth (m), shape [n_cells].
        manning_n: Manning's n (s/m^(1/3)), shape [n_cells].

    Returns:
        c_f (-), shape [n_cells].
    """
    h_floor = jnp.maximum(h, m.h_small)
    return m.g * manning_n**2 / (m.k_n**2 * h_floor ** (1.0 / 3.0))


def equilibrium(m: SWEModel, h: Array, speed: Array, manning_n: Array) -> tuple[Array, Array]:
    """Local-equilibrium k and epsilon of uniform flow (P_h = 0).

    Args:
        m: the model (``turbulence`` constants, ``g``, ``k_n``, ``h_small``).
        h: water depth (m), shape [n].
        speed: depth-averaged speed |U| (m/s), shape [n].
        manning_n: Manning's n (s/m^(1/3)), shape [n].

    Returns:
        (k_eq (m^2/s^2), epsilon_eq (m^2/s^3)), each of shape [n], floored at the model's
        k_min and eps_min.
    """
    settings = m.turbulence
    h_floor = jnp.maximum(h, m.h_small)
    c_f = bed_friction_coefficient(m, h, manning_n)
    u_star_squared = c_f * speed**2

    epsilon = c_f * speed**3 / h_floor
    k = u_star_squared / (3.6 * jnp.sqrt(settings.c_mu) * c_f**0.25)
    return jnp.maximum(k, settings.k_min), jnp.maximum(epsilon, settings.eps_min)


def turbulence_variables(m: SWEModel, h: Array, U: Array) -> tuple[Array, Array, Array]:
    """k, epsilon and nu_t per cell from the state rows K = h k and E = h epsilon.

    Args:
        m: the model (``turbulence`` constants, ``h_small``).
        h: water depth (m), shape [n_cells].
        U: state [xi, q_x, q_y, K, E], shape [5, n_cells].

    Returns:
        (k (m^2/s^2), epsilon (m^2/s^3), nu_t (m^2/s)), each of shape [n_cells]; k and
        epsilon smoothly floored at k_min and eps_min.
    """
    settings = m.turbulence
    h_floor = jnp.maximum(h, m.h_small)
    k = _smooth_floor(U[3] / h_floor, settings.k_min)
    epsilon = _smooth_floor(U[4] / h_floor, settings.eps_min)
    nu_t = settings.c_mu * k * k / epsilon
    return k, epsilon, nu_t


def _upwind_transport(
    m: SWEModel,
    mass_flux: Array,
    boundary_mass_flux: Array,
    field: Array,
    inflow_value: Array,
) -> Array:
    """Net upwind advective outflow of h * field per cell (field x m^3/s).

    Args:
        m: the model (face connectivity).
        mass_flux: interior-face mass flux h u_n times the face length, from owner to
            neighbour (m^3/s), shape [n_fi].
        boundary_mass_flux: boundary-face mass flux, outward positive, times the face
            length (m^3/s), shape [n_fb].
        field: cell values of k or epsilon, shape [n_cells].
        inflow_value: value carried in by inflow boundary faces, per boundary face,
            shape [n_fb].

    Returns:
        Net outflow per cell, shape [n_cells].
    """
    owner = m.fi_o
    neighbour = m.fi_r
    n_cells = m.n_cells

    # Interior faces: the upwind cell value.
    upwind = jnp.where(mass_flux >= 0.0, field[owner], field[neighbour])
    interior_flux = mass_flux * upwind

    # Boundary faces: the cell value on outflow, the inflow value on inflow.
    boundary_value = jnp.where(boundary_mass_flux >= 0.0, field[m.fb_c], inflow_value)
    boundary_flux = boundary_mass_flux * boundary_value

    return (
        jax.ops.segment_sum(interior_flux, owner, n_cells)
        - jax.ops.segment_sum(interior_flux, neighbour, n_cells)
        + jax.ops.segment_sum(boundary_flux, m.fb_c, n_cells)
    )


def _face_diffusion(m: SWEModel, h: Array, diffusivity: Array, field: Array) -> Array:
    """Net diffusive inflow div(h D grad field) times the cell area (two-point faces).

    Args:
        m: the model (face connectivity and geometry).
        h: water depth (m), shape [n_cells].
        diffusivity: diffusivity D per cell (m^2/s), shape [n_cells].
        field: cell values, shape [n_cells].

    Returns:
        Net inflow per cell (field x m^3/s), shape [n_cells].
    """
    owner = m.fi_o
    neighbour = m.fi_r
    d_x = m.fi_ro[:, 0] - m.fi_rr[:, 0]
    d_y = m.fi_ro[:, 1] - m.fi_rr[:, 1]
    normal_distance = d_x * m.fi_nx + d_y * m.fi_ny

    h_face = jnp.minimum(h[owner], h[neighbour])
    diffusivity_face = 0.5 * (diffusivity[owner] + diffusivity[neighbour])
    flux = h_face * diffusivity_face * m.fi_L / normal_distance * (field[neighbour] - field[owner])

    return (
        jax.ops.segment_sum(flux, owner, m.n_cells)
        - jax.ops.segment_sum(flux, neighbour, m.n_cells)
    )


def wall_function_terms(
    m: SWEModel, theta: Theta, h: Array, u: Array, v: Array, k: Array, epsilon: Array,
) -> tuple[Array, Array, Array]:
    """Side-wall production of k and relaxation of epsilon in wall-adjacent cells.

    Args:
        m: the model (``wall_friction`` and ``turbulence`` settings, boundary faces).
        theta: differentiable parameters (``wall_ks``).
        h: water depth (m), shape [n_cells].
        u, v: depth-averaged velocities (m/s), shape [n_cells].
        k, epsilon: floored k (m^2/s^2) and epsilon (m^2/s^3), shape [n_cells].

    Returns:
        (k_source, epsilon_source, epsilon_sink_rate), each of shape [n_cells]:
        h P_kW (m^3/s^3) and h eps_w eps / k (m^3/s^4) per unit plan area, and the extra
        sink rate eps / k (1/s) of E in wall cells (zero elsewhere).
    """
    settings = m.turbulence
    kappa = m.wall_friction.kappa
    cell = m.fb_c
    is_wall, _, _, distance, _, speed, friction_velocity = wall_face_state(m, theta, u, v)

    # Work of the wall shear stress per unit plan area (m^3/s^3): u_*^2 |u_t| h L_f / A.
    work = is_wall * friction_velocity**2 * speed * h[cell] * m.fb_L
    k_source = jax.ops.segment_sum(work, cell, m.n_cells) / m.area

    # Log-layer dissipation at the wall-cell centroid, averaged over the cell's wall faces.
    epsilon_wall_face = settings.c_mu**0.75 * k[cell] ** 1.5 / (kappa * distance)
    wall_faces = jax.ops.segment_sum(is_wall, cell, m.n_cells)
    epsilon_wall_sum = jax.ops.segment_sum(is_wall * epsilon_wall_face, cell, m.n_cells)
    is_wall_cell = wall_faces > 0.0
    epsilon_wall = epsilon_wall_sum / jnp.maximum(wall_faces, 1.0)

    # Relaxation h (eps_w - eps) eps / k: source h eps_w eps / k, sink rate eps / k.
    rate = jnp.where(is_wall_cell, epsilon / k, 0.0)
    epsilon_source = h * epsilon_wall * rate
    return k_source, epsilon_source, rate


def k_epsilon_rates(
    m: SWEModel,
    theta: Theta,
    h: Array,
    u: Array,
    v: Array,
    U: Array,
    manning_n: Array,
    mass_flux: Array,
    boundary_mass_flux: Array,
) -> tuple[Array, Array, Array]:
    """Rates of change of K = h k and E = h epsilon, their sink rates, and nu_t.

    Args:
        m: the model (``turbulence`` constants, mesh).
        theta: differentiable parameters (``wall_ks`` for the wall functions).
        h: water depth (m), shape [n_cells].
        u, v: depth-averaged velocities (m/s), shape [n_cells].
        U: state [xi, q_x, q_y, K, E], shape [5, n_cells].
        manning_n: Manning's n (s/m^(1/3)), shape [n_cells].
        mass_flux: interior-face mass flux times face length (m^3/s), owner to
            neighbour, shape [n_fi].
        boundary_mass_flux: boundary-face mass flux times face length, outward
            positive (m^3/s), shape [n_fb].

    Returns:
        (rates, sink_rates, nu_t):
        rates: dK/dt and dE/dt without the sinks (m^3/s^3 and m^3/s^4), shape [2, n_cells];
        sink_rates: the sinks are -sink_rates * [K, E], with the rates epsilon / k and
            c_2 epsilon / k (1/s), shape [2, n_cells];
        nu_t: eddy viscosity (m^2/s), shape [n_cells].
    """
    settings = m.turbulence
    k, epsilon, nu_t = turbulence_variables(m, h, U)
    h_floor = jnp.maximum(h, m.h_small)

    # 1. Advection (upwind with the scheme's mass fluxes); inflow brings equilibrium values.
    speed = jnp.sqrt(u * u + v * v + 1.0e-12)
    k_eq, epsilon_eq = equilibrium(m, h, speed, manning_n)
    bcell = m.fb_c
    is_wall = m.fb_kind == KIND_WALL
    boundary_flux = jnp.where(is_wall, 0.0, boundary_mass_flux)
    advection_k = _upwind_transport(m, mass_flux, boundary_flux, k, k_eq[bcell])
    advection_e = _upwind_transport(m, mass_flux, boundary_flux, epsilon, epsilon_eq[bcell])

    # 2. Diffusion with nu_t / sigma.
    diffusion_k = _face_diffusion(m, h, nu_t / settings.sigma_k, k)
    diffusion_e = _face_diffusion(m, h, nu_t / settings.sigma_e, epsilon)

    # 3. Production by the horizontal shear: P_h = nu_t |S|^2 (m^2/s^3).
    grad_u = green_gauss(m, u)
    grad_v = green_gauss(m, v)
    strain_squared = (
        2.0 * grad_u[:, 0] ** 2
        + 2.0 * grad_v[:, 1] ** 2
        + (grad_u[:, 1] + grad_v[:, 0]) ** 2
    )
    production_h = nu_t * strain_squared

    # 4. Production by the bed shear: P_kV = c_f |U|^3 / h, P_eV (see the module docstring).
    c_f = bed_friction_coefficient(m, h, manning_n)
    production_kv = c_f * speed**3 / h_floor
    production_ev = (
        3.6 * settings.c_2 * jnp.sqrt(settings.c_mu) * c_f**1.25 * speed**4 / h_floor**2
    )

    # 5. Assemble per unit plan area: d(hk)/dt and d(h eps)/dt without the sinks.
    source_k = h * (production_h + production_kv)
    source_e = h * (settings.c_1 * epsilon / k * production_h + production_ev)
    rate_k = (diffusion_k - advection_k) / m.area + source_k
    rate_e = (diffusion_e - advection_e) / m.area + source_e

    # Sinks -h eps = -(eps / k) K and -c_2 h eps^2 / k = -(c_2 eps / k) E.
    sink_k = epsilon / k
    sink_e = settings.c_2 * epsilon / k

    # Side-wall functions (only with log-law wall friction; free-slip walls add nothing).
    if m.wall_friction.mode != "none":
        wall_k, wall_e, wall_sink = wall_function_terms(m, theta, h, u, v, k, epsilon)
        rate_k = rate_k + wall_k
        rate_e = rate_e + wall_e
        sink_e = sink_e + wall_sink

    sink_rates = jnp.stack([sink_k, sink_e])
    return jnp.stack([rate_k, rate_e]), sink_rates, nu_t
