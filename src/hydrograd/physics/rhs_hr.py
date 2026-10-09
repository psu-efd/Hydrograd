"""Wet/dry residual: hydrostatic reconstruction (Audusse et al. 2004) with the HLLC flux.

This is the default spatial scheme of hydrograd (``SWEModel.scheme = "hr_hllc"``).

State layout
------------
The state has the same layout as the Julia-ported scheme:

    U = [xi, q_x, q_y],   shape [3, n_cells]

- ``xi``  : free-surface displacement above the still-water level (m),
- ``q_x``, ``q_y`` : unit discharges (m^2/s),
- water depth ``h = xi + h_still`` (m), where ``h_still`` is fixed in time.

Because ``h_still`` does not change, d(xi)/dt = dh/dt, so every application (forward,
inversion, sensitivity, UDE) works unchanged with this scheme.

Interior faces (first order)
----------------------------
For a face between the owner cell L and the neighbour cell R, with unit normal n
pointing from L to R:

    z_f   = max(z_L, z_R)                       face bed elevation (m)
    h_L*  = max(0, h_L + z_L - z_f)             reconstructed left depth (m)
    h_R*  = max(0, h_R + z_R - z_f)             reconstructed right depth (m)
    F     = HLLC(U_L*, U_R*),   U* = (h*, h* u)

    cell L receives   F + (0, g/2 (h_L^2 - h_L*^2) n)
    cell R receives  -F - (0, g/2 (h_R^2 - h_R*^2) n)

The pressure corrections g/2 (h^2 - h*^2) replace the bed-slope source term. As a result
the scheme:
- is well balanced for a lake at rest, including wet/dry fronts and emerged bed;
- conserves mass to round-off;
- keeps h >= 0 with first-order explicit SSP time stepping for CFL <= 1/2
  (use ``ode_solver = "SSPRK3()"`` for wet/dry runs).

Inlet faces
-----------
Inlet-Q faces carry exactly the prescribed share of the discharge: the flux is imposed
directly, not computed from a Riemann problem. The inflow volume therefore equals the
time integral of Q(t).

Second order (``SWEModel.order = 2``)
-------------------------------------
- MUSCL reconstruction of h, eta = h + z, u and v at interior faces, with a slope limiter
  (Venkatakrishnan by default, or Barth-Jespersen).
- First order at boundary faces and, through a continuous blend, next to dry cells.
- Hydrostatic reconstruction on the face states, with z_f = eta_f - h_f.
- The centred bed source of Audusse & Bristeau (2005).
- Use CFL <= 0.25 for wet/dry runs.

Smoothing option
----------------
``SWEModel.dry_smoothing = delta > 0`` replaces max(x, 0) (cell depth and reconstructed
face depths) by (x + sqrt(x^2 + delta^2)) / 2. The residual then becomes smooth, which
helps Newton in the implicit scheme at wet fronts, at the price of a water film of about
delta/2 on dry cells. Default: 0 (exact max).

Velocities near dry cells
-------------------------
Depth thresholds enter only through the velocities (Kurganov & Petrova 2007):

    u = sqrt(2) h q / sqrt(h^4 + max(h^4, eps^4)),   eps = h_small

so u -> 0 as h -> 0 without clipping depth or mass. Friction and Manning laws use the
same velocities, and floor the depth at h_small only inside the friction and closure
formulas.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from hydrograd.bed import bed_slope  # noqa: F401  (kept for API symmetry)
from hydrograd.physics.dispersion import dispersive_flux_source
from hydrograd.physics.hllc import hllc_flux
from hydrograd.physics.k_epsilon import k_epsilon_rates
from hydrograd.physics.model import KIND_INLET, SWEModel
from hydrograd.physics.rhs import (
    _ghosts,
    friction_coefficient,
    manning_n_cells,
    still_depth,
)
from hydrograd.physics.turbulence import turbulent_diffusion
from hydrograd.physics.walls import wall_shear_source

__all__ = ["desingularized_velocity", "max_eddy_viscosity", "max_wave_speed", "rhs_hr"]

# Differentiable parameters, e.g. {"n_zone": ..., "zb": ..., "Q": ..., "nn": ...}.
Theta = dict[str, Any]

SQRT2 = 1.4142135623730951


def desingularized_velocity(h: Array, q: Array, eps: float) -> Array:
    """Velocity that stays bounded as the depth goes to zero (Kurganov & Petrova 2007).

    u = sqrt(2) h q / sqrt(h^4 + max(h^4, eps^4)). For h >= eps this equals q / h exactly;
    for h < eps it tends smoothly to zero.

    Args:
        h: water depth (m), any shape.
        q: unit discharge component (m^2/s), same shape as ``h``.
        eps: depth below which the velocity is damped (m), normally ``h_small``.

    Returns:
        Velocity component (m/s), same shape as ``h``.
    """
    h4 = h ** 4
    return SQRT2 * h * q / jnp.sqrt(h4 + jnp.maximum(h4, eps ** 4))


def _pos(x: Array, delta: float) -> Array:
    """Positive part of ``x``: exact max(x, 0), or a smooth version when ``delta > 0``.

    Args:
        x: input values (m), any shape.
        delta: smoothing width (m). 0 gives the exact max(x, 0); delta > 0 gives
            (x + sqrt(x^2 + delta^2)) / 2, which is differentiable everywhere.

    Returns:
        Non-negative values (m), same shape as ``x``.
    """
    if delta > 0.0:
        return 0.5 * (x + jnp.sqrt(x * x + delta * delta))

    return jnp.maximum(x, 0.0)


def _cell_state(m: SWEModel, theta: Theta, U: Array) -> tuple[Array, Array, Array, Array]:
    """Primitive cell variables from the conserved state.

    Args:
        m: the model (mesh, bed, settings).
        theta: differentiable parameters (``theta["zb"]`` changes the still-water depth).
        U: state [xi, q_x, q_y], shape [3, n_cells].

    Returns:
        (h_still, h, u, v), each of shape [n_cells]:
        still-water depth (m), water depth (m), and velocities u, v (m/s).
    """
    h_still = still_depth(m, theta)

    # Depth from the free-surface displacement; clipped (or smoothly clipped) at zero.
    h = _pos(U[0] + h_still, m.dry_smoothing)

    # Velocities that vanish smoothly in (nearly) dry cells.
    u = desingularized_velocity(h, U[1], m.h_small)
    v = desingularized_velocity(h, U[2], m.h_small)

    return h_still, h, u, v


def _face_flux(
    hL: Array, uL: Array, vL: Array,
    hR: Array, uR: Array, vR: Array,
    nx: Array, ny: Array, g: float,
) -> tuple[Array, Array, Array, Array]:
    """HLLC flux across faces, returned in global x/y components.

    The HLLC solver works in the face frame (normal and tangential velocity). This
    function rotates the velocities into that frame, calls the solver, and rotates the
    momentum flux back to x/y.

    Args:
        hL, uL, vL: left depth (m) and velocities (m/s), shape [n_faces].
        hR, uR, vR: right depth (m) and velocities (m/s), shape [n_faces].
        nx, ny: components of the unit face normal, pointing from left to right.
        g: gravitational acceleration (m/s^2).

    Returns:
        (F_h, F_x, F_y, s_max), each of shape [n_faces]: mass flux (m^2/s), x- and
        y-momentum flux (m^3/s^2) per unit face length, and the largest wave speed (m/s).
    """
    # Rotate velocities into the face frame: normal (un) and tangential (ut) components.
    unL = uL * nx + vL * ny
    utL = -uL * ny + vL * nx
    unR = uR * nx + vR * ny
    utR = -uR * ny + vR * nx

    # Riemann solve in the face frame.
    F_h, F_n, F_t, s_max = hllc_flux(hL, unL, utL, hR, unR, utR, g)

    # Rotate the momentum flux back to global x/y components.
    F_x = F_n * nx - F_t * ny
    F_y = F_n * ny + F_t * nx

    return F_h, F_x, F_y, s_max


def _muscl_states(
    m: SWEModel, h: Array, zb: Array, u: Array, v: Array,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    """Second-order (limited MUSCL) face states on both sides of each interior face.

    The free surface eta = h + z is reconstructed instead of the bed z, so a lake at rest
    stays exactly flat. Near dry cells the reconstruction blends continuously to first
    order (see the weight ``w`` below).

    Args:
        m: the model (mesh connectivity and limiter choice).
        h: cell water depth (m), shape [n_cells].
        zb: cell bed elevation (m), shape [n_cells].
        u, v: cell velocities (m/s), shape [n_cells].

    Returns:
        (hL, hR, zL, zR, uL, uR, vL, vR), each of shape [n_interior_faces]: depth (m),
        bed elevation (m) and velocities (m/s) on the owner (L) and neighbour (R) side.
    """
    from hydrograd.physics.reconstruct import limited_face_values

    n_cells = m.n_cells

    # Smallest depth among each cell and its face neighbours (both face orientations).
    h_min_owner_side = jax.ops.segment_min(h[m.fi_r], m.fi_o, n_cells)
    h_min_neighbour_side = jax.ops.segment_min(h[m.fi_o], m.fi_r, n_cells)
    h_min = jnp.minimum(h, jnp.minimum(h_min_owner_side, h_min_neighbour_side))

    # Blend weight w: 0 (pure first order) at h_min <= h_small, rising linearly to 1
    # (full second order) at h_min >= 10 h_small. Dry neighbours still give pure first
    # order, which preserves positivity, but the residual stays continuous in the state.
    # Newton (implicit and steady solves) needs that continuity at moving wet fronts.
    w = jnp.clip((h_min - m.h_small) / (9.0 * m.h_small), 0.0, 1.0)

    # Reconstruct the four fields in one batched pass (gradients, neighbour ranges and
    # limiters are vectorized over the field axis).
    fields = jnp.stack([h, h + zb, u, v])

    def reconstruct(q: Array) -> tuple[Array, Array]:
        return limited_face_values(m, q, w, m.limiter)

    face_left, face_right = jax.vmap(reconstruct)(fields)
    hL, etaL, uL, vL = face_left
    hR, etaR, uR, vR = face_right

    # The Venkatakrishnan limiter may overshoot local bounds slightly: keep depths >= 0.
    hL = jnp.maximum(hL, 0.0)
    hR = jnp.maximum(hR, 0.0)

    # Bed elevation on each side follows from the reconstructed free surface.
    zL = etaL - hL
    zR = etaR - hR

    return hL, hR, zL, zR, uL, uR, vL, vR


def rhs_hr(
    m: SWEModel,
    theta: Theta,
    U: Array,
    K: Array | None = None,
    split: bool = False,
) -> Array | tuple[Array, Array]:
    """Semi-discrete right-hand side dU/dt of the SWEs (see the module docstring).

    Args:
        m: the model (mesh, bed, boundary conditions, scheme settings).
        theta: differentiable parameters (Manning's n, bed, inflow, NN weights, ...).
        U: state [xi, q_x, q_y], shape [3, n_cells]; with the k-epsilon model
            [xi, q_x, q_y, h k, h epsilon], shape [5, n_cells].
        K: inlet conveyance weights for the discharge split, or None to compute them.
        split: if True, return the linear sinks separately (for point-implicit stepping).

    Returns:
        If ``split`` is False: dU/dt, shape [n_rows, n_cells] (units of U per second).

        If ``split`` is True: a tuple ``(dU_flux, c)`` where ``dU_flux`` is dU/dt without
        the sinks, and ``c`` the sink rate of every row (1/s), shape [n_rows, n_cells]:
        0 for xi, the friction rate c_f for q_x and q_y, and epsilon / k and
        c_2 epsilon / k for h k and h epsilon. The sinks are -c * U, which the SSP-RK
        stepper treats point-implicitly.
    """
    g = m.g

    # ------------------------------------------------------------------
    # 1. Cell variables
    # ------------------------------------------------------------------
    _, h, u, v = _cell_state(m, theta, U)
    zb = theta["zb"] if "zb" in theta else m.zb

    # Depth floored at h_small; used only inside the friction and closure formulas.
    h_floor = jnp.maximum(h, m.h_small)

    qx = h * u
    qy = h * v

    # Manning's n per cell (constant per zone, a depth law, or the NN closure).
    manning_n = manning_n_cells(m, theta, h_floor, qx, qy)

    # ------------------------------------------------------------------
    # 2. Interior faces
    # ------------------------------------------------------------------
    owner = m.fi_o
    neighbour = m.fi_r

    # Face states: cell values at first order, limited MUSCL values at second order.
    if m.order == 2:
        hL, hR, zL, zR, uL, uR, vL, vR = _muscl_states(m, h, zb, u, v)
    else:
        hL = h[owner]
        hR = h[neighbour]
        zL = zb[owner]
        zR = zb[neighbour]
        uL = u[owner]
        uR = u[neighbour]
        vL = v[owner]
        vR = v[neighbour]

    # Hydrostatic reconstruction: depths measured above the higher of the two beds.
    z_face = jnp.maximum(zL, zR)
    hL_star = _pos(hL + zL - z_face, m.dry_smoothing)
    hR_star = _pos(hR + zR - z_face, m.dry_smoothing)

    # Riemann flux between the reconstructed states.
    F_h, F_x, F_y, _ = _face_flux(hL_star, uL, vL, hR_star, uR, vR, m.fi_nx, m.fi_ny, g)

    # Pressure corrections g/2 (h^2 - h*^2) on each side; they replace the bed-slope source.
    corr_L = 0.5 * g * (hL ** 2 - hL_star ** 2)
    corr_R = 0.5 * g * (hR ** 2 - hR_star ** 2)

    if m.order == 2:
        # Centred bed source -g sum_f (h_i + h_if)/2 (z_if - z_i) n L (Audusse & Bristeau
        # 2005), written as an outward "flux" term. At rest it balances the face terms
        # exactly.
        corr_L = corr_L + 0.5 * g * (h[owner] + hL) * (zL - zb[owner])
        corr_R = corr_R + 0.5 * g * (h[neighbour] + hR) * (zR - zb[neighbour])

    # Outward face fluxes (times face length) for the owner and the neighbour cell.
    face_length = m.fi_L
    flux_owner = jnp.stack(
        [
            F_h * face_length,
            (F_x + corr_L * m.fi_nx) * face_length,
            (F_y + corr_L * m.fi_ny) * face_length,
        ],
        axis=1,
    )
    flux_neighbour = jnp.stack(
        [
            -F_h * face_length,
            (-F_x - corr_R * m.fi_nx) * face_length,
            (-F_y - corr_R * m.fi_ny) * face_length,
        ],
        axis=1,
    )

    # ------------------------------------------------------------------
    # 3. Boundary faces
    # ------------------------------------------------------------------
    # Ghost states from the boundary conditions. The ghost bed equals the cell bed, so
    # no hydrostatic reconstruction is needed.
    bcell = m.fb_c
    h_ghost, qx_ghost, qy_ghost = _ghosts(m, theta, zb, h, qx, qy, manning_n, K)
    h_ghost = jnp.maximum(h_ghost, 0.0)
    u_ghost = desingularized_velocity(h_ghost, qx_ghost, m.h_small)
    v_ghost = desingularized_velocity(h_ghost, qy_ghost, m.h_small)

    G_h, G_x, G_y, _ = _face_flux(
        h[bcell], u[bcell], v[bcell], h_ghost, u_ghost, v_ghost, m.fb_nx, m.fb_ny, g
    )

    # Inlet-Q faces: impose the face flux directly, so the admitted discharge is exactly
    # the prescribed share of Q(t). A Riemann flux against the ghost state would only
    # approximate it.
    #   mass flux      q_n = h_g u_n             (u_n < 0 for inflow)
    #   momentum flux  q_n u_n + g h_c^2 / 2     along the normal
    is_inlet = m.fb_kind == KIND_INLET
    qn_inlet = qx_ghost * m.fb_nx + qy_ghost * m.fb_ny      # = h_g u_n (m^2/s)
    h_inlet = jnp.maximum(h_ghost, m.h_small)
    momentum_inlet = qn_inlet * qn_inlet / h_inlet + 0.5 * g * h[bcell] ** 2

    G_h = jnp.where(is_inlet, qn_inlet, G_h)
    G_x = jnp.where(is_inlet, momentum_inlet * m.fb_nx, G_x)
    G_y = jnp.where(is_inlet, momentum_inlet * m.fb_ny, G_y)

    flux_boundary = jnp.stack([G_h, G_x, G_y], axis=1) * m.fb_L[:, None]

    # ------------------------------------------------------------------
    # 4. Assemble: sum the face fluxes per cell and divide by the cell area
    # ------------------------------------------------------------------
    net_outflow = (
        jax.ops.segment_sum(flux_owner, owner, m.n_cells)
        + jax.ops.segment_sum(flux_neighbour, neighbour, m.n_cells)
        + jax.ops.segment_sum(flux_boundary, bcell, m.n_cells)
    )
    dU = -(net_outflow / m.area[:, None]).T

    # Optional k-epsilon transport (rows 3 and 4); its nu_t drives the momentum diffusion.
    k_epsilon = m.turbulence.mode == "k_epsilon"
    if k_epsilon:
        turbulence_rates, turbulence_sinks, nu_ke = k_epsilon_rates(
            m, h, u, v, U, manning_n, F_h * face_length, G_h * m.fb_L
        )

    # Optional horizontal turbulent diffusion div(h nu_t grad u) (explicit part of the
    # rate; models without a turbulence block skip it and stay bit-identical).
    if k_epsilon:
        nu_momentum = nu_ke + m.turbulence.nu0
        dU = dU + turbulent_diffusion(m, theta, h, u, v, manning_n, nu_momentum)
    elif m.turbulence.mode != "none":
        dU = dU + turbulent_diffusion(m, theta, h, u, v, manning_n)

    # Optional shear stress of the side walls (log-law wall function); without it WALL
    # faces are free-slip.
    if m.wall_friction.mode != "none":
        dU = dU + wall_shear_source(m, theta, h, u, v)

    # Optional dispersive momentum flux (beta - 1) h U U from sub-cell velocity
    # variations (momentum coefficient beta > 1, e.g. inside stem patches).
    if m.dispersion.mode != "none":
        dU = dU + dispersive_flux_source(m, theta, h, u, v)

    # ------------------------------------------------------------------
    # 5. Bed friction: -c_f q in the momentum equations
    # ------------------------------------------------------------------
    c_f = friction_coefficient(m, theta, h_floor, qx, qy, manning_n)

    # Sink rate of every row (1/s): none for xi, friction for the discharges, and the
    # dissipation sinks of h k and h epsilon.
    sink_rates = jnp.stack([jnp.zeros_like(c_f), c_f, c_f])
    if k_epsilon:
        dU = jnp.concatenate([dU, turbulence_rates])
        sink_rates = jnp.concatenate([sink_rates, turbulence_sinks])

    if split:
        return dU, sink_rates

    if not k_epsilon:
        friction = jnp.stack([jnp.zeros_like(c_f), c_f * qx, c_f * qy])
        return dU - friction
    return dU - sink_rates * jnp.concatenate([jnp.stack([U[0], qx, qy]), U[3:]])


def max_eddy_viscosity(m: SWEModel, theta: Theta, U: Array) -> Array:
    """Largest eddy viscosity nu_t over all cells (m^2/s), for the explicit diffusion limit.

    Args:
        m: the model (turbulence settings).
        theta: differentiable parameters.
        U: state, shape [n_rows, n_cells].

    Returns:
        Scalar maximum nu_t (m^2/s); 0 without turbulent diffusion.
    """
    from hydrograd.physics.k_epsilon import turbulence_variables
    from hydrograd.physics.turbulence import eddy_viscosity

    if m.turbulence.mode == "none":
        return jnp.asarray(0.0)

    _, h, u, v = _cell_state(m, theta, U)
    if m.turbulence.mode == "k_epsilon":
        nu_t = turbulence_variables(m, h, U)[2] + m.turbulence.nu0
    else:
        h_floor = jnp.maximum(h, m.h_small)
        manning_n = manning_n_cells(m, theta, h_floor, h * u, h * v)
        nu_t = eddy_viscosity(m, theta, h, u, v, manning_n)
    return jnp.max(nu_t)


def max_wave_speed(m: SWEModel, theta: Theta, U: Array) -> Array:
    """Largest wave speed |u| + sqrt(g h) over all cells, for CFL time-step control.

    Args:
        m: the model.
        theta: differentiable parameters.
        U: state [xi, q_x, q_y], shape [3, n_cells].

    Returns:
        Scalar maximum wave speed (m/s).
    """
    _, h, u, v = _cell_state(m, theta, U)
    speed = jnp.sqrt(u * u + v * v) + jnp.sqrt(m.g * h)

    return jnp.max(speed)
