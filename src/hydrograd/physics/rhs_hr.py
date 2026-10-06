"""Wet/dry residual: hydrostatic reconstruction (Audusse et al. 2004) + HLLC.

Default scheme of hydrograd (``SWEModel.scheme = "hr_hllc"``). Same state layout
as the Julia-ported scheme, U = [xi, q_x, q_y] with h = xi + h_still (h_still fixed),
so dxi/dt = dh/dt and all applications are unchanged.

Per interior face (owner L, neighbour R, unit normal n from L to R):

    z_f = max(z_L, z_R),   h_L* = max(0, h_L + z_L - z_f),   h_R* = max(0, h_R + z_R - z_f)
    F   = HLLC(U_L*, U_R*),  U* = (h*, h* u)
    cell L:  F + (0, g/2 (h_L^2 - h_L*^2) n);   cell R: -F - (0, g/2 (h_R^2 - h_R*^2) n)

Inlet-Q faces carry exactly the prescribed discharge share (flux imposed directly,
not through a Riemann problem), so the inflow volume equals the integral of Q(t).

The pressure corrections replace the bed-slope source, so the scheme is well
balanced for lake at rest *including* wet/dry fronts and emerged bed, conserves
mass to round-off, and keeps h >= 0 with first-order explicit SSP time stepping
under CFL <= 1/2 (use ``ode_solver = "SSPRK3()"`` for wet/dry runs).

Second order (``SWEModel.order = 2``): MUSCL reconstruction of h, eta = h + z, u, v with
limiting (Venkatakrishnan, smooth, default; or Barth-Jespersen) at interior faces (first order next to dry cells and at boundary
faces), hydrostatic reconstruction on the face states (z_f = eta_f - h_f), and the centred
bed source of Audusse & Bristeau (2005). Use cfl <= 0.25 for wet/dry runs.

``SWEModel.dry_smoothing`` = delta > 0 replaces the state-dependent max(., 0) (cell depth and
reconstructed face depths) by (x + sqrt(x^2 + delta^2))/2: G becomes smooth for Newton in the
implicit scheme at wet fronts, at the price of a film of ~delta/2 on dry cells (default 0).

Depth thresholds enter only through velocities: u = sqrt(2) h q / sqrt(h^4 +
max(h^4, eps^4)) (Kurganov & Petrova 2007) with eps = h_small, so u -> 0 as h -> 0
without clipping depth or mass. Friction and Manning laws use the same velocities,
with depth floored at h_small inside the friction/closure formulas only.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from hydrograd.bed import bed_slope  # noqa: F401  (kept for API symmetry)
from hydrograd.physics.hllc import hllc_flux
from hydrograd.physics.model import KIND_INLET, SWEModel
from hydrograd.physics.rhs import _ghosts, friction_coefficient, manning_n_cells, still_depth

__all__ = ["desingularized_velocity", "max_wave_speed", "rhs_hr"]

Theta = dict[str, Any]
SQRT2 = 1.4142135623730951


def desingularized_velocity(h: Array, q: Array, eps: float) -> Array:
    """u = sqrt(2) h q / sqrt(h^4 + max(h^4, eps^4)) (m/s); equals q/h for h >= eps."""
    h4 = h ** 4
    return SQRT2 * h * q / jnp.sqrt(h4 + jnp.maximum(h4, eps ** 4))


def _pos(x: Array, delta: float) -> Array:
    """max(x, 0), or its smooth version (x + sqrt(x^2 + delta^2))/2 when delta > 0."""
    if delta > 0.0:
        return 0.5 * (x + jnp.sqrt(x * x + delta * delta))
    return jnp.maximum(x, 0.0)


def _cell_state(m: SWEModel, theta: Theta, U: Array):
    hs = still_depth(m, theta)
    h = _pos(U[0] + hs, m.dry_smoothing)
    u = desingularized_velocity(h, U[1], m.h_small)
    v = desingularized_velocity(h, U[2], m.h_small)
    return hs, h, u, v


def _face_flux(hL, uL, vL, hR, uR, vR, nx, ny, g):
    """HLLC flux in x/y components plus the maximum wave speed."""
    unL, utL = uL * nx + vL * ny, -uL * ny + vL * nx
    unR, utR = uR * nx + vR * ny, -uR * ny + vR * nx
    Fh, Fn, Ft, s = hllc_flux(hL, unL, utL, hR, unR, utR, g)
    return Fh, Fn * nx - Ft * ny, Fn * ny + Ft * nx, s


def _muscl_states(m: SWEModel, h: Array, zb: Array, u: Array, v: Array):
    """Limited face states (owner side, neighbour side) of h, z = eta - h, u, v.

    The free surface eta = h + z (not z) is reconstructed, so a lake at rest stays flat.
    Cells near dry cells blend continuously to first order (see below).
    """
    from hydrograd.physics.reconstruct import limited_face_values

    n = m.n_cells
    # Continuous blend to first order near dry cells: weight w = ramp of the smallest depth
    # among the cell and its face neighbours, 0 at h <= h_small, 1 at h >= 10 h_small. Dry
    # neighbours still give pure first order (positivity), but the residual stays continuous,
    # which Newton (implicit and steady solves) needs at moving wet fronts.
    hmin = jnp.minimum(h, jnp.minimum(jax.ops.segment_min(h[m.fi_r], m.fi_o, n),
                                      jax.ops.segment_min(h[m.fi_o], m.fi_r, n)))
    w = jnp.clip((hmin - m.h_small) / (9.0 * m.h_small), 0.0, 1.0)
    # One batched pass for the four fields (gradients, neighbour ranges and limiters vectorized).
    Q = jnp.stack([h, h + zb, u, v])
    QL, QR = jax.vmap(lambda q: limited_face_values(m, q, w, m.limiter))(Q)
    hL, eL, uL, vL = QL
    hR, eR, uR, vR = QR
    # Venkatakrishnan may overshoot local bounds slightly: keep reconstructed depths >= 0.
    hL, hR = jnp.maximum(hL, 0.0), jnp.maximum(hR, 0.0)
    return hL, hR, eL - hL, eR - hR, uL, uR, vL, vR


def rhs_hr(m: SWEModel, theta: Theta, U: Array, K: Array | None = None, split: bool = False):
    """dU/dt [3, n_cells] with hydrostatic reconstruction + HLLC (see module docstring).

    With ``split=True`` returns (flux + bed part [3, n_cells], friction rate c [n_cells] (1/s)),
    friction being -c q, for point-implicit friction in the SSP-RK stepper.
    """
    g = m.g
    hs, h, u, v = _cell_state(m, theta, U)
    zb = theta["zb"] if "zb" in theta else m.zb
    hf = jnp.maximum(h, m.h_small)           # depth floor for friction / closures only
    qx, qy = h * u, h * v
    n = manning_n_cells(m, theta, hf, qx, qy)

    # Interior faces: face states (first order: cell values; second order: limited MUSCL).
    o, r = m.fi_o, m.fi_r
    if m.order == 2:
        hL_, hR_, zL, zR, uL, uR, vL, vR = _muscl_states(m, h, zb, u, v)
    else:
        hL_, hR_, zL, zR, uL, uR, vL, vR = h[o], h[r], zb[o], zb[r], u[o], u[r], v[o], v[r]
    zf = jnp.maximum(zL, zR)
    hLs = _pos(hL_ + zL - zf, m.dry_smoothing)
    hRs = _pos(hR_ + zR - zf, m.dry_smoothing)
    Fh, Fx, Fy, _ = _face_flux(hLs, uL, vL, hRs, uR, vR, m.fi_nx, m.fi_ny, g)
    cL = 0.5 * g * (hL_ ** 2 - hLs ** 2)
    cR = 0.5 * g * (hR_ ** 2 - hRs ** 2)
    if m.order == 2:
        # Centred bed source -g sum_f (h_i + h_if)/2 (z_if - z_i) n L (Audusse & Bristeau 2005),
        # written as an outward "flux" term; it balances the face terms exactly at rest.
        cL = cL + 0.5 * g * (h[o] + hL_) * (zL - zb[o])
        cR = cR + 0.5 * g * (h[r] + hR_) * (zR - zb[r])
    L = m.fi_L
    own = jnp.stack([Fh * L, (Fx + cL * m.fi_nx) * L, (Fy + cL * m.fi_ny) * L], axis=1)
    nbr = jnp.stack([-Fh * L, (-Fx - cR * m.fi_nx) * L, (-Fy - cR * m.fi_ny) * L], axis=1)

    # Boundary faces: ghost states (z_ghost = z_cell, so no reconstruction is needed).
    c = m.fb_c
    hg, qxg, qyg = _ghosts(m, theta, zb, h, qx, qy, n, K)
    hg = jnp.maximum(hg, 0.0)
    ug = desingularized_velocity(hg, qxg, m.h_small)
    vg = desingularized_velocity(hg, qyg, m.h_small)
    Gh, Gx, Gy, _ = _face_flux(h[c], u[c], v[c], hg, ug, vg, m.fb_nx, m.fb_ny, g)
    # Inlet-Q faces: impose the face flux directly so the admitted discharge is exactly the
    # prescribed share of Q(t) (a Riemann flux against the ghost state only approximates it):
    # mass q_n = h_g u_n (u_n < 0 inflow), momentum q_n u_n + g h_c^2 / 2 along n.
    is_in = m.fb_kind == KIND_INLET
    un_in = qxg * m.fb_nx + qyg * m.fb_ny                     # = h_g * u_n (m^2/s)
    hb = jnp.maximum(hg, m.h_small)
    Fn_in = un_in * un_in / hb + 0.5 * g * h[c] ** 2
    Gh = jnp.where(is_in, un_in, Gh)
    Gx = jnp.where(is_in, Fn_in * m.fb_nx, Gx)
    Gy = jnp.where(is_in, Fn_in * m.fb_ny, Gy)
    bnd = jnp.stack([Gh, Gx, Gy], axis=1) * m.fb_L[:, None]

    acc = (jax.ops.segment_sum(own, o, m.n_cells) + jax.ops.segment_sum(nbr, r, m.n_cells)
           + jax.ops.segment_sum(bnd, c, m.n_cells))
    dU = -(acc / m.area[:, None]).T
    c_f = friction_coefficient(m, theta, hf, qx, qy, n)
    if split:
        return dU, c_f
    return dU - jnp.stack([jnp.zeros_like(c_f), c_f * qx, c_f * qy])


def max_wave_speed(m: SWEModel, theta: Theta, U: Array) -> Array:
    """max over cells of |u| + sqrt(g h) (m/s), for CFL time steps."""
    _, h, u, v = _cell_state(m, theta, U)
    return jnp.max(jnp.sqrt(u * u + v * v) + jnp.sqrt(m.g * h))
