"""Semi-discrete residual dQ/dt of the 2D SWEs (port of ``swe_2d_rhs``).

State ``U`` has shape [3, n_cells]: rows ``xi`` (m), ``q_x``, ``q_y`` (m^2/s).
Hydrograd.jl's flat state ``vcat(xi, q_x, q_y)`` is ``U.reshape(-1)``.

Steps, exactly as in Julia (``semi_discretize_swe_2D.jl``, ``bc_2D.jl``):

1. ``h = xi + h_still``; where ``h <= h_small``: ``h = h_small``, ``q = 0``.
2. Resistance: Manning's n per cell (zone / law / NN), or NN friction.
3. Ghost states on boundary faces (inlet-Q, exit-H, wall, symmetry).
4. Roe fluxes on all faces; ``-(1/A) sum F.n L``.
5. Sources where ``h > h_small``: ``g xi S0 - friction``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from hydrograd.bed import bed_slope
from hydrograd.closures.mlp import apply_mlp, normalize
from hydrograd.params.manning import manning_law
from hydrograd.physics.model import (
    KIND_EXIT,
    KIND_INLET,
    KIND_SYMM,
    KIND_WALL,
    SWEModel,
)
from hydrograd.physics.roe import roe_flux, smooth_sqrt

__all__ = ["boundary_values_at", "clipped_state", "friction", "inlet_conveyance", "manning_n_cells", "rhs", "rhs_julia", "still_depth"]

Theta = dict[str, Any]


def _safe_speed(qx: Array, qy: Array, h: Array) -> Array:
    """|U| = sqrt((qx/h)^2 + (qy/h)^2) with a NaN-free derivative at U = 0."""
    s = (qx / h) ** 2 + (qy / h) ** 2
    pos = s > 0.0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, s, 1.0)), 0.0)


def still_depth(m: SWEModel, theta: Theta) -> Array:
    """Still-water depth h_still (m).

    With ``zb`` as a parameter, ``h_still = w_still - zb`` so that the depth
    ``h = xi + h_still`` follows the bed being estimated. (Deviation from
    Hydrograd.jl, which keeps h_still from zb = 0 during bathymetry inversion,
    so its fluxes and friction see the depth above datum instead of above the bed.)
    """
    return m.wstill - theta["zb"] if "zb" in theta else m.hstill


def clipped_state(m: SWEModel, U: Array, hstill: Array | None = None) -> tuple[Array, Array, Array, Array]:
    """(xi, h, q_x, q_y) after the dry-cell clipping of step 1."""
    xi, qx, qy = U[0], U[1], U[2]
    h = xi + (m.hstill if hstill is None else hstill)
    dry = h <= m.h_small
    return xi, jnp.where(dry, m.h_small, h), jnp.where(dry, 0.0, qx), jnp.where(dry, 0.0, qy)


def manning_n_cells(m: SWEModel, theta: Theta, h: Array, qx: Array, qy: Array) -> Array:
    """Manning's n per cell (s/m^(1/3)); zone values for ``nn_FlowResistance``."""
    r = m.resistance
    if r.mode == "law":
        assert r.law is not None
        return manning_law(r.law, h, _safe_speed(qx, qy, h), m.ks)[0]
    if r.mode == "nn_ManningN_h":
        x = normalize(h, *r.h_bounds)[:, None]
        return apply_mlp(theta["nn"], x, r.activations, *r.out_bounds, r.ln_axes)[:, 0]
    if r.mode == "nn_ManningN_h_Umag_ks":
        x = jnp.stack([normalize(h, *r.h_bounds), normalize(_safe_speed(qx, qy, h), *r.umag_bounds),
                       normalize(m.ks, *r.ks_bounds)], axis=1)
        return apply_mlp(theta["nn"], x, r.activations, *r.out_bounds, r.ln_axes)[:, 0]
    n_zone = theta.get("n_zone", m.n_zone)
    return n_zone[m.mat_id]


def friction_coefficient(m: SWEModel, theta: Theta, h: Array, qx: Array, qy: Array, n: Array) -> Array:
    """Friction rate c (1/s) per cell, with friction = c q (Manning, or NN magnitude / |q|)."""
    q_mag = smooth_sqrt(qx * qx + qy * qy)
    if m.resistance.mode == "nn_FlowResistance":
        r = m.resistance
        x = jnp.stack([normalize(h, *r.h_bounds), normalize(q_mag, *r.q_bounds)], axis=1)
        tau = apply_mlp(theta["nn"], x, r.activations, *r.out_bounds, r.ln_axes)[:, 0]
        return tau / q_mag
    return m.g * n * n / m.k_n ** 2 / (h + m.h_small) ** (7.0 / 3.0) * q_mag


def friction(m: SWEModel, theta: Theta, h: Array, qx: Array, qy: Array, n: Array) -> tuple[Array, Array]:
    """Bed friction (m^2/s^2) per cell, x and y components (= c q)."""
    c = friction_coefficient(m, theta, h, qx, qy, n)
    return c * qx, c * qy


def _inlet_face_terms(m: SWEModel, h: Array, n: Array) -> tuple[Array, Array, Array]:
    """Per boundary face: (wet flag, conveyance K_f, velocity shape) for inlet-Q faces."""
    c, L = m.fb_c, m.fb_L
    hc, nc = h[c], n[c]
    is_in = m.fb_kind == KIND_INLET
    wet = jnp.where(is_in & (hc > m.h_small), 1.0, 0.0)
    if m.inlet_split == "julia":
        return wet, L ** (5.0 / 3.0) * hc / nc * wet, L ** (2.0 / 3.0) / nc
    hw = jnp.maximum(hc, m.h_small)  # wet faces have h > h_small; avoids d(h^(2/3))/dh = inf at h = 0
    return wet, L * hw ** (5.0 / 3.0) / nc * wet, hw ** (2.0 / 3.0) / nc


def inlet_conveyance(m: SWEModel, theta: Theta, U: Array) -> Array:
    """Total conveyance K of each inlet-Q boundary, shape [max(n_inlets, 1)].

    All inlet cells interact only through K, so dR/dU = J_local + (dR/dK)(dK/dU):
    the Jacobian tools treat that coupling as a low-rank term.
    """
    if m.scheme == "hr_hllc":
        from hydrograd.physics.rhs_hr import _cell_state
        _, h, u, v = _cell_state(m, theta, U)
        n = manning_n_cells(m, theta, jnp.maximum(h, m.h_small), h * u, h * v)
    else:
        _, h, qx, qy = clipped_state(m, U, still_depth(m, theta))
        n = manning_n_cells(m, theta, h, qx, qy)
    _, k_face, _ = _inlet_face_terms(m, h, n)
    return jax.ops.segment_sum(k_face, m.fb_inlet, max(m.n_inlets, 1))


def _ghosts(m: SWEModel, theta: Theta, zb: Array, h: Array, qx: Array, qy: Array,
            n: Array, K: Array | None = None) -> tuple[Array, Array, Array]:
    """Ghost (h, q_x, q_y) on boundary faces (``process_all_boundaries_2d``)."""
    c, nx, ny, kind = m.fb_c, m.fb_nx, m.fb_ny, m.fb_kind
    hc, qxc, qyc = h[c], qx[c], qy[c]
    is_in, is_ex = kind == KIND_INLET, kind == KIND_EXIT
    is_wall, is_sym = kind == KIND_WALL, kind == KIND_SYMM

    # Inlet-Q: split Q over wet faces by Manning conveyance K_f = L h^(5/3) / n, so that the
    # face velocity is v_n = Q/K * h^(2/3)/n and sum_f h v_n L = Q. Hydrograd.jl swaps L and h
    # (K_f = L^(5/3) h / n, v_n = Q/K * L^(2/3)/n); inlet_split = "julia" reproduces that.
    Q = theta.get("Q", m.Q)
    wet, k_face, v_shape = _inlet_face_terms(m, h, n)
    if K is None:
        K = jax.ops.segment_sum(k_face, m.fb_inlet, max(m.n_inlets, 1))
    K_f = K[m.fb_inlet]
    ok = is_in & (K_f > 0.0)  # Julia asserts K > 0; here an all-dry inlet carries no discharge
    vn = jnp.where(ok, (Q[m.fb_inlet] if m.n_inlets else 0.0) / jnp.where(ok, K_f, 1.0), 0.0) * v_shape
    qx_in, qy_in = -hc * vn * nx * wet, -hc * vn * ny * wet

    # Exit-H: prescribed WSE.
    WSE = theta.get("WSE", m.WSE)
    h_ex = jnp.maximum(m.h_small, (WSE[m.fb_exit] if m.n_exits else 0.0) - zb[c])

    # Symmetry: reflect the normal component.
    vdn = qxc * nx + qyc * ny
    hg = jnp.where(is_ex, h_ex, hc)
    qxg = jnp.where(is_in, qx_in, jnp.where(is_wall, -qxc, jnp.where(is_sym, qxc - 2.0 * vdn * nx, qxc)))
    qyg = jnp.where(is_in, qy_in, jnp.where(is_wall, -qyc, jnp.where(is_sym, qyc - 2.0 * vdn * ny, qyc)))
    return hg, qxg, qyg


def boundary_values_at(m: SWEModel, theta: Theta, t: Array | float | None) -> Theta:
    """theta with inlet Q and exit WSE evaluated at time t (s), linear in time.

    Time series (arrays [n_boundaries, n_times] on ``m.Q_times`` / ``m.WSE_times``)
    may come from the model or be parameters in theta (hydrograph inversion).
    With t = None (steady solves) the first value of each series is used.
    """
    out = dict(theta)
    for key, times in (("Q", m.Q_times), ("WSE", m.WSE_times)):
        v = theta.get(key, getattr(m, key))
        if v.ndim == 2:
            if t is None:
                out[key] = v[:, 0]
            else:
                out[key] = jax.vmap(lambda row, tt=times: jnp.interp(t, tt, row))(v)
    return out


def rhs(m: SWEModel, theta: Theta, U: Array, K: Array | None = None, t: Array | float | None = None) -> Array:
    """dU/dt [3, n_cells] with the model's scheme (``m.scheme``) at time t (s).

    ``hr_hllc`` (default): hydrostatic reconstruction + HLLC, wet/dry capable
    (:mod:`hydrograd.physics.rhs_hr`). ``julia_xi_roe``: Hydrograd.jl's xi-form Roe
    scheme (:func:`rhs_julia`), for parity with Julia; requires a wet domain.
    """
    theta = boundary_values_at(m, theta, t)
    if m.scheme == "julia_xi_roe":
        return rhs_julia(m, theta, U, K)
    if m.scheme == "hr_hllc":
        from hydrograd.physics.rhs_hr import rhs_hr
        return rhs_hr(m, theta, U, K)
    raise ValueError(f"unknown scheme '{m.scheme}'")


def rhs_julia(m: SWEModel, theta: Theta, U: Array, K: Array | None = None) -> Array:
    """dU/dt, shape [3, n_cells].

    Args:
        m: static model.
        theta: parameter overrides (``zb``, ``n_zone``, ``Q``, ``WSE``, ``nn``); may be empty.
        U: state [xi, q_x, q_y], shape [3, n_cells].
        K: optional inlet conveyances to use instead of computing them from U
            (Jacobian splitting; see :func:`inlet_conveyance`).
    """
    g, hmin = m.g, m.h_small
    hs = still_depth(m, theta)
    xi, h, qx, qy = clipped_state(m, U, hs)
    if "zb" in theta:
        zb = theta["zb"]
        S0 = bed_slope(m.bed, zb)
    else:
        zb, S0 = m.zb, m.S0
    n = manning_n_cells(m, theta, h, qx, qy)

    # Interior faces.
    o, r = m.fi_o, m.fi_r
    fi = roe_flux(xi[o], hs[o], h[o], qx[o], qy[o], zb[o],
                  xi[r], hs[r], h[r], qx[r], qy[r], zb[r], m.fi_nx, m.fi_ny, g, hmin)
    Fi = jnp.stack(fi, axis=1) * m.fi_L[:, None]

    # Boundary faces with ghost cells (zb_ghost = zb_cell, h_still_ghost = h_still_cell).
    c = m.fb_c
    hg, qxg, qyg = _ghosts(m, theta, zb, h, qx, qy, n, K)
    fb = roe_flux(xi[c], hs[c], h[c], qx[c], qy[c], zb[c],
                  hg - hs[c], hs[c], hg, qxg, qyg, zb[c], m.fb_nx, m.fb_ny, g, hmin)
    Fb = jnp.stack(fb, axis=1) * m.fb_L[:, None]

    acc = (jax.ops.segment_sum(Fi, o, m.n_cells) - jax.ops.segment_sum(Fi, r, m.n_cells)
           + jax.ops.segment_sum(Fb, c, m.n_cells))
    inviscid = -(acc / m.area[:, None]).T

    fx, fy = friction(m, theta, h, qx, qy, n)
    wet = h > hmin
    sx = jnp.where(wet, g * xi * S0[:, 0] - fx, 0.0)
    sy = jnp.where(wet, g * xi * S0[:, 1] - fy, 0.0)
    return inviscid + jnp.stack([jnp.zeros_like(sx), sx, sy])
