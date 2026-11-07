"""Semi-discrete residual dQ/dt of the 2D SWEs (port of ``swe_2d_rhs``), plus shared pieces.

This module holds

- the scheme dispatcher :func:`rhs` (``hr_hllc`` in :mod:`hydrograd.physics.rhs_hr`, or
  ``julia_xi_roe`` here);
- Hydrograd.jl's xi-form Roe residual :func:`rhs_julia`;
- pieces shared by both schemes: still-water depth, Manning's n per cell, bed friction,
  inlet-Q discharge split and boundary ghost states, time-varying boundary values.

State layout
------------
State ``U`` has shape [3, n_cells]: rows ``xi`` (m), ``q_x``, ``q_y`` (m^2/s), where
``xi = h - h_still`` is the free-surface displacement above the still-water level and
q = h u the unit discharge. Hydrograd.jl's flat state ``vcat(xi, q_x, q_y)`` is
``U.reshape(-1)``.

Steps of ``rhs_julia``
----------------------
Exactly as in Julia (``semi_discretize_swe_2D.jl``, ``bc_2D.jl``):

1. ``h = xi + h_still``; where ``h <= h_small``: ``h = h_small``, ``q = 0``.
2. Resistance: Manning's n per cell (zone / law / NN), or NN friction.
3. Ghost states on boundary faces (inlet-Q, exit-H, wall, symmetry).
4. Roe fluxes on all faces; ``-(1/A) sum F.n L``.
5. Sources where ``h > h_small``: ``g xi S0 - friction``.

Bed friction
------------
Manning friction per unit mass in the momentum equations (m^2/s^2):

    friction = c_f q,   c_f = g n^2 |q| / (k_n^2 (h + h_small)^(7/3))   (1/s)

with n Manning's coefficient (s/m^(1/3)) and k_n the unit factor (1 in SI).
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
from hydrograd.physics.porous import porous_drag_rate
from hydrograd.physics.roe import roe_flux, smooth_sqrt
from hydrograd.physics.vegetation import vegetation_drag_rate

__all__ = [
    "boundary_values_at",
    "clipped_state",
    "friction",
    "inlet_conveyance",
    "manning_n_cells",
    "rhs",
    "rhs_julia",
    "still_depth",
]

# Differentiable parameters, e.g. {"n_zone": ..., "zb": ..., "Q": ..., "nn": ...}.
Theta = dict[str, Any]


def _safe_speed(qx: Array, qy: Array, h: Array) -> Array:
    """|U| = sqrt((qx/h)^2 + (qy/h)^2) with a NaN-free derivative at U = 0.

    Args:
        qx, qy: unit discharges (m^2/s), shape [n_cells].
        h: water depth (m), > 0, shape [n_cells].

    Returns:
        Speed |U| (m/s), shape [n_cells].
    """
    speed_sq = (qx / h) ** 2 + (qy / h) ** 2
    pos = speed_sq > 0.0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, speed_sq, 1.0)), 0.0)


def still_depth(m: SWEModel, theta: Theta) -> Array:
    """Still-water depth h_still (m).

    With ``zb`` as a parameter, ``h_still = w_still - zb`` so that the depth
    ``h = xi + h_still`` follows the bed being estimated. (Deviation from
    Hydrograd.jl, which keeps h_still from zb = 0 during bathymetry inversion,
    so its fluxes and friction see the depth above datum instead of above the bed.)

    Args:
        m: the model.
        theta: differentiable parameters (``theta["zb"]``, if present, is the bed in m).

    Returns:
        Still-water depth (m), shape [n_cells].
    """
    return m.wstill - theta["zb"] if "zb" in theta else m.hstill


def clipped_state(
    m: SWEModel, U: Array, hstill: Array | None = None,
) -> tuple[Array, Array, Array, Array]:
    """(xi, h, q_x, q_y) after the dry-cell clipping of step 1.

    Args:
        m: the model (``h_small`` and the default ``hstill``).
        U: state [xi, q_x, q_y], shape [3, n_cells].
        hstill: still-water depth (m), shape [n_cells]; None uses ``m.hstill``.

    Returns:
        (xi, h, q_x, q_y), each of shape [n_cells]: displacement (m), depth (m) set to
        h_small in dry cells, and unit discharges (m^2/s) set to 0 in dry cells.
    """
    xi = U[0]
    qx = U[1]
    qy = U[2]

    # Depth, and dry cells h <= h_small.
    h = xi + (m.hstill if hstill is None else hstill)
    dry = h <= m.h_small

    # Dry cells: depth clipped to h_small, discharge set to zero.
    h_clipped = jnp.where(dry, m.h_small, h)
    qx_clipped = jnp.where(dry, 0.0, qx)
    qy_clipped = jnp.where(dry, 0.0, qy)

    return xi, h_clipped, qx_clipped, qy_clipped


def manning_n_cells(m: SWEModel, theta: Theta, h: Array, qx: Array, qy: Array) -> Array:
    """Manning's n per cell (s/m^(1/3)); zone values for ``nn_FlowResistance``.

    Args:
        m: the model (its ``resistance`` selects the mode).
        theta: differentiable parameters (``n_zone`` or the NN weights ``nn``).
        h: water depth (m), > 0, shape [n_cells].
        qx, qy: unit discharges (m^2/s), shape [n_cells].

    Returns:
        Manning's n (s/m^(1/3)), shape [n_cells].
    """
    r = m.resistance

    # Variable-n law n(h[, |U|, ks]).
    if r.mode == "law":
        assert r.law is not None
        speed = _safe_speed(qx, qy, h)
        return manning_law(r.law, h, speed, m.ks)[0]

    # NN closure n = f(h), inputs normalized to [-1, 1].
    if r.mode == "nn_ManningN_h":
        x = normalize(h, *r.h_bounds)[:, None]
        return apply_mlp(theta["nn"], x, r.activations, *r.out_bounds, r.ln_axes)[:, 0]

    # NN closure n = f(h, |U|, ks).
    if r.mode == "nn_ManningN_h_Umag_ks":
        h_input = normalize(h, *r.h_bounds)
        speed_input = normalize(_safe_speed(qx, qy, h), *r.umag_bounds)
        ks_input = normalize(m.ks, *r.ks_bounds)
        x = jnp.stack([h_input, speed_input, ks_input], axis=1)
        return apply_mlp(theta["nn"], x, r.activations, *r.out_bounds, r.ln_axes)[:, 0]

    # Default: constant n per material zone.
    n_zone = theta.get("n_zone", m.n_zone)
    return n_zone[m.mat_id]


def friction_coefficient(
    m: SWEModel, theta: Theta, h: Array, qx: Array, qy: Array, n: Array,
) -> Array:
    """Friction rate c (1/s) per cell, with friction = c q.

    c is the bed friction (Manning, or the NN magnitude / |q|) plus, if the model has
    vegetation, the stem drag rate c_veg = 1/2 C_D a |u| (``physics.vegetation``), plus,
    if it has porous zones, the Darcy-Forchheimer rate c_df = alpha + beta |u|
    (``physics.porous``).

    Args:
        m: the model (``g``, ``k_n``, ``h_small``, resistance mode, vegetation).
        theta: differentiable parameters (NN weights, ``veg_CD``, ``veg_a``,
            ``df_alpha``, ``df_beta``).
        h: water depth (m), shape [n_cells].
        qx, qy: unit discharges (m^2/s), shape [n_cells].
        n: Manning's n (s/m^(1/3)), shape [n_cells].

    Returns:
        Friction rate c (1/s), shape [n_cells].
    """
    c = _bed_friction_coefficient(m, theta, h, qx, qy, n)

    # Stem drag adds to the bed friction. Without vegetation the bed friction is returned
    # unchanged (keeps earlier results bit-identical).
    if m.vegetation.mode != "none":
        c = c + vegetation_drag_rate(m, theta, h, qx, qy)

    # Darcy-Forchheimer drag of porous zones adds in the same way.
    if m.porous.mode != "none":
        c = c + porous_drag_rate(m, theta, h, qx, qy)

    return c


def _bed_friction_coefficient(
    m: SWEModel, theta: Theta, h: Array, qx: Array, qy: Array, n: Array,
) -> Array:
    """Bed friction rate c_bed (1/s) per cell (Manning, or NN magnitude / |q|).

    Args:
        m: the model (``g``, ``k_n``, ``h_small``, resistance mode).
        theta: differentiable parameters (NN weights for ``nn_FlowResistance``).
        h: water depth (m), shape [n_cells].
        qx, qy: unit discharges (m^2/s), shape [n_cells].
        n: Manning's n (s/m^(1/3)), shape [n_cells].

    Returns:
        c_bed (1/s), shape [n_cells].
    """
    # Smoothed |q| (m^2/s); finite derivative at q = 0.
    q_mag = smooth_sqrt(qx * qx + qy * qy)

    # NN closure: the network predicts the friction magnitude tau (m^2/s^2) from (h, |q|).
    if m.resistance.mode == "nn_FlowResistance":
        r = m.resistance
        h_input = normalize(h, *r.h_bounds)
        q_input = normalize(q_mag, *r.q_bounds)
        x = jnp.stack([h_input, q_input], axis=1)
        tau = apply_mlp(theta["nn"], x, r.activations, *r.out_bounds, r.ln_axes)[:, 0]
        return tau / q_mag

    # Manning: c = g n^2 |q| / (k_n^2 h_eff^(7/3)).
    # - hr_hllc: smooth depth floor h_eff = sqrt(h^2 + h_small^2). The relative friction
    #   error in wet cells is about 7/6 (h_small / h)^2 (1e-5 at h = 0.3 m), and near-dry
    #   cells stay bounded and smooth for Newton (h_eff -> h_small as h -> 0).
    # - julia_xi_roe: Hydrograd.jl's (h + h_small)^(7/3), kept for parity. It lowers the
    #   friction by about 7/3 h_small / h in wet cells (0.8% at h = 0.3 m).
    if m.scheme == "julia_xi_roe":
        depth_term = (h + m.h_small) ** (7.0 / 3.0)
    else:
        h_eff = jnp.sqrt(h * h + m.h_small * m.h_small)
        depth_term = h_eff ** (7.0 / 3.0)
    return m.g * n * n / m.k_n ** 2 / depth_term * q_mag


def friction(
    m: SWEModel, theta: Theta, h: Array, qx: Array, qy: Array, n: Array,
) -> tuple[Array, Array]:
    """Bed friction (m^2/s^2) per cell, x and y components (= c q).

    Args:
        m: the model.
        theta: differentiable parameters.
        h: water depth (m), shape [n_cells].
        qx, qy: unit discharges (m^2/s), shape [n_cells].
        n: Manning's n (s/m^(1/3)), shape [n_cells].

    Returns:
        (friction_x, friction_y) (m^2/s^2), each of shape [n_cells].
    """
    c = friction_coefficient(m, theta, h, qx, qy, n)
    return c * qx, c * qy


def _inlet_face_terms(m: SWEModel, h: Array, n: Array) -> tuple[Array, Array, Array]:
    """Per boundary face: (wet flag, conveyance K_f, velocity shape) for inlet-Q faces.

    Default split (``inlet_split = "conveyance"``): K_f = L h^(5/3) / n and the velocity
    shape is h^(2/3) / n. Hydrograd.jl (``"julia"``): K_f = L^(5/3) h / n, shape L^(2/3)/n.

    Args:
        m: the model (boundary faces and ``inlet_split``).
        h: cell water depth (m), shape [n_cells].
        n: cell Manning's n (s/m^(1/3)), shape [n_cells].

    Returns:
        (wet, k_face, v_shape), each of shape [n_boundary_faces]: 1.0 on wet inlet faces
        and 0.0 elsewhere; face conveyance K_f (zero on non-inlet and dry faces); and the
        velocity shape factor, so that the face normal velocity is Q/K * v_shape.
    """
    face_length = m.fb_L
    hc = h[m.fb_c]
    nc = n[m.fb_c]

    # Wet inlet faces carry discharge; dry ones and other kinds do not.
    is_in = m.fb_kind == KIND_INLET
    wet = jnp.where(is_in & (hc > m.h_small), 1.0, 0.0)

    # Hydrograd.jl's split (L and h swapped).
    if m.inlet_split == "julia":
        k_face = face_length ** (5.0 / 3.0) * hc / nc * wet
        v_shape = face_length ** (2.0 / 3.0) / nc
        return wet, k_face, v_shape

    # Conveyance split. Wet faces have h > h_small; the floor avoids
    # d(h^(2/3))/dh = inf at h = 0.
    hw = jnp.maximum(hc, m.h_small)
    k_face = face_length * hw ** (5.0 / 3.0) / nc * wet
    v_shape = hw ** (2.0 / 3.0) / nc
    return wet, k_face, v_shape


def inlet_conveyance(m: SWEModel, theta: Theta, U: Array) -> Array:
    """Total conveyance K of each inlet-Q boundary, shape [max(n_inlets, 1)].

    All inlet cells interact only through K, so dR/dU = J_local + (dR/dK)(dK/dU):
    the Jacobian tools treat that coupling as a low-rank term.

    Args:
        m: the model.
        theta: differentiable parameters.
        U: state [xi, q_x, q_y], shape [3, n_cells].

    Returns:
        Conveyance K = sum of K_f over the wet faces of each inlet (units of L h^(5/3)/n,
        m^3/s in SI), shape [max(n_inlets, 1)].
    """
    # Cell depth and Manning's n, computed as in the selected scheme.
    if m.scheme == "hr_hllc":
        from hydrograd.physics.rhs_hr import _cell_state
        _, h, u, v = _cell_state(m, theta, U)
        n = manning_n_cells(m, theta, jnp.maximum(h, m.h_small), h * u, h * v)
    else:
        _, h, qx, qy = clipped_state(m, U, still_depth(m, theta))
        n = manning_n_cells(m, theta, h, qx, qy)

    # Sum the face conveyances per inlet boundary.
    _, k_face, _ = _inlet_face_terms(m, h, n)
    return jax.ops.segment_sum(k_face, m.fb_inlet, max(m.n_inlets, 1))


def _ghosts(m: SWEModel, theta: Theta, zb: Array, h: Array, qx: Array, qy: Array,
            n: Array, K: Array | None = None) -> tuple[Array, Array, Array]:
    """Ghost (h, q_x, q_y) on boundary faces (``process_all_boundaries_2d``).

    - Inlet-Q: cell depth, discharge q = -h v_n n with v_n from the conveyance split.
    - Exit-H: depth from the prescribed WSE, max(h_small, WSE - zb); cell discharge.
    - Wall: cell depth, reversed discharge -q.
    - Symmetry: cell depth, discharge with its normal component reflected.

    Args:
        m: the model (boundary faces).
        theta: parameters; ``Q`` (m^3/s) and ``WSE`` (m) override the model values.
        zb: cell bed elevation (m), shape [n_cells].
        h: cell water depth (m), shape [n_cells].
        qx, qy: cell unit discharges (m^2/s), shape [n_cells].
        n: cell Manning's n (s/m^(1/3)), shape [n_cells].
        K: inlet conveyances, shape [max(n_inlets, 1)]; None computes them from h.

    Returns:
        (h_ghost, qx_ghost, qy_ghost), each of shape [n_boundary_faces]: depth (m) and
        unit discharges (m^2/s) of the ghost states.
    """
    cell = m.fb_c
    nx = m.fb_nx
    ny = m.fb_ny
    kind = m.fb_kind

    # Cell values next to each boundary face.
    hc = h[cell]
    qxc = qx[cell]
    qyc = qy[cell]

    # Boundary-kind masks.
    is_in = kind == KIND_INLET
    is_ex = kind == KIND_EXIT
    is_wall = kind == KIND_WALL
    is_sym = kind == KIND_SYMM

    # Inlet-Q: split Q over wet faces by Manning conveyance K_f = L h^(5/3) / n, so that the
    # face velocity is v_n = Q/K * h^(2/3)/n and sum_f h v_n L = Q. Hydrograd.jl swaps L and h
    # (K_f = L^(5/3) h / n, v_n = Q/K * L^(2/3)/n); inlet_split = "julia" reproduces that.
    Q = theta.get("Q", m.Q)
    wet, k_face, v_shape = _inlet_face_terms(m, h, n)
    if K is None:
        K = jax.ops.segment_sum(k_face, m.fb_inlet, max(m.n_inlets, 1))
    K_f = K[m.fb_inlet]

    # Julia asserts K > 0; here an all-dry inlet carries no discharge.
    ok = is_in & (K_f > 0.0)
    Q_face = Q[m.fb_inlet] if m.n_inlets else 0.0
    Q_over_K = jnp.where(ok, Q_face / jnp.where(ok, K_f, 1.0), 0.0)
    vn = Q_over_K * v_shape

    # Inflow discharge points against the outward normal.
    qx_in = -hc * vn * nx * wet
    qy_in = -hc * vn * ny * wet

    # Exit-H: depth from the prescribed WSE.
    WSE = theta.get("WSE", m.WSE)
    WSE_face = WSE[m.fb_exit] if m.n_exits else 0.0
    h_ex = jnp.maximum(m.h_small, WSE_face - zb[cell])

    # Symmetry: reflect the normal component, q - 2 (q.n) n.
    vdn = qxc * nx + qyc * ny
    qx_sym = qxc - 2.0 * vdn * nx
    qy_sym = qyc - 2.0 * vdn * ny

    # Assemble the ghost states by boundary kind (other kinds copy the cell values).
    hg = jnp.where(is_ex, h_ex, hc)
    qxg = jnp.where(is_in, qx_in, jnp.where(is_wall, -qxc, jnp.where(is_sym, qx_sym, qxc)))
    qyg = jnp.where(is_in, qy_in, jnp.where(is_wall, -qyc, jnp.where(is_sym, qy_sym, qyc)))

    return hg, qxg, qyg


def boundary_values_at(m: SWEModel, theta: Theta, t: Array | float | None) -> Theta:
    """theta with inlet Q and exit WSE evaluated at time t (s), linear in time.

    Time series (arrays [n_boundaries, n_times] on ``m.Q_times`` / ``m.WSE_times``)
    may come from the model or be parameters in theta (hydrograph inversion).
    With t = None (steady solves) the first value of each series is used.

    Args:
        m: the model (default values and time grids).
        theta: differentiable parameters.
        t: time (s), or None for steady solves.

    Returns:
        A copy of theta whose ``Q`` (m^3/s) and ``WSE`` (m) entries are [n_boundaries]
        vectors when the model or theta holds time series; unchanged otherwise.
    """
    out = dict(theta)

    for key, times in (("Q", m.Q_times), ("WSE", m.WSE_times)):
        values = theta.get(key, getattr(m, key))

        # Constant boundary values: nothing to evaluate.
        if values.ndim != 2:
            continue

        # Steady solve: first value of each series.
        if t is None:
            out[key] = values[:, 0]
            continue

        # Linear interpolation of each boundary's series at time t.
        def interp_row(row: Array, tt: Array = times) -> Array:
            """Value of one boundary's series [n_times] at time t (s)."""
            return jnp.interp(t, tt, row)

        out[key] = jax.vmap(interp_row)(values)

    return out


def rhs(
    m: SWEModel,
    theta: Theta,
    U: Array,
    K: Array | None = None,
    t: Array | float | None = None,
) -> Array:
    """dU/dt [3, n_cells] with the model's scheme (``m.scheme``) at time t (s).

    ``hr_hllc`` (default): hydrostatic reconstruction + HLLC, wet/dry capable
    (:mod:`hydrograd.physics.rhs_hr`). ``julia_xi_roe``: Hydrograd.jl's xi-form Roe
    scheme (:func:`rhs_julia`), for parity with Julia; requires a wet domain.

    Args:
        m: the model.
        theta: differentiable parameters.
        U: state [xi, q_x, q_y], shape [3, n_cells].
        K: inlet conveyances, shape [max(n_inlets, 1)], or None to compute them from U.
        t: time (s) for time-varying boundary values, or None (steady).

    Returns:
        dU/dt, shape [3, n_cells] (units of U per second).

    Raises:
        ValueError: for an unknown scheme.
    """
    theta = boundary_values_at(m, theta, t)

    if m.scheme == "julia_xi_roe":
        return rhs_julia(m, theta, U, K)

    if m.scheme == "hr_hllc":
        from hydrograd.physics.rhs_hr import rhs_hr
        return rhs_hr(m, theta, U, K)

    raise ValueError(f"unknown scheme '{m.scheme}'")


def rhs_julia(m: SWEModel, theta: Theta, U: Array, K: Array | None = None) -> Array:
    """dU/dt, shape [3, n_cells], with Hydrograd.jl's xi-form Roe scheme.

    Args:
        m: static model.
        theta: parameter overrides (``zb``, ``n_zone``, ``Q``, ``WSE``, ``nn``); may be empty.
        U: state [xi, q_x, q_y], shape [3, n_cells].
        K: optional inlet conveyances to use instead of computing them from U
            (Jacobian splitting; see :func:`inlet_conveyance`).

    Returns:
        dU/dt, shape [3, n_cells] (units of U per second).
    """
    g = m.g
    hmin = m.h_small

    # ------------------------------------------------------------------
    # 1. Cell variables (dry cells clipped) and bed
    # ------------------------------------------------------------------
    hs = still_depth(m, theta)
    xi, h, qx, qy = clipped_state(m, U, hs)

    # Bed elevation and slope: parameter values if zb is estimated, else the defaults.
    if "zb" in theta:
        zb = theta["zb"]
        S0 = bed_slope(m.bed, zb)
    else:
        zb = m.zb
        S0 = m.S0

    # ------------------------------------------------------------------
    # 2. Resistance: Manning's n per cell
    # ------------------------------------------------------------------
    n = manning_n_cells(m, theta, h, qx, qy)

    # ------------------------------------------------------------------
    # 3. Interior faces: Roe flux from owner to neighbour, times face length
    # ------------------------------------------------------------------
    owner = m.fi_o
    neighbour = m.fi_r
    flux_interior = roe_flux(
        xi[owner], hs[owner], h[owner], qx[owner], qy[owner], zb[owner],
        xi[neighbour], hs[neighbour], h[neighbour], qx[neighbour], qy[neighbour], zb[neighbour],
        m.fi_nx, m.fi_ny, g, hmin,
    )
    Fi = jnp.stack(flux_interior, axis=1) * m.fi_L[:, None]

    # ------------------------------------------------------------------
    # 4. Boundary faces with ghost cells (zb_ghost = zb_cell, h_still_ghost = h_still_cell)
    # ------------------------------------------------------------------
    cell = m.fb_c
    hg, qxg, qyg = _ghosts(m, theta, zb, h, qx, qy, n, K)
    flux_boundary = roe_flux(
        xi[cell], hs[cell], h[cell], qx[cell], qy[cell], zb[cell],
        hg - hs[cell], hs[cell], hg, qxg, qyg, zb[cell],
        m.fb_nx, m.fb_ny, g, hmin,
    )
    Fb = jnp.stack(flux_boundary, axis=1) * m.fb_L[:, None]

    # ------------------------------------------------------------------
    # 5. Assemble: -(1/A) sum of the outward face fluxes
    # ------------------------------------------------------------------
    net_outflow = (
        jax.ops.segment_sum(Fi, owner, m.n_cells)
        - jax.ops.segment_sum(Fi, neighbour, m.n_cells)
        + jax.ops.segment_sum(Fb, cell, m.n_cells)
    )
    inviscid = -(net_outflow / m.area[:, None]).T

    # ------------------------------------------------------------------
    # 6. Sources in wet cells: bed slope g xi S0 minus friction
    # ------------------------------------------------------------------
    fx, fy = friction(m, theta, h, qx, qy, n)
    wet = h > hmin
    sx = jnp.where(wet, g * xi * S0[:, 0] - fx, 0.0)
    sy = jnp.where(wet, g * xi * S0[:, 1] - fy, 0.0)

    return inviscid + jnp.stack([jnp.zeros_like(sx), sx, sy])
