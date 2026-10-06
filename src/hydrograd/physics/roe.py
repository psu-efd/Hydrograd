"""Roe approximate Riemann solver for the 2D SWEs in the pre-balanced xi-form.

Port of ``Riemann_2D_Roe`` in Hydrograd.jl
(``src/fvm/discretization/Riemman_solvers/swe_2D_solvers.jl``), vectorized over faces.
Used by the ``julia_xi_roe`` scheme (:func:`hydrograd.physics.rhs.rhs_julia`).

Variables
---------
The conserved variables are ``(xi, q_x, q_y)``, where ``xi = h - h_still`` is the
free-surface displacement (m) above the still-water level, h the water depth (m), and
``q = h u`` the unit discharge (m^2/s), with u = (u, v) the depth-averaged velocity (m/s).

Pre-balanced flux
-----------------
The hydrostatic pressure flux is written as ``g (xi^2/2 + xi h_still)`` (Rogers et al.
2003) instead of g h^2 / 2. Together with the bed source ``g xi S0`` this makes the scheme
well balanced (a lake at rest stays at rest). The physical flux normal to a face with unit
normal n = (nx, ny) is

    F.n = ( q.n,
            (q_x u + p) nx + q_x v ny,
            q_y u nx + (q_y v + p) ny ),    p = g (xi^2/2 + xi h_still).

Roe flux
--------
F = (F_L + F_R)/2 - R |Lambda| L (U_R - U_L) / 2, with Roe averages
h_roe = (h_L + h_R)/2, u_roe = (sqrt(h_L) u_L + sqrt(h_R) u_R) / (sqrt(h_L) + sqrt(h_R)),
eigenvalues u_n, u_n - c, u_n + c (c = sqrt(g h_roe)), and left/right eigenvector
matrices L and R. Absolute values and square roots use Julia's eps-smoothed versions.

Dry faces
---------
Dry-face handling follows the Julia branches exactly (evaluated in order):

1. both sides dry (``h <= h_min``)                     -> zero flux;
2. left WSE below right bed and right dry              -> right state := mirrored left;
3. right WSE below left bed and left dry               -> left state := mirrored right;
4. left dry                                            -> one-sided flux of the right state;
5. right dry                                           -> one-sided flux of the left state;
6. otherwise                                           -> Roe flux.

In branches 2-3 only ``h`` and ``q`` are mirrored; ``xi`` and ``h_still`` keep
their original values, as in Julia.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

EPS = float(jnp.finfo(jnp.float64).eps)  # eps(Float64) used by Julia's smooth_* functions


def smooth_abs(x: Array) -> Array:
    """sqrt(x^2 + eps): differentiable |x| (Julia ``smooth_abs``).

    Args:
        x: input values, any shape and units.

    Returns:
        Smoothed |x|, same shape and units as ``x``.
    """
    return jnp.sqrt(x * x + EPS)


def smooth_sqrt(x: Array) -> Array:
    """sqrt(x + eps) (Julia ``smooth_sqrt``); finite derivative at x = 0.

    Args:
        x: non-negative input values, any shape.

    Returns:
        Smoothed sqrt(x), same shape as ``x``.
    """
    return jnp.sqrt(x + EPS)


def smooth_pow2(x: Array) -> Array:
    """(x + eps)^2 (Julia ``smooth_pow(x, 2)``).

    Args:
        x: input values, any shape.

    Returns:
        (x + eps)^2, same shape as ``x``.
    """
    return (x + EPS) ** 2


def _one_sided(h: Array, qx: Array, qy: Array, nx: Array, ny: Array, g: float) -> tuple[Array, ...]:
    """Physical flux F(U).n of one wet state (used when the other side is dry).

    Uses the full hydrostatic pressure p = g h^2 / 2 (not the xi-form).

    Args:
        h: water depth (m), shape [n_faces].
        qx, qy: unit discharges (m^2/s), shape [n_faces].
        nx, ny: components of the unit face normal, shape [n_faces].
        g: gravitational acceleration (m/s^2).

    Returns:
        (F_xi, F_qx, F_qy), each of shape [n_faces]: mass flux (m^2/s) and x-, y-momentum
        fluxes (m^3/s^2) per unit face length.
    """
    # Velocities (m/s) and hydrostatic pressure term (m^3/s^2).
    u = qx / h
    v = qy / h
    p = 0.5 * g * smooth_pow2(h)

    # Normal flux of mass, x-momentum and y-momentum.
    flux_mass = qx * nx + qy * ny
    flux_x = (qx * u + p) * nx + qx * v * ny
    flux_y = (qy * u) * nx + (qy * v + p) * ny

    return flux_mass, flux_x, flux_y


def _roe(
    xiL: Array, hsL: Array, hL: Array, qxL: Array, qyL: Array,
    xiR: Array, hsR: Array, hR: Array, qxR: Array, qyR: Array,
    nx: Array, ny: Array, g: float,
) -> tuple[Array, ...]:
    """Roe flux for two wet states (branch 6).

    Args:
        xiL, xiR: free-surface displacement (m), shape [n_faces].
        hsL, hsR: still-water depth h_still (m), shape [n_faces].
        hL, hR: water depth (m), > 0, shape [n_faces].
        qxL, qyL, qxR, qyR: unit discharges (m^2/s), shape [n_faces].
        nx, ny: components of the unit normal, from left to right, shape [n_faces].
        g: gravitational acceleration (m/s^2).

    Returns:
        (F_xi, F_qx, F_qy), each of shape [n_faces]: mass flux (m^2/s) and x-, y-momentum
        fluxes (m^3/s^2) per unit face length.
    """
    # ------------------------------------------------------------------
    # 1. Velocities and Roe averages
    # ------------------------------------------------------------------
    uL = qxL / hL
    vL = qyL / hL
    uR = qxR / hR
    vR = qyR / hR

    # sqrt(h) weights of the Roe-averaged velocity.
    sL = smooth_sqrt(hL)
    sR = smooth_sqrt(hR)
    h_roe = 0.5 * (hL + hR)
    u_roe = (sL * uL + sR * uR) / (sL + sR)
    v_roe = (sL * vL + sR * vR) / (sL + sR)

    # Normal Roe velocity (m/s), celerity c (m/s), and k = 1/(2c) (s/m).
    un = u_roe * nx + v_roe * ny
    c = smooth_sqrt(g * h_roe)
    k = 0.5 / c

    # ------------------------------------------------------------------
    # 2. Upwind dissipation R |Lambda| L dU
    # ------------------------------------------------------------------
    # Jumps of the conserved variables across the face.
    d0 = xiR - xiL
    d1 = qxR - qxL
    d2 = qyR - qyL

    # L_mat * dU  (left eigenvectors).
    w0 = -(u_roe * ny - v_roe * nx) * d0 + ny * d1 - nx * d2
    w1 = (un * k + 0.5) * d0 - nx * k * d1 - ny * k * d2
    w2 = (-un * k + 0.5) * d0 + nx * k * d1 + ny * k * d2

    # |Lambda| * (L dU), eigenvalues u_n, u_n - c, u_n + c.
    w0_scaled = smooth_abs(un) * w0
    w1_scaled = smooth_abs(un - c) * w1
    w2_scaled = smooth_abs(un + c) * w2

    # R_mat * (...)  (right eigenvectors).
    a0 = w1_scaled + w2_scaled
    a1 = ny * w0_scaled + (u_roe - c * nx) * w1_scaled + (u_roe + c * nx) * w2_scaled
    a2 = -nx * w0_scaled + (v_roe - c * ny) * w1_scaled + (v_roe + c * ny) * w2_scaled

    # ------------------------------------------------------------------
    # 3. Average of the physical fluxes minus half the dissipation
    # ------------------------------------------------------------------
    # Pre-balanced pressure terms p = g (xi^2/2 + xi h_still) (m^3/s^2).
    pL = 0.5 * g * (smooth_pow2(xiL) + 2.0 * xiL * hsL)
    pR = 0.5 * g * (smooth_pow2(xiR) + 2.0 * xiR * hsR)

    # Left flux terms, then right flux terms, then the dissipation (summed left to right).
    f0 = 0.5 * ((qxL * nx + qyL * ny) + (qxR * nx + qyR * ny) - a0)
    f1 = 0.5 * (
        (qxL * uL + pL) * nx + qxL * vL * ny
        + (qxR * uR + pR) * nx + qxR * vR * ny
        - a1
    )
    f2 = 0.5 * (
        qyL * uL * nx + (qyL * vL + pL) * ny
        + qyR * uR * nx + (qyR * vR + pR) * ny
        - a2
    )

    return f0, f1, f2


def roe_flux(xiL: Array, hsL: Array, hL: Array, qxL: Array, qyL: Array, zbL: Array,
             xiR: Array, hsR: Array, hR: Array, qxR: Array, qyR: Array, zbR: Array,
             nx: Array, ny: Array, g: float, h_min: float) -> tuple[Array, Array, Array]:
    """Numerical flux F.n per face (per unit face length) with dry-face branches.

    Args:
        xiL, xiR: free-surface displacement (m) left/right, shape [n_faces].
        hsL, hsR: still-water depth h_still (m), shape [n_faces].
        hL, hR: water depth (m), already clipped to ``>= h_min``, shape [n_faces].
        qxL, qyL, qxR, qyR: unit discharges (m^2/s), shape [n_faces].
        zbL, zbR: bed elevation (m) of the left/right cell, shape [n_faces].
        nx, ny: unit normal pointing from left to right, shape [n_faces].
        g: gravitational acceleration (m/s^2).
        h_min: dry threshold h_small (m).

    Returns:
        (F_xi, F_qx, F_qy), each with the shape of the inputs: mass flux (m^2/s) and
        x-, y-momentum fluxes (m^3/s^2) per unit face length.
    """
    # ------------------------------------------------------------------
    # 1. Branch masks (mutually exclusive, in Julia's order; see module docstring)
    # ------------------------------------------------------------------
    dryL = hL <= h_min
    dryR = hR <= h_min
    both_dry = dryL & dryR
    mirror_R = ((hL + zbL) < (zbR + h_min)) & dryR & ~both_dry
    mirror_L = ((hR + zbR) < (zbL + h_min)) & dryL & ~both_dry & ~mirror_R
    one_R = dryL & ~both_dry & ~mirror_R & ~mirror_L       # left dry: flux of right state
    one_L = dryR & ~both_dry & ~mirror_R & ~mirror_L & ~one_R

    # ------------------------------------------------------------------
    # 2. Mirrored states (branches 2/3 modify h, q only, then use the Roe flux)
    # ------------------------------------------------------------------
    hR_ = jnp.where(mirror_R, hL, hR)
    qxR_ = jnp.where(mirror_R, -qxL, qxR)
    qyR_ = jnp.where(mirror_R, -qyL, qyR)
    hL_ = jnp.where(mirror_L, hR, hL)
    qxL_ = jnp.where(mirror_L, -qxR, qxL)
    qyL_ = jnp.where(mirror_L, -qyR, qyL)

    # ------------------------------------------------------------------
    # 3. Candidate fluxes of all branches
    # ------------------------------------------------------------------
    # Guard the unused branches against division by tiny h (keeps gradients finite):
    # depths of faces that do not use a branch are replaced by 1 in that branch.
    safe = ~(both_dry | one_R | one_L)
    one = jnp.ones_like(hL)
    roe = _roe(xiL, hsL, jnp.where(safe, hL_, one), qxL_, qyL_,
               xiR, hsR, jnp.where(safe, hR_, one), qxR_, qyR_, nx, ny, g)
    flux_right_state = _one_sided(jnp.where(one_R, hR, one), qxR, qyR, nx, ny, g)
    flux_left_state = _one_sided(jnp.where(one_L, hL, one), qxL, qyL, nx, ny, g)

    # ------------------------------------------------------------------
    # 4. Select the branch per face and component; zero flux if both sides are dry
    # ------------------------------------------------------------------
    fluxes = []
    for k in range(3):
        f = jnp.where(one_R, flux_right_state[k], jnp.where(one_L, flux_left_state[k], roe[k]))
        fluxes.append(jnp.where(both_dry, 0.0, f))

    return fluxes[0], fluxes[1], fluxes[2]
