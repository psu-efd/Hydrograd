"""Roe approximate Riemann solver for the 2D SWEs in the pre-balanced xi-form.

Port of ``Riemann_2D_Roe`` in Hydrograd.jl
(``src/fvm/discretization/Riemman_solvers/swe_2D_solvers.jl``), vectorized
over faces. The conserved variables are ``(xi, q_x, q_y)``, where
``xi = h - h_still`` is the free-surface displacement (m) above the still-water
level, and ``q = h u`` the unit discharge (m^2/s). The hydrostatic pressure
flux is written as ``g (xi^2/2 + xi h_still)`` (Rogers et al. 2003), which,
together with the bed source ``g xi S0``, makes the scheme well balanced.

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
    """sqrt(x^2 + eps): differentiable |x| (Julia ``smooth_abs``)."""
    return jnp.sqrt(x * x + EPS)


def smooth_sqrt(x: Array) -> Array:
    """sqrt(x + eps) (Julia ``smooth_sqrt``)."""
    return jnp.sqrt(x + EPS)


def smooth_pow2(x: Array) -> Array:
    """(x + eps)^2 (Julia ``smooth_pow(x, 2)``)."""
    return (x + EPS) ** 2


def _one_sided(h: Array, qx: Array, qy: Array, nx: Array, ny: Array, g: float) -> tuple[Array, ...]:
    """Physical flux F(U).n of one wet state (used when the other side is dry)."""
    u, v = qx / h, qy / h
    p = 0.5 * g * smooth_pow2(h)
    return qx * nx + qy * ny, (qx * u + p) * nx + qx * v * ny, (qy * u) * nx + (qy * v + p) * ny


def _roe(xiL, hsL, hL, qxL, qyL, xiR, hsR, hR, qxR, qyR, nx, ny, g: float) -> tuple[Array, ...]:
    """Roe flux for two wet states (branch 6)."""
    uL, vL, uR, vR = qxL / hL, qyL / hL, qxR / hR, qyR / hR
    sL, sR = smooth_sqrt(hL), smooth_sqrt(hR)
    h_roe = 0.5 * (hL + hR)
    u_roe = (sL * uL + sR * uR) / (sL + sR)
    v_roe = (sL * vL + sR * vR) / (sL + sR)
    un = u_roe * nx + v_roe * ny
    c = smooth_sqrt(g * h_roe)
    k = 0.5 / c

    d0, d1, d2 = xiR - xiL, qxR - qxL, qyR - qyL
    # L_mat * dQ  (left eigenvectors)
    w0 = -(u_roe * ny - v_roe * nx) * d0 + ny * d1 - nx * d2
    w1 = (un * k + 0.5) * d0 - nx * k * d1 - ny * k * d2
    w2 = (-un * k + 0.5) * d0 + nx * k * d1 + ny * k * d2
    # |Lambda| * (L dQ)
    w0, w1, w2 = smooth_abs(un) * w0, smooth_abs(un - c) * w1, smooth_abs(un + c) * w2
    # R_mat * (...)  (right eigenvectors)
    a0 = w1 + w2
    a1 = ny * w0 + (u_roe - c * nx) * w1 + (u_roe + c * nx) * w2
    a2 = -nx * w0 + (v_roe - c * ny) * w1 + (v_roe + c * ny) * w2

    pL = 0.5 * g * (smooth_pow2(xiL) + 2.0 * xiL * hsL)
    pR = 0.5 * g * (smooth_pow2(xiR) + 2.0 * xiR * hsR)
    f0 = 0.5 * ((qxL * nx + qyL * ny) + (qxR * nx + qyR * ny) - a0)
    f1 = 0.5 * ((qxL * uL + pL) * nx + qxL * vL * ny + (qxR * uR + pR) * nx + qxR * vR * ny - a1)
    f2 = 0.5 * (qyL * uL * nx + (qyL * vL + pL) * ny + qyR * uR * nx + (qyR * vR + pR) * ny - a2)
    return f0, f1, f2


def roe_flux(xiL: Array, hsL: Array, hL: Array, qxL: Array, qyL: Array, zbL: Array,
             xiR: Array, hsR: Array, hR: Array, qxR: Array, qyR: Array, zbR: Array,
             nx: Array, ny: Array, g: float, h_min: float) -> tuple[Array, Array, Array]:
    """Numerical flux F.n per face (per unit face length) with dry-face branches.

    Args:
        xiL, xiR: free-surface displacement (m) left/right.
        hsL, hsR: still-water depth h_still (m).
        hL, hR: water depth (m), already clipped to ``>= h_min``.
        qxL, qyL, qxR, qyR: unit discharges (m^2/s).
        zbL, zbR: bed elevation (m) of the left/right cell.
        nx, ny: unit normal pointing from left to right.
        g: gravity (m/s^2); h_min: dry threshold h_small (m).

    Returns:
        (F_xi, F_qx, F_qy), each with the shape of the inputs.
    """
    dryL, dryR = hL <= h_min, hR <= h_min
    both_dry = dryL & dryR
    mirror_R = ((hL + zbL) < (zbR + h_min)) & dryR & ~both_dry
    mirror_L = ((hR + zbR) < (zbL + h_min)) & dryL & ~both_dry & ~mirror_R
    one_R = dryL & ~both_dry & ~mirror_R & ~mirror_L       # left dry: flux of right state
    one_L = dryR & ~both_dry & ~mirror_R & ~mirror_L & ~one_R

    # Branches 2/3 modify h, q only; then fall through to the Roe flux.
    hR_ = jnp.where(mirror_R, hL, hR)
    qxR_ = jnp.where(mirror_R, -qxL, qxR)
    qyR_ = jnp.where(mirror_R, -qyL, qyR)
    hL_ = jnp.where(mirror_L, hR, hL)
    qxL_ = jnp.where(mirror_L, -qxR, qxL)
    qyL_ = jnp.where(mirror_L, -qyR, qyL)

    # Guard the unused branches against division by tiny h (keeps gradients finite).
    safe = ~(both_dry | one_R | one_L)
    one = jnp.ones_like(hL)
    roe = _roe(xiL, hsL, jnp.where(safe, hL_, one), qxL_, qyL_,
               xiR, hsR, jnp.where(safe, hR_, one), qxR_, qyR_, nx, ny, g)
    fr = _one_sided(jnp.where(one_R, hR, one), qxR, qyR, nx, ny, g)
    fl = _one_sided(jnp.where(one_L, hL, one), qxL, qyL, nx, ny, g)

    out = []
    for k in range(3):
        f = jnp.where(one_R, fr[k], jnp.where(one_L, fl[k], roe[k]))
        out.append(jnp.where(both_dry, 0.0, f))
    return out[0], out[1], out[2]
