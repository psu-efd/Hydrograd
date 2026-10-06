"""HLLC approximate Riemann solver for the shallow water equations (face-normal frame).

Reference: Toro (2001), *Shock-Capturing Methods for Free-Surface Shallow Flows*.

Face frame
----------
The solver works per face in the frame of the unit normal n (pointing from the left
state L to the right state R): u_n is the normal velocity, u_t the tangential velocity
(m/s), h the depth (m) and c = sqrt(g h) the celerity (m/s). The fluxes are

    F_h = h u_n                    mass flux (m^2/s)
    F_n = h u_n^2 + g h^2 / 2      normal-momentum flux (m^3/s^2)
    F_t = h u_n u_t                tangential-momentum flux (m^3/s^2)

HLLC flux
---------
The mass and normal-momentum fluxes are the HLL fluxes

    F_hll = (S_R F_L - S_L F_R + S_L S_R (U_R - U_L)) / (S_R - S_L),

with F = F_L if S_L >= 0 and F = F_R if S_R <= 0. The tangential momentum is advected
with the contact (middle) wave speed S_M: F_t = F_h u_t,L if S_M >= 0, else F_h u_t,R.

Wave speeds
-----------
Einfeldt/Toro two-rarefaction estimates, exact for dry states:

* both sides wet: S_L = min(u_L - c_L, u* - c*),  S_R = max(u_R + c_R, u* + c*),
  with u* = (u_L + u_R)/2 + c_L - c_R and c* = (c_L + c_R)/2 + (u_L - u_R)/4;
* dry right (h_R = 0): S_L = u_L - c_L,  S_R = u_L + 2 c_L;
* dry left  (h_L = 0): S_L = u_R - 2 c_R, S_R = u_R + c_R.

HLL is positivity preserving and satisfies the entropy condition (no expansion
shocks), which the Roe solver without an entropy fix does not.
All divisions are guarded so that forward- and reverse-mode derivatives are finite.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

__all__ = ["hllc_flux"]

# Denominators with |b| <= TINY are treated as zero (division returns 0).
TINY = 1.0e-14


def _safe_div(a: Array, b: Array) -> Array:
    """Division a / b that returns 0 where |b| <= TINY, with finite derivatives.

    The inner ``where`` replaces tiny denominators by 1 before dividing, so neither the
    value nor its derivative is inf/NaN in the masked entries.

    Args:
        a: numerator, any shape.
        b: denominator, same shape as ``a``.

    Returns:
        a / b where |b| > TINY, else 0; same shape as ``a``.
    """
    ok = jnp.abs(b) > TINY
    return jnp.where(ok, a / jnp.where(ok, b, 1.0), 0.0)


def _celerity(h: Array, g: float) -> Array:
    """Gravity-wave celerity sqrt(g h), 0 for h <= 0, with a finite derivative at h = 0.

    Args:
        h: water depth (m), any shape.
        g: gravitational acceleration (m/s^2).

    Returns:
        Celerity (m/s), same shape as ``h``.
    """
    wet = h > 0.0
    return jnp.where(wet, jnp.sqrt(g * jnp.where(wet, h, 1.0)), 0.0)


def hllc_flux(hL: Array, unL: Array, utL: Array, hR: Array, unR: Array, utR: Array,
              g: float) -> tuple[Array, Array, Array, Array]:
    """Numerical flux in the face frame.

    Args:
        hL, hR: depths (m, >= 0), shape [n_faces].
        unL, unR: normal velocities (m/s, positive from L to R), shape [n_faces].
        utL, utR: tangential velocities (m/s), shape [n_faces].
        g: gravitational acceleration (m/s^2).

    Returns:
        (F_h, F_n, F_t, s_max), each of shape [n_faces]: mass (m^2/s), normal- and
        tangential-momentum (m^3/s^2) fluxes per unit face length, and the largest wave
        speed (m/s). All four are zero on faces where both sides are dry.
    """
    # ------------------------------------------------------------------
    # 1. Wave speed estimates S_L, S_R
    # ------------------------------------------------------------------
    cL = _celerity(hL, g)
    cR = _celerity(hR, g)
    dryL = hL <= 0.0
    dryR = hR <= 0.0

    # Two-rarefaction middle state (u*, c*); c* is floored at 0.
    u_star = 0.5 * (unL + unR) + cL - cR
    c_star = jnp.maximum(0.5 * (cL + cR) + 0.25 * (unL - unR), 0.0)

    # Wet/wet speeds, then the exact dry-bed speeds where one side is dry.
    SL_wet = jnp.minimum(unL - cL, u_star - c_star)
    SR_wet = jnp.maximum(unR + cR, u_star + c_star)
    SL = jnp.where(dryL, unR - 2.0 * cR, jnp.where(dryR, unL - cL, SL_wet))
    SR = jnp.where(dryR, unL + 2.0 * cL, jnp.where(dryL, unR + cR, SR_wet))

    # ------------------------------------------------------------------
    # 2. Physical fluxes of the left and right states
    # ------------------------------------------------------------------
    # Normal unit discharges q = h u_n (m^2/s); they are also the mass fluxes.
    qL = hL * unL
    qR = hR * unR
    FL_h = qL
    FR_h = qR

    # Normal-momentum fluxes q u_n + g h^2 / 2 (m^3/s^2).
    FL_n = qL * unL + 0.5 * g * hL * hL
    FR_n = qR * unR + 0.5 * g * hR * hR

    # ------------------------------------------------------------------
    # 3. HLL fluxes for mass and normal momentum (upwinded outside the fan)
    # ------------------------------------------------------------------
    denom = SR - SL
    hll_h = _safe_div(SR * FL_h - SL * FR_h + SL * SR * (hR - hL), denom)
    hll_n = _safe_div(SR * FL_n - SL * FR_n + SL * SR * (qR - qL), denom)

    F_h = jnp.where(SL >= 0.0, FL_h, jnp.where(SR <= 0.0, FR_h, hll_h))
    F_n = jnp.where(SL >= 0.0, FL_n, jnp.where(SR <= 0.0, FR_n, hll_n))

    # ------------------------------------------------------------------
    # 4. Contact speed S_M and upwinded tangential velocity
    # ------------------------------------------------------------------
    SM_num = SL * hR * (unR - SR) - SR * hL * (unL - SL)
    SM_den = hR * (unR - SR) - hL * (unL - SL)
    SM = _safe_div(SM_num, SM_den)
    F_t = F_h * jnp.where(SM >= 0.0, utL, utR)

    # ------------------------------------------------------------------
    # 5. Zero everything on faces where both sides are dry
    # ------------------------------------------------------------------
    both_dry = dryL & dryR
    zero = jnp.zeros_like(F_h)
    s_max = jnp.maximum(jnp.abs(SL), jnp.abs(SR))

    F_h_out = jnp.where(both_dry, zero, F_h)
    F_n_out = jnp.where(both_dry, zero, F_n)
    F_t_out = jnp.where(both_dry, zero, F_t)
    s_max_out = jnp.where(both_dry, zero, s_max)

    return F_h_out, F_n_out, F_t_out, s_max_out
