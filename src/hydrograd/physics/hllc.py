"""HLLC approximate Riemann solver for the shallow water equations (face-normal frame).

Toro (2001), *Shock-Capturing Methods for Free-Surface Shallow Flows*: the mass
and normal-momentum fluxes are the HLL fluxes; the tangential momentum is
advected with the contact (middle) wave speed S_M.

Wave speeds (Einfeldt/Toro two-rarefaction estimates, exact for dry states):

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

TINY = 1.0e-14


def _safe_div(a: Array, b: Array) -> Array:
    ok = jnp.abs(b) > TINY
    return jnp.where(ok, a / jnp.where(ok, b, 1.0), 0.0)


def _celerity(h: Array, g: float) -> Array:
    """sqrt(g h), 0 for h <= 0, with a finite derivative at h = 0."""
    wet = h > 0.0
    return jnp.where(wet, jnp.sqrt(g * jnp.where(wet, h, 1.0)), 0.0)


def hllc_flux(hL: Array, unL: Array, utL: Array, hR: Array, unR: Array, utR: Array,
              g: float) -> tuple[Array, Array, Array, Array]:
    """Numerical flux in the face frame.

    Args:
        hL, hR: depths (m, >= 0); unL, unR: normal velocities (m/s, + from L to R);
        utL, utR: tangential velocities (m/s); g: gravity (m/s^2).

    Returns:
        (F_h, F_n, F_t, s_max): mass (m^2/s), normal- and tangential-momentum
        (m^3/s^2) fluxes per unit face length, and the largest wave speed (m/s).
    """
    cL, cR = _celerity(hL, g), _celerity(hR, g)
    dryL, dryR = hL <= 0.0, hR <= 0.0
    u_star = 0.5 * (unL + unR) + cL - cR
    c_star = jnp.maximum(0.5 * (cL + cR) + 0.25 * (unL - unR), 0.0)
    SL = jnp.where(dryL, unR - 2.0 * cR, jnp.where(dryR, unL - cL, jnp.minimum(unL - cL, u_star - c_star)))
    SR = jnp.where(dryR, unL + 2.0 * cL, jnp.where(dryL, unR + cR, jnp.maximum(unR + cR, u_star + c_star)))

    qL, qR = hL * unL, hR * unR
    FL_h, FR_h = qL, qR
    FL_n, FR_n = qL * unL + 0.5 * g * hL * hL, qR * unR + 0.5 * g * hR * hR

    denom = SR - SL
    hll_h = _safe_div(SR * FL_h - SL * FR_h + SL * SR * (hR - hL), denom)
    hll_n = _safe_div(SR * FL_n - SL * FR_n + SL * SR * (qR - qL), denom)
    F_h = jnp.where(SL >= 0.0, FL_h, jnp.where(SR <= 0.0, FR_h, hll_h))
    F_n = jnp.where(SL >= 0.0, FL_n, jnp.where(SR <= 0.0, FR_n, hll_n))

    # Contact speed and upwinded tangential velocity.
    SM = _safe_div(SL * hR * (unR - SR) - SR * hL * (unL - SL), hR * (unR - SR) - hL * (unL - SL))
    F_t = F_h * jnp.where(SM >= 0.0, utL, utR)

    both_dry = dryL & dryR
    zero = jnp.zeros_like(F_h)
    s_max = jnp.maximum(jnp.abs(SL), jnp.abs(SR))
    return (jnp.where(both_dry, zero, F_h), jnp.where(both_dry, zero, F_n),
            jnp.where(both_dry, zero, F_t), jnp.where(both_dry, zero, s_max))
