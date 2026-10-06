"""Manning's n parameterizations (port of ``process_ManningN_2D.jl``).

Laws for a *variable* n (forward ``ManningN_option = "variable"``), with
``h`` the water depth (m) and ``U`` the speed (m/s):

* ``inverse``   : n = n_lo + (n_hi - n_lo) / (1 + k h)
* ``power_law`` : n = n_lo + (n_hi - n_lo) (h + eps)^(-k)
* ``sigmoid``   : n = n_lo + (n_hi - n_lo) / (1 + exp(+k (h - h_mid)))
                  (Julia uses ``+k``: n *decreases* from n_hi to n_lo with depth)
* ``h_Umag_ks`` : Cheng (2008) friction factor f(Re, h/ks), n = sqrt(f/8) h^(1/6) / sqrt(g),
                  with nu = 1e-6 m^2/s and g = 9.81 m/s^2 hard-coded (SI only, as in Julia).

All functions are vectorized ``jax.numpy`` and differentiable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["ManningLaw", "cheng_friction", "manning_law"]

EPS = float(jnp.finfo(jnp.float64).eps)
NU = 1.0e-6  # kinematic viscosity of water (m^2/s)
RE_MIN = 2.1 * (1.0 + 1.0e-6)        # Cheng law: log10(Re/2.1) > 0
HKS_MIN = (1.0 + 1.0e-6) / 11.8      # Cheng law: log10(11.8 h/ks) > 0


@dataclass(frozen=True)
class ManningLaw:
    """A named n(h[, U, ks]) law with its parameters (``run_control.json`` keys)."""

    kind: str
    params: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ("inverse", "power_law", "sigmoid", "h_Umag_ks"):
            raise ValueError(f"unknown ManningN function type '{self.kind}'")
        if self.kind in ("inverse", "sigmoid") and not self.params["k"] > 0:
            raise ValueError("ManningN law requires k > 0")
        if self.kind == "sigmoid" and not self.params["h_mid"] > 0:
            raise ValueError("sigmoid ManningN law requires h_mid > 0")

    def __hash__(self) -> int:  # usable as a static jit argument
        return hash((self.kind, tuple(sorted(self.params.items()))))


def cheng_friction(h: Array, umag: Array, ks: Array) -> tuple[Array, Array, Array, Array]:
    """Cheng (2008) Darcy-Weisbach friction factor for flow over rough beds.

    Args:
        h: depth (m); umag: speed (m/s); ks: roughness height (m).

    Returns:
        (n, h/ks, f, Re): Manning's n (s/m^(1/3)), relative submergence (-),
        friction factor (-), Reynolds number U h / nu (-).
    """
    # Floors keep the logs positive; Julia's formula is NaN/inf below them (Re < 2.1, h/ks < 1/11.8).
    re = jnp.maximum(umag * h / NU, RE_MIN)
    h_ks = jnp.maximum(h / ks, HKS_MIN)
    alpha = 1.0 / (1.0 + (re / 850.0) ** 9)
    beta = 1.0 / (1.0 + (re / (160.0 * h_ks)) ** 2)
    f = 1.0 / ((re / 24.0) ** alpha
               * (1.8 * jnp.log10(re / 2.1)) ** (2.0 * (1.0 - alpha) * beta)
               * (2.0 * jnp.log10(11.8 * h_ks)) ** (2.0 * (1.0 - alpha) * (1.0 - beta)))
    n = jnp.sqrt(f / 8.0) * h ** (1.0 / 6.0) / jnp.sqrt(9.81)
    return n, h_ks, f, re


def manning_law(law: ManningLaw, h: Array, umag: Array, ks: Array) -> tuple[Array, Array, Array, Array]:
    """Evaluate a variable-n law per cell.

    Returns:
        (n, h/ks, f, Re); the last three are zeros except for ``h_Umag_ks``.
    """
    p = law.params
    if law.kind == "h_Umag_ks":
        return cheng_friction(h, umag, ks)
    lo, hi = p["n_lower"], p["n_upper"]
    if law.kind == "inverse":
        n = lo + (hi - lo) / (1.0 + p["k"] * h)
    elif law.kind == "power_law":
        n = lo + (hi - lo) * (h + EPS) ** (-p["k"])
    else:  # sigmoid
        n = lo + (hi - lo) * jax.nn.sigmoid(-p["k"] * (h - p["h_mid"]))  # = 1/(1+exp(k(h-h_mid)))
    z = jnp.zeros_like(h)
    return n, z, z, z
