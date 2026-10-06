"""Manning's n parameterizations (port of ``process_ManningN_2D.jl``).

Variable-n laws
---------------
Laws for a *variable* n (forward ``ManningN_option = "variable"``), with ``h`` the water
depth (m), ``U`` the speed (m/s), n_lo and n_hi the lower and upper bounds of n
(s/m^(1/3)), and k, h_mid shape parameters:

* ``inverse``   : n = n_lo + (n_hi - n_lo) / (1 + k h)
* ``power_law`` : n = n_lo + (n_hi - n_lo) (h + eps)^(-k)
* ``sigmoid``   : n = n_lo + (n_hi - n_lo) / (1 + exp(+k (h - h_mid)))
                  (Julia uses ``+k``: n *decreases* from n_hi to n_lo with depth)
* ``h_Umag_ks`` : Cheng (2008) friction factor f(Re, h/ks), n = sqrt(f/8) h^(1/6) / sqrt(g),
                  with nu = 1e-6 m^2/s and g = 9.81 m/s^2 hard-coded (SI only, as in Julia).

Cheng (2008) friction factor
----------------------------
With Re = U h / nu and ks the roughness height (m):

    1/f = (Re/24)^alpha (1.8 log10(Re/2.1))^(2 (1 - alpha) beta)
          (2 log10(11.8 h/ks))^(2 (1 - alpha)(1 - beta)),
    alpha = 1 / (1 + (Re/850)^9),    beta = 1 / (1 + (Re / (160 h/ks))^2).

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
    """A named n(h[, U, ks]) law with its parameters (``run_control.json`` keys).

    Attributes:
        kind: ``inverse``, ``power_law``, ``sigmoid`` or ``h_Umag_ks``.
        params: law parameters: ``n_lower``, ``n_upper`` (s/m^(1/3)), ``k`` (1/m for
            ``inverse`` and ``sigmoid``, - for ``power_law``), ``h_mid`` (m, ``sigmoid``).
    """

    kind: str
    params: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check the law name and its parameters.

        Raises:
            ValueError: on an unknown law, k <= 0 (inverse, sigmoid) or h_mid <= 0 (sigmoid).
        """
        if self.kind not in ("inverse", "power_law", "sigmoid", "h_Umag_ks"):
            raise ValueError(f"unknown ManningN function type '{self.kind}'")
        if self.kind in ("inverse", "sigmoid") and not self.params["k"] > 0:
            raise ValueError("ManningN law requires k > 0")
        if self.kind == "sigmoid" and not self.params["h_mid"] > 0:
            raise ValueError("sigmoid ManningN law requires h_mid > 0")

    def __hash__(self) -> int:
        """Hash of the law and its parameters, so it is usable as a static jit argument.

        Returns:
            Hash of (kind, sorted parameter items).
        """
        return hash((self.kind, tuple(sorted(self.params.items()))))


def cheng_friction(h: Array, umag: Array, ks: Array) -> tuple[Array, Array, Array, Array]:
    """Cheng (2008) Darcy-Weisbach friction factor for flow over rough beds.

    Args:
        h: depth (m), shape [n_cells].
        umag: speed (m/s), shape [n_cells].
        ks: roughness height (m), shape [n_cells].

    Returns:
        (n, h/ks, f, Re), each of shape [n_cells]: Manning's n (s/m^(1/3)), relative
        submergence (-), friction factor (-), Reynolds number U h / nu (-).
    """
    # Floors keep the logs positive; Julia's formula is NaN/inf below them
    # (Re < 2.1, h/ks < 1/11.8).
    re = jnp.maximum(umag * h / NU, RE_MIN)
    h_ks = jnp.maximum(h / ks, HKS_MIN)

    # Blending weights between the laminar, smooth-turbulent and rough-turbulent laws.
    alpha = 1.0 / (1.0 + (re / 850.0) ** 9)
    beta = 1.0 / (1.0 + (re / (160.0 * h_ks)) ** 2)

    # 1/f as the product of the three regime factors.
    laminar = (re / 24.0) ** alpha
    smooth = (1.8 * jnp.log10(re / 2.1)) ** (2.0 * (1.0 - alpha) * beta)
    rough = (2.0 * jnp.log10(11.8 * h_ks)) ** (2.0 * (1.0 - alpha) * (1.0 - beta))
    f = 1.0 / (laminar * smooth * rough)

    # Manning's n from f: n = sqrt(f/8) h^(1/6) / sqrt(g), g = 9.81 m/s^2.
    n = jnp.sqrt(f / 8.0) * h ** (1.0 / 6.0) / jnp.sqrt(9.81)

    return n, h_ks, f, re


def manning_law(
    law: ManningLaw, h: Array, umag: Array, ks: Array,
) -> tuple[Array, Array, Array, Array]:
    """Evaluate a variable-n law per cell.

    Args:
        law: the law and its parameters.
        h: water depth (m), shape [n_cells].
        umag: speed (m/s), shape [n_cells]; used only by ``h_Umag_ks``.
        ks: roughness height (m), shape [n_cells]; used only by ``h_Umag_ks``.

    Returns:
        (n, h/ks, f, Re), each of shape [n_cells]: Manning's n (s/m^(1/3)); the last three
        are zeros except for ``h_Umag_ks``.
    """
    p = law.params

    # Cheng's law returns all four fields.
    if law.kind == "h_Umag_ks":
        return cheng_friction(h, umag, ks)

    # Depth-only laws between n_lo and n_hi.
    lo = p["n_lower"]
    hi = p["n_upper"]
    if law.kind == "inverse":
        n = lo + (hi - lo) / (1.0 + p["k"] * h)
    elif law.kind == "power_law":
        n = lo + (hi - lo) * (h + EPS) ** (-p["k"])
    else:
        # sigmoid(-k (h - h_mid)) = 1 / (1 + exp(k (h - h_mid))).
        n = lo + (hi - lo) * jax.nn.sigmoid(-p["k"] * (h - p["h_mid"]))

    z = jnp.zeros_like(h)
    return n, z, z, z
