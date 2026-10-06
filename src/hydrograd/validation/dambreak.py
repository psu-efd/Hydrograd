"""Exact 1D dam-break solutions (flat, frictionless channel), for verification.

Initial state: still water, depth h_l for x < x0 and h_r for x > x0 (m).

* Stoker (1957), h_r > 0: rarefaction | constant state (h_m, u_m) | bore.
  The middle state solves  2 (c_l - c_m) = (h_m - h_r) sqrt(g (h_m + h_r) / (2 h_m h_r)),
  with c = sqrt(g h); bore speed S = h_m u_m / (h_m - h_r).
* Ritter (1892), h_r = 0: rarefaction from x0 - c_l t to the wet front x0 + 2 c_l t.

Inside the rarefaction: u = 2/3 (c_l + xi), h = (2 c_l - xi)^2 / (9 g), xi = (x - x0)/t.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

__all__ = ["dam_break", "stoker_middle_state"]


def stoker_middle_state(h_l: float, h_r: float, g: float = 9.81) -> tuple[float, float, float]:
    """(h_m (m), u_m (m/s), bore speed S (m/s)) of the wet-bed dam break."""
    c_l = np.sqrt(g * h_l)

    def f(h_m: float) -> float:
        return 2.0 * (c_l - np.sqrt(g * h_m)) - (h_m - h_r) * np.sqrt(g * (h_m + h_r) / (2.0 * h_m * h_r))

    h_m = brentq(f, h_r * (1 + 1e-12), h_l)
    u_m = 2.0 * (c_l - np.sqrt(g * h_m))
    return h_m, u_m, h_m * u_m / (h_m - h_r)


def dam_break(x: np.ndarray, t: float, h_l: float, h_r: float, x0: float = 0.0,
              g: float = 9.81) -> tuple[np.ndarray, np.ndarray]:
    """Exact depth h (m) and velocity u (m/s) at positions x (m), time t > 0 (s)."""
    x = np.asarray(x, dtype=float)
    c_l = np.sqrt(g * h_l)
    xi = (x - x0) / t
    h = np.where(xi < -c_l, h_l, 0.0)
    u = np.zeros_like(x)
    raref_h = (2.0 * c_l - xi) ** 2 / (9.0 * g)
    raref_u = 2.0 / 3.0 * (c_l + xi)
    if h_r <= 0.0:  # Ritter
        inr = (xi >= -c_l) & (xi <= 2.0 * c_l)
        h = np.where(inr, raref_h, h)
        u = np.where(inr, raref_u, u)
        return h, u
    h_m, u_m, S = stoker_middle_state(h_l, h_r, g)
    c_m = np.sqrt(g * h_m)
    inr = (xi >= -c_l) & (xi <= u_m - c_m)
    mid = (xi > u_m - c_m) & (xi <= S)
    h = np.where(inr, raref_h, np.where(mid, h_m, np.where(xi > S, h_r, h)))
    u = np.where(inr, raref_u, np.where(mid, u_m, u))
    return h, u
