"""Exact 1D dam-break solutions (flat, frictionless channel), for verification.

Initial state
-------------
Still water, depth h_l for x < x0 and h_r for x > x0 (m); g is the gravitational
acceleration (m/s^2) and c = sqrt(g h) the celerity (m/s).

Solutions
---------
* Stoker (1957), h_r > 0: rarefaction | constant state (h_m, u_m) | bore.
  The middle state solves  2 (c_l - c_m) = (h_m - h_r) sqrt(g (h_m + h_r) / (2 h_m h_r)),
  with u_m = 2 (c_l - c_m); bore speed S = h_m u_m / (h_m - h_r).
* Ritter (1892), h_r = 0: rarefaction from x0 - c_l t to the wet front x0 + 2 c_l t.

Inside the rarefaction: u = 2/3 (c_l + xi), h = (2 c_l - xi)^2 / (9 g), xi = (x - x0)/t.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

__all__ = ["dam_break", "stoker_middle_state"]


def stoker_middle_state(h_l: float, h_r: float, g: float = 9.81) -> tuple[float, float, float]:
    """(h_m (m), u_m (m/s), bore speed S (m/s)) of the wet-bed dam break.

    Args:
        h_l: upstream depth (m).
        h_r: downstream depth (m), 0 < h_r < h_l.
        g: gravitational acceleration (m/s^2).

    Returns:
        (h_m, u_m, S): middle-state depth (m) and velocity (m/s), and bore speed (m/s).
    """
    c_l = np.sqrt(g * h_l)

    def f(h_m: float) -> float:
        """Residual of the middle-state equation (m/s); zero at the Stoker depth h_m (m)."""
        rarefaction_side = 2.0 * (c_l - np.sqrt(g * h_m))
        bore_side = (h_m - h_r) * np.sqrt(g * (h_m + h_r) / (2.0 * h_m * h_r))
        return rarefaction_side - bore_side

    # The middle depth lies between h_r and h_l.
    h_m = brentq(f, h_r * (1 + 1e-12), h_l)
    u_m = 2.0 * (c_l - np.sqrt(g * h_m))
    bore_speed = h_m * u_m / (h_m - h_r)

    return h_m, u_m, bore_speed


def dam_break(x: np.ndarray, t: float, h_l: float, h_r: float, x0: float = 0.0,
              g: float = 9.81) -> tuple[np.ndarray, np.ndarray]:
    """Exact depth h (m) and velocity u (m/s) at positions x (m), time t > 0 (s).

    Args:
        x: positions (m), any shape.
        t: time after the dam break (s), > 0.
        h_l: upstream depth (m).
        h_r: downstream depth (m); 0 gives the dry-bed (Ritter) solution.
        x0: dam position (m).
        g: gravitational acceleration (m/s^2).

    Returns:
        (h, u): depth (m) and velocity (m/s), each with the shape of ``x``.
    """
    x = np.asarray(x, dtype=float)
    c_l = np.sqrt(g * h_l)

    # Similarity variable xi = (x - x0)/t (m/s); undisturbed upstream state for xi < -c_l.
    xi = (x - x0) / t
    h = np.where(xi < -c_l, h_l, 0.0)
    u = np.zeros_like(x)

    # Rarefaction-fan values (used where xi lies inside the fan).
    raref_h = (2.0 * c_l - xi) ** 2 / (9.0 * g)
    raref_u = 2.0 / 3.0 * (c_l + xi)

    # Ritter (dry bed): the fan reaches the wet front at xi = 2 c_l.
    if h_r <= 0.0:
        inr = (xi >= -c_l) & (xi <= 2.0 * c_l)
        h = np.where(inr, raref_h, h)
        u = np.where(inr, raref_u, u)
        return h, u

    # Stoker (wet bed): fan | middle state up to the bore | undisturbed downstream state.
    h_m, u_m, S = stoker_middle_state(h_l, h_r, g)
    c_m = np.sqrt(g * h_m)
    inr = (xi >= -c_l) & (xi <= u_m - c_m)
    mid = (xi > u_m - c_m) & (xi <= S)
    downstream = xi > S

    h = np.where(inr, raref_h, np.where(mid, h_m, np.where(downstream, h_r, h)))
    u = np.where(inr, raref_u, np.where(mid, u_m, u))

    return h, u
