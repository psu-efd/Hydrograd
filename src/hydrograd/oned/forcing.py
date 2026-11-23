"""Boundary conditions of a 1-D run as time series, one entry per morphological step.

Each step has its own length dt, so a long record can be represented by a few flow classes
of different duration (a flow-duration curve) as well as by a regular hydrograph.

Inflow load: if ``load_in`` is None, the load of each headwater follows the rating curve

    Q_t,in,k = p_in,k a_r Q^b_r                                   (m^3/s of solids)

with the calibrated parameters ``a_r``, ``b_r`` and the inflow size fractions ``p_in``.

Units: SI.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
from jax import Array

__all__ = ["Forcing", "constant_forcing", "rating_curve_load"]

# Differentiable parameters (see :func:`hydrograd.oned.config.default_parameters`).
Theta = dict[str, Any]


class Forcing(NamedTuple):
    """Boundary conditions of T morphological steps.

    Attributes:
        Q_in: inflow discharge (m^3/s), shape [T] (the same for every headwater) or [T, H]
            (one column per headwater).
        eta_outlet: water-surface elevation at the outlet (m), shape [T].
        dt: length of each step (s), shape [T].
        load_in: inflow load per size class (m^3/s of solids), shape [K, T] (the same for
            every headwater) or [K, T, H]; None uses the rating curve.
        Q_withdrawal: discharge of the lateral withdrawals of the network (m^3/s), shape
            [T] (one withdrawal) or [T, W]; None for no withdrawal. It must stay below
            the discharge that arrives at the withdrawal section.
    """

    Q_in: Array
    eta_outlet: Array
    dt: Array
    load_in: Array | None = None
    Q_withdrawal: Array | None = None

    @property
    def n_steps(self) -> int:
        """Number of steps T."""
        return int(self.Q_in.shape[0])

    def time_end(self) -> Array:
        """Time at the end of each step (s), shape [T]."""
        return jnp.cumsum(self.dt)


def constant_forcing(Q: float, eta_outlet: float, dt: float, n_steps: int) -> Forcing:
    """Steady boundary conditions.

    Args:
        Q: inflow discharge (m^3/s).
        eta_outlet: water-surface elevation at the outlet (m).
        dt: step length (s).
        n_steps: number of steps T.

    Returns:
        The forcing, without an inflow load (set ``load_in`` with ``_replace``).
    """
    return Forcing(
        Q_in=jnp.full(n_steps, Q),
        eta_outlet=jnp.full(n_steps, eta_outlet),
        dt=jnp.full(n_steps, dt),
    )


def rating_curve_load(theta: Theta, Q_headwaters: Array) -> Array:
    """Inflow load of each headwater from the rating curve p_in,k a_r Q^b_r.

    Args:
        theta: parameters ``p_in`` (-, shape [K]), ``a_r`` and ``b_r`` (scalars).
        Q_headwaters: discharge of each headwater (m^3/s), shape [H].

    Returns:
        Load (m^3/s of solids), shape [K, H].
    """
    total_load = theta["a_r"] * Q_headwaters ** theta["b_r"]
    return theta["p_in"][:, None] * total_load[None, :]
