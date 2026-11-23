"""Source and rate of the non-equilibrium total-load equation.

Governing form
--------------
Along the flow direction x the total load Q_t,k (m^3/s of solids) of size class k follows

    d Q_t,k / dx = S_k - r_k Q_t,k                                            (m^2/s)

with a source S_k (m^2/s) and a rate r_k (1/m). The right-hand side is the volume of
solids the flow takes from the bed per time and per length of channel, so

    net erosion per length = S_k - r_k Q_t,k,     net deposition per length = r_k Q_t,k - S_k.

The same expression serves 2-D transport: with closures evaluated per unit width (B = 1,
see :mod:`hydrograd.sediment.closures`) and Q_t replaced by the unit load q_t (m^2/s),
S - r q_t is the net erosion flux per unit bed area (m/s), the bed exchange term of the
depth-averaged transport equation and of the Exner equation.

Non-cohesive classes
--------------------
S = Q* / L and r = 1 / L (SRH-1D eq. 3.4), i.e. relaxation to the capacity Q* over the
adaptation length (eq. 3.6)

    L = f_s L_s + (1 - f_s) L_b,     L_s = sw L_s(alpha_d) + (1 - sw) L_s(alpha_e).

sw is a smooth deposition/erosion switch, sw = sigmoid((Q_t - Q*) / (Q_t + Q*) / eps_switch):
1 where the load exceeds the capacity (deposition), 0 where it is below (erosion).

Supply limit
------------
The bed can only give what its active layer holds. With ``supply_cap`` c (m^2/s; the
volume of the class available per time and per length), the erosion excess
X = max(Q* - Q_t, 0) / L is limited to X c / (X + c), which tends to X for X << c and to c
for X >> c:

    S = Q* / L - (X - X c / (X + c)).

Deposition towards the capacity is not limited. The cohesive source is pure erosion and
is limited in the same way, S = S_coh c / (S_coh + c); the cohesive rate is unchanged.

Lagged load
-----------
The switch and the erosion excess use a load supplied by the caller (``load_lagged``). The
1-D routing passes the load of the previous time step, which keeps the switch out of the
fixed point of the current step; this is one of the conditions for a stable adjoint.

At Q* = Q_t exactly, the derivative of max(Q* - Q_t, 0) is 1/2 (the convention of JAX at
a tie).

Units: SI.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from hydrograd.sediment.closures import GUARD, ClosureState
from hydrograd.sediment.config import SedimentConfig
from hydrograd.smooth import sigmoid_eps

__all__ = ["select_locations", "source_and_rate"]

# Smallest adaptation length (m).
LENGTH_FLOOR = 1.0e-3

# Guard of the supply limiter X c / (X + c) (m^2/s): it keeps 0 / 0 out where both the
# erosion excess X and the supply c are zero.
SUPPLY_GUARD = 1.0e-30


def select_locations(closures: ClosureState, index: Array) -> ClosureState:
    """Closure results at a subset (or a reordering) of the locations.

    The 1-D routing needs the closures of each reach at the cross-section where the reach
    deposits (its downstream section); ``index`` then lists that section for every reach.

    Args:
        closures: closure results at N locations.
        index: integer indices into the N locations, shape [M].

    Returns:
        Closure results at the M selected locations (arrays of shape [K, M] and [M]).
    """
    return ClosureState(
        settling_velocity=closures.settling_velocity,
        suspended_fraction=closures.suspended_fraction[:, index],
        capacity=closures.capacity[:, index],
        capacity_per_fraction=closures.capacity_per_fraction[:, index],
        length_suspended_deposition=closures.length_suspended_deposition[:, index],
        length_suspended_erosion=closures.length_suspended_erosion[:, index],
        length_bedload=closures.length_bedload[index],
        cohesive_source=closures.cohesive_source[:, index],
        cohesive_rate=closures.cohesive_rate[:, index],
    )


def source_and_rate(
    closures: ClosureState, load_lagged: Array, supply_cap: Array, config: SedimentConfig,
) -> tuple[Array, Array]:
    """Source S and rate r of d Q_t / dx = S - r Q_t for every class and location.

    Args:
        closures: closure results at the M locations where the exchange takes place
            (use :func:`select_locations` to pick them).
        load_lagged: load Q_t (m^3/s of solids) used for the deposition/erosion switch and
            the erosion excess, shape [K, M].
        supply_cap: volume of each class the bed can supply per time and per length
            (m^2/s), shape [K, M].
        config: sediment settings (cohesive classes, switch width).

    Returns:
        Tuple ``(source, rate)``: S (m^2/s of solids) and r (1/m), each of shape [K, M].
    """
    is_cohesive = config.cohesive_mask()[:, None]
    capacity = closures.capacity
    f_s = closures.suspended_fraction

    # Smooth switch: 1 where the load exceeds the capacity (deposition), 0 in erosion.
    relative_excess = (load_lagged - capacity) / (load_lagged + capacity + GUARD)
    switch = sigmoid_eps(relative_excess, config.smoothing.eps_switch)

    # Adaptation length: suspended part with the recovery coefficient chosen by the
    # switch, bed-load part, weighted by the suspended fraction.
    length_deposition = closures.length_suspended_deposition
    length_erosion = closures.length_suspended_erosion
    length_suspended = switch * length_deposition + (1.0 - switch) * length_erosion
    length = f_s * length_suspended + (1.0 - f_s) * closures.length_bedload[None, :]
    length = jnp.maximum(length, LENGTH_FLOOR)

    # Non-cohesive classes: relaxation to the capacity, with only the erosion excess
    # limited by the supply of the bed. The hard maximum (not the smooth switch) is used,
    # so that nothing passes the supply limit.
    excess = jnp.maximum(capacity - load_lagged, 0.0) / length
    excess_limited = excess * supply_cap / (excess + supply_cap + SUPPLY_GUARD)
    source_noncohesive = capacity / length - (excess - excess_limited)
    rate_noncohesive = 1.0 / length

    # Cohesive classes: the erosion source is limited by the supply directly.
    source_cohesive = closures.cohesive_source
    source_cohesive = source_cohesive * supply_cap / (source_cohesive + supply_cap + SUPPLY_GUARD)

    # Pick the cohesive or the non-cohesive form for each class.
    source = jnp.where(is_cohesive, source_cohesive, source_noncohesive)
    rate = jnp.where(is_cohesive, closures.cohesive_rate, rate_noncohesive)

    return source, rate
