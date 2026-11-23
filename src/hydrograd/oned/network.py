"""Channel-network topology for tree-shaped systems (tributaries joining, no loops).

Numbering
---------
The network has N cross-sections and M = N - 1 reaches. Sections are numbered in
topological order: reach i runs from section i to a downstream section ``downstream[i]``
with ``downstream[i] > i``, and the outlet (the dam) is the last section, N - 1. A single
channel is the special case ``downstream[i] = i + 1``.

Sections that no reach flows into are headwaters. Discharge and inflow load are prescribed
there and accumulate downstream: ``reach_map[i, h]`` is 1 where the water of headwater h
passes section i, so the discharge at all sections is ``reach_map @ Q_headwaters``.

Use by the solvers
------------------
- Hydraulics sweep from the outlet upstream. The energy equation of reach i couples the
  water surfaces of sections i and ``downstream[i]`` only, so the Newton system has two
  entries per row and is solved by back substitution in decreasing section order.
- Sediment routing sweeps in increasing order. At a junction the loads of the incoming
  reaches add; the capacity and supply of the junction section are shared among the
  incoming reaches in proportion to their discharge.

Deposition bookkeeping
----------------------
The deposit of a reach is assigned to its downstream section ("collocated" scheme, SRH-1D
eq. 3.2). A headwater has no incoming reach, so it gets a virtual inflow reach of length
``head_fraction`` times its own outgoing reach, over which the prescribed inflow load
relaxes to the headwater's capacity; otherwise its bed could never change.

The topology (``downstream``, ``headwaters``, ``reach_map``) is stored as NumPy arrays: it
is fixed data, not something to differentiate. The reach lengths are a JAX array.

Withdrawals
-----------
``withdrawals`` lists sections at which water leaves the channel sideways (an intake, a
diversion). The withdrawn discharge is taken out just below the section: the section
itself still carries its full discharge and every section downstream of it carries less.
The suspended load leaves in proportion to the water (a fully mixed section), so the
reach that starts at the section receives the fraction 1 - Q_withdrawal / Q of its load.
A negative withdrawal is a lateral inflow of clear water.

Where ``withdrawal_takes_load`` is False the exchange moves water only and the whole load
stays in the channel. This describes water that goes into storage and comes back out
(the rise and fall of a flood-control pool): the discharge along the lake then differs
from the inflow, Q(x) = Q_in - (rate of storage above x), while the sediment stays in the
flow.

Units: SI (lengths in m, discharges in m^3/s).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

__all__ = [
    "Network",
    "as_network",
    "chain_network",
    "control_volume_lengths",
    "deposit_index",
    "head_lengths",
    "headwater_matrix",
    "network_from_downstream",
    "network_from_positions",
    "passing_fraction",
    "reach_to_section",
    "section_discharge",
    "water_fraction",
    "with_withdrawals",
]

# Smallest control-volume length (m).
LENGTH_FLOOR = 1.0e-6


class Network(NamedTuple):
    """A tree-shaped channel network (see the module docstring for the numbering).

    Attributes:
        reach_length: length of each reach (m), shape [M].
        downstream: downstream section of each reach (integers), shape [M].
        headwaters: sections without an incoming reach (integers), shape [H].
        reach_map: 1.0 where headwater h drains through section i, shape [N, H].
        withdrawals: sections with a lateral withdrawal (integers), shape [W], or None.
        withdrawal_takes_load: for each withdrawal, True if the withdrawn water carries
            its share of the load away and False if the whole load stays in the channel
            (storage exchange), shape [W]; None means True for all.
        withdrawal_map: 1.0 where section i lies downstream of withdrawal w, shape
            [N, W], or None.
    """

    reach_length: Array
    downstream: np.ndarray
    headwaters: np.ndarray
    reach_map: np.ndarray
    withdrawals: np.ndarray | None = None
    withdrawal_map: np.ndarray | None = None
    withdrawal_takes_load: np.ndarray | None = None

    @property
    def n_reaches(self) -> int:
        """Number of reaches M."""
        return int(self.reach_length.shape[0])

    @property
    def n_sections(self) -> int:
        """Number of cross-sections N = M + 1."""
        return self.n_reaches + 1

    @property
    def n_headwaters(self) -> int:
        """Number of headwater sections H."""
        return int(self.headwaters.shape[0])

    @property
    def is_chain(self) -> bool:
        """True for a single channel (``downstream[i] = i + 1``)."""
        expected = np.arange(1, self.n_reaches + 1)
        return bool(np.array_equal(self.downstream, expected))


def network_from_downstream(reach_length, downstream) -> Network:
    """Build a network from the reach lengths and the downstream section of each reach.

    Args:
        reach_length: length of each reach (m), shape [M].
        downstream: downstream section of each reach, shape [M]; must satisfy
            ``i < downstream[i] <= M``.

    Returns:
        The network, with its headwaters and reach map.
    """
    reach_length = jnp.asarray(reach_length, dtype=float)
    downstream = np.asarray(downstream, dtype=np.int64)
    n_reaches = int(reach_length.shape[0])
    n_sections = n_reaches + 1

    # Check the numbering: one entry per reach, each reach ends downstream of its start.
    if downstream.shape[0] != n_reaches:
        raise ValueError("downstream must have one entry per reach (N - 1)")
    reach_index = np.arange(n_reaches)
    if np.any(downstream <= reach_index) or np.any(downstream > n_sections - 1):
        raise ValueError(
            "sections must be numbered so that downstream[i] > i and the outlet is N - 1"
        )

    # Headwaters: sections that are the downstream end of no reach.
    n_incoming = np.zeros(n_sections, dtype=np.int64)
    np.add.at(n_incoming, downstream, 1)
    headwaters = np.flatnonzero(n_incoming == 0)
    n_headwaters = int(headwaters.shape[0])

    # Reach map: start with 1 at each headwater and pass every row to the downstream
    # section. In topological order all contributions arrive before a section is read.
    reach_map = np.zeros((n_sections, n_headwaters))
    reach_map[headwaters, np.arange(n_headwaters)] = 1.0
    for i in range(n_reaches):
        reach_map[downstream[i]] = reach_map[downstream[i]] + reach_map[i]

    return Network(
        reach_length=reach_length,
        downstream=downstream,
        headwaters=headwaters,
        reach_map=reach_map,
    )


def chain_network(reach_length) -> Network:
    """A single channel from its reach lengths.

    Args:
        reach_length: length of each reach (m), shape [M], upstream to downstream.

    Returns:
        The chain network with N = M + 1 sections.
    """
    n_reaches = int(np.shape(reach_length)[0])
    return network_from_downstream(reach_length, np.arange(1, n_reaches + 1))


def network_from_positions(x) -> Network:
    """A single channel from the section positions.

    Args:
        x: streamwise position of each section (m), increasing downstream, shape [N].

    Returns:
        The chain network.
    """
    x = jnp.asarray(x, dtype=float)
    return chain_network(x[1:] - x[:-1])


def as_network(network_or_lengths) -> Network:
    """Accept a :class:`Network` or the reach lengths of a single channel.

    Args:
        network_or_lengths: a network, or reach lengths (m), shape [M].

    Returns:
        The network.
    """
    if isinstance(network_or_lengths, Network):
        return network_or_lengths
    return chain_network(network_or_lengths)


def with_withdrawals(network: Network, sections, takes_load=True) -> Network:
    """The same network with lateral withdrawals at the given sections.

    Args:
        network: the network.
        sections: sections just below which water is withdrawn (integers), shape [W];
            the outlet is not allowed.
        takes_load: whether the withdrawn water carries its share of the load away; one
            value for all withdrawals or one per withdrawal, shape [W].

    Returns:
        The network with ``withdrawals``, ``withdrawal_map`` and
        ``withdrawal_takes_load`` set.
    """
    sections = np.atleast_1d(np.asarray(sections, dtype=np.int64))
    n_sections = network.n_sections
    if np.any(sections < 0) or np.any(sections >= n_sections - 1):
        raise ValueError("a withdrawal must lie at a section above the outlet")

    # Sections downstream of each withdrawal, by following the reaches to the outlet.
    below = np.zeros((n_sections, sections.shape[0]))
    for column, section in enumerate(sections):
        current = int(network.downstream[section])
        while True:
            below[current, column] = 1.0
            if current == n_sections - 1:
                break
            current = int(network.downstream[current])
    takes = np.broadcast_to(np.asarray(takes_load, dtype=bool), sections.shape).copy()
    return network._replace(withdrawals=sections, withdrawal_map=below, withdrawal_takes_load=takes)


def section_discharge(
    network: Network, Q_headwaters: Array, Q_withdrawal: Array | None = None,
) -> Array:
    """Discharge at every section from the headwater discharges.

    Args:
        network: the network.
        Q_headwaters: discharge of each headwater (m^3/s), shape [H]; a scalar or a single
            value is given to every headwater.
        Q_withdrawal: discharge of each lateral withdrawal (m^3/s), shape [W], smaller
            than the discharge arriving at its section; None for a network without
            withdrawals.

    Returns:
        Discharge Q (m^3/s) at the sections, shape [N].
    """
    Q = jnp.reshape(Q_headwaters, (-1,))

    # One value for all headwaters.
    if Q.shape[0] != network.n_headwaters:
        if Q.shape[0] != 1:
            raise ValueError("Q_headwaters must have one entry per headwater")
        Q = jnp.broadcast_to(Q, (network.n_headwaters,))

    discharge = jnp.asarray(network.reach_map) @ Q
    if network.withdrawals is None or Q_withdrawal is None:
        return discharge
    return discharge - jnp.asarray(network.withdrawal_map) @ jnp.reshape(Q_withdrawal, (-1,))


def water_fraction(network: Network, discharge: Array, Q_withdrawal: Array) -> Array:
    """Fraction of the water of each section that enters its outgoing reach.

    Args:
        network: a network with withdrawals.
        discharge: discharge at the sections (m^3/s), shape [N].
        Q_withdrawal: discharge of each withdrawal (m^3/s), shape [W].

    Returns:
        The fraction (-), shape [M]: 1 - Q_withdrawal / Q at a withdrawal section, not
        negative (above 1 for a lateral inflow), and 1 elsewhere.
    """
    sections = network.withdrawals
    fraction = 1.0 - jnp.reshape(Q_withdrawal, (-1,)) / discharge[sections]
    ones = jnp.ones(network.n_reaches, dtype=discharge.dtype)
    return ones.at[sections].set(jnp.maximum(fraction, 0.0))


def passing_fraction(network: Network, discharge: Array, Q_withdrawal: Array) -> Array:
    """Fraction of the load of each section that enters its outgoing reach.

    Args:
        network: a network with withdrawals.
        discharge: discharge at the sections (m^3/s), shape [N].
        Q_withdrawal: discharge of each withdrawal (m^3/s), shape [W].

    Returns:
        The fraction (-), shape [M]: the water fraction limited to [0, 1] (a lateral
        inflow brings no load); 1 below a withdrawal that takes no load.
    """
    fraction = jnp.minimum(water_fraction(network, discharge, Q_withdrawal), 1.0)
    if network.withdrawal_takes_load is None:
        return fraction
    keeps = np.zeros(network.n_reaches, dtype=bool)
    keeps[network.withdrawals[~network.withdrawal_takes_load]] = True
    return jnp.where(jnp.asarray(keeps), 1.0, fraction)


def headwater_matrix(network: Network, headwater_values: Array) -> Array:
    """Place values given at the headwaters into an array over all sections.

    Args:
        network: the network.
        headwater_values: values per size class at the headwaters, shape [K] or [K, H]
            (a single column is given to every headwater).

    Returns:
        Array of shape [K, N]: the values at the headwater sections, zero elsewhere.
    """
    n_classes = headwater_values.shape[0]
    values = jnp.reshape(headwater_values, (n_classes, -1))
    if values.shape[1] == 1 and network.n_headwaters > 1:
        values = jnp.broadcast_to(values, (n_classes, network.n_headwaters))

    placed = jnp.zeros((n_classes, network.n_sections), dtype=values.dtype)
    return placed.at[:, network.headwaters].add(values)


def head_lengths(network: Network, head_fraction: float) -> Array:
    """Length of the virtual inflow reach above each headwater.

    Args:
        network: the network.
        head_fraction: length of the virtual reach as a fraction (-) of the headwater's own
            outgoing reach; 0 freezes the bed of the headwaters.

    Returns:
        Lengths (m), shape [H].
    """
    return head_fraction * network.reach_length[network.headwaters]


def deposit_index(network: Network, with_heads: bool) -> np.ndarray:
    """Section that receives the deposit of each reach.

    Args:
        network: the network.
        with_heads: if True, the H virtual inflow reaches follow the M real reaches.

    Returns:
        Section indices, shape [M] or [M + H].
    """
    if with_heads:
        return np.concatenate([network.downstream, network.headwaters])
    return network.downstream


def control_volume_lengths(
    network: Network, scheme: str = "collocated", head_fraction: float = 1.0,
) -> Array:
    """Streamwise length of the control volume of each section.

    Args:
        network: the network.
        scheme: ``"collocated"``: a section owns the reaches that end at it (a junction
            owns all incoming reaches; a headwater owns its virtual inflow reach).
            ``"upstream_half"``: a section owns half of each adjacent reach.
        head_fraction: relative length of the virtual inflow reach (-), see
            :func:`head_lengths`.

    Returns:
        Lengths (m), shape [N].
    """
    n_reaches = network.n_reaches
    dx = network.reach_length
    zero = jnp.zeros(network.n_sections, dtype=dx.dtype)

    # Half of each adjacent reach.
    if scheme == "upstream_half":
        upstream_part = zero.at[np.arange(n_reaches)].add(0.5 * dx)
        return upstream_part.at[network.downstream].add(0.5 * dx)

    # Collocated: reaches that end at the section, plus the virtual reach of a headwater.
    lengths = zero.at[network.downstream].add(dx)
    lengths = lengths.at[network.headwaters].add(head_lengths(network, head_fraction))
    return jnp.maximum(lengths, LENGTH_FLOOR)


def reach_to_section(
    network: Network, deposition_reach: Array, scheme: str = "collocated",
) -> Array:
    """Assign the deposition of each reach to the sections.

    Args:
        network: the network.
        deposition_reach: deposition rate of each reach per size class (m^3/s of solids),
            shape [K, M], or [K, M + H] with the virtual inflow reaches appended.
        scheme: ``"collocated"`` (all to the downstream section) or ``"upstream_half"``
            (half to each end).

    Returns:
        Deposition rate at the sections (m^3/s of solids), shape [K, N].
    """
    n_classes = deposition_reach.shape[0]
    n_reaches = network.n_reaches
    zero = jnp.zeros((n_classes, network.n_sections), dtype=deposition_reach.dtype)

    # Half to each end of the reach.
    if scheme == "upstream_half":
        upstream_part = zero.at[:, np.arange(n_reaches)].add(0.5 * deposition_reach)
        return upstream_part.at[:, network.downstream].add(0.5 * deposition_reach)

    # Collocated: to the section where the reach ends.
    with_heads = deposition_reach.shape[1] > n_reaches
    index = deposit_index(network, with_heads)
    return zero.at[:, index].add(deposition_reach)
