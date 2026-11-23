"""Non-equilibrium total-load routing on a network, with a locally implicit bed feedback.

Explicit form
-------------
Along reach i (from section i to its downstream section j), with the source S and the
rate r of :mod:`hydrograd.sediment.exchange` taken at the section where the reach
deposits, d Q_t / dx = S - r Q_t has the exact solution

    Q_out = Q_in e^(-a) + Q_eff (1 - e^(-a)),     a = r dx,     Q_eff = S / r,

(SRH-1D eq. 3.4 for the non-cohesive classes and eq. 3.10 for cohesive settling). The
deposition of the reach, Q_in - Q_out, is assigned to the downstream section (eq. 3.2,
"collocated"). Evaluating the closures and placing the deposit at the same section makes
a bed perturbation damp itself; evaluating upstream and depositing downstream produces an
odd-even instability.

Locally implicit form
---------------------
The capacity of a section depends strongly on its own depth (about h^-5.5 for
Engelund-Hansen with Manning friction) and on its active-layer fractions. With a daily
step in a river 2 m deep, the explicit bed-capacity loop has a gain of order 10 and the
solution blows up. The capacity is therefore linearized in the bed change of the same
section within the step,

    Q*_k <- Q*_k + s_k dz + (q_k / delta_a) (dz_k - p_k dz),

with s_k = d Q*_k / d z_b, q_k = Q*_k / p_k, the bed change dz_k = c_k (Q_in,k - Q_out,k)
of class k and the Exner factor c_k. With f_k = 1 - e^(-a_k) this gives

    Q_out = A Q_in + b,     A = I - diag(1 / c) G,     b = diag(1 / c) G Q_eff,
    G = (I + M)^-1 diag(c f),     M_kj = c_k f_k [s_k + (q_k / delta_a) (delta_kj - p_k)].

(``implicit_coupling="full"``). The row index k of the composition term carries the
fraction: the layer fraction of class k changes by (dz_k - p_k dz) / delta_a, and dz is
the sum of the dz_j of all classes.

``implicit_coupling="diagonal"`` (default) keeps only the diagonal of M:

    Q_out,k = (1 - f_k d_k) Q_in,k + f_k d_k Q_eff,k,
    d_k = 1 / (1 + c_k f_k g_k),     g_k = s_k + (q_k / delta_a) (1 - p_k),

so that each relaxation factor f_k d_k stays in (0, 1) and the loads stay non-negative.
For s = q = 0 this is the explicit form. Mass conservation is not affected (the
deposition is still Q_in - Q_out); the linearization only changes how far the load
relaxes within the step, like an implicit Euler step on the stiff local mode.

Only a stabilizing depth feedback is taken implicitly: both forms use max(s_k, 0). The
capacity normally grows when the bed rises under a fixed water surface (s > 0), and the
implicit term then damps the step. Where the water just spills onto a wide floodplain a
bed rise takes width away and the capacity falls (s < 0); the linearized feedback would
then amplify the step, without bound as c f g approaches -1 (and I + M becomes singular).
Such a sensitivity holds only for a bed change of millimetres, so it is left out and the
step falls back towards the explicit form there. With s >= 0, g_k >= 0 and I + M is
non-singular.

Network sweep
-------------
Sections are numbered so that every reach ends downstream of its start. One sweep in
increasing order therefore finds the load of section i complete when reach i is
processed; loads add at junctions. Each headwater has a virtual inflow reach in which the
prescribed inflow load relaxes to the headwater's own capacity (see
:mod:`hydrograd.oned.network`). At a junction the capacity, source and supply of the
section are divided among the incoming reaches in proportion to their discharge
(``share``). Below a lateral withdrawal only the fraction ``passing`` of the load of the
section enters the reach; the rest leaves with the withdrawn water.

Lagged switch
-------------
The deposition/erosion switch and the erosion excess use the load of the previous time
step, which keeps them out of the fixed point of the current step. By default this load
is differentiated through (exact). ``lag_switch=True`` holds it constant in derivatives,
and ``attached_sweeps > 0`` iterates the switch within the step; both exist for the
adjoint-stability study.

Units: SI (loads in m^3/s of solids).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hydrograd.oned.config import OneDConfig
from hydrograd.oned.network import Network, head_lengths
from hydrograd.sediment.closures import ClosureState
from hydrograd.sediment.exchange import select_locations, source_and_rate
from hydrograd.smooth import expm1_ratio

__all__ = ["SedimentState", "route"]


class SedimentState(NamedTuple):
    """Result of routing the load through the network for one step.

    Attributes:
        load: total load Q_t at the sections (m^3/s of solids), shape [K, N].
        deposition_reach: deposition rate of each reach, positive for deposition (m^3/s of
            solids), shape [K, M], or [K, M + H] with the virtual inflow reaches appended.
    """

    load: Array
    deposition_reach: Array


def _lag(config: OneDConfig, value: Array) -> Array:
    """Hold ``value`` constant in derivatives if ``lag_switch`` is on.

    Args:
        config: settings of the 1-D mode.
        value: any array.

    Returns:
        ``value``, or ``stop_gradient(value)``.
    """
    if config.morph.lag_switch:
        return jax.lax.stop_gradient(value)
    return value


def _reach_coefficients(
    config: OneDConfig,
    f: Array,
    f_Q_eff: Array,
    exner: Array | None,
    capacity_sensitivity: Array,
    capacity_per_fraction: Array,
    p_active: Array | None,
    delta_a: Array | None,
) -> tuple[Array, Array]:
    """Coefficients A and b of the recurrence Q_out = A Q_in + b of every reach.

    Args:
        config: settings of the 1-D mode (``implicit_exner``, ``implicit_coupling``).
        f: relaxation factor 1 - exp(-a) (-), shape [K, R] for R reaches.
        f_Q_eff: f Q_eff (m^3/s), shape [K, R].
        exner: Exner factor c at the depositing sections (s/m^2), shape [K, R], or None
            for the explicit form.
        capacity_sensitivity: s = d Q* / d z_b (m^2/s), shape [K, R]; negative values
            are taken as zero.
        capacity_per_fraction: q = Q* / p (m^3/s), shape [K, R].
        p_active: active-layer fractions (-), shape [K, R].
        delta_a: active-layer thickness (m), scalar.

    Returns:
        Tuple ``(A, b)``. ``b`` has shape [R, K]. ``A`` has shape [R, K] (the diagonal)
        for the explicit and the diagonal implicit form, and [R, K, K] for full coupling.
    """
    morph = config.morph

    # Explicit recurrence.
    if exner is None or not morph.implicit_exner:
        return (1.0 - f).T, f_Q_eff.T

    c = exner
    # Only a stabilizing depth feedback (s > 0) is implicit; see the module docstring.
    s = jnp.maximum(capacity_sensitivity, 0.0)
    q = capacity_per_fraction

    if morph.implicit_coupling == "full":
        n_classes = f.shape[0]
        identity = jnp.eye(n_classes, dtype=f.dtype)[None]

        # M_kj = c_k f_k [s_k + (q_k / delta_a) (delta_kj - p_k)], shape [R, K, K].
        c_f = (c * f).T
        composition = (q / delta_a).T[:, :, None] * (identity - p_active.T[:, :, None])
        M = c_f[:, :, None] * (s.T[:, :, None] + composition)

        # H = diag(1 / c) (I + M)^-1 diag(c);  A = I - H diag(f);  b = H (f Q_eff).
        c_diagonal = jax.vmap(jnp.diag)(c.T)
        G = jnp.linalg.solve(identity + M, c_diagonal)
        H = (1.0 / c).T[:, :, None] * G
        A = identity - H * f.T[:, None, :]
        b = jnp.einsum("mkj,mj->mk", H, f_Q_eff.T)
        return A, b

    # Diagonal: each class sees its own depth feedback s_k and the diagonal part of the
    # composition feedback. The damping factor is in (0, 1] and finite as f -> 0.
    g = s + (q / delta_a) * (1.0 - p_active)
    damping = 1.0 / (1.0 + c * f * g)
    return (1.0 - f * damping).T, (f_Q_eff * damping).T


def _apply(A_reach: Array, load: Array) -> Array:
    """A Q for one reach, with A a diagonal (shape [K]) or a matrix (shape [K, K]).

    Args:
        A_reach: coefficient of one reach.
        load: load entering the reach (m^3/s), shape [K].

    Returns:
        A Q (m^3/s), shape [K].
    """
    if A_reach.ndim == 1:
        return A_reach * load
    return A_reach @ load


def _sweep(
    config: OneDConfig,
    network: Network,
    closures: ClosureState,
    capacity_sensitivity: Array,
    load_headwaters: Array,
    supply_cap: Array,
    load_lagged: Array,
    exner: Array | None,
    p_active: Array | None,
    delta_a: Array | None,
    share: Array | None,
    passing: Array | None,
) -> tuple[Array, Array]:
    """One downstream sweep of the routing.

    Args:
        config: settings of the 1-D mode.
        network: the network (N sections, M reaches, H headwaters).
        closures: closure results at the N sections.
        capacity_sensitivity: d Q* / d z_b (m^2/s), shape [K, N].
        load_headwaters: inflow load at each headwater (m^3/s), shape [K, H].
        supply_cap: supply limit of the bed (m^2/s), shape [K, N].
        load_lagged: load used for the switch and the erosion excess (m^3/s), shape [K, N].
        exner: Exner factor (s/m^2), shape [K, N], or None for the explicit form.
        p_active: active-layer fractions (-), shape [K, N] (implicit form).
        delta_a: active-layer thickness (m), scalar (implicit form).
        share: share of the downstream section's discharge carried by each reach (-),
            shape [M], or None for 1.
        passing: fraction of the load of its upstream section that enters each reach
            (-), shape [M], or None for 1.

    Returns:
        Tuple ``(load, deposition_reach)``; see :class:`SedimentState`.
    """
    morph = config.morph
    n_sections = closures.capacity.shape[1]
    n_reaches = n_sections - 1
    downstream = network.downstream
    headwaters = network.headwaters
    dtype = closures.capacity.dtype

    # Section at which the closures of each reach are evaluated: its downstream section
    # (collocated) or its upstream section (the "upstream" schemes of the study).
    collocated = not morph.deposition_scheme.startswith("upstream")
    section = downstream if collocated else np.arange(n_reaches)
    weight = jnp.ones(n_reaches, dtype=dtype) if share is None else share
    dx = network.reach_length
    entering = jnp.ones(n_reaches, dtype=dtype) if passing is None else passing
    load_upstream = load_lagged[:, :-1] * entering[None, :]

    # Virtual inflow reaches above the headwaters, appended after the M real reaches so
    # that one set of coefficients covers both.
    use_heads = collocated and morph.head_inflow_fraction > 0.0
    if use_heads:
        n_headwaters = headwaters.shape[0]
        section = np.concatenate([section, headwaters])
        weight = jnp.concatenate([weight, jnp.ones(n_headwaters, dtype=dtype)])
        dx = jnp.concatenate([dx, head_lengths(network, morph.head_inflow_fraction)])
        load_upstream = jnp.concatenate([load_upstream, _lag(config, load_headwaters)], axis=1)

    # Source and rate of every reach; a junction's capacity and supply are shared among
    # its incoming reaches.
    at_reaches = select_locations(closures, section)
    source, rate = source_and_rate(
        at_reaches, load_upstream, supply_cap[:, section] * weight[None, :], config.sediment,
    )
    source = source * weight[None, :]

    # Relaxation over the reach: f = 1 - exp(-a), and f Q_eff = S dx phi(a), which stays
    # finite as the rate goes to zero.
    a = rate * dx[None, :]
    f = -jnp.expm1(-a)
    f_Q_eff = source * dx[None, :] * expm1_ratio(a)

    # Recurrence coefficients, explicit or locally implicit.
    implicit = exner is not None and morph.implicit_exner
    A, b = _reach_coefficients(
        config,
        f,
        f_Q_eff,
        exner[:, section] if implicit else None,
        capacity_sensitivity[:, section],
        at_reaches.capacity_per_fraction,
        p_active[:, section] if implicit else None,
        delta_a,
    )

    # Loads at the headwaters: after the virtual inflow reach, or the inflow itself.
    load_sections = jnp.zeros((n_sections, load_headwaters.shape[0]), dtype=dtype)
    inflow = load_headwaters.T
    if use_heads:
        A_heads = A[n_reaches:]
        b_heads = b[n_reaches:]
        after_head = jax.vmap(_apply)(A_heads, inflow) + b_heads
        deposition_heads = inflow - after_head
        load_sections = load_sections.at[headwaters].set(after_head)
    else:
        load_sections = load_sections.at[headwaters].set(inflow)

    def route_reach(loads: Array, reach: Array) -> tuple[Array, Array]:
        """Pass the load of section ``reach`` through its reach and add it downstream."""
        outflow = _apply(A[reach], loads[reach] * entering[reach]) + b[reach]
        return loads.at[jnp.asarray(downstream)[reach]].add(outflow), outflow

    # Sweep over the real reaches in topological order.
    load_sections, outflows = jax.lax.scan(route_reach, load_sections, jnp.arange(n_reaches))
    load = load_sections.T

    # Deposition of each reach: load entering it minus its outflow.
    deposition_reach = load[:, :-1] * entering[None, :] - outflows.T
    if use_heads:
        deposition_reach = jnp.concatenate([deposition_reach, deposition_heads.T], axis=1)

    return load, deposition_reach


def route(
    config: OneDConfig,
    network: Network,
    closures: ClosureState,
    load_headwaters: Array,
    supply_cap: Array,
    load_previous: Array | None = None,
    capacity_sensitivity: Array | None = None,
    exner: Array | None = None,
    p_active: Array | None = None,
    delta_a: Array | None = None,
    share: Array | None = None,
    passing: Array | None = None,
) -> SedimentState:
    """Route the load through the network for one step.

    Args:
        config: settings of the 1-D mode.
        network: the network (N sections, M reaches, H headwaters).
        closures: closure results at the N sections.
        load_headwaters: inflow load at each headwater (m^3/s of solids), shape [K, H].
        supply_cap: supply limit of the bed (m^2/s), shape [K, N].
        load_previous: load at the sections in the previous step (m^3/s), shape [K, N];
            None (first step) uses the total inflow load at every section.
        capacity_sensitivity: d Q* / d z_b (m^2/s), shape [K, N]; None means zero.
        exner: Exner factor (s/m^2), shape [K, N]. Together with ``p_active`` and
            ``delta_a`` it enables the locally implicit bed feedback; None gives the
            explicit recurrence.
        p_active: active-layer fractions (-), shape [K, N].
        delta_a: active-layer thickness (m), scalar.
        share: share of the downstream section's discharge carried by each reach (-),
            shape [M]; None means 1 (a single channel).
        passing: fraction of the load of its upstream section that enters each reach
            (-), shape [M]; None means 1 (no lateral withdrawal).

    Returns:
        Load at the sections and deposition of the reaches (:class:`SedimentState`).
    """
    n_classes = closures.capacity.shape[0]
    n_sections = closures.capacity.shape[1]

    if capacity_sensitivity is None:
        capacity_sensitivity = jnp.zeros_like(closures.capacity)

    # Lagged load for the switch: the previous step's, or the total inflow at the start.
    if load_previous is None:
        total_inflow = jnp.sum(_lag(config, load_headwaters), axis=1)
        load_lagged = jnp.broadcast_to(total_inflow[:, None], (n_classes, n_sections))
    else:
        load_lagged = _lag(config, load_previous)

    arguments = (exner, p_active, delta_a, share, passing)
    load, deposition_reach = _sweep(
        config, network, closures, capacity_sensitivity, load_headwaters, supply_cap,
        load_lagged, *arguments,
    )

    # Study option: further sweeps with the switch at the load of the current step,
    # differentiated through. This is what makes the adjoint grow geometrically.
    for _ in range(int(config.morph.attached_sweeps)):
        load, deposition_reach = _sweep(
            config, network, closures, capacity_sensitivity, load_headwaters, supply_cap,
            load, *arguments,
        )

    return SedimentState(load=load, deposition_reach=deposition_reach)
