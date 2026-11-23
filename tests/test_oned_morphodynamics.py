"""Tests of the 1-D sediment routing, bed update, observation operators and time loop.

Bed composition (``hydrograd.sediment.bed``):

1. A thick deposit of one class turns the active layer into that class; erosion refills
   the layer from the substrate; fractions stay normalized; no supply on a bedrock floor.

Routing (``hydrograd.oned.routing``):

2. Pure settling of a cohesive class decays exponentially from reach to reach, and the
   deposition equals inflow minus outflow.
3. The virtual inflow reach lets a headwater deposit; without it the inflow passes the
   first section unchanged.
4. A non-cohesive load far above capacity decreases monotonically along the channel.

Time loop (``hydrograd.oned.stepper``):

5. A short run conserves sediment volume, erodes the steep upstream reach and fills the
   pool, and keeps the fractions normalized.
6. Sub-steps, a hydraulic solve every second step, recording intervals and bed snapshots.
7. The checkpoint setting changes neither the results nor the derivatives (to round-off),
   and the derivatives match central differences.
8. A non-erodible (fixed) bed does not erode by more than its active layer.
9. A Y-shaped network conserves sediment volume; its branches erode and its pool fills.

10. Fully coupled implicit routing (``implicit_coupling="full"``): identical to the
    diagonal coupling for a single class, volume-conserving, and without negative loads
    at the outlet of the Y-shaped network.

Observation:

11. Consolidated deposit volumes are smaller than uncompacted ones after several years.
12. The storage curve of a trapezoidal channel equals the analytic volume.

15. With a depth-weighted deposit the volume balance closes, and a section with a flat
    bottom and dry bank tops changes as with the uniform shift.

17. With a channel deficit of the deposit and a separate depth exponent of erosion the
    volume balance closes, and the deepest point of a compound section rises less than the
    ground beside it.

16. With ``tau_ce_fresh`` equal to ``tau_ce`` the run is unchanged; with a smaller value
    the volume balance closes, less sediment stays in the channel, and the derivative
    with respect to it is finite.

Implicit bed feedback of the routing:

13. A negative capacity sensitivity (capacity falling as the bed rises, as where the water
    just spills onto a floodplain) is left out of the implicit term: the loads equal
    those for zero sensitivity and stay finite and non-negative, for both couplings.

Lateral withdrawal:

14. A withdrawal lowers the discharge of the sections below it only; a zero withdrawal
    reproduces the run without one; the volume balance closes with the withdrawn load,
    in a single channel and on the Y-shaped network; less load reaches the outlet.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hydrograd.oned import (
    MorphNumerics,
    OneDConfig,
    constant_forcing,
    default_parameters,
    network_from_positions,
    run,
    solve,
)
from hydrograd.oned.backwater import sediment_flow_state
from hydrograd.oned.network import (
    control_volume_lengths,
    section_discharge,
    with_withdrawals,
)
from hydrograd.oned.observation import capacity_curve
from hydrograd.oned.routing import route
from hydrograd.oned.synthetic import trapezoid_channel, uniform_bed, y_network
from hydrograd.sediment import (
    BedComposition,
    InterfaceMix,
    SedimentConfig,
    bulk_volume_uncompacted,
    compute_closures,
    consolidated_volume,
    interface_parameters,
    update_composition,
)
from hydrograd.sediment.bed import available_thickness

DAY = 86400.0           # s


def _channel_case(n_steps: int = 20, dt: float = DAY, morph: MorphNumerics | None = None):
    """A 10 km trapezoidal channel with a 12 m deep pool at the dam and three classes.

    Args:
        n_steps: number of steps T.
        dt: step length (s).
        morph: morphodynamic settings, or None for the defaults.

    Returns:
        Tuple ``(config, network, theta, forcing, bed0)``.
    """
    config = OneDConfig(morph=morph if morph is not None else MorphNumerics())
    x, z, s = trapezoid_channel(length=10000.0, n_sections=21)
    network = network_from_positions(x)
    bed0 = uniform_bed(z, s, n_classes=3)
    theta = default_parameters(3)

    # Constant inflow of 200 m^3/s carrying 0.02, 0.01 and 0.005 m^3/s of the classes.
    load_in = jnp.broadcast_to(jnp.asarray([0.02, 0.01, 0.005])[:, None], (3, n_steps))
    forcing = constant_forcing(200.0, 112.0, dt, n_steps)._replace(load_in=load_in)

    return config, network, theta, forcing, bed0


def _run_jit(config, network, theta, forcing, bed0):
    """Run the model jit-compiled, with the settings and the network closed over.

    The network holds its topology as NumPy arrays, which must stay constants of the
    compiled function, so only the parameters are passed as an argument.
    """

    def run_with(parameters):
        """The run as a function of the parameters."""
        return run(config, network, parameters, forcing, bed0)

    return jax.jit(run_with)(theta)


def _routing_setup(diameters: tuple[float, ...]):
    """Hydraulics and closures of the channel for the routing tests.

    Args:
        diameters: grain diameters of the size classes (m).

    Returns:
        Tuple ``(config, network, theta, hydro, bed)``.
    """
    n_classes = len(diameters)
    config = OneDConfig(sediment=SedimentConfig(diameters=diameters))
    x, z, s = trapezoid_channel(length=10000.0, n_sections=21)
    network = network_from_positions(x)
    bed = uniform_bed(z, s, n_classes=n_classes)
    theta = default_parameters(n_classes)
    hydro = solve(
        config.hydraulics, network, z, s, jnp.asarray(200.0), jnp.asarray(112.0),
        theta["n_manning"],
    )
    return config, network, theta, hydro, bed


def test_bed_composition_update():
    """Deposition, erosion and the bedrock floor in the composition update."""
    smoothing = SedimentConfig().smoothing
    delta_a = jnp.asarray(0.05)
    composition = BedComposition(
        active=jnp.asarray([[0.2, 0.2], [0.8, 0.8]]),
        substrate=jnp.asarray([[0.6, 0.6], [0.4, 0.4]]),
        substrate_thickness=jnp.asarray([5.0, 5.0]),
    )

    # Location 0: 1 m of class 0 deposits (20 layer thicknesses). Location 1: 1 cm of
    # class 1 is eroded.
    thickness_change = jnp.asarray([[1.0, 0.0], [0.0, -0.01]])
    updated = update_composition(composition, thickness_change, delta_a, smoothing)

    # The thick deposit replaces the layer: it becomes class 0 only.
    np.testing.assert_allclose(updated.active[:, 0], [1.0, 0.0], atol=1e-8)

    # Erosion of class 1: its fraction drops by 0.01 / 0.05 = 0.2, refilled from the
    # substrate (60 % class 0): p_1 = 0.8 + (-0.01 + 0.4 * 0.01) / 0.05 = 0.68.
    np.testing.assert_allclose(updated.active[:, 1], [0.32, 0.68], atol=1e-10)

    # Fractions sum to one; the substrate thickness follows the bed change.
    np.testing.assert_allclose(jnp.sum(updated.active, axis=0), 1.0, rtol=1e-14)
    np.testing.assert_allclose(jnp.sum(updated.substrate, axis=0), 1.0, rtol=1e-14)
    np.testing.assert_allclose(updated.substrate_thickness, [6.0, 4.99], atol=1e-10)

    # Erodible thickness: nothing on the floor, one active layer far above it or without
    # a floor.
    bed_elevation = jnp.asarray([10.0, 10.0])
    floor = jnp.asarray([10.0, 5.0])
    thickness = available_thickness(bed_elevation, floor, delta_a, smoothing)
    np.testing.assert_allclose(thickness, [0.0, 0.05], atol=1e-12)
    np.testing.assert_allclose(
        available_thickness(bed_elevation, None, delta_a, smoothing), 0.05, rtol=1e-14,
    )


def test_interface_exchange_with_a_share_of_the_load():
    """Aggradation with part of the load passed down to the substrate ("bedload_mix").

    Without a share of the load it is the standard update; with it the surface settles
    at the published steady state, coarser than the deposit where the load is finer;
    the sand follows the surface or the relation of Cui (2007); erosion is unchanged.
    """
    smoothing = SedimentConfig().smoothing
    delta_a = jnp.asarray(0.05)
    # Two sand classes (0.25 and 1 mm) and three gravel classes (4, 16 and 64 mm).
    diameters = jnp.asarray([0.25e-3, 1.0e-3, 4.0e-3, 16.0e-3, 64.0e-3])
    start = jnp.asarray([0.1, 0.2, 0.3, 0.25, 0.15])[:, None]
    composition = BedComposition(
        active=start, substrate=start, substrate_thickness=jnp.asarray([5.0]),
    )
    # Deposit of each step (2 mm) and the load: the load is finer than the deposit. Within
    # the gravel the deposit is (0.30, 0.35, 0.35) and the load (0.40, 0.35, 0.25).
    deposit = jnp.asarray([0.10, 0.15, 0.225, 0.2625, 0.2625])[:, None]
    load = jnp.asarray([0.25, 0.25, 0.20, 0.175, 0.125])[:, None]
    thickness_change = 0.002 * deposit

    def aggrade(share, rule, n_steps=1, state=composition, change=thickness_change):
        interface = InterfaceMix(load, diameters, jnp.asarray(share), rule)
        for _ in range(n_steps):
            state = update_composition(state, change, delta_a, smoothing, interface=interface)
        return state

    # No share of the load, sand as on the surface: the standard update (the substrate
    # receives the exact balance of the layer instead of its first-order form).
    standard = update_composition(composition, thickness_change, delta_a, smoothing)
    same = aggrade(0.0, "surface")
    np.testing.assert_allclose(same.active, standard.active, rtol=1e-12)
    np.testing.assert_allclose(same.substrate, standard.substrate, atol=1e-6)

    # Steady aggradation, share 0.7, sand as on the surface. The sand fraction of the
    # surface is that of the deposit; within the gravel, Ft_j = (pd_j - c pt_j) / (1 - c).
    steady = aggrade(0.7, "surface", n_steps=3000)
    np.testing.assert_allclose(jnp.sum(steady.active, axis=0), 1.0, rtol=1e-12)
    deposit_gravel = np.asarray(deposit[2:, 0] / jnp.sum(deposit[2:, 0]))
    load_gravel = np.asarray(load[2:, 0] / jnp.sum(load[2:, 0]))
    expected_gravel = (deposit_gravel - 0.7 * load_gravel) / 0.3
    surface = np.asarray(steady.active[:, 0])
    np.testing.assert_allclose(surface[:2].sum(), 0.25, rtol=1e-6)
    np.testing.assert_allclose(surface[2:] / surface[2:].sum(), expected_gravel, rtol=1e-6)

    # The surface gravel is coarser than the gravel of the deposit (mean psi), and the
    # material passed down in one more step is the deposit.
    psi = np.log2(np.asarray(diameters[2:]))
    assert (expected_gravel * psi).sum() > (deposit_gravel * psi).sum() + 0.5
    assert expected_gravel.min() > 0.0
    one_more = aggrade(0.7, "surface", state=steady)
    gained = (
        one_more.substrate[:, 0] * one_more.substrate_thickness[0]
        - steady.substrate[:, 0] * steady.substrate_thickness[0]
    )
    np.testing.assert_allclose(gained / jnp.sum(gained), deposit[:, 0], rtol=1e-5)

    # Sand by the relation of Cui (2007): the steady sand fraction of the surface solves
    # f_Is(F_s) = sand fraction of the deposit, f_Is = 0.4 - 0.1 s + (0.6 + 0.1 s) F_s
    # with the geometric standard deviation s of the gravel passed down (the deposit's).
    steady_cui = aggrade(0.7, "cui", n_steps=3000)
    mean = (deposit_gravel * psi).sum()
    sigma = 2.0 ** np.sqrt((deposit_gravel * (psi - mean) ** 2).sum())
    expected_sand = (0.25 - (0.4 - 0.1 * sigma)) / (0.6 + 0.1 * sigma)
    surface = np.asarray(steady_cui.active[:, 0])
    assert 0.0 < expected_sand < 0.25
    np.testing.assert_allclose(surface[:2].sum(), expected_sand, rtol=2e-3)
    np.testing.assert_allclose(surface[2:] / surface[2:].sum(), expected_gravel, rtol=1e-5)

    # A gravel class that is in the load but not in the deposit is emptied, not negative.
    without_fine = jnp.asarray([0.10, 0.15, 0.0, 0.40, 0.35])[:, None]
    emptied = aggrade(0.7, "surface", n_steps=3000, change=0.002 * without_fine)
    assert float(jnp.min(emptied.active)) >= 0.0
    np.testing.assert_allclose(emptied.active[2, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(jnp.sum(emptied.active, axis=0), 1.0, rtol=1e-12)

    # Erosion does not depend on the exchange in aggradation.
    erosion = -0.002 * deposit
    np.testing.assert_allclose(
        aggrade(0.7, "cui", change=erosion).active,
        update_composition(composition, erosion, delta_a, smoothing).active,
        rtol=1e-14,
    )

    # Derivative of the surface composition with respect to the share of the load.
    def coarse_fraction(share):
        interface = InterfaceMix(load, diameters, share, "cui")
        state = composition
        for _ in range(20):
            state = update_composition(state, thickness_change, delta_a, smoothing, interface=interface)
        return state.active[4, 0]

    gradient = jax.grad(coarse_fraction)(jnp.asarray(0.7))
    step = 1.0e-6
    difference = (coarse_fraction(0.7 + step) - coarse_fraction(0.7 - step)) / (2.0 * step)
    np.testing.assert_allclose(gradient, difference, rtol=1e-6)
    assert float(gradient) > 0.0


def test_run_with_interface_exchange():
    """A run with the interface exchange "bedload_mix": volume balance, and the surface
    of the aggrading reach coarser than with the exchange "surface"."""
    config, network, theta, forcing, bed0 = _channel_case()
    # Fine sand, coarse sand and gravel.
    diameters = (2.0e-4, 1.0e-3, 8.0e-3)
    results = {}
    for exchange in ("surface", "bedload_mix"):
        sediment = dataclasses.replace(
            config.sediment, diameters=diameters, cohesive_dmax=0.0,
            interface_exchange=exchange, interface_sand="cui",
        )
        parameters = {**theta, **interface_parameters()}
        results[exchange] = _run_jit(
            dataclasses.replace(config, sediment=sediment), network, parameters, forcing, bed0,
        )

    mixed = results["bedload_mix"]
    inflow = jnp.sum(mixed.load_in, axis=(1, 2))
    outflow = jnp.sum(mixed.load_out, axis=1)
    np.testing.assert_allclose(
        jnp.sum(mixed.deposition), jnp.sum((inflow - outflow) * forcing.dt), rtol=1e-8,
    )
    np.testing.assert_allclose(jnp.sum(mixed.bed.p_active, axis=0), 1.0, rtol=1e-12)
    np.testing.assert_allclose(jnp.sum(mixed.bed.p_substrate, axis=0), 1.0, rtol=1e-12)
    assert float(jnp.min(mixed.bed.p_active)) >= 0.0

    # Where the bed has aggraded, the surface holds less sand than with the exchange
    # "surface" (the relation of Cui passes more sand down than the surface holds).
    rise = jnp.min(mixed.bed.z, axis=1) - jnp.min(bed0.z, axis=1)
    aggraded = np.asarray(rise > 0.05)
    assert aggraded.sum() >= 3
    sand_mixed = np.asarray(jnp.sum(mixed.bed.p_active[:2], axis=0))[aggraded]
    sand_surface = np.asarray(jnp.sum(results["surface"].bed.p_active[:2], axis=0))[aggraded]
    assert np.all(sand_mixed <= sand_surface + 1e-9)
    assert np.mean(sand_surface - sand_mixed) > 0.01


def test_pure_settling_is_exponential_and_conservative():
    """One cohesive class without erosion: exponential decay of the load."""
    config, network, theta, hydro, bed = _routing_setup((2.0e-5,))

    # No erosion and full deposition everywhere: S = 0 and r = B w / Q.
    theta["tau_ce"] = jnp.asarray(1.0e9)
    theta["tau_cd"] = jnp.asarray(1.0e9)
    closures = compute_closures(sediment_flow_state(hydro), bed.p_active, theta, config.sediment)

    load_in = jnp.asarray([[1.0e-3]])
    supply = jnp.full((1, 21), 1.0e6)
    sediment = route(config, network, closures, load_in, supply)

    # Decay over the virtual inflow reach (half of the first reach), then reach by reach
    # with the rate of the downstream section of each reach.
    dx = network.reach_length
    rate = closures.cohesive_rate[0]
    load_first = load_in[0, 0] * jnp.exp(-rate[0] * 0.5 * dx[0])
    exponent = jnp.cumsum(rate[1:] * dx)
    np.testing.assert_allclose(sediment.load[0, 0], load_first, rtol=1e-8)
    np.testing.assert_allclose(sediment.load[0, 1:], load_first * jnp.exp(-exponent), rtol=1e-8)

    # Deposition equals inflow minus outflow.
    np.testing.assert_allclose(
        jnp.sum(sediment.deposition_reach), load_in[0, 0] - sediment.load[0, -1], rtol=1e-10,
    )
    assert float(sediment.load[0, -1]) < float(load_in[0, 0])


def test_headwater_reach_and_frozen_headwater():
    """The virtual inflow reach lets the first section deposit; without it, it cannot."""
    config, network, theta, hydro, bed = _routing_setup((2.0e-5,))
    theta["tau_ce"] = jnp.asarray(1.0e9)
    theta["tau_cd"] = jnp.asarray(1.0e9)
    closures = compute_closures(sediment_flow_state(hydro), bed.p_active, theta, config.sediment)
    load_in = jnp.asarray([[1.0e-3]])
    supply = jnp.full((1, 21), 1.0e6)

    # Default: 20 real reaches plus one virtual reach, which receives a deposit.
    sediment = route(config, network, closures, load_in, supply)
    assert sediment.deposition_reach.shape[1] == 21
    assert float(sediment.deposition_reach[0, -1]) > 0.0

    # head_inflow_fraction = 0: the inflow passes the first section unchanged.
    frozen = dataclasses.replace(config, morph=MorphNumerics(head_inflow_fraction=0.0))
    sediment = route(frozen, network, closures, load_in, supply)
    assert sediment.deposition_reach.shape[1] == 20
    np.testing.assert_allclose(sediment.load[0, 0], load_in[0, 0], rtol=1e-14)


def test_noncohesive_load_relaxes_to_capacity():
    """A sand load far above capacity decreases monotonically and conserves volume."""
    config, network, theta, hydro, bed = _routing_setup((5.0e-4,))
    closures = compute_closures(sediment_flow_state(hydro), bed.p_active, theta, config.sediment)

    load_in = 10.0 * closures.capacity[:, :1] + 1.0e-3
    sediment = route(config, network, closures, load_in, jnp.full((1, 21), 1.0e6))

    assert bool(jnp.all(sediment.load[0, 1:] <= sediment.load[0, :-1] + 1e-12))
    np.testing.assert_allclose(
        jnp.sum(sediment.deposition_reach), load_in[0, 0] - sediment.load[0, -1], rtol=1e-10,
    )


@pytest.mark.parametrize("coupling", ["diagonal", "full"])
def test_negative_capacity_sensitivity_is_not_amplified(coupling: str):
    """A capacity that falls as the bed rises must not amplify the implicit step."""
    config, network, theta, hydro, bed = _routing_setup((5.0e-4,))
    config = dataclasses.replace(config, morph=MorphNumerics(implicit_coupling=coupling))
    closures = compute_closures(sediment_flow_state(hydro), bed.p_active, theta, config.sediment)
    load_in = closures.capacity[:, :1]
    supply = jnp.full((1, 21), 1.0e6)
    exner = jnp.full((1, 21), 5.0)

    def routed(sensitivity):
        """Route with the given capacity sensitivity d Q* / d z_b, shape [1, 21]."""
        return route(
            config, network, closures, load_in, supply, None, sensitivity, exner,
            bed.p_active, theta["delta_a"],
        )

    # For one class the implicit factor is 1 / (1 + c f s) with f in (0, 1]: with c = 5
    # it would be singular for some s in [-1, -0.2] and negative below it.
    zero = routed(jnp.zeros((1, 21)))
    negative = routed(jnp.linspace(-1.0, -0.01, 21)[None, :])
    positive = routed(jnp.full((1, 21), 0.05))

    np.testing.assert_allclose(negative.load, zero.load, rtol=1e-12)
    np.testing.assert_allclose(negative.deposition_reach, zero.deposition_reach, rtol=1e-12, atol=1e-18)
    assert bool(jnp.all(jnp.isfinite(negative.load)))
    assert bool(jnp.all(negative.load >= 0.0))

    # The implicit term is active for a positive sensitivity (the test is not vacuous).
    assert float(jnp.max(jnp.abs(positive.load - zero.load))) > 1.0e-3 * float(jnp.max(zero.load))


def test_short_run_conserves_volume():
    """Deposited volume = (inflow - outflow) dt; upstream erodes, the pool fills."""
    config, network, theta, forcing, bed0 = _channel_case()
    result = _run_jit(config, network, theta, forcing, bed0)

    # Volume balance over the run.
    inflow = jnp.sum(result.load_in, axis=(1, 2))
    outflow = jnp.sum(result.load_out, axis=1)
    np.testing.assert_allclose(
        jnp.sum(result.deposition), jnp.sum((inflow - outflow) * forcing.dt), rtol=1e-8,
    )

    # The steep upstream reach erodes (the sand capacity exceeds the supply); the pool
    # aggrades.
    bed_change = jnp.min(result.bed.z, axis=1) - jnp.min(bed0.z, axis=1)
    assert float(jnp.min(bed_change[:2])) < 0.0
    assert float(jnp.max(bed_change[5:])) > 0.0
    assert bool(jnp.all(bed_change[-5:] >= -1e-9))
    assert 0.0 < float(result.trap_efficiency[-1]) < 1.0

    # Fractions stay normalized.
    np.testing.assert_allclose(jnp.sum(result.bed.p_active, axis=0), 1.0, rtol=1e-12)
    np.testing.assert_allclose(jnp.sum(result.bed.p_substrate, axis=0), 1.0, rtol=1e-12)

    # Time at the end of each step.
    np.testing.assert_allclose(result.time_end, DAY * np.arange(1, 21), rtol=1e-14)


def test_depth_weighted_deposit_conserves_volume():
    """With ``deposition_depth_exponent`` the balance closes; a flat bottom is unchanged."""
    morph = MorphNumerics(deposition_depth_exponent=1.0)
    config, network, theta, forcing, bed0 = _channel_case(morph=morph)
    result = _run_jit(config, network, theta, forcing, bed0)

    inflow = jnp.sum(result.load_in, axis=(1, 2))
    outflow = jnp.sum(result.load_out, axis=1)
    np.testing.assert_allclose(
        jnp.sum(result.deposition), jnp.sum((inflow - outflow) * forcing.dt), rtol=1e-8,
    )

    # The wetted points of the trapezoid (its flat bottom) all have the largest depth, so
    # the result equals that of the uniform shift.
    plain = _run_jit(*_channel_case())
    np.testing.assert_allclose(result.bed.z, plain.bed.z, rtol=1e-12, atol=1e-12)


def test_channel_deficit_and_erosion_exponent_conserve_volume():
    """Balance with ``deposition_channel_deficit`` and ``erosion_depth_exponent``; open channel."""
    morph = MorphNumerics(deposition_channel_deficit=0.8, erosion_depth_exponent=2.0)
    config, network, theta, forcing, bed0 = _channel_case(morph=morph)
    # A compound section: lower one of the two bottom points of the trapezoid by 1 m.
    bed0 = bed0._replace(z=bed0.z.at[:, 1].add(-1.0))
    result = _run_jit(config, network, theta, forcing, bed0)

    inflow = jnp.sum(result.load_in, axis=(1, 2))
    outflow = jnp.sum(result.load_out, axis=1)
    np.testing.assert_allclose(
        jnp.sum(result.deposition), jnp.sum((inflow - outflow) * forcing.dt), rtol=1e-8,
    )

    # In the pool (deposition) the deep point rises less than the bottom point beside it.
    change = result.bed.z - bed0.z
    assert float(change[-3, 2]) > 0.0
    assert float(change[-3, 1]) < 0.6 * float(change[-3, 2])


def test_fresh_deposits_erode_more_easily():
    """``tau_ce_fresh``: same run for the value of ``tau_ce``; balance and less deposit below it."""
    config, network, theta, forcing, bed0 = _channel_case()
    # Shear stress of the pool lies between the two thresholds, so that only new deposits erode.
    theta = {**theta, "tau_ce": jnp.asarray(5.0)}
    plain = _run_jit(config, network, theta, forcing, bed0)
    same = _run_jit(config, network, {**theta, "tau_ce_fresh": theta["tau_ce"]}, forcing, bed0)
    np.testing.assert_allclose(same.bed.z, plain.bed.z, rtol=1e-12, atol=1e-12)

    weak = {**theta, "tau_ce_fresh": jnp.asarray(0.02)}
    result = _run_jit(config, network, weak, forcing, bed0)
    inflow = jnp.sum(result.load_in, axis=(1, 2))
    outflow = jnp.sum(result.load_out, axis=1)
    np.testing.assert_allclose(
        jnp.sum(result.deposition), jnp.sum((inflow - outflow) * forcing.dt), rtol=1e-8,
    )
    assert float(jnp.sum(result.deposition)) < float(jnp.sum(plain.deposition))

    def deposited(value):
        """Deposited volume as a function of the threshold of new deposits."""
        return jnp.sum(run(config, network, {**theta, "tau_ce_fresh": value}, forcing, bed0).deposition)

    slope = jax.jit(jax.grad(deposited))(jnp.asarray(0.02))
    assert bool(jnp.isfinite(slope)) and float(slope) > 0.0


def test_lateral_withdrawal_discharge_and_volume_balance():
    """Water and load leave at the intake; the volume balance closes with them."""
    config, network, theta, forcing, bed0 = _channel_case()
    intake = 12
    with_intake = with_withdrawals(network, [intake])

    # Discharge: unchanged down to the intake section, lower below it.
    Q = section_discharge(with_intake, jnp.asarray(200.0), jnp.asarray([150.0]))
    np.testing.assert_allclose(Q[: intake + 1], 200.0)
    np.testing.assert_allclose(Q[intake + 1 :], 50.0)

    # A zero withdrawal changes nothing.
    plain = _run_jit(config, network, theta, forcing, bed0)
    zero = _run_jit(config, with_intake, theta, forcing._replace(Q_withdrawal=jnp.zeros(20)), bed0)
    np.testing.assert_allclose(zero.deposition, plain.deposition, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(zero.load_withdrawn, 0.0, atol=1e-30)

    # Three quarters of the water withdrawn.
    drawn = _run_jit(
        config, with_intake, theta, forcing._replace(Q_withdrawal=jnp.full(20, 150.0)), bed0,
    )
    assert drawn.load_withdrawn.shape == (20, 3, 1)
    inflow = jnp.sum(drawn.load_in, axis=(1, 2))
    outflow = jnp.sum(drawn.load_out, axis=1) + jnp.sum(drawn.load_withdrawn, axis=(1, 2))
    np.testing.assert_allclose(
        jnp.sum(drawn.deposition), jnp.sum((inflow - outflow) * forcing.dt), rtol=1e-8,
    )
    assert float(jnp.sum(drawn.load_withdrawn)) > 0.0

    # Less water and less load reach the dam.
    assert float(jnp.sum(drawn.load_out)) < float(jnp.sum(plain.load_out))

    # Water-only exchange (storage): no load leaves sideways, and the balance closes with
    # the load at the outlet alone.
    storage = with_withdrawals(network, [intake], takes_load=False)
    stored = _run_jit(
        config, storage, theta, forcing._replace(Q_withdrawal=jnp.full(20, 150.0)), bed0,
    )
    np.testing.assert_allclose(stored.load_withdrawn, 0.0, atol=1e-30)
    np.testing.assert_allclose(
        jnp.sum(stored.deposition),
        jnp.sum((jnp.sum(stored.load_in, axis=(1, 2)) - jnp.sum(stored.load_out, axis=1)) * forcing.dt),
        rtol=1e-8,
    )


def test_lateral_withdrawal_on_network_conserves_volume():
    """Withdrawal on one branch of the Y-shaped network: balance and junction shares."""
    network, theta, forcing, bed0 = _y_network_case(n_steps=10)
    with_intake = with_withdrawals(network, [4])
    forcing = forcing._replace(Q_withdrawal=jnp.full((10, 1), 60.0))
    result = _run_jit(OneDConfig(), with_intake, theta, forcing, bed0)

    inflow = jnp.sum(result.load_in, axis=(1, 2))
    outflow = jnp.sum(result.load_out, axis=1) + jnp.sum(result.load_withdrawn, axis=(1, 2))
    np.testing.assert_allclose(
        jnp.sum(result.deposition), jnp.sum((inflow - outflow) * forcing.dt), rtol=1e-8,
    )
    assert bool(jnp.all(jnp.isfinite(result.deposition)))
    with pytest.raises(ValueError):
        with_withdrawals(network, [network.n_sections - 1])


def test_substeps_hydro_interval_records_and_snapshots():
    """Options of the time loop keep the volume balance and the output layout."""
    morph = MorphNumerics(
        n_substeps=2, hydro_every=2, record_every=3, snapshot_steps=(4, 9),
    )
    config, network, theta, forcing, bed0 = _channel_case(n_steps=10, morph=morph)
    result = run(config, network, theta, forcing, bed0)

    # Histories keep every third step (steps 0, 3, 6, 9); deposition keeps all steps.
    assert result.deposition.shape == (10, 3, 21)
    assert result.thalweg.shape == (4, 21)
    assert result.load_out.shape == (4, 3)

    # The same run with every step recorded.
    config_all = dataclasses.replace(config, morph=dataclasses.replace(morph, record_every=1))
    full = run(config_all, network, theta, forcing, bed0)

    # Volume balance with a hydraulic solve every second step. (Without sub-steps: the
    # recorded outflow is that of the last sub-step, not the mean over the step.)
    config_balance = dataclasses.replace(config, morph=MorphNumerics(hydro_every=2))
    balance = run(config_balance, network, theta, forcing, bed0)
    inflow = jnp.sum(balance.load_in, axis=(1, 2))
    outflow = jnp.sum(balance.load_out, axis=1)
    np.testing.assert_allclose(
        jnp.sum(balance.deposition), jnp.sum((inflow - outflow) * forcing.dt), rtol=1e-8,
    )

    # Snapshots: the bed after steps 4 and 9; the last one is the final bed.
    assert result.z_snapshots.shape == (2, 21, 4)
    np.testing.assert_allclose(result.z_snapshots[1], result.bed.z, rtol=1e-15)
    np.testing.assert_allclose(jnp.min(result.z_snapshots[0], axis=1), full.thalweg[4], rtol=1e-15)

    # With hydro_every = 2 the water surface is the same in steps 0 and 1.
    np.testing.assert_allclose(full.eta[0], full.eta[1], rtol=1e-15)


def _deposit_objective(config, network, theta, forcing, bed0):
    """Sum of squares of the deposit volume of each section (m^6)."""
    result = run(config, network, theta, forcing, bed0)
    return jnp.sum(jnp.sum(result.deposition, axis=(0, 1)) ** 2)


def test_checkpointing_and_gradients():
    """Checkpoint settings agree; derivatives match central differences."""
    names = ("a_eh", "kappa_w", "alpha_d", "delta_a", "n_manning", "tau_cd")
    values = {}
    gradients = {}

    # The same 7-step run with the whole history stored, with segments of three steps
    # (two full segments and a remainder), and with the automatic segment length.
    for checkpoint in ("none", 3, "auto"):
        morph = MorphNumerics(checkpoint=checkpoint)
        config, network, theta, forcing, bed0 = _channel_case(n_steps=7, morph=morph)

        def objective(parameters, config=config, network=network, forcing=forcing, bed0=bed0):
            """Deposit objective as a function of the parameters."""
            return _deposit_objective(config, network, parameters, forcing, bed0)

        value, gradient = jax.jit(jax.value_and_grad(objective))(theta)
        values[checkpoint] = value
        gradients[checkpoint] = gradient

    for checkpoint in (3, "auto"):
        np.testing.assert_allclose(values[checkpoint], values["none"], rtol=1e-10)
        for name in names:
            np.testing.assert_allclose(
                gradients[checkpoint][name], gradients["none"][name], rtol=1e-10, err_msg=name,
            )

    # Central differences in each parameter, for a vector parameter in the entry with
    # the largest derivative. The objective J is of order 1e7 and carries a round-off of
    # about 1e-12 J (the eroded volumes upstream are large and enter squared), so the
    # difference quotient has an absolute error of about 1e-12 J / step. This is allowed
    # for; it matters only for derivatives of order 100 or less (tau_cd), which are 1e-5
    # of the others.
    objective_jit = jax.jit(objective)
    for name in names:
        parameter = np.atleast_1d(np.asarray(theta[name])).copy()
        derivative_all = np.atleast_1d(np.asarray(gradients["auto"][name]))
        entry = int(np.argmax(np.abs(derivative_all)))
        step = 1.0e-4 * abs(parameter[entry])

        plus = dict(theta)
        parameter_plus = parameter.copy()
        parameter_plus[entry] += step
        plus[name] = jnp.asarray(parameter_plus.reshape(np.shape(theta[name])))

        minus = dict(theta)
        parameter_minus = parameter.copy()
        parameter_minus[entry] -= step
        minus[name] = jnp.asarray(parameter_minus.reshape(np.shape(theta[name])))

        difference = (objective_jit(plus) - objective_jit(minus)) / (2.0 * step)
        roundoff = 5.0e-12 * abs(float(values["auto"])) / step
        np.testing.assert_allclose(
            derivative_all[entry], difference, rtol=1e-3, atol=roundoff, err_msg=name,
        )


def test_fixed_bed_does_not_erode():
    """A non-erodible reach loses at most its active layer; the run stays finite."""
    n_steps = 30
    config = OneDConfig()
    x, z, s = trapezoid_channel(length=20000.0, n_sections=41, bottom_width=80.0)
    network = network_from_positions(x)
    fixed = x < 2500.0
    bed0 = uniform_bed(z, s, n_classes=3, fractions=(0.2, 0.3, 0.5), fixed=fixed)
    theta = default_parameters(3)
    theta["tau_cd"] = jnp.asarray(1.0)
    load_in = jnp.broadcast_to(jnp.asarray([0.02, 0.01, 0.003])[:, None], (3, n_steps))
    forcing = constant_forcing(200.0, 118.0, DAY, n_steps)._replace(load_in=load_in)

    result = _run_jit(config, network, theta, forcing, bed0)

    # The only erodible material on the fixed reach is the initial active layer plus the
    # 1 cm of initial substrate.
    delta_a = float(theta["delta_a"])
    bed_change = jnp.min(result.bed.z, axis=1) - jnp.min(bed0.z, axis=1)
    assert bool(jnp.all(bed_change[fixed] > -(2.0 * delta_a + 0.02)))
    assert bool(jnp.all(jnp.isfinite(result.eta)))
    assert 0.0 < float(result.trap_efficiency[-1]) < 1.0


def test_y_network_conserves_volume():
    """Volume balance of a Y-shaped network with the sediment rating curve."""
    n_steps = 20
    config = OneDConfig()
    y = y_network(
        branch_lengths=(5000.0, 4000.0),
        n_branch=(10, 8),
        branch_widths=(40.0, 30.0),
        branch_slopes=(1.2e-3, 1.5e-3),
    )
    bed0 = uniform_bed(y.z, y.s, n_classes=3)
    theta = default_parameters(3)
    theta["tau_cd"] = jnp.asarray(1.0)
    theta["a_r"] = jnp.asarray(2.0e-6)
    theta["p_in"] = jnp.asarray([0.6, 0.3, 0.1])

    # 120 and 60 m^3/s on the branches; the inflow load follows the rating curve.
    Q_in = jnp.broadcast_to(jnp.asarray([120.0, 60.0]), (n_steps, 2))
    forcing = constant_forcing(0.0, 112.0, DAY, n_steps)._replace(Q_in=Q_in)

    result = _run_jit(config, y.network, theta, forcing, bed0)

    # Rating curve: a_r Q^2 split by p_in at each headwater.
    np.testing.assert_allclose(
        result.load_in[0, :, 0], np.array([0.6, 0.3, 0.1]) * 2.0e-6 * 120.0**2, rtol=1e-13,
    )

    # Inflow - outflow = deposition.
    inflow = jnp.sum(result.load_in) * DAY
    outflow = jnp.sum(result.load_out) * DAY
    deposited = jnp.sum(result.deposition)
    assert abs(float(inflow - outflow - deposited)) < 1e-8 * float(inflow)
    assert bool(jnp.all(jnp.isfinite(result.thalweg)))

    # Both branches are fed below capacity and erode; the pool fills.
    per_section = jnp.sum(result.deposition, axis=(0, 1))
    assert float(jnp.sum(per_section[1:10])) < 0.0
    assert float(jnp.sum(per_section[11:18])) < 0.0
    assert float(jnp.sum(per_section[18:])) > 0.0


def _y_network_case(n_steps: int):
    """Y-shaped network with different branches and the sediment rating curve.

    Args:
        n_steps: number of daily steps T.

    Returns:
        Tuple ``(network, theta, forcing, bed0)``.
    """
    y = y_network(
        branch_lengths=(5000.0, 4000.0),
        n_branch=(10, 8),
        branch_widths=(40.0, 30.0),
        branch_slopes=(1.2e-3, 1.5e-3),
    )
    bed0 = uniform_bed(y.z, y.s, n_classes=3)
    theta = default_parameters(3)
    theta["tau_cd"] = jnp.asarray(1.0)
    theta["a_r"] = jnp.asarray(2.0e-6)
    theta["p_in"] = jnp.asarray([0.5, 0.3, 0.2])
    Q_in = jnp.broadcast_to(jnp.asarray([120.0, 60.0]), (n_steps, 2))
    forcing = constant_forcing(0.0, 112.0, DAY, n_steps)._replace(Q_in=Q_in)
    return y.network, theta, forcing, bed0


def test_full_implicit_coupling():
    """Full coupling: equal to diagonal for one class; conservative; no negative loads."""
    full = OneDConfig(morph=MorphNumerics(implicit_coupling="full"))

    # One size class: the composition feedback vanishes (delta_kk - p_k = 0 with p = 1),
    # so the full and the diagonal coupling are the same scheme.
    sand = SedimentConfig(diameters=(5.0e-4,))
    x, z, s = trapezoid_channel(length=10000.0, n_sections=21)
    network = network_from_positions(x)
    bed0 = uniform_bed(z, s, n_classes=1)
    theta = default_parameters(1)
    load_in = jnp.full((1, 10), 0.005)
    forcing = constant_forcing(200.0, 112.0, DAY, 10)._replace(load_in=load_in)
    result_full = run(dataclasses.replace(full, sediment=sand), network, theta, forcing, bed0)
    result_diagonal = run(OneDConfig(sediment=sand), network, theta, forcing, bed0)
    np.testing.assert_allclose(result_full.bed.z, result_diagonal.bed.z, atol=1e-10)

    # Y-shaped network, 200 daily steps. With the composition term of the coupling
    # matrix indexed by the column class (an index error), the outlet load of this
    # case was negative in 189 of the 200 steps.
    network, theta, forcing, bed0 = _y_network_case(200)
    result = _run_jit(full, network, theta, forcing, bed0)
    load_scale = float(jnp.max(jnp.abs(result.load_out)))
    assert float(jnp.min(result.load_out)) >= -1e-12 * load_scale

    # Volume balance.
    inflow = jnp.sum(result.load_in) * DAY
    outflow = jnp.sum(result.load_out) * DAY
    deposited = jnp.sum(result.deposition)
    assert abs(float(inflow - outflow - deposited)) < 1e-8 * float(inflow)


def test_consolidation_and_storage_curve():
    """Consolidated volumes are smaller after years; the storage curve is exact."""
    config, network, theta, forcing, bed0 = _channel_case(n_steps=10, dt=30.0 * DAY)
    result = run(config, network, theta, forcing, bed0)

    # Survey five years after the end of the run.
    seconds_per_year = config.sediment.constants.seconds_per_year
    time_survey = float(result.time_end[-1]) + 5.0 * seconds_per_year
    consolidated = consolidated_volume(
        result.deposition, result.time_end, time_survey, theta, config.sediment,
    )
    uncompacted = bulk_volume_uncompacted(result.deposition, theta, config.sediment)
    assert bool(jnp.all(consolidated <= uncompacted + 1e-12))
    assert float(jnp.sum(consolidated)) < float(jnp.sum(uncompacted))

    # A survey before the first step ends counts nothing.
    early = consolidated_volume(result.deposition, result.time_end, 0.0, theta, config.sediment)
    np.testing.assert_array_equal(early, 0.0)

    # Storage curve of the initial bed at 110 m: the flow area of a trapezoid with depth
    # y_i = 110 - z_bed,i is (b + m y_i) y_i (b = 60 m, m = 2), zero where the bed is higher.
    cv_lengths = control_volume_lengths(network)
    levels = jnp.asarray([105.0, 110.0])
    volume = capacity_curve(bed0.z, bed0.s, cv_lengths, levels, config.hydraulics.smoothing)
    depth = jnp.maximum(110.0 - jnp.min(bed0.z, axis=1), 0.0)
    expected = jnp.sum((60.0 + 2.0 * depth) * depth * cv_lengths)
    np.testing.assert_allclose(volume[1], expected, rtol=2e-3)
    assert float(volume[0]) < float(volume[1])


@pytest.mark.parametrize("scheme", ["upstream_half", "upstream_downstream"])
def test_study_schemes_conserve_volume(scheme: str):
    """The deposition schemes of the stability study still conserve sediment volume."""
    morph = MorphNumerics(deposition_scheme=scheme)
    config, network, theta, forcing, bed0 = _channel_case(n_steps=5, dt=3600.0, morph=morph)
    result = run(config, network, theta, forcing, bed0)

    inflow = jnp.sum(result.load_in, axis=(1, 2))
    outflow = jnp.sum(result.load_out, axis=1)
    np.testing.assert_allclose(
        jnp.sum(result.deposition), jnp.sum((inflow - outflow) * forcing.dt), rtol=1e-8,
    )
