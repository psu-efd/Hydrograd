"""Tests of the sediment closures against analytic results (``hydrograd.sediment``).

The tests check:

1. Settling velocity (Ferguson and Church 2004): the Stokes limit R_s g d^2 / (18 nu) for
   fine grains and the constant-drag limit sqrt(4 R_s g d / 3) for coarse grains.
2. Suspended fraction: 1 for strong turbulence, 0 for weak turbulence.
3. Engelund-Hansen capacity of a single class equals the classical form
   q_t = 0.05 V^2 theta^(3/2) sqrt(d / (R_s g)), with V the mean velocity.
4. One formulation for 1-D and 2-D: evaluating the closures for a stream tube (Q, B) and
   per unit width (Q / B, 1) gives the same lengths and rates, and capacities and sources
   that differ by the factor B.
5. A location without shear stress (a dry cell in 2-D) has zero capacity and finite
   derivatives.
6. Exchange: no net exchange when the load equals the capacity; the erosion of a class
   never exceeds the supply of the bed; cohesive classes deposit with the Krone
   probability and do not erode below tau_ce.
7. The closures can be jit-compiled and give the same values.
8. Automatic derivatives with respect to all closure parameters match central differences.
9. Hiding factor: the two power laws reduce to the power law for equal exponents; the
   network gives 1 at the start and for a grain of the mean size, and its capacity has
   finite derivatives with respect to the network parameters that match central
   differences.
10. Wilcock-Crowe capacity: the two branches of the transport function at known values
    of phi = tau_b / tau_r and their junction; the reference Shields stress with and
    without sand; equal mobility for the hiding exponent -1; the published hiding
    function; zero capacity and finite derivatives without shear stress; derivatives
    against central differences.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from hydrograd.sediment import (
    FlowState,
    SedimentConfig,
    compute_closures,
    default_sediment_parameters,
    source_and_rate,
    wilcock_crowe_parameters,
)
from hydrograd.sediment.closures import (
    hiding_factor,
    init_hiding_network,
    init_transport_network,
    settling_velocity,
    suspended_fraction,
    transport_capacity,
    transport_function,
    wilcock_crowe_capacity,
    wilcock_crowe_stress_ratio,
)

# Physical constants of the default configuration (SI).
G = 9.81                # m/s^2
RHO = 1000.0            # kg/m^3
RS = 1.65               # submerged specific gravity (-)
NU = 1.0e-6             # m^2/s


def _flow_state(discharge, width, depth, friction_coefficient) -> FlowState:
    """Flow state of a stream tube from discharge, width, depth and friction coefficient.

    Args:
        discharge: Q (m^3/s), shape [N].
        width: B (m), shape [N].
        depth: h (m), shape [N].
        friction_coefficient: c_f (-), shape [N].

    Returns:
        Flow state with tau_b = rho c_f u^2 (Pa), u = Q / (B h), and u_* = sqrt(tau_b / rho).
    """
    discharge = jnp.asarray(discharge)
    width = jnp.asarray(width)
    depth = jnp.asarray(depth)
    friction_coefficient = jnp.asarray(friction_coefficient)

    velocity = discharge / (width * depth)
    shear_stress = RHO * friction_coefficient * velocity**2

    return FlowState(
        discharge=discharge,
        width=width,
        depth=depth,
        shear_stress=shear_stress,
        shear_velocity=jnp.sqrt(shear_stress / RHO),
        friction_coefficient=friction_coefficient,
    )


def _example_flow() -> FlowState:
    """Five locations from a fast river (2 m/s) to a nearly still pool (0.02 m/s)."""
    return _flow_state(
        discharge=[200.0, 200.0, 200.0, 200.0, 200.0],
        width=[50.0, 60.0, 80.0, 150.0, 400.0],
        depth=[2.0, 2.5, 4.0, 10.0, 25.0],
        friction_coefficient=[4.0e-3, 3.5e-3, 3.0e-3, 2.5e-3, 2.0e-3],
    )


def _example_fractions(n_locations: int) -> jnp.ndarray:
    """Active-layer fractions of three classes that vary along the channel, shape [3, N]."""
    position = jnp.linspace(0.0, 1.0, n_locations)
    fine = 0.1 + 0.6 * position
    coarse = 0.5 - 0.4 * position
    medium = 1.0 - fine - coarse
    return jnp.stack([fine, medium, coarse])


def test_settling_velocity_limits():
    """Stokes limit for fine grains and constant-drag limit for coarse grains."""
    config = SedimentConfig()
    diameters = jnp.asarray([2.0e-6, 0.1])
    w = settling_velocity(diameters, jnp.ones(2), config.constants)

    # Fine grain: Stokes, w = R_s g d^2 / (18 nu).
    stokes = RS * G * diameters[0] ** 2 / (18.0 * NU)
    np.testing.assert_allclose(w[0], stokes, rtol=1e-3)

    # Coarse grain: w = sqrt(4 R_s g d / 3), i.e. a drag coefficient of 1.
    constant_drag = jnp.sqrt(4.0 * RS * G * diameters[1] / 3.0)
    np.testing.assert_allclose(w[1], constant_drag, rtol=1e-3)

    # The multiplier scales the velocity.
    w_doubled = settling_velocity(diameters, jnp.asarray([2.0, 1.0]), config.constants)
    np.testing.assert_allclose(w_doubled, w * jnp.asarray([2.0, 1.0]), rtol=1e-14)


def test_suspended_fraction_limits():
    """f_s -> 1 where u_* >> w and f_s -> 0 where u_* << w."""
    config = SedimentConfig()
    w = jnp.asarray([1.0e-4, 5.0e-2])
    u_star = jnp.asarray([0.2, 1.0e-4])
    f_s = suspended_fraction(w, u_star, config)

    # Fine class in strong turbulence: fully suspended.
    np.testing.assert_allclose(f_s[0, 0], 1.0, atol=1e-12)

    # Coarse class in weak turbulence: bed load only.
    np.testing.assert_allclose(f_s[1, 1], 0.0, atol=1e-12)

    # The fraction stays within [0, 1].
    assert bool(jnp.all((f_s >= 0.0) & (f_s <= 1.0)))


def test_engelund_hansen_classical_form():
    """A single class gives q_t = 0.05 V^2 theta^(3/2) sqrt(d / (R_s g))."""
    diameter = 3.0e-4
    config = SedimentConfig(diameters=(diameter,), cohesive_dmax=0.0)
    theta = default_sediment_parameters(1)

    flow = _flow_state([30.0, 80.0], [20.0, 40.0], [1.5, 2.0], [3.0e-3, 4.0e-3])
    fractions = jnp.ones((1, 2))
    closures = compute_closures(flow, fractions, theta, config)

    # Classical Engelund-Hansen formula per unit width, times the width.
    velocity = flow.discharge / (flow.width * flow.depth)
    shields = flow.shear_stress / (RHO * RS * G * diameter)
    q_classical = 0.05 * velocity**2 * shields**1.5 * jnp.sqrt(diameter / (RS * G))
    np.testing.assert_allclose(closures.capacity[0], flow.width * q_classical, rtol=1e-9)


def test_stream_tube_and_unit_width_agree():
    """Closures for (Q, B) and for (Q / B, 1) differ only by the factor B (1-D vs 2-D)."""
    config = SedimentConfig()
    theta = default_sediment_parameters(config.n_classes)
    theta["m_h"] = jnp.asarray(-0.3)

    # 1-D: the stream tube of the cross-section.
    flow = _example_flow()
    fractions = _example_fractions(flow.discharge.shape[0])
    tube = compute_closures(flow, fractions, theta, config)

    # 2-D: unit width, with the unit discharge q = Q / B.
    flow_unit = flow._replace(discharge=flow.discharge / flow.width, width=jnp.ones(5))
    unit = compute_closures(flow_unit, fractions, theta, config)

    # Capacities and cohesive sources scale with the width.
    width = flow.width[None, :]
    np.testing.assert_allclose(tube.capacity, width * unit.capacity, rtol=1e-12)
    np.testing.assert_allclose(tube.cohesive_source, width * unit.cohesive_source, rtol=1e-12)

    # Lengths and rates are the same. The tolerance allows for the guard constants, which
    # act on B-dependent denominators.
    np.testing.assert_allclose(
        tube.length_suspended_deposition, unit.length_suspended_deposition, rtol=1e-6,
    )
    np.testing.assert_allclose(tube.cohesive_rate, unit.cohesive_rate, rtol=1e-6)
    np.testing.assert_allclose(tube.suspended_fraction, unit.suspended_fraction, rtol=1e-14)


def test_no_shear_gives_zero_capacity_and_finite_gradients():
    """Without shear stress the capacity is zero and its derivatives are finite."""
    config = SedimentConfig()
    theta = default_sediment_parameters(config.n_classes)
    zeros = jnp.zeros(2)
    flow = FlowState(
        discharge=zeros,
        width=jnp.ones(2),
        depth=jnp.asarray([0.0, 1.0e-3]),
        shear_stress=zeros,
        shear_velocity=zeros,
        friction_coefficient=zeros,
    )
    fractions = jnp.full((3, 2), 1.0 / 3.0)

    def total_capacity(parameters):
        """Sum of the capacities and cohesive terms (scalar)."""
        closures = compute_closures(flow, fractions, parameters, config)
        return jnp.sum(closures.capacity) + jnp.sum(closures.cohesive_source)

    closures = compute_closures(flow, fractions, theta, config)
    np.testing.assert_array_equal(closures.capacity, 0.0)

    gradient = jax.grad(total_capacity)(theta)
    for name, value in gradient.items():
        assert bool(jnp.all(jnp.isfinite(value))), name


def test_exchange_equilibrium_and_supply_limit():
    """No net exchange at capacity; erosion never exceeds the supply of the bed."""
    config = SedimentConfig()
    theta = default_sediment_parameters(config.n_classes)
    flow = _example_flow()
    fractions = _example_fractions(5)
    closures = compute_closures(flow, fractions, theta, config)
    noncohesive = ~config.cohesive_mask()

    # Load equal to the capacity, ample supply: S - r Q_t = 0 for the non-cohesive classes.
    ample_supply = jnp.full((3, 5), 1.0e6)
    source, rate = source_and_rate(closures, closures.capacity, ample_supply, config)
    net_erosion = source - rate * closures.capacity
    scale = jnp.max(jnp.abs(source[noncohesive]))
    np.testing.assert_allclose(net_erosion[noncohesive], 0.0, atol=1e-12 * scale)

    # Clear water (no load), small supply: the net erosion S stays below the supply.
    small_supply = jnp.full((3, 5), 1.0e-9)
    source, rate = source_and_rate(closures, jnp.zeros((3, 5)), small_supply, config)
    assert bool(jnp.all(source <= small_supply))
    assert bool(jnp.all(source >= 0.0))

    # Clear water, ample supply: the full erosion excess Q* / L is reached.
    source, rate = source_and_rate(closures, jnp.zeros((3, 5)), ample_supply, config)
    np.testing.assert_allclose(
        source[noncohesive], (closures.capacity * rate)[noncohesive], rtol=1e-6,
    )


def test_cohesive_deposition_and_erosion():
    """Krone deposition below tau_cd; no erosion below tau_ce; erosion above it."""
    config = SedimentConfig()
    theta = default_sediment_parameters(config.n_classes)

    # Three locations with tau_b = 0.02 Pa (deposition), 0.3 Pa (neither), 2 Pa (erosion).
    shear_stress = jnp.asarray([0.02, 0.3, 2.0])
    flow = FlowState(
        discharge=jnp.full(3, 100.0),
        width=jnp.full(3, 50.0),
        depth=jnp.full(3, 4.0),
        shear_stress=shear_stress,
        shear_velocity=jnp.sqrt(shear_stress / RHO),
        friction_coefficient=jnp.full(3, 3.0e-3),
    )
    fractions = jnp.full((3, 3), 1.0 / 3.0)
    closures = compute_closures(flow, fractions, theta, config)

    # Class 0 (20 micron) is the cohesive one.
    assert list(config.cohesive_mask()) == [True, False, False]
    w_fine = closures.settling_velocity[0]
    source = closures.cohesive_source[0]
    rate = closures.cohesive_rate[0]

    # Deposition at 0.02 Pa: r = B P_d w / Q with P_d = 1 - 0.02 / 0.1 = 0.8.
    np.testing.assert_allclose(rate[0], 50.0 * 0.8 * w_fine / 100.0, rtol=1e-3)

    # Between tau_cd and tau_ce: almost no deposition and no erosion.
    assert float(rate[1]) < 1.0e-6 * float(rate[0])
    assert float(source[1]) < 1.0e-6 * float(source[2])

    # Erosion at 2 Pa: S = B p M_e (tau_b / tau_ce - 1) = 50 * (1/3) * 1e-8 * 3.
    np.testing.assert_allclose(source[2], 50.0 * (1.0 / 3.0) * 1.0e-8 * 3.0, rtol=1e-9)

    # The capacity of the cohesive class is zero.
    np.testing.assert_array_equal(closures.capacity[0], 0.0)

    # "step" form of Krone: P_d is a smoothed indicator, about 1 well below tau_cd.
    config_step = dataclasses.replace(config, krone_form="step")
    closures_step = compute_closures(flow, fractions, theta, config_step)
    np.testing.assert_allclose(
        closures_step.cohesive_rate[0, 0], 50.0 * w_fine / 100.0, rtol=1e-3,
    )


def _exchange_objective(theta, flow, fractions, load, supply, config):
    """Scalar function of the source and rate, used for the derivative tests.

    Returns:
        sum(S) / mean|S| scale + sum(r) scale: 1e6 sum(S) + 1e3 sum(r) (mixed units; the
        factors only bring both terms to order one).
    """
    closures = compute_closures(flow, fractions, theta, config)
    source, rate = source_and_rate(closures, load, supply, config)
    return 1.0e6 * jnp.sum(source) + 1.0e3 * jnp.sum(rate)


def test_jit_and_gradients_match_finite_differences():
    """jit gives the same values; reverse-mode derivatives match central differences."""
    config = SedimentConfig()
    flow = _example_flow()
    fractions = _example_fractions(5)

    # Parameters away from their defaults, so that every term is active.
    theta = default_sediment_parameters(config.n_classes)
    theta["m_h"] = jnp.asarray(-0.3)
    theta["b_eh"] = jnp.asarray(2.3)
    theta["kappa_w"] = jnp.asarray([3.0, 0.8, 1.2])
    theta["tau_cd"] = jnp.asarray(0.4)

    # A load between zero and twice the capacity, and a supply of the order of the source.
    closures = compute_closures(flow, fractions, theta, config)
    factor = jnp.linspace(0.0, 2.0, 5)[None, :]
    load = factor * closures.capacity + 1.0e-4
    supply = jnp.full((3, 5), 1.0e-5)

    def objective(parameters):
        """The scalar objective as a function of the parameters only."""
        return _exchange_objective(parameters, flow, fractions, load, supply, config)

    # jit-compiled and eager values agree.
    value = objective(theta)
    value_jit = jax.jit(objective)(theta)
    np.testing.assert_allclose(value_jit, value, rtol=1e-13)

    # Reverse-mode gradient against central differences in each parameter entry.
    gradient = jax.grad(objective)(theta)
    for name, parameter in theta.items():
        flat = np.atleast_1d(np.asarray(parameter))
        for entry in range(flat.size):
            # Relative step; the parameters differ by many orders of magnitude (M_e ~ 1e-8).
            step = 1.0e-6 * abs(flat[entry]) if flat[entry] != 0.0 else 1.0e-8

            plus = flat.copy()
            plus[entry] += step
            minus = flat.copy()
            minus[entry] -= step

            theta_plus = dict(theta)
            theta_plus[name] = jnp.asarray(plus.reshape(np.shape(parameter)))
            theta_minus = dict(theta)
            theta_minus[name] = jnp.asarray(minus.reshape(np.shape(parameter)))

            difference = (objective(theta_plus) - objective(theta_minus)) / (2.0 * step)
            derivative = np.atleast_1d(np.asarray(gradient[name]))[entry]
            np.testing.assert_allclose(
                derivative, difference, rtol=1e-5, atol=1e-7 * abs(float(value)),
                err_msg=f"{name}[{entry}]",
            )


def test_hiding_factor_forms():
    """Two power laws and the network: limits, and derivatives of the network."""
    diameters = jnp.asarray([1.0e-4, 5.0e-4, 2.0e-3])
    mean_diameter = jnp.asarray([5.0e-4, 1.0e-3])
    shear_stress = jnp.asarray([1.0, 3.0])

    # Equal exponents of the two power laws are the power law.
    power = hiding_factor(diameters, mean_diameter, shear_stress, {"m_h": jnp.asarray(-0.4)}, SedimentConfig())
    two = hiding_factor(
        diameters, mean_diameter, shear_stress,
        {"m_h_fine": jnp.asarray(-0.4), "m_h_coarse": jnp.asarray(-0.4)},
        SedimentConfig(hiding_form="two_power"),
    )
    np.testing.assert_allclose(two, power, rtol=1e-13)

    for inputs in (("size_ratio",), ("size_ratio", "shields")):
        config = SedimentConfig(diameters=(1.0e-4, 5.0e-4, 2.0e-3), cohesive_dmax=0.0,
                                hiding_form="network", hiding_inputs=inputs)
        network = init_hiding_network(jax.random.PRNGKey(0), config)

        # No hiding at the start.
        start = hiding_factor(diameters, mean_diameter, shear_stress, {"hiding_network": network}, config)
        np.testing.assert_allclose(start, 1.0, rtol=1e-13)

        # A network with a non-zero output: 1 for the grain of the mean size (class 1 at
        # the first location), different from 1 elsewhere.
        network[-1]["W"] = network[-1]["W"] + 0.5
        factor = hiding_factor(diameters, mean_diameter, shear_stress, {"hiding_network": network}, config)
        np.testing.assert_allclose(factor[1, 0], 1.0, rtol=1e-13)
        assert abs(float(factor[0, 0]) - 1.0) > 1.0e-2

        # Derivative of the capacity with respect to one weight against central differences.
        flow = _example_flow()
        fractions = _example_fractions(5)
        theta = default_sediment_parameters(config.n_classes)

        def total(weight, network=network, theta=theta, flow=flow, fractions=fractions, config=config):
            layers = [dict(layer) for layer in network]
            layers[0]["W"] = layers[0]["W"].at[0, 0].set(weight)
            return jnp.sum(transport_capacity(flow, fractions, {**theta, "hiding_network": layers}, config))

        weight = network[0]["W"][0, 0]
        derivative = jax.grad(total)(weight)
        step = 1.0e-6
        difference = (total(weight + step) - total(weight - step)) / (2.0 * step)
        assert np.isfinite(float(derivative))
        np.testing.assert_allclose(derivative, difference, rtol=1e-6)


def _shear_flow(shear_stress) -> FlowState:
    """Flow state with a given bed shear stress (Pa) in a tube of unit width."""
    tau = jnp.atleast_1d(jnp.asarray(shear_stress, dtype=float))
    ones = jnp.ones_like(tau)
    return FlowState(
        discharge=ones, width=ones, depth=ones, shear_stress=tau,
        shear_velocity=jnp.sqrt(tau / RHO), friction_coefficient=0.005 * ones,
    )


def test_wilcock_crowe_capacity():
    """Wilcock-Crowe (2003): transport function, reference stress, hiding, derivatives."""
    theta = wilcock_crowe_parameters()
    scale = (2650.0 - RHO) * G

    # One gravel class (no sand): tau_r = 0.036 (rho_s - rho) g d, and
    # q* = W* u_*^3 / (R_s g) with W* = 0.002 at phi = 1 and 14 (1 - 0.894 / 2)^4.5 at phi = 4.
    gravel = SedimentConfig(diameters=(0.01,), cohesive_dmax=0.0, transport_formula="wilcock_crowe",
                            hiding_form="wilcock_crowe")
    tau_reference = 0.036 * scale * 0.01
    for phi, expected in ((1.0, 0.002), (4.0, 14.0 * (1.0 - 0.894 / 2.0) ** 4.5)):
        flow = _shear_flow(phi * tau_reference)
        capacity = transport_capacity(flow, jnp.ones((1, 1)), theta, gravel)
        np.testing.assert_allclose(capacity[0, 0], expected * float(flow.shear_velocity[0]) ** 3 / (RS * G), rtol=1e-9)

    # The two branches meet at phi = 1.35 (to the rounding of the published constants).
    below = transport_capacity(_shear_flow(1.3499 * tau_reference), jnp.ones((1, 1)), theta, gravel)
    above = transport_capacity(_shear_flow(1.3501 * tau_reference), jnp.ones((1, 1)), theta, gravel)
    np.testing.assert_allclose(below, above, rtol=5e-3)

    # A bed of sand only: reference Shields stress 0.021 + 0.015 exp(-20).
    sand = dataclasses.replace(gravel, diameters=(0.001,))
    tau_sand = (0.021 + 0.015 * np.exp(-20.0)) * scale * 0.001
    flow = _shear_flow(tau_sand)
    capacity = transport_capacity(flow, jnp.ones((1, 1)), theta, sand)
    np.testing.assert_allclose(capacity[0, 0], 0.002 * float(flow.shear_velocity[0]) ** 3 / (RS * G), rtol=1e-9)

    # Mixture. The published hiding function: tau_r,k / tau_r,m = ratio^b.
    diameters = jnp.asarray([5.0e-4, 4.0e-3, 3.2e-2])
    mixture = dataclasses.replace(gravel, diameters=tuple(float(d) for d in diameters))
    mean = jnp.asarray([4.0e-3])
    ratio = np.asarray(diameters) / 4.0e-3
    exponent = 0.67 / (1.0 + np.exp(1.5 - ratio))
    factor = hiding_factor(diameters, mean, jnp.asarray([1.0]), theta, mixture)
    np.testing.assert_allclose(ratio / np.asarray(factor[:, 0]), ratio**exponent, rtol=1e-12)
    np.testing.assert_allclose(factor[1, 0], 1.0, rtol=1e-12)

    # Equal mobility (power law with m_h = -1): the capacity per unit fraction is the same
    # for all classes; without hiding (m_h = 0) the coarse class moves least.
    fractions = jnp.asarray([[0.2], [0.5], [0.3]])
    flow = _shear_flow(8.0)
    power = dataclasses.replace(mixture, hiding_form="power")
    equal = transport_capacity(flow, fractions, {**theta, "m_h": jnp.asarray(-1.0)}, power) / fractions
    np.testing.assert_allclose(equal, equal[0, 0], rtol=1e-9)
    selective = transport_capacity(flow, fractions, {**theta, "m_h": jnp.asarray(0.0)}, power) / fractions
    assert float(selective[0, 0]) > float(selective[1, 0]) > float(selective[2, 0])

    # No shear stress: zero capacity and finite derivatives. Derivatives with respect to
    # the two parameters and the fractions against central differences, on both branches.
    def total(a_wc, tau_r_wc, fractions, shear):
        parameters = {"a_wc": a_wc, "tau_r_wc": tau_r_wc}
        return jnp.sum(wilcock_crowe_capacity(_shear_flow(shear), diameters, fractions, parameters, mixture))

    arguments = (jnp.asarray(1.0), jnp.asarray(1.0), fractions)
    assert float(total(*arguments, 0.0)) == 0.0
    for gradient in jax.grad(total, argnums=(0, 1, 2))(*arguments, 0.0):
        assert bool(jnp.all(jnp.isfinite(gradient)))
    for shear in (2.0, 8.0, 40.0):
        value_function = jax.jit(lambda a, t, f, shear=shear: total(a, t, f, shear))
        gradients = jax.grad(value_function, argnums=(0, 1, 2))(*arguments)
        step = 1.0e-6
        np.testing.assert_allclose(
            gradients[0], (value_function(1.0 + step, 1.0, fractions) - value_function(1.0 - step, 1.0, fractions)) / (2 * step),
            rtol=1e-6)
        np.testing.assert_allclose(
            gradients[1], (value_function(1.0, 1.0 + step, fractions) - value_function(1.0, 1.0 - step, fractions)) / (2 * step),
            rtol=1e-5)
        bump = jnp.zeros_like(fractions).at[2, 0].set(step)
        np.testing.assert_allclose(
            gradients[2][2, 0], (value_function(1.0, 1.0, fractions + bump) - value_function(1.0, 1.0, fractions - bump)) / (2 * step),
            rtol=1e-5)


def test_transport_functions_of_the_wilcock_crowe_formula():
    """The transport function W*(phi): Wilcock-Crowe, Parker (as in Cui, 2007), network."""
    theta = wilcock_crowe_parameters()
    published = SedimentConfig(transport_function="wilcock_crowe", hiding_form="wilcock_crowe")
    parker = SedimentConfig(transport_function="parker", hiding_form="wilcock_crowe")
    learned = SedimentConfig(transport_function="network", hiding_form="wilcock_crowe")

    # Both published functions are 0.002 at the reference stress and continuous where
    # their branches meet (phi = 1.35; phi = 1 and 1.59).
    one = jnp.asarray([1.0])
    np.testing.assert_allclose(transport_function(one, theta, published), 0.002, rtol=1e-12)
    np.testing.assert_allclose(transport_function(one, theta, parker), 0.002, rtol=1e-12)
    for config, breaks, tolerance in ((published, (1.35,), 2e-3), (parker, (1.0, 1.59), 2e-3)):
        for point in breaks:
            below, above = transport_function(jnp.asarray([point - 1e-9, point + 1e-9]), theta, config)
            np.testing.assert_allclose(below, above, rtol=tolerance)

    # Values of the function of Parker: 0.002 phi^14.2 below the reference stress, and
    # 10.95 (1 - 0.853 / phi)^4.5 far above it.
    phi = jnp.asarray([0.5, 2.0, 4.0])
    values = np.asarray(transport_function(phi, theta, parker))
    np.testing.assert_allclose(values[0], 0.002 * 0.5**14.2, rtol=1e-12)
    np.testing.assert_allclose(values[1], 10.95 * (1.0 - 0.853 / 2.0) ** 4.5, rtol=1e-12)
    np.testing.assert_allclose(values[2], 10.95 * (1.0 - 0.853 / 4.0) ** 4.5, rtol=1e-12)

    # It is steeper: smaller below the reference stress, 5.8 and 3.8 times larger at
    # phi = 2 and 4.
    ratio = values / np.asarray(transport_function(phi, theta, published))
    assert ratio[0] < 0.02
    np.testing.assert_allclose(ratio[1:], [5.8, 3.8], rtol=0.02)

    # Network: with its output layer at zero it is the function of Wilcock and Crowe;
    # with any weights it keeps W* = 0.002 at phi = 1.
    network = init_transport_network(jax.random.PRNGKey(0), learned)
    with_network = {**theta, "transport_network": network}
    wide = jnp.asarray(np.geomspace(0.05, 200.0, 40))
    np.testing.assert_allclose(
        transport_function(wide, with_network, learned),
        transport_function(wide, theta, published), rtol=1e-12,
    )
    perturbed = jax.tree.map(lambda leaf: leaf + 0.3, network)
    changed = {**theta, "transport_network": perturbed}
    np.testing.assert_allclose(transport_function(one, changed, learned), 0.002, rtol=1e-12)
    factor = transport_function(wide, changed, learned) / transport_function(wide, theta, published)
    assert float(jnp.max(jnp.abs(jnp.log(factor)))) > 0.1
    assert float(jnp.max(jnp.abs(jnp.log(factor)))) <= 2.0 * learned.transport_log_bound

    # The capacity uses the chosen function, and derivatives with respect to the weights
    # of the network match central differences.
    flow = _shear_flow(jnp.asarray([30.0]))
    diameters = jnp.asarray([0.5e-3, 4.0e-3, 32.0e-3])
    fractions = jnp.asarray([[0.3], [0.4], [0.3]])
    capacity_published = wilcock_crowe_capacity(flow, diameters, fractions, theta, published)
    capacity_parker = wilcock_crowe_capacity(flow, diameters, fractions, theta, parker)
    assert bool(jnp.all(capacity_parker > 0.0))
    assert not np.allclose(capacity_parker, capacity_published, rtol=0.05)

    # The stress ratio of the formula is available on its own: the capacity follows from
    # it, and for a bed of one gravel size it is tau_b / (0.036 (rho_s - rho) g d).
    ratio_classes = wilcock_crowe_stress_ratio(flow, diameters, fractions, theta, published)
    assert ratio_classes.shape == (3, 1)
    from_ratio = (
        theta["a_wc"] * fractions * transport_function(ratio_classes, theta, published)
        * flow.shear_velocity[None, :] ** 3 / ((2650.0 / RHO - 1.0) * G) * flow.width[None, :]
    )
    np.testing.assert_allclose(capacity_published, from_ratio, rtol=1e-12)
    one_size = wilcock_crowe_stress_ratio(flow, jnp.asarray([8.0e-3]), jnp.ones((1, 1)), theta, published)
    np.testing.assert_allclose(one_size, 30.0 / (0.036 * (2650.0 - RHO) * G * 8.0e-3), rtol=1e-9)

    def total(weight):
        params = [dict(layer) for layer in perturbed]
        params[-1]["W"] = params[-1]["W"].at[0, 0].set(weight)
        return jnp.sum(
            wilcock_crowe_capacity(flow, diameters, fractions, {**theta, "transport_network": params}, learned)
        )

    weight = perturbed[-1]["W"][0, 0]
    gradient = jax.grad(total)(weight)
    step = 1.0e-6
    difference = (total(weight + step) - total(weight - step)) / (2.0 * step)
    np.testing.assert_allclose(gradient, difference, rtol=1e-6)
