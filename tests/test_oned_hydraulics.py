"""Tests of the 1-D hydraulics: network, cross-section geometry and backwater solver.

Geometry (``hydrograd.oned.geometry``):

1. Rectangular and trapezoidal sections give the exact area, top width and wetted
   perimeter (with a sharp waterline).
2. The deposition width of a trapezoid with dry bank tops is the bank-to-bank mean width
   b + m H, independent of the water depth.
3. dA / d eta equals the top width, and derivatives with respect to the bed are finite,
   also for a table padded with repeated points.
3a. Conveyance: both formulas (``"segments"``, ``"section"``) give their exact values for
   a trapezoid and a rectangle; on a section with a bench the conveyance of the whole
   section falls with rising water surface and the sum over the segments does not.

Network (``hydrograd.oned.network``):

4. Headwaters, discharge accumulation and control-volume lengths of a chain and of a
   Y-shaped network; the bookkeeping of deposits conserves volume.

Backwater solver (``hydrograd.oned.backwater``):

5. Uniform flow: with the outlet at normal depth, the depth is normal everywhere.
6. M1 profile behind a dam: the depth grows towards the dam and the flow is subcritical.
7. A bed crest above the pool becomes a critical-flow control; the energy equation holds
   at all other sections; repeated solves from the previous solution do not drift.
8. Implicit-function-theorem derivatives with respect to Manning's n, the outlet level,
   the discharge and the bed match central differences, also through a control section;
   forward mode gives the same derivative as reverse mode.
9. Y-shaped network: the discharges add at the junction and the two branches have the
   same water surface where they meet.
10. Channel with a bench at the flow depth: with the default conveyance the Newton
    iterations converge for a sequence of discharges and outlet levels.

Deposit distribution (``hydrograd.oned.geometry``):

11. Depth-weighted deposit on a section with a flooded bench: the bench rises in
    proportion to its depth, and the deposition width still gives the deposited area.

Resistance law and shear stress (``hydrograd.oned.geometry``, ``hydrograd.oned.backwater``):

12. Bed shear stress with the hydraulic radius and with the mean depth.
13. Variable-power resistance law: coefficient and its limits, conveyance, uniform flow,
    reduced shear stress, derivatives with respect to the roughness length.
14. Logarithmic (Keulegan) resistance law: coefficient and its continuation to zero depth,
    conveyance, uniform flow, reduced shear stress, roughness length from the mean size of
    the active layer, derivatives.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hydrograd.oned import (
    BackwaterNumerics,
    GeometrySmoothing,
    HydraulicsConfig,
    converge,
    hydraulic_properties,
    network_from_positions,
    section_discharge,
    solve,
)
from hydrograd.oned.backwater import _residual_and_jacobian
from hydrograd.oned.geometry import critical_depth, deposition_width, thalweg
from hydrograd.oned.network import control_volume_lengths, reach_to_section
from hydrograd.oned.synthetic import (
    normal_depth_trapezoid,
    trapezoid_channel,
    y_network,
)

# Trapezoid of the synthetic channels (SI units).
BOTTOM_WIDTH = 60.0     # m
SIDE_SLOPE = 2.0        # horizontal per vertical (-)
BANK_HEIGHT = 30.0      # m
SLOPE = 1.0e-3          # -
N_MANNING = 0.03        # s/m^(1/3)
G = 9.81                # m/s^2

# A sharp waterline, for comparisons with exact geometry.
SHARP = GeometrySmoothing(beta_softplus=1000.0, eps_abs=1.0e-6, eps_wet=1.0e-3)

# Tight Newton tolerance, for derivative checks.
TIGHT = HydraulicsConfig(numerics=BackwaterNumerics(newton_tol=1.0e-11))


def _trapezoid_section() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Tables (z, s) of one trapezoidal section with its bottom at z = 0, shape [1, 4]."""
    bank_offset = SIDE_SLOPE * BANK_HEIGHT
    half_width = BOTTOM_WIDTH / 2.0
    s = jnp.asarray([[-(half_width + bank_offset), -half_width, half_width, half_width + bank_offset]])
    z = jnp.asarray([[BANK_HEIGHT, 0.0, 0.0, BANK_HEIGHT]])
    return z, s


def _bench_section() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Tables (z, s) of one section with a bench, bottom at z = 0, shape [1, 8].

    A channel with a 16 m wide bottom; on the right a 40 m wide bench 2.0 to 2.1 m above
    the bottom, then the outer bank. The last point is repeated twice (a padded table).
    """
    s = jnp.asarray([[0.0, 24.0, 40.0, 44.0, 84.0, 108.0, 108.0, 108.0]])
    z = jnp.asarray([[12.0, 0.0, 0.0, 2.0, 2.1, 14.1, 14.1, 14.1]])
    return z, s


def _channel(n_sections: int, length: float = 5000.0):
    """A trapezoidal channel and its network.

    Returns:
        Tuple ``(network, z, s)`` for ``n_sections`` sections over ``length`` (m).
    """
    x, z, s = trapezoid_channel(length=length, n_sections=n_sections, slope=SLOPE)
    return network_from_positions(x), z, s


def _channel_with_crest():
    """A 10 km channel of 21 sections with section 8 raised by 6 m (a crest)."""
    network, z, s = _channel(21, length=10000.0)
    z = z.at[8].add(6.0)
    return network, z, s


def test_rectangle_and_trapezoid_properties():
    """Exact area, top width and wetted perimeter with a sharp waterline."""
    # Rectangle 10 m wide with vertical walls 5 m high, water depth 2 m.
    s = jnp.asarray([[-5.0, -5.0, 5.0, 5.0]])
    z = jnp.asarray([[5.0, 0.0, 0.0, 5.0]])
    rectangle = hydraulic_properties(jnp.asarray([2.0]), z, s, jnp.asarray([0.03]), SHARP)
    np.testing.assert_allclose(rectangle.area, 20.0, atol=1e-3)
    np.testing.assert_allclose(rectangle.top_width, 10.0, atol=1e-3)
    np.testing.assert_allclose(rectangle.wetted_perimeter, 14.0, atol=1e-3)

    # Trapezoid with the water on the banks, depth y = 3 m.
    z, s = _trapezoid_section()
    y = 3.0
    trapezoid = hydraulic_properties(jnp.asarray([y]), z, s, jnp.asarray([0.03]), SHARP)
    np.testing.assert_allclose(trapezoid.area, (BOTTOM_WIDTH + SIDE_SLOPE * y) * y, atol=1e-2)
    np.testing.assert_allclose(trapezoid.top_width, BOTTOM_WIDTH + 2 * SIDE_SLOPE * y, atol=1e-2)



def test_conveyance_formulas():
    """Exact conveyance of a trapezoid and a rectangle for both formulas."""
    n_manning = 0.03
    y = 3.0

    # Trapezoid, whole section: K = A R^(2/3) / n.
    z, s = _trapezoid_section()
    eta = jnp.asarray([y])
    section = hydraulic_properties(eta, z, s, jnp.asarray([n_manning]), SHARP, "section")
    radius = section.area / section.wetted_perimeter
    np.testing.assert_allclose(
        section.conveyance, section.area * radius ** (2.0 / 3.0) / n_manning, rtol=1e-13,
    )
    area = (BOTTOM_WIDTH + SIDE_SLOPE * y) * y
    perimeter = BOTTOM_WIDTH + 2.0 * y * (1.0 + SIDE_SLOPE**2) ** 0.5
    np.testing.assert_allclose(
        section.conveyance, area * (area / perimeter) ** (2.0 / 3.0) / n_manning, rtol=1e-3,
    )

    # Trapezoid, segments (default): the bottom slice and the two bank slices.
    segments = hydraulic_properties(eta, z, s, jnp.asarray([n_manning]), SHARP)
    bank_area = 0.5 * SIDE_SLOPE * y * y
    bank_perimeter = y * (1.0 + SIDE_SLOPE**2) ** 0.5
    expected = (
        BOTTOM_WIDTH * y ** (5.0 / 3.0)
        + 2.0 * bank_area ** (5.0 / 3.0) / bank_perimeter ** (2.0 / 3.0)
    ) / n_manning
    np.testing.assert_allclose(segments.conveyance, expected, rtol=1e-3)

    # The other properties do not depend on the formula.
    np.testing.assert_allclose(segments.area, section.area, rtol=1e-14)
    np.testing.assert_allclose(segments.hydraulic_radius, section.hydraulic_radius, rtol=1e-14)

    # Rectangle 10 m wide, depth 2 m: the vertical walls add nothing to the sum over the
    # segments, K = b h^(5/3) / n; the whole section has R = 20 / 14 m.
    s = jnp.asarray([[-5.0, -5.0, 5.0, 5.0]])
    z = jnp.asarray([[5.0, 0.0, 0.0, 5.0]])
    eta = jnp.asarray([2.0])
    segments = hydraulic_properties(eta, z, s, jnp.asarray([n_manning]), SHARP)
    section = hydraulic_properties(eta, z, s, jnp.asarray([n_manning]), SHARP, "section")
    np.testing.assert_allclose(
        segments.conveyance, 10.0 * 2.0 ** (5.0 / 3.0) / n_manning, rtol=1e-3,
    )
    np.testing.assert_allclose(
        section.conveyance, 20.0 * (20.0 / 14.0) ** (2.0 / 3.0) / n_manning, rtol=1e-3,
    )

    # An unknown formula is an error.
    with pytest.raises(ValueError, match="conveyance"):
        hydraulic_properties(eta, z, s, jnp.asarray([n_manning]), SHARP, "sections")


def test_conveyance_on_a_bench():
    """The whole-section conveyance falls where the water floods a bench; the sum does not."""
    z, s = _bench_section()
    smoothing = GeometrySmoothing()
    n_levels = 400
    eta = jnp.linspace(0.3, 6.0, n_levels)
    z_levels = jnp.broadcast_to(z, (n_levels, z.shape[1]))
    s_levels = jnp.broadcast_to(s, (n_levels, s.shape[1]))

    def largest_fall(formula: str) -> float:
        """Largest fall of K below its running maximum, relative to that maximum (-)."""
        properties = hydraulic_properties(
            eta, z_levels, s_levels, jnp.asarray(0.03), smoothing, formula,
        )
        K = np.asarray(properties.conveyance)
        highest = np.maximum.accumulate(K)
        return float(np.max((highest - K) / highest))

    assert largest_fall("section") > 0.2
    assert largest_fall("segments") == 0.0


def test_deposition_width_is_polygon_mean_width():
    """B_d = b + m H for a trapezoid with dry bank tops, at any water depth."""
    z, s = _trapezoid_section()
    smoothing = GeometrySmoothing(eps_wet=1.0e-3)
    expected = BOTTOM_WIDTH + SIDE_SLOPE * BANK_HEIGHT

    _, width_shallow = deposition_width(jnp.asarray([3.0]), z, s, smoothing)
    _, width_deep = deposition_width(jnp.asarray([10.0]), z, s, smoothing)

    np.testing.assert_allclose(width_shallow, expected, atol=1.0)
    np.testing.assert_allclose(width_deep, width_shallow, atol=1.0)


def test_depth_weighted_deposit_fills_the_channel_first():
    """Weights (h / h_max)^m: bench against channel, and B_d dz = deposited area."""
    z, s = _bench_section()
    smoothing = GeometrySmoothing(eps_wet=1.0e-3)
    eta = jnp.asarray([3.0])

    # Uniform shift: the flooded bench rises like the channel bottom.
    uniform, width_uniform = deposition_width(eta, z, s, smoothing)
    np.testing.assert_allclose(uniform[0, [1, 3]], 1.0, atol=1e-6)

    # Depth-weighted: channel bottom 1, bench points by their depth (1.0 m and 0.9 m of 3 m).
    weights, width = deposition_width(eta, z, s, smoothing, depth_exponent=1.0)
    np.testing.assert_allclose(weights[0, 1:5], [1.0, 1.0, 1.0 / 3.0, 0.3], atol=1e-6)
    np.testing.assert_allclose(weights[0, [0, 5]], 0.0, atol=1e-12)
    assert float(width[0]) < float(width_uniform[0])

    # The width is the area placed per unit bed change of the deepest point.
    dz = 0.05
    ones = jnp.ones(1)
    datum = jnp.asarray([20.0])
    before = hydraulic_properties(datum, z, s, ones, smoothing).area
    after = hydraulic_properties(datum, z + dz * weights, s, ones, smoothing).area
    np.testing.assert_allclose(before - after, width * dz, rtol=1e-6)


def test_geometry_derivatives():
    """dA / d eta is the top width; derivatives with respect to the bed are finite."""
    z, s = _trapezoid_section()
    smoothing = GeometrySmoothing()
    eta = jnp.asarray([2.0])
    n_manning = jnp.asarray([0.03])

    def total_area(eta_value, z_value):
        """Sum of the flow areas (m^2)."""
        return jnp.sum(hydraulic_properties(eta_value, z_value, s, n_manning, smoothing).area)

    dA_deta, dA_dz = jax.grad(total_area, argnums=(0, 1))(eta, z)
    top_width = hydraulic_properties(eta, z, s, n_manning, smoothing).top_width

    np.testing.assert_allclose(dA_deta, top_width, rtol=5e-3)
    assert bool(jnp.all(jnp.isfinite(dA_dz)))

    # A table padded with repeated points and with dry segments far above the water:
    # the derivatives of the wetted perimeter and of both conveyances are finite, and
    # zero at the padding.
    z, s = _bench_section()
    eta = jnp.asarray([3.0])

    for formula in ("segments", "section"):

        def perimeter_and_conveyance(z_value, formula=formula):
            """Wetted perimeter (m) plus conveyance (m^3/s) of the bench section."""
            properties = hydraulic_properties(eta, z_value, s, n_manning, smoothing, formula)
            return jnp.sum(properties.wetted_perimeter + properties.conveyance)

        gradient = jax.grad(perimeter_and_conveyance)(z)
        assert bool(jnp.all(jnp.isfinite(gradient)))
        np.testing.assert_allclose(gradient[0, -2:], 0.0, atol=1e-12)


def test_network_topology_and_bookkeeping():
    """Headwaters, discharge, control volumes and deposit bookkeeping of chain and Y."""
    # Chain of five sections: one headwater, the same discharge everywhere.
    chain = network_from_positions(jnp.linspace(0.0, 1000.0, 5))
    assert chain.is_chain
    assert chain.headwaters.tolist() == [0]
    np.testing.assert_allclose(section_discharge(chain, jnp.asarray(3.0)), 3.0)

    # Y network: branch A (3 sections), branch B (2 sections), main stem (5 sections).
    y = y_network(n_main=5, n_branch=(3, 2))
    network = y.network
    assert not network.is_chain
    assert network.headwaters.tolist() == [0, 3]

    # The discharges of the branches add on the main stem.
    Q = section_discharge(network, jnp.asarray([10.0, 5.0]))
    np.testing.assert_allclose(Q[:3], 10.0)
    np.testing.assert_allclose(Q[3:5], 5.0)
    np.testing.assert_allclose(Q[5:], 15.0)

    # Collocated control volumes: the junction (section 5) owns both incoming reaches.
    lengths = control_volume_lengths(network)
    dx = network.reach_length
    np.testing.assert_allclose(lengths[5], dx[2] + dx[4], rtol=1e-14)

    # Deposits per reach (with the two virtual inflow reaches) keep their total volume.
    n_reaches = network.n_reaches
    deposition = jnp.arange(1.0, 2.0 * (n_reaches + 2) + 1.0).reshape(2, n_reaches + 2)
    at_sections = reach_to_section(network, deposition)
    np.testing.assert_allclose(jnp.sum(at_sections, axis=1), jnp.sum(deposition, axis=1))
    half = reach_to_section(network, deposition[:, :n_reaches], scheme="upstream_half")
    np.testing.assert_allclose(jnp.sum(half, axis=1), jnp.sum(deposition[:, :n_reaches], axis=1))


def test_uniform_flow_and_m1_profile():
    """Normal depth everywhere for a normal-depth outlet; M1 profile behind a dam."""
    network, z, s = _channel(11)
    Q = 200.0
    bed = jnp.min(z, axis=1)

    def profile(eta_outlet):
        """Hydraulic state for the outlet level ``eta_outlet`` (m)."""
        return solve(TIGHT, network, z, s, jnp.asarray(Q), eta_outlet, jnp.asarray([N_MANNING]))

    profile = jax.jit(profile)

    # Outlet at normal depth: uniform flow.
    y_normal = normal_depth_trapezoid(Q, N_MANNING, SLOPE, BOTTOM_WIDTH, SIDE_SLOPE)
    uniform = profile(jnp.asarray(100.0 + y_normal))
    np.testing.assert_allclose(uniform.eta - bed, y_normal, atol=2e-3)

    # The same with the conveyance of the whole section, which has its own normal depth.
    whole_section = HydraulicsConfig(
        numerics=BackwaterNumerics(newton_tol=1.0e-11), conveyance="section",
    )
    y_normal_section = normal_depth_trapezoid(
        Q, N_MANNING, SLOPE, BOTTOM_WIDTH, SIDE_SLOPE, conveyance="section",
    )
    uniform_section = solve(
        whole_section, network, z, s, jnp.asarray(Q), jnp.asarray(100.0 + y_normal_section),
        jnp.asarray([N_MANNING]),
    )
    np.testing.assert_allclose(uniform_section.eta - bed, y_normal_section, atol=2e-3)
    assert y_normal < y_normal_section

    # Outlet 12 m above the bed: depth increases towards the dam, flow is subcritical.
    backwater = profile(jnp.asarray(112.0))
    depth = backwater.eta - bed
    assert bool(jnp.all(depth[1:] > depth[:-1]))
    froude = backwater.velocity / jnp.sqrt(G * backwater.depth)
    assert bool(jnp.all(froude < 1.0))

    # Derived quantities are consistent: tau_b = rho g R S_f and u_* = sqrt(tau_b / rho).
    expected_stress = 1000.0 * G * backwater.hydraulic_radius * backwater.friction_slope
    np.testing.assert_allclose(backwater.shear_stress, expected_stress, rtol=1e-13)
    np.testing.assert_allclose(
        backwater.shear_velocity, jnp.sqrt(backwater.shear_stress / 1000.0), rtol=1e-13,
    )


def test_crest_becomes_control():
    """A crest above the pool is a critical-flow control; the solve does not drift."""
    network, z, s = _channel_with_crest()
    Q = jnp.asarray(150.0)
    eta_outlet = jnp.asarray(103.0)
    n_manning = jnp.asarray([N_MANNING])

    newton = converge(TIGHT, network, z, s, Q, eta_outlet, n_manning)
    hydro = solve(TIGHT, network, z, s, Q, eta_outlet, n_manning)
    assert bool(jnp.all(jnp.isfinite(hydro.eta)))

    # Section 8 (the crest) is a control at critical depth.
    assert bool(newton.is_control[8])
    y_critical = critical_depth(jnp.full(21, 150.0), z, s, TIGHT.smoothing)
    np.testing.assert_allclose(hydro.eta[8], jnp.min(z[8]) + y_critical[8], atol=1e-6)

    # Subcritical flow up to and including the crest.
    froude = hydro.velocity / jnp.sqrt(G * hydro.depth)
    assert bool(jnp.all(froude[:9] <= 1.0 + 1e-6))

    # The energy equation holds at the sections that are not controls.
    g = TIGHT.constants.g
    head = hydro.eta + hydro.discharge**2 / (2.0 * g * hydro.area**2)
    loss = 0.5 * network.reach_length * (hydro.friction_slope[:-1] + hydro.friction_slope[1:])
    energy_residual = head[:-1] - head[1:] - loss
    assert float(jnp.max(jnp.abs(energy_residual[~newton.is_control]))) < 1e-6

    # Repeated solves that start from the previous solution stay on it.
    def resolve(eta_guess):
        """Solve again from ``eta_guess`` and return the water surface (m)."""
        return solve(TIGHT, network, z, s, Q, eta_outlet, n_manning, eta_guess).eta

    resolve = jax.jit(resolve)
    eta = hydro.eta
    for _ in range(20):
        eta = resolve(eta)
    np.testing.assert_allclose(eta, hydro.eta, atol=1e-8)


def test_gradients_match_finite_differences():
    """IFT derivatives against central differences, through a control section."""
    network, z, s = _channel_with_crest()

    # Direction of a bed perturbation: random shifts of all station points (m per unit).
    rng = np.random.default_rng(3)
    direction = jnp.asarray(rng.uniform(-1.0, 1.0, size=z.shape))

    def objective(n_manning, eta_outlet, Q, bed_shift):
        """Sum of the water surface upstream of the crest plus a shear-stress term."""
        z_shifted = z + bed_shift * direction
        hydro = solve(TIGHT, network, z_shifted, s, Q, eta_outlet, n_manning)
        return jnp.sum(hydro.eta[:9]) + 1.0e-2 * jnp.sum(hydro.shear_stress)

    values = (jnp.asarray([N_MANNING]), jnp.asarray(103.0), jnp.asarray(150.0), jnp.asarray(0.0))
    objective_jit = jax.jit(objective)
    gradient = jax.jit(jax.grad(objective, argnums=(0, 1, 2, 3)))(*values)

    # Central differences in each argument.
    steps = (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-5)
    for position, step in enumerate(steps):
        plus = list(values)
        plus[position] = values[position] + step
        minus = list(values)
        minus[position] = values[position] - step

        difference = (objective_jit(*plus) - objective_jit(*minus)) / (2.0 * step)
        derivative = jnp.sum(gradient[position])
        np.testing.assert_allclose(derivative, difference, rtol=1e-5, err_msg=str(position))

    # Raising the outlet level lowers nothing upstream of the crest (a control hides the
    # pool), but it changes the shear stress in the pool: the derivative is not zero.
    assert float(gradient[1]) != 0.0

    # Forward mode gives the same derivative with respect to Manning's n.
    def objective_of_n(n_manning):
        """The objective as a function of Manning's n only."""
        return objective(n_manning, values[1], values[2], values[3])

    _, tangent = jax.jvp(objective_of_n, (values[0],), (jnp.ones(1),))
    np.testing.assert_allclose(tangent, gradient[0][0], rtol=1e-10)


def test_y_network_backwater():
    """Discharges add at the junction; both branches meet the same water surface."""
    y = y_network(
        branch_lengths=(5000.0, 4000.0),
        n_branch=(10, 8),
        branch_widths=(40.0, 30.0),
        branch_slopes=(1.2e-3, 1.5e-3),
    )
    network = y.network
    Q_headwaters = jnp.asarray([120.0, 60.0])
    n_manning = jnp.asarray([N_MANNING])

    hydro = solve(TIGHT, network, y.z, y.s, Q_headwaters, jnp.asarray(112.0), n_manning)
    assert bool(jnp.all(jnp.isfinite(hydro.eta)))

    # Discharge: 120 on branch A, 60 on branch B, 180 on the main stem.
    np.testing.assert_allclose(hydro.discharge[:10], 120.0)
    np.testing.assert_allclose(hydro.discharge[10:18], 60.0)
    np.testing.assert_allclose(hydro.discharge[18:], 180.0)

    # The last section of each branch satisfies the energy equation towards the junction
    # (section 18), so both see the same downstream head.
    g = TIGHT.constants.g
    head = hydro.eta + hydro.discharge**2 / (2.0 * g * hydro.area**2)
    for last in (9, 17):
        loss = 0.5 * network.reach_length[last] * (
            hydro.friction_slope[last] + hydro.friction_slope[18]
        )
        np.testing.assert_allclose(head[last] - head[18] - loss, 0.0, atol=1e-8)

    # The water surface falls in the flow direction along every reach.
    assert bool(jnp.all(hydro.eta[:-1] >= hydro.eta[network.downstream] - 1e-9))


def test_backwater_converges_with_a_bench():
    """Newton converges when the water surface lies at the level of a bench.

    With the conveyance of the whole section the energy equation of this channel has
    several roots and the iterations cycle (up to the iteration limit); with the default
    conveyance they converge in a few iterations.
    """
    # 21 sections with the bench 2.0 to 2.1 m above the bottom, 200 m apart, slope 0.001.
    n_sections = 21
    x = jnp.linspace(0.0, 4000.0, n_sections)
    bottom = 100.0 + SLOPE * (x[-1] - x)
    z_one, s_one = _bench_section()
    z = bottom[:, None] + z_one
    s = jnp.broadcast_to(s_one, z.shape)
    network = network_from_positions(x)
    n_manning = jnp.asarray([N_MANNING])
    config = HydraulicsConfig()

    def converged(Q, eta_outlet, eta_guess):
        """Newton state for the discharge ``Q`` and the outlet level, from a guess."""
        return converge(config, network, z, s, Q, eta_outlet, n_manning, eta_guess)

    converged = jax.jit(converged)

    # Discharges (m^3/s) and outlet depths (m) around the bench, each solve starting from
    # the previous water surface, as in a time loop.
    eta_guess = None
    for Q, outlet_depth in ((20.0, 2.0), (30.0, 2.15), (40.0, 2.3), (50.0, 2.2), (45.0, 2.05)):
        Q = jnp.asarray(Q)
        eta_outlet = jnp.asarray(100.0 + outlet_depth)
        if eta_guess is None:
            eta_guess = jnp.full(n_sections, eta_outlet)
        newton = converged(Q, eta_outlet, eta_guess)
        assert int(newton.n_iter) <= 10

        # The energy equation holds at the returned water surface.
        discharge = section_discharge(network, Q)
        y_critical = critical_depth(discharge, z, s, config.smoothing)
        floor = thalweg(z)[:-1] + y_critical[:-1]
        residual, _, _ = _residual_and_jacobian(
            config, network, newton.eta_own, eta_outlet, discharge, z, s,
            jnp.broadcast_to(n_manning, (n_sections,)), newton.is_control, floor,
        )
        assert not bool(jnp.any(newton.is_control))
        np.testing.assert_allclose(residual, 0.0, atol=1e-9)

        eta_guess = jnp.concatenate([newton.eta_own, jnp.reshape(eta_outlet, (1,))])


def test_bed_shear_stress_radius():
    """The bed shear stress with the hydraulic radius of the section and with the mean depth."""
    from hydrograd.oned.backwater import hydro_state_from_surface

    # A flume 0.3 m wide with vertical walls, 0.15 m of water, discharge 0.049 m^3/s.
    width, depth, discharge, manning = 0.3, 0.15, 0.049, 0.03
    s = jnp.asarray([[0.0, 0.0, width, width]])
    z = jnp.asarray([[1.0, 0.0, 0.0, 1.0]])
    eta = jnp.asarray([depth])
    Q = jnp.asarray([discharge])
    n = jnp.asarray([manning])
    section = HydraulicsConfig(smoothing=SHARP)
    by_depth = HydraulicsConfig(smoothing=SHARP, shear_radius="depth")

    # Friction slope of the default conveyance K = b h^(5/3) / n.
    friction_slope = (discharge * manning / (width * depth ** (5.0 / 3.0))) ** 2
    with_section = hydro_state_from_surface(section, eta, Q, z, s, n)
    with_depth = hydro_state_from_surface(by_depth, eta, Q, z, s, n)
    np.testing.assert_allclose(with_section.friction_slope, friction_slope, rtol=1e-6)

    # Hydraulic radius b h / (b + 2 h) = half the depth here; mean depth h.
    radius = width * depth / (width + 2.0 * depth)
    np.testing.assert_allclose(with_section.shear_stress, 1000.0 * G * radius * friction_slope, rtol=1e-6)
    np.testing.assert_allclose(with_depth.shear_stress, 1000.0 * G * depth * friction_slope, rtol=1e-6)
    np.testing.assert_allclose(with_depth.shear_stress / with_section.shear_stress, 2.0, rtol=1e-6)
    np.testing.assert_allclose(
        with_depth.shear_velocity, jnp.sqrt(with_depth.shear_stress / 1000.0), rtol=1e-12,
    )

    # A wide trapezoid (60 m bottom, 3 m of water): the two differ by a few percent.
    z_wide, s_wide = _trapezoid_section()
    wide_section = hydro_state_from_surface(section, jnp.asarray([3.0]), jnp.asarray([300.0]), z_wide, s_wide, n)
    wide_depth = hydro_state_from_surface(by_depth, jnp.asarray([3.0]), jnp.asarray([300.0]), z_wide, s_wide, n)
    ratio = float(wide_depth.shear_stress[0] / wide_section.shear_stress[0])
    assert 1.0 < ratio < 1.12

    with pytest.raises(ValueError, match="shear_radius must be one of"):
        hydro_state_from_surface(
            HydraulicsConfig(shear_radius="width"), eta, Q, z, s, n,
        )


def test_variable_power_resistance():
    """The variable-power resistance law: coefficient, conveyance, uniform flow, reduced
    shear stress, derivatives with respect to the roughness length."""
    from hydrograd.oned.backwater import hydro_state_from_surface
    from hydrograd.oned.config import VPE_A1, VPE_A2
    from hydrograd.oned.geometry import variable_power_coefficient

    # Values of C = U / u_* and its two limits: Manning-Strickler for deep flow, linear in
    # the relative depth for shallow flow.
    np.testing.assert_allclose(
        variable_power_coefficient(jnp.asarray([6.3, 2.5])), [7.70, 4.82], rtol=2e-3,
    )
    deep = jnp.asarray([1.0e4])
    np.testing.assert_allclose(
        variable_power_coefficient(deep), VPE_A1 * deep ** (1.0 / 6.0), rtol=1e-5,
    )
    shallow = jnp.asarray([1.0e-3])
    np.testing.assert_allclose(variable_power_coefficient(shallow), VPE_A2 * shallow, rtol=1e-4)

    # Conveyance of a flume 0.3 m wide with vertical walls: K = b h sqrt(g R) C(R / k),
    # with R = h for the segments (no wall friction) and b h / (b + 2 h) for the section.
    width, roughness = 0.3, 0.022
    s = jnp.asarray([[0.0, 0.0, width, width]])
    z = jnp.asarray([[1.0, 0.0, 0.0, 1.0]])
    k = jnp.asarray([roughness])
    for depth in (0.05, 0.14):
        eta = jnp.asarray([depth])
        for formula, radius in (("segments", depth), ("section", width * depth / (width + 2.0 * depth))):
            properties = hydraulic_properties(eta, z, s, k, SHARP, formula, "variable_power", G)
            coefficient = float(variable_power_coefficient(jnp.asarray(radius / roughness)))
            np.testing.assert_allclose(
                properties.conveyance, width * depth * np.sqrt(G * radius) * coefficient,
                rtol=1e-5, err_msg=f"{formula} {depth}",
            )

    # For deep flow the law is Manning's with n = k^(1/6) / (a1 sqrt(g)).
    z_wide, s_wide = _trapezoid_section()
    strickler = roughness ** (1.0 / 6.0) / (VPE_A1 * np.sqrt(G))
    eta_deep = jnp.asarray([20.0])
    with_k = hydraulic_properties(eta_deep, z_wide, s_wide, k, SHARP, "segments", "variable_power", G)
    with_n = hydraulic_properties(eta_deep, z_wide, s_wide, jnp.asarray([strickler]), SHARP)
    np.testing.assert_allclose(with_k.conveyance, with_n.conveyance, rtol=2e-3)

    # Reduced shear stress: tau_b (C / C_deep)^1.5, C_deep = a1 x^(1/6); 0.81 of the total
    # at a depth of 6 roughness lengths and 0.50 at 2.5. The friction coefficient keeps
    # the total stress.
    total = HydraulicsConfig(smoothing=SHARP, shear_radius="depth", resistance="variable_power")
    reduced = HydraulicsConfig(
        smoothing=SHARP, shear_radius="depth", resistance="variable_power", transport_stress="reduced",
    )
    for depth, expected in ((0.139, 0.81), (0.055, 0.50)):
        eta, Q = jnp.asarray([depth]), jnp.asarray([0.049])
        state_total = hydro_state_from_surface(total, eta, Q, z, s, k)
        state_reduced = hydro_state_from_surface(reduced, eta, Q, z, s, k)
        x = depth / roughness
        factor = (float(variable_power_coefficient(jnp.asarray(x))) / (VPE_A1 * x ** (1.0 / 6.0))) ** 1.5
        np.testing.assert_allclose(state_reduced.shear_stress / state_total.shear_stress, factor, rtol=1e-5)
        np.testing.assert_allclose(factor, expected, atol=0.01)
        np.testing.assert_allclose(state_reduced.friction_coefficient, state_total.friction_coefficient, rtol=1e-12)
        np.testing.assert_allclose(state_total.shear_stress, 1000.0 * G * depth * state_total.friction_slope, rtol=1e-6)
    with pytest.raises(ValueError, match="variable_power"):
        hydro_state_from_surface(
            HydraulicsConfig(transport_stress="reduced"), jnp.asarray([0.1]), jnp.asarray([0.049]), z, s, k,
        )
    with pytest.raises(ValueError, match="resistance"):
        hydraulic_properties(jnp.asarray([0.1]), z, s, k, SHARP, "segments", "chezy")

    # Uniform flow of the backwater solver in a flume on a mild slope: the depth solves
    # q = C(h / k) h^(3/2) sqrt(g S).
    slope, discharge = 0.002, 0.049
    x_flume, z_flume, s_flume = trapezoid_channel(
        length=60.0, n_sections=13, slope=slope, bottom_width=width, side_slope=1.0e-4,
        bank_height=1.0, z_dam_bed=0.0,
    )
    network = network_from_positions(x_flume)
    low, high = 0.05, 1.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        flow = float(variable_power_coefficient(jnp.asarray(middle / roughness))) * middle**1.5 * np.sqrt(G * slope)
        low, high = (middle, high) if flow < discharge / width else (low, middle)
    normal_depth = 0.5 * (low + high)
    law = HydraulicsConfig(
        numerics=BackwaterNumerics(newton_tol=1.0e-11), smoothing=SHARP, resistance="variable_power",
    )
    uniform = solve(law, network, z_flume, s_flume, jnp.asarray(discharge), jnp.asarray(normal_depth), k)
    np.testing.assert_allclose(uniform.eta - jnp.min(z_flume, axis=1), normal_depth, atol=1e-4)

    # Derivatives with respect to the roughness length match central differences, with
    # both stresses and through the solver (a deeper outlet, so that the profile varies).
    for config in (law, dataclasses.replace(law, shear_radius="depth", transport_stress="reduced")):
        def objective(roughness_length, config=config):
            hydro = solve(config, network, z_flume, s_flume, jnp.asarray(discharge), jnp.asarray(0.35), roughness_length)
            return jnp.sum(hydro.eta) + 1.0e-2 * jnp.sum(hydro.shear_stress)

        gradient = jax.grad(objective)(k)
        step = 1.0e-7
        difference = (objective(k + step) - objective(k - step)) / (2.0 * step)
        np.testing.assert_allclose(jnp.sum(gradient), difference, rtol=1e-5)


def test_keulegan_resistance():
    """The logarithmic resistance law: coefficient, conveyance, uniform flow, reduced shear
    stress, roughness from the bed surface, derivatives."""
    import math

    from hydrograd.oned.backwater import hydro_state_from_surface
    from hydrograd.oned.config import resistance_parameter
    from hydrograd.oned.geometry import keulegan_coefficient, stress_reduction
    from hydrograd.sediment.closures import grain_size_percentile

    # C = 2.5 ln(11 x); below x0 = e / 11 the tangent through the origin, with the same
    # value (2.5) and slope at x0; positive and increasing down to zero.
    np.testing.assert_allclose(
        keulegan_coefficient(jnp.asarray([1.0, 2.0, 10.0])), 2.5 * np.log([11.0, 22.0, 110.0]), rtol=1e-12,
    )
    x0 = math.e / 11.0
    np.testing.assert_allclose(keulegan_coefficient(jnp.asarray([x0])), 2.5, rtol=1e-12)
    slope = jax.vmap(jax.grad(lambda x: keulegan_coefficient(x)))(jnp.asarray([x0 - 1e-6, x0 + 1e-6]))
    np.testing.assert_allclose(slope, 2.5 / x0, rtol=1e-5)
    low = keulegan_coefficient(jnp.asarray([0.0, 0.01, 0.05, 0.1, 0.2]))
    np.testing.assert_allclose(low, (2.5 / x0) * np.array([0.0, 0.01, 0.05, 0.1, 0.2]), rtol=1e-12)
    assert bool(jnp.all(jnp.diff(low) > 0.0))

    # Conveyance of a flume 0.3 m wide: K = b h sqrt(g h) C(h / k_s), also for a depth of
    # a fifth of the roughness height (the tangent).
    width, roughness = 0.3, 0.03
    s = jnp.asarray([[0.0, 0.0, width, width]])
    z = jnp.asarray([[1.0, 0.0, 0.0, 1.0]])
    k = jnp.asarray([roughness])
    for depth, height in ((0.06, 0.3), (0.06, roughness), (0.14, roughness)):
        properties = hydraulic_properties(
            jnp.asarray([depth]), z, s, jnp.asarray([height]), SHARP, "segments", "keulegan", G,
        )
        coefficient = float(keulegan_coefficient(jnp.asarray(depth / height)))
        np.testing.assert_allclose(
            properties.conveyance, width * depth * np.sqrt(G * depth) * coefficient, rtol=1e-5,
        )

    # Reduced shear stress: (C / (8.1 x^(1/6)))^1.5 in shallow flow, 1 from x = 14 on and
    # in deep flow, where the power law runs away from the logarithmic one.
    x = jnp.asarray([1.0, 2.5, 5.0])
    expected = (2.5 * np.log(11.0 * np.asarray(x)) / (8.1 * np.asarray(x) ** (1.0 / 6.0))) ** 1.5
    np.testing.assert_allclose(stress_reduction(x, "keulegan"), expected, rtol=1e-12)
    np.testing.assert_allclose(expected, [0.64, 0.82, 0.92], atol=0.005)
    np.testing.assert_allclose(stress_reduction(jnp.asarray([14.0, 30.0, 36.0, 40.0, 1.0e3, 1.0e5]), "keulegan"), 1.0)
    assert bool(jnp.all(jnp.diff(stress_reduction(jnp.asarray(np.geomspace(0.01, 1.0e4, 200)), "keulegan")) >= 0.0))
    total = HydraulicsConfig(smoothing=SHARP, shear_radius="depth", resistance="keulegan")
    reduced = dataclasses.replace(total, transport_stress="reduced")
    eta, Q = jnp.asarray([0.075]), jnp.asarray([0.02])
    ratio = hydro_state_from_surface(reduced, eta, Q, z, s, k).shear_stress / hydro_state_from_surface(total, eta, Q, z, s, k).shear_stress
    np.testing.assert_allclose(ratio, stress_reduction(jnp.asarray([0.075 / roughness]), "keulegan"), rtol=1e-6)

    # Roughness length from the bed surface: the factor times the geometric mean size of
    # the active layer of each section.
    diameters = jnp.asarray([0.5e-3, 8.0e-3, 32.0e-3])
    p_active = jnp.asarray([[0.2, 1.0], [0.3, 0.0], [0.5, 0.0]])
    theta = {"n_manning": jnp.asarray(0.03), "roughness_factor": jnp.asarray(2.5), "roughness_length": jnp.asarray(0.02)}
    surface = HydraulicsConfig(resistance="keulegan", roughness_scale="surface_mean")
    mean_size = np.exp(0.2 * np.log(0.5e-3) + 0.3 * np.log(8.0e-3) + 0.5 * np.log(32.0e-3))
    np.testing.assert_allclose(
        resistance_parameter(theta, surface, p_active, diameters), 2.5 * np.array([mean_size, 0.5e-3]), rtol=1e-9,
    )
    np.testing.assert_allclose(resistance_parameter(theta, HydraulicsConfig(resistance="keulegan")), 0.02)

    # The same with the D90 of the active layer. Classes of one psi unit: the limits are
    # the geometric means of the diameters. Section 1: 0.2 / 0.3 / 0.5 from fine to
    # coarse, so that 90 % lies four fifths of the way through the coarsest class (in
    # ln d); section 2: all in the finest class.
    octaves = jnp.asarray([2.0e-3, 4.0e-3, 8.0e-3])
    d90 = np.array([np.sqrt(2.0) * 4.0e-3 * 2.0**0.8, np.sqrt(2.0) * 1.0e-3 * 2.0**0.9])
    np.testing.assert_allclose(grain_size_percentile(octaves, p_active, 90.0), d90, rtol=1e-6)
    np.testing.assert_allclose(grain_size_percentile(octaves, p_active, 50.0)[0], np.sqrt(2.0) * 4.0e-3, rtol=1e-6)
    by_d90 = HydraulicsConfig(resistance="keulegan", roughness_scale="surface_d90")
    np.testing.assert_allclose(resistance_parameter(theta, by_d90, p_active, octaves), 2.5 * d90, rtol=1e-6)
    slope_of_d90 = jax.grad(lambda p: jnp.sum(grain_size_percentile(octaves, p, 90.0)))(p_active)
    assert bool(jnp.all(jnp.isfinite(slope_of_d90)))
    np.testing.assert_allclose(resistance_parameter(theta, HydraulicsConfig()), 0.03)
    with pytest.raises(ValueError, match="surface_mean"):
        resistance_parameter(theta, HydraulicsConfig(roughness_scale="surface_mean"), p_active, diameters)
    with pytest.raises(ValueError, match="active layer"):
        resistance_parameter(theta, surface)

    # Uniform flow of the backwater solver: the depth solves q = C(h / k_s) h^(3/2) sqrt(g S).
    slope_bed, discharge = 0.002, 0.049
    x_flume, z_flume, s_flume = trapezoid_channel(
        length=60.0, n_sections=13, slope=slope_bed, bottom_width=width, side_slope=1.0e-4,
        bank_height=1.0, z_dam_bed=0.0,
    )
    network = network_from_positions(x_flume)
    low_depth, high_depth = 0.05, 1.0
    for _ in range(60):
        middle = 0.5 * (low_depth + high_depth)
        flow = float(keulegan_coefficient(jnp.asarray(middle / roughness))) * middle**1.5 * np.sqrt(G * slope_bed)
        low_depth, high_depth = (middle, high_depth) if flow < discharge / width else (low_depth, middle)
    normal_depth = 0.5 * (low_depth + high_depth)
    law = HydraulicsConfig(numerics=BackwaterNumerics(newton_tol=1.0e-11), smoothing=SHARP, resistance="keulegan")
    uniform = solve(law, network, z_flume, s_flume, jnp.asarray(discharge), jnp.asarray(normal_depth), k)
    np.testing.assert_allclose(uniform.eta - jnp.min(z_flume, axis=1), normal_depth, atol=1e-4)

    # Derivatives with respect to the roughness height match central differences, with
    # both stresses.
    for config in (law, dataclasses.replace(law, shear_radius="depth", transport_stress="reduced")):
        def objective(roughness_height, config=config):
            hydro = solve(config, network, z_flume, s_flume, jnp.asarray(discharge), jnp.asarray(0.35), roughness_height)
            return jnp.sum(hydro.eta) + 1.0e-2 * jnp.sum(hydro.shear_stress)

        gradient = jax.grad(objective)(k)
        step = 1.0e-7
        difference = (objective(k + step) - objective(k - step)) / (2.0 * step)
        np.testing.assert_allclose(jnp.sum(gradient), difference, rtol=1e-5)
