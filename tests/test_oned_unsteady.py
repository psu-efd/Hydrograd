"""Tests of the unsteady 1-D flow solver (``hydrograd.oned.unsteady``).

Geometry (``hydrograd.oned.geometry``):

1. The first moment I_1 of a rectangle and of a trapezoid is exact, and dI_1 / d eta is
   the flow area on an irregular section. The inverse of the area returns the water
   surface, and its derivatives with respect to the area and the bed match central
   differences, also for a flat bottom under a few centimetres of water, where the slope
   dA / d eta of the smooth area differs from the top width.

Flow solver:

2. Lake at rest: over irregular sections at uneven spacing a level water surface stays
   level and the discharge stays zero (to rounding).
3. Water volume: during a rise of the inflow the change of the stored volume equals the
   inflow minus the outflow, step by step.
4. Dam break on a wet bed (frictionless): depth and discharge behind the bore and the
   position of the bore match the analytical solution of Stoker.
5. Steady flow over a bump with a hydraulic jump (frictionless): the discharge is
   uniform, the flow passes through critical depth at the crest, the depth upstream
   approaches the value of the energy equation with grid refinement, and the jump stands
   where the analytical solution has it.
6. Subcritical flow behind a dam: the steady water surface agrees with the backwater
   solver, and the difference falls in proportion to the spacing.
7. Steep channel into a pool, headwater as a critical-flow control: critical depth at
   the headwater, a discharge equal to the inflow on every reach, supercritical flow that
   approaches the normal depth, a jump at the position that the specific force and the
   backwater curve of the pool give, and that backwater curve behind the jump.
8. Steep channel with a low tailwater: the supercritical flow leaves freely, at the
   normal depth. With the headwater as a critical-flow control and a momentum coefficient
   beta the headwater stands at the critical depth of beta Q^2 B / (g A^3) = 1; with the
   default headwater condition (uniform flow) the depth is normal from the headwater on.
9. Derivatives of a function of the final state with respect to Manning's n, the inflow,
   the outlet level and the bed match central differences, in reverse mode (with and
   without recomputing the flow steps) and in forward mode; also through a hydraulic
   jump, and for a shallow flow with the default (wide) smoothing of the waterline.

Time loop (``hydrograd.oned.stepper`` with ``flow_solver="unsteady"``):

10. A subcritical channel with a pool: the deposit of a run with the unsteady flow solver
    agrees with that of the backwater solver, the sediment volume balance closes, and a
    derivative through the run matches a central difference.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq

from hydrograd.oned import (
    GeometrySmoothing,
    HydraulicsConfig,
    OneDConfig,
    UnsteadyNumerics,
    constant_forcing,
    default_parameters,
    network_from_positions,
    run,
    solve,
)
from hydrograd.oned.geometry import (
    area_width_moment,
    hydraulic_properties,
    water_surface_from_area,
)
from hydrograd.oned.network import control_volume_lengths
from hydrograd.oned.synthetic import trapezoid_channel, uniform_bed
from hydrograd.oned.unsteady import (
    advance,
    courant_number,
    froude_number,
    link_discharge,
    state_from_surface,
)
from hydrograd.validation.dambreak import dam_break

G = 9.81                # m/s^2

# A sharp waterline, for comparisons with exact geometry.
SHARP = GeometrySmoothing(beta_softplus=1000.0, eps_abs=1.0e-6, eps_wet=1.0e-3)
CONFIG = HydraulicsConfig(smoothing=SHARP)
# The same with the headwater as a critical-flow control.
CONFIG_CRITICAL = HydraulicsConfig(
    smoothing=SHARP, unsteady=UnsteadyNumerics(headwater="critical"),
)

# Steep rectangular channel of tests 7 to 9 (SI units).
CHUTE_WIDTH = 0.3               # m
CHUTE_SLOPE = 0.02              # -
CHUTE_MANNING = 0.012           # s/m^(1/3)
CHUTE_DISCHARGE = 0.049         # m^3/s
CHUTE_LENGTH = 30.0             # m


def _rectangle(bed: jnp.ndarray, width: float, height: float = 5.0):
    """Tables (z, s) of rectangular sections with vertical walls, shape [N, 4]."""
    n_sections = bed.shape[0]
    s = jnp.tile(jnp.asarray([0.0, 0.0, width, width]), (n_sections, 1))
    z = bed[:, None] + jnp.asarray([height, 0.0, 0.0, height])[None, :]
    return z, s


def _irregular(n_sections: int, n_points: int = 9, seed: int = 0):
    """Irregular sections at uneven spacing: positions x [N] and tables (z, s) [N, J]."""
    rng = np.random.default_rng(seed)
    x = jnp.asarray(np.cumsum(rng.uniform(50.0, 300.0, n_sections)))
    s = jnp.asarray(np.sort(rng.uniform(0.0, 200.0, (n_sections, n_points)), axis=1))
    z = jnp.asarray(rng.uniform(0.0, 6.0, (n_sections, n_points)))
    z = z.at[:, 0].set(12.0).at[:, -1].set(12.0)
    return x, z, s


def _chute(n_sections: int):
    """The steep channel: network, bed elevation [N], tables and depths (normal, critical)."""
    x = jnp.linspace(0.0, CHUTE_LENGTH, n_sections)
    bed = CHUTE_SLOPE * (CHUTE_LENGTH - x)
    z, s = _rectangle(bed, CHUTE_WIDTH, 2.0)
    q = CHUTE_DISCHARGE / CHUTE_WIDTH
    # Normal depth of the default conveyance K = b h^(5/3) / n, and critical depth.
    normal = (q * CHUTE_MANNING / np.sqrt(CHUTE_SLOPE)) ** 0.6
    critical = (q * q / G) ** (1.0 / 3.0)
    return network_from_positions(x), x, bed, z, s, normal, critical


def _chute_steady(
    n_sections: int, eta_tail: float, beta=None, time: float = 400.0, config=CONFIG_CRITICAL,
):
    """Steady state of the steep channel for a tailwater level (m)."""
    network, _, bed, z, s, _, critical = _chute(n_sections)
    eta_start = jnp.maximum(bed + critical, eta_tail)
    state = state_from_surface(config, eta_start, jnp.full(n_sections, CHUTE_DISCHARGE), z, s)
    dt = 0.4 * (CHUTE_LENGTH / (n_sections - 1)) / 4.0
    n_steps = int(time / dt)
    run = jax.jit(
        lambda st: advance(
            config, network, st, z, s, CHUTE_DISCHARGE, eta_tail, CHUTE_MANNING, dt, n_steps, beta,
        )
    )
    return run(state), dt


def test_first_moment_and_inverse_of_area():
    """Test 1."""
    # Rectangle (b = 4 m) and trapezoid (b = 60 m, side slope 2), depth 1.5 m.
    z_rect, s_rect = _rectangle(jnp.zeros(1), 4.0)
    _, _, moment = area_width_moment(jnp.asarray([1.5]), z_rect, s_rect, SHARP)
    np.testing.assert_allclose(moment, 4.0 * 1.5**2 / 2.0, rtol=1e-9)

    s_trap = jnp.asarray([[-90.0, -30.0, 30.0, 90.0]])
    z_trap = jnp.asarray([[30.0, 0.0, 0.0, 30.0]])
    _, _, moment = area_width_moment(jnp.asarray([1.5]), z_trap, s_trap, SHARP)
    np.testing.assert_allclose(moment, 60.0 * 1.5**2 / 2.0 + 2.0 * 1.5**3 / 3.0, rtol=1e-6)

    # dI_1 / d eta = A on irregular sections (default smoothing).
    _, z, s = _irregular(6)
    smoothing = GeometrySmoothing()
    eta = jnp.asarray([3.1, 4.4, 5.2, 6.3, 7.7, 9.0])

    def total_moment(level):
        return jnp.sum(area_width_moment(level, z, s, smoothing)[2])

    area, width, _ = area_width_moment(eta, z, s, smoothing)
    np.testing.assert_allclose(jax.grad(total_moment)(eta), area, rtol=2e-3)

    # Inverse of the area, from a start 0.4 m off.
    eta_back = water_surface_from_area(area, z, s, eta + 0.4, smoothing, n_iter=8)
    np.testing.assert_allclose(eta_back, eta, atol=1e-10)

    # Derivatives of the inverse against central differences: with respect to the area
    # (section by section) and to a shift of the bed of the third point of every section.
    def surface(target, shift):
        table = z.at[:, 3].add(shift)
        return jnp.sum(water_surface_from_area(target, table, s, eta, smoothing, n_iter=8))

    d_area, d_shift = jax.grad(surface, argnums=(0, 1))(area, jnp.asarray(0.0))
    step = 1.0e-4
    for index in range(6):
        change = jnp.zeros(6).at[index].set(step)
        difference = (surface(area + change, 0.0) - surface(area - change, 0.0)) / (2.0 * step)
        np.testing.assert_allclose(d_area[index], difference, rtol=1e-7)
    np.testing.assert_allclose(d_area, 1.0 / width, rtol=2e-3)
    step = 1.0e-5
    difference = (surface(area, step) - surface(area, -step)) / (2.0 * step)
    np.testing.assert_allclose(d_shift, difference, rtol=1e-6)

    # A flat bottom (0.3 m wide) under 6 cm of water, default smoothing: the slope
    # dA / d eta is 11 % below the top width, and the derivative of the inverse follows
    # the slope.
    s_flume = jnp.asarray([[0.0, 0.0, 0.3, 0.3]])
    z_flume = jnp.asarray([[1.0, 0.0, 0.0, 1.0]])
    level = jnp.asarray([0.06])
    area_flume, width_flume, _ = area_width_moment(level, z_flume, s_flume, smoothing)

    def surface_flume(target):
        return water_surface_from_area(target, z_flume, s_flume, level, smoothing)[0]

    derivative = jax.grad(surface_flume)(area_flume)[0]
    step = 1.0e-7
    difference = (surface_flume(area_flume + step) - surface_flume(area_flume - step)) / (2.0 * step)
    np.testing.assert_allclose(derivative, difference, rtol=1e-6)
    assert float(derivative * width_flume[0]) > 1.1

    # The properties used by the two flow solvers are the same.
    properties = hydraulic_properties(eta, z, s, jnp.ones(6), smoothing)
    np.testing.assert_allclose(properties.area, area, rtol=1e-14)
    np.testing.assert_allclose(properties.top_width, width, rtol=1e-14)


def test_lake_at_rest():
    """Test 2."""
    x, z, s = _irregular(30)
    network = network_from_positions(x)
    eta = jnp.full(30, 8.0)
    state = state_from_surface(CONFIG, eta, jnp.zeros(30), z, s)

    final = advance(CONFIG, network, state, z, s, 0.0, 8.0, 0.03, 2.0, 2000)

    assert float(jnp.max(jnp.abs(final.eta - 8.0))) < 1e-11
    assert float(jnp.max(jnp.abs(link_discharge(final.area, final.velocity)))) < 1e-8


def test_water_volume_is_conserved():
    """Test 3."""
    n_sections = 21
    x, z, s = trapezoid_channel(length=10000.0, n_sections=n_sections, slope=1.0e-3)
    network = network_from_positions(x)
    lengths = control_volume_lengths(network, "upstream_half")
    back = solve(CONFIG, network, z, s, 300.0, 108.0, 0.03)
    state = state_from_surface(CONFIG, back.eta, jnp.full(n_sections, 300.0), z, s)

    # The inflow rises from 300 to 450 m^3/s; one flow step per call.
    dt, Q_in = 15.0, 450.0
    one = jax.jit(lambda st: advance(CONFIG, network, st, z, s, Q_in, 108.0, 0.03, dt, 1))
    volume_start = float(jnp.sum(state.area[:-1] * lengths[:-1]))
    balance = 0.0
    for _ in range(300):
        new = one(state)
        # Outflow of the step: new velocity of the last reach times its old upwind area.
        balance += dt * (Q_in - float(new.velocity[-1] * state.area[-2]))
        state = new
    volume_end = float(jnp.sum(state.area[:-1] * lengths[:-1]))

    assert balance > 1.0e5
    np.testing.assert_allclose(volume_end - volume_start, balance, rtol=1e-10)


def test_dam_break_matches_stoker():
    """Test 4."""
    length, n_sections, depth_left, depth_right, time = 100.0, 401, 1.0, 0.2, 6.0
    x = jnp.linspace(0.0, length, n_sections)
    z, s = _rectangle(jnp.zeros(n_sections), 1.0)
    network = network_from_positions(x)
    state = state_from_surface(
        CONFIG, jnp.where(x < 50.0, depth_left, depth_right), jnp.zeros(n_sections), z, s,
    )
    spacing = length / (n_sections - 1)
    n_steps = int(np.ceil(time / (0.4 * spacing / 6.0)))
    dt = time / n_steps

    final = jax.jit(
        lambda st: advance(CONFIG, network, st, z, s, 0.0, depth_right, 1.0e-6, dt, n_steps)
    )(state)
    assert float(jnp.max(courant_number(CONFIG, network, final, z, s, dt))) < 0.5

    depth = np.asarray(final.eta)
    discharge = np.asarray(link_discharge(final.area, final.velocity))
    depth_exact, velocity_exact = dam_break(np.asarray(x), time, depth_left, depth_right, x0=50.0)
    positions = np.asarray(x)

    # State between the rarefaction and the bore.
    middle = (positions > 60.0) & (positions < 66.0)
    np.testing.assert_allclose(depth[middle].mean(), depth_exact[middle].mean(), rtol=0.01)
    np.testing.assert_allclose(
        discharge[middle[:-1]].mean(), (depth_exact * velocity_exact)[middle].mean(), rtol=0.01,
    )

    # Position of the bore: last point above the mean of the two states.
    level = 0.5 * (depth_exact[middle].mean() + depth_right)
    bore = positions[np.where(depth > level)[0].max()]
    bore_exact = positions[np.where(depth_exact > level)[0].max()]
    assert abs(bore - bore_exact) <= 2.0 * spacing

    # Mean error of the depth, and no overshoot above the reservoir or below the tail.
    assert np.abs(depth - depth_exact).mean() < 0.005
    assert depth.max() <= depth_left + 1e-9
    assert depth.min() >= depth_right - 1e-9


def _bump_steady(n_sections: int):
    """Steady flow over the parabolic bump: positions, depth, Froude number, discharge."""
    discharge, depth_out = 0.18, 0.33
    x = jnp.linspace(0.0, 25.0, n_sections)
    bump = jnp.where((x > 8.0) & (x < 12.0), 0.2 - 0.05 * (x - 10.0) ** 2, 0.0)
    z, s = _rectangle(bump, 1.0)
    network = network_from_positions(x)
    state = state_from_surface(
        CONFIG, jnp.full(n_sections, depth_out), jnp.full(n_sections, discharge), z, s,
    )
    dt = 0.4 * (25.0 / (n_sections - 1)) / 4.0
    n_steps = int(400.0 / dt)
    final = jax.jit(
        lambda st: advance(CONFIG, network, st, z, s, discharge, depth_out, 1.0e-6, dt, n_steps)
    )(state)
    froude = np.asarray(froude_number(CONFIG, final, z, s, discharge))
    flow = np.asarray(link_discharge(final.area, final.velocity))
    return np.asarray(x), np.asarray(final.eta - bump), froude, flow


def test_hydraulic_jump_over_a_bump():
    """Test 5."""
    discharge = 0.18
    critical = (discharge**2 / G) ** (1.0 / 3.0)
    # Depth upstream of the bump from the energy equation, critical flow at the crest.
    head = 0.2 + 1.5 * critical
    upstream_exact = brentq(lambda y: y + discharge**2 / (2.0 * G * y**2) - head, critical, 2.0)
    jump_exact = 11.6657

    errors = []
    for n_sections in (126, 251):
        x, depth, froude, flow = _bump_steady(n_sections)
        spacing = 25.0 / (n_sections - 1)

        # Uniform discharge.
        np.testing.assert_allclose(flow, discharge, rtol=1e-6)

        # Subcritical upstream, critical near the crest (x = 10 m), supercritical on the
        # downstream face, subcritical again behind the jump.
        assert np.all(froude[x < 9.5] < 1.0)
        crest = int(np.argmin(np.abs(x - 10.0)))
        np.testing.assert_allclose(depth[crest], critical, rtol=0.03)
        last_supercritical = x[np.where(froude > 1.0)[0].max()]
        assert abs(last_supercritical - jump_exact) <= 3.0 * spacing
        np.testing.assert_allclose(depth[x > 14.0], 0.33, atol=1e-4)

        errors.append(abs(depth[2] - upstream_exact))

    # The depth upstream is within 1 % of the energy equation and converges (first order).
    assert errors[1] < 0.01 * upstream_exact
    assert errors[1] < 0.6 * errors[0]


def _difference_from_backwater(n_sections: int) -> float:
    """Largest difference (m) of the steady water surface from the backwater solver."""
    discharge, manning, eta_outlet = 300.0, 0.03, 108.0
    x, z, s = trapezoid_channel(length=10000.0, n_sections=n_sections, slope=1.0e-3)
    network = network_from_positions(x)
    back = solve(CONFIG, network, z, s, discharge, eta_outlet, manning)
    state = state_from_surface(CONFIG, back.eta, jnp.full(n_sections, discharge), z, s)
    dt = 0.5 * (10000.0 / (n_sections - 1)) / 14.0
    n_steps = int(40000.0 / dt)

    final = jax.jit(
        lambda st: advance(
            CONFIG, network, st, z, s, discharge, eta_outlet, manning, dt, n_steps,
        )
    )(state)

    np.testing.assert_allclose(
        link_discharge(final.area, final.velocity), discharge, rtol=1e-8,
    )
    assert float(final.eta[-1]) == eta_outlet

    return float(jnp.max(jnp.abs(final.eta - back.eta)))


def test_subcritical_flow_agrees_with_backwater_solver():
    """Test 6."""
    coarse = _difference_from_backwater(21)
    fine = _difference_from_backwater(41)

    # 8 mm at 500 m spacing (depths of 3.5 to 8 m), halved at 250 m.
    assert coarse < 0.01
    assert fine < 0.6 * coarse


def test_steep_channel_with_jump_into_a_pool():
    """Test 7."""
    n_sections, eta_tail = 301, 0.35
    final, dt = _chute_steady(n_sections, eta_tail, time=500.0)
    network, x, bed, z, s, normal, critical = _chute(n_sections)
    depth = np.asarray(final.eta - bed)
    froude = np.asarray(froude_number(CONFIG, final, z, s, CHUTE_DISCHARGE))
    positions = np.asarray(x)
    spacing = CHUTE_LENGTH / (n_sections - 1)

    # Steady: 2000 more steps change nothing.
    later = advance(
        CONFIG_CRITICAL, network, final, z, s, CHUTE_DISCHARGE, eta_tail, CHUTE_MANNING, dt, 2000,
    )
    assert float(jnp.max(jnp.abs(later.eta - final.eta))) < 1e-10

    # The discharge of every reach is the inflow: the headwater control adds no water.
    np.testing.assert_allclose(
        link_discharge(final.area, final.velocity), CHUTE_DISCHARGE, rtol=1e-9,
    )

    # Critical depth at the headwater, then a supercritical flow that approaches the
    # normal depth from above (the foot of the jump is spread over a few sections).
    np.testing.assert_allclose(depth[0], critical, rtol=1e-3)
    jump_index = np.where(froude > 1.0)[0].max()
    supercritical = depth[1 : jump_index - 5]
    assert np.all(np.diff(supercritical) < 0.0)
    assert normal < supercritical[-1] < 1.05 * normal

    # Jump position: the backwater curve of the pool, integrated upstream from the
    # outlet, reaches the depth conjugate to the normal depth.
    q = CHUTE_DISCHARGE / CHUTE_WIDTH
    froude_normal = q / (normal * np.sqrt(G * normal))
    conjugate = 0.5 * normal * (np.sqrt(1.0 + 8.0 * froude_normal**2) - 1.0)

    def slope_of_depth(y):
        friction = (q * CHUTE_MANNING) ** 2 / y ** (10.0 / 3.0)
        return (CHUTE_SLOPE - friction) / (1.0 - q * q / (G * y**3))

    position, y, step = CHUTE_LENGTH, eta_tail, -0.001
    curve_positions, curve_depths = [position], [y]
    while y > conjugate:
        y += step * slope_of_depth(y + 0.5 * step * slope_of_depth(y))
        position += step
        curve_positions.append(position)
        curve_depths.append(y)
    assert abs(positions[jump_index] - position) <= 2.0 * spacing

    # Behind the jump the flow is subcritical and follows the backwater curve of the pool.
    pool = positions > position + 1.0
    assert np.all(froude[pool] < 1.0)
    curve = np.interp(positions[pool], curve_positions[::-1], curve_depths[::-1])
    np.testing.assert_allclose(depth[pool], curve, rtol=0.01)


def test_free_outflow_and_momentum_coefficient():
    """Test 8."""
    n_sections = 151
    _, _, bed, z, s, normal, critical = _chute(n_sections)

    for beta in (None, 1.2):
        # Tailwater far below the outlet bed.
        final, _ = _chute_steady(n_sections, -1.0, beta=beta, time=300.0)
        depth = np.asarray(final.eta - bed)
        froude = np.asarray(froude_number(CONFIG, final, z, s, CHUTE_DISCHARGE, beta))

        # Critical depth at the headwater: beta Q^2 B / (g A^3) = 1.
        factor = 1.0 if beta is None else beta ** (1.0 / 3.0)
        np.testing.assert_allclose(depth[0], factor * critical, rtol=1e-3)
        np.testing.assert_allclose(froude[0], 1.0, rtol=2e-3)

        # Supercritical everywhere below the headwater, at the normal depth at the end;
        # the outlet has the depth of the section above it.
        assert np.all(froude[1:] > 1.0)
        np.testing.assert_allclose(depth[-2], normal, rtol=0.01)
        np.testing.assert_allclose(depth[-1], depth[-2], rtol=1e-9)
        np.testing.assert_allclose(
            link_discharge(final.area, final.velocity), CHUTE_DISCHARGE, rtol=1e-9,
        )

    # Default headwater condition: uniform flow from the headwater on.
    final, _ = _chute_steady(n_sections, -1.0, time=300.0, config=CONFIG)
    np.testing.assert_allclose(np.asarray(final.eta - bed), normal, rtol=1e-3)
    np.testing.assert_allclose(
        link_discharge(final.area, final.velocity), CHUTE_DISCHARGE, rtol=1e-9,
    )


@pytest.mark.parametrize("case", ["pool", "jump", "shallow"])
def test_gradients_match_central_differences(case: str):
    """Test 9."""
    config = CONFIG
    if case == "shallow":
        # Steep channel with free outflow (depth of 8 cm) and the default smoothing of
        # the waterline (2 cm wide), 2000 steps from the steady state.
        config = HydraulicsConfig()
        n_sections = 31
        network, _, _, z, s, _, _ = _chute(n_sections)
        start, dt = _chute_steady(n_sections, -1.0, time=200.0, config=config)
        n_steps = 2000
        point = {"shift": 0.0, "Q": 0.050, "eta": -1.0, "n": 0.0122}
        steps = {"shift": 1e-6, "Q": 1e-7, "n": 1e-7}
        tolerance = 1e-5
    elif case == "pool":
        # Subcritical channel behind a dam, 600 steps from the backwater profile.
        n_sections = 21
        x, z, s = trapezoid_channel(length=10000.0, n_sections=n_sections, slope=1.0e-3)
        network = network_from_positions(x)
        back = solve(CONFIG, network, z, s, 300.0, 108.0, 0.03)
        start = state_from_surface(CONFIG, back.eta, jnp.full(n_sections, 300.0), z, s)
        dt, n_steps = 0.5 * 500.0 / 14.0, 600
        point = {"shift": 0.0, "Q": 330.0, "eta": 108.3, "n": 0.033}
        steps = {"shift": 1e-4, "Q": 1e-2, "eta": 1e-4, "n": 1e-6}
        tolerance = 1e-6
    else:
        # Steep channel with a jump, 3000 steps from the steady state.
        n_sections = 61
        network, _, _, z, s, _, _ = _chute(n_sections)
        start, dt = _chute_steady(n_sections, 0.35, time=300.0, config=CONFIG)
        n_steps = 3000
        point = {"shift": 0.0, "Q": 0.050, "eta": 0.352, "n": 0.0122}
        steps = {"shift": 1e-6, "Q": 1e-7, "eta": 1e-6, "n": 1e-7}
        tolerance = 1e-4

    def objective(parameters, config=config):
        final = advance(
            config, network, start, z + parameters["shift"], s, parameters["Q"],
            parameters["eta"], parameters["n"], dt, n_steps,
        )
        return 1.0e-2 * jnp.sum(final.eta[:-1] ** 2) + jnp.sum(final.velocity)

    point = {name: jnp.asarray(value) for name, value in point.items()}
    reverse = jax.jit(jax.grad(objective))(point)
    value = jax.jit(objective)

    for name, step in steps.items():
        up = dict(point)
        down = dict(point)
        up[name] = point[name] + step
        down[name] = point[name] - step
        difference = (value(up) - value(down)) / (2.0 * step)
        np.testing.assert_allclose(reverse[name], difference, rtol=tolerance, err_msg=name)

    # Reverse mode without recomputing the flow steps, and forward mode.
    stored = HydraulicsConfig(
        smoothing=config.smoothing, unsteady=UnsteadyNumerics(checkpoint_substeps=False),
    )
    reverse_stored = jax.jit(jax.grad(lambda p: objective(p, stored)))(point)
    forward = jax.jit(jax.jacfwd(objective))(point)
    for name in point:
        np.testing.assert_allclose(reverse_stored[name], reverse[name], rtol=1e-10)
        np.testing.assert_allclose(forward[name], reverse[name], rtol=1e-10)


def _pool_case(flow_solver: str, n_steps: int):
    """A 10 km trapezoidal channel with a 12 m deep pool at the dam and three classes."""
    hydraulics = HydraulicsConfig(
        flow_solver=flow_solver, unsteady=UnsteadyNumerics(n_substeps=1800),
    )
    config = OneDConfig(hydraulics=hydraulics)
    x, z, s = trapezoid_channel(length=10000.0, n_sections=21)
    network = network_from_positions(x)
    bed0 = uniform_bed(z, s, n_classes=3)
    theta = default_parameters(3)

    # Constant inflow of 200 m^3/s carrying 0.02, 0.01 and 0.005 m^3/s of the classes;
    # steps of half a day (flow steps of 24 s with the unsteady solver).
    load_in = jnp.broadcast_to(jnp.asarray([0.02, 0.01, 0.005])[:, None], (3, n_steps))
    forcing = constant_forcing(200.0, 112.0, 43200.0, n_steps)._replace(load_in=load_in)

    return config, network, theta, forcing, bed0


def test_time_loop_with_unsteady_flow():
    """Test 10."""
    results = {}
    for flow_solver in ("backwater", "unsteady"):
        config, network, theta, forcing, bed0 = _pool_case(flow_solver, 10)
        results[flow_solver] = jax.jit(
            lambda parameters, c=config, n=network, f=forcing, b=bed0: run(c, n, parameters, f, b)
        )(theta)

    unsteady = results["unsteady"]
    backwater = results["backwater"]

    # Sediment volume balance of the run with the unsteady solver.
    inflow = jnp.sum(unsteady.load_in, axis=(1, 2))
    outflow = jnp.sum(unsteady.load_out, axis=1)
    np.testing.assert_allclose(
        jnp.sum(unsteady.deposition), jnp.sum((inflow - outflow) * 43200.0), rtol=1e-8,
    )

    # Same water surface and the same deposit per section as with the backwater solver.
    np.testing.assert_allclose(unsteady.eta[-1], backwater.eta[-1], atol=0.02)
    deposit_unsteady = np.asarray(jnp.sum(unsteady.deposition, axis=(0, 1)))
    deposit_backwater = np.asarray(jnp.sum(backwater.deposition, axis=(0, 1)))
    scale = np.abs(deposit_backwater).max()
    np.testing.assert_allclose(deposit_unsteady, deposit_backwater, atol=0.05 * scale)

    # Derivative of the deposited volume in the pool with respect to Manning's n.
    config, network, theta, forcing, bed0 = _pool_case("unsteady", 2)

    def pool_deposit(n_manning):
        result = run(config, network, {**theta, "n_manning": n_manning}, forcing, bed0)
        return jnp.sum(result.deposition[:, :, 12:])

    point = jnp.asarray(0.03)
    gradient = jax.jit(jax.grad(pool_deposit))(point)
    value = jax.jit(pool_deposit)
    step = 1.0e-6
    difference = (value(point + step) - value(point - step)) / (2.0 * step)
    np.testing.assert_allclose(gradient, difference, rtol=1e-5)
