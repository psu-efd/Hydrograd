"""Unsteady shallow-water (Saint-Venant) flow on cross-sections of arbitrary shape.

Equations
---------
The unknowns are the flow area A(x, t) and the discharge Q(x, t). With the water surface
eta, the width b(x, z) of the section at elevation z and the first moment of the area
about the water surface, I_1 = integral of (eta - z) b dz (:mod:`hydrograd.oned.geometry`):

    dA/dt + dQ/dx = q_L                                               continuity
    dQ/dt + d/dx (beta Q^2 / A) + g A d eta/dx + g A S_f = 0          momentum
    S_f = Q |Q| / K^2                                                 friction slope

with the conveyance K(eta) and the momentum coefficient beta (1 for a uniform velocity
over the section). The pressure and bed terms of the conservative form,
g dI_1/dx - g I_2 - g A S_0 (I_2: pressure of the banks where the section changes along
the channel, S_0: thalweg slope), are equal to g A d eta/dx. This form needs neither the
thalweg slope nor the change of the section shape between two sections, and it vanishes
for a level water surface. Since dI_1 / d eta = A at a fixed section,

    integral of g A d eta = g [ I_1(eta_2) - I_1(eta_1) ]     for a fixed section,

which is how the term is evaluated here: in a prismatic reach it is the difference of the
conservative pressure flux, so that a hydraulic jump conserves the specific force
beta Q^2 / A + g I_1.

For steady flow with constant Q and beta = 1, and away from a jump, the momentum equation
divided by g A is the energy equation of the backwater solver,
d/dx (eta + Q^2 / (2 g A^2)) + S_f = 0. At a jump the two differ: momentum is conserved
and energy is lost.

Discretisation
--------------
Staggered in space (Stelling and Duinmeijer, 2003, Int. J. Numer. Meth. Fluids 43,
1329-1354): the area A_i and the water surface eta_i belong to the cross-sections
(nodes), the velocity u_l to the reaches (links). Link l runs from section i = l to
section j = i + 1 and has the length dx_l; node i has the length L_i = half of its
adjacent reaches. The discharge of a link is its velocity times the area of its upwind
end,

    Q_l = u_l A_i   if u_l >= 0,      Q_l = u_l A_j   otherwise.

Continuity of node i:

    L_i dA_i/dt = Q_(i-1) - Q_i          (+ the inflow Q_in at the headwater, node 0)

Momentum of link l, from the balance d(Abar_l u_l)/dt dx_l = Phi_i - Phi_j - ... of the
link and its continuity dAbar_l/dt dx_l = Qbar_i - Qbar_j:

    Abar_l dx_l du_l/dt = Qbar_i (beta_i u*_i - u_l) - Qbar_j (beta_j u*_j - u_l)
                          - g T_l - g Abar_l dx_l Q_l |Q_l| (1/K_i^2 + 1/K_j^2) / 2

- Phi_i = beta_i Qbar_i u*_i is the momentum flux through node i: Qbar_i is the mean of
  the discharges of the two links at the node, and u*_i the velocity of the link on the
  upwind side. The two links at a node see the same flux, so momentum is conserved. For
  beta = 1 and flow in the positive direction the advection term is the first-order
  upwind difference -Qbar_i (u_l - u_(l-1)).
- T_l = [ I_1,i(eta_j) - I_1,i(eta_i) + I_1,j(eta_j) - I_1,j(eta_i) ] / 2 is the pressure
  term: the mean of the two end sections, each evaluated at the two water surfaces. It
  is zero for a level water surface whatever the sections are, and equal to
  I_1(eta_j) - I_1(eta_i) in a prismatic reach.
- Abar_l is the mean area of the two ends. The friction term uses the mean friction
  slope of the two end sections (as the backwater solver).

At steady state the continuity equation makes the discharge the same on all links (the
sediment routing sees a uniform discharge), and the momentum equation becomes
Q^2 / A_i - Q^2 / A_(i-1) + g T_l + friction = 0: across a hydraulic jump in a prismatic
reach the specific force Q^2 / A + g I_1 is conserved.

The velocity, not the discharge, is the unknown of a link on purpose: with the discharge
as the unknown the mass flux has no upwinding, and the fast wave of a supercritical flow
is amplified (negative numerical diffusion u (u - c) dx / (4 c)).

Time stepping (explicit, first order): the velocities are advanced with the water surface
and the momentum fluxes of the old time level and the friction term taken implicitly
(divided by 1 + dt g |Q| A_upwind (1/K_i^2 + 1/K_j^2) / 2); then the areas with the
discharges of the new velocities and the old upwind areas; then the water surface from
the areas (:func:`hydrograd.oned.geometry.water_surface_from_area`). The step must keep
the Courant number (|u| + c) dt / dx below 1 (:func:`courant_number`), c = sqrt(g A / B).

Boundary conditions
-------------------
- Headwater: the inflow Q_in enters node 0 with the velocity Q_in / A_0. Where the flow
  leaves the headwater supercritically a second condition is needed: the section is then
  a critical-flow control (the entrance of a steep reach). This is imposed as a limit on
  the velocity of the first reach, u_0 <= Q_in / A_c with A_c the area of the headwater
  at its critical depth for Q_in. The headwater then settles at A_c with the discharge
  Q_in; no water is added or removed.
- Outlet (node N - 1): the water surface is prescribed, but not below the critical depth
  of the outlet for the inflow of the step (a free overfall). If the flow arrives
  supercritically and with more specific force than the tailwater has, nothing is imposed:
  the outlet takes the depth above the thalweg of the section upstream of it. Otherwise
  the tailwater is imposed and a hydraulic jump forms inside the channel.

Limits: single channels only (no junctions, no lateral withdrawals); a section must not
fall dry.

Units: SI.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from hydrograd.oned.backwater import HydroState, hydro_state_from_surface
from hydrograd.oned.config import HEADWATER_CONDITIONS, HydraulicsConfig
from hydrograd.oned.geometry import (
    PROPERTY_FLOOR,
    area_width_moment,
    critical_depth_attached,
    hydraulic_properties,
    thalweg,
    water_surface_from_area,
)
from hydrograd.oned.network import Network, control_volume_lengths

__all__ = [
    "UnsteadyState",
    "advance",
    "courant_number",
    "froude_number",
    "hydro_state",
    "link_discharge",
    "momentum_coefficient",
    "state_from_surface",
    "step",
]

# Newton iterations that recover the water surface at the start of :func:`advance`, where
# the bed may have changed since the areas were computed.
START_SURFACE_ITER = 12

# Velocity limit of the first reach where there is none (m/s).
NO_LIMIT = 1.0e30


class UnsteadyState(NamedTuple):
    """State of the unsteady flow solver (N sections, N - 1 reaches).

    Attributes:
        area: flow area A at the sections (m^2), shape [N]. This is the conserved water
            volume per unit length; it is kept when the bed changes.
        velocity: velocity u of the reaches, positive downstream (m/s), shape [N - 1].
        eta: water-surface elevation at the sections (m), shape [N].
    """

    area: Array
    velocity: Array
    eta: Array


class _Bounds(NamedTuple):
    """Boundary quantities that stay constant during a call of :func:`advance`.

    Attributes:
        Q_in: inflow at the headwater (m^3/s), scalar.
        velocity_cap_head: largest velocity of the first reach, Q_in / A_c with the area
            A_c of the headwater at its critical depth for Q_in (m/s), scalar; no limit
            (a large number) without inflow.
        eta_tail: water surface imposed at the outlet: the prescribed level, or the
            critical level of the outlet for Q_in if that is higher (m), scalar.
        area_tail: flow area of the outlet at ``eta_tail`` (m^2), scalar.
        moment_tail: first moment I_1 of the outlet at ``eta_tail`` (m^3), scalar.
        thalweg: lowest bed elevation of the sections (m), shape [N].
        lengths: length of the control volume of each node (m), shape [N].
    """

    Q_in: Array
    velocity_cap_head: Array
    eta_tail: Array
    area_tail: Array
    moment_tail: Array
    thalweg: Array
    lengths: Array


def _require_chain(network: Network) -> None:
    """Raise if the network is not a single channel without withdrawals."""
    if not network.is_chain:
        raise NotImplementedError(
            "the unsteady flow solver handles single channels only (no junctions)"
        )
    if network.withdrawals is not None:
        raise NotImplementedError(
            "the unsteady flow solver does not handle lateral withdrawals"
        )


def momentum_coefficient(theta, n_sections: int, dtype=None) -> Array:
    """Momentum coefficient beta of each section from the parameter dictionary.

    Args:
        theta: parameters; the optional entry ``"beta_momentum"`` is a scalar or has one
            value per section.
        n_sections: number of cross-sections N.
        dtype: data type of the result.

    Returns:
        beta (-), shape [N]; ones if ``theta`` has no entry ``"beta_momentum"``.
    """
    if theta is None or "beta_momentum" not in theta:
        return jnp.ones(n_sections, dtype=dtype)
    beta = jnp.asarray(theta["beta_momentum"], dtype=dtype)
    return jnp.broadcast_to(jnp.reshape(beta, (-1,)), (n_sections,))


def state_from_surface(
    config: HydraulicsConfig, eta: Array, Q_sections: Array, z: Array, s: Array,
) -> UnsteadyState:
    """Solver state for a given water surface and discharge at the sections.

    Args:
        config: hydraulics settings.
        eta: water-surface elevation (m), shape [N].
        Q_sections: discharge at the sections (m^3/s), shape [N]; each reach gets the
            discharge of its upstream section.
        z: bed elevation of the station points (m), shape [N, J].
        s: station of the points (m), shape [N, J].

    Returns:
        The state (:class:`UnsteadyState`).
    """
    area, _, _ = area_width_moment(eta, z, s, config.smoothing)
    discharge = Q_sections[:-1]
    area_upwind = jnp.where(discharge >= 0.0, area[:-1], area[1:])
    return UnsteadyState(area=area, velocity=discharge / area_upwind, eta=eta)


def link_discharge(area: Array, velocity: Array) -> Array:
    """Discharge of each reach: its velocity times the area of its upwind end.

    Args:
        area: flow area at the sections (m^2), shape [N].
        velocity: velocity of the reaches (m/s), shape [N - 1].

    Returns:
        Discharge of the reaches (m^3/s), shape [N - 1].
    """
    return velocity * jnp.where(velocity >= 0.0, area[:-1], area[1:])


def _node_discharges(Q_in: Array, discharge: Array) -> tuple[Array, Array]:
    """Discharge that enters and that leaves each node of a single channel.

    Args:
        Q_in: inflow at the headwater (m^3/s), scalar.
        discharge: discharge of the reaches (m^3/s), shape [N - 1].

    Returns:
        Tuple ``(entering, leaving)`` (m^3/s), each of shape [N]. The outlet has no
        outgoing reach; its leaving discharge is set to the entering one.
    """
    entering = jnp.concatenate([jnp.reshape(Q_in, (1,)), discharge])
    leaving = jnp.concatenate([discharge, discharge[-1:]])
    return entering, leaving


def _bounds(
    config: HydraulicsConfig,
    network: Network,
    z: Array,
    s: Array,
    Q_headwaters: Array,
    eta_outlet: Array,
    beta: Array,
) -> _Bounds:
    """Boundary quantities of a call of :func:`advance` (see :class:`_Bounds`).

    Args:
        config: hydraulics settings.
        network: the network (a single channel).
        z: bed elevation of the station points (m), shape [N, J].
        s: station of the points (m), shape [N, J].
        Q_headwaters: inflow at the headwater (m^3/s), scalar or shape [1].
        eta_outlet: prescribed water surface at the outlet (m), scalar.
        beta: momentum coefficient of the sections (-), shape [N].

    Returns:
        The boundary quantities.
    """
    smoothing = config.smoothing
    Q_in = jnp.reshape(jnp.asarray(Q_headwaters, dtype=z.dtype), (-1,))[0]
    bottom = thalweg(z)

    # Critical depth of the headwater and of the outlet for the inflow of the step. With
    # a momentum coefficient the critical condition is beta Q^2 B / (g A^3) = 1.
    ends = jnp.asarray([0, z.shape[0] - 1])
    z_ends = z[ends]
    s_ends = s[ends]
    Q_critical = jnp.sqrt(beta[ends]) * jnp.abs(Q_in)
    y_critical = critical_depth_attached(
        Q_critical, z_ends, s_ends, smoothing, config.constants.g,
        config.numerics.critical_depth_iter,
    )
    eta_critical = bottom[ends] + y_critical

    eta_tail = jnp.maximum(jnp.asarray(eta_outlet, dtype=z.dtype), eta_critical[1])
    levels = jnp.stack([eta_critical[0], eta_tail])
    area_ends, _, moment_ends = area_width_moment(levels, z_ends, s_ends, smoothing)

    # Critical velocity of the headwater if it is a critical-flow control; no limit
    # otherwise, and none without inflow.
    if config.unsteady.headwater not in HEADWATER_CONDITIONS:
        raise ValueError(
            f"headwater must be one of {HEADWATER_CONDITIONS}, not '{config.unsteady.headwater}'"
        )
    if config.unsteady.headwater == "critical":
        velocity_cap = jnp.where(Q_in > 0.0, Q_in / area_ends[0], NO_LIMIT)
    else:
        velocity_cap = jnp.asarray(NO_LIMIT, dtype=z.dtype)

    return _Bounds(
        Q_in=Q_in,
        velocity_cap_head=velocity_cap,
        eta_tail=eta_tail,
        area_tail=area_ends[1],
        moment_tail=moment_ends[1],
        thalweg=bottom,
        lengths=control_volume_lengths(network, "upstream_half"),
    )


def step(
    config: HydraulicsConfig,
    network: Network,
    state: UnsteadyState,
    z: Array,
    s: Array,
    n_manning: Array,
    beta: Array,
    bounds: _Bounds,
    dt: Array,
) -> UnsteadyState:
    """One explicit time step of the flow (see the module docstring).

    Args:
        config: hydraulics settings.
        network: the network (a single channel of N sections).
        state: state at the start of the step.
        z: bed elevation of the station points (m), shape [N, J].
        s: station of the points (m), shape [N, J].
        n_manning: Manning's n at the sections (s/m^(1/3)), shape [N].
        beta: momentum coefficient of the sections (-), shape [N].
        bounds: boundary quantities of the step (:func:`_bounds`).
        dt: step length (s), scalar.

    Returns:
        The state at the end of the step.
    """
    g = config.constants.g
    smoothing = config.smoothing
    dx = network.reach_length
    n_sections = z.shape[0]

    area = state.area
    velocity = state.velocity
    eta = state.eta
    discharge = link_discharge(area, velocity)

    # Top width and conveyance of the sections at their own water surface.
    properties = hydraulic_properties(
        eta, z, s, n_manning, smoothing, config.conveyance, config.resistance,
        config.constants.g,
    )
    conveyance = properties.conveyance
    top_width = properties.top_width

    # First moment I_1 of every section at its own water surface, and of the two end
    # sections of every reach at the water surface of the other end.
    eta_stacked = jnp.concatenate([eta, eta[1:], eta[:-1]])
    z_stacked = jnp.concatenate([z, z[:-1], z[1:]])
    s_stacked = jnp.concatenate([s, s[:-1], s[1:]])
    _, _, moment = area_width_moment(eta_stacked, z_stacked, s_stacked, smoothing)
    moment_own = moment[:n_sections]
    moment_up_at_down = moment[n_sections : 2 * n_sections - 1]
    moment_down_at_up = moment[2 * n_sections - 1 :]

    # Pressure term T_l of each reach.
    thrust = 0.5 * (
        (moment_up_at_down - moment_own[:-1]) + (moment_own[1:] - moment_down_at_up)
    )

    # Discharge through each node (mean of its two reaches) and the velocity it carries
    # (that of the upwind side). The inflow enters the headwater with the velocity of the
    # headwater section; a backflow enters the outlet with that of the outlet section.
    entering, leaving = _node_discharges(bounds.Q_in, discharge)
    through = 0.5 * (entering + leaving)
    velocity_above = jnp.concatenate([jnp.reshape(bounds.Q_in / area[0], (1,)), velocity])
    velocity_below = jnp.concatenate([velocity, jnp.reshape(discharge[-1] / area[-1], (1,))])
    carried = beta * jnp.where(through >= 0.0, velocity_above, velocity_below)

    # Advection of each reach: Qbar_i (beta_i u*_i - u_l) - Qbar_j (beta_j u*_j - u_l).
    advection = through[:-1] * (carried[:-1] - velocity) - through[1:] * (carried[1:] - velocity)

    # Friction: mean friction slope of the two ends, taken implicitly in the velocity.
    area_mean = 0.5 * (area[:-1] + area[1:])
    area_upwind = jnp.where(velocity >= 0.0, area[:-1], area[1:])
    resistance = 0.5 * (
        1.0 / (conveyance[:-1] * conveyance[:-1]) + 1.0 / (conveyance[1:] * conveyance[1:])
    )
    friction_factor = 1.0 + dt * g * jnp.abs(discharge) * area_upwind * resistance

    # Momentum: new velocity of each reach.
    explicit = velocity + dt * (advection - g * thrust) / (area_mean * dx)
    velocity_new = explicit / friction_factor

    # Headwater as a critical-flow control: the first reach is not faster than the
    # critical velocity of the headwater.
    velocity_new = velocity_new.at[0].set(
        jnp.minimum(velocity_new[0], bounds.velocity_cap_head)
    )

    # Continuity: new area of each node, with the new velocities and the old upwind
    # areas. The area of the outlet follows from its boundary condition below.
    discharge_new = link_discharge(area, velocity_new)
    entering_new, leaving_new = _node_discharges(bounds.Q_in, discharge_new)
    area_new = area + dt * (entering_new - leaving_new) / bounds.lengths
    area_new = jnp.maximum(area_new, PROPERTY_FLOOR)

    # Water surface of the new areas, starting from the old water surface.
    eta_new = water_surface_from_area(
        area_new, z, s, eta, smoothing, config.unsteady.surface_iter,
    )

    # Outlet. The tailwater is imposed unless the flow arrives supercritically with more
    # specific force than the tailwater has (old time level).
    Q_last = discharge[-1]
    celerity_above = jnp.sqrt(g * area[-2] / top_width[-2])
    froude_above = jnp.sqrt(beta[-2]) * jnp.abs(Q_last) / (area[-2] * celerity_above)
    force_above = beta[-2] * Q_last * Q_last / area[-2] + g * moment_own[-2]
    force_tail = beta[-1] * Q_last * Q_last / bounds.area_tail + g * bounds.moment_tail
    tail_controls = (froude_above < 1.0) | (force_tail >= force_above) | (Q_last <= 0.0)

    eta_free = bounds.thalweg[-1] + (eta_new[-2] - bounds.thalweg[-2])
    area_free, _, _ = area_width_moment(
        jnp.reshape(eta_free, (1,)), z[-1:], s[-1:], smoothing,
    )
    eta_outlet = jnp.where(tail_controls, bounds.eta_tail, eta_free)
    area_outlet = jnp.where(tail_controls, bounds.area_tail, area_free[0])

    eta_new = eta_new.at[-1].set(eta_outlet)
    area_new = area_new.at[-1].set(area_outlet)

    return UnsteadyState(area=area_new, velocity=velocity_new, eta=eta_new)


def advance(
    config: HydraulicsConfig,
    network: Network,
    state: UnsteadyState,
    z: Array,
    s: Array,
    Q_headwaters: Array,
    eta_outlet: Array,
    n_manning: Array,
    dt: Array,
    n_steps: int,
    beta: Array | None = None,
) -> UnsteadyState:
    """Advance the flow by ``n_steps`` time steps with constant boundary conditions.

    The water volume of the state is kept: if the bed ``z`` differs from the bed the
    state was computed for (after a deposit or an erosion), the water surface is first
    recovered from the areas on the new bed.

    Args:
        config: hydraulics settings.
        network: the network (a single channel of N sections).
        state: state at the start.
        z: bed elevation of the station points (m), shape [N, J].
        s: station of the points (m), shape [N, J].
        Q_headwaters: inflow at the headwater (m^3/s), scalar or shape [1].
        eta_outlet: prescribed water surface at the outlet (m), scalar.
        n_manning: Manning's n (s/m^(1/3)), scalar, shape [1] or shape [N].
        dt: length of one time step (s), scalar.
        n_steps: number of time steps.
        beta: momentum coefficient (-), scalar or shape [N]; None for 1.

    Returns:
        The state after the steps.
    """
    _require_chain(network)
    n_sections = z.shape[0]
    n_manning = jnp.broadcast_to(jnp.reshape(n_manning, (-1,)), (n_sections,))
    if beta is None:
        beta = jnp.ones(n_sections, dtype=z.dtype)
    beta = jnp.broadcast_to(jnp.reshape(jnp.asarray(beta, dtype=z.dtype), (-1,)), (n_sections,))
    dt = jnp.asarray(dt, dtype=z.dtype)

    def run_steps(state, z, s, Q_headwaters, eta_outlet, n_manning, beta, dt):
        """The time steps as a function of arrays only (so that it can be checkpointed)."""
        bounds = _bounds(config, network, z, s, Q_headwaters, eta_outlet, beta)

        # Water surface of the stored areas on the current bed.
        eta = water_surface_from_area(
            state.area, z, s, state.eta, config.smoothing, START_SURFACE_ITER,
        )
        start = UnsteadyState(area=state.area, velocity=state.velocity, eta=eta)

        def one_step(carry: UnsteadyState, _):
            """One flow step."""
            return step(config, network, carry, z, s, n_manning, beta, bounds, dt), None

        final, _ = jax.lax.scan(one_step, start, None, length=n_steps)
        return final

    if config.unsteady.checkpoint_substeps:
        run_steps = jax.checkpoint(run_steps)

    return run_steps(state, z, s, Q_headwaters, eta_outlet, n_manning, beta, dt)


def hydro_state(
    config: HydraulicsConfig,
    state: UnsteadyState,
    z: Array,
    s: Array,
    n_manning: Array,
    Q_headwaters: Array,
) -> HydroState:
    """Hydraulic state at the sections, as the sediment routing needs it.

    The discharge of a section is the mean of the discharge that enters it and the
    discharge that leaves it; at steady state this is the inflow at every section.

    Args:
        config: hydraulics settings.
        state: state of the flow solver.
        z: bed elevation of the station points (m), shape [N, J].
        s: station of the points (m), shape [N, J].
        n_manning: Manning's n (s/m^(1/3)), scalar, shape [1] or shape [N].
        Q_headwaters: inflow at the headwater (m^3/s), scalar or shape [1].

    Returns:
        The hydraulic state (:class:`hydrograd.oned.backwater.HydroState`).
    """
    n_sections = z.shape[0]
    n_manning = jnp.broadcast_to(jnp.reshape(n_manning, (-1,)), (n_sections,))
    Q_in = jnp.reshape(jnp.asarray(Q_headwaters, dtype=z.dtype), (-1,))[0]
    entering, leaving = _node_discharges(Q_in, link_discharge(state.area, state.velocity))

    return hydro_state_from_surface(
        config, state.eta, 0.5 * (entering + leaving), z, s, n_manning,
    )


def froude_number(
    config: HydraulicsConfig,
    state: UnsteadyState,
    z: Array,
    s: Array,
    Q_headwaters: Array,
    beta: Array | None = None,
) -> Array:
    """Froude number sqrt(beta) |Q| / (A sqrt(g A / B)) at the sections.

    Args:
        config: hydraulics settings.
        state: state of the flow solver.
        z: bed elevation of the station points (m), shape [N, J].
        s: station of the points (m), shape [N, J].
        Q_headwaters: inflow at the headwater (m^3/s), scalar or shape [1].
        beta: momentum coefficient (-), scalar or shape [N]; None for 1.

    Returns:
        Froude number (-), shape [N].
    """
    _, top_width, _ = area_width_moment(state.eta, z, s, config.smoothing)
    Q_in = jnp.reshape(jnp.asarray(Q_headwaters, dtype=z.dtype), (-1,))[0]
    entering, leaving = _node_discharges(Q_in, link_discharge(state.area, state.velocity))
    Q_sections = 0.5 * (entering + leaving)
    celerity = jnp.sqrt(config.constants.g * state.area / top_width)
    factor = 1.0 if beta is None else jnp.sqrt(jnp.asarray(beta))

    return factor * jnp.abs(Q_sections) / (state.area * celerity)


def courant_number(
    config: HydraulicsConfig,
    network: Network,
    state: UnsteadyState,
    z: Array,
    s: Array,
    dt: Array,
) -> Array:
    """Courant number (|u| + c) dt / dx of each reach for a time step ``dt``.

    Args:
        config: hydraulics settings.
        network: the network.
        state: state of the flow solver.
        z: bed elevation of the station points (m), shape [N, J].
        s: station of the points (m), shape [N, J].
        dt: time step (s), scalar.

    Returns:
        Courant number (-), shape [N - 1], with the larger celerity of the two ends of
        the reach. The explicit scheme needs values below 1.
    """
    _, top_width, _ = area_width_moment(state.eta, z, s, config.smoothing)
    celerity = jnp.sqrt(config.constants.g * state.area / top_width)
    speed = jnp.abs(state.velocity) + jnp.maximum(celerity[:-1], celerity[1:])

    return speed * dt / network.reach_length
