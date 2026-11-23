"""Quasi-steady backwater solver on a channel network, with implicit-function gradients.

Equations
---------
For a reach from section i to its downstream section j = ``downstream[i]``, the
standard-step energy equation is

    F_i = H_i - H_j - dx_i (S_f,i + S_f,j) / 2 = 0,
    H = eta + Q^2 / (2 g A^2)            total head (m),
    S_f = Q^2 / K^2                      friction slope (-),

with the water surface eta, the discharge Q, and the area A(eta) and conveyance K(eta) of
the cross-section (:mod:`hydrograd.oned.geometry`; the formula of K is the setting
``HydraulicsConfig.conveyance``). The water surface at the outlet
(section N - 1) is prescribed, and the discharge at every section is the sum of the
headwater inflows that pass it. The N - 1 unknowns are the water surfaces of sections
0 .. N - 2. F_i depends on eta_i and eta_j only, so the Jacobian has two entries per row
(d F_i / d eta_i and d F_i / d eta_j) and the Newton system is solved by back substitution
from the outlet upstream.

Hydraulic controls
------------------
Where the head downstream plus the friction loss is below the minimum specific energy of
a section, the energy equation has no subcritical solution: the section is a control at
critical depth (for example the crest of a delta acting as a weir), as in the steady mode
of SRH-1D. With the "floor" eta_c,i = thalweg_i + y_c,i (critical depth y_c), section i is
a control if

    H_i(eta_c,i) > H_j + dx_i (S_f,i(eta_c,i) + S_f,j) / 2,

and its residual is then F_i = eta_i - eta_c,i. The control set is decided anew in each
Newton iteration, and the water surface is kept at or above the floor.

Gradients
---------
The Newton iterations run on constants (no derivatives pass through them). After
convergence to eta*, one more Newton step is taken with the dependence on the inputs
theta (bed tables, Manning's n, inflows, outlet level) kept in the residual only:

    eta = eta* - J^-1 F(eta*, theta),     J = dF/d eta at eta*, held constant.

Its value is eta* (F vanishes there), and its derivative is -J^-1 dF/d theta, which is
the implicit-function-theorem derivative of the converged solution. For the given control
set the result is exact, including the derivatives through the critical depth. Both
reverse and forward mode work.

Convergence
-----------
The iterations stop at ``newton_max_iter`` whether they have converged or not. With
``conveyance="segments"`` (default) the conveyance rises with the water surface. With
``conveyance="section"`` it can fall where the water spills onto a bench; the energy
equation then has several roots and the iterations can cycle between them, so that the
returned water surface does not satisfy it (104 of 12,417 daily steps of a real
reservoir with benches in its cross-sections, with residuals up to metres).

Units: SI.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from hydrograd.oned.config import SHEAR_RADII, TRANSPORT_STRESSES, HydraulicsConfig
from hydrograd.oned.geometry import (
    critical_depth,
    critical_depth_attached,
    hydraulic_properties,
    stress_reduction,
    thalweg,
)
from hydrograd.oned.network import Network, section_discharge
from hydrograd.sediment.closures import FlowState

__all__ = [
    "HydroState",
    "NewtonState",
    "attach",
    "converge",
    "hydro_state_from_surface",
    "initial_water_surface",
    "sediment_flow_state",
    "solve",
]

# Floor of the bed shear stress inside the square root of the shear velocity (Pa).
SHEAR_STRESS_FLOOR = 1.0e-12

# Guard of the denominator rho u^2 of the friction coefficient (Pa).
GUARD = 1.0e-12


class HydroState(NamedTuple):
    """Hydraulic state at the N cross-sections.

    Attributes:
        eta: water-surface elevation (m), shape [N].
        discharge: Q (m^3/s), shape [N].
        area: flow area A (m^2), shape [N].
        top_width: B (m), shape [N].
        hydraulic_radius: R (m), shape [N].
        depth: mean depth h = A / B (m), shape [N].
        velocity: mean velocity u = Q / A (m/s), shape [N].
        friction_slope: S_f = Q^2 / K^2 (-), shape [N].
        shear_stress: bed shear stress tau_b = rho g R S_f (Pa), shape [N]; R is the
            hydraulic radius or the mean depth (``HydraulicsConfig.shear_radius``).
        shear_velocity: u_* = sqrt(tau_b / rho) (m/s), shape [N].
        friction_coefficient: c_f = tau_b / (rho u^2) (-), shape [N].
    """

    eta: Array
    discharge: Array
    area: Array
    top_width: Array
    hydraulic_radius: Array
    depth: Array
    velocity: Array
    friction_slope: Array
    shear_stress: Array
    shear_velocity: Array
    friction_coefficient: Array


class NewtonState(NamedTuple):
    """Converged quantities of the Newton solve (constants in derivatives).

    Attributes:
        eta_own: converged water surface at sections 0 .. N - 2 (m), shape [N - 1].
        jacobian_own: d F_i / d eta_i, shape [N - 1].
        jacobian_next: d F_i / d eta_j, j the downstream section, shape [N - 1].
        is_control: True where the section is a critical-flow control, shape [N - 1].
        n_iter: number of Newton iterations used (scalar integer).
    """

    eta_own: Array
    jacobian_own: Array
    jacobian_next: Array
    is_control: Array
    n_iter: Array


def sediment_flow_state(hydro: HydroState) -> FlowState:
    """The flow quantities the sediment closures need, taken from a hydraulic state.

    Args:
        hydro: hydraulic state at the N sections.

    Returns:
        Flow state for :mod:`hydrograd.sediment`, with the top width as the tube width.
    """
    return FlowState(
        discharge=hydro.discharge,
        width=hydro.top_width,
        depth=hydro.depth,
        shear_stress=hydro.shear_stress,
        shear_velocity=hydro.shear_velocity,
        friction_coefficient=hydro.friction_coefficient,
    )


def _eta_next(network: Network, eta_own: Array, eta_outlet: Array) -> Array:
    """Water surface at the downstream section of each reach.

    Args:
        network: the network.
        eta_own: water surface at sections 0 .. N - 2 (m), shape [N - 1].
        eta_outlet: water surface at the outlet (m), scalar.

    Returns:
        Water surface at ``downstream[i]`` (m), shape [N - 1].
    """
    eta_all = jnp.concatenate([eta_own, jnp.reshape(eta_outlet, (1,))])
    return eta_all[network.downstream]


def _heads(
    config: HydraulicsConfig,
    network: Network,
    eta_own: Array,
    eta_next: Array,
    Q: Array,
    z: Array,
    s: Array,
    n_manning: Array,
) -> tuple[Array, Array]:
    """Both sides of the energy equation of every reach.

    Args:
        config: hydraulics settings.
        network: the network.
        eta_own: water surface at the upstream section of each reach (m), shape [N - 1].
        eta_next: water surface at the downstream section of each reach (m), shape [N - 1].
        Q: discharge at the sections (m^3/s), shape [N].
        z: bed elevation of the station points (m), shape [N, J].
        s: station of the points (m), shape [N, J].
        n_manning: Manning's n at the sections (s/m^(1/3)), shape [N].

    Returns:
        Tuple ``(head_own, head_required)`` (m), each of shape [N - 1]: the total head
        H_i at the upstream section, and H_j + dx (S_f,i + S_f,j) / 2.
    """
    g = config.constants.g
    smoothing = config.smoothing
    downstream = network.downstream

    # Discharge and section properties at the two ends of each reach.
    Q_own = Q[:-1]
    Q_next = Q[downstream]
    own = hydraulic_properties(
        eta_own, z[:-1], s[:-1], n_manning[:-1], smoothing, config.conveyance,
        config.resistance, config.constants.g,
    )
    nxt = hydraulic_properties(
        eta_next, z[downstream], s[downstream], n_manning[downstream], smoothing,
        config.conveyance, config.resistance, config.constants.g,
    )

    # Friction slopes S_f = Q^2 / K^2.
    slope_own = Q_own * Q_own / (own.conveyance * own.conveyance)
    slope_next = Q_next * Q_next / (nxt.conveyance * nxt.conveyance)

    # Total heads H = eta + Q^2 / (2 g A^2).
    head_own = eta_own + Q_own * Q_own / (2.0 * g * own.area * own.area)
    head_next = eta_next + Q_next * Q_next / (2.0 * g * nxt.area * nxt.area)

    # Head required at the upstream section: downstream head plus the friction loss.
    head_required = head_next + 0.5 * network.reach_length * (slope_own + slope_next)

    return head_own, head_required


def _residual(
    config: HydraulicsConfig,
    network: Network,
    eta_own: Array,
    eta_next: Array,
    Q: Array,
    z: Array,
    s: Array,
    n_manning: Array,
    is_control: Array,
    floor: Array,
) -> Array:
    """Residual F of every reach: energy equation, or eta - floor at control sections.

    Args:
        config, network, eta_own, eta_next, Q, z, s, n_manning: as in :func:`_heads`.
        is_control: True where the upstream section is a control, shape [N - 1].
        floor: critical water surface thalweg + y_c of the upstream sections (m),
            shape [N - 1].

    Returns:
        F (m), shape [N - 1].
    """
    head_own, head_required = _heads(config, network, eta_own, eta_next, Q, z, s, n_manning)
    return jnp.where(is_control, eta_own - floor, head_own - head_required)


def _residual_and_jacobian(
    config: HydraulicsConfig,
    network: Network,
    eta_own: Array,
    eta_outlet: Array,
    Q: Array,
    z: Array,
    s: Array,
    n_manning: Array,
    is_control: Array,
    floor: Array,
) -> tuple[Array, Array, Array]:
    """Residual and the two Jacobian entries of every row.

    F_i depends on eta_i ("own") and on the water surface of its downstream section
    ("next") only. Treating the two as separate arguments, the gradient of sum(F) with
    respect to each gives d F_i / d eta_i and d F_i / d eta_j exactly.

    Args:
        config, network, Q, z, s, n_manning, is_control, floor: as in :func:`_residual`.
        eta_own: water surface at sections 0 .. N - 2 (m), shape [N - 1].
        eta_outlet: water surface at the outlet (m), scalar.

    Returns:
        Tuple ``(F, jacobian_own, jacobian_next)``, each of shape [N - 1].
    """
    eta_next = _eta_next(network, eta_own, eta_outlet)

    def residual_sum(own: Array, nxt: Array) -> tuple[Array, Array]:
        """Sum of the residuals (for the gradient) and the residuals themselves."""
        F = _residual(config, network, own, nxt, Q, z, s, n_manning, is_control, floor)
        return jnp.sum(F), F

    gradients, F = jax.grad(residual_sum, argnums=(0, 1), has_aux=True)(eta_own, eta_next)
    jacobian_own = gradients[0]
    jacobian_next = gradients[1]

    return F, jacobian_own, jacobian_next


def _back_substitute(
    network: Network, jacobian_own: Array, jacobian_next: Array, rhs: Array,
) -> Array:
    """Solve J x = rhs for the two-entry-per-row Jacobian.

    Row i reads ``jacobian_own[i] x_i + jacobian_next[i] x_j = rhs_i`` with
    j = ``downstream[i]`` > i and x = 0 at the outlet, so going through the rows in
    decreasing order every x_j is known when it is needed. For a single channel this is
    the back substitution of an upper bidiagonal matrix.

    Args:
        network: the network.
        jacobian_own: d F_i / d eta_i, shape [N - 1].
        jacobian_next: d F_i / d eta_j, shape [N - 1].
        rhs: right-hand side, shape [N - 1].

    Returns:
        Solution x, shape [N - 1].
    """
    n_rows = rhs.shape[0]
    downstream = jnp.asarray(network.downstream)

    def solve_row(x: Array, row: Array) -> tuple[Array, None]:
        """Solve one row for x[row], given x at its downstream section."""
        value = (rhs[row] - jacobian_next[row] * x[downstream[row]]) / jacobian_own[row]
        return x.at[row].set(value), None

    # x has one extra entry, the outlet, where x = 0.
    x_start = jnp.zeros(n_rows + 1, dtype=rhs.dtype)
    rows = jnp.arange(n_rows - 1, -1, -1)
    x, _ = jax.lax.scan(solve_row, x_start, rows)

    return x[:n_rows]


def _control_set(
    config: HydraulicsConfig,
    network: Network,
    eta_own: Array,
    eta_outlet: Array,
    Q: Array,
    z: Array,
    s: Array,
    n_manning: Array,
    floor: Array,
) -> Array:
    """Sections whose minimum specific energy exceeds the downstream head plus losses.

    Args:
        config, network, eta_own, eta_outlet, Q, z, s, n_manning: as in
            :func:`_residual_and_jacobian`.
        floor: critical water surface of sections 0 .. N - 2 (m), shape [N - 1].

    Returns:
        Boolean array, True at control sections, shape [N - 1].
    """
    eta_next = _eta_next(network, eta_own, eta_outlet)
    head_critical, head_required = _heads(config, network, floor, eta_next, Q, z, s, n_manning)
    return head_critical > head_required


def _default_start(eta_outlet: Array, floor: Array, y_critical: Array) -> Array:
    """Default initial water surface of the Newton iterations.

    The outlet level everywhere, but at least one critical depth above the critical water
    surface ("floor") of each section.

    Args:
        eta_outlet: water surface at the outlet (m), scalar.
        floor: critical water surface of sections 0 .. N - 2 (m), shape [N - 1].
        y_critical: critical depth of all sections (m), shape [N].

    Returns:
        Initial water surface at all sections (m), shape [N].
    """
    level = jnp.broadcast_to(eta_outlet, floor.shape)
    eta_own = jnp.maximum(level, floor + y_critical[:-1])
    return jnp.concatenate([eta_own, jnp.reshape(eta_outlet, (1,))])


def initial_water_surface(
    config: HydraulicsConfig,
    network: Network,
    z: Array,
    s: Array,
    Q_headwaters: Array,
    eta_outlet: Array,
    Q_withdrawal: Array | None = None,
) -> Array:
    """The water surface from which :func:`converge` starts when it is given no guess.

    The time stepper passes it as the guess of the first step, so that the first step
    needs no special case. It is a constant in derivatives.

    Args:
        config: hydraulics settings.
        network: the network (N sections).
        z: bed elevation of the station points (m), shape [N, J].
        s: station of the points (m), shape [N, J].
        Q_headwaters: discharge of each headwater (m^3/s), scalar or shape [H].
        eta_outlet: water surface at the outlet (m), scalar.
        Q_withdrawal: discharge of each lateral withdrawal (m^3/s), shape [W], or None.

    Returns:
        Initial water surface at all sections (m), shape [N].
    """
    z = jax.lax.stop_gradient(z)
    s = jax.lax.stop_gradient(s)
    Q = jax.lax.stop_gradient(section_discharge(network, Q_headwaters, Q_withdrawal))
    eta_outlet = jax.lax.stop_gradient(jnp.asarray(eta_outlet, dtype=z.dtype))

    y_critical = critical_depth(
        Q, z, s, config.smoothing, config.constants.g, config.numerics.critical_depth_iter,
    )
    floor = thalweg(z)[:-1] + y_critical[:-1]

    return _default_start(eta_outlet, floor, y_critical)


def hydro_state_from_surface(
    config: HydraulicsConfig, eta: Array, Q: Array, z: Array, s: Array, n_manning: Array,
) -> HydroState:
    """Hydraulic state for a given water surface and discharge.

    Used for the converged water surface, and by the sediment routing for a shifted bed
    under the same water surface.

    Args:
        config: hydraulics settings.
        eta: water-surface elevation (m), shape [N].
        Q: discharge at the sections (m^3/s), shape [N].
        z: bed elevation of the station points (m), shape [N, J].
        s: station of the points (m), shape [N, J].
        n_manning: Manning's n at the sections (s/m^(1/3)), shape [N].

    Returns:
        The hydraulic state (:class:`HydroState`).
    """
    constants = config.constants

    # Section properties and the derived flow quantities.
    properties = hydraulic_properties(
        eta, z, s, n_manning, config.smoothing, config.conveyance, config.resistance,
        constants.g,
    )
    area = properties.area
    friction_slope = Q * Q / (properties.conveyance * properties.conveyance)
    velocity = Q / area
    depth = area / properties.top_width

    # Bed shear stress tau_b = rho g R S_f, with the hydraulic radius of the section or
    # the mean depth; shear velocity and friction coefficient.
    if config.shear_radius not in SHEAR_RADII:
        raise ValueError(
            f"shear_radius must be one of {SHEAR_RADII}, not '{config.shear_radius}'"
        )
    radius = depth if config.shear_radius == "depth" else properties.hydraulic_radius
    shear_stress = constants.rho * constants.g * radius * friction_slope
    friction_coefficient = shear_stress / (constants.rho * velocity * velocity + GUARD)

    # Shear stress for the sediment: the total, or the part left after the drag on the
    # largest grains in shallow flow, tau_b (C / C_deep)^e with the resistance law C and
    # the law of a deep flow C_deep (:func:`hydrograd.oned.geometry.stress_reduction`).
    if config.transport_stress not in TRANSPORT_STRESSES:
        raise ValueError(
            f"transport_stress must be one of {TRANSPORT_STRESSES}, "
            f"not '{config.transport_stress}'"
        )
    if config.transport_stress == "reduced":
        if config.resistance == "manning":
            raise ValueError(
                'transport_stress = "reduced" needs resistance = "variable_power" or "keulegan"'
            )
        submergence = jnp.maximum(radius, GUARD) / n_manning
        shear_stress = shear_stress * stress_reduction(submergence, config.resistance)
    shear_velocity = jnp.sqrt(jnp.maximum(shear_stress, SHEAR_STRESS_FLOOR) / constants.rho)

    return HydroState(
        eta=eta,
        discharge=Q,
        area=area,
        top_width=properties.top_width,
        hydraulic_radius=properties.hydraulic_radius,
        depth=depth,
        velocity=velocity,
        friction_slope=friction_slope,
        shear_stress=shear_stress,
        shear_velocity=shear_velocity,
        friction_coefficient=friction_coefficient,
    )


def converge(
    config: HydraulicsConfig,
    network: Network,
    z: Array,
    s: Array,
    Q_headwaters: Array,
    eta_outlet: Array,
    n_manning: Array,
    eta_guess: Array | None = None,
    Q_withdrawal: Array | None = None,
) -> NewtonState:
    """Newton iterations to the converged water surface; no derivatives pass through.

    Args:
        config: hydraulics settings.
        network: the network (N sections).
        z: bed elevation of the station points (m), shape [N, J].
        s: station of the points (m), shape [N, J].
        Q_headwaters: discharge of each headwater (m^3/s), scalar or shape [H].
        eta_outlet: water surface at the outlet (m), scalar.
        n_manning: Manning's n (s/m^(1/3)), scalar, shape [1] or shape [N].
        eta_guess: initial water surface at all sections (m), shape [N], e.g. the solution
            of the previous time step; None starts from the outlet level.
        Q_withdrawal: discharge of each lateral withdrawal (m^3/s), shape [W], or None.

    Returns:
        The converged state (:class:`NewtonState`).
    """
    numerics = config.numerics
    n_sections = z.shape[0]

    # All inputs are constants here.
    z = jax.lax.stop_gradient(z)
    s = jax.lax.stop_gradient(s)
    Q = jax.lax.stop_gradient(section_discharge(network, Q_headwaters, Q_withdrawal))
    eta_outlet = jax.lax.stop_gradient(jnp.asarray(eta_outlet, dtype=z.dtype))
    n_manning = jax.lax.stop_gradient(jnp.broadcast_to(n_manning, (n_sections,)))

    # Critical water surface ("floor") of sections 0 .. N - 2.
    y_critical = critical_depth(
        Q, z, s, config.smoothing, config.constants.g, numerics.critical_depth_iter,
    )
    floor = thalweg(z)[:-1] + y_critical[:-1]

    # Initial water surface: the outlet level (at least one critical depth above the
    # floor), or the guess, kept at or above the floor.
    if eta_guess is None:
        eta_guess = _default_start(eta_outlet, floor, y_critical)
    eta_start = jnp.maximum(jax.lax.stop_gradient(eta_guess)[:-1], floor)

    def not_converged(carry: tuple[Array, Array, Array]) -> Array:
        """Continue while the last update exceeds the tolerance and iterations remain."""
        _, n_iter, largest_update = carry
        return (n_iter < numerics.newton_max_iter) & (largest_update >= numerics.newton_tol)

    def newton_step(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        """One Newton iteration with the control set of the current iterate."""
        eta_own, n_iter, _ = carry

        # Control set, residual and Jacobian at the current iterate.
        is_control = _control_set(
            config, network, eta_own, eta_outlet, Q, z, s, n_manning, floor,
        )
        F, jacobian_own, jacobian_next = _residual_and_jacobian(
            config, network, eta_own, eta_outlet, Q, z, s, n_manning, is_control, floor,
        )

        # Newton update, limited in size, and kept at or above the floor.
        update = _back_substitute(network, jacobian_own, jacobian_next, -F)
        update = jnp.clip(update, -numerics.newton_max_step, numerics.newton_max_step)
        eta_new = jnp.maximum(eta_own + update, floor)

        return eta_new, n_iter + 1, jnp.max(jnp.abs(update))

    start = (eta_start, jnp.asarray(0), jnp.asarray(jnp.inf, dtype=z.dtype))
    eta_own, n_iter, _ = jax.lax.while_loop(not_converged, newton_step, start)

    # Control set and Jacobian at the converged water surface, for the gradient step.
    is_control = _control_set(config, network, eta_own, eta_outlet, Q, z, s, n_manning, floor)
    _, jacobian_own, jacobian_next = _residual_and_jacobian(
        config, network, eta_own, eta_outlet, Q, z, s, n_manning, is_control, floor,
    )

    return NewtonState(
        eta_own=eta_own,
        jacobian_own=jacobian_own,
        jacobian_next=jacobian_next,
        is_control=is_control,
        n_iter=n_iter,
    )


def attach(
    config: HydraulicsConfig,
    network: Network,
    newton: NewtonState,
    z: Array,
    s: Array,
    Q_headwaters: Array,
    eta_outlet: Array,
    n_manning: Array,
    Q_withdrawal: Array | None = None,
) -> HydroState:
    """Hydraulic state with derivatives, from the converged Newton state.

    Takes the implicit-function-theorem step eta = eta* - J^-1 F(eta*, theta) (see the
    module docstring) and evaluates the section properties at the result.

    Args:
        config: hydraulics settings.
        network: the network (N sections).
        newton: converged Newton state from :func:`converge`.
        z, s, Q_headwaters, eta_outlet, n_manning: as in :func:`converge`; derivatives
            with respect to z, Q_headwaters, eta_outlet and n_manning are carried.
        Q_withdrawal: discharge of each lateral withdrawal (m^3/s), shape [W], or None.

    Returns:
        The hydraulic state (:class:`HydroState`).
    """
    constants = config.constants
    numerics = config.numerics
    n_sections = z.shape[0]

    Q = section_discharge(network, Q_headwaters, Q_withdrawal)
    eta_outlet = jnp.asarray(eta_outlet, dtype=z.dtype)
    n_manning = jnp.broadcast_to(n_manning, (n_sections,))

    # Critical water surface with its dependence on Q and z.
    y_critical = critical_depth_attached(
        Q,
        z,
        s,
        config.smoothing,
        constants.g,
        numerics.critical_depth_iter,
    )
    floor = thalweg(z)[:-1] + y_critical[:-1]

    # Residual at the converged water surface, as a function of the inputs only.
    eta_star = jax.lax.stop_gradient(newton.eta_own)
    eta_next = _eta_next(network, eta_star, eta_outlet)
    is_control = newton.is_control
    F = _residual(config, network, eta_star, eta_next, Q, z, s, n_manning, is_control, floor)

    # One Newton step with the constant Jacobian: value eta*, derivative -J^-1 dF/d theta.
    jacobian_own = jax.lax.stop_gradient(newton.jacobian_own)
    jacobian_next = jax.lax.stop_gradient(newton.jacobian_next)
    update = _back_substitute(network, jacobian_own, jacobian_next, -F)
    eta = jnp.concatenate([eta_star + update, jnp.reshape(eta_outlet, (1,))])

    return hydro_state_from_surface(config, eta, Q, z, s, n_manning)


def solve(
    config: HydraulicsConfig,
    network: Network,
    z: Array,
    s: Array,
    Q_headwaters: Array,
    eta_outlet: Array,
    n_manning: Array,
    eta_guess: Array | None = None,
    Q_withdrawal: Array | None = None,
) -> HydroState:
    """Converge the backwater profile and return the hydraulic state with derivatives.

    Args:
        config: hydraulics settings.
        network: the network (N sections).
        z: bed elevation of the station points (m), shape [N, J].
        s: station of the points (m), shape [N, J].
        Q_headwaters: discharge of each headwater (m^3/s), scalar or shape [H].
        eta_outlet: water surface at the outlet (m), scalar.
        n_manning: Manning's n (s/m^(1/3)), scalar, shape [1] or shape [N].
        eta_guess: initial water surface (m), shape [N], or None.
        Q_withdrawal: discharge of each lateral withdrawal (m^3/s), shape [W], or None.

    Returns:
        The hydraulic state (:class:`HydroState`). Wrap the call in ``jax.jit`` (with
        ``config`` and ``network`` closed over) when it is called repeatedly.
    """
    newton = converge(
        config, network, z, s, Q_headwaters, eta_outlet, n_manning, eta_guess, Q_withdrawal,
    )
    return attach(
        config, network, newton, z, s, Q_headwaters, eta_outlet, n_manning, Q_withdrawal,
    )
