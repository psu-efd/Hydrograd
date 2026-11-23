"""Time loop of the 1-D mode: hydraulics, closures, routing and bed update per step.

One morphological step
----------------------
1. Hydraulics, by the flow solver chosen in ``HydraulicsConfig.flow_solver``:

   - ``"backwater"`` (every ``hydro_every`` steps): converge the backwater profile for
     the current bed and boundary conditions, starting from the previous water surface,
     and attach the derivatives by the implicit function theorem
     (:mod:`hydrograd.oned.backwater`).
   - ``"unsteady"`` (every step): advance the shallow-water equations over the step in
     ``UnsteadyNumerics.n_substeps`` flow steps on the current bed, with the boundary
     conditions of the step (:mod:`hydrograd.oned.unsteady`). The water volume of every
     section is kept when the bed changes. The run starts from the backwater profile of
     the first step.
2. Sediment and bed, in ``n_substeps`` sub-steps with the hydraulics frozen:
   closures -> supply limit and Exner factor -> routing -> bed update
   (:mod:`hydrograd.sediment.closures`, :mod:`hydrograd.oned.routing`,
   :mod:`hydrograd.oned.bed`).

The step is a pure function of the state it carries (bed tables, bed composition,
hydraulic state, load of the previous step), the boundary conditions of the step and the
parameters, so the loop is a ``jax.lax.scan``.

Derivatives and memory
----------------------
Reverse-mode differentiation of a scan stores the intermediate values of every step. With
``checkpoint="auto"`` (default) the T steps are run in segments of about sqrt(T) steps
wrapped in ``jax.checkpoint``: the forward pass stores only the state at the start of each
segment, and the backward pass recomputes one segment at a time. Memory then grows like
sqrt(T) instead of T, at the cost of one extra forward pass. The results do not depend on
the setting.

The Newton iterations of the backwater solver run on constants, so they are part of the
scan without being differentiated. The flow steps of the unsteady solver are differentiated
through; by default they are recomputed in the backward pass instead of stored
(``UnsteadyNumerics.checkpoint_substeps``).

Units: SI.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hydrograd.oned.backwater import (
    HydroState,
    attach,
    converge,
    initial_water_surface,
    sediment_flow_state,
    solve,
)
from hydrograd.oned.bed import (
    BedState,
    capacity_bed_sensitivity,
    exner_factor,
    supply_cap_per_length,
    update_bed,
)
from hydrograd.oned.config import FLOW_SOLVERS, OneDConfig, resistance_parameter
from hydrograd.oned.forcing import Forcing, rating_curve_load
from hydrograd.oned.geometry import thalweg
from hydrograd.oned.network import (
    Network,
    as_network,
    control_volume_lengths,
    passing_fraction,
    water_fraction,
)
from hydrograd.oned.routing import route
from hydrograd.oned.unsteady import (
    UnsteadyState,
    advance,
    momentum_coefficient,
    state_from_surface,
)
from hydrograd.oned.unsteady import hydro_state as unsteady_hydro_state
from hydrograd.sediment.closures import compute_closures

__all__ = ["RunResult", "run", "step"]

# Differentiable parameters (see :func:`hydrograd.oned.config.default_parameters`).
Theta = dict[str, Any]

# Guard of the inflow load in the trap efficiency (m^3/s).
LOAD_GUARD = 1.0e-30


class RunResult(NamedTuple):
    """Result of a 1-D run of T steps (T_rec recorded steps, S snapshots).

    Attributes:
        bed: bed at the end of the run.
        deposition: solids volume deposited per step, class and section (m^3), positive
            for deposition, shape [T, K, N].
        time_end: time at the end of each step (s), shape [T].
        thalweg: lowest bed elevation of each section (m), shape [T_rec, N].
        eta: water-surface elevation (m), shape [T_rec, N].
        load_out: load leaving through the outlet (m^3/s of solids), shape [T_rec, K].
        load_in: inflow load at the headwaters (m^3/s of solids), shape [T_rec, K, H].
        trap_efficiency: 1 - (load out) / (load in), summed over classes, shape [T_rec];
            the load out includes the load of the lateral withdrawals.
        z_snapshots: bed tables after the steps ``snapshot_steps`` (m), shape [S, N, J],
            or None.
        load_withdrawn: load leaving through the lateral withdrawals (m^3/s of solids),
            shape [T_rec, K, W], or None for a network without withdrawals.
    """

    bed: BedState
    deposition: Array
    time_end: Array
    thalweg: Array
    eta: Array
    load_out: Array
    load_in: Array
    trap_efficiency: Array
    z_snapshots: Array | None = None
    load_withdrawn: Array | None = None


class _Carry(NamedTuple):
    """State carried from one step to the next.

    Attributes:
        z: bed elevation of the station points (m), shape [N, J].
        p_active: active-layer fractions (-), shape [K, N].
        p_substrate: substrate fractions (-), shape [K, N].
        substrate_thickness: substrate thickness (m), shape [N].
        hydro: hydraulic state of the last hydraulic solve.
        load: load at the sections in the previous step (m^3/s), shape [K, N].
        z_snapshots: bed tables stored so far (m), shape [S, N, J].
        flow: state of the unsteady flow solver (zeros with the backwater solver).
    """

    z: Array
    p_active: Array
    p_substrate: Array
    substrate_thickness: Array
    hydro: HydroState
    load: Array
    z_snapshots: Array
    flow: UnsteadyState


class _StepInput(NamedTuple):
    """Boundary conditions and bookkeeping of one step.

    Attributes:
        Q: discharge of each headwater (m^3/s), shape [H].
        eta_outlet: water surface at the outlet (m), scalar.
        dt: step length (s), scalar.
        load_in: inflow load of each headwater (m^3/s), shape [K, H]; a dummy [K, H] of
            zeros when the rating curve is used.
        solve_hydro: True if the hydraulics are solved in this step, scalar.
        index: step number, scalar integer.
        Q_withdrawal: discharge of each lateral withdrawal (m^3/s), shape [W] (W = 0
            without withdrawals).
    """

    Q: Array
    eta_outlet: Array
    dt: Array
    load_in: Array
    solve_hydro: Array
    index: Array
    Q_withdrawal: Array


def step(
    config: OneDConfig,
    network: Network,
    cv_lengths: Array,
    theta: Theta,
    bed: BedState,
    hydro: HydroState,
    load_previous: Array,
    load_in: Array,
    dt: Array,
    Q_withdrawal: Array | None = None,
) -> tuple[BedState, Array, Array]:
    """Sediment and bed part of one morphological step, for a given hydraulic state.

    Args:
        config: settings of the 1-D mode.
        network: the network (N sections, H headwaters).
        cv_lengths: control-volume length of each section (m), shape [N].
        theta: parameters.
        bed: bed at the start of the step.
        hydro: hydraulic state of the step.
        load_previous: load at the sections in the previous step (m^3/s), shape [K, N].
        load_in: inflow load of each headwater (m^3/s of solids), shape [K, H].
        dt: step length (s), scalar.
        Q_withdrawal: discharge of each lateral withdrawal (m^3/s), shape [W], or None.

    Returns:
        Tuple ``(bed, deposition, load)``: the bed after the step, the solids volume
        deposited at each section (m^3), shape [K, N], and the load at the sections
        (m^3/s), shape [K, N].
    """
    morph = config.morph
    n_substeps = max(1, int(morph.n_substeps))
    dt_sub = dt / n_substeps
    flow = sediment_flow_state(hydro)

    # At a junction the capacity and supply are shared by the incoming reaches in
    # proportion to their discharge.
    # Below a lateral withdrawal only part of the load (and of the water) enters the reach.
    passing = None
    reach_discharge = hydro.discharge[:-1]
    if network.withdrawals is not None and Q_withdrawal is not None:
        passing = passing_fraction(network, hydro.discharge, Q_withdrawal)
        reach_discharge = reach_discharge * water_fraction(network, hydro.discharge, Q_withdrawal)
    if network.is_chain:
        share = None
    else:
        share = reach_discharge / hydro.discharge[network.downstream]

    deposition_total = jnp.zeros_like(bed.p_active)
    load = load_previous

    for _ in range(n_substeps):
        # Closures for the current bed composition, and the capacity sensitivity.
        closures = compute_closures(flow, bed.p_active, theta, config.sediment)
        sensitivity = capacity_bed_sensitivity(config, hydro, bed, theta, closures.capacity)

        # Supply limit of the bed and Exner factor for this sub-step.
        supply_cap = supply_cap_per_length(config, bed, hydro.eta, theta, dt_sub)
        exner = exner_factor(config, bed, hydro.eta, theta, dt_sub, cv_lengths)

        # Route the load and update the bed.
        sediment = route(
            config,
            network,
            closures,
            load_in,
            supply_cap,
            load,
            sensitivity,
            exner,
            bed.p_active,
            theta["delta_a"],
            share,
            passing,
        )
        update = update_bed(
            config, network, cv_lengths, bed, hydro.eta, sediment.deposition_reach, theta, dt_sub,
            sediment.load,
        )

        bed = update.bed
        load = sediment.load
        deposition_total = deposition_total + update.deposition

    return bed, deposition_total, load


def _segment_length(checkpoint: int | str, n_steps: int) -> int | None:
    """Number of steps per checkpointed segment, or None for no checkpointing.

    Args:
        checkpoint: ``"auto"``, ``"none"`` or a segment length.
        n_steps: number of steps T.

    Returns:
        Segment length, or None.
    """
    if checkpoint in ("none", None, False):
        return None
    if checkpoint == "auto":
        length = math.ceil(math.sqrt(n_steps))
    else:
        length = int(checkpoint)
    if length <= 0 or length >= n_steps:
        return None
    return length


def _scan_segments(step_function, theta: Theta, carry, inputs, n_steps: int, length: int | None):
    """Scan ``step_function`` over the steps, in checkpointed segments if ``length`` is set.

    Args:
        step_function: ``(theta, carry, step_input) -> (carry, output)``.
        theta: parameters, passed to every step.
        carry: initial carry.
        inputs: step inputs with a leading axis of length T.
        n_steps: number of steps T.
        length: steps per segment, or None for one plain scan.

    Returns:
        Tuple ``(carry, outputs)`` as ``jax.lax.scan`` would return for all T steps.
    """

    def scan_steps(parameters, start, segment_inputs):
        """Plain scan over a block of steps."""

        def body(state, step_input):
            """One step with the parameters closed over."""
            return step_function(parameters, state, step_input)

        return jax.lax.scan(body, start, segment_inputs)

    if length is None:
        return scan_steps(theta, carry, inputs)

    # Each segment is recomputed in the backward pass instead of being stored.
    scan_segment = jax.checkpoint(scan_steps)

    # Full segments: reshape the inputs to [n_segments, length, ...] and scan over them.
    n_segments = n_steps // length
    n_main = n_segments * length

    def to_segments(array):
        """Leading axis [T] -> [n_segments, length]."""
        return jnp.reshape(array[:n_main], (n_segments, length) + array.shape[1:])

    def from_segments(array):
        """Leading axes [n_segments, length] -> [n_main]."""
        return jnp.reshape(array, (n_main,) + array.shape[2:])

    def segment_body(state, segment_inputs):
        """One checkpointed segment."""
        return scan_segment(theta, state, segment_inputs)

    main_inputs = jax.tree_util.tree_map(to_segments, inputs)
    carry, outputs = jax.lax.scan(segment_body, carry, main_inputs)
    outputs = jax.tree_util.tree_map(from_segments, outputs)

    # Remaining steps: one more (shorter) segment.
    if n_main < n_steps:

        def tail(array):
            """The steps after the last full segment."""
            return array[n_main:]

        def join(first, second):
            """Concatenate the outputs along the step axis."""
            return jnp.concatenate([first, second], axis=0)

        rest_inputs = jax.tree_util.tree_map(tail, inputs)
        carry, rest_outputs = scan_segment(theta, carry, rest_inputs)
        outputs = jax.tree_util.tree_map(join, outputs, rest_outputs)

    return carry, outputs


def _step_inputs(network: Network, forcing: Forcing, n_classes: int, hydro_every: int):
    """Boundary conditions in the layout of the scan (leading axis T).

    Args:
        network: the network (H headwaters).
        forcing: boundary conditions.
        n_classes: number of size classes K.
        hydro_every: the hydraulics are solved every this many steps.

    Returns:
        The step inputs (:class:`_StepInput` with arrays of leading length T).
    """
    n_steps = forcing.n_steps
    n_headwaters = network.n_headwaters

    # Discharge of each headwater, shape [T, H].
    Q = jnp.asarray(forcing.Q_in)
    if Q.ndim == 1:
        Q = jnp.broadcast_to(Q[:, None], (n_steps, n_headwaters))

    # Inflow load, shape [T, K, H] (zeros if the rating curve is used).
    if forcing.load_in is None:
        load_in = jnp.zeros((n_steps, n_classes, n_headwaters), dtype=Q.dtype)
    else:
        load = jnp.asarray(forcing.load_in)
        if load.ndim == 2:
            load = load[:, :, None]
        load = jnp.broadcast_to(load, (n_classes, n_steps, n_headwaters))
        load_in = jnp.transpose(load, (1, 0, 2))

    # Lateral withdrawals, shape [T, W].
    if forcing.Q_withdrawal is None or network.withdrawals is None:
        Q_withdrawal = jnp.zeros((n_steps, 0), dtype=Q.dtype)
    else:
        Q_withdrawal = jnp.reshape(
            jnp.asarray(forcing.Q_withdrawal), (n_steps, network.withdrawals.shape[0]),
        )

    index = np.arange(n_steps)
    return _StepInput(
        Q=Q,
        eta_outlet=jnp.asarray(forcing.eta_outlet),
        dt=jnp.asarray(forcing.dt),
        load_in=load_in,
        solve_hydro=jnp.asarray(index % hydro_every == 0),
        index=jnp.asarray(index),
        Q_withdrawal=Q_withdrawal,
    )


def run(
    config: OneDConfig,
    network_or_lengths,
    theta: Theta,
    forcing: Forcing,
    bed0: BedState,
) -> RunResult:
    """Run the 1-D model over all steps of the forcing.

    The function is pure and differentiable with respect to ``theta`` (and the arrays of
    ``forcing`` and ``bed0``). Wrap it, or a loss built on it, in ``jax.jit``.

    Args:
        config: settings of the 1-D mode.
        network_or_lengths: a :class:`hydrograd.oned.network.Network`, or the reach
            lengths (m) of a single channel, shape [N - 1].
        theta: parameters (see :func:`hydrograd.oned.config.default_parameters`).
        forcing: boundary conditions of the T steps.
        bed0: initial bed.

    Returns:
        The run result (:class:`RunResult`).
    """
    morph = config.morph
    hydraulics = config.hydraulics
    network = as_network(network_or_lengths)
    n_steps = forcing.n_steps
    n_classes = bed0.p_active.shape[0]
    n_sections = bed0.z.shape[0]
    s = bed0.s
    z_floor = bed0.z_floor
    thalweg0 = thalweg(bed0.z)
    diameters = jnp.asarray(config.sediment.diameters, dtype=bed0.z.dtype)
    uses_rating = forcing.load_in is None
    if hydraulics.flow_solver not in FLOW_SOLVERS:
        raise ValueError(
            f"flow_solver must be one of {FLOW_SOLVERS}, not '{hydraulics.flow_solver}'"
        )
    unsteady_flow = hydraulics.flow_solver == "unsteady"
    if unsteady_flow and morph.hydro_every != 1:
        raise ValueError("the unsteady flow solver needs hydro_every = 1")
    flow_substeps = max(1, int(hydraulics.unsteady.n_substeps))
    has_withdrawals = network.withdrawals is not None and forcing.Q_withdrawal is not None
    snapshot_steps = np.asarray(morph.snapshot_steps, dtype=np.int64)

    cv_lengths = control_volume_lengths(
        network, morph.deposition_scheme, morph.head_inflow_fraction,
    )
    inputs = _step_inputs(network, forcing, n_classes, morph.hydro_every)

    def inflow_load(parameters: Theta, step_input: _StepInput) -> Array:
        """Inflow load of the step, prescribed or from the rating curve, shape [K, H]."""
        if uses_rating:
            return rating_curve_load(parameters, step_input.Q)
        return step_input.load_in

    def solve_hydraulics(
        parameters: Theta, z: Array, step_input: _StepInput, eta_guess: Array, p_active: Array,
    ):
        """Converge the backwater profile for the bed ``z`` and attach the derivatives."""
        n_manning = resistance_parameter(parameters, hydraulics, p_active, diameters)
        withdrawal = step_input.Q_withdrawal if has_withdrawals else None
        newton = converge(
            hydraulics, network, z, s, step_input.Q, step_input.eta_outlet, n_manning, eta_guess,
            withdrawal,
        )
        return attach(
            hydraulics, network, newton, z, s, step_input.Q, step_input.eta_outlet, n_manning,
            withdrawal,
        )

    def advance_flow(
        parameters: Theta, z: Array, step_input: _StepInput, flow: UnsteadyState, p_active: Array,
    ):
        """Advance the unsteady flow over the step on the bed ``z``."""
        n_manning = resistance_parameter(parameters, hydraulics, p_active, diameters)
        beta = momentum_coefficient(parameters, n_sections, z.dtype)
        flow = advance(
            hydraulics, network, flow, z, s, step_input.Q, step_input.eta_outlet, n_manning,
            step_input.dt / flow_substeps, flow_substeps, beta,
        )
        return flow, unsteady_hydro_state(hydraulics, flow, z, s, n_manning, step_input.Q)

    def one_step(parameters: Theta, carry: _Carry, step_input: _StepInput):
        """One morphological step: hydraulics, then sediment and bed."""
        # Hydraulics: the unsteady flow over the step, a new backwater solve from the
        # previous water surface, or the previous state.
        flow = carry.flow
        if unsteady_flow:
            flow, hydro = advance_flow(parameters, carry.z, step_input, flow, carry.p_active)
        elif morph.hydro_every == 1:
            hydro = solve_hydraulics(parameters, carry.z, step_input, carry.hydro.eta, carry.p_active)
        else:

            def solved(operand):
                """New hydraulic solve."""
                return solve_hydraulics(
                    parameters, operand.z, step_input, operand.hydro.eta, operand.p_active,
                )

            def reused(operand):
                """Hydraulic state of the last solve."""
                return operand.hydro

            hydro = jax.lax.cond(step_input.solve_hydro, solved, reused, carry)

        # Sediment and bed.
        bed = BedState(
            z=carry.z,
            s=s,
            p_active=carry.p_active,
            p_substrate=carry.p_substrate,
            substrate_thickness=carry.substrate_thickness,
            z_floor=z_floor,
        )
        load_in = inflow_load(parameters, step_input)
        withdrawal = step_input.Q_withdrawal if has_withdrawals else None
        step_parameters = parameters
        if "tau_ce_fresh" in parameters:
            # Deposits laid down during the run erode at a lower critical shear stress than
            # the initial bed: tau_ce goes from its value on the initial bed to
            # tau_ce_fresh as the deposit above the initial thalweg grows past delta_a.
            rise = jnp.maximum(thalweg(carry.z) - thalweg0, 0.0)
            fresh = rise / (rise + parameters["delta_a"])
            tau_ce = parameters["tau_ce"] + fresh * (parameters["tau_ce_fresh"] - parameters["tau_ce"])
            step_parameters = {**parameters, "tau_ce": tau_ce}
        bed, deposition, load = step(
            config, network, cv_lengths, step_parameters, bed, hydro, carry.load, load_in,
            step_input.dt, withdrawal,
        )

        # Store the bed table if this step is a snapshot step.
        is_snapshot = jnp.asarray(snapshot_steps) == step_input.index
        z_snapshots = jnp.where(is_snapshot[:, None, None], bed.z[None], carry.z_snapshots)

        # Recorded quantities of the step.
        load_out = load[:, -1]
        if has_withdrawals:
            leaving = 1.0 - passing_fraction(network, hydro.discharge, withdrawal)
            load_withdrawn = (load[:, :-1] * leaving[None, :])[:, network.withdrawals]
        else:
            load_withdrawn = jnp.zeros((n_classes, 0), dtype=load.dtype)
        load_lost = jnp.sum(load_out) + jnp.sum(load_withdrawn)
        trap_efficiency = 1.0 - load_lost / (jnp.sum(load_in) + LOAD_GUARD)
        output = (
            deposition, thalweg(bed.z), hydro.eta, load_out, load_in, trap_efficiency,
            load_withdrawn,
        )

        carry_new = _Carry(
            z=bed.z,
            p_active=bed.p_active,
            p_substrate=bed.p_substrate,
            substrate_thickness=bed.substrate_thickness,
            hydro=hydro,
            load=load,
            z_snapshots=z_snapshots,
            flow=flow,
        )
        return carry_new, output

    # Initial carry. The water surface is the default start of the Newton iterations; the
    # other fields of the hydraulic state are placeholders (the first step solves).
    first_input = jax.tree_util.tree_map(_first, inputs)
    eta_start = initial_water_surface(
        hydraulics, network, bed0.z, s, first_input.Q, first_input.eta_outlet,
        first_input.Q_withdrawal if has_withdrawals else None,
    )
    zeros = jnp.zeros(n_sections, dtype=bed0.z.dtype)
    hydro_start = HydroState(eta_start, *([zeros] * (len(HydroState._fields) - 1)))

    # Unsteady flow: start from the backwater profile of the first step.
    if unsteady_flow:
        backwater_start = solve(
            hydraulics, network, bed0.z, s, first_input.Q, first_input.eta_outlet,
            resistance_parameter(theta, hydraulics, bed0.p_active, diameters),
        )
        flow_start = state_from_surface(
            hydraulics, backwater_start.eta, backwater_start.discharge, bed0.z, s,
        )
    else:
        flow_start = UnsteadyState(area=zeros, velocity=zeros[:-1], eta=zeros)

    # Lagged load of the first step: the total inflow load at every section.
    load_first = inflow_load(theta, first_input)
    if morph.lag_switch:
        load_first = jax.lax.stop_gradient(load_first)
    load_start = jnp.broadcast_to(
        jnp.sum(load_first, axis=1)[:, None], (n_classes, n_sections),
    )

    carry = _Carry(
        z=bed0.z,
        p_active=bed0.p_active,
        p_substrate=bed0.p_substrate,
        substrate_thickness=bed0.substrate_thickness,
        hydro=hydro_start,
        load=load_start,
        z_snapshots=jnp.zeros((snapshot_steps.shape[0],) + bed0.z.shape, dtype=bed0.z.dtype),
        flow=flow_start,
    )

    # Time loop.
    length = _segment_length(morph.checkpoint, n_steps)
    carry, outputs = _scan_segments(one_step, theta, carry, inputs, n_steps, length)
    deposition, thalweg_history, eta_history, load_out, load_in, trap_efficiency = outputs[:6]
    load_withdrawn = outputs[6]

    bed = BedState(
        z=carry.z,
        s=s,
        p_active=carry.p_active,
        p_substrate=carry.p_substrate,
        substrate_thickness=carry.substrate_thickness,
        z_floor=z_floor,
    )
    every = morph.record_every
    return RunResult(
        bed=bed,
        deposition=deposition,
        time_end=forcing.time_end(),
        thalweg=thalweg_history[::every],
        eta=eta_history[::every],
        load_out=load_out[::every],
        load_in=load_in[::every],
        trap_efficiency=trap_efficiency[::every],
        z_snapshots=carry.z_snapshots if snapshot_steps.shape[0] > 0 else None,
        load_withdrawn=load_withdrawn[::every] if has_withdrawals else None,
    )


def _first(array: Array) -> Array:
    """First entry along the step axis."""
    return array[0]
