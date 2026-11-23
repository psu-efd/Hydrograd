"""Synthetic river reach for learning the hiding factor of a sediment mixture.

A straight trapezoidal channel carries a steady discharge over a bed of five sand classes.
Two scenarios use the same channel:

- ``"aggradation"``: the inflow load is ``FEED_FACTOR`` times the load the channel carries
  in equilibrium; a wedge of sediment builds up from the upstream end and fines downstream.
- ``"degradation"``: clear water enters; the bed erodes and its surface coarsens.

The observations of a scenario are the solids volume deposited (negative: eroded) per size
class and cross-section at the survey times, and the composition of the bed surface (active
layer) at the end.

Units: SI.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from hydrograd.oned import OneDConfig, constant_forcing, default_parameters, run
from hydrograd.oned.forcing import Forcing
from hydrograd.oned.synthetic import (
    normal_depth_trapezoid,
    trapezoid_channel,
    uniform_bed,
)

# Channel.
LENGTH = 6000.0          # m
N_SECTIONS = 61
SLOPE = 1.0e-3
BOTTOM_WIDTH = 40.0      # m
SIDE_SLOPE = 2.0
DISCHARGE = 80.0         # m^3/s
N_MANNING = 0.03

# Sediment: five sand classes, equal fractions in the initial bed and in the feed.
DIAMETERS = (1.25e-4, 2.5e-4, 5.0e-4, 1.0e-3, 2.0e-3)   # m
FEED_FACTOR = 3.0

# Time: steps of 6 hours with 6 sub-steps, surveys at the given steps.
DT = 6.0 * 3600.0
N_STEPS = 480
SURVEY_STEPS = (120, 240, 480)

# Hiding law of the synthetic truth: fine classes are sheltered strongly, coarse classes
# are exposed weakly (exponents of ``hiding_form = "two_power"``).
TRUTH = {"m_h_fine": -0.9, "m_h_coarse": -0.2}


class Case(NamedTuple):
    """Everything a run needs.

    Attributes:
        config: settings of the 1-D mode.
        lengths: reach lengths (m), shape [N - 1].
        theta: parameters with the default closures.
        forcing: boundary conditions.
        bed0: initial bed.
        x: position of the sections (m), shape [N].
    """

    config: OneDConfig
    lengths: jnp.ndarray
    theta: dict
    forcing: Forcing
    bed0: object
    x: np.ndarray


def with_hiding(case: Case, form: str, **settings) -> Case:
    """The same case with another form of the hiding factor.

    Args:
        case: the case.
        form: ``"power"``, ``"two_power"`` or ``"network"``.
        **settings: further fields of the sediment settings (``hiding_*``).

    Returns:
        The case with the changed settings.
    """
    sediment = dataclasses.replace(case.config.sediment, hiding_form=form, **settings)
    return case._replace(config=dataclasses.replace(case.config, sediment=sediment))


def build(scenario: str = "aggradation", n_steps: int = N_STEPS) -> Case:
    """Build a scenario.

    Args:
        scenario: ``"aggradation"`` or ``"degradation"``.
        n_steps: number of steps.

    Returns:
        The case, with the power-law hiding factor and m_h = 0 (no hiding).
    """
    x, z, s = trapezoid_channel(
        length=LENGTH, n_sections=N_SECTIONS, slope=SLOPE, bottom_width=BOTTOM_WIDTH,
        side_slope=SIDE_SLOPE, bank_height=10.0, z_dam_bed=0.0,
    )
    n_classes = len(DIAMETERS)
    config = OneDConfig()
    sediment = dataclasses.replace(config.sediment, diameters=DIAMETERS, cohesive_dmax=0.0)
    morph = dataclasses.replace(config.morph, n_substeps=6)
    config = dataclasses.replace(config, sediment=sediment, morph=morph)

    theta = default_parameters(n_classes, N_SECTIONS)
    theta["n_manning"] = jnp.asarray(N_MANNING)
    theta["delta_a"] = jnp.asarray(0.1)

    depth = normal_depth_trapezoid(DISCHARGE, N_MANNING, SLOPE, BOTTOM_WIDTH, SIDE_SLOPE)
    forcing = constant_forcing(DISCHARGE, float(depth), DT, n_steps)
    bed0 = uniform_bed(z, s, n_classes)
    case = Case(config, jnp.diff(x), theta, forcing, bed0, np.asarray(x))

    # Inflow load: a multiple of the load of uniform flow with the truth closure.
    truth = with_hiding(case, "two_power")
    factor = FEED_FACTOR if scenario == "aggradation" else 0.0
    load = factor * equilibrium_load(truth, depth)
    forcing = forcing._replace(load_in=jnp.broadcast_to(load[:, None], (n_classes, n_steps)))
    return case._replace(forcing=forcing)


def as_arrays(values: dict) -> dict:
    """Dictionary of floats as JAX arrays."""
    return {name: jnp.asarray(value) for name, value in values.items()}


def equilibrium_load(case: Case, depth) -> jnp.ndarray:
    """Load per class in uniform flow over the initial bed (m^3/s of solids), shape [K]."""
    from hydrograd.sediment.closures import FlowState, transport_capacity

    area = (BOTTOM_WIDTH + SIDE_SLOPE * depth) * depth
    top = BOTTOM_WIDTH + 2.0 * SIDE_SLOPE * depth
    perimeter = BOTTOM_WIDTH + 2.0 * depth * np.sqrt(1.0 + SIDE_SLOPE**2)
    constants = case.config.sediment.constants
    tau = constants.rho * constants.g * (area / perimeter) * SLOPE
    velocity = DISCHARGE / area
    flow = FlowState(
        discharge=jnp.asarray([DISCHARGE]), width=jnp.asarray([top]), depth=jnp.asarray([area / top]),
        shear_stress=jnp.asarray([tau]), shear_velocity=jnp.sqrt(jnp.asarray([tau]) / constants.rho),
        friction_coefficient=jnp.asarray([tau / (constants.rho * velocity**2)]),
    )
    fractions = case.bed0.p_active[:, :1]
    theta = {**case.theta, **as_arrays(TRUTH)}
    return transport_capacity(flow, fractions, theta, case.config.sediment)[:, 0]


def observe(case: Case, theta: dict, survey_steps=SURVEY_STEPS):
    """Run the case and return its observations.

    Args:
        case: the case.
        theta: parameters.
        survey_steps: steps after which the deposits are surveyed.

    Returns:
        Tuple ``(deposit, surface, thalweg)``: solids volume deposited per class and
        section at the surveys (m^3), shape [S, K, N]; fractions of the active layer at
        the end (-), shape [K, N]; thalweg at the surveys (m), shape [S, N].
    """
    result = run(case.config, case.lengths, theta, case.forcing, case.bed0)
    cumulative = jnp.cumsum(result.deposition, axis=0)
    index = jnp.asarray([step - 1 for step in survey_steps if step <= case.forcing.n_steps])
    return cumulative[index], result.bed.p_active, result.thalweg[index]
