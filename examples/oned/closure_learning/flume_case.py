"""The downstream-fining flume runs of Seal et al. (1997) in the 1-D mode, and run 6 of
Toro-Escobar et al. (2000) (same flume width, other conditions; see ``flume_data.py``).

A rectangular flume (45 m long, 0.3 m wide, smooth non-erodible floor on a slope of 0.002)
carries 49 L/s; a gravel-sand mixture of 20 size classes is fed at the upstream end and a
tailgate holds the water level at the downstream end (a different level in each run).
The model covers the 45 m from the feed point to the end of the measured reach, without
the ramp and the tailgate pool beyond it: the water there is deep and slow, so that its
surface is level from the gravel front to the tailgate. The deposit is a wedge whose gravel
fines downstream. The data are in ``flume_data.py``.

The observations of a run are the thickness of the deposit at the measuring stations and
times of the bed profiles, the grain sizes (psi = log2(d in mm) of given percentiles) of
the deposit laid down between two times, and the mean water depth over the deposit.

The flow is computed either by the backwater solver (default), which holds a section at
critical depth where the flow would be supercritical, or by the unsteady shallow-water
solver, which follows the flow through both regimes (``build(..., flow="unsteady")``).

Units: SI; psi for grain-size percentiles.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from hydrograd.oned import OneDConfig, constant_forcing, default_parameters, run
from hydrograd.oned.backwater import hydro_state_from_surface, sediment_flow_state
from hydrograd.oned.config import resistance_parameter
from hydrograd.oned.forcing import Forcing
from hydrograd.oned.synthetic import trapezoid_channel, uniform_bed
from hydrograd.sediment import interface_parameters, wilcock_crowe_parameters
from hydrograd.sediment.closures import wilcock_crowe_stress_ratio

import flume_data as data

N_SECTIONS = 46                 # runs 1 to 3: one section per metre, at the measuring stations
DT = 60.0                       # s
N_SUBSTEPS = 4
# Flow steps per step with the unsteady flow solver: 0.25 s, a Courant number of about
# 0.6 for a section spacing of 1 m and |u| + c up to 2.5 m/s.
FLOW_SUBSTEPS = 240
BULK_DENSITY = 2140.0           # kg/m^3: mass fed over the volume of the surveyed wedges
ACTIVE_LAYER = 0.05             # m, about the coarsest grains of the mixture
N_MANNING = 0.025
ROUGHNESS_LENGTH = 0.02         # m, start value for the resistance laws with a roughness length
ROUGHNESS_FACTOR = 2.5          # roughness length over the mean size of the surface (Cui, 2007)
# Class limits (psi), finest first, and the feed fractions in the same order.
PSI_LIMITS = np.linspace(-3.5, 6.5, 21)
DIAMETERS = tuple(float(v) for v in 1.0e-3 * 2.0 ** (0.5 * (PSI_LIMITS[:-1] + PSI_LIMITS[1:])))


class Case(NamedTuple):
    """Everything a run needs.

    Attributes:
        run: number of the experimental run (1 to 3, or 6).
        config: settings of the 1-D mode.
        lengths: reach lengths (m), shape [N - 1].
        theta: parameters with the default closures.
        forcing: boundary conditions.
        bed0: initial bed.
        x: position of the sections, from the feed point (m), shape [N].
        discharge: water discharge (m^3/s).
    """

    run: int
    config: OneDConfig
    lengths: jnp.ndarray
    theta: dict
    forcing: Forcing
    bed0: object
    x: np.ndarray
    discharge: float = data.DISCHARGE


class Layout(NamedTuple):
    """Reach and boundary conditions of a run.

    Attributes:
        x_inlet: position of the first section (m from the feed point).
        length: length of the modelled reach (m).
        n_sections: number of sections.
        slope: slope of the fixed floor (-).
        discharge: water discharge (m^3/s).
        tailwater: water level at the last section, above the floor there (m).
        feed: feed rate (kg/s).
        hours: duration (h).
    """

    x_inlet: float
    length: float
    n_sections: int
    slope: float
    discharge: float
    tailwater: float
    feed: float
    hours: float


# Run 6: sections every 0.875 m, half the spacing of the survey sections, which are every
# second section from the second one (x = 13 m). The first section, at 12.125 m, stands
# for the head of the deposit: the feed enters there (the fast, shallow flow from the feed
# point to the deposit is not modelled).
RUN6_SPACING = 0.875
RUN6_SECTIONS = 44
# Sharpness (1/m) and width (m) of the smooth waterline for run 6 (see ``build``).
RUN6_SOFTPLUS = 200.0
RUN6_EPS_WET = 0.005


def layout(run_number: int, feed_kg_per_min: float | None = None) -> Layout:
    """Reach and boundary conditions of a run (``feed_kg_per_min`` replaces the feed rate)."""
    if run_number == 6:
        x_inlet = float(data.RUN6_STATIONS[0]) - RUN6_SPACING
        feed = data.RUN6["feed_kg_per_min"] if feed_kg_per_min is None else feed_kg_per_min
        return Layout(
            x_inlet, RUN6_SPACING * (RUN6_SECTIONS - 1), RUN6_SECTIONS, data.RUN6["floor_slope"],
            data.RUN6["discharge_m3s"], data.RUN6["tailwater_m"], feed / 60.0, data.RUN6["hours"],
        )
    feed = data.feed_rate(run_number) if feed_kg_per_min is None else feed_kg_per_min / 60.0
    return Layout(
        0.0, data.LENGTH, N_SECTIONS, data.FLOOR_SLOPE, data.DISCHARGE, data.TAILWATER[run_number], feed,
        data.RUNS[run_number]["hours"],
    )


def feed_fractions(run_number: int) -> np.ndarray:
    """Fractions of the feed per class, finest first (run 2: as fed; run 6: the sandy
    mixture; else the target mixture)."""
    percent = {2: data.FEED_RUN2_PERCENT, 6: data.FEED_RUN6_PERCENT}.get(run_number, data.FEED_TARGET_PERCENT)
    return percent[::-1] / percent.sum()


def step_of(hours: float) -> int:
    """Number of steps up to a time (h)."""
    return round(hours * 3600.0 / DT)


def build(
    run_number: int, formula: str = "engelund_hansen", flow: str = "backwater", exchange: str = "surface",
    feed_kg_per_min: float | None = None, resistance: str = "manning", transport_stress: str = "total",
    roughness_scale: str = "length",
) -> Case:
    """Build an experimental run.

    Args:
        run_number: 1, 2, 3 or 6.
        formula: transport formula, ``"engelund_hansen"`` or ``"wilcock_crowe"``.
        flow: flow solver, ``"backwater"`` or ``"unsteady"``.
        exchange: what the surface layer passes down to the deposit in aggradation:
            ``"surface"`` (its own composition) or ``"bedload_mix"`` (for the gravel, 70 %
            load and 30 % surface, Toro-Escobar et al., 1996; for the sand, the relation
            of Cui, 2007).
        feed_kg_per_min: feed rate in place of that of the run (kg/min).
        resistance: flow-resistance law, ``"manning"`` (Manning's n),
            ``"variable_power"`` (Ferguson, 2007) or ``"keulegan"`` (logarithmic law); the
            last two have a roughness length, which makes a shallow flow rougher than a
            deep one over the same bed.
        transport_stress: ``"total"``, or ``"reduced"`` for the drag on the largest grains
            in shallow flow (not with ``"manning"``); see ``hydrograd.oned.config``.
        roughness_scale: ``"length"`` (one roughness length for the run),
            ``"surface_mean"`` (a factor times the geometric mean size of the surface
            layer of each section, as in Cui, 2007) or ``"surface_d90"`` (the same with
            the D90 of the surface layer).

    Returns:
        The case, with the power-law hiding factor and m_h = 0 (no hiding).
    """
    reach = layout(run_number, feed_kg_per_min)
    n_sections = reach.n_sections
    x, z, s = trapezoid_channel(
        length=reach.length, n_sections=n_sections, slope=reach.slope, bottom_width=data.WIDTH,
        side_slope=1.0e-3, bank_height=data.WALL_HEIGHT, z_dam_bed=0.0,
    )
    n_classes = len(DIAMETERS)
    config = OneDConfig()
    sediment = dataclasses.replace(
        config.sediment, diameters=DIAMETERS, cohesive_dmax=0.0, transport_formula=formula,
        interface_exchange=exchange, interface_sand="cui" if exchange == "bedload_mix" else "surface",
    )
    morph = dataclasses.replace(config.morph, n_substeps=N_SUBSTEPS)
    unsteady = dataclasses.replace(config.hydraulics.unsteady, n_substeps=FLOW_SUBSTEPS)
    # The walls of the flume are smooth and the conveyance has no wall friction, so the
    # whole friction force acts on the bed: bed shear stress with the mean depth.
    hydraulics = dataclasses.replace(
        config.hydraulics, flow_solver=flow, unsteady=unsteady, shear_radius="depth",
        resistance=resistance, transport_stress=transport_stress, roughness_scale=roughness_scale,
    )
    if run_number == 6:
        # The flow of run 6 is 0.03 to 0.07 m deep. The default widths of the smooth
        # waterline (0.02 and 0.05 m) distort so shallow a flow: the deposit at the first
        # survey section comes out 0.09 m thicker than with a sharp waterline. With the
        # widths below the results no longer change (runs 1 to 3, 0.12 m deep and more,
        # do not depend on them).
        sharp = dataclasses.replace(hydraulics.smoothing, beta_softplus=RUN6_SOFTPLUS, eps_wet=RUN6_EPS_WET)
        hydraulics = dataclasses.replace(hydraulics, smoothing=sharp)
    config = dataclasses.replace(config, hydraulics=hydraulics, sediment=sediment, morph=morph)

    theta = default_parameters(n_classes, n_sections)
    theta["n_manning"] = jnp.asarray(N_MANNING)
    theta["roughness_length"] = jnp.asarray(ROUGHNESS_LENGTH)
    theta["roughness_factor"] = jnp.asarray(ROUGHNESS_FACTOR)
    theta["delta_a"] = jnp.asarray(ACTIVE_LAYER)
    theta["gamma1"] = jnp.full(n_classes, BULK_DENSITY)
    theta.update(wilcock_crowe_parameters())
    theta.update(interface_parameters())

    fractions = feed_fractions(run_number)
    n_steps = step_of(reach.hours)
    forcing = constant_forcing(reach.discharge, reach.tailwater, DT, n_steps)
    load = reach.feed / data.GRAIN_DENSITY * fractions          # m^3/s of solids
    forcing = forcing._replace(load_in=jnp.broadcast_to(jnp.asarray(load)[:, None], (n_classes, n_steps)))
    bed0 = uniform_bed(z, s, n_classes, fractions=tuple(fractions), fixed=jnp.ones(n_sections, bool))
    return Case(run_number, config, jnp.diff(x), theta, forcing, bed0, reach.x_inlet + np.asarray(x), reach.discharge)


def with_hiding(case: Case, form: str, **settings) -> Case:
    """The same case with another form of the hiding factor (see ``case.with_hiding``)."""
    sediment = dataclasses.replace(case.config.sediment, hiding_form=form, **settings)
    return case._replace(config=dataclasses.replace(case.config, sediment=sediment))


def percentile_psi(fractions, percent):
    """Grain size (psi) below which ``percent`` of a mixture lies.

    Args:
        fractions: fractions per class, finest first (-), shape [K, N].
        percent: percentiles (0 to 100), shape [P].

    Returns:
        psi, shape [P, N]; linear in psi within a class.
    """
    fractions = fractions / jnp.sum(fractions, axis=0, keepdims=True)
    cumulative = jnp.concatenate([jnp.zeros((1, fractions.shape[1])), jnp.cumsum(fractions, axis=0)], axis=0)
    # A small ramp keeps the cumulative curve increasing where a class is empty.
    cumulative = cumulative + 1.0e-9 * jnp.arange(cumulative.shape[0])[:, None]
    limits = jnp.asarray(PSI_LIMITS)
    return jax.vmap(lambda c: jnp.interp(jnp.asarray(percent) / 100.0, c, limits), in_axes=1, out_axes=1)(cumulative)


def run_case(case: Case, theta: dict):
    """Run a case; returns the result of ``hydrograd.oned.run``."""
    return run(case.config, case.lengths, theta, case.forcing, case.bed0)


def observables(case: Case, result):
    """What the measurements are compared with, from the result of a run.

    Returns:
        Tuple ``(thickness, cumulative, surface, depth)``: thickness of the deposit after
        every step (m), shape [T, N]; solids volume deposited per class up to every step
        (m^3), shape [T, K, N]; fractions of the active layer at the end (-), shape
        [K, N]; water depth at the end (m), shape [N].
    """
    floor = jnp.min(case.bed0.z, axis=1)
    depth = result.eta[-1] - result.thalweg[-1]
    return result.thalweg - floor[None, :], jnp.cumsum(result.deposition, axis=0), result.bed.p_active, depth


def simulate(case: Case, theta: dict):
    """Run a case; returns ``(thickness, cumulative, surface, depth)`` (see ``observables``)."""
    return observables(case, run_case(case, theta))


def stress_ratio(case: Case, theta: dict, result):
    """Ratio of the bed shear stress to the reference shear stress of every class at the
    end of a run (Wilcock-Crowe formula), shape [K, N].

    The flow is that of the water surface and the bed at the end of the run with the
    inflow discharge at every section (the flow is steady but for the slow growth of the
    deposit).
    """
    bed = result.bed
    n_sections = case.x.size
    discharge = jnp.full(n_sections, case.discharge)
    roughness = resistance_parameter(theta, case.config.hydraulics, bed.p_active, jnp.asarray(DIAMETERS))
    n_manning = jnp.broadcast_to(roughness, (n_sections,))
    hydro = hydro_state_from_surface(case.config.hydraulics, result.eta[-1], discharge, bed.z, bed.s, n_manning)
    return wilcock_crowe_stress_ratio(
        sediment_flow_state(hydro), jnp.asarray(DIAMETERS), bed.p_active, theta, case.config.sediment,
    )


def deposit_sections(run_number: int) -> np.ndarray:
    """Sections on the deposit at the end of a run: from 2 m to 3 m above the gravel front."""
    return np.arange(2, int(data.RUNS[run_number]["front_m"]) - 2)


def deposit_fractions(cumulative, hours: tuple[float, float]):
    """Fractions of the deposit laid down between two times (h), shape [K, N]."""
    laid = cumulative[step_of(hours[1]) - 1] - cumulative[step_of(hours[0]) - 1]
    laid = jnp.maximum(laid, 0.0) + 1.0e-12
    return laid / jnp.sum(laid, axis=0, keepdims=True)
