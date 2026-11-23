"""Bed of the 1-D mode: state, supply limit, Exner factor and bed update of cross-sections.

This module connects the geometry of the cross-sections to the dimension-independent bed
composition of :mod:`hydrograd.sediment.bed`.

Control volumes and deposition width
------------------------------------
Section i owns a control volume of streamwise length dx_cv,i
(:func:`hydrograd.oned.network.control_volume_lengths`) and of width B_d,i, the deposition
width (:func:`hydrograd.oned.geometry.deposition_width`): a bed change dz of the wetted
points changes the volume of the section's control volume by B_d dx_cv dz.

Exner equation
--------------
A solids volume V_k = (deposition rate of class k at the section) dt MF deposited in a
step (MF is the morphological factor) changes the bed by

    D_k = V_k / ((1 - lambda_k) B_d dx_cv),     dz = sum_k D_k,

with the porosity lambda_k. The wetted station points are raised by dz times their wet
weights (a uniform shift over the wetted perimeter, as in SRH-1D). With the setting
``deposition_depth_exponent`` m > 0 the shift of a point is also proportional to
(h / h_max)^m, so that the deposit fills the deep parts of the section first; dz is then
the bed change of the deepest point. ``deposition_channel_deficit`` reduces the change of
the deepest part of the section (a deposit beside a channel that stays open), and
``erosion_depth_exponent`` gives a net erosion of a section its own depth exponent (an
erosion that cuts the deep part while deposits form a veneer). The active layer and
the substrate are updated by :func:`hydrograd.sediment.bed.update_composition`.

Quantities for the routing
--------------------------
- Supply limit: the solids volume of class k the bed can give per time and per length of
  channel,

      cap_k = p_a,k h_avail (1 - lambda_k) B_d / (dt MF)                  (m^2/s),

  with h_avail the smaller of the active-layer thickness and the erodible thickness above
  a non-erodible floor.
- Exner factor: bed thickness per unit deposition rate,

      c_k = dt MF / ((1 - lambda_k) B_d dx_cv)                            (s/m^2).

- Sensitivity of the capacity to the bed: d Q*_k / d z_b at a fixed water surface and
  discharge, by a finite difference with the wetted points raised by 1 mm. Raising the bed
  under a fixed water surface speeds the flow up and increases the capacity; this is the
  feedback the locally implicit routing uses.

Units: SI.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
from jax import Array

from hydrograd.oned.backwater import (
    HydroState,
    hydro_state_from_surface,
    sediment_flow_state,
)
from hydrograd.oned.config import OneDConfig, resistance_parameter
from hydrograd.oned.geometry import deposition_weights, deposition_width, thalweg
from hydrograd.oned.network import Network, reach_to_section
from hydrograd.sediment.bed import (
    BedComposition,
    InterfaceMix,
    available_thickness,
    porosity,
    update_composition,
)
from hydrograd.sediment.closures import transport_capacity
from hydrograd.sediment.config import INTERFACE_EXCHANGES

__all__ = [
    "BedState",
    "BedUpdate",
    "capacity_bed_sensitivity",
    "exner_factor",
    "supply_cap_per_length",
    "update_bed",
]

# Differentiable parameters (see :func:`hydrograd.oned.config.default_parameters`).
Theta = dict[str, Any]

# Bed shift (m) of the finite difference for the sensitivity of the capacity.
SHIFT = 1.0e-3


class BedState(NamedTuple):
    """Bed of N cross-sections with J station points and K size classes.

    Attributes:
        z: bed elevation of the station points (m), shape [N, J].
        s: station of the points (m), shape [N, J]; does not change.
        p_active: fractions of the active layer (-), shape [K, N].
        p_substrate: fractions of the substrate (-), shape [K, N].
        substrate_thickness: thickness of the substrate (m), shape [N].
        z_floor: elevation of the non-erodible floor (m), shape [N], or None for a fully
            erodible bed.
    """

    z: Array
    s: Array
    p_active: Array
    p_substrate: Array
    substrate_thickness: Array
    z_floor: Array | None = None


class BedUpdate(NamedTuple):
    """Result of one bed update.

    Attributes:
        bed: bed after the step.
        dz: bed elevation change of the step (m), shape [N].
        deposition: solids volume deposited at each section in the step (m^3), positive
            for deposition, shape [K, N].
    """

    bed: BedState
    dz: Array
    deposition: Array


def _shift_weights(config: OneDConfig, bed: BedState, eta: Array, erosion: bool) -> tuple[Array, Array]:
    """Point weights and width of a bed change: of a deposit, or of an erosion.

    Args:
        config: settings of the 1-D mode.
        bed: bed state.
        eta: water-surface elevation (m), shape [N].
        erosion: True for the weights of a net erosion (``erosion_depth_exponent``, if
            set), False for those of a net deposit.

    Returns:
        Tuple ``(weights, width)`` of :func:`hydrograd.oned.geometry.deposition_width`.
    """
    morph = config.morph
    if erosion and morph.erosion_depth_exponent is not None:
        return deposition_width(
            eta, bed.z, bed.s, config.hydraulics.smoothing, morph.deposition_width,
            morph.erosion_depth_exponent,
        )
    return deposition_width(
        eta, bed.z, bed.s, config.hydraulics.smoothing, morph.deposition_width,
        morph.deposition_depth_exponent, morph.deposition_channel_deficit,
    )


def supply_cap_per_length(
    config: OneDConfig, bed: BedState, eta: Array, theta: Theta, dt: Array,
) -> Array:
    """Solids each class of the bed can supply per time and per length of channel.

    Args:
        config: settings of the 1-D mode.
        bed: bed state.
        eta: water-surface elevation (m), shape [N].
        theta: parameters ``delta_a`` (m) and ``gamma1`` (kg/m^3).
        dt: time step (s), scalar.

    Returns:
        Supply limit (m^2/s of solids), shape [K, N].
    """
    morph = config.morph
    solid_fraction = 1.0 - porosity(theta, config.sediment.constants)

    # Thickness the flow can erode in the step, and the deposition width.
    thickness = available_thickness(
        thalweg(bed.z), bed.z_floor, theta["delta_a"], config.sediment.smoothing,
    )
    _, width = _shift_weights(config, bed, eta, erosion=True)

    # Solids volume of each class per length, divided by the (accelerated) time step.
    volume_per_length = bed.p_active * solid_fraction[:, None] * (thickness * width)[None, :]
    return volume_per_length / (dt * morph.morph_factor)


def exner_factor(
    config: OneDConfig,
    bed: BedState,
    eta: Array,
    theta: Theta,
    dt: Array,
    cv_lengths: Array,
) -> Array:
    """Bed thickness per unit deposition rate, c_k = dt MF / ((1 - lambda_k) B_d dx_cv).

    Args:
        config: settings of the 1-D mode.
        bed: bed state.
        eta: water-surface elevation (m), shape [N].
        theta: parameters with ``gamma1`` (kg/m^3).
        dt: time step (s), scalar.
        cv_lengths: control-volume length of each section (m), shape [N].

    Returns:
        Exner factor (s/m^2), shape [K, N].
    """
    morph = config.morph
    solid_fraction = 1.0 - porosity(theta, config.sediment.constants)
    _, width = _shift_weights(config, bed, eta, erosion=False)

    return dt * morph.morph_factor / (solid_fraction[:, None] * (width * cv_lengths)[None, :])


def capacity_bed_sensitivity(
    config: OneDConfig, hydro: HydroState, bed: BedState, theta: Theta, capacity: Array,
) -> Array:
    """Sensitivity d Q* / d z_b of the capacity to a bed shift at a fixed water surface.

    Args:
        config: settings of the 1-D mode.
        hydro: hydraulic state at the N sections.
        bed: bed state.
        theta: parameters (that of the resistance law and the capacity parameters).
        capacity: transport capacity Q* of the unshifted bed (m^3/s), shape [K, N].

    Returns:
        d Q* / d z_b (m^2/s of solids), shape [K, N]; zero for the cohesive classes.
    """
    hydraulics = config.hydraulics
    n_sections = bed.z.shape[0]
    roughness = resistance_parameter(
        theta, hydraulics, bed.p_active, jnp.asarray(config.sediment.diameters, dtype=bed.z.dtype),
    )
    n_manning = jnp.broadcast_to(roughness, (n_sections,))

    # Raise the wetted points by SHIFT under the same water surface and discharge.
    weights = deposition_weights(
        hydro.eta, bed.z, hydraulics.smoothing, config.morph.deposition_depth_exponent,
        config.morph.deposition_channel_deficit,
    )
    z_shifted = bed.z + SHIFT * weights
    hydro_shifted = hydro_state_from_surface(
        hydraulics, hydro.eta, hydro.discharge, z_shifted, bed.s, n_manning,
    )

    # Capacity for the shifted bed, and the finite difference.
    capacity_shifted = transport_capacity(
        sediment_flow_state(hydro_shifted), bed.p_active, theta, config.sediment,
    )
    return (capacity_shifted - capacity) / SHIFT


def update_bed(
    config: OneDConfig,
    network: Network,
    cv_lengths: Array,
    bed: BedState,
    eta: Array,
    deposition_reach: Array,
    theta: Theta,
    dt: Array,
    load: Array | None = None,
) -> BedUpdate:
    """Apply the deposition of one step to the bed (Exner, active layer, substrate).

    Args:
        config: settings of the 1-D mode.
        network: the network.
        cv_lengths: control-volume length of each section (m), shape [N].
        bed: bed before the step.
        eta: water-surface elevation (m), shape [N].
        deposition_reach: deposition rate of each reach per class (m^3/s of solids),
            shape [K, M] or [K, M + H] (with the virtual inflow reaches).
        theta: parameters ``gamma1`` (kg/m^3) and ``delta_a`` (m); ``c_interface`` (-)
            with the interface exchange ``"bedload_mix"``.
        dt: time step (s), scalar.
        load: load of each class at the sections (m^3/s), shape [K, N]; needed with the
            interface exchange ``"bedload_mix"``
            (``SedimentConfig.interface_exchange``).

    Returns:
        The updated bed, the bed change and the deposited volumes (:class:`BedUpdate`).
    """
    morph = config.morph
    sediment = config.sediment
    if sediment.interface_exchange not in INTERFACE_EXCHANGES:
        raise ValueError(
            f"interface_exchange must be one of {INTERFACE_EXCHANGES}, "
            f"not '{sediment.interface_exchange}'"
        )
    interface = None
    if sediment.interface_exchange == "bedload_mix":
        if load is None:
            raise ValueError("the interface exchange 'bedload_mix' needs the load at the sections")
        interface = InterfaceMix(
            load=load,
            diameters=jnp.asarray(sediment.diameters),
            bedload_share=theta["c_interface"],
            sand_rule=sediment.interface_sand,
        )
    solid_fraction = 1.0 - porosity(theta, config.sediment.constants)

    # Solids volume deposited at each section in the step (m^3), shape [K, N].
    rate_section = reach_to_section(network, deposition_reach, morph.deposition_scheme)
    deposition = rate_section * dt * morph.morph_factor

    # Bulk thickness change of each class and the bed change.
    weights, width = _shift_weights(config, bed, eta, erosion=False)
    if morph.erosion_depth_exponent is not None:
        # A net erosion of the section is placed with its own weights and width.
        erosion_weights, erosion_width = _shift_weights(config, bed, eta, erosion=True)
        erodes = jnp.sum(deposition / solid_fraction[:, None], axis=0) < 0.0
        weights = jnp.where(erodes[:, None], erosion_weights, weights)
        width = jnp.where(erodes, erosion_width, width)
    bed_area = width * cv_lengths
    thickness_change = deposition / solid_fraction[:, None] / bed_area[None, :]
    dz = jnp.sum(thickness_change, axis=0)

    # Active layer and substrate.
    composition = BedComposition(
        active=bed.p_active,
        substrate=bed.p_substrate,
        substrate_thickness=bed.substrate_thickness,
    )
    composition = update_composition(
        composition,
        thickness_change,
        theta["delta_a"],
        config.sediment.smoothing,
        morph.composition_update,
        interface,
    )

    # Raise the wetted station points.
    z_new = bed.z + weights * dz[:, None]

    bed_new = BedState(
        z=z_new,
        s=bed.s,
        p_active=composition.active,
        p_substrate=composition.substrate,
        substrate_thickness=composition.substrate_thickness,
        z_floor=bed.z_floor,
    )
    return BedUpdate(bed=bed_new, dz=dz, deposition=deposition)
