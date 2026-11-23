"""Bed composition: Hirano active layer, substrate exchange and erodible thickness.

The functions work per location (a cross-section in 1-D, a cell in 2-D) and know nothing
about geometry: the caller converts a deposited volume into the thickness change of each
size class and applies the resulting bed change to its own geometry.

Layers
------
The bed has a thin active layer of thickness delta_a (m), which exchanges sediment with
the flow and has the composition p_a,k, above a substrate of thickness delta_sub (m) and
composition p_sub,k (fractions per size class k, each summing to one).

Thickness change
----------------
A solids volume V_k (m^3) of class k deposited on the bed area A_bed (m^2) changes the bed
by the bulk thickness

    D_k = V_k / ((1 - lambda_k) A_bed),     lambda_k = 1 - gamma1_k / rho_s,

with the porosity lambda_k from the initial dry bulk density gamma1_k (kg/m^3). The bed
change is dz = sum_k D_k. D_k is split into its deposition part D_k^+ = max(D_k, 0) and
erosion part D_k^- = min(D_k, 0), with dz^+ and dz^- their sums over the classes.

Active layer (default ``method="relaxation"``)
----------------------------------------------
Erosion (explicit, SRH-1D eq. 3.102): the layer loses D_k^- and is refilled from below
with the composition p_below (the substrate's, or the layer's own where no substrate is
left),

    p_1,k = p_a,k + (D_k^- - p_below,k dz^-) / delta_a.

It is stable because the supply limit of the exchange keeps |D_k^-| <= p_a,k delta_a.

Deposition (exact relaxation): the layer composition relaxes to the composition of the
deposit, p_dep,k = D_k^+ / dz^+,

    p_a,k <- p_1,k + (p_dep,k - p_1,k) (1 - exp(-dz^+ / delta_a)),

which is the exact solution of the layer equation for a steady deposit and is stable
however thick the deposit of one step is. The explicit form p_a + (D - p dz) / delta_a
has the coefficient 1 - dz / delta_a, below -1 for dz > 2 delta_a, which makes the adjoint
grow geometrically; it is kept as ``method="explicit"`` for the adjoint-stability study.

Substrate
---------
The substrate receives what the layer passes down in deposition (composition p_1) and
gives what is taken from below in erosion:

    exchange_k = p_1,k dz^+ + p_below,k dz^-,
    delta_sub <- eps_floor + softplus(delta_sub + dz - eps_floor),
    p_sub,k <- (p_sub,k delta_sub + exchange_k) / delta_sub(new),   renormalized.

Interface exchange in aggradation (``SedimentConfig.interface_exchange``)
-------------------------------------------------------------------------
With the bed elevation eta and the layer thickness L_a, the sediment balance of class k
in the active layer is (Parker, in Garcia, 2008, ASCE Manual 110, eq. 3-23)

    (1 - lambda) [ f_I,k d(eta - L_a)/dt + d(L_a F_k)/dt ] = - dq_k/dx,

with the surface fractions F_k (here p_a,k) and the fractions f_I,k of the material that
crosses the interface between layer and substrate. In erosion f_I,k is the substrate
(Hirano, 1971; all later models). In aggradation:

- ``"surface"`` (default): f_I,k = F_k, the layer passes down its own composition
  (Hirano, 1971). This is the relaxation above. In steady aggradation the surface then
  has the composition of the deposit.
- ``"bedload_mix"``: the classes are split into sand (d < 2 mm) and gravel. With the
  sand fraction F_s of the surface, the fractions Ft_j of gravel class j within the
  surface gravel, and the fractions pt_j of class j within the gravel of the load,

      f_I,j = (1 - f_Is) [ c pt_j + (1 - c) Ft_j ]              gravel class j,
      f_I,k = f_Is F_k / F_s                                    sand class k,

  with c = ``theta["c_interface"]`` the share of the load (Hoey and Ferguson, 1994,
  Water Resour. Res. 30(7), who write the weight on the surface; c = 0.7 from the flume
  experiments of Toro-Escobar et al., 1996, J. Hydraul. Res. 34(1), as given by Cui,
  2007, Water Resour. Res. 43, W10436, whose equations (7), (16b) and (16c) these are).
  f_Is is the sand fraction of what is passed down (``SedimentConfig.interface_sand``):
  ``"surface"``: f_Is = F_s; ``"cui"`` (Cui, 2007, eq. 8, for sigma < 4; F_s above):

      f_Is = 0.4 - 0.1 sigma + (0.6 + 0.1 sigma) F_s,

  with the geometric standard deviation sigma of the gravel that is passed down. The
  published relations have one sand class; here the sand that is passed down has the
  proportions of the surface sand. c = 0 with f_Is = F_s is the exchange ``"surface"``.

  With c > 0 the surface is no longer the deposit: in steady aggradation the gravel of
  the surface is Ft_j = (pd_j - c pt_j) / (1 - c), with pd_j the fractions within the
  gravel of the deposit, so that where the load is finer than the deposit the surface
  is coarser than the deposit (a mobile armour).

  Within a step the coefficients (F_s, f_Is, sigma, pt_j) are those at its start, which
  makes the layer equation linear, delta_a dp_k/dzeta = p_dep,k - A_k - r_k p_k, with
  A_k and r_k from the relations above; its exact solution over the deposit dz^+ is used,
  as for the relaxation. A class of which more would be passed down than is deposited
  and present is emptied (its fraction is floored at zero); the fractions are
  renormalized. The substrate receives D_k^+ - delta_a (change of p_a,k), the exact
  balance of the layer.

Units: SI.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
from jax import Array

from hydrograd.sediment.config import (
    INTERFACE_SAND_RULES,
    SAND_DMAX,
    PhysicalConstants,
    SedimentSmoothing,
)
from hydrograd.smooth import (
    expm1_ratio,
    normalize_fractions,
    sigmoid_eps,
    soft_min_one,
    softplus_beta,
)

__all__ = [
    "BedComposition",
    "InterfaceMix",
    "available_thickness",
    "interface_coefficients",
    "porosity",
    "update_composition",
]

# Differentiable parameters, e.g. {"delta_a": ..., "gamma1": ...}.
Theta = dict[str, Any]

# Guard of the deposit thickness dz^+ in the deposit composition (m).
THICKNESS_GUARD = 1.0e-30

# Width (m) of the aggradation/degradation switch of the explicit Hirano update.
EXPLICIT_SWITCH_WIDTH = 1.0e-4

# Interface exchange "bedload_mix": guards of the divisions by the gravel load, by the
# gravel fraction of the surface and by its sand fraction (-); below the last one the
# sand that is passed down goes to zero with the sand of the surface.
FRACTION_GUARD = 1.0e-12
SAND_FRACTION_GUARD = 1.0e-4

# Cui (2007), eq. (8): the relation holds for a geometric standard deviation of the
# gravel below this value; above it the sand fraction passed down is that of the surface.
CUI_SIGMA_MAX = 4.0


class BedComposition(NamedTuple):
    """Composition of the bed at N locations for K size classes.

    Attributes:
        active: fractions p_a,k of the active layer (-), shape [K, N].
        substrate: fractions p_sub,k of the substrate (-), shape [K, N].
        substrate_thickness: delta_sub (m), shape [N].
    """

    active: Array
    substrate: Array
    substrate_thickness: Array


class InterfaceMix(NamedTuple):
    """Data of the interface exchange ``"bedload_mix"`` (see the module docstring).

    Attributes:
        load: load of each class at the location (any unit; only the proportions are
            used), shape [K, N].
        diameters: grain diameter of each class (m), shape [K].
        bedload_share: share c of the load in the gravel that is passed down (-), scalar.
        sand_rule: ``"surface"`` or ``"cui"``.
    """

    load: Array
    diameters: Array
    bedload_share: Array
    sand_rule: str = "surface"


def interface_coefficients(p_active: Array, interface: InterfaceMix) -> tuple[Array, Array]:
    """Coefficients of the fractions passed down in aggradation, f_I,k = A_k + r_k p_a,k.

    Args:
        p_active: fractions of the active layer (-), shape [K, N].
        interface: load, diameters, share of the load and rule for the sand.

    Returns:
        Tuple ``(A, r)`` (-), each of shape [K, N]: the part of f_I,k that does not
        depend on the fraction of the class in the layer, and the factor of that
        fraction. Their dependence on the other fractions (through the sand fraction of
        the surface) is kept in derivatives.
    """
    if interface.sand_rule not in INTERFACE_SAND_RULES:
        raise ValueError(
            f"interface_sand must be one of {INTERFACE_SAND_RULES}, not '{interface.sand_rule}'"
        )
    diameters = jnp.asarray(interface.diameters)
    is_sand = (diameters < SAND_DMAX)[:, None]
    share = interface.bedload_share

    # Sand fraction of the surface, and the gravel classes as fractions of the gravel.
    sand_fraction = jnp.sum(jnp.where(is_sand, p_active, 0.0), axis=0, keepdims=True)
    gravel_fraction = jnp.maximum(1.0 - sand_fraction, FRACTION_GUARD)
    surface_gravel = jnp.where(is_sand, 0.0, p_active) / gravel_fraction

    # Gravel classes as fractions of the gravel of the load; without gravel in the load,
    # the gravel of the surface.
    load_gravel = jnp.where(is_sand, 0.0, jnp.maximum(interface.load, 0.0))
    load_total = jnp.sum(load_gravel, axis=0, keepdims=True)
    has_load = load_total > FRACTION_GUARD * jnp.max(load_total)
    load_fractions = jnp.where(
        has_load, load_gravel / jnp.where(has_load, load_total, 1.0), surface_gravel,
    )

    # Sand fraction of what is passed down.
    if interface.sand_rule == "cui":
        # Geometric standard deviation of the gravel that is passed down, in psi units.
        passed_gravel = share * load_fractions + (1.0 - share) * surface_gravel
        psi = jnp.log2(diameters)[:, None]
        mean = jnp.sum(passed_gravel * psi, axis=0, keepdims=True)
        variance = jnp.sum(passed_gravel * (psi - mean) ** 2, axis=0, keepdims=True)
        sigma = 2.0 ** jnp.sqrt(jnp.maximum(variance, 0.0) + FRACTION_GUARD)
        sigma = jnp.minimum(sigma, CUI_SIGMA_MAX)
        passed_sand = 0.4 - 0.1 * sigma + (0.6 + 0.1 * sigma) * sand_fraction
    else:
        passed_sand = sand_fraction

    # Sand classes: in the proportions of the surface sand; no sand is passed down from a
    # surface without sand.
    sand_rate = passed_sand / jnp.maximum(sand_fraction, SAND_FRACTION_GUARD)
    passed_sand = sand_rate * sand_fraction

    # Gravel classes: (1 - f_Is) [c pt_j + (1 - c) p_j / (1 - F_s)].
    offset = jnp.where(is_sand, 0.0, (1.0 - passed_sand) * share * load_fractions)
    gravel_rate = (1.0 - passed_sand) * (1.0 - share) / gravel_fraction
    rate = jnp.where(is_sand, sand_rate, gravel_rate)

    return offset, jnp.broadcast_to(rate, p_active.shape)


def porosity(theta: Theta, constants: PhysicalConstants) -> Array:
    """Porosity of a fresh deposit of each class, lambda_k = 1 - gamma1_k / rho_s.

    Args:
        theta: parameters with ``gamma1``, the initial dry bulk density (kg/m^3), shape [K].
        constants: physical constants (grain density rho_s).

    Returns:
        Porosity (-), shape [K].
    """
    return 1.0 - theta["gamma1"] / constants.rho_s


def available_thickness(
    bed_elevation: Array,
    floor_elevation: Array | None,
    delta_a: Array,
    smoothing: SedimentSmoothing,
) -> Array:
    """Thickness of bed the flow can erode within one step.

    The smaller of the active-layer thickness and the erodible thickness above a
    non-erodible floor (bedrock), both smoothed.

    Args:
        bed_elevation: bed elevation (m), shape [N]; the thalweg in 1-D.
        floor_elevation: elevation of the non-erodible floor (m), shape [N], or None for a
            fully erodible bed.
        delta_a: active-layer thickness (m), scalar.
        smoothing: smoothing widths (``beta_softplus``).

    Returns:
        Available thickness (m), shape [N].
    """
    # Fully erodible: one active layer everywhere.
    if floor_elevation is None:
        return jnp.broadcast_to(delta_a, bed_elevation.shape)

    # Erodible thickness above the floor: a softplus shifted so that it is exactly zero
    # on the floor. (A plain softplus would supply ln(2) / beta of sediment wherever the
    # bed sits on bedrock.)
    height = bed_elevation - floor_elevation
    beta = smoothing.beta_softplus
    shifted = softplus_beta(height, beta) - softplus_beta(jnp.zeros_like(height), beta)
    erodible = jnp.maximum(shifted, 0.0)

    # Smooth minimum of the erodible thickness and delta_a.
    return delta_a * soft_min_one(erodible / delta_a, beta)


def update_composition(
    composition: BedComposition,
    thickness_change: Array,
    delta_a: Array,
    smoothing: SedimentSmoothing,
    method: str = "relaxation",
    interface: InterfaceMix | None = None,
) -> BedComposition:
    """Update the active layer and the substrate for the thickness change of one step.

    Args:
        composition: composition before the step.
        thickness_change: bulk thickness change D_k of each class (m), positive for
            deposition, shape [K, N].
        delta_a: active-layer thickness (m), scalar.
        smoothing: smoothing widths (``eps_floor``, ``beta_softplus``).
        method: ``"relaxation"`` (default) or ``"explicit"`` (see the module docstring).
        interface: data of the interface exchange ``"bedload_mix"``, or None for the
            exchange ``"surface"``; only with ``method="relaxation"``.

    Returns:
        Composition after the step.
    """
    if interface is not None and method != "relaxation":
        raise ValueError("the interface exchange 'bedload_mix' needs the relaxation update")

    p_active = composition.active
    p_substrate = composition.substrate
    substrate_thickness = composition.substrate_thickness
    D = thickness_change
    dz = jnp.sum(D, axis=0)

    # Composition supplied from below in erosion: the substrate's, or the layer's own
    # where the substrate is (nearly) used up.
    has_substrate = sigmoid_eps(
        substrate_thickness - 3.0 * smoothing.eps_floor, smoothing.eps_floor,
    )[None, :]
    p_below = has_substrate * p_substrate + (1.0 - has_substrate) * p_active

    # Deposition and erosion parts of the thickness change.
    D_plus = jnp.maximum(D, 0.0)
    D_minus = jnp.minimum(D, 0.0)
    dz_plus = jnp.sum(D_plus, axis=0)[None, :]
    dz_minus = jnp.sum(D_minus, axis=0)[None, :]

    if method == "explicit":
        # Explicit Hirano update with a sharp interface switch (adjoint-unstable; kept
        # for the stability study).
        aggrading = sigmoid_eps(dz, EXPLICIT_SWITCH_WIDTH)[None, :]
        p_interface = aggrading * p_active + (1.0 - aggrading) * p_below
        p_after_erosion = p_active
        p_new = p_active + (D - p_interface * dz[None, :]) / delta_a
        p_active_new = normalize_fractions(p_new, axis=0)
    else:
        # Erosion: explicit, refilled from below.
        p_after_erosion = p_active + (D_minus - p_below * dz_minus) / delta_a

        p_deposit = D_plus / (dz_plus + THICKNESS_GUARD)
        if interface is None:
            # Deposition: exact relaxation to the composition of the deposit.
            relaxation = -jnp.expm1(-dz_plus / delta_a)
            p_new = p_after_erosion + (p_deposit - p_after_erosion) * relaxation
        else:
            # Deposition with f_I,k = A_k + r_k p_k: exact solution of
            # delta_a dp_k/dzeta = p_dep,k - A_k - r_k p_k over the deposit dz^+.
            offset, rate = interface_coefficients(p_after_erosion, interface)
            decay = rate * dz_plus / delta_a
            p_new = p_after_erosion * jnp.exp(-decay) + (
                (p_deposit - offset) * (dz_plus / delta_a) * expm1_ratio(decay)
            )
            p_new = jnp.maximum(p_new, 0.0)
        p_active_new = normalize_fractions(p_new, axis=0)

    # Material exchanged with the substrate: the layer passes its composition down in
    # deposition (with the interface exchange "bedload_mix": the exact balance of the
    # layer); erosion takes the composition from below.
    if interface is None:
        passed_down = p_after_erosion * dz_plus
    else:
        passed_down = jnp.maximum(D_plus - delta_a * (p_new - p_after_erosion), 0.0)
    exchange = passed_down + p_below * dz_minus

    # Substrate thickness (floored smoothly at eps_floor) and composition.
    thickness_new = smoothing.eps_floor + softplus_beta(
        substrate_thickness + dz - smoothing.eps_floor, smoothing.beta_softplus,
    )
    substrate_content = p_substrate * substrate_thickness[None, :] + exchange
    p_substrate_new = normalize_fractions(substrate_content / thickness_new[None, :], axis=0)

    return BedComposition(
        active=p_active_new,
        substrate=p_substrate_new,
        substrate_thickness=thickness_new,
    )
