"""Sediment closures: settling, suspended fraction, capacity, cohesive fluxes, adaptation.

All functions are pure and work on arrays over the size classes [K] and the locations [N].
A location is a cross-section in the 1-D mode and a mesh cell in 2-D. The functions know
nothing about cross-section geometry or meshes; the flow enters only through
:class:`FlowState`.

One formulation for 1-D and 2-D
-------------------------------
The closures are written for a stream tube of width B (m) that carries the discharge
Q (m^3/s):

- 1-D mode: B is the top width of the cross-section and Q its discharge. Capacities are
  volumes of solids per time, Q* (m^3/s).
- 2-D: set B = 1 and Q = |q|, the magnitude of the unit discharge (m^2/s). Every result is
  then per unit width: the capacity is q* (m^2/s), and the cohesive source is the erosion
  flux E (m/s).

Formulas (SRH-1D v4.0 manual, chapter 3; equation numbers refer to it)
---------------------------------------------------------------------
With R_s = rho_s / rho - 1, grain diameter d_k, bed shear stress tau_b, shear velocity
u_*, friction coefficient c_f = tau_b / (rho u^2) and the fraction p_k of class k in the
active layer of the bed:

Settling velocity (Ferguson and Church 2004, C_1 = 18, C_2 = 1), with a multiplier kappa_w:

    w_k = kappa_w,k R_s g d_k^2 / (C_1 nu + sqrt(0.75 C_2 R_s g d_k^3))            (m/s)

Suspended fraction of the total load (eq. 3.8), with the minimum smoothed:

    f_s,k = min(1, 2.5 exp(-Z_k)),      Z_k = w_k / (kappa u_*)                    (-)

Transport capacity of the non-cohesive classes, Engelund-Hansen (eq. 3.46) weighted by the
active-layer fraction (eq. 3.34) and a hiding factor:

    q*_k = p_k xi_k (a_eh / c_f) theta_k^b_eh sqrt(R_s g d_k^3),     Q*_k = B q*_k
    theta_k = tau_b / ((rho_s - rho) g d_k),    xi_k = (d_k / d_m)^(-m_h),
    d_m = sum_k p_k d_k

The defaults a_eh = 0.05 and b_eh = 2.5 are the original formula. The hiding factor is an
addition: SRH-1D applies a hiding correction to a critical shear stress (eq. 3.42), which
Engelund-Hansen does not have, so here it multiplies the capacity. m_h < 0 shelters the
fine classes, m_h > 0 exposes them, and m_h = 0 is SRH-1D. ``hiding_form = "two_power"``
uses the exponent ``m_h_fine`` for d_k < d_m and ``m_h_coarse`` for d_k > d_m.

Transport capacity of a sand-gravel mixture, Wilcock and Crowe (2003)
(``SedimentConfig.transport_formula = "wilcock_crowe"``), a surface-based bed-load formula
with a reference shear stress for every size:

    q*_k = a_wc p_k W*_k u_*^3 / (R_s g),     Q*_k = B q*_k
    W*_k = 0.002 phi_k^7.5                          for phi_k < 1.35
         = 14 (1 - 0.894 / sqrt(phi_k))^4.5         for phi_k >= 1.35
    phi_k = tau_b / tau_r,k,     tau_r,k = tau_r,m (d_k / d_g) / xi_k
    tau_r,m = tau_r_wc (0.021 + 0.015 exp(-20 F_s)) (rho_s - rho) g d_g

with the geometric mean size d_g and the sand fraction F_s (d_k < 2 mm) of the active
layer. a_wc = 1 and tau_r_wc = 1 are the original formula. The transport function W*
can be replaced (``SedimentConfig.transport_function``) by that of Parker (1990), scaled
to W* = 0.002 at phi = 1 as in Cui (2007, Water Resour. Res. 43, W10436, eq. 22),

    W*_k = 0.002 phi_k^14.2                                         for phi_k <= 1
         = 0.002 exp(14.2 (phi_k - 1) - 9.28 (phi_k - 1)^2)         for 1 < phi_k <= 1.59
         = 10.95 (1 - 0.853 / phi_k)^4.5                            for phi_k > 1.59

or by the function of Wilcock and Crowe times a learned factor,
W*_k = W*_WC(phi_k) exp(f(ln phi_k) - f(0)), with a small neural network f (parameters
``theta["transport_network"]``). The factor is 1 at phi = 1, so that the reference
stress keeps its meaning (the stress at which W* = 0.002); |f| <= ``transport_log_bound``. The hiding factor xi_k is the
same quantity as for Engelund-Hansen (it raises the mobility of class k, xi = 1 for a
grain of the mean size), here applied to the reference stress: xi = 1 for all classes is
no hiding (every size moves at its own Shields stress, tau_r,k proportional to d_k), and
xi_k = d_k / d_g is equal mobility (one reference stress for all sizes). The hiding
function of Wilcock and Crowe, tau_r,k / tau_r,m = (d_k / d_g)^b with
b = 0.67 / (1 + exp(1.5 - d_k / d_g)), is ``hiding_form = "wilcock_crowe"``:
xi_k = (d_k / d_g)^(1 - b). The power law (xi_k = (d_k / d_g)^(-m_h): m_h = 0 no hiding,
m_h = -1 equal mobility) and the network can be used instead. The mean size of the hiding
factor is d_g for this formula and d_m for Engelund-Hansen.

Learned hiding factor (``SedimentConfig.hiding_form = "network"``): the power law is
replaced by a small neural network f with the parameters ``theta["hiding_network"]``,

    ln xi_k = f(r_k) - f(0),      r_k = ln(d_k / d_m),

so that xi = 1 for a grain of the mean size, as for the power law (the level of the
capacity stays with a_eh). f is bounded, |f| <= ``hiding_log_bound``. With
``hiding_inputs = ("size_ratio", "shields")`` the network has a second input, the
logarithm of the Shields number of the mean size, theta_m = tau_b / ((rho_s - rho) g d_m),
and ln xi_k = f(r_k, ln theta_m) - f(0, ln theta_m). See :func:`hiding_factor`.

Adaptation lengths (eq. 3.6 and 3.7): suspended load, with the recovery coefficient alpha
(alpha_d in deposition, alpha_e in erosion), and bed load:

    L_s,k = Q / (alpha w_k B),      L_b = c_b h                                     (m)

Cohesive classes (d_k < cohesive_dmax), Krone deposition and Partheniades erosion
(eq. 3.89 and 3.90), written as a source S and a rate r of d Q_t / dx = S - r Q_t:

    S_k = B E_k,    E_k = p_k M_e max(tau_b / tau_ce - 1, 0)                        (m^2/s)
    r_k = B P_d w_k / Q,    P_d = max(1 - tau_b / tau_cd, 0)                        (1/m)

The two maxima are smoothed (see :class:`hydrograd.sediment.config.SedimentSmoothing`).

The combination of these quantities into the source and rate of every class is in
:mod:`hydrograd.sediment.exchange`.

Units: SI.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from hydrograd.closures.mlp import apply_mlp, init_mlp, normalize
from hydrograd.sediment.config import (
    SAND_DMAX,
    TRANSPORT_FUNCTIONS,
    PhysicalConstants,
    SedimentConfig,
)
from hydrograd.smooth import sigmoid_eps, soft_min_one, softplus_beta

__all__ = [
    "ClosureState",
    "FlowState",
    "cohesive_fluxes",
    "compute_closures",
    "engelund_hansen_capacity",
    "grain_size_percentile",
    "hiding_factor",
    "init_hiding_network",
    "init_transport_network",
    "settling_velocity",
    "suspended_adaptation_length",
    "suspended_fraction",
    "transport_capacity",
    "transport_function",
    "wilcock_crowe_capacity",
    "wilcock_crowe_stress_ratio",
]

# Differentiable closure parameters, e.g. {"a_eh": ..., "kappa_w": ..., "tau_cd": ...}.
Theta = dict[str, Any]

# Guard added to denominators that are positive in any wet location (units of the
# denominator). It only prevents a division by zero where there is no flow.
GUARD = 1.0e-12

# Floor of the mean grain diameter d_m (m) in the hiding factor.
MEAN_DIAMETER_FLOOR = 1.0e-9

# Wilcock and Crowe (2003): value of phi = tau_b / tau_r at which the two branches of the
# transport function W* meet. (The upper size of sand, SAND_DMAX, is in the settings.)
WC_PHI_BREAK = 1.35

# Parker (1990): value of phi at which the upper two branches of the transport function
# meet (the lower two meet at phi = 1).
PARKER_PHI_BREAK = 1.59


class FlowState(NamedTuple):
    """Flow quantities the sediment closures need, one value per location.

    In 2-D set ``width`` to 1 and ``discharge`` to the magnitude of the unit discharge
    (m^2/s); all closure results are then per unit width.

    Attributes:
        discharge: discharge Q (m^3/s), shape [N].
        width: width B of the stream tube (m), shape [N]; the top width in 1-D.
        depth: mean water depth h (m), shape [N]; A / B in 1-D.
        shear_stress: bed shear stress tau_b (Pa), shape [N].
        shear_velocity: shear velocity u_* = sqrt(tau_b / rho) (m/s), shape [N].
        friction_coefficient: c_f = tau_b / (rho u^2) (-), u the mean velocity, shape [N].
    """

    discharge: Array
    width: Array
    depth: Array
    shear_stress: Array
    shear_velocity: Array
    friction_coefficient: Array


class ClosureState(NamedTuple):
    """Closure results for K size classes at N locations.

    Attributes:
        settling_velocity: w_k (m/s), shape [K].
        suspended_fraction: f_s (-), shape [K, N].
        capacity: transport capacity Q* (m^3/s of solids), shape [K, N]; zero for the
            cohesive classes.
        capacity_per_fraction: Q*_k / p_k (m^3/s), the capacity per unit active-layer
            fraction, shape [K, N]; zero for the cohesive classes. The implicit bed
            feedback of the routing uses it.
        length_suspended_deposition: L_s with alpha_d (m), shape [K, N].
        length_suspended_erosion: L_s with alpha_e (m), shape [K, N].
        length_bedload: L_b (m), shape [N].
        cohesive_source: S = B E (m^2/s of solids), shape [K, N].
        cohesive_rate: r = B P_d w / Q (1/m), shape [K, N].
    """

    settling_velocity: Array
    suspended_fraction: Array
    capacity: Array
    capacity_per_fraction: Array
    length_suspended_deposition: Array
    length_suspended_erosion: Array
    length_bedload: Array
    cohesive_source: Array
    cohesive_rate: Array


def grain_size_percentile(diameters: Array, fractions: Array, percent: float) -> Array:
    """Grain size below which a given percentage of a mixture lies (D50, D90, ...).

    The diameters are the representative (geometric middle) sizes of the classes, in
    ascending order. The limit between two classes is the geometric mean of their
    diameters; the two outer limits lie as far from the end diameters (in ln d) as the
    neighbouring inner limits. Within a class the mixture is spread evenly in ln d, so the
    percentile is linear in ln d between the class limits.

    Args:
        diameters: grain diameters d_k (m), ascending, shape [K].
        fractions: fractions of the classes (-), shape [K, N]; need not sum to one.
        percent: percentage finer (0 to 100).

    Returns:
        The grain size (m), shape [N].
    """
    n_classes = diameters.shape[0]
    if n_classes == 1:
        return jnp.broadcast_to(diameters, (fractions.shape[1],))
    log_d = jnp.log(diameters)
    inner = 0.5 * (log_d[:-1] + log_d[1:])
    limits = jnp.concatenate([2.0 * log_d[:1] - inner[:1], inner, 2.0 * log_d[-1:] - inner[-1:]])
    share = fractions / (jnp.sum(fractions, axis=0, keepdims=True) + GUARD)
    cumulative = jnp.concatenate([jnp.zeros_like(share[:1]), jnp.cumsum(share, axis=0)], axis=0)
    # A small ramp keeps the cumulative curve increasing where a class is empty.
    cumulative = cumulative + 1.0e-9 * jnp.arange(n_classes + 1, dtype=cumulative.dtype)[:, None]
    log_size = jax.vmap(lambda curve: jnp.interp(percent / 100.0, curve, limits), in_axes=1)(cumulative)
    return jnp.exp(log_size)


def settling_velocity(diameters: Array, kappa_w: Array, constants: PhysicalConstants) -> Array:
    """Settling velocity of Ferguson and Church (2004), times a multiplier per class.

    Args:
        diameters: grain diameters d_k (m), shape [K].
        kappa_w: multiplier of each class (-), shape [K] or scalar; 1 is the formula.
        constants: physical constants (g, densities, nu).

    Returns:
        Settling velocity w_k (m/s), shape [K].
    """
    Rs = constants.submerged_specific_gravity
    g = constants.g

    # Constants of the formula for natural grains: C_1 (laminar) and C_2 (turbulent drag).
    C1 = 18.0
    C2 = 1.0

    # w = R_s g d^2 / (C_1 nu + sqrt(0.75 C_2 R_s g d^3)): Stokes for fine grains,
    # constant drag coefficient for coarse grains.
    viscous_term = C1 * constants.nu
    inertial_term = jnp.sqrt(0.75 * C2 * Rs * g * diameters**3)
    velocity = Rs * g * diameters * diameters / (viscous_term + inertial_term)

    return kappa_w * velocity


def suspended_fraction(
    velocity_settling: Array, shear_velocity: Array, config: SedimentConfig,
) -> Array:
    """Fraction of the total load that travels in suspension (SRH-1D eq. 3.8, smoothed).

    f_s = min(1, 2.5 exp(-Z)) with the Rouse-type number Z = w / (kappa u_*).

    Args:
        velocity_settling: settling velocity w_k (m/s), shape [K].
        shear_velocity: shear velocity u_* (m/s), shape [N].
        config: sediment settings (von Karman constant, smoothing sharpness).

    Returns:
        f_s (-), shape [K, N].
    """
    kappa = config.constants.kappa

    # Z = w / (kappa u_*); the guard keeps Z finite where there is no flow (u_* = 0).
    Z = velocity_settling[:, None] / (kappa * shear_velocity[None, :] + GUARD)

    return soft_min_one(2.5 * jnp.exp(-Z), config.smoothing.beta_softplus)


def init_hiding_network(key: Array, config: SedimentConfig) -> list[dict[str, Array]]:
    """Initial parameters of the hiding network (``theta["hiding_network"]``).

    The output layer starts at zero, so the initial hiding factor is 1 for all classes.

    Args:
        key: JAX PRNG key.
        config: sediment settings (inputs and hidden widths of the network).

    Returns:
        Parameters of the network: one dictionary per layer (weights ``W``, biases ``b``).
    """
    params = init_mlp(key, len(config.hiding_inputs), config.hiding_hidden, 1, layernorm=False)
    params[-1]["W"] = jnp.zeros_like(params[-1]["W"])
    return params


def hiding_factor(
    diameters: Array, mean_diameter: Array, shear_stress: Array, theta: Theta, config: SedimentConfig,
) -> Array:
    """Hiding factor xi_k of every class: the power law or the network.

    Args:
        diameters: grain diameters d_k (m), shape [K].
        mean_diameter: mean diameter of the active layer (m), shape [N]: the arithmetic
            mean d_m for Engelund-Hansen, the geometric mean d_g for Wilcock-Crowe.
        shear_stress: bed shear stress tau_b (Pa), shape [N]; used only by a network with
            the input ``"shields"``.
        theta: closure parameters: ``m_h`` (-) for the power law, ``m_h_fine`` and
            ``m_h_coarse`` (-) for the two power laws, ``hiding_network`` for the network;
            none for the hiding function of Wilcock and Crowe.
        config: sediment settings (form of the hiding factor, settings of the network).

    Returns:
        xi (-), shape [K, N]; 1 for a grain of the mean size.
    """
    ratio = diameters[:, None] / mean_diameter[None, :]
    if config.hiding_form == "power":
        return ratio ** (-theta["m_h"])
    if config.hiding_form == "wilcock_crowe":
        # tau_r,k / tau_r,m = ratio^b, written as the mobility factor xi = ratio^(1 - b).
        exponent = 0.67 / (1.0 + jnp.exp(1.5 - ratio))
        return ratio ** (1.0 - exponent)
    if config.hiding_form == "two_power":
        # One exponent for the classes finer than the mean size, one for the coarser.
        log_ratio = jnp.log(ratio)
        exponent = jnp.where(log_ratio < 0.0, theta["m_h_fine"], theta["m_h_coarse"])
        return jnp.exp(-exponent * log_ratio)
    if config.hiding_form != "network":
        raise ValueError(f"Unknown hiding form {config.hiding_form!r}")

    # Inputs of the network, each mapped from its range to [-1, 1]: ln(d_k / d_m), and
    # (optional) the logarithm of the Shields number of the mean size.
    log_ratio = jnp.log(ratio)                                              # [K, N]
    low, high = config.hiding_size_ratio_range
    features = [normalize(log_ratio, jnp.log(low), jnp.log(high))]
    reference = [normalize(jnp.zeros_like(log_ratio), jnp.log(low), jnp.log(high))]
    if "shields" in config.hiding_inputs:
        constants = config.constants
        shields = shear_stress / ((constants.rho_s - constants.rho) * constants.g * mean_diameter)
        low, high = config.hiding_shields_range
        log_shields = jnp.log(jnp.clip(shields, low, high))
        feature = jnp.broadcast_to(normalize(log_shields, jnp.log(low), jnp.log(high)), log_ratio.shape)
        features.append(feature)
        reference.append(feature)

    def network(inputs: list[Array]) -> Array:
        x = jnp.stack([item.reshape(-1) for item in inputs], axis=1)        # [K N, inputs]
        bound = config.hiding_log_bound
        out = apply_mlp(theta["hiding_network"], x, config.hiding_activations, -bound, bound)
        return out.reshape(log_ratio.shape)

    # ln xi = f(r) - f(0): no hiding for a grain of the mean size.
    return jnp.exp(network(features) - network(reference))


def engelund_hansen_capacity(
    flow: FlowState, diameters: Array, fractions: Array, theta: Theta, config: SedimentConfig,
) -> Array:
    """Engelund-Hansen capacity of every class, with active-layer fraction and hiding.

    Q*_k = B p_k xi_k (a_eh / c_f) theta_k^b_eh sqrt(R_s g d_k^3); see the module
    docstring. The result is not yet set to zero for the cohesive classes
    (:func:`transport_capacity` does that).

    Args:
        flow: flow quantities at the N locations.
        diameters: grain diameters d_k (m), shape [K].
        fractions: active-layer fractions p_k (-), shape [K, N].
        theta: closure parameters ``a_eh`` (-), ``b_eh`` (-) and ``m_h`` (-) (or
            ``hiding_network``, see :func:`hiding_factor`).
        config: sediment settings (physical constants, form of the hiding factor).

    Returns:
        Q* (m^3/s of solids), shape [K, N].
    """
    constants = config.constants
    Rs = constants.submerged_specific_gravity
    g = constants.g
    d = diameters[:, None]

    # Shields number of each class, theta_k = tau_b / ((rho_s - rho) g d_k), shape [K, N].
    shields = flow.shear_stress[None, :] / ((constants.rho_s - constants.rho) * g * d)

    # theta_k^b_eh. Where tau_b = 0 (a dry cell in 2-D) the power is set to zero directly,
    # because the derivative of 0^b with respect to b is not finite.
    is_sheared = shields > 0.0
    shields_safe = jnp.where(is_sheared, shields, 1.0)
    shields_power = jnp.where(is_sheared, shields_safe ** theta["b_eh"], 0.0)

    # Hiding factor xi_k (power law or network), d_m the mean diameter of the active layer.
    mean_diameter = jnp.sum(fractions * d, axis=0)
    mean_diameter = jnp.maximum(mean_diameter, MEAN_DIAMETER_FLOOR)
    hiding = hiding_factor(diameters, mean_diameter, flow.shear_stress, theta, config)

    # Capacity per unit width q*_k (m^2/s).
    coefficient = theta["a_eh"] / (flow.friction_coefficient[None, :] + GUARD)
    grain_scale = jnp.sqrt(Rs * g * d**3)
    capacity_unit_width = fractions * hiding * coefficient * shields_power * grain_scale

    return flow.width[None, :] * capacity_unit_width


def init_transport_network(key: Array, config: SedimentConfig) -> list[dict[str, Array]]:
    """Initial parameters of the transport-function network (``theta["transport_network"]``).

    The output layer starts at zero, so the initial transport function is that of
    Wilcock and Crowe.

    Args:
        key: JAX PRNG key.
        config: sediment settings (hidden widths of the network).

    Returns:
        Parameters of the network: one dictionary per layer (weights ``W``, biases ``b``).
    """
    params = init_mlp(key, 1, config.transport_hidden, 1, layernorm=False)
    params[-1]["W"] = jnp.zeros_like(params[-1]["W"])
    return params


def transport_function(phi: Array, theta: Theta, config: SedimentConfig) -> Array:
    """Transport function W*(phi) of the Wilcock-Crowe formula (see the module docstring).

    Args:
        phi: ratio of the bed shear stress to the reference shear stress of the class
            (-), any shape, not negative.
        theta: ``transport_network`` for ``SedimentConfig.transport_function = "network"``;
            not used otherwise.
        config: sediment settings (which function, settings of the network).

    Returns:
        W* (-), same shape as ``phi``; 0.002 at phi = 1 for every choice.
    """
    if config.transport_function not in TRANSPORT_FUNCTIONS:
        raise ValueError(
            f"transport_function must be one of {TRANSPORT_FUNCTIONS}, "
            f"not '{config.transport_function}'"
        )

    # Each branch is evaluated with an argument of its own range, so that the unused
    # branches have finite values and derivatives.
    if config.transport_function == "parker":
        is_low = phi <= 1.0
        is_high = phi > PARKER_PHI_BREAK
        phi_low = jnp.where(is_low, phi, 1.0)
        excess = jnp.clip(phi, 1.0, PARKER_PHI_BREAK) - 1.0
        phi_high = jnp.where(is_high, phi, PARKER_PHI_BREAK)
        low = 0.002 * phi_low**14.2
        middle = 0.002 * jnp.exp(14.2 * excess - 9.28 * excess * excess)
        high = 10.95 * (1.0 - 0.853 / phi_high) ** 4.5
        return jnp.where(is_low, low, jnp.where(is_high, high, middle))

    is_low = phi < WC_PHI_BREAK
    phi_low = jnp.where(is_low, phi, 1.0)
    phi_high = jnp.where(is_low, WC_PHI_BREAK, phi)
    base = jnp.where(
        is_low, 0.002 * phi_low**7.5, 14.0 * (1.0 - 0.894 / jnp.sqrt(phi_high)) ** 4.5,
    )
    if config.transport_function == "wilcock_crowe":
        return base

    # Learned factor exp(f(ln phi) - f(0)) on the function of Wilcock and Crowe.
    low, high = config.transport_stress_ratio_range
    bound = config.transport_log_bound

    def network(ratio: Array) -> Array:
        x = normalize(jnp.log(jnp.clip(ratio, low, high)), jnp.log(low), jnp.log(high))
        out = apply_mlp(
            theta["transport_network"], x.reshape(-1, 1), config.transport_activations,
            -bound, bound,
        )
        return out.reshape(ratio.shape)

    return base * jnp.exp(network(phi) - network(jnp.ones_like(phi)))


def wilcock_crowe_stress_ratio(
    flow: FlowState, diameters: Array, fractions: Array, theta: Theta, config: SedimentConfig,
) -> Array:
    """Ratio phi_k = tau_b / tau_r,k of the bed shear stress to the reference shear stress
    of every class in the Wilcock-Crowe (2003) formula.

    See the module docstring. The mean size and the sand fraction are taken over all
    classes of the active layer.

    Args:
        flow: flow quantities at the N locations.
        diameters: grain diameters d_k (m), shape [K].
        fractions: active-layer fractions p_k (-), shape [K, N].
        theta: closure parameters ``tau_r_wc`` (-) and those of the hiding factor (see
            :func:`hiding_factor`).
        config: sediment settings (physical constants, form of the hiding factor).

    Returns:
        phi (-), shape [K, N].
    """
    constants = config.constants
    g = constants.g
    d = diameters[:, None]

    # Geometric mean size and sand fraction of the active layer, shape [N].
    total = jnp.sum(fractions, axis=0) + GUARD
    mean_diameter = jnp.exp(jnp.sum(fractions * jnp.log(d), axis=0) / total)
    mean_diameter = jnp.maximum(mean_diameter, MEAN_DIAMETER_FLOOR)
    is_sand = (diameters < SAND_DMAX)[:, None]
    sand_fraction = jnp.sum(jnp.where(is_sand, fractions, 0.0), axis=0) / total

    # Reference shear stress of the mean size and of each class, shape [N] and [K, N].
    shields_reference = theta["tau_r_wc"] * (0.021 + 0.015 * jnp.exp(-20.0 * sand_fraction))
    tau_reference_mean = shields_reference * (constants.rho_s - constants.rho) * g * mean_diameter
    hiding = hiding_factor(diameters, mean_diameter, flow.shear_stress, theta, config)
    tau_reference = tau_reference_mean[None, :] * (d / mean_diameter[None, :]) / hiding

    return flow.shear_stress[None, :] / tau_reference


def wilcock_crowe_capacity(
    flow: FlowState, diameters: Array, fractions: Array, theta: Theta, config: SedimentConfig,
) -> Array:
    """Wilcock-Crowe (2003) capacity of every class of a sand-gravel mixture.

    See the module docstring. The result is not yet set to zero for the cohesive classes
    (:func:`transport_capacity` does that).

    Args:
        flow: flow quantities at the N locations.
        diameters: grain diameters d_k (m), shape [K].
        fractions: active-layer fractions p_k (-), shape [K, N].
        theta: closure parameters ``a_wc`` (-), ``tau_r_wc`` (-), those of the hiding
            factor (see :func:`hiding_factor`) and of a learned transport function (see
            :func:`transport_function`).
        config: sediment settings (physical constants, form of the hiding factor,
            transport function).

    Returns:
        Q* (m^3/s of solids), shape [K, N].
    """
    constants = config.constants
    Rs = constants.submerged_specific_gravity
    g = constants.g

    # Transport function W*(phi).
    phi = wilcock_crowe_stress_ratio(flow, diameters, fractions, theta, config)
    transport_rate = transport_function(phi, theta, config)

    capacity_unit_width = theta["a_wc"] * fractions * transport_rate * flow.shear_velocity[None, :] ** 3 / (Rs * g)

    return flow.width[None, :] * capacity_unit_width


def transport_capacity(
    flow: FlowState, fractions: Array, theta: Theta, config: SedimentConfig,
) -> Array:
    """Transport capacity Q* of every class: the chosen formula, zero for cohesive classes.

    The 1-D routing also calls this for a flow state with the bed shifted, to get the
    sensitivity of the capacity to the bed elevation.

    Args:
        flow: flow quantities at the N locations.
        fractions: active-layer fractions p_k (-), shape [K, N].
        theta: closure parameters of the chosen formula: ``a_eh``, ``b_eh`` (Engelund-Hansen)
            or ``a_wc``, ``tau_r_wc`` (Wilcock-Crowe), and those of the hiding factor.
        config: sediment settings (size classes, formula choice).

    Returns:
        Q* (m^3/s of solids), shape [K, N].
    """
    if config.transport_formula == "engelund_hansen":
        capacity = engelund_hansen_capacity(flow, config.diameter_array(), fractions, theta, config)
    elif config.transport_formula == "wilcock_crowe":
        capacity = wilcock_crowe_capacity(flow, config.diameter_array(), fractions, theta, config)
    else:
        raise ValueError(f"Unknown transport formula {config.transport_formula!r}")

    # Cohesive classes have no capacity; their exchange is by erosion and deposition fluxes.
    is_cohesive = config.cohesive_mask()[:, None]
    return jnp.where(is_cohesive, 0.0, capacity)


def suspended_adaptation_length(
    flow: FlowState, velocity_settling: Array, recovery_coefficient: Array,
) -> Array:
    """Adaptation length of the suspended load, L_s = Q / (alpha w B) (SRH-1D eq. 3.6).

    Args:
        flow: flow quantities at the N locations.
        velocity_settling: settling velocity w_k (m/s), shape [K].
        recovery_coefficient: alpha (-), scalar: ``alpha_d`` or ``alpha_e``.

    Returns:
        L_s (m), shape [K, N].
    """
    settling_width = recovery_coefficient * velocity_settling[:, None] * flow.width[None, :]

    return flow.discharge[None, :] / (settling_width + GUARD)


def cohesive_fluxes(
    flow: FlowState,
    fractions: Array,
    velocity_settling: Array,
    theta: Theta,
    config: SedimentConfig,
) -> tuple[Array, Array]:
    """Erosion source and deposition rate of the cohesive classes.

    The values are computed for every class; the caller uses them for the cohesive ones.

    Args:
        flow: flow quantities at the N locations.
        fractions: active-layer fractions p_k (-), shape [K, N].
        velocity_settling: settling velocity w_k (m/s), shape [K].
        theta: closure parameters ``M_e`` (m/s), ``tau_ce`` (Pa) and ``tau_cd`` (Pa).
        config: sediment settings (Krone form, smoothing widths).

    Returns:
        Tuple ``(source, rate)``: S = B E (m^2/s of solids) and r = B P_d w / Q (1/m),
        each of shape [K, N].
    """
    smoothing = config.smoothing
    shear_stress = flow.shear_stress[None, :]
    width = flow.width[None, :]

    # Krone deposition probability P_d. "ramp" is 1 - tau_b / tau_cd, a linear decrease to
    # zero at tau_cd (SRH-1D eq. 3.90), with only the kink at tau_cd smoothed. "step" is a
    # smoothed indicator of tau_b < tau_cd.
    deposition_deficit = 1.0 - shear_stress / theta["tau_cd"]
    if config.krone_form == "step":
        probability = sigmoid_eps(deposition_deficit, smoothing.eps_pd)
    else:
        probability = softplus_beta(deposition_deficit, 1.0 / smoothing.eps_pd)

    # Partheniades erosion flux E = p M_e max(tau_b / tau_ce - 1, 0) (m/s), smoothed.
    erosion_excess = softplus_beta(shear_stress / theta["tau_ce"] - 1.0, smoothing.beta_softplus)
    erosion_flux = fractions * theta["M_e"] * erosion_excess

    # Source and rate of d Q_t / dx = S - r Q_t.
    source = width * erosion_flux
    rate = width * probability * velocity_settling[:, None] / (flow.discharge[None, :] + GUARD)

    return source, rate


def compute_closures(
    flow: FlowState, fractions: Array, theta: Theta, config: SedimentConfig,
) -> ClosureState:
    """Evaluate all closures at the N locations.

    Args:
        flow: flow quantities at the N locations.
        fractions: active-layer fractions p_k (-), shape [K, N].
        theta: closure parameters (see :mod:`hydrograd.sediment.config`).
        config: sediment settings.

    Returns:
        The closure results (:class:`ClosureState`).
    """
    is_cohesive = config.cohesive_mask()[:, None]

    # Settling velocity [K] and suspended fraction [K, N].
    w = settling_velocity(config.diameter_array(), theta["kappa_w"], config.constants)
    f_s = suspended_fraction(w, flow.shear_velocity, config)

    # Transport capacity (zero for cohesive classes) and capacity per unit fraction.
    capacity = transport_capacity(flow, fractions, theta, config)
    capacity_per_fraction = capacity / jnp.maximum(fractions, GUARD)
    capacity_per_fraction = jnp.where(is_cohesive, 0.0, capacity_per_fraction)

    # Adaptation lengths: suspended load in deposition and in erosion, and bed load.
    length_deposition = suspended_adaptation_length(flow, w, theta["alpha_d"])
    length_erosion = suspended_adaptation_length(flow, w, theta["alpha_e"])
    length_bedload = theta["c_b"] * flow.depth

    # Cohesive erosion source and deposition rate.
    cohesive_source, cohesive_rate = cohesive_fluxes(flow, fractions, w, theta, config)

    return ClosureState(
        settling_velocity=w,
        suspended_fraction=f_s,
        capacity=capacity,
        capacity_per_fraction=capacity_per_fraction,
        length_suspended_deposition=length_deposition,
        length_suspended_erosion=length_erosion,
        length_bedload=length_bedload,
        cohesive_source=cohesive_source,
        cohesive_rate=cohesive_rate,
    )
