"""Settings and default parameters of the sediment model.

Everything here is independent of the flow model: it is used unchanged by the 1-D mode
(cross-sections) and is meant to be used unchanged by 2-D sediment transport (mesh cells).

Three frozen dataclasses hold the settings that are *not* calibrated:

- :class:`PhysicalConstants`: gravity, densities, viscosity, the von Karman constant;
- :class:`SedimentSmoothing`: the widths of the smooth switches inside the closures;
- :class:`SedimentConfig`: the size classes and the choice of formulas, plus the two above.

The calibrated quantities are plain arrays in a dictionary ``theta`` (the convention of the
rest of Hydrograd). :func:`default_sediment_parameters` returns the defaults:

    name      default  units       meaning
    a_eh      0.05     -           Engelund-Hansen coefficient
    b_eh      2.5      -           Engelund-Hansen exponent of the Shields number
    m_h       0.0      -           hiding exponent, xi_k = (d_k / d_m)^(-m_h)
    kappa_w   1.0 [K]  -           multiplier of the settling velocity of each class
    c_b       7.0      -           bed-load adaptation length L_b = c_b h
    alpha_d   0.25     -           recovery coefficient in deposition
    alpha_e   1.0      -           recovery coefficient in erosion
    M_e       1e-8     m/s         cohesive surface erosion rate (volume of solids)
    tau_ce    0.5      Pa          critical shear stress for cohesive erosion
    tau_cd    0.1      Pa          critical shear stress for full cohesive deposition

:func:`default_bed_parameters` returns the parameters of the bed layers and of the
consolidation of deposits:

    name      default        units                  meaning
    delta_a   0.05           m                      active-layer thickness
    gamma1    800, 1500 [K]  kg/m^3                 initial dry bulk density (first class
                                                    800, the others 1500)
    K_cons    250, 1 [K]     kg/m^3 per log10(yr)   Lane-Koelzer consolidation coefficient
                                                    (first class 250, the others 1)

[K] marks one value per size class.

Units: SI only (m, s, kg, Pa). The 2-D solver can also run in US customary units; the
sediment formulas contain dimensional constants and are written for SI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
from jax import Array

__all__ = [
    "INTERFACE_EXCHANGES",
    "INTERFACE_SAND_RULES",
    "SAND_DMAX",
    "TRANSPORT_FUNCTIONS",
    "PhysicalConstants",
    "SedimentConfig",
    "SedimentSmoothing",
    "default_bed_parameters",
    "default_sediment_parameters",
    "interface_parameters",
    "wilcock_crowe_parameters",
]

# Classes finer than this diameter (m) are sand; the others are gravel.
SAND_DMAX = 2.0e-3

# Transport functions W*(phi) of the Wilcock-Crowe formula (see :class:`SedimentConfig`).
TRANSPORT_FUNCTIONS = ("wilcock_crowe", "parker", "network")

# What the active layer passes down to the substrate when the bed aggrades, and the rule
# for the sand part of it (see :class:`SedimentConfig` and :mod:`hydrograd.sediment.bed`).
INTERFACE_EXCHANGES = ("surface", "bedload_mix")
INTERFACE_SAND_RULES = ("surface", "cui")

# Names of the calibrated closure parameters, in the order of the table above.
CLOSURE_PARAMETER_NAMES = (
    "a_eh", "b_eh", "m_h", "kappa_w", "c_b", "alpha_d", "alpha_e", "M_e", "tau_ce", "tau_cd",
)


@dataclass(frozen=True)
class PhysicalConstants:
    """Physical constants of water and sediment (SI).

    Attributes:
        g: gravitational acceleration (m/s^2).
        rho: density of water (kg/m^3).
        rho_s: density of the sediment grains (kg/m^3).
        nu: kinematic viscosity of water (m^2/s).
        kappa: von Karman constant (-).
        seconds_per_year: length of a year (s), for consolidation ages.
    """

    g: float = 9.81
    rho: float = 1000.0
    rho_s: float = 2650.0
    nu: float = 1.0e-6
    kappa: float = 0.41
    seconds_per_year: float = 365.25 * 86400.0

    @property
    def submerged_specific_gravity(self) -> float:
        """R_s = rho_s / rho - 1 (-), 1.65 for quartz in water."""
        return self.rho_s / self.rho - 1.0


@dataclass(frozen=True)
class SedimentSmoothing:
    """Widths of the smooth switches in the sediment closures.

    Attributes:
        beta_softplus: sharpness (-) of the smooth max and min applied to dimensionless
            arguments: the cap of the suspended fraction at 1 and the onset of cohesive
            erosion in tau_b / tau_ce - 1. The transition width is about 1 / beta.
        eps_switch: width (-) of the deposition/erosion switch in (Q_t - Q*) / (Q_t + Q*),
            which selects the recovery coefficient (alpha_d or alpha_e).
        eps_pd: width (-) of the smoothed kink of the Krone deposition probability, in
            tau_b / tau_cd.
        eps_floor: floor (m) on the thickness of the bed substrate (used by the bed
            update, not by the closures).
    """

    beta_softplus: float = 50.0
    eps_switch: float = 0.5
    eps_pd: float = 0.1
    eps_floor: float = 0.01


@dataclass(frozen=True)
class SedimentConfig:
    """Size classes and formula choices of the sediment model.

    Attributes:
        diameters: representative grain diameter d_k of each size class (m), finest first.
        cohesive_dmax: classes finer than this diameter (m) are cohesive (silt and clay;
            0.0625 mm is the SRH-1D default). They are transported by the Krone and
            Partheniades deposition and erosion fluxes instead of a transport capacity.
        transport_formula: capacity formula of the non-cohesive classes:
            ``"engelund_hansen"`` (total load of sand) or ``"wilcock_crowe"`` (bed load of
            sand-gravel mixtures; needs the parameters of
            :func:`wilcock_crowe_parameters`).
        krone_form: deposition probability P_d of the cohesive classes. ``"ramp"`` is
            P_d = 1 - tau_b / tau_cd (SRH-1D eq. 3.90); ``"step"`` is a smoothed indicator
            of tau_b < tau_cd (kept for ablation studies).
        hiding_form: hiding factor xi_k of the capacity. ``"power"`` is
            xi_k = (d_k / d_m)^(-m_h); ``"two_power"`` is the same with the exponent
            ``m_h_fine`` for d_k < d_m and ``m_h_coarse`` for d_k > d_m;
            ``"wilcock_crowe"`` is the hiding function of Wilcock and Crowe (2003) (for
            their formula); ``"network"`` is a neural network with the
            parameters ``theta["hiding_network"]`` (see
            :func:`hydrograd.sediment.closures.hiding_factor`).
        hiding_inputs: inputs of the network: ``"size_ratio"`` (ln(d_k / d_m)), and
            optionally ``"shields"`` (logarithm of the Shields number of the mean size).
        hiding_hidden: widths of the hidden layers of the network.
        hiding_activations: activation of each hidden layer.
        hiding_log_bound: bound of the network output; |ln xi_k| is at most twice this.
        hiding_size_ratio_range: range of d_k / d_m that is mapped to [-1, 1].
        hiding_shields_range: range of the Shields number that is mapped to [-1, 1];
            values outside are clipped.
        transport_function: transport function W*(phi) of the Wilcock-Crowe formula, the
            dimensionless transport rate of a class as a function of phi = tau_b / tau_r,k.
            ``"wilcock_crowe"`` (default): the function of Wilcock and Crowe (2003).
            ``"parker"``: the function of Parker (1990) scaled to the same value at
            phi = 1, as used by Cui (2007) in the Wilcock-Crowe formula; it is steeper
            and gives a four to six times larger transport rate for phi of 2 to 4.
            ``"network"``: the function of Wilcock and Crowe times
            exp(f(ln phi) - f(0)) with a neural network f, the parameters
            ``theta["transport_network"]`` (see
            :func:`hydrograd.sediment.closures.transport_function`).
        transport_hidden: widths of the hidden layers of that network.
        transport_activations: activation of each hidden layer.
        transport_log_bound: bound of the network output; the logarithm of the factor on
            W* is at most twice this in magnitude.
        transport_stress_ratio_range: range of phi that is mapped to [-1, 1]; values
            outside are clipped.
        interface_exchange: what the active layer passes down to the substrate when the
            bed aggrades. ``"surface"`` (default): its own composition (Hirano, 1971).
            ``"bedload_mix"``: for the gravel classes a mixture of the load and the
            surface, with the share ``theta["c_interface"]`` of the load (Hoey and
            Ferguson, 1994; Toro-Escobar et al., 1996); needs the parameter of
            :func:`interface_parameters`. See :mod:`hydrograd.sediment.bed`.
        interface_sand: sand fraction (classes finer than 2 mm) of what is passed down,
            with ``interface_exchange = "bedload_mix"``. ``"surface"`` (default): the sand
            fraction of the active layer. ``"cui"``: the relation of Cui (2007), in which
            the deposit is sandier than the surface.
        constants: physical constants.
        smoothing: widths of the smooth switches.
    """

    diameters: tuple[float, ...] = (2.0e-5, 1.0e-4, 5.0e-4)
    cohesive_dmax: float = 6.25e-5
    transport_formula: str = "engelund_hansen"
    krone_form: str = "ramp"
    hiding_form: str = "power"
    hiding_inputs: tuple[str, ...] = ("size_ratio",)
    hiding_hidden: tuple[int, ...] = (8, 8)
    hiding_activations: tuple[str, ...] = ("tanh", "tanh")
    hiding_log_bound: float = 3.0
    hiding_size_ratio_range: tuple[float, float] = (0.01, 100.0)
    hiding_shields_range: tuple[float, float] = (0.01, 100.0)
    transport_function: str = "wilcock_crowe"
    transport_hidden: tuple[int, ...] = (8, 8)
    transport_activations: tuple[str, ...] = ("tanh", "tanh")
    transport_log_bound: float = 3.0
    transport_stress_ratio_range: tuple[float, float] = (0.1, 100.0)
    interface_exchange: str = "surface"
    interface_sand: str = "surface"
    constants: PhysicalConstants = field(default_factory=PhysicalConstants)
    smoothing: SedimentSmoothing = field(default_factory=SedimentSmoothing)

    @property
    def n_classes(self) -> int:
        """Number of size classes K."""
        return len(self.diameters)

    def diameter_array(self) -> Array:
        """Grain diameters d_k (m) as an array of shape [n_classes]."""
        return jnp.asarray(self.diameters)

    def cohesive_mask(self) -> np.ndarray:
        """Boolean mask of the cohesive classes (d_k < cohesive_dmax), shape [n_classes].

        A NumPy array, so it is a constant (not a traced value) inside jitted functions.
        """
        diameters = np.asarray(self.diameters)
        return diameters < self.cohesive_dmax


def default_sediment_parameters(n_classes: int) -> dict[str, Array]:
    """Default values of the calibrated closure parameters (see the module docstring).

    Args:
        n_classes: number of size classes K.

    Returns:
        Dictionary ``theta`` of physical (constrained) values. ``kappa_w`` has shape
        [n_classes]; all other entries are scalars.
    """
    return {
        "a_eh": jnp.asarray(0.05),
        "b_eh": jnp.asarray(2.5),
        "m_h": jnp.asarray(0.0),
        "kappa_w": jnp.ones(n_classes),
        "c_b": jnp.asarray(7.0),
        "alpha_d": jnp.asarray(0.25),
        "alpha_e": jnp.asarray(1.0),
        "M_e": jnp.asarray(1.0e-8),
        "tau_ce": jnp.asarray(0.5),
        "tau_cd": jnp.asarray(0.1),
    }


def wilcock_crowe_parameters() -> dict[str, Array]:
    """Default values of the parameters of the Wilcock-Crowe formula (the original formula).

    Returns:
        ``a_wc``: multiplier of the transport function W* (-); ``tau_r_wc``: multiplier of
        the reference Shields stress of the mean size (-). Add them to ``theta`` when
        ``SedimentConfig.transport_formula = "wilcock_crowe"``.
    """
    return {"a_wc": jnp.asarray(1.0), "tau_r_wc": jnp.asarray(1.0)}


def interface_parameters() -> dict[str, Array]:
    """Default value of the parameter of the interface exchange ``"bedload_mix"``.

    Returns:
        ``c_interface``: share (-, 0 to 1) of the load in the gravel that the active
        layer passes down to the substrate when the bed aggrades; the rest is surface
        material. 0.7 is the value of Toro-Escobar et al. (1996) for a gravel-sand
        mixture in a flume, as given by Cui (2007); 0 is the exchange ``"surface"``. Add
        it to ``theta`` when ``SedimentConfig.interface_exchange = "bedload_mix"``.
    """
    return {"c_interface": jnp.asarray(0.7)}


def default_bed_parameters(n_classes: int) -> dict[str, Array]:
    """Default parameters of the bed layers and of consolidation (see the module docstring).

    Args:
        n_classes: number of size classes K.

    Returns:
        Dictionary of physical values: ``delta_a`` (scalar), ``gamma1`` and ``K_cons``
        (shape [n_classes]). The first (finest) class has the low density and the high
        consolidation coefficient of a cohesive deposit.
    """
    is_first = np.arange(n_classes) == 0
    return {
        "delta_a": jnp.asarray(0.05),
        "gamma1": jnp.asarray(np.where(is_first, 800.0, 1500.0)),
        "K_cons": jnp.asarray(np.where(is_first, 250.0, 1.0)),
    }
