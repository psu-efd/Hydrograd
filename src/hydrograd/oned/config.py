"""Settings and default parameters of the 1-D mode.

Hydraulics: :class:`GeometrySmoothing`, :class:`BackwaterNumerics` and
:class:`UnsteadyNumerics` hold the numerical settings, and :class:`HydraulicsConfig`
combines them with the physical constants (shared with the sediment model, so that gravity
and the water density are defined once) and the choice of the flow solver
(``flow_solver``: the quasi-steady backwater computation or the unsteady shallow-water
equations).

Morphodynamics: :class:`MorphNumerics` holds the settings of the sediment routing, the bed
update and the time loop. :class:`OneDConfig` combines everything, including the sediment
settings of :mod:`hydrograd.sediment`.

The calibrated quantities are plain arrays in a dictionary ``theta``.
:func:`default_parameters` returns all of them: the closure and bed parameters of
:mod:`hydrograd.sediment.config`, plus those of the 1-D mode,

    name       default    units         meaning
    n_manning  0.03 [N]   s/m^(1/3)     Manning's n of each cross-section
    a_r        1e-6       (see below)   coefficient of the sediment rating curve
    b_r        2.0        -             exponent of the sediment rating curve
    p_in       1/K [K]    -             size fractions of the inflow load

with the rating curve Q_t,in = a_r Q^b_r (m^3/s of solids for Q in m^3/s), used when the
forcing gives no inflow load.

An optional entry ``tau_ce_fresh`` (Pa; not in the defaults) gives deposits laid down
during the run their own critical shear stress for cohesive erosion: at a section whose
thalweg has risen by d above the initial one, the model uses

    tau_ce + d / (d + delta_a) (tau_ce_fresh - tau_ce),

so that the initial bed keeps ``tau_ce`` and a deposit thicker than the active layer
erodes at about ``tau_ce_fresh``. Without the entry all of the bed has ``tau_ce``.

An optional entry ``roughness_length`` (m; not in the defaults; a scalar or one value per
cross-section, shape [N]) is the roughness length k of the resistance laws
``HydraulicsConfig.resistance = "variable_power"`` and ``"keulegan"``, which do not use
``n_manning``; it is required for them with ``roughness_scale = "length"``. With
``roughness_scale = "surface_mean"`` or ``"surface_d90"`` the entry ``roughness_factor``
(-; not in the defaults) is required in its place: the roughness length of a section is
then that factor times the geometric mean grain size, or the D90, of its active layer.

An optional entry ``beta_momentum`` (-; not in the defaults; a scalar or one value per
cross-section, shape [N]) is the momentum coefficient beta of the unsteady flow solver
(:mod:`hydrograd.oned.unsteady`). Without the entry beta = 1 (uniform velocity over the
section). The backwater solver does not use it.

:func:`default_parameter_specs` gives the transform of each parameter between its physical
value and the unconstrained variable an optimizer works on (see
:mod:`hydrograd.params.transforms`): positive parameters use ``log``; the exponents
``b_eh`` in (2, 3), ``m_h`` in (-1, 1) and ``b_r`` in (1, 3) use ``affine``; the inflow
fractions ``p_in`` use ``softmax``.

Units: SI (m, s, kg).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import Array

from hydrograd.params.transforms import ParamSpec
from hydrograd.sediment.closures import grain_size_percentile
from hydrograd.sediment.config import (
    PhysicalConstants,
    SedimentConfig,
    default_bed_parameters,
    default_sediment_parameters,
)

__all__ = [
    "KEULEGAN_OFFSET",
    "KEULEGAN_SLOPE",
    "REDUCED_STRESS_EXPONENT",
    "RESISTANCE_LAWS",
    "ROUGHNESS_SCALES",
    "STRICKLER_COEFFICIENT",
    "TRANSPORT_STRESSES",
    "VPE_A1",
    "VPE_A2",
    "BackwaterNumerics",
    "GeometrySmoothing",
    "HydraulicsConfig",
    "MorphNumerics",
    "OneDConfig",
    "UnsteadyNumerics",
    "default_parameter_specs",
    "default_parameters",
    "resistance_parameter",
]

# Flow-resistance laws (``HydraulicsConfig.resistance``) and the shear stress handed to the
# sediment closures (``HydraulicsConfig.transport_stress``).
RESISTANCE_LAWS = ("manning", "variable_power", "keulegan")
TRANSPORT_STRESSES = ("total", "reduced")

# Where the roughness length of the laws "variable_power" and "keulegan" comes from
# (``HydraulicsConfig.roughness_scale``).
ROUGHNESS_SCALES = ("length", "surface_mean", "surface_d90")

# Logarithmic resistance law of Keulegan (1938): U / u_* = 2.5 ln(11 R / k_s), and the
# power law that follows it for depths of about 1 to 100 roughness heights (Manning-
# Strickler; Parker, 1991): U / u_* = 8.1 (R / k_s)^(1/6).
KEULEGAN_SLOPE = 2.5
KEULEGAN_OFFSET = 11.0
STRICKLER_COEFFICIENT = 8.1

# Variable-power resistance equation of Ferguson (2007, Water Resour. Res. 43, W05427):
# U / u_* = a1 a2 x / sqrt(a1^2 + a2^2 x^(5/3)), x = R / k. For deep flow it tends to the
# Manning-Strickler law a1 x^(1/6), for shallow flow to a2 x.
VPE_A1 = 6.5
VPE_A2 = 2.5

# Exponent e of the reduced shear stress tau' = tau (C / C_deep)^e, with C = U / u_* of the
# resistance law and C_deep its deep-flow limit: 1.5 as in Meyer-Peter and Mueller (1948)
# and in Rickenmann and Recking (2011, Water Resour. Res. 47, W07538) and Nitsche et al.
# (2011, Water Resour. Res. 47, W08513), who use values of 1 to 2.
REDUCED_STRESS_EXPONENT = 1.5


@dataclass(frozen=True)
class GeometrySmoothing:
    """Widths of the smooth waterline in the cross-section geometry.

    Attributes:
        beta_softplus: sharpness (1/m) of the smooth positive part of the water depth at a
            station point; the transition width is about 1 / beta (2 cm for 50).
        eps_abs: width (m) of the smooth absolute value of the depth at a station point.
        eps_wet: width (m) of the smooth indicator that a station point is under water.
    """

    beta_softplus: float = 50.0
    eps_abs: float = 1.0e-3
    eps_wet: float = 0.05


@dataclass(frozen=True)
class BackwaterNumerics:
    """Newton settings of the backwater solver.

    Attributes:
        newton_max_iter: largest number of Newton iterations.
        newton_tol: convergence tolerance on the largest water-surface update (m).
        newton_max_step: limit on the water-surface update of one iteration (m).
        critical_depth_iter: number of Newton iterations for the critical depth.
    """

    newton_max_iter: int = 40
    newton_tol: float = 1.0e-9
    newton_max_step: float = 1.0
    critical_depth_iter: int = 40


# Flow solvers of the 1-D mode (see :class:`HydraulicsConfig`).
FLOW_SOLVERS = ("backwater", "unsteady")

# Conditions at a supercritical headwater of the unsteady flow solver (see
# :class:`UnsteadyNumerics`).
HEADWATER_CONDITIONS = ("uniform", "critical")

# Length that multiplies rho g S_f in the bed shear stress (see :class:`HydraulicsConfig`).
SHEAR_RADII = ("section", "depth")


@dataclass(frozen=True)
class UnsteadyNumerics:
    """Settings of the unsteady flow solver (:mod:`hydrograd.oned.unsteady`).

    Attributes:
        n_substeps: number of flow time steps per step of the time loop (per
            morphological step). The scheme is explicit: the flow step dt / n_substeps
            must keep the Courant number (|u| + c) dt / dx below 1 on every reach
            (:func:`hydrograd.oned.unsteady.courant_number`).
        surface_iter: Newton iterations per flow step that recover the water surface from
            the flow area, starting from the water surface of the previous flow step.
        headwater: second condition at the headwater where the flow leaves it
            supercritically (a subcritical headwater needs none). ``"uniform"``
            (default): no acceleration over the first reach, so that the water-surface
            slope of the first reach equals its friction slope (uniform flow).
            ``"critical"``: the headwater is a critical-flow control (the crest of a weir,
            the entrance of a steep reach from a pool).
        checkpoint_substeps: recompute the flow steps of a morphological step in the
            backward pass of reverse-mode derivatives instead of storing them (less
            memory, one more forward pass).
    """

    n_substeps: int = 1
    surface_iter: int = 3
    headwater: str = "uniform"
    checkpoint_substeps: bool = True


@dataclass(frozen=True)
class HydraulicsConfig:
    """All settings of the 1-D hydraulics.

    Attributes:
        constants: physical constants (g and rho are used).
        smoothing: widths of the smooth waterline.
        numerics: Newton settings of the backwater solver.
        shear_radius: length R in the bed shear stress tau_b = rho g R S_f.
            ``"section"`` (default): the hydraulic radius A / P of the whole section, the
            mean shear stress over the wetted perimeter. ``"depth"``: the mean depth
            A / B, which puts the whole friction force on the bed. The two are nearly
            equal in a wide section. In a narrow channel with smooth vertical walls (a
            flume) the hydraulic radius spreads the force over bed and walls and gives a
            bed shear stress that is too small (half, for a depth of half the width),
            while the conveyance ``"segments"`` has no wall friction: use ``"depth"``
            there.
        resistance: flow-resistance law, with the velocity U = C sqrt(g R S_f).
            ``"manning"`` (default): C = R^(1/6) / (n sqrt(g)) with ``theta["n_manning"]``.
            ``"variable_power"``: the equation of Ferguson (2007),
            C = a1 a2 x / sqrt(a1^2 + a2^2 x^(5/3)) with x = R / k, a1 = 6.5, a2 = 2.5 and
            the roughness length k = ``theta["roughness_length"]`` (m). For deep flow it
            is the Manning-Strickler law (n = k^(1/6) / (a1 sqrt(g))); as the depth falls
            towards k the resistance rises faster than Manning's, as observed in shallow
            flow over gravel. ``"keulegan"``: the logarithmic law C = 2.5 ln(11 R / k_s)
            with the roughness height k_s (same parameter as k); below R = 0.247 k_s,
            where the law would turn negative, C goes to zero along its tangent through
            the origin, C = 10.1 R / k_s. See :mod:`hydrograd.oned.geometry`.
        roughness_scale: the roughness length of the laws ``"variable_power"`` and
            ``"keulegan"``. ``"length"`` (default): ``theta["roughness_length"]`` (m), one
            value or one per section. ``"surface_mean"``: ``theta["roughness_factor"]``
            times the geometric mean grain size of the active layer of each section, taken
            at the start of every step, so that the roughness follows the bed surface as
            it coarsens or fines (Cui, 2007, uses the factor 2.5 with the logarithmic law).
            ``"surface_d90"``: the same with the D90 of the active layer, the size below
            which 90 % of it lies (:func:`hydrograd.sediment.closures.grain_size_percentile`;
            Parker, 1990, uses the factor 2). The mean size falls when sand covers the
            bed; the D90 follows the coarse grains that make the roughness.
        transport_stress: the bed shear stress handed to the sediment closures.
            ``"total"`` (default): tau_b = rho g R S_f. ``"reduced"`` (not with
            ``resistance = "manning"``): tau_b (C / C_deep)^1.5, where C_deep is the
            resistance law of a deep flow: a1 x^(1/6) for ``"variable_power"``, where the
            factor is (1 + (a1 / a2)^2 x^(-5/3))^(-3/4) with x = R / k, and 8.1 x^(1/6) for
            ``"keulegan"``, where the factor is 1 for x above 14 and 0.92, 0.82 and 0.64
            at x = 5, 2.5 and 1. In shallow flow part of the resistance is drag on the
            largest grains, which does not move sediment; the reduction is that of
            Rickenmann and Recking (2011) for steep, shallow streams, with the exponent
            of Meyer-Peter and Mueller (1948).
        flow_solver: ``"backwater"`` (default): quasi-steady standard-step energy
            equation, subcritical flow with critical-flow controls
            (:mod:`hydrograd.oned.backwater`). ``"unsteady"``: the unsteady shallow-water
            (Saint-Venant) equations, sub- and supercritical flow with hydraulic jumps
            (:mod:`hydrograd.oned.unsteady`).
        unsteady: settings of the unsteady flow solver.
        conveyance: formula of the conveyance K of a cross-section (see
            :mod:`hydrograd.oned.geometry`). ``"segments"`` (default): sum of the
            conveyances of the slices between the station points, which rises with the
            water surface for every section. ``"section"``: K = A R^(2/3) / n with the
            hydraulic radius of the whole section; it can fall with rising water surface
            where the water spills onto a bench, and the backwater solver may then not
            converge.
    """

    constants: PhysicalConstants = field(default_factory=PhysicalConstants)
    smoothing: GeometrySmoothing = field(default_factory=GeometrySmoothing)
    numerics: BackwaterNumerics = field(default_factory=BackwaterNumerics)
    conveyance: str = "segments"
    shear_radius: str = "section"
    resistance: str = "manning"
    roughness_scale: str = "length"
    transport_stress: str = "total"
    flow_solver: str = "backwater"
    unsteady: UnsteadyNumerics = field(default_factory=UnsteadyNumerics)


@dataclass(frozen=True)
class MorphNumerics:
    """Settings of the sediment routing, the bed update and the time loop.

    The defaults are the validated scheme. Several options exist only to study the
    stability of the adjoint (derivatives that grow with the length of a run while the
    forward solution stays bounded); they are marked "study".

    Attributes:
        hydro_every: solve the hydraulics every this many morphological steps.
        n_substeps: sediment and bed sub-steps per morphological step, with the hydraulics
            frozen. The active-layer composition equilibrates on the time
            T_a = delta_a B dx (1 - lambda) p_k / Q*_k (hours for sand at 1.5 m/s with a
            5 cm layer); an explicit update needs dt / n_substeps < T_a / 2.
        deposition_scheme: where the closures of a reach are evaluated and where its
            deposit goes. ``"collocated"`` (default): both at the downstream section
            (SRH-1D eq. 3.2 and 3.4). Study: ``"upstream_half"`` (closures upstream,
            deposit half to each end) and ``"upstream_downstream"`` (closures upstream,
            deposit downstream); both produce odd-even bed oscillations.
        implicit_exner: locally implicit feedback of the bed change on the capacity in the
            routing (default True; False is the explicit recurrence, study).
        implicit_coupling: ``"diagonal"`` (default): implicit factor per size class, every
            recurrence coefficient stays in (0, 1) and the loads stay non-negative.
            ``"full"``: K x K coupling through the active-layer composition (can give
            small negative loads at the dam).
        composition_update: ``"relaxation"`` (default) or ``"explicit"`` (study); see
            :mod:`hydrograd.sediment.bed`.
        attached_sweeps: extra sweeps of the routing within a step in which the
            deposition/erosion switch uses the load of the current step, differentiated
            through (default 0; larger values reproduce the growth of the adjoint, study).
        lag_switch: hold the lagged load of the switch constant in derivatives (default
            False: the lagged switch is differentiated through, which is exact; True gives
            the "frozen" derivative, study).
        morph_factor: morphological acceleration factor (-).
        head_inflow_fraction: length of the virtual inflow reach above each headwater as a
            fraction (-) of its own outgoing reach; 0 freezes the bed of the headwaters.
        deposition_width: ``"polygon"`` (default) or ``"wetted"`` (study); see
            :func:`hydrograd.oned.geometry.deposition_width`.
        deposition_depth_exponent: exponent m of the local depth in the share of a bed
            change each station point receives, w = wet (h / h_max)^m
            (:func:`hydrograd.oned.geometry.deposition_weights`). 0 (default): the same
            change at all wetted points. 1: in proportion to the depth, which fills the
            channel of a compound section before its flooded terraces.
        deposition_channel_deficit: part a (0 to 1) by which a deposit is reduced at the
            deepest part of the section, w = w (1 - a (h / h_max)^4); 0 (default): none.
            With a > 0 a deposit builds up beside a channel that stays open.
        erosion_depth_exponent: depth exponent of the weights of a net erosion of a
            section; None (default): the weights of a deposit. A value above
            ``deposition_depth_exponent`` cuts the deep part of the section in erosion.
        checkpoint: memory saving of reverse-mode derivatives through the time loop.
            ``"auto"`` (default): segments of about sqrt(T) steps are recomputed in the
            backward pass. An integer: the segment length in steps. ``"none"``: the whole
            history is stored.
        record_every: the recorded histories keep every this many steps.
        snapshot_steps: steps after which the full bed table z is stored (survey dates).
    """

    hydro_every: int = 1
    n_substeps: int = 1
    deposition_scheme: str = "collocated"
    implicit_exner: bool = True
    implicit_coupling: str = "diagonal"
    composition_update: str = "relaxation"
    attached_sweeps: int = 0
    lag_switch: bool = False
    morph_factor: float = 1.0
    head_inflow_fraction: float = 0.5
    deposition_width: str = "polygon"
    deposition_depth_exponent: float = 0.0
    deposition_channel_deficit: float = 0.0
    erosion_depth_exponent: float | None = None
    checkpoint: int | str = "auto"
    record_every: int = 1
    snapshot_steps: tuple[int, ...] = ()


@dataclass(frozen=True)
class OneDConfig:
    """All settings of the 1-D mode.

    Attributes:
        hydraulics: geometry smoothing and backwater solver.
        sediment: size classes, formulas and smoothing of the sediment closures.
        morph: routing, bed update and time loop.
    """

    hydraulics: HydraulicsConfig = field(default_factory=HydraulicsConfig)
    sediment: SedimentConfig = field(default_factory=SedimentConfig)
    morph: MorphNumerics = field(default_factory=MorphNumerics)


def resistance_parameter(
    theta: dict[str, Array],
    hydraulics: HydraulicsConfig,
    p_active: Array | None = None,
    diameters: Array | None = None,
) -> Array:
    """The parameter of the resistance law of the sections.

    Args:
        theta: parameters.
        hydraulics: hydraulics settings (resistance law, roughness scale).
        p_active: fractions of the active layer (-), shape [K, N]; needed for
            ``roughness_scale = "surface_mean"``.
        diameters: grain diameters of the classes (m), shape [K]; needed with
            ``p_active``.

    Returns:
        Manning's n (s/m^(1/3)) for ``resistance = "manning"``, else the roughness length
        (m): ``theta["roughness_length"]``, or ``theta["roughness_factor"]`` times the
        geometric mean grain size or the D90 of the active layer; a scalar or shape [N].
    """
    if hydraulics.resistance not in RESISTANCE_LAWS:
        raise ValueError(
            f"resistance must be one of {RESISTANCE_LAWS}, not '{hydraulics.resistance}'"
        )
    if hydraulics.roughness_scale not in ROUGHNESS_SCALES:
        raise ValueError(
            f"roughness_scale must be one of {ROUGHNESS_SCALES}, "
            f"not '{hydraulics.roughness_scale}'"
        )
    if hydraulics.resistance == "manning":
        if hydraulics.roughness_scale != "length":
            raise ValueError(
                f'roughness_scale = "{hydraulics.roughness_scale}" needs a resistance law with a roughness length'
            )
        return theta["n_manning"]
    if hydraulics.roughness_scale != "length":
        scale = hydraulics.roughness_scale
        if "roughness_factor" not in theta:
            raise KeyError(f'roughness_scale = "{scale}" needs the parameter "roughness_factor" (-)')
        if p_active is None or diameters is None:
            raise ValueError(f'roughness_scale = "{scale}" needs the active layer and the grain diameters')
        if scale == "surface_d90":
            return theta["roughness_factor"] * grain_size_percentile(diameters, p_active, 90.0)
        # Geometric mean size of the active layer, all classes (as in the formula of
        # Wilcock and Crowe).
        total = jnp.sum(p_active, axis=0) + 1.0e-12
        mean_size = jnp.exp(jnp.sum(p_active * jnp.log(diameters)[:, None], axis=0) / total)
        return theta["roughness_factor"] * mean_size
    if "roughness_length" not in theta:
        raise KeyError(
            f'resistance = "{hydraulics.resistance}" needs the parameter "roughness_length" (m)'
        )
    return theta["roughness_length"]


def default_parameters(n_classes: int, n_sections: int = 1) -> dict[str, Array]:
    """Default values of all calibrated parameters of the 1-D mode.

    Args:
        n_classes: number of size classes K.
        n_sections: number of entries of ``n_manning``: the number of cross-sections N, or
            1 for one value for all sections.

    Returns:
        Dictionary ``theta`` of physical values (see the module docstring and
        :mod:`hydrograd.sediment.config`).
    """
    theta = default_sediment_parameters(n_classes)
    theta.update(default_bed_parameters(n_classes))
    theta.update(
        {
            "n_manning": jnp.full(n_sections, 0.03),
            "a_r": jnp.asarray(1.0e-6),
            "b_r": jnp.asarray(2.0),
            "p_in": jnp.full(n_classes, 1.0 / n_classes),
        }
    )
    return theta


def default_parameter_specs() -> dict[str, ParamSpec]:
    """Transform of every calibrated parameter of the 1-D mode, by name.

    No priors are set (``prior_sd`` is None); a calibration sets them for its active
    parameters.

    Returns:
        Dictionary name -> :class:`hydrograd.params.transforms.ParamSpec`.
    """
    specs = [
        ParamSpec("a_eh", "log", doc="Engelund-Hansen coefficient (-)"),
        ParamSpec("b_eh", "affine", lo=2.0, hi=3.0, doc="Engelund-Hansen exponent (-)"),
        ParamSpec("m_h", "affine", lo=-1.0, hi=1.0, doc="hiding exponent (-)"),
        ParamSpec("kappa_w", "log", doc="settling-velocity multiplier per class (-)"),
        ParamSpec("c_b", "log", doc="bed-load adaptation length factor, L_b = c_b h (-)"),
        ParamSpec("alpha_d", "log", doc="recovery coefficient in deposition (-)"),
        ParamSpec("alpha_e", "log", doc="recovery coefficient in erosion (-)"),
        ParamSpec("M_e", "log", doc="cohesive erosion rate (m/s of solids)"),
        ParamSpec("tau_ce", "log", doc="critical shear stress for erosion (Pa)"),
        ParamSpec("tau_cd", "log", doc="critical shear stress for full deposition (Pa)"),
        ParamSpec("delta_a", "log", doc="active-layer thickness (m)"),
        ParamSpec("gamma1", "log", doc="initial dry bulk density per class (kg/m^3)"),
        ParamSpec("K_cons", "log", doc="consolidation coefficient per class (kg/m^3)"),
        ParamSpec("a_r", "log", doc="sediment rating coefficient, Q_t = a_r Q^b_r"),
        ParamSpec("b_r", "affine", lo=1.0, hi=3.0, doc="sediment rating exponent (-)"),
        ParamSpec("p_in", "softmax", doc="size fractions of the inflow load (-)"),
        ParamSpec("n_manning", "log", doc="Manning's n per section (s/m^(1/3))"),
        ParamSpec("roughness_length", "log", doc="roughness length of the resistance law (m)"),
        ParamSpec("roughness_factor", "log", doc="roughness length over the mean size or D90 of the surface (-)"),
    ]
    return {spec.name: spec for spec in specs}
