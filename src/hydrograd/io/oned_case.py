"""Reader and writer of a 1-D case file (JSON).

A 1-D case is one JSON file that holds everything a run of :mod:`hydrograd.oned` needs:
the network, the cross-sections, the initial bed, the boundary conditions, the size
classes, the parameter values and the numerical settings. ``run_control.json`` names it
with ``"case_file_name"`` (see :mod:`hydrograd.io.control`).

File layout
-----------
Arrays are nested lists in row-major order with the shapes given below (N sections, J
station points, K size classes, T steps, H headwaters). All values are in SI units. ::

    {
      "format": "hydrograd-1d-case",
      "version": 1,
      "sediment": {
        "diameters": [K],                 grain diameters (m), finest first
        "cohesive_dmax": 6.25e-5,         classes finer than this are cohesive (m)
        "transport_formula": "engelund_hansen",   or "wilcock_crowe"
        "hiding_form": "power",           or "wilcock_crowe" (see hydrograd.sediment.config)
        "krone_form": "ramp",
        "interface_exchange": "surface",  or "bedload_mix" (see hydrograd.sediment.bed)
        "interface_sand": "surface"       or "cui"
      },
      "hydraulics": {
        "conveyance": "segments",         formula of the conveyance: "segments" or
                                          "section" (see hydrograd.oned.geometry)
        "flow_solver": "backwater",       or "unsteady" (see hydrograd.oned.config)
        "shear_radius": "section",        or "depth": length in the bed shear stress
        "resistance": "manning",          or "variable_power", "keulegan" (need the
                                          parameter "roughness_length")
        "roughness_scale": "length",      or "surface_mean", "surface_d90" (need the
                                          parameter "roughness_factor" in place of the
                                          length)
        "transport_stress": "total"       or "reduced" (not with "manning")
      },
      "network": {
        "reach_length": [N-1],            length of each reach (m)
        "downstream": [N-1]               downstream section of each reach (0-based)
      },
      "sections": {
        "s": [N][J],                      station of the points (m), non-decreasing
        "z": [N][J]                       initial bed elevation of the points (m)
      },
      "bed": {
        "p_active": [K][N],               active-layer fractions (-)
        "p_substrate": [K][N],            substrate fractions (-)
        "substrate_thickness": [N],       (m)
        "z_floor": [N] or null            non-erodible floor (m)
      },
      "forcing": {
        "Q_in": [T] or [T][H],            inflow discharge (m^3/s)
        "eta_outlet": [T],                water surface at the outlet (m)
        "dt": [T],                        step lengths (s)
        "load_in": [K][T], [K][T][H] or null   inflow load (m^3/s of solids);
                                               null uses the rating curve
      },
      "parameters": { "a_eh": 0.05, "kappa_w": [K], ... },
      "numerics": {
        "morph": { ... },                 fields of MorphNumerics
        "backwater": { ... },             fields of BackwaterNumerics
        "unsteady": { ... },              fields of UnsteadyNumerics
        "geometry_smoothing": { ... },    fields of GeometrySmoothing
        "sediment_smoothing": { ... }     fields of SedimentSmoothing
      }
    }

``"hydraulics"``, ``"parameters"`` and ``"numerics"`` may be partial or absent: missing
entries take the defaults of :func:`hydrograd.oned.config.default_parameters` and of the
settings classes.
The optional parameter ``"beta_momentum"`` (momentum coefficient of the unsteady flow
solver; one value, or one per section) has no default: without it the coefficient is 1.
Unknown parameter names and unknown settings are errors, so a misspelled name is noticed.

Units: SI.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from hydrograd.oned.bed import BedState
from hydrograd.oned.config import (
    FLOW_SOLVERS,
    RESISTANCE_LAWS,
    ROUGHNESS_SCALES,
    SHEAR_RADII,
    TRANSPORT_STRESSES,
    BackwaterNumerics,
    GeometrySmoothing,
    HydraulicsConfig,
    MorphNumerics,
    OneDConfig,
    UnsteadyNumerics,
    default_parameters,
)
from hydrograd.oned.forcing import Forcing
from hydrograd.oned.geometry import CONVEYANCE_OPTIONS
from hydrograd.oned.network import Network, network_from_downstream
from hydrograd.sediment.config import (
    SedimentConfig,
    SedimentSmoothing,
    interface_parameters,
    wilcock_crowe_parameters,
)

logger = logging.getLogger(__name__)

__all__ = ["FORMAT", "OneDCase", "read_case", "write_case"]

# Identifier and version of the file format.
FORMAT = "hydrograd-1d-case"
VERSION = 1

# Differentiable parameters: name -> array of physical values.
Theta = dict[str, Any]

# Parameters that a case file may give although they have no default value.
OPTIONAL_PARAMETERS = ("beta_momentum", "roughness_length", "roughness_factor")


@dataclass(frozen=True)
class OneDCase:
    """Everything a 1-D run needs.

    Attributes:
        config: settings of the 1-D mode.
        network: the channel network.
        theta: parameter values (physical), by name.
        forcing: boundary conditions of all steps.
        bed0: initial bed.
    """

    config: OneDConfig
    network: Network
    theta: Theta
    forcing: Forcing
    bed0: BedState


def _to_list(array: Any) -> Any:
    """Array as nested lists (None stays None).

    Args:
        array: a NumPy or JAX array, or None.

    Returns:
        Nested lists of Python numbers, or None.
    """
    if array is None:
        return None
    return np.asarray(array).tolist()


def _settings_to_dict(settings: Any) -> dict[str, Any]:
    """Fields of a settings dataclass as a JSON-compatible dictionary.

    Args:
        settings: a frozen dataclass of numbers, strings, booleans and tuples.

    Returns:
        Dictionary field name -> value (tuples as lists).
    """
    values = {}
    for field in dataclasses.fields(settings):
        value = getattr(settings, field.name)
        values[field.name] = list(value) if isinstance(value, tuple) else value
    return values


def _settings_from_dict(settings_class: type, values: dict[str, Any] | None, block: str) -> Any:
    """Build a settings dataclass from a (partial) dictionary.

    Args:
        settings_class: the dataclass to build.
        values: field name -> value; missing fields take their defaults; None is empty.
        block: name of the JSON block, for the error message.

    Returns:
        An instance of ``settings_class``.
    """
    values = dict(values or {})
    known = {field.name for field in dataclasses.fields(settings_class)}

    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"1-D case file: unknown setting(s) {unknown} in '{block}'")

    # JSON has no tuples: convert lists back.
    for name, value in values.items():
        if isinstance(value, list):
            values[name] = tuple(value)

    return settings_class(**values)


def write_case(path: str | Path, case: OneDCase) -> None:
    """Write a 1-D case file.

    Args:
        path: output ``.json`` file.
        case: the case.
    """
    config = case.config
    sediment = config.sediment
    bed = case.bed0
    forcing = case.forcing

    parameters = {name: _to_list(value) for name, value in case.theta.items()}
    data = {
        "format": FORMAT,
        "version": VERSION,
        "sediment": {
            "diameters": list(sediment.diameters),
            "cohesive_dmax": sediment.cohesive_dmax,
            "transport_formula": sediment.transport_formula,
            "hiding_form": sediment.hiding_form,
            "krone_form": sediment.krone_form,
            "interface_exchange": sediment.interface_exchange,
            "interface_sand": sediment.interface_sand,
        },
        "hydraulics": {
            "conveyance": config.hydraulics.conveyance,
            "flow_solver": config.hydraulics.flow_solver,
            "shear_radius": config.hydraulics.shear_radius,
            "resistance": config.hydraulics.resistance,
            "roughness_scale": config.hydraulics.roughness_scale,
            "transport_stress": config.hydraulics.transport_stress,
        },
        "network": {
            "reach_length": _to_list(case.network.reach_length),
            "downstream": _to_list(case.network.downstream),
        },
        "sections": {"s": _to_list(bed.s), "z": _to_list(bed.z)},
        "bed": {
            "p_active": _to_list(bed.p_active),
            "p_substrate": _to_list(bed.p_substrate),
            "substrate_thickness": _to_list(bed.substrate_thickness),
            "z_floor": _to_list(bed.z_floor),
        },
        "forcing": {
            "Q_in": _to_list(forcing.Q_in),
            "eta_outlet": _to_list(forcing.eta_outlet),
            "dt": _to_list(forcing.dt),
            "load_in": _to_list(forcing.load_in),
        },
        "parameters": parameters,
        "numerics": {
            "morph": _settings_to_dict(config.morph),
            "backwater": _settings_to_dict(config.hydraulics.numerics),
            "unsteady": _settings_to_dict(config.hydraulics.unsteady),
            "geometry_smoothing": _settings_to_dict(config.hydraulics.smoothing),
            "sediment_smoothing": _settings_to_dict(sediment.smoothing),
        },
    }

    Path(path).write_text(json.dumps(data))
    logger.info("wrote 1-D case %s", path)


def _array(data: dict[str, Any], block: str, key: str) -> jnp.ndarray:
    """A required array of the case file as a float JAX array.

    Args:
        data: the decoded JSON.
        block: name of the block.
        key: name of the array inside the block.

    Returns:
        The array.
    """
    if block not in data or key not in data[block]:
        raise KeyError(f"1-D case file: '{block}.{key}' is missing")
    return jnp.asarray(data[block][key], dtype=float)


def read_case(path: str | Path) -> OneDCase:
    """Read a 1-D case file and check that its arrays fit together.

    Args:
        path: the ``.json`` case file.

    Returns:
        The case.
    """
    data = json.loads(Path(path).read_text())
    if data.get("format") != FORMAT:
        raise ValueError(f"{path}: not a 1-D case file ('format' must be '{FORMAT}')")

    # ------------------------------------------------------------------
    # 1. Settings
    # ------------------------------------------------------------------
    numerics = data.get("numerics", {})
    sediment_block = dict(data["sediment"])
    sediment_block["diameters"] = tuple(sediment_block["diameters"])
    sediment_smoothing = _settings_from_dict(
        SedimentSmoothing, numerics.get("sediment_smoothing"), "numerics.sediment_smoothing",
    )
    sediment = SedimentConfig(smoothing=sediment_smoothing, **sediment_block)

    hydraulics_block = dict(data.get("hydraulics", {}))
    known = {"conveyance", "flow_solver", "shear_radius", "resistance", "roughness_scale", "transport_stress"}
    unknown = sorted(set(hydraulics_block) - known)
    if unknown:
        raise ValueError(f"1-D case file: unknown setting(s) {unknown} in 'hydraulics'")
    conveyance = hydraulics_block.get("conveyance", HydraulicsConfig().conveyance)
    if conveyance not in CONVEYANCE_OPTIONS:
        raise ValueError(
            f"1-D case file: hydraulics.conveyance must be one of {CONVEYANCE_OPTIONS}, "
            f"not '{conveyance}'"
        )

    flow_solver = hydraulics_block.get("flow_solver", HydraulicsConfig().flow_solver)
    if flow_solver not in FLOW_SOLVERS:
        raise ValueError(
            f"1-D case file: hydraulics.flow_solver must be one of {FLOW_SOLVERS}, "
            f"not '{flow_solver}'"
        )

    shear_radius = hydraulics_block.get("shear_radius", HydraulicsConfig().shear_radius)
    if shear_radius not in SHEAR_RADII:
        raise ValueError(
            f"1-D case file: hydraulics.shear_radius must be one of {SHEAR_RADII}, "
            f"not '{shear_radius}'"
        )

    resistance = hydraulics_block.get("resistance", HydraulicsConfig().resistance)
    if resistance not in RESISTANCE_LAWS:
        raise ValueError(
            f"1-D case file: hydraulics.resistance must be one of {RESISTANCE_LAWS}, "
            f"not '{resistance}'"
        )
    transport_stress = hydraulics_block.get("transport_stress", HydraulicsConfig().transport_stress)
    if transport_stress not in TRANSPORT_STRESSES:
        raise ValueError(
            f"1-D case file: hydraulics.transport_stress must be one of {TRANSPORT_STRESSES}, "
            f"not '{transport_stress}'"
        )
    if transport_stress == "reduced" and resistance == "manning":
        raise ValueError(
            '1-D case file: hydraulics.transport_stress "reduced" needs resistance '
            '"variable_power" or "keulegan"'
        )
    roughness_scale = hydraulics_block.get("roughness_scale", HydraulicsConfig().roughness_scale)
    if roughness_scale not in ROUGHNESS_SCALES:
        raise ValueError(
            f"1-D case file: hydraulics.roughness_scale must be one of {ROUGHNESS_SCALES}, "
            f"not '{roughness_scale}'"
        )
    if roughness_scale != "length" and resistance == "manning":
        raise ValueError(
            f'1-D case file: hydraulics.roughness_scale "{roughness_scale}" needs resistance '
            '"variable_power" or "keulegan"'
        )

    hydraulics = HydraulicsConfig(
        smoothing=_settings_from_dict(
            GeometrySmoothing, numerics.get("geometry_smoothing"), "numerics.geometry_smoothing",
        ),
        numerics=_settings_from_dict(
            BackwaterNumerics, numerics.get("backwater"), "numerics.backwater",
        ),
        conveyance=conveyance,
        flow_solver=flow_solver,
        shear_radius=shear_radius,
        resistance=resistance,
        roughness_scale=roughness_scale,
        transport_stress=transport_stress,
        unsteady=_settings_from_dict(
            UnsteadyNumerics, numerics.get("unsteady"), "numerics.unsteady",
        ),
    )
    morph = _settings_from_dict(MorphNumerics, numerics.get("morph"), "numerics.morph")
    config = OneDConfig(hydraulics=hydraulics, sediment=sediment, morph=morph)

    # ------------------------------------------------------------------
    # 2. Network, sections and bed
    # ------------------------------------------------------------------
    network = network_from_downstream(
        _array(data, "network", "reach_length"), data["network"]["downstream"],
    )
    z = _array(data, "sections", "z")
    s = _array(data, "sections", "s")
    z_floor = data["bed"].get("z_floor")
    bed0 = BedState(
        z=z,
        s=s,
        p_active=_array(data, "bed", "p_active"),
        p_substrate=_array(data, "bed", "p_substrate"),
        substrate_thickness=_array(data, "bed", "substrate_thickness"),
        z_floor=None if z_floor is None else jnp.asarray(z_floor, dtype=float),
    )

    # ------------------------------------------------------------------
    # 3. Forcing
    # ------------------------------------------------------------------
    load_in = data["forcing"].get("load_in")
    forcing = Forcing(
        Q_in=_array(data, "forcing", "Q_in"),
        eta_outlet=_array(data, "forcing", "eta_outlet"),
        dt=_array(data, "forcing", "dt"),
        load_in=None if load_in is None else jnp.asarray(load_in, dtype=float),
    )

    # ------------------------------------------------------------------
    # 4. Parameters: defaults, overridden by the file
    # ------------------------------------------------------------------
    n_sections = network.n_sections
    n_classes = sediment.n_classes
    theta = default_parameters(n_classes)
    if sediment.transport_formula == "wilcock_crowe":
        theta.update(wilcock_crowe_parameters())
    if sediment.interface_exchange == "bedload_mix":
        theta.update(interface_parameters())
    given = data.get("parameters", {})

    unknown = sorted(set(given) - set(theta) - set(OPTIONAL_PARAMETERS))
    if unknown:
        raise ValueError(f"1-D case file: unknown parameter(s) {unknown}")
    for name, value in given.items():
        theta[name] = jnp.asarray(value, dtype=float)

    # ------------------------------------------------------------------
    # 5. Consistency of the shapes
    # ------------------------------------------------------------------
    n_steps = forcing.n_steps
    checks = [
        ("sections.z", z.shape[0], n_sections),
        ("sections.s", s.shape, z.shape),
        ("bed.p_active", bed0.p_active.shape, (n_classes, n_sections)),
        ("bed.p_substrate", bed0.p_substrate.shape, (n_classes, n_sections)),
        ("bed.substrate_thickness", bed0.substrate_thickness.shape, (n_sections,)),
        ("forcing.eta_outlet", forcing.eta_outlet.shape, (n_steps,)),
        ("forcing.dt", forcing.dt.shape, (n_steps,)),
    ]
    if forcing.Q_in.ndim == 2:
        checks.append(("forcing.Q_in", forcing.Q_in.shape, (n_steps, network.n_headwaters)))
    if forcing.load_in is not None:
        checks.append(("forcing.load_in", forcing.load_in.shape[:2], (n_classes, n_steps)))
    if theta["n_manning"].size != 1:
        checks.append(("parameters.n_manning", theta["n_manning"].shape, (n_sections,)))
    for name in ("beta_momentum", "roughness_length"):
        if name in theta and theta[name].size != 1:
            checks.append((f"parameters.{name}", theta[name].shape, (n_sections,)))
    if hydraulics.resistance != "manning":
        needed = "roughness_length" if hydraulics.roughness_scale == "length" else "roughness_factor"
        if needed not in theta:
            raise ValueError(
                f'1-D case file: hydraulics.resistance "{hydraulics.resistance}" with roughness_scale '
                f'"{hydraulics.roughness_scale}" needs the parameter "{needed}"'
            )
    for name in ("kappa_w", "gamma1", "K_cons", "p_in"):
        checks.append((f"parameters.{name}", theta[name].shape, (n_classes,)))

    for name, actual, expected in checks:
        if actual != expected:
            raise ValueError(f"1-D case file: {name} has shape {actual}, expected {expected}")

    # Stations must not decrease along a section (segments of negative width).
    if bool(jnp.any(s[:, 1:] < s[:, :-1])):
        raise ValueError("1-D case file: sections.s must be non-decreasing along each section")

    logger.info(
        "1-D case: %d sections, %d headwater(s), %d size classes, %d steps",
        n_sections, network.n_headwaters, n_classes, n_steps,
    )
    return OneDCase(config=config, network=network, theta=theta, forcing=forcing, bed0=bed0)
