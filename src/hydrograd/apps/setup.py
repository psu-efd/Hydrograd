"""Shared case setup for all run modes: files -> mesh -> bed -> IC -> SWEModel.

Every run mode (forward simulation, sensitivity analysis, inversion, UDE training)
starts from the same :class:`Case`, built here from ``run_control.json`` and the SRH-2D
files of the case directory. The steps follow ``solve_swe_2D.jl``.

Pipeline
--------
1. Read the SRH-2D files (``.srhhydro``, ``.srhgeom``, ``.srhmat``) and build the mesh.
2. Time span and unit system, from the run control or from the ``.srhhydro`` file.
3. Bed elevation per cell (m), interpolated from the mesh nodes.
4. Initial condition per cell: water-surface elevation (WSE, m), still-water elevation
   w_still (m), and unit discharges q_x, q_y (m^2/s), as in ``process_ICs_2D.jl``.
5. Flow-resistance model: constant Manning's n per material zone, an n(h, |U|, ks) law,
   or a neural-network closure (UDE mode).
6. Optional vegetation drag (``"vegetation"`` block, see :func:`_vegetation`) and
   horizontal eddy viscosity (``"turbulence"`` block, see :func:`_turbulence`).
7. The static :class:`~hydrograd.physics.model.SWEModel`, optional ``"numerics"``
   overrides (scheme, spatial order, limiter, checkpointing), and the ODE settings.

The initial state is

    U0 = [xi, q_x, q_y],   xi = WSE - w_still   (m),   shape [3, n_cells].

Differences from Julia
----------------------
All are inert for the Hydrograd.jl examples. The ``.srhhydro`` time/unit branch
(``bUse_srhhydro_time_settings = true``) reads ``SimTime`` in SRH order and honours
``GridUnit`` (Julia swaps the SimTime fields and its unit check always fails).

Files read
----------
``run_control.json`` (through :class:`~hydrograd.io.control.Control`), the SRH-2D case
files, and, for ``"from_file"`` initial conditions, a JSON file with the keys ``wse``,
``wstill``, ``q_x`` and ``q_y``. Nothing is written.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

from hydrograd.bed import zb_nodes_to_cells
from hydrograd.io.control import Control
from hydrograd.io.srh2d import SRHCase, read_srh_case
from hydrograd.mesh import Mesh, build_mesh
from hydrograd.params.manning import ManningLaw
from hydrograd.physics.model import (
    Resistance,
    SWEModel,
    Turbulence,
    Vegetation,
    build_model,
)
from hydrograd.solvers.ode import ODESettings

logger = logging.getLogger(__name__)

__all__ = ["Case", "setup_case"]

# Per-mode prefixes of the initial-condition option keys in run_control.json.
_IC_PREFIX = {
    "forward": "forward_simulation_initial_condition",
    "inversion": "inversion_forward_simulation_initial_condition",
    "sensitivity": "sensitivity_forward_simulation_initial_condition",
    "ude": "UDE_forward_simulation_initial_condition",
}

# Per-mode names of the ODE-solver option block (the forward mode uses flat keys instead).
_ODE_BLOCK = {
    "inversion": "inversion_ode_solver_options",
    "sensitivity": "sensitivity_ode_solver_options",
    "ude": "UDE_ode_solver_options",
}

# SRH-2D GridUnit (lower case) -> unit system.
_UNIT_NAMES = {"meters": "SI", "meter": "SI", "feet": "EN", "foot": "EN"}


@dataclass(frozen=True)
class Case:
    """Everything a run mode needs.

    Attributes:
        ctrl: run control (``run_control.json``).
        srh: the parsed SRH-2D files.
        mesh: the unstructured mesh.
        model: static SWE model; its defaults are the truth parameters from the files.
        U0: initial state [xi, q_x, q_y], shape [3, n_cells]; xi in m, q in m^2/s.
        ode: time-integration settings.
        wse0: initial water-surface elevation (m), shape [n_cells].
        wstill: still-water elevation (m), shape [n_cells].
        unit_system: ``"SI"`` or ``"EN"``.
    """

    ctrl: Control
    srh: SRHCase
    mesh: Mesh
    model: SWEModel
    U0: Array
    ode: ODESettings
    wse0: np.ndarray
    wstill: np.ndarray
    unit_system: str


def _time_and_units(ctrl: Control, srh: SRHCase) -> tuple[tuple[float, float], float, str]:
    """Time span, time step and unit system of the run.

    By default these come from the run control and the unit system is SI, as in
    Hydrograd.jl. With ``bUse_srhhydro_time_settings = true`` they come from the
    ``.srhhydro`` ``SimTime`` entry (start and end in hours, step in seconds) and the
    ``GridUnit`` of the ``.srhgeom`` file.

    Args:
        ctrl: run control.
        srh: the parsed SRH-2D files.

    Returns:
        ``((t_start, t_end), dt, unit)``: time span (s), time step (s) and the unit
        system ``"SI"`` or ``"EN"``.
    """
    # Default: time settings from the run control, SI constants regardless of GridUnit.
    if not ctrl.use_srhhydro_time:
        if _UNIT_NAMES.get(srh.geom.grid_unit.lower()) == "EN":
            logger.warning(
                "GridUnit is %s but SI constants (g = 9.81, k_n = 1) are used, as in "
                "Hydrograd.jl when bUse_srhhydro_time_settings is false",
                srh.geom.grid_unit,
            )
        return ctrl.tspan, ctrl.dt, "SI"

    # SRH-2D settings: SimTime = (start [h], dt [s], end [h]).
    t_start_h, dt_s, t_end_h = srh.hydro.sim_time

    unit = _UNIT_NAMES.get(srh.geom.grid_unit.lower())
    if unit is None:
        raise ValueError(f"unsupported GridUnit '{srh.geom.grid_unit}'")

    return (3600.0 * t_start_h, 3600.0 * t_end_h), dt_s, unit


def _initial_condition(ctrl: Control, zb: np.ndarray) -> tuple[np.ndarray, ...]:
    """Initial condition per cell (``process_ICs_2D.jl``).

    Two options, chosen by ``<prefix>_options``:
    - ``"constant"``: ``<prefix>_constant_values = [wse, wstill, q_x, q_y]`` in every cell;
    - ``"from_file"``: a JSON file with per-cell arrays ``wse``, ``wstill``, ``q_x``, ``q_y``.

    Args:
        ctrl: run control (its ``mode`` selects the option-key prefix).
        zb: bed elevation (m), shape [n_cells].

    Returns:
        ``(wse, wstill, qx, qy)``, each of shape [n_cells]: water-surface elevation (m),
        still-water elevation (m) and unit discharges (m^2/s).
    """
    prefix = _IC_PREFIX[ctrl.mode]
    n_cells = len(zb)
    kind = ctrl.opt(f"{prefix}_options")

    # Read the four fields, either as constants or from a JSON file.
    if kind == "constant":
        values = ctrl.opt(f"{prefix}_constant_values")
        if len(values) < 4:
            raise ValueError(
                f"{prefix}_constant_values needs [wse, wstill, q_x, q_y], got {values}"
            )
        wse = np.full(n_cells, float(values[0]))
        wstill = np.full(n_cells, float(values[1]))
        qx = np.full(n_cells, float(values[2]))
        qy = np.full(n_cells, float(values[3]))
    elif kind == "from_file":
        ic_path = ctrl.case_dir / ctrl.opt(f"{prefix}_file_name")
        ic_data = json.loads(ic_path.read_text())
        wse = np.asarray(ic_data["wse"], dtype=float)
        wstill = np.asarray(ic_data["wstill"], dtype=float)
        qx = np.asarray(ic_data["q_x"], dtype=float)
        qy = np.asarray(ic_data["q_y"], dtype=float)
    else:
        raise ValueError(f"unknown IC option '{kind}'")

    # The water surface may not lie below the bed.
    if np.any(wse < zb - 1.0e-12):
        raise ValueError("initial condition has WSE below the bed elevation")

    # Hydrograd.jl requires a wet IC; here dry cells are allowed and handled by the
    # h_small thresholds, so they are only reported.
    n_dry = int(np.sum(wse - zb < 1.0e-3))
    if n_dry:
        logger.info("initial condition has %d dry cells (h < h_small)", n_dry)

    return wse, wstill, qx, qy


def _ode_settings(ctrl: Control, tspan: tuple[float, float], dt: float) -> ODESettings:
    """Time-integration settings of the run mode.

    The forward mode reads flat ``forward_simulation_*`` keys; the other modes read the
    ``*_ode_solver_options`` block.

    Args:
        ctrl: run control.
        tspan: (t_start, t_end) (s).
        dt: time step (s); the initial step for adaptive solvers.

    Returns:
        The ODE settings (solver name, adaptivity, number of saved frames).
    """
    # Forward mode: flat keys; a "customized" Julia solver becomes fixed-step Euler.
    if ctrl.mode == "forward":
        solver = ctrl.opt("forward_simulation_ode_solver")
        if ctrl.opt("forward_simulation_solver") == "customized":
            logger.warning("'customized' solver mapped to fixed-step Euler()")
            solver = "Euler()"
            adaptive = False
        else:
            adaptive = bool(ctrl.opt("forward_simulation_adaptive"))

        n_save = int(ctrl.opt("forward_simulation_nSave"))
        return ODESettings(tspan[0], tspan[1], dt, n_save, solver, adaptive)

    # Inversion, sensitivity and UDE modes: one option block.
    options = ctrl.opt(_ODE_BLOCK[ctrl.mode])
    return ODESettings(
        tspan[0],
        tspan[1],
        dt,
        int(options["ode_solver_nSave"]),
        options["ode_solver"],
        bool(options["ode_solver_adaptive"]),
    )


def _ks_cells(ks_list: Any, mat_id: np.ndarray) -> np.ndarray:
    """Roughness height ks per cell from the per-material list.

    Args:
        ks_list: ``ks_list[i]`` is the roughness height (m) of material i.
        mat_id: material ID of each cell, shape [n_cells].

    Returns:
        ks per cell (m), shape [n_cells].
    """
    return np.asarray(ks_list, dtype=float)[mat_id]


def _resistance(ctrl: Control, mat_id: np.ndarray) -> tuple[Resistance, Any]:
    """Flow-resistance model of the run and, if needed, the roughness height per cell.

    Three cases:
    - forward mode with ``ManningN_option = "variable"``: an n(h, |U|, ks) law;
    - UDE mode: a neural-network closure (``UDE_choice``) with normalization bounds
      from ``UDE_NN_config``;
    - otherwise: constant Manning's n per material zone.

    Args:
        ctrl: run control.
        mat_id: material ID of each cell, shape [n_cells].

    Returns:
        ``(resistance, ks)``: the resistance model and the roughness height per cell (m),
        shape [n_cells], or None when the model does not use ks.
    """
    # ------------------------------------------------------------------
    # 1. Forward mode with a Manning's n law
    # ------------------------------------------------------------------
    if ctrl.mode == "forward" and ctrl.opt("forward_simulation_ManningN_option") == "variable":
        params: dict[str, Any] = dict(ctrl.opt("forward_simulation_ManningN_function_parameters"))
        kind = ctrl.opt("forward_simulation_ManningN_function_type")

        # The h_Umag_ks law takes ks per cell; remove it from the scalar parameters.
        ks = _ks_cells(params.pop("ks"), mat_id) if kind == "h_Umag_ks" else None

        # Scalar law parameters only (list-valued entries are dropped).
        scalar_params: dict[str, float] = {}
        for key, value in params.items():
            if not isinstance(value, list):
                scalar_params[key] = float(value)

        law = ManningLaw(kind, scalar_params)
        return Resistance(mode="law", law=law), ks

    # ------------------------------------------------------------------
    # 2. UDE mode: neural-network closure
    # ------------------------------------------------------------------
    if ctrl.mode == "ude" and ctrl.opt("UDE_choice") == "VegetationCD":
        # The network predicts the stem drag coefficient (see _vegetation); the bed keeps
        # a constant Manning's n per material zone.
        return Resistance(mode="zone"), None

    if ctrl.mode == "ude":
        choice = ctrl.opt("UDE_choice")
        nn = ctrl.opt("UDE_NN_config")

        def b(k: str) -> tuple[float, float]:
            """Normalization bounds ``(lower, upper)`` of key ``k``; (0, 1) if absent.

            Args:
                k: key in ``UDE_NN_config``, e.g. ``"h_bounds"``.

            Returns:
                (lower, upper) as floats, in the units of the bounded quantity.
            """
            lo, hi = nn.get(k, (0.0, 1.0))
            return float(lo), float(hi)

        # Roughness height per cell, only for the n(h, |U|, ks) closure.
        ks = None
        if choice == "ManningN_h_Umag_ks":
            ks = _ks_cells(ctrl.opt("UDE_ManningN_function_parameters")["ks"], mat_id)

        # Velocity (m/s) and depth (m) bounds; the unit-discharge bounds (m^2/s) are
        # their products.
        velocity_bounds = b("velocity_magnitude_bounds")
        h_bounds = b("h_bounds")
        q_bounds = (h_bounds[0] * velocity_bounds[0], h_bounds[1] * velocity_bounds[1])

        if nn.get("layernorm_axes", "features") not in ("features", "all"):
            raise ValueError("UDE_NN_config.layernorm_axes must be 'features' or 'all'")

        resistance = Resistance(
            mode=f"nn_{choice}",
            activations=tuple(nn["activations"]),
            out_bounds=b("output_bounds"),
            h_bounds=h_bounds,
            umag_bounds=b("Umag_bounds"),
            ks_bounds=b("ks_bounds"),
            q_bounds=q_bounds,
            ln_axes=nn.get("layernorm_axes", "features"),
        )
        return resistance, ks

    # ------------------------------------------------------------------
    # 3. Default: constant Manning's n per material zone
    # ------------------------------------------------------------------
    return Resistance(mode="zone"), None


def _per_material(values: Any, n_materials: int, default: float, name: str) -> tuple[float, ...]:
    """Expand a per-material specification to one value per material ID.

    Args:
        values: a number (same for all materials), a list with one value per material ID,
            or a dict ``{"<material ID>": value}`` (other materials get ``default``).
        n_materials: number of material IDs (0 .. n_materials - 1).
        default: value for materials missing from a dict.
        name: entry name, for error messages.

    Returns:
        One float per material ID.
    """
    # A single number applies to every material.
    if isinstance(values, (int, float)):
        return tuple(float(values) for _ in range(n_materials))

    # A list gives the value of every material in ID order.
    if isinstance(values, list):
        if len(values) != n_materials:
            raise ValueError(f"vegetation.{name}: expected {n_materials} values, got {len(values)}")
        return tuple(float(v) for v in values)

    # A dict gives values for some material IDs.
    expanded = [default] * n_materials
    for key, value in values.items():
        material = int(key)
        if not 0 <= material < n_materials:
            raise ValueError(f"vegetation.{name}: no material ID {material}")
        expanded[material] = float(value)
    return tuple(expanded)


def _vegetation(ctrl: Control, n_materials: int) -> Vegetation:
    """Vegetation drag settings from the optional ``"vegetation"`` block of the run control.

    Example (material 2 is a stem patch with d = 0.01 m, dS = 0.1 m, so a = 1 /m)::

        "vegetation": {
            "a":  {"2": 1.0},          # frontal area per unit volume (1/m), per material
            "CD": {"2": 1.0}           # drag coefficient (-), per material
        }

    Instead of ``a``, the stem diameter ``d`` (m) and spacing ``spacing`` (m) may be given
    per material; then a = d / spacing^2. In UDE mode with ``UDE_choice = "VegetationCD"``
    the drag coefficient comes from a local network C_D = f(h, |u|, a); its normalization
    bounds are ``h_bounds`` (m), ``Umag_bounds`` (m/s), ``a_bounds`` (1/m) and
    ``output_bounds`` (C_D) in ``UDE_NN_config``.

    Args:
        ctrl: run control.
        n_materials: number of material IDs.

    Returns:
        The vegetation settings (mode ``none`` if the block is absent).
    """
    block = ctrl.raw.get("vegetation")
    is_cd_network = ctrl.mode == "ude" and ctrl.opt("UDE_choice") == "VegetationCD"

    if not block:
        if is_cd_network:
            raise ValueError("UDE_choice 'VegetationCD' needs a 'vegetation' block with 'a'")
        return Vegetation()

    # Frontal area per unit volume a (1/m): given directly, or from d and the spacing.
    if "a" in block:
        a_zone = _per_material(block["a"], n_materials, 0.0, "a")
    else:
        diameters = _per_material(block["d"], n_materials, 0.0, "d")
        spacings = _per_material(block["spacing"], n_materials, 1.0, "spacing")
        a_zone = tuple(d / s ** 2 for d, s in zip(diameters, spacings, strict=True))

    CD_zone = _per_material(block.get("CD", 1.0), n_materials, 1.0, "CD")

    # Constant C_D per zone (forward, sensitivity, inversion of veg_CD).
    if not is_cd_network:
        return Vegetation(mode="zone", a_zone=a_zone, CD_zone=CD_zone)

    # Local network C_D = f(h, |u|, a).
    nn = ctrl.opt("UDE_NN_config")

    def bounds(key: str, default: tuple[float, float]) -> tuple[float, float]:
        """Normalization bounds ``(lower, upper)`` from ``UDE_NN_config``."""
        lower, upper = nn.get(key, default)
        return float(lower), float(upper)

    return Vegetation(
        mode="nn_CD",
        a_zone=a_zone,
        CD_zone=CD_zone,
        activations=tuple(nn["activations"]),
        CD_bounds=bounds("output_bounds", (0.0, 3.0)),
        h_bounds=bounds("h_bounds", (0.0, 1.0)),
        umag_bounds=bounds("Umag_bounds", (0.0, 1.0)),
        a_bounds=bounds("a_bounds", (0.0, max(a_zone) if a_zone else 1.0)),
    )


def _turbulence(ctrl: Control) -> Turbulence:
    """Eddy-viscosity settings from the optional ``"turbulence"`` block of the run control.

    Examples::

        "turbulence": {"model": "parabolic", "coefficient": 0.7, "nu0": 1e-6}
        "turbulence": {"model": "constant", "nu_t": 0.01}

    The parabolic model is nu_t = C_t u_* h + nu_0 (SRH-2D); C_t is differentiable as
    ``theta["eddy_Ct"]``.

    Args:
        ctrl: run control.

    Returns:
        The turbulence settings (mode ``none`` if the block is absent).
    """
    block = ctrl.raw.get("turbulence")
    if not block:
        return Turbulence()

    model = block.get("model", "parabolic")
    if model not in ("parabolic", "constant"):
        raise ValueError(f"turbulence.model must be 'parabolic' or 'constant', got {model!r}")

    return Turbulence(
        mode=model,
        coefficient=float(block.get("coefficient", 0.7)),
        nu0=float(block.get("nu0", 1.0e-6)),
        nu_constant=float(block.get("nu_t", 0.0)),
    )


def setup_case(ctrl: Control) -> Case:
    """Read all case files and build the model, initial condition and ODE settings.

    Args:
        ctrl: run control (``run_control.json`` of the case directory).

    Returns:
        The ready-to-run case.
    """
    # ------------------------------------------------------------------
    # 1. SRH-2D files, mesh, time span and unit system
    # ------------------------------------------------------------------
    srh = read_srh_case(ctrl.case_dir / ctrl.srhhydro_file)
    mesh = build_mesh(srh.geom, srh.hydro.bc)
    tspan, dt, unit = _time_and_units(ctrl, srh)

    # ------------------------------------------------------------------
    # 2. Bed, materials, initial condition and resistance model
    # ------------------------------------------------------------------
    zb = zb_nodes_to_cells(mesh)

    # Julia zeroes the node elevations for bathymetry inversion.
    if ctrl.mode == "inversion" and ctrl.opt("active_param_names") == ["zb"]:
        zb = np.zeros_like(zb)

    mat_id = srh.mat.cell_material_ids(mesh.n_cells)
    n_zone = [srh.hydro.manning_n[i] for i in range(srh.mat.n_materials)]

    wse, wstill, qx, qy = _initial_condition(ctrl, zb)
    resistance, ks = _resistance(ctrl, mat_id)

    # ------------------------------------------------------------------
    # 3. The SWE model (first value of each inlet-Q and exit-H parameter list)
    # ------------------------------------------------------------------
    inlet_q_params = {key: values[0] for key, values in srh.hydro.iq_params.items()}
    exit_h_params = {key: values[0] for key, values in srh.hydro.ews_params.items()}
    model = build_model(
        mesh, zb, wstill, mat_id, n_zone, inlet_q_params, exit_h_params, resistance, ks, unit
    )

    # Extension: optional vegetation drag ("vegetation" block).
    vegetation = _vegetation(ctrl, srh.mat.n_materials)
    if vegetation.mode != "none":
        model = dataclasses.replace(model, vegetation=vegetation)

    # Extension: optional horizontal eddy viscosity ("turbulence" block).
    turbulence = _turbulence(ctrl)
    if turbulence.mode != "none":
        model = dataclasses.replace(model, turbulence=turbulence)

    # Extension: optional "numerics" block (scheme, spatial_order, limiter, checkpoints,
    # checkpoint_memory_gb).
    num = ctrl.raw.get("numerics", {})
    if num:
        model = dataclasses.replace(
            model,
            scheme=num.get("scheme", model.scheme),
            order=int(num.get("spatial_order", model.order)),
            limiter=num.get("limiter", model.limiter),
        )
        if model.order not in (1, 2) or (model.order == 2 and model.scheme != "hr_hllc"):
            raise ValueError("numerics: spatial_order 2 is available for scheme 'hr_hllc' only")

    if model.turbulence.mode != "none" and model.scheme != "hr_hllc":
        raise ValueError("the 'turbulence' block is available for scheme 'hr_hllc' only")

    # ------------------------------------------------------------------
    # 4. Initial state U0 = [xi, q_x, q_y] and ODE settings
    # ------------------------------------------------------------------
    U0 = jnp.stack([jnp.asarray(wse - wstill), jnp.asarray(qx), jnp.asarray(qy)])
    ode = _ode_settings(ctrl, tspan, dt)

    # Extension (numerics block): checkpointing of the reverse-mode adjoint.
    if "checkpoints" in num or "checkpoint_memory_gb" in num:
        checkpoints = num.get("checkpoints", ode.checkpoints)
        if not (checkpoints == "auto" or (isinstance(checkpoints, int) and checkpoints > 0)):
            raise ValueError("numerics.checkpoints must be a positive integer or 'auto'")

        memory_gb = float(num.get("checkpoint_memory_gb", ode.checkpoint_memory_gb))
        ode = dataclasses.replace(ode, checkpoints=checkpoints, checkpoint_memory_gb=memory_gb)

    logger.info(
        "Case ready: %d cells, %d faces, BCs %s, resistance=%s, vegetation=%s, solver=%s",
        mesh.n_cells, mesh.n_faces, mesh.bc_types, resistance.mode, model.vegetation.mode,
        ode.solver,
    )
    return Case(ctrl, srh, mesh, model, U0, ode, wse, wstill, unit)
