"""Shared case setup for all run modes: files -> mesh -> bed -> IC -> SWEModel.

Follows ``solve_swe_2D.jl``. Differences from Julia, all inert for its examples:
the ``.srhhydro`` time/unit branch (``bUse_srhhydro_time_settings = true``)
reads ``SimTime`` in SRH order and honours ``GridUnit`` (Julia swaps the
SimTime fields and its unit check always fails).
"""

from __future__ import annotations

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
from hydrograd.physics.model import Resistance, SWEModel, build_model
from hydrograd.solvers.ode import ODESettings

logger = logging.getLogger(__name__)

__all__ = ["Case", "setup_case"]

# Per-mode prefixes of the IC and ODE option keys.
_IC_PREFIX = {"forward": "forward_simulation_initial_condition",
              "inversion": "inversion_forward_simulation_initial_condition",
              "sensitivity": "sensitivity_forward_simulation_initial_condition",
              "ude": "UDE_forward_simulation_initial_condition"}
_ODE_BLOCK = {"inversion": "inversion_ode_solver_options",
              "sensitivity": "sensitivity_ode_solver_options",
              "ude": "UDE_ode_solver_options"}
_UNIT_NAMES = {"meters": "SI", "meter": "SI", "feet": "EN", "foot": "EN"}


@dataclass(frozen=True)
class Case:
    """Everything a run mode needs.

    Attributes:
        ctrl: run control; srh: SRH-2D files; mesh: mesh.
        model: static SWE model (defaults = truth parameters from the files).
        U0: initial state [xi, q_x, q_y], shape [3, n_cells].
        ode: time-integration settings.
        wse0: initial WSE (m); wstill: still-water elevation (m) at cells.
        unit_system: ``SI`` or ``EN``.
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
    if not ctrl.use_srhhydro_time:
        if _UNIT_NAMES.get(srh.geom.grid_unit.lower()) == "EN":
            logger.warning("GridUnit is %s but SI constants (g = 9.81, k_n = 1) are used, as in "
                           "Hydrograd.jl when bUse_srhhydro_time_settings is false", srh.geom.grid_unit)
        return ctrl.tspan, ctrl.dt, "SI"
    t_start_h, dt_s, t_end_h = srh.hydro.sim_time
    unit = _UNIT_NAMES.get(srh.geom.grid_unit.lower())
    if unit is None:
        raise ValueError(f"unsupported GridUnit '{srh.geom.grid_unit}'")
    return (3600.0 * t_start_h, 3600.0 * t_end_h), dt_s, unit


def _initial_condition(ctrl: Control, zb: np.ndarray) -> tuple[np.ndarray, ...]:
    """(wse, wstill, q_x, q_y) per cell (``process_ICs_2D.jl``)."""
    p = _IC_PREFIX[ctrl.mode]
    n = len(zb)
    kind = ctrl.opt(f"{p}_options")
    if kind == "constant":
        vals = ctrl.opt(f"{p}_constant_values")
        if len(vals) < 4:
            raise ValueError(f"{p}_constant_values needs [wse, wstill, q_x, q_y], got {vals}")
        wse, wstill, qx, qy = (np.full(n, float(v)) for v in vals[:4])
    elif kind == "from_file":
        d = json.loads((ctrl.case_dir / ctrl.opt(f"{p}_file_name")).read_text())
        wse, wstill, qx, qy = (np.asarray(d[k], dtype=float) for k in ("wse", "wstill", "q_x", "q_y"))
    else:
        raise ValueError(f"unknown IC option '{kind}'")
    if np.any(wse < zb - 1.0e-12):
        raise ValueError("initial condition has WSE below the bed elevation")
    n_dry = int(np.sum(wse - zb < 1.0e-3))
    if n_dry:  # Hydrograd.jl requires a wet IC; dry cells are handled by the h_small clipping
        logger.info("initial condition has %d dry cells (h < h_small)", n_dry)
    return wse, wstill, qx, qy


def _ode_settings(ctrl: Control, tspan: tuple[float, float], dt: float) -> ODESettings:
    if ctrl.mode == "forward":
        solver = ctrl.opt("forward_simulation_ode_solver")
        if ctrl.opt("forward_simulation_solver") == "customized":
            logger.warning("'customized' solver mapped to fixed-step Euler()")
            solver, adaptive = "Euler()", False
        else:
            adaptive = bool(ctrl.opt("forward_simulation_adaptive"))
        return ODESettings(tspan[0], tspan[1], dt, int(ctrl.opt("forward_simulation_nSave")),
                           solver, adaptive)
    o = ctrl.opt(_ODE_BLOCK[ctrl.mode])
    return ODESettings(tspan[0], tspan[1], dt, int(o["ode_solver_nSave"]), o["ode_solver"],
                       bool(o["ode_solver_adaptive"]))


def _ks_cells(ks_list: Any, mat_id: np.ndarray) -> np.ndarray:
    """ks per cell; ``ks_list[i]`` is the roughness height (m) of material i."""
    return np.asarray(ks_list, dtype=float)[mat_id]


def _resistance(ctrl: Control, mat_id: np.ndarray) -> tuple[Resistance, Any]:
    if ctrl.mode == "forward" and ctrl.opt("forward_simulation_ManningN_option") == "variable":
        params: dict[str, Any] = dict(ctrl.opt("forward_simulation_ManningN_function_parameters"))
        kind = ctrl.opt("forward_simulation_ManningN_function_type")
        ks = _ks_cells(params.pop("ks"), mat_id) if kind == "h_Umag_ks" else None
        law = ManningLaw(kind, {k: float(v) for k, v in params.items() if not isinstance(v, list)})
        return Resistance(mode="law", law=law), ks
    if ctrl.mode == "ude":
        choice = ctrl.opt("UDE_choice")
        nn = ctrl.opt("UDE_NN_config")
        b = lambda k: tuple(float(v) for v in nn.get(k, (0.0, 1.0)))
        ks = None
        if choice == "ManningN_h_Umag_ks":
            ks = _ks_cells(ctrl.opt("UDE_ManningN_function_parameters")["ks"], mat_id)
        vb = b("velocity_magnitude_bounds")
        hb = b("h_bounds")
        if nn.get("layernorm_axes", "features") not in ("features", "all"):
            raise ValueError("UDE_NN_config.layernorm_axes must be 'features' or 'all'")
        return Resistance(mode=f"nn_{choice}", activations=tuple(nn["activations"]),
                          out_bounds=b("output_bounds"), h_bounds=hb, umag_bounds=b("Umag_bounds"),
                          ks_bounds=b("ks_bounds"), q_bounds=(hb[0] * vb[0], hb[1] * vb[1]),
                          ln_axes=nn.get("layernorm_axes", "features")), ks
    return Resistance(mode="zone"), None


def setup_case(ctrl: Control) -> Case:
    """Read all case files and build the model, IC and ODE settings."""
    srh = read_srh_case(ctrl.case_dir / ctrl.srhhydro_file)
    mesh = build_mesh(srh.geom, srh.hydro.bc)
    tspan, dt, unit = _time_and_units(ctrl, srh)
    zb = zb_nodes_to_cells(mesh)
    if ctrl.mode == "inversion" and ctrl.opt("active_param_names") == ["zb"]:
        zb = np.zeros_like(zb)  # Julia zeroes the node elevations for bathymetry inversion
    mat_id = srh.mat.cell_material_ids(mesh.n_cells)
    n_zone = [srh.hydro.manning_n[i] for i in range(srh.mat.n_materials)]
    wse, wstill, qx, qy = _initial_condition(ctrl, zb)
    resistance, ks = _resistance(ctrl, mat_id)
    model = build_model(mesh, zb, wstill, mat_id, n_zone,
                        {k: v[0] for k, v in srh.hydro.iq_params.items()},
                        {k: v[0] for k, v in srh.hydro.ews_params.items()},
                        resistance, ks, unit)
    num = ctrl.raw.get("numerics", {})   # extension: scheme, spatial_order, limiter, checkpoints, checkpoint_memory_gb
    if num:
        import dataclasses
        model = dataclasses.replace(model, scheme=num.get("scheme", model.scheme),
                                    order=int(num.get("spatial_order", model.order)),
                                    limiter=num.get("limiter", model.limiter))
        if model.order not in (1, 2) or (model.order == 2 and model.scheme != "hr_hllc"):
            raise ValueError("numerics: spatial_order 2 is available for scheme 'hr_hllc' only")
    U0 = jnp.stack([jnp.asarray(wse - wstill), jnp.asarray(qx), jnp.asarray(qy)])
    ode = _ode_settings(ctrl, tspan, dt)
    if "checkpoints" in num or "checkpoint_memory_gb" in num:   # extension (numerics block)
        import dataclasses
        ck = num.get("checkpoints", ode.checkpoints)
        if not (ck == "auto" or (isinstance(ck, int) and ck > 0)):
            raise ValueError("numerics.checkpoints must be a positive integer or 'auto'")
        ode = dataclasses.replace(ode, checkpoints=ck,
                                  checkpoint_memory_gb=float(num.get("checkpoint_memory_gb", ode.checkpoint_memory_gb)))
    logger.info("Case ready: %d cells, %d faces, BCs %s, resistance=%s, solver=%s",
                mesh.n_cells, mesh.n_faces, mesh.bc_types, resistance.mode, ode.solver)
    return Case(ctrl, srh, mesh, model, U0, ode, wse, wstill, unit)
