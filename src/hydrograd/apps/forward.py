"""Forward simulation (port of ``swe_2D_forward_simulation.jl`` + post-processing).

The mode integrates the SWEs from the initial condition with the parameters given in
the case files (no differentiation). Three solver families are supported:

* ``SteadyNewton()``: Newton iteration to the steady state R(U) = 0;
* ``BackwardEuler()`` / ``CrankNicolson()``: implicit theta-scheme with fixed dt;
* any explicit solver (``SSPRK3()``, ``Tsit5()``, ...) through diffrax.

Derived fields
--------------
For each saved frame, with ``h = xi + h_still`` (m, *not* clipped):

    u = q_x / (h + h_small),   v = q_y / (h + h_small)   (m/s)
    WSE = xi + w_still                                     (m)

plus Manning's n per cell and the friction terms (Julia post-processing conventions).

Files written
-------------
In the case directory, named as in Hydrograd.jl:

* ``forward_simulation_results_%04d.vtk`` for each saved frame (1-based);
* the truth JSON (``forward_simulation_save_solution_truth_file_name``) with the
  same keys as Julia, from the last frame;
* ``total_water_volume.csv`` (sum h A per frame, m^3);
* ``forward_simulation_solution.npz`` (all frames; replaces Julia's .jld2).

Difference from Julia: ``friction_x/y_truth`` use the *actual* (possibly
variable) n; Julia always uses the constant zone n there.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hydrograd.apps.setup import Case
from hydrograd.io.output import write_csv, write_json, write_vtk
from hydrograd.params.manning import manning_law
from hydrograd.physics.model import SWEModel
from hydrograd.physics.rhs import (
    _safe_speed,
    clipped_state,
    friction,
    manning_n_cells,
    still_depth,
)
from hydrograd.solvers.final import (
    STEADY_TOL,
    implicit_settings,
    is_implicit,
    is_steady,
    newton_settings,
)
from hydrograd.solvers.implicit import solve_implicit
from hydrograd.solvers.ode import solve_swe
from hydrograd.solvers.steady import newton_solve

logger = logging.getLogger(__name__)

__all__ = ["derived_fields", "run_forward"]


def derived_fields(
    m: SWEModel, theta: dict[str, Any], U: np.ndarray, wstill: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-cell diagnostic fields of one state (Julia post-processing conventions).

    Velocities are ``q/(h + h_small)``; ``h = xi + h_still`` *without* clipping. Manning's
    n and the friction terms use the clipped state, as in the residual.

    Args:
        m: the model.
        theta: differentiable parameters (empty dict for the file values).
        U: state [xi, q_x, q_y], shape [3, n_cells].
        wstill: still-water elevation (m), shape [n_cells].

    Returns:
        Dict of NumPy arrays, each of shape [n_cells]: ``xi``, ``wstill``, ``hstill``,
        ``h``, ``wse`` (m); ``u``, ``v`` (m/s); ``q_x``, ``q_y`` (m^2/s); ``zb`` (m);
        ``ManningN`` (s/m^(1/3)); ``friction_x``, ``friction_y`` (momentum source,
        m^2/s^2). For an n-law model also ``ks``, ``h_ks`` (m), ``friction_factor`` (-)
        and ``Re`` (-).
    """
    Uj = jnp.asarray(U)

    # Still-water depth, clipped state (for n and friction) and unclipped depth.
    hs = still_depth(m, theta)
    _, hc, qxc, qyc = clipped_state(m, Uj, hs)
    h = np.asarray(Uj[0] + hs)

    # Manning's n and the friction source terms.
    n = manning_n_cells(m, theta, hc, qxc, qyc)
    fx, fy = friction(m, theta, hc, qxc, qyc, n)

    # Collect the fields (the key order sets the order in the output files).
    xi = np.asarray(Uj[0])
    qx = np.asarray(Uj[1])
    qy = np.asarray(Uj[2])
    out = {
        "xi": xi,
        "wstill": wstill,
        "hstill": np.asarray(hs),
        "h": h,
        "wse": np.asarray(Uj[0]) + wstill,
        "u": qx / (h + m.h_small),
        "v": qy / (h + m.h_small),
        "q_x": np.asarray(Uj[1]),
        "q_y": np.asarray(Uj[2]),
        "zb": np.asarray(theta.get("zb", m.zb)),
        "ManningN": np.asarray(n),
        "friction_x": np.asarray(fx),
        "friction_y": np.asarray(fy),
    }

    # Extra diagnostics of the n(h, |U|, ks) law.
    if m.resistance.mode == "law":
        assert m.resistance.law is not None
        speed = _safe_speed(qxc, qyc, hc)
        _, h_ks, friction_factor, reynolds = manning_law(m.resistance.law, hc, speed, m.ks)
        out.update(
            ks=np.asarray(m.ks),
            h_ks=np.asarray(h_ks),
            friction_factor=np.asarray(friction_factor),
            Re=np.asarray(reynolds),
        )

    return out


def _truth_dict(case: Case, fields: dict[str, np.ndarray]) -> dict[str, Any]:
    """Contents of the truth JSON file, with Hydrograd.jl's keys.

    Args:
        case: the case.
        fields: derived fields of the last frame (see :func:`derived_fields`).

    Returns:
        Dict of per-cell arrays (``*_truth``), plus the zone Manning's n
        (s/m^(1/3)) and the inlet discharges (m^3/s). Law diagnostics that the model
        does not compute are written as zeros.
    """
    m = case.model
    zeros = np.zeros(m.n_cells)

    return {
        "xi_truth": fields["xi"],
        "wstill_truth": fields["wstill"],
        "hstill_truth": fields["hstill"],
        "wse_truth": fields["wse"],
        "h_truth": fields["h"],
        "u_truth": fields["u"],
        "v_truth": fields["v"],
        "zb_cell_truth": fields["zb"],
        "ManningN_cells_truth": fields["ManningN"],
        "h_ks_cells_truth": fields.get("h_ks", zeros),
        "friction_factor_cells_truth": fields.get("friction_factor", zeros),
        "Re_cells_truth": fields.get("Re", zeros),
        "friction_x_truth": fields["friction_x"],
        "friction_y_truth": fields["friction_y"],
        "S0_cells_truth": np.asarray(m.S0),
        "ManningN_zone_values_truth": np.asarray(m.n_zone),
        "inlet_discharges_truth": np.asarray(m.Q),
    }


def write_frame_vtk(case: Case, path: str, fields: dict[str, np.ndarray]) -> None:
    """Write a VTK file with the main cell fields of one frame.

    Args:
        case: the case (mesh, model, case directory).
        path: file name relative to the case directory.
        fields: derived fields of the frame (see :func:`derived_fields`).
    """
    names = ["xi", "wstill", "hstill", "h", "q_x", "q_y", "ManningN", "zb", "wse",
             "friction_x", "friction_y", "ks", "h_ks", "friction_factor", "Re"]

    # Scalar cell fields present in this frame; "wse" is written as "WSE".
    scalars = {}
    for name in names:
        if name in fields:
            vtk_name = name if name != "wse" else "WSE"
            scalars[vtk_name] = fields[name]

    # Vector cell fields: velocity (m/s) and bed slope (-).
    vectors = {"U": np.stack([fields["u"], fields["v"]], 1), "slope": np.asarray(case.model.S0)}

    write_vtk(case.ctrl.case_dir / path, case.mesh, scalars, vectors)


def run_forward(case: Case) -> dict[str, Any]:
    """Run the forward simulation and write all outputs.

    Args:
        case: the case.

    Returns:
        Dict with ``ts`` (s), shape [n_frames]; ``ys``, shape [n_frames, 3, n_cells];
        the last-frame ``fields``; and the wall time ``wall_s`` (s, including compilation).
    """
    m = case.model
    s = case.ode
    theta: dict[str, Any] = {}
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Solve: steady Newton, implicit theta-scheme, or explicit diffrax
    # ------------------------------------------------------------------
    if is_steady(s):

        def steady_solve(U0: Array) -> Any:
            """Newton solve to the steady state from U0 [3, n_cells]."""
            return newton_solve(m, theta, U0, newton_settings(s))

        U, it, rn = jax.jit(steady_solve)(case.U0)
        wall = time.perf_counter() - t0

        if not float(rn) < STEADY_TOL:
            raise RuntimeError(f"steady Newton solve did not converge (max|R| = {float(rn):.2e} "
                               f"after {int(it)} iterations); no results written")
        logger.info("steady state: %d Newton iterations, max|R| %.1e, %.2f s incl. compile",
                    int(it), float(rn), wall)

        # Two frames: the initial condition and the steady state.
        ys = np.stack([np.asarray(case.U0), np.asarray(U)])
        ts = np.array([s.t0, s.t1])

    elif is_implicit(s):

        def implicit_solve(U0: Array) -> Any:
            """Implicit theta-scheme time integration from U0 [3, n_cells]."""
            return solve_implicit(m, theta, U0, implicit_settings(s))

        ts_j, ys_j = jax.jit(implicit_solve)(case.U0)
        ys = np.asarray(ys_j)
        ts = np.asarray(ts_j)
        wall = time.perf_counter() - t0

        if not np.all(np.isfinite(ys)):
            raise RuntimeError(
                "implicit solve failed (non-finite state); reduce dt. No results written."
            )
        logger.info("implicit %s: dt = %g s, %.2f s incl. compile", s.solver,
                    implicit_settings(s).steps()[1], wall)

    else:

        def explicit_solve(U0: Array) -> Any:
            """Explicit (diffrax) time integration from U0 [3, n_cells]."""
            return solve_swe(m, theta, U0, s)

        sol = jax.jit(explicit_solve)(case.U0)
        ys = np.asarray(sol.ys)
        ts = np.asarray(sol.ts)
        wall = time.perf_counter() - t0

        if not bool(sol.result == dfx.RESULTS.successful):
            raise RuntimeError(f"ODE solve failed ({sol.result}); no results written. "
                               "Check dt, tspan and the max_steps limit.")
        logger.info("forward simulation: %d steps (%d rejected), %.2f s incl. compile",
                    int(sol.stats["num_steps"]), int(sol.stats["num_rejected_steps"]), wall)

    # ------------------------------------------------------------------
    # 2. Per-frame VTK files and the total water volume sum(h A) (m^3)
    # ------------------------------------------------------------------
    vol = []
    for k, U in enumerate(ys):
        fields = derived_fields(m, theta, U, case.wstill)
        write_frame_vtk(case, f"forward_simulation_results_{k + 1:04d}.vtk", fields)

        total_volume = float(np.sum(fields["h"] * np.asarray(m.area)))
        vol.append((float(ts[k]), total_volume))

    # ------------------------------------------------------------------
    # 3. Volume history, all frames, and the truth file (last frame)
    # ------------------------------------------------------------------
    d = case.ctrl.case_dir
    write_csv(d / "total_water_volume.csv", ["t", "total_water_volume"], vol)
    np.savez_compressed(d / "forward_simulation_solution.npz", t=ts, u=ys)

    truth_file = case.ctrl.opt("forward_simulation_save_solution_truth_file_name")
    write_json(d / truth_file, _truth_dict(case, fields))

    return {"ts": ts, "ys": ys, "fields": fields, "wall_s": wall}
