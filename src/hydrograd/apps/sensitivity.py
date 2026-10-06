"""Sensitivity analysis (port of ``swe_2D_sensitivity.jl``).

What it computes
----------------
The Jacobian of the final state with respect to the active parameters p (Manning's n
per zone, zb per cell, or Q per inlet):

    J = d vcat(xi, q_x, q_y)(t_end) / dp,   shape [3N, n_params],

where N is the number of cells. Rows 0..N-1 are d(xi)/dp = dh/dp (m per unit of p),
rows N..2N-1 are d(q_x)/dp and rows 2N..3N-1 are d(q_y)/dp (m^2/s per unit of p).

How
---
* Transient runs: forward-mode AD through the ODE solve (Julia: ``ForwardDiff.jacobian``
  with ``Tsit5``), ``JVP_CHUNK`` tangent directions at a time to bound memory.
* ``SteadyNewton()``: the implicit function theorem, dU*/dp = -(dR/dU)^{-1} dR/dp
  (one sparse LU).
* Implicit theta-schemes are not supported (reverse-mode gradients only).

Files written
-------------
Julia names: ``sensitivity_results.json`` (``sensitivity_results`` flattened
column-major, ``parameter_name``, ``params_vector``),
``sensitivity_results_<name>_<i>.{json,vtk}`` per parameter, and
``forward_simulation_results.json`` (final frames at the evaluation point).
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from hydrograd.apps.params import initial_params, to_theta
from hydrograd.apps.setup import Case
from hydrograd.io.output import write_json, write_vtk
from hydrograd.solvers.final import final_state, is_implicit, is_steady
from hydrograd.solvers.ode import solve_swe
from hydrograd.solvers.steady import steady_jacobian

logger = logging.getLogger(__name__)

__all__ = ["DERIV_RTOL", "final_state_jacobian", "run_sensitivity"]

DERIV_RTOL = 1.0e-7  # relative tolerance for derivative computations
JVP_CHUNK = 64       # parameters per batch of forward-mode tangents (bounds memory)


def _chunked_jacfwd(
    f: Callable[[jax.Array], jax.Array], p: jax.Array, chunk: int
) -> np.ndarray:
    """Jacobian of f at p by forward mode, ``chunk`` tangents at a time.

    Same result as ``jax.jacfwd`` but memory grows with ``chunk``, not with len(p)
    (e.g. per-cell zb parameters).

    Args:
        f: function p [n_params] -> output [n_out].
        p: evaluation point, shape [n_params].
        chunk: number of tangent directions per batch.

    Returns:
        Jacobian df/dp, shape [n_out, n_params].
    """
    n = p.shape[0]

    def jvp_column(tangent: jax.Array) -> jax.Array:
        """Directional derivative of f at p along ``tangent`` [n_params]; shape [n_out]."""
        return jax.jvp(f, (p,), (tangent,))[1]

    jvps = jax.jit(jax.vmap(jvp_column))

    # One batch of unit tangents per chunk of parameters. The batch always has ``chunk``
    # rows (zero padding in the last batch), so the function is compiled only once.
    cols = []
    for k in range(0, n, chunk):
        idx = np.arange(k, min(k + chunk, n))
        T = np.zeros((chunk, n))
        T[np.arange(len(idx)), idx] = 1.0

        batch_columns = np.asarray(jvps(jnp.asarray(T)))
        cols.append(batch_columns[:len(idx)])

    return np.concatenate(cols, axis=0).T


def final_state_jacobian(case: Case, name: str, p: jax.Array) -> dict[str, Any]:
    """d(final state)/dp (see the module docstring).

    Args:
        case: the case.
        name: active parameter name (``"ManningN"``, ``"zb"`` or ``"Q"``).
        p: evaluation point, shape [n_params].

    Returns:
        Dict with ``"J"``, shape [3 n_cells, n_params], and the solution frames ``"ys"``,
        shape [n_frames, 3, n_cells].
    """
    # Julia's ForwardDiff duals enter SciML's step-size error norm, so its derivatives
    # are controlled to the nominal tolerance; diffrax controls only the primal error.
    # A tighter tolerance gives comparable derivative accuracy (validated vs Julia: ~1e-5).
    m = case.model

    if is_implicit(case.ode):
        raise NotImplementedError(
            "sensitivity analysis uses forward-mode Jacobians; the implicit "
            "theta-scheme has reverse-mode (IFT) gradients only. Use SSPRK3() "
            "or SteadyNewton(), or jax.jacrev on solve_implicit."
        )

    # ------------------------------------------------------------------
    # 1. Steady state: dU*/dp = -J^{-1} dR/dp (one sparse LU)
    # ------------------------------------------------------------------
    if is_steady(case.ode):
        t0 = time.perf_counter()

        def steady_state(p: jax.Array) -> Any:
            """Steady state and convergence flag for parameters p [n_params]."""
            return final_state(m, to_theta(name, p), case.U0, case.ode)

        U, ok = jax.jit(steady_state)(p)
        if not bool(ok):
            raise RuntimeError("steady Newton solve did not converge")

        J = steady_jacobian(m, to_theta(name, p), U)
        logger.info("steady sensitivity Jacobian %s computed in %.2f s", J.shape,
                    time.perf_counter() - t0)
        return {"J": J, "ys": np.stack([np.asarray(case.U0), np.asarray(U)])}

    # ------------------------------------------------------------------
    # 2. Transient: forward-mode AD through the ODE solve
    # ------------------------------------------------------------------
    s = dataclasses.replace(case.ode, adjoint="forward", rtol=min(case.ode.rtol, DERIV_RTOL),
                            atol=min(case.ode.atol, DERIV_RTOL * 1.0e-3))

    def final_flat(p: jax.Array) -> jax.Array:
        """Final state vcat(xi, q_x, q_y) for parameters p [n_params]; shape [3 n_cells]."""
        sol = solve_swe(m, to_theta(name, p), case.U0, s)
        assert sol.ys is not None
        return sol.ys[-1].reshape(-1)

    def solve_at(p: jax.Array) -> Any:
        """Full ODE solution (all saved frames) for parameters p [n_params]."""
        return solve_swe(m, to_theta(name, p), case.U0, s)

    t0 = time.perf_counter()
    J = _chunked_jacfwd(final_flat, p, JVP_CHUNK)

    # The solution frames at the evaluation point (for the output file).
    sol = jax.jit(solve_at)(p)
    assert sol.ys is not None

    logger.info("sensitivity Jacobian %s computed in %.2f s", J.shape, time.perf_counter() - t0)
    return {"J": J, "ys": np.asarray(sol.ys)}


def run_sensitivity(case: Case) -> dict[str, Any]:
    """Run the sensitivity analysis and write Julia-compatible outputs.

    Args:
        case: the case.

    Returns:
        The result of :func:`final_state_jacobian` (``"J"`` and ``"ys"``).
    """
    # Evaluation point and the Jacobian there.
    name, p = initial_params(case, "sensitivity")
    res = final_state_jacobian(case, name, p)
    J = res["J"]
    n = case.model.n_cells
    d = case.ctrl.case_dir
    pv = np.asarray(p)

    # Whole Jacobian.
    write_json(d / "sensitivity_results.json",
               {"sensitivity_results": J, "parameter_name": name, "params_vector": pv})

    # One JSON and one VTK file per parameter: the columns split into dh, d(hu), d(hv).
    for i in range(J.shape[1]):
        dh = J[:n, i]
        dhu = J[n:2 * n, i]
        dhv = J[2 * n:, i]
        tag = f"sensitivity_results_{name}_{i + 1}"

        write_json(d / f"{tag}.json", {"dh_dparam": dh, "dhu_dparam": dhu, "dhv_dparam": dhv,
                                       "parameter_name": name, "parameter_value": float(pv[i])})
        write_vtk(d / f"{tag}.vtk", case.mesh,
                  {"dh_dparam": dh, "dhu_dparam": dhu, "dhv_dparam": dhv})

    # Solution frames at the evaluation point, one column per frame.
    ys = res["ys"]
    write_json(d / "forward_simulation_results.json",
               {"forward_simulation_results": ys.reshape(len(ys), -1).T,
                "zb_cells": np.asarray(case.model.zb), "wstill": case.wstill,
                "hstill": np.asarray(case.model.hstill)})

    return res
