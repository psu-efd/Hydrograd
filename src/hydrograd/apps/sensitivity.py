"""Sensitivity analysis (port of ``swe_2D_sensitivity.jl``).

Computes the Jacobian of the final state ``vcat(xi, q_x, q_y)`` (length 3N)
with respect to the active parameters with forward-mode AD through the ODE
solve (Julia: ``ForwardDiff.jacobian`` with ``Tsit5``).

Outputs (Julia names): ``sensitivity_results.json`` (``sensitivity_results``
flattened column-major, ``parameter_name``, ``params_vector``),
``sensitivity_results_<name>_<i>.{json,vtk}`` per parameter, and
``forward_simulation_results.json`` (final frames at the evaluation point).
"""

from __future__ import annotations

import dataclasses
import logging
import time
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


def _chunked_jacfwd(f, p: jax.Array, chunk: int) -> np.ndarray:
    """Jacobian of f at p by forward mode, ``chunk`` tangents at a time.

    Same result as ``jax.jacfwd`` but memory grows with ``chunk``, not with len(p)
    (e.g. per-cell zb parameters).
    """
    n = p.shape[0]
    jvps = jax.jit(jax.vmap(lambda t: jax.jvp(f, (p,), (t,))[1]))
    cols = []
    for k in range(0, n, chunk):
        idx = np.arange(k, min(k + chunk, n))
        T = np.zeros((chunk, n))
        T[np.arange(len(idx)), idx] = 1.0       # fixed batch shape: one compilation
        cols.append(np.asarray(jvps(jnp.asarray(T)))[:len(idx)])
    return np.concatenate(cols, axis=0).T


def final_state_jacobian(case: Case, name: str, p: jax.Array) -> dict[str, Any]:
    """d(final state)/dp by forward-mode AD; returns J [3N, n_params] and the solution."""
    # Julia's ForwardDiff duals enter SciML's step-size error norm, so its derivatives
    # are controlled to the nominal tolerance; diffrax controls only the primal error.
    # A tighter tolerance gives comparable derivative accuracy (validated vs Julia: ~1e-5).
    m = case.model
    if is_implicit(case.ode):
        raise NotImplementedError("sensitivity analysis uses forward-mode Jacobians; the implicit "
                                  "theta-scheme has reverse-mode (IFT) gradients only. Use SSPRK3() "
                                  "or SteadyNewton(), or jax.jacrev on solve_implicit.")
    if is_steady(case.ode):  # steady state: dU*/dp = -J^{-1} dR/dp (one sparse LU)
        t0 = time.perf_counter()
        U, ok = jax.jit(lambda p: final_state(m, to_theta(name, p), case.U0, case.ode))(p)
        if not bool(ok):
            raise RuntimeError("steady Newton solve did not converge")
        J = steady_jacobian(m, to_theta(name, p), U)
        logger.info("steady sensitivity Jacobian %s computed in %.2f s", J.shape, time.perf_counter() - t0)
        return {"J": J, "ys": np.stack([np.asarray(case.U0), np.asarray(U)])}
    s = dataclasses.replace(case.ode, adjoint="forward", rtol=min(case.ode.rtol, DERIV_RTOL),
                            atol=min(case.ode.atol, DERIV_RTOL * 1.0e-3))

    def final_flat(p: jax.Array) -> jax.Array:
        sol = solve_swe(m, to_theta(name, p), case.U0, s)
        assert sol.ys is not None
        return sol.ys[-1].reshape(-1)

    t0 = time.perf_counter()
    J = _chunked_jacfwd(final_flat, p, JVP_CHUNK)
    sol = jax.jit(lambda p: solve_swe(m, to_theta(name, p), case.U0, s))(p)
    assert sol.ys is not None
    logger.info("sensitivity Jacobian %s computed in %.2f s", J.shape, time.perf_counter() - t0)
    return {"J": J, "ys": np.asarray(sol.ys)}


def run_sensitivity(case: Case) -> dict[str, Any]:
    """Run the sensitivity analysis and write Julia-compatible outputs."""
    name, p = initial_params(case, "sensitivity")
    res = final_state_jacobian(case, name, p)
    J, n = res["J"], case.model.n_cells
    d = case.ctrl.case_dir
    pv = np.asarray(p)
    write_json(d / "sensitivity_results.json",
               {"sensitivity_results": J, "parameter_name": name, "params_vector": pv})
    for i in range(J.shape[1]):
        dh, dhu, dhv = J[:n, i], J[n:2 * n, i], J[2 * n:, i]
        tag = f"sensitivity_results_{name}_{i + 1}"
        write_json(d / f"{tag}.json", {"dh_dparam": dh, "dhu_dparam": dhu, "dhv_dparam": dhv,
                                       "parameter_name": name, "parameter_value": float(pv[i])})
        write_vtk(d / f"{tag}.vtk", case.mesh, {"dh_dparam": dh, "dhu_dparam": dhu, "dhv_dparam": dhv})
    ys = res["ys"]
    write_json(d / "forward_simulation_results.json",
               {"forward_simulation_results": ys.reshape(len(ys), -1).T,
                "zb_cells": np.asarray(case.model.zb), "wstill": case.wstill,
                "hstill": np.asarray(case.model.hstill)})
    return res
