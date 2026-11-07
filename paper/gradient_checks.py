"""Gradient verification for the Hydrograd v2.0 paper (Table 1: AD vs finite differences).

Usage (with the Hydrograd venv active; about 1-2 h on 8 CPU cores)::

    JAX_PLATFORMS=cpu python paper/gradient_checks.py [check ...]

Writes ``paper/results/gradient_checks.json`` (one entry per check; a partial file is
written after every check). Without arguments all checks run. The earlier version of this
script, whose rows ii-v used synthetic cases that appear nowhere else in the paper, is kept
as ``paper/results/gradient_checks_synthetic_cases.json``.

Every check uses a case of the paper's Results section, so that no row of the table rests
on a configuration that is not described elsewhere.

What is checked
---------------
Each check differentiates a scalar loss L(p) = sum_i w_i X_i(p), a fixed random weighting
(w ~ N(0, 1), numpy seed 0) of the steady or final state X (either the whole state
U = [xi, q_x, q_y] in m and m^2/s, or the final depth h in m, as stated per check), with
respect to physical parameters p. The AD derivative dL/dp_k is compared with the central
finite difference

    FD_k = (L(p + eps e_k) - L(p - eps e_k)) / (2 eps),

and the relative difference |AD_k - FD_k| / |FD_k| is reported per parameter. Costs are
wall-clock times of jitted calls after compilation (the first call, which includes the
compilation, is stored separately).

(i)   Steady state by pseudo-transient Newton + implicit-function-theorem (IFT) adjoint,
      dU*/dp = -(dR/dU)^{-1} dR/dp. Savannah River of Sect. 4.2 (1,306 cells), p = Manning's
      n of the five zones (s/m^(1/3)); and, for the steady-solver verification of Sect. 4.1,
      the transcritical bump (200 cells), p = n of the channel. Both also compare the Newton
      steady state with a long Tsit5 run (rtol 1e-8).
(ii)  Steady IFT on the closure-learning case of Sect. 4.4 (``fig06_steady_ude.py``, refined
      bump, 200 cells, steady Newton): p = the 22 weights of the NN closure n = N(h), at the
      initial weights of the training.
(iii) Explicit SSPRK3 (CFL steps, second order, point-implicit friction) with checkpointed
      reverse mode (diffrax RecursiveCheckpointAdjoint) through the 2 h flood of the
      hydrograph-inversion valley of Sect. 4.5 (500 cells): p = the 9 inflow knots Q_k
      (m^3/s, 15 min apart) at the true hydrograph. The loss weights the depths saved every
      15 min, so that every knot acts on it (a final-state loss would be insensitive to the
      early knots once the flood wave has passed).
(iv)  Implicit backward Euler (dt = 60 s, Newton tol 1e-6 and 1e-10, the inversion model of
      Sect. 4.5) with the per-step IFT adjoint, same case, loss and parameters as (iii), at
      the hydrograph recovered by the noise-free inversion (the knots the inversion model
      actually solved at). At the exact true knots, backward Euler with dt = 60 s diverges in
      the fourth step (t = 180 to 240 s, when the flood first climbs the 2:5 bank: Newton
      returns a non-finite residual for every damping fraction, while a 30 s step from the
      same state converges), so the true knots are checked with dt = 30 s.
(v)   Forward mode (JVP per weight) against checkpointed reverse mode and FD on the
      time-marching training configuration of Sect. 4.4 (Tsit5, rtol 1e-7, 100 s): p = the
      22 NN weights. FD at two steps: the AD modes differentiate the rtol 1e-7 discrete
      solution while FD samples rtol 1e-10 runs, so the FD comparison carries an O(rtol / eps)
      floor that the larger step lowers.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import hydrograd  # noqa: F401  (enables float64)
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.io.synthetic import write_channel_case

# Paths: this folder, the repository, the case input files and the output file.
PAPER_DIR = Path(__file__).resolve().parent
REPO_DIR = PAPER_DIR.parent
CASES_DIR = PAPER_DIR / "cases"     # input files of the paper cases (read only)
RESULT_FILE = PAPER_DIR / "results" / "gradient_checks.json"

# Closure-learning cases of Sect. 4.4, as written by fig06_steady_ude.py (steady and
# transient training folders with the steady truth file).
UDE_DATA_DIR = PAPER_DIR / "data" / "fig06_steady_ude"
# Hydrograph recovered by the noise-free inversion of Sect. 4.5 (prep_hydrograph_inversion.py).
INVERSION_SUMMARY = PAPER_DIR / "data" / "hydrograph_inversion" / "summary_noise0mm.json"

# Hydrograph rows: frames saved every 15 min (the knot spacing) enter the loss, and the true
# knots are run implicitly with a halved step (dt = 60 s diverges at the first bank climb).
N_SAVE_HYDROGRAPH = 8
DT_IMPLICIT_TRUE_KNOTS = 30.0        # s

# Finite differences through adaptive Tsit5 runs: tolerance and step.
FD_RTOL = 1.0e-10
FD_STEPS_WEIGHTS_TRANSIENT = (1.0e-4, 1.0e-3)   # NN weights (-), transient row
FD_STEP_WEIGHTS_STEADY = 1.0e-5      # NN weights (-), steady row (1e-6 left FD round-off of 5e-6 on the smallest components)
FD_STEP_ZONE_N = 1.0e-6              # Manning's n (s/m^(1/3))
FD_STEPS_Q_EXPLICIT = (1.0e-3, 1.0e-2)   # hydrograph knots (m^3/s), explicit row
FD_STEP_Q_IMPLICIT = 5.0e-4              # hydrograph knots (m^3/s), implicit row

sys.path.insert(0, str(PAPER_DIR))
sys.path.insert(0, str(REPO_DIR / "benchmarks" / "validation"))
from fig03_verification import hardware_info


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def copy_example(rel: str, work_dir: Path) -> Path:
    """Copy a paper case into ``work_dir`` (never write into ``paper/cases``).

    Args:
        rel: case folder name under ``paper/cases``.
        work_dir: scratch folder.

    Returns:
        The copied case folder.
    """
    destination = work_dir / rel.replace("/", "_")
    shutil.copytree(CASES_DIR / rel, destination,
                    ignore=shutil.ignore_patterns("*.gif", "*.h5", "SRH_2D"))
    return destination


def timed(function: Callable, *args: Any) -> tuple[Any, float]:
    """Call ``function(*args)``, wait for the JAX result and measure the wall-clock time.

    Args:
        function: a (jitted) function.
        *args: its arguments.

    Returns:
        (result, seconds).
    """
    start = time.perf_counter()
    result = jax.block_until_ready(function(*args))
    return result, time.perf_counter() - start


def compare_with_fd(loss: Callable, gradient: np.ndarray, p: jax.Array, indices: list,
                    step: float) -> list[dict[str, Any]]:
    """Central finite differences of ``loss`` at p and their relative difference to AD.

    Args:
        loss: jitted scalar function of p.
        gradient: AD gradient dL/dp, same shape as p.
        p: evaluation point (any shape).
        indices: entries of p to check (tuples or ints).
        step: finite-difference step eps (units of p).

    Returns:
        List of {"index", "ad", "fd", "rel_diff"} per checked entry.
    """
    rows = []
    for index in indices:
        perturbation = jnp.zeros_like(p).at[index].set(step)
        loss_plus = float(loss(p + perturbation))
        loss_minus = float(loss(p - perturbation))
        fd = (loss_plus - loss_minus) / (2.0 * step)
        ad = float(np.asarray(gradient)[index])
        rows.append({
            "index": list(np.atleast_1d(index).tolist()),
            "ad": ad,
            "fd": fd,
            "rel_diff": abs(ad - fd) / abs(fd),
        })
    return rows


def random_weights(shape: Any) -> jax.Array:
    """Fixed random weights (seed 0) of the scalar loss, shape ``shape``."""
    return jnp.asarray(np.random.default_rng(0).normal(size=shape))


def summarise_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Max and min relative difference of a list of FD rows, and the index of the max.

    Args:
        rows: output of :func:`compare_with_fd`.

    Returns:
        Dict with ``max_rel_diff``, ``min_rel_diff`` and ``index_of_max_rel_diff``.
    """
    worst = max(rows, key=lambda row: row["rel_diff"])
    return {
        "max_rel_diff": worst["rel_diff"],
        "min_rel_diff": min(row["rel_diff"] for row in rows),
        "index_of_max_rel_diff": worst["index"],
    }


# ----------------------------------------------------------------------
# (i) steady IFT gradient, and steady vs long transient (Sects. 4.1 and 4.2)
# ----------------------------------------------------------------------
def steady_checks(rel: str, t_transient: float, work_dir: Path) -> dict[str, Any]:
    """Steady Newton + IFT on a paper case: gradient vs FD and steady vs long transient.

    Args:
        rel: example path relative to ``examples/SWE_2D`` (a forward-simulation case).
        t_transient: end time of the long transient run (s).
        work_dir: scratch folder.

    Returns:
        Dict with Newton statistics, solve and gradient times (s), the per-zone gradient
        check and the steady-vs-transient WSE difference (m).
    """
    from hydrograd.physics.rhs import rhs
    from hydrograd.solvers.final import newton_settings
    from hydrograd.solvers.ode import solve_swe
    from hydrograd.solvers.steady import newton_solve, steady_state

    # 1. Case with the steady Newton solver.
    case = setup_case(read_control(copy_example(rel, work_dir) / "run_control.json"))
    model = case.model
    settings = newton_settings(case.ode)
    n_zone = jnp.asarray(model.n_zone)
    zones_with_cells = sorted(set(np.asarray(model.mat_id).tolist()))

    # 2. Newton solve: first call (with compilation) and a second, timed call.
    def newton(U0: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Newton solve from U0; returns (U, iterations, max residual)."""
        return newton_solve(model, {}, U0, settings)

    newton_jit = jax.jit(newton)
    _, first_seconds = timed(newton_jit, case.U0)
    (U_steady, iterations, residual), solve_seconds = timed(newton_jit, case.U0)

    # 3. IFT gradient of the weighted-sum loss with respect to the zone n.
    weights = random_weights(case.U0.shape)

    def loss(n_values: jax.Array) -> jax.Array:
        """Weighted sum of the steady state for zone n values (s/m^(1/3))."""
        return jnp.sum(weights * steady_state(model, {"n_zone": n_values}, case.U0, settings))

    loss_jit = jax.jit(loss)
    gradient_jit = jax.jit(jax.grad(loss))
    _, gradient_first_seconds = timed(gradient_jit, n_zone)
    gradient, gradient_seconds = timed(gradient_jit, n_zone)
    rows = compare_with_fd(loss_jit, np.asarray(gradient), n_zone, zones_with_cells,
                           FD_STEP_ZONE_N)

    # 4. Long transient run (Tsit5, tight tolerances) from the same initial state.
    transient_settings = dataclasses.replace(
        case.ode, t1=t_transient, solver="Tsit5()", adaptive=True, rtol=1.0e-8, atol=1.0e-10,
        n_save=1)

    def transient(U0: jax.Array) -> jax.Array:
        """Tsit5 solution to t_transient; returns all saved frames."""
        return solve_swe(model, {}, U0, transient_settings).ys

    ys, transient_seconds = timed(jax.jit(transient), case.U0)
    U_transient = ys[-1]
    transient_residual = float(jnp.max(jnp.abs(rhs(model, {}, U_transient))))
    wse_difference = np.abs(np.asarray(U_transient[0] - U_steady[0]))

    return {
        "case": rel,
        "loss": "sum(w * U*) over the whole steady state",
        "parameters": "Manning's n per zone (s/m^(1/3))",
        "n_cells": int(model.n_cells),
        "order": int(model.order),
        "newton_dtau0_s": settings.dtau0,
        "newton_iterations": int(iterations),
        "newton_final_max_residual": float(residual),
        "newton_solve_time_s": solve_seconds,
        "newton_first_call_incl_compile_s": first_seconds,
        "ift_gradient_time_s": gradient_seconds,
        "ift_gradient_first_call_incl_compile_s": gradient_first_seconds,
        "fd_step": FD_STEP_ZONE_N,
        "gradient_vs_fd": rows,
        **summarise_rows(rows),
        "transient_t_end_s": t_transient,
        "transient_rtol": 1.0e-8,
        "transient_time_incl_compile_s": transient_seconds,
        "transient_final_max_abs_dUdt": transient_residual,
        "steady_vs_transient_max_abs_wse_diff_m": float(wse_difference.max()),
    }


# ----------------------------------------------------------------------
# NN closure of Sect. 4.4: case and initial weights
# ----------------------------------------------------------------------
def ude_case_and_weights(label: str) -> tuple[Any, Any]:
    """The closure-learning case of ``fig06_steady_ude.py`` and its initial NN weights.

    Args:
        label: ``"steady"`` or ``"transient"`` (training folder of fig06).

    Returns:
        (case, params): the set-up case and the initial MLP weights (pytree, 22 values),
        exactly the weights from which the training of Sect. 4.4 starts.
    """
    from hydrograd.apps import ude as ude_app

    control_path = UDE_DATA_DIR / label / "run_control.json"
    if not control_path.exists():
        raise FileNotFoundError(f"{control_path} missing: run paper/fig06_steady_ude.py first")
    case = setup_case(read_control(control_path))
    params = ude_app._init_params(case)
    return case, params


# ----------------------------------------------------------------------
# (ii) steady IFT with respect to the NN weights (Sect. 4.4, steady path)
# ----------------------------------------------------------------------
def steady_ude_check() -> dict[str, Any]:
    """Steady Newton + IFT gradient with respect to the 22 NN weights vs central FD.

    Returns:
        Dict with Newton statistics, timings (s) and the per-weight gradient check.
    """
    from hydrograd.solvers.final import newton_settings
    from hydrograd.solvers.steady import newton_solve, steady_state

    # 1. Case, initial weights (flattened for the FD loop) and Newton settings.
    case, params = ude_case_and_weights("steady")
    model = case.model
    settings = newton_settings(case.ode)
    p0, unravel = ravel_pytree(params)
    n_params = int(p0.shape[0])

    # 2. Newton solve at the initial weights: compile, then time.
    def newton(p: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Newton solve for weights p; returns (U, iterations, max residual)."""
        return newton_solve(model, {"nn": unravel(p)}, case.U0, settings)

    newton_jit = jax.jit(newton)
    _, first_seconds = timed(newton_jit, p0)
    (_U_steady, iterations, residual), solve_seconds = timed(newton_jit, p0)

    # 3. Loss, IFT gradient (through the custom VJP of steady_state) and FD per weight.
    weights = random_weights(case.U0.shape)

    def loss(p: jax.Array) -> jax.Array:
        """Weighted sum of the steady state for the flattened weights p."""
        U = steady_state(model, {"nn": unravel(p)}, case.U0, settings)
        return jnp.sum(weights * U)

    loss_jit = jax.jit(loss)
    gradient_jit = jax.jit(jax.grad(loss))
    _, gradient_first_seconds = timed(gradient_jit, p0)
    gradient, gradient_seconds = timed(gradient_jit, p0)
    rows = compare_with_fd(loss_jit, np.asarray(gradient), p0, list(range(n_params)),
                           FD_STEP_WEIGHTS_STEADY)

    return {
        "case": "Sect. 4.4 closure learning, steady path (fig06_steady_ude.py, refined bump)",
        "loss": "sum(w * U*) over the whole steady state",
        "parameters": "22 NN weights of n = N(h) (initial weights of the training, seed 1111)",
        "n_cells": int(model.n_cells),
        "order": int(model.order),
        "n_params": n_params,
        "newton_dtau0_s": settings.dtau0,
        "newton_iterations": int(iterations),
        "newton_final_max_residual": float(residual),
        "newton_solve_time_s": solve_seconds,
        "newton_first_call_incl_compile_s": first_seconds,
        "ift_gradient_time_s": gradient_seconds,
        "ift_gradient_first_call_incl_compile_s": gradient_first_seconds,
        "fd_step": FD_STEP_WEIGHTS_STEADY,
        "gradient_vs_fd": rows,
        **summarise_rows(rows),
    }


# ----------------------------------------------------------------------
# Hydrograph-inversion valley of Sect. 4.5 at the true hydrograph
# ----------------------------------------------------------------------
def hydrograph_case(work_dir: Path) -> tuple[Any, jax.Array, np.ndarray]:
    """The inversion case of Sect. 4.5 with the true hydrograph at the 9 knots.

    The geometry, boundary conditions and initial state are those of
    ``benchmarks/validation/hydrograph_inversion.py``; the inflow series holds the true
    hydrograph sampled at the 15 min knots (the parameters of the inversion), so that the
    flood inundates the floodplains as in the paper.

    Args:
        work_dir: scratch folder.

    Returns:
        (case, q_knots, t_knots): the set-up case (explicit SSPRK3 settings), the knot
        discharges [1, 9] (m^3/s) and the knot times [9] (s).
    """
    import hydrograph_inversion as valley

    t_knots = np.arange(0.0, valley.T_END + 1.0, 900.0)
    h_n = 1.03   # base-flow depth of the main channel (m), as in the inversion script

    def initial_wse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Initial WSE (m): plane parallel to the bed, h_n above the channel bed."""
        return valley.S0 * (valley.L - x) + h_n

    case_dir = work_dir / "hydrograph_valley"
    write_channel_case(
        case_dir, valley.NX, valley.NY, length=valley.L, width=valley.W,
        n_manning=valley.N_MAN, zb_fn=valley.bed, wstill=0.0, dt=0.1, t_end=valley.T_END,
        solver="SSPRK3()", inlet="INLET-Q", outlet="EXIT-H", wse_out=h_n,
        Q_series=(t_knots, valley.q_true(t_knots)), ic_wse=initial_wse)
    case = setup_case(read_control(case_dir / "run_control.json"))
    return case, jnp.asarray(case.model.Q), t_knots


# ----------------------------------------------------------------------
# (iii) explicit SSPRK3, checkpointed reverse mode (Sect. 4.5 flood)
# ----------------------------------------------------------------------
def hydrograph_loss_weights(model: Any) -> jax.Array:
    """Random weights of the hydrograph-row loss, shape [N_SAVE_HYDROGRAPH, n_cells]."""
    return random_weights((N_SAVE_HYDROGRAPH, model.n_cells))


def inverted_knots() -> jax.Array:
    """Hydrograph recovered by the noise-free inversion of Sect. 4.5, shape [1, 9] (m^3/s)."""
    if not INVERSION_SUMMARY.exists():
        raise FileNotFoundError(f"{INVERSION_SUMMARY} missing: run prep_hydrograph_inversion.py")
    summary = json.loads(INVERSION_SUMMARY.read_text())
    return jnp.asarray([summary["q_inv"]])


def explicit_hydrograph_check(work_dir: Path) -> dict[str, Any]:
    """Reverse-mode gradient through SSPRK3 (2 h flood of Sect. 4.5) vs central FD.

    Args:
        work_dir: scratch folder.

    Returns:
        Dict with the number of time steps, checkpoints, timings (s) and the gradient
        check for every hydrograph knot, at each FD step of ``FD_STEPS_Q_EXPLICIT``.
    """
    from hydrograd.solvers.ode import n_checkpoints, solve_swe

    # 1. Case (true hydrograph at the 9 knots), explicit settings, a frame every 15 min.
    case, q0, t_knots = hydrograph_case(work_dir)
    model = case.model
    settings = dataclasses.replace(case.ode, n_save=N_SAVE_HYDROGRAPH)

    # 2. Forward run: number of steps and the wetted cells before and after.
    solution, _ = timed(jax.jit(lambda U0: solve_swe(model, {}, U0, settings)), case.U0)
    h_initial = np.asarray(case.U0[0] + model.hstill)
    h_final = np.asarray(solution.ys[-1][0] + model.hstill)

    # 3. Loss on the depths saved every 15 min (every knot acts on it, as on the gauge series
    #    of the inversion) and its reverse-mode gradient with respect to the knots.
    weights = hydrograph_loss_weights(model)

    def loss(q: jax.Array) -> jax.Array:
        """Weighted sum of the saved depths (m) for hydrograph knots q [1, 9] (m^3/s)."""
        frames = solve_swe(model, {"Q": q}, case.U0, settings).ys[1:]
        return jnp.sum(weights * (frames[:, 0] + model.hstill))

    loss_jit = jax.jit(loss)
    gradient_jit = jax.jit(jax.grad(loss))
    _, gradient_first_seconds = timed(gradient_jit, q0)
    gradient, gradient_seconds = timed(gradient_jit, q0)
    _, forward_seconds = timed(loss_jit, q0)
    _, forward_seconds = timed(loss_jit, q0)

    # 4. FD on every knot at each step size (small steps avoid crossing the wet/dry kink;
    #    the CFL step sequence, which is not differentiated, limits the agreement).
    indices = [(0, k) for k in range(q0.shape[1])]
    checks = {}
    for step in FD_STEPS_Q_EXPLICIT:
        rows = compare_with_fd(loss_jit, np.asarray(gradient), q0, indices, step)
        checks[f"fd_step_{step:g}"] = {"fd_step_m3_per_s": step, "gradient_vs_fd": rows,
                                       **summarise_rows(rows)}

    return {
        "case": "Sect. 4.5 hydrograph-inversion valley, 2 h flood at the true hydrograph",
        "loss": f"sum(w * h) over the depths saved every 900 s ({N_SAVE_HYDROGRAPH} frames)",
        "parameters": "9 inflow knots Q_k (m^3/s), 15 min apart",
        "q_knots_m3_per_s": np.asarray(q0)[0].tolist(),
        "t_knots_s": t_knots.tolist(),
        "n_cells": int(model.n_cells),
        "order": int(model.order),
        "solver": "SSPRK3(), CFL-limited steps, point-implicit friction",
        "time_steps": int(solution.stats["num_steps"]),
        "checkpoints": n_checkpoints(settings, case.U0),
        "wet_cells_initial": int((h_initial > 0.01).sum()),
        "wet_cells_final": int((h_final > 0.01).sum()),
        "forward_time_s": forward_seconds,
        "gradient_time_s": gradient_seconds,
        "gradient_first_call_incl_compile_s": gradient_first_seconds,
        "gradient_over_forward_cost": gradient_seconds / forward_seconds,
        "checks": checks,
    }


# ----------------------------------------------------------------------
# (iv) implicit backward Euler with per-step IFT adjoints (Sect. 4.5 inversion model)
# ----------------------------------------------------------------------
def implicit_hydrograph_check(work_dir: Path) -> dict[str, Any]:
    """Reverse-mode gradient through backward Euler (2 h flood of Sect. 4.5) vs central FD.

    Three runs: the inversion settings (theta = 1, dt = 60 s, Newton tolerance 1e-6) at
    the hydrograph recovered by the noise-free inversion; the same with a Newton tolerance
    of 1e-10, which removes the Newton stopping error from both sides; and the true
    hydrograph with dt = 30 s (dt = 60 s diverges there, see the module docstring).

    Args:
        work_dir: scratch folder.

    Returns:
        Dict with the knots, timings (s) and the gradient check for every knot of each run.
    """
    import hydrograph_inversion as valley

    from hydrograd.solvers.implicit import ImplicitSettings, solve_implicit

    case, q0, _t_knots = hydrograph_case(work_dir)
    model = case.model
    weights = hydrograph_loss_weights(model)
    indices = [(0, k) for k in range(q0.shape[1])]
    q_inv = inverted_knots()
    runs = [
        ("inverted_knots_dt60_tol1e-06", q_inv, valley.DT_IMPLICIT, 1.0e-6),
        ("inverted_knots_dt60_tol1e-10", q_inv, valley.DT_IMPLICIT, 1.0e-10),
        ("true_knots_dt30_tol1e-06", q0, DT_IMPLICIT_TRUE_KNOTS, 1.0e-6),
    ]
    results: dict[str, Any] = {}

    for name, q, dt, tol in runs:
        settings = ImplicitSettings(0.0, valley.T_END, dt, theta=1.0, n_save=N_SAVE_HYDROGRAPH,
                                    tol=tol)

        def loss(q: jax.Array, settings: ImplicitSettings = settings) -> jax.Array:
            """Weighted sum of the saved depths (m) for hydrograph knots q [1, 9]."""
            frames = solve_implicit(model, {"Q": q}, case.U0, settings)[1][1:]
            return jnp.sum(weights * (frames[:, 0] + model.hstill))

        loss_jit = jax.jit(loss)
        gradient_jit = jax.jit(jax.grad(loss))
        _, gradient_first_seconds = timed(gradient_jit, q)
        gradient, gradient_seconds = timed(gradient_jit, q)
        _, forward_seconds = timed(loss_jit, q)
        _, forward_seconds = timed(loss_jit, q)
        rows = compare_with_fd(loss_jit, np.asarray(gradient), q, indices, FD_STEP_Q_IMPLICIT)

        results[name] = {
            "q_knots_m3_per_s": np.asarray(q)[0].tolist(),
            "dt_s": dt,
            "time_steps": round(valley.T_END / dt),
            "newton_tol": tol,
            "forward_time_s": forward_seconds,
            "gradient_time_s": gradient_seconds,
            "gradient_first_call_incl_compile_s": gradient_first_seconds,
            "gradient_over_forward_cost": gradient_seconds / forward_seconds,
            "gradient_vs_fd": rows,
            **summarise_rows(rows),
        }

    return {
        "case": "Sect. 4.5 hydrograph-inversion valley, 2 h flood",
        "loss": f"sum(w * h) over the depths saved every 900 s ({N_SAVE_HYDROGRAPH} frames)",
        "parameters": "9 inflow knots Q_k (m^3/s), 15 min apart",
        "knots": "inverted: hydrograph recovered by the noise-free inversion of Sect. 4.5; "
                 "true: the true hydrograph sampled at the knots",
        "note": "at the true knots, backward Euler with dt = 60 s diverges in step 4 "
                "(t = 180 to 240 s, first climb onto the bank; all Newton damping fractions "
                "give a non-finite residual, a 30 s step from the same state converges); "
                "the inversion of Sect. 4.5 never visited those knots",
        "n_cells": int(model.n_cells),
        "order": int(model.order),
        "solver": "backward Euler (theta = 1), per-step IFT adjoint",
        "fd_step_m3_per_s": FD_STEP_Q_IMPLICIT,
        "checks": results,
    }


# ----------------------------------------------------------------------
# (v) forward mode vs reverse mode vs FD (Sect. 4.4, time-marching path)
# ----------------------------------------------------------------------
def forward_vs_reverse_ude_check() -> dict[str, Any]:
    """Forward-mode (JVP per weight) vs checkpointed reverse-mode gradients vs central FD.

    Returns:
        Dict with the per-weight derivatives of the three methods, their relative
        differences and the timings (s).
    """
    from hydrograd.apps.sensitivity import DERIV_RTOL, _chunked_jacfwd
    from hydrograd.solvers.ode import solve_swe

    # 1. Transient training case of Sect. 4.4 (Tsit5, 100 s) at the initial weights.
    case, params = ude_case_and_weights("transient")
    model = case.model
    p0, unravel = ravel_pytree(params)
    n_params = int(p0.shape[0])
    weights = random_weights(case.U0.shape)

    # 2. Forward mode: JVPs of the final state, one per weight, at the training tolerance.
    forward_settings = dataclasses.replace(case.ode, adjoint="forward", rtol=DERIV_RTOL,
                                           atol=DERIV_RTOL * 1.0e-3, n_save=1)

    def final_flat(p: jax.Array) -> jax.Array:
        """Final state vcat(xi, q_x, q_y) for the flattened weights p; shape [3 n_cells]."""
        return solve_swe(model, {"nn": unravel(p)}, case.U0, forward_settings).ys[-1].reshape(-1)

    _ = _chunked_jacfwd(final_flat, p0, n_params)             # compile
    start = time.perf_counter()
    jacobian = _chunked_jacfwd(final_flat, p0, n_params)
    forward_seconds = time.perf_counter() - start
    gradient_forward = np.asarray(weights).reshape(-1) @ jacobian

    # 3. Reverse mode with checkpointing at the same tolerance.
    reverse_settings = dataclasses.replace(case.ode, adjoint="checkpoint", rtol=DERIV_RTOL,
                                           atol=DERIV_RTOL * 1.0e-3, n_save=1)

    def loss(p: jax.Array) -> jax.Array:
        """Weighted sum of the final state for the flattened weights p."""
        U_final = solve_swe(model, {"nn": unravel(p)}, case.U0, reverse_settings).ys[-1]
        return jnp.sum(weights * U_final)

    gradient_jit = jax.jit(jax.grad(loss))
    _, reverse_first_seconds = timed(gradient_jit, p0)
    gradient_reverse, reverse_seconds = timed(gradient_jit, p0)
    gradient_reverse = np.asarray(gradient_reverse)
    forward_run_jit = jax.jit(loss)
    _, forward_run_seconds = timed(forward_run_jit, p0)
    _, forward_run_seconds = timed(forward_run_jit, p0)
    solution = solve_swe(model, {"nn": params}, case.U0,
                         dataclasses.replace(reverse_settings, adjoint="direct"))

    # 4. FD at a much tighter tolerance (the adaptive step selection changes between the
    #    perturbed runs, and the FD quotient carries O(rtol / eps) noise otherwise).
    fd_settings = dataclasses.replace(reverse_settings, rtol=FD_RTOL, atol=FD_RTOL * 1.0e-3)

    def loss_fd(p: jax.Array) -> jax.Array:
        """The same loss, integrated at the tight FD tolerance."""
        U_final = solve_swe(model, {"nn": unravel(p)}, case.U0, fd_settings).ys[-1]
        return jnp.sum(weights * U_final)

    loss_fd_jit = jax.jit(loss_fd)
    checks = {}
    for step in FD_STEPS_WEIGHTS_TRANSIENT:
        rows = compare_with_fd(loss_fd_jit, gradient_reverse, p0, list(range(n_params)), step)
        for row in rows:
            k = row["index"][0]
            row["forward_mode"] = float(gradient_forward[k])
            row["reverse_mode"] = row.pop("ad")
            row["rel_diff_reverse_fd"] = row.pop("rel_diff")
            row["rel_diff_forward_fd"] = abs(row["forward_mode"] - row["fd"]) / abs(row["fd"])
        checks[f"fd_step_{step:g}"] = {
            "fd_step": step,
            "per_weight": rows,
            "max_rel_diff_forward_fd": max(r["rel_diff_forward_fd"] for r in rows),
            "max_rel_diff_reverse_fd": max(r["rel_diff_reverse_fd"] for r in rows),
        }

    # 5. Forward against reverse mode, independent of FD.
    forward_vs_reverse = []
    for k in range(n_params):
        forward_k = float(gradient_forward[k])
        reverse_k = float(gradient_reverse[k])
        forward_vs_reverse.append({"index": [k], "forward_mode": forward_k,
                                   "reverse_mode": reverse_k,
                                   "rel_diff": abs(forward_k - reverse_k) / abs(reverse_k)})

    return {
        "case": "Sect. 4.4 closure learning, time-marching path (fig06_steady_ude.py)",
        "loss": "sum(w * U(t_end)) over the whole final state",
        "parameters": "22 NN weights of n = N(h) (initial weights of the training, seed 1111)",
        "n_cells": int(model.n_cells),
        "order": int(model.order),
        "n_params": n_params,
        "solver": f"Tsit5(), rtol {DERIV_RTOL:g}, atol {DERIV_RTOL * 1.0e-3:g}",
        "t_end_s": float(case.ode.t1),
        "tsit5_steps": int(solution.stats["num_steps"]),
        "forward_run_time_s": forward_run_seconds,
        "forward_mode_jacobian_time_s": forward_seconds,
        "forward_mode_over_forward_run_cost": forward_seconds / forward_run_seconds,
        "reverse_mode_gradient_time_s": reverse_seconds,
        "reverse_mode_first_call_incl_compile_s": reverse_first_seconds,
        "reverse_mode_over_forward_run_cost": reverse_seconds / forward_run_seconds,
        "fd_rtol": FD_RTOL,
        "forward_vs_reverse": forward_vs_reverse,
        "max_rel_diff_forward_reverse": max(r["rel_diff"] for r in forward_vs_reverse),
        "checks": checks,
    }


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def main() -> None:
    """Run the selected checks (all by default) and update the results file."""
    work_dir = Path(tempfile.mkdtemp(prefix="gradient_checks_"))

    # Every check writes its entry as soon as it is done, so partial results survive and
    # single checks can be rerun without losing the others.
    checks = {
        "i_steady_ift_savannah":
            lambda: steady_checks("savannah_river_steady", 3000.0, work_dir),
        "i_steady_ift_bump":
            lambda: steady_checks("channel_bump_transcritical", 2000.0, work_dir),
        "ii_steady_ift_nn_closure": steady_ude_check,
        "iii_explicit_ssprk3_hydrograph": lambda: explicit_hydrograph_check(work_dir),
        "iv_implicit_backward_euler_hydrograph": lambda: implicit_hydrograph_check(work_dir),
        "v_forward_vs_reverse_nn_closure": forward_vs_reverse_ude_check,
    }
    selected = sys.argv[1:] or list(checks)
    unknown = [name for name in selected if name not in checks]
    if unknown:
        raise SystemExit(f"unknown checks {unknown}; choose from {list(checks)}")

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(RESULT_FILE.read_text()) if RESULT_FILE.exists() else {}
    results["hardware"] = hardware_info()
    results["loss"] = ("sum(w * X), w ~ N(0, 1), seed 0; X = steady or final state, or the "
                       "depths saved every 15 min (hydrograph rows)")
    results["rel_diff"] = "|AD - FD| / |FD| (central FD of the same discrete model)"

    for name in selected:
        start = time.perf_counter()
        results[name] = checks[name]()
        results[name]["check_wall_time_s"] = time.perf_counter() - start
        print(name, json.dumps(results[name], indent=1), flush=True)
        RESULT_FILE.write_text(json.dumps(results, indent=2))

    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
