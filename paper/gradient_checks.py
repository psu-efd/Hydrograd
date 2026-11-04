"""Gradient verification for the Hydrograd v2.0 paper (table of AD vs finite differences).

Usage (with the Hydrograd venv active; about 10-20 min on 8 CPU cores)::

    JAX_PLATFORMS=cpu python paper/gradient_checks.py

Writes ``paper/results/gradient_checks.json``.

What is checked
---------------
Each check differentiates a scalar loss L(p) = sum_i w_i U_i(p), a fixed random weighting
(seed 0) of the final or steady state U = [xi, q_x, q_y] (m, m^2/s, m^2/s), with respect to
physical parameters p. The AD derivative dL/dp_k is compared with the central finite difference

    FD_k = (L(p + eps e_k) - L(p - eps e_k)) / (2 eps),

and the relative difference |AD_k - FD_k| / |FD_k| is reported. The four derivative paths are:

(i)   Steady state by pseudo-transient Newton + implicit-function-theorem (IFT) adjoint,
      dU*/dp = -(dR/dU)^{-1} dR/dp: Savannah River (1,306 cells), p = Manning's n per zone
      (s/m^(1/3)); and the transcritical bump (200 cells), p = n of the channel.
(ii)  Explicit SSPRK3 (CFL-limited steps, point-implicit friction) with checkpointed reverse
      mode (diffrax RecursiveCheckpointAdjoint, 64 checkpoints) through a rising flood that
      wets the floodplain (500 cells, 1200 s); p = inlet-hydrograph knots Q_k (m^3/s).
(iii) Implicit backward Euler (fixed dt = 30 s) with the per-step IFT adjoint, hydrograph
      valley of ``benchmarks/validation/hydrograph_inversion.py`` (600 s); p = Q_k (m^3/s).
(iv)  Forward-mode sensitivity (the ``sensitivity`` app: chunked JVPs through Tsit5, rtol 1e-7)
      against checkpointed reverse mode (same tolerance) and FD (rtol 1e-10, eps 1e-5), Julia example
      ``sensitivity_analysis/ManningN/oneD_channel_with_bump`` (t = 200 s); p = zone n.

Also recorded: the steady Newton solve against a long transient run (Tsit5, rtol 1e-8) for
the bump (t = 2,000 s) and Savannah (t = 3,000 s): max |WSE difference| (m), Newton
iterations, final residual and wall-clock times (s), with and without compilation.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

import hydrograd  # noqa: F401  (enables float64)
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.io.synthetic import write_channel_case

# Paths: this folder, the repository, the Julia examples and the output file.
PAPER_DIR = Path(__file__).resolve().parent
REPO_DIR = PAPER_DIR.parent
JULIA_DIR = Path(os.environ.get("HYDROGRAD_JL", REPO_DIR.parent / "Hydrograd.jl"))
EXAMPLES = JULIA_DIR / "examples" / "SWE_2D"
RESULT_FILE = PAPER_DIR / "results" / "gradient_checks.json"

# Finite differences through adaptive Tsit5 runs: tolerance and step (s/m^(1/3)).
FD_RTOL = 1.0e-10
FD_STEP_TRANSIENT = 1.0e-5

sys.path.insert(0, str(PAPER_DIR))
sys.path.insert(0, str(REPO_DIR / "benchmarks" / "validation"))
from fig03_verification import hardware_info


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def copy_example(rel: str, work_dir: Path) -> Path:
    """Copy a Hydrograd.jl example into ``work_dir`` (never write into Hydrograd.jl).

    Args:
        rel: example path relative to ``examples/SWE_2D``.
        work_dir: scratch folder.

    Returns:
        The copied case folder.
    """
    destination = work_dir / rel.replace("/", "_")
    shutil.copytree(EXAMPLES / rel, destination,
                    ignore=shutil.ignore_patterns("*.gif", "*.h5", "SRH_2D"))
    return destination


def timed(function, *args):
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


def compare_with_fd(loss, gradient: np.ndarray, p: jax.Array, indices, step: float) -> list:
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


def random_weights(shape) -> jax.Array:
    """Fixed random weights (seed 0) of the scalar loss, shape ``shape``."""
    return jnp.asarray(np.random.default_rng(0).normal(size=shape))


# ----------------------------------------------------------------------
# (i) steady IFT gradient, and steady vs long transient
# ----------------------------------------------------------------------
def steady_checks(rel: str, t_transient: float, work_dir: Path) -> dict:
    """Steady Newton + IFT on a Julia example: gradient vs FD and steady vs long transient.

    Args:
        rel: example path relative to ``examples/SWE_2D`` (a forward-simulation case).
        t_transient: end time of the long transient run (s).
        work_dir: scratch folder.

    Returns:
        Dict with Newton statistics, solve and gradient times (s), the per-zone gradient
        check and the steady-vs-transient WSE difference (m).
    """
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
    def newton(U0):
        """Newton solve from U0; returns (U, iterations, max residual)."""
        return newton_solve(model, {}, U0, settings)

    newton_jit = jax.jit(newton)
    _, first_seconds = timed(newton_jit, case.U0)
    (U_steady, iterations, residual), solve_seconds = timed(newton_jit, case.U0)

    # 3. IFT gradient of the weighted-sum loss with respect to the zone n.
    weights = random_weights(case.U0.shape)

    def loss(n_values):
        """Weighted sum of the steady state for zone n values (s/m^(1/3))."""
        return jnp.sum(weights * steady_state(model, {"n_zone": n_values}, case.U0, settings))

    loss_jit = jax.jit(loss)
    gradient_jit = jax.jit(jax.grad(loss))
    _, gradient_first_seconds = timed(gradient_jit, n_zone)
    gradient, gradient_seconds = timed(gradient_jit, n_zone)
    rows = compare_with_fd(loss_jit, np.asarray(gradient), n_zone, zones_with_cells, 1.0e-6)

    # 4. Long transient run (Tsit5, tight tolerances) from the same initial state.
    transient_settings = dataclasses.replace(
        case.ode, t1=t_transient, solver="Tsit5()", adaptive=True, rtol=1.0e-8, atol=1.0e-10,
        n_save=1)

    def transient(U0):
        """Tsit5 solution to t_transient; returns all saved frames."""
        return solve_swe(model, {}, U0, transient_settings).ys

    ys, transient_seconds = timed(jax.jit(transient), case.U0)
    U_transient = ys[-1]
    from hydrograd.physics.rhs import rhs
    transient_residual = float(jnp.max(jnp.abs(rhs(model, {}, U_transient))))
    wse_difference = np.abs(np.asarray(U_transient[0] - U_steady[0]))

    return {
        "case": rel,
        "n_cells": int(model.n_cells),
        "order": int(model.order),
        "newton_dtau0_s": settings.dtau0,
        "newton_iterations": int(iterations),
        "newton_final_max_residual": float(residual),
        "newton_solve_time_s": solve_seconds,
        "newton_first_call_incl_compile_s": first_seconds,
        "ift_gradient_time_s": gradient_seconds,
        "ift_gradient_first_call_incl_compile_s": gradient_first_seconds,
        "fd_step": 1.0e-6,
        "gradient_vs_fd": rows,
        "max_rel_diff": max(row["rel_diff"] for row in rows),
        "transient_t_end_s": t_transient,
        "transient_rtol": 1.0e-8,
        "transient_time_incl_compile_s": transient_seconds,
        "transient_final_max_abs_dUdt": transient_residual,
        "steady_vs_transient_max_abs_wse_diff_m": float(wse_difference.max()),
    }


# ----------------------------------------------------------------------
# (ii) explicit SSPRK3, checkpointed reverse mode through a wetting flood
# ----------------------------------------------------------------------
def valley_bed(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Valley bed (m): slope 1e-3, 20 m channel, 5 m banks rising 2 m, gentle floodplains.

    Args:
        X, Y: coordinates (m), any (equal) shape.

    Returns:
        Bed elevation z_b (m).
    """
    r = np.abs(Y - 50.0)
    floodplain = 2.0 + 0.01 * (r - 15.0)
    bank = 0.4 * (r - 10.0)
    cross_section = np.where(r <= 10.0, 0.0, np.where(r <= 15.0, bank, floodplain))
    return 1.0e-3 * (500.0 - X) + cross_section


def valley_initial_wse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Initial WSE (m): 1 m above a plane with the bed slope."""
    return 1.0e-3 * (500.0 - x) + 1.0


def explicit_reverse_check(work_dir: Path) -> dict:
    """Reverse-mode gradient through SSPRK3 (wetting flood) vs central FD.

    Same set-up as the slow test in ``tests/test_wetdry.py``: inlet hydrograph rising from
    10 to 150 m^3/s over 600 s, then constant; wall outlet; second-order scheme.

    Args:
        work_dir: scratch folder.

    Returns:
        Dict with the number of time steps, checkpoints, timings (s) and the gradient check.
    """
    from hydrograd.solvers.ode import n_checkpoints, solve_swe

    # 1. Case.
    case_dir = work_dir / "wetting"
    write_channel_case(
        case_dir, 50, 10, length=500.0, width=100.0, n_manning=0.035, zb_fn=valley_bed,
        wstill=0.0, inlet="INLET-Q", outlet="WALL", dt=0.1, solver="SSPRK3()", t_end=1200.0,
        Q_series=([0.0, 600.0, 1200.0], [10.0, 150.0, 150.0]), ic_wse=valley_initial_wse)
    case = setup_case(read_control(case_dir / "run_control.json"))
    model = case.model
    settings = dataclasses.replace(case.ode, n_save=1)

    # 2. Forward run: number of steps and the wetted cells (h > 1 cm) before and after.
    solution, _ = timed(jax.jit(lambda U0: solve_swe(model, {}, U0, settings)),
                                      case.U0)
    h_initial = np.asarray(case.U0[0] + model.hstill)
    h_final = np.asarray(solution.ys[-1][0] + model.hstill)

    # 3. Loss and its reverse-mode gradient with respect to the hydrograph knots.
    weights = random_weights(model.n_cells)

    def loss(q):
        """Weighted sum of the final depths (m) for hydrograph values q (m^3/s)."""
        U_final = solve_swe(model, {"Q": q}, case.U0, settings).ys[-1]
        return jnp.sum(weights * (U_final[0] + model.hstill))

    q0 = jnp.asarray(model.Q)
    gradient_jit = jax.jit(jax.grad(loss))
    _, gradient_first_seconds = timed(gradient_jit, q0)
    gradient, gradient_seconds = timed(gradient_jit, q0)
    _, forward_warm_seconds = timed(jax.jit(loss), q0)

    # 4. FD on every knot (step 0.01 m^3/s; larger steps let cells cross the wet/dry kink).
    indices = [(0, k) for k in range(q0.shape[1])]
    rows = compare_with_fd(jax.jit(loss), np.asarray(gradient), q0, indices, 0.01)

    return {
        "description": "SSPRK3 (order 2, CFL 0.25), 500 cells, 1200 s, wetting flood",
        "note": ("FD-AD gap (~1e-4 to 6e-4) comes from the CFL step-size choice, which "
                 "depends on Q but is not differentiated (dt is treated as a constant in "
                 "the adjoint)"),
        "n_cells": int(model.n_cells),
        "order": int(model.order),
        "time_steps": int(solution.stats["num_steps"]),
        "checkpoints": n_checkpoints(settings, case.U0),
        "wet_cells_initial": int((h_initial > 0.01).sum()),
        "wet_cells_final": int((h_final > 0.01).sum()),
        "forward_time_s": forward_warm_seconds,
        "gradient_time_s": gradient_seconds,
        "gradient_first_call_incl_compile_s": gradient_first_seconds,
        "gradient_over_forward_cost": gradient_seconds / forward_warm_seconds,
        "fd_step_m3_per_s": 0.01,
        "gradient_vs_fd": rows,
        "max_rel_diff": max(row["rel_diff"] for row in rows),
    }


# ----------------------------------------------------------------------
# (iii) implicit backward Euler with per-step IFT adjoints
# ----------------------------------------------------------------------
def implicit_check(work_dir: Path) -> dict:
    """Reverse-mode gradient through backward Euler (per-step IFT adjoint) vs central FD.

    Same set-up as ``tests/test_implicit.py``: hydrograph-inversion valley, Q rising from
    about 40 to about 75 m^3/s over 600 s, exit-H outlet, dt = 30 s.

    The test uses dt = 60 s and checks one knot only. At dt = 60 s, perturbing other knots
    by +-0.05 m^3/s makes damped Newton fail (non-finite or 1e1 residual) in step 9 at the
    advancing wet front, so the finite differences, not the adjoint, are wrong there
    (phase-2 report section 4g). At dt = 30 s every Newton solve of the FD runs converges
    to max|G| < 1e-10; the FD step is 5e-4 m^3/s (see step 3).

    Args:
        work_dir: scratch folder.

    Returns:
        Dict with timings (s) and the gradient check for every hydrograph knot.
    """
    import hydrograph_inversion as valley

    from hydrograd.solvers.implicit import ImplicitSettings, solve_implicit

    # 1. Case (five hydrograph knots over 600 s).
    t_end = 600.0
    knot_times = np.linspace(0.0, t_end, 5)

    def initial_wse(x, y):
        """Initial WSE (m): plane parallel to the bed, 1.03 m at the outlet."""
        return valley.S0 * (valley.L - x) + 1.03

    case_dir = work_dir / "implicit"
    write_channel_case(
        case_dir, valley.NX, valley.NY, length=valley.L, width=valley.W,
        n_manning=valley.N_MAN, zb_fn=valley.bed, wstill=0.0, inlet="INLET-Q",
        outlet="EXIT-H", dt=0.1, solver="SSPRK3()", t_end=t_end, wse_out=1.03,
        Q_series=(knot_times, valley.q_true(knot_times)), ic_wse=initial_wse)
    case = setup_case(read_control(case_dir / "run_control.json"))
    model = case.model
    dt = 30.0
    settings = ImplicitSettings(0.0, t_end, dt, n_save=1, tol=1.0e-10)

    # 2. Loss and its reverse-mode gradient.
    weights = random_weights(model.n_cells)

    def loss(q):
        """Weighted sum of the final depths (m) for hydrograph values q (m^3/s)."""
        U_final = solve_implicit(model, {"Q": q}, case.U0, settings)[1][-1]
        return jnp.sum(weights * (U_final[0] + model.hstill))

    q0 = jnp.asarray(model.Q)
    loss_jit = jax.jit(loss)
    gradient_jit = jax.jit(jax.grad(loss))
    _, gradient_first_seconds = timed(gradient_jit, q0)
    gradient, gradient_seconds = timed(gradient_jit, q0)
    _, forward_seconds = timed(loss_jit, q0)
    _, forward_seconds = timed(loss_jit, q0)

    # 3. FD on every hydrograph knot. The step must be small (5e-4 m^3/s): with 0.05 m^3/s,
    #    cells near the wet front cross the wet/dry kink between the +-eps runs and the FD
    #    quotient differs from the derivative by 2 % (at 5e-3: 1e-7; at 5e-4: 1e-8).
    indices = [(0, k) for k in range(q0.shape[1])]
    rows = compare_with_fd(loss_jit, np.asarray(gradient), q0, indices, 5.0e-4)

    return {
        "description": "backward Euler, dt = 30 s, 600 s, hydrograph valley",
        "note": ("at dt = 60 s, FD runs with Q knots 2 and 4 perturbed by +0.05 m^3/s fail "
                 "in Newton (step 9, wet front); FD there is invalid, not the adjoint"),
        "n_cells": int(model.n_cells),
        "order": int(model.order),
        "time_steps": round(t_end / dt),
        "newton_tol": 1.0e-10,
        "forward_time_s": forward_seconds,
        "gradient_time_s": gradient_seconds,
        "gradient_first_call_incl_compile_s": gradient_first_seconds,
        "gradient_over_forward_cost": gradient_seconds / forward_seconds,
        "fd_step_m3_per_s": 5.0e-4,
        "gradient_vs_fd": rows,
        "max_rel_diff": max(row["rel_diff"] for row in rows),
    }


# ----------------------------------------------------------------------
# (iv) forward-mode sensitivity vs reverse mode vs FD
# ----------------------------------------------------------------------
def forward_vs_reverse_check(work_dir: Path) -> dict:
    """Forward-mode sensitivity Jacobian vs checkpointed reverse mode vs central FD.

    Args:
        work_dir: scratch folder.

    Returns:
        Dict with the per-zone derivatives of the three methods and their relative
        differences, and the timings (s).
    """
    from hydrograd.apps.sensitivity import DERIV_RTOL, final_state_jacobian
    from hydrograd.solvers.ode import solve_swe

    # 1. Sensitivity example (Tsit5, t = 0-200 s, zone n = [0.03, 0.02, 0.03]).
    rel = "sensitivity_analysis/ManningN/oneD_channel_with_bump"
    case = setup_case(read_control(copy_example(rel, work_dir) / "run_control.json"))
    model = case.model
    n_values = jnp.asarray(np.asarray(case.ctrl.opt("sensitivity_ManningN_values"), dtype=float))
    zones_with_cells = sorted(set(np.asarray(model.mat_id).tolist()))

    # 2. Forward mode: the full Jacobian d U(t_end) / d n, shape [3 n_cells, n_zones].
    start = time.perf_counter()
    jacobian = final_state_jacobian(case, "ManningN", n_values)["J"]
    forward_seconds = time.perf_counter() - start
    weights = random_weights(case.U0.shape)
    gradient_forward = np.asarray(weights).reshape(-1) @ jacobian

    # 3. Reverse mode with the same tolerances as the forward-mode app.
    settings = dataclasses.replace(case.ode, adjoint="checkpoint", rtol=DERIV_RTOL,
                                   atol=DERIV_RTOL * 1.0e-3, n_save=1)

    def loss(n_zone):
        """Weighted sum of the final state for zone n values (s/m^(1/3))."""
        return jnp.sum(weights * solve_swe(model, {"n_zone": n_zone}, case.U0, settings).ys[-1])

    gradient_jit = jax.jit(jax.grad(loss))
    _, reverse_first_seconds = timed(gradient_jit, n_values)
    gradient_reverse, reverse_seconds = timed(gradient_jit, n_values)
    gradient_reverse = np.asarray(gradient_reverse)

    # 4. FD and the pairwise relative differences. The FD runs use a much tighter
    #    tolerance (rtol 1e-10): at rtol 1e-7 the adaptive step selection changes between
    #    the perturbed runs, and the FD quotient then carries O(rtol / eps) noise (~1e-3).
    fd_settings = dataclasses.replace(settings, rtol=FD_RTOL, atol=FD_RTOL * 1.0e-3)

    def loss_fd(n_zone):
        """The same loss, integrated at the tight FD tolerance."""
        U_final = solve_swe(model, {"n_zone": n_zone}, case.U0, fd_settings).ys[-1]
        return jnp.sum(weights * U_final)

    rows = compare_with_fd(jax.jit(loss_fd), gradient_reverse, n_values, zones_with_cells,
                           FD_STEP_TRANSIENT)
    for row in rows:
        zone = row["index"][0]
        row["forward_mode"] = float(gradient_forward[zone])
        row["reverse_mode"] = row.pop("ad")
        row["rel_diff_reverse_fd"] = row.pop("rel_diff")
        row["rel_diff_forward_fd"] = abs(row["forward_mode"] - row["fd"]) / abs(row["fd"])
        row["rel_diff_forward_reverse"] = (abs(row["forward_mode"] - row["reverse_mode"])
                                           / abs(row["reverse_mode"]))

    return {
        "case": rel,
        "n_cells": int(model.n_cells),
        "order": int(model.order),
        "solver": "Tsit5(), rtol 1e-7, atol 1e-10",
        "t_end_s": float(case.ode.t1),
        "forward_mode_jacobian_time_incl_compile_s": forward_seconds,
        "reverse_mode_gradient_time_s": reverse_seconds,
        "reverse_mode_first_call_incl_compile_s": reverse_first_seconds,
        "fd_step": FD_STEP_TRANSIENT,
        "fd_rtol": FD_RTOL,
        "per_zone": rows,
        "max_rel_diff_forward_reverse": max(r["rel_diff_forward_reverse"] for r in rows),
        "max_rel_diff_forward_fd": max(r["rel_diff_forward_fd"] for r in rows),
        "max_rel_diff_reverse_fd": max(r["rel_diff_reverse_fd"] for r in rows),
    }


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def main() -> None:
    """Run all checks and write ``paper/results/gradient_checks.json``."""
    work_dir = Path(tempfile.mkdtemp(prefix="gradient_checks_"))
    results = {"hardware": hardware_info(), "loss": "sum(w * U), w ~ N(0, 1), seed 0",
               "rel_diff": "|AD - FD| / |FD| (central FD)"}

    # Each check is written as soon as it is done, so partial results survive.
    checks = (
        ("i_steady_ift_savannah",
         lambda: steady_checks("forward_simulation/Savannah_River", 3000.0, work_dir)),
        ("i_steady_ift_bump",
         lambda: steady_checks("forward_simulation/oneD_channel_with_bump", 2000.0, work_dir)),
        ("iii_implicit_backward_euler", lambda: implicit_check(work_dir)),
        ("iv_forward_vs_reverse", lambda: forward_vs_reverse_check(work_dir)),
        ("ii_explicit_ssprk3_checkpointed_reverse", lambda: explicit_reverse_check(work_dir)),
    )
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    for name, check in checks:
        start = time.perf_counter()
        results[name] = check()
        results[name]["check_wall_time_s"] = time.perf_counter() - start
        print(name, json.dumps(results[name], indent=1), flush=True)
        RESULT_FILE.write_text(json.dumps(results, indent=2))

    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
