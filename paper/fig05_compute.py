"""Computations behind Figure 5: steady and transient Manning's n inversion on the Savannah River.

Problem
-------
Five Manning zones (s m^(-1/3)) with true values n* = (0.040, 0.050, 0.030, 0.045, 0.050)
(the ``.srhhydro`` values; zone 0 of SRH-2D has no cells). Synthetic observations are the
water-surface elevation WSE (m) and velocities (u, v) (m/s) of a forward run at n*. Starting
from n = 0.03 in every zone, the loss of ``hydrograd.apps.losses``

    L(n) = L_WSE + L_uv + L_bound,
    L_WSE = mean(((WSE - WSE_obs) / range(WSE_obs))^2),
    L_uv  = sum(((u - u_obs)/V)^2 + ((v - v_obs)/V)^2) / N,   V = max |u_obs|,

is minimised by Adam (learning rate 1e-3) followed by L-BFGS, through the package's own
inversion application ``hydrograd.apps.inversion.run_inversion``.

Runs
----
``steady``     steady state R(U*, n) = 0 by Newton; gradients by the implicit function
               theorem (one adjoint solve per gradient). Observations: a steady forward run.
               30 Adam iterations, then at most 50 L-BFGS iterations (stop at |grad| < 1e-6).
``transient``  the same optimiser stages, but the state is the time-marched solution at
               t = 200 s (Tsit5, rtol 1e-7 for derivatives); observations from a transient
               run at n* with the same tolerance (``_transient_truth``). Both stages use
               reverse-mode AD with checkpointing (``AutoZygote()``).
``julia300``   the Hydrograd.jl example configuration: 300 Adam iterations, transient,
               Hydrograd.jl's own truth file, first-order scheme (re-times the run reported
               in the phase-1 parity report as 9 min, made before second order existed).
               Adam with forward-mode AD (``AutoForwardDiff()``), as in the example.
``steady_o1``, ``transient_o1``: ``steady`` and ``transient`` with the first-order scheme.
``gradcheck``  IFT gradient of the steady loss against central finite differences (step
               1e-6 s m^(-1/3)) in every zone, at two parameter points; plus the warm (compiled)
               cost of one loss-and-gradient evaluation for the steady and transient losses.

``steady``, ``transient`` and ``gradcheck`` use the default second-order MUSCL scheme.

Each run writes ``paper/data/fig05_<run>.json``; ``fig05_steady_inversion.py`` plots them and
writes ``paper/results/steady_inversion.json``. Wall times are measured on the machine that
runs this script (recorded with each run; use ``hpc/roar/paper_fig04_05.sbatch``).

Usage
-----
    python paper/fig05_compute.py steady gradcheck transient steady_o1 transient_o1 julia300
"""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from savannah_common import (
    DATA_DIR,
    INVERSION_CASE,
    copy_case,
    machine_info,
    read_json_with_commas,
    write_control,
)

import hydrograd  # noqa: F401  (enables float64)
import hydrograd.apps.inversion as inversion_app
from hydrograd.apps.forward import derived_fields, run_forward
from hydrograd.apps.losses import bound_loss, data_losses, load_truth
from hydrograd.apps.sensitivity import DERIV_RTOL
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.solvers.final import final_state

# Starting value of every zone (s m^(-1/3)) and the optimiser stages of the runs.
N_START = 0.03
ADAM_LR = 1.0e-3
ADAM_ITERATIONS = 30
LBFGS_ITERATIONS = 50
GRAD_TOL = 1.0e-6

# Central finite-difference step (s m^(-1/3)) and timing repetitions of the gradient check.
FD_STEP = 1.0e-6
N_TIMING_REPEATS = 3

# Solver names in the run control.
STEADY = "SteadyNewton()"
TRANSIENT = "Tsit5()"


# ----------------------------------------------------------------------
# Case preparation
# ----------------------------------------------------------------------
def _forward_truth(case_dir: Path, control: dict[str, Any], solver: str, name: str) -> None:
    """Forward run at the true n that writes the synthetic observations ``name``.

    Args:
        case_dir: copied inversion case folder.
        control: the example's inversion run control.
        solver: ``"SteadyNewton()"`` or ``"Tsit5()"``.
        name: truth file name to write.
    """
    forward = json.loads(json.dumps(control))
    forward["control_variables"].update(bPerform_Inversion=False, bPerform_Forward_Simulation=True)
    forward["forward_simulation_options"] = {
        "forward_simulation_ManningN_option": "constant",
        "forward_simulation_solver": "SciML",
        "forward_simulation_ode_solver": solver,
        "forward_simulation_adaptive": True,
        "forward_simulation_nSave": 1,
        "forward_simulation_initial_condition_options": "from_file",
        "forward_simulation_initial_condition_file_name":
            "forward_simulation_initial_condition.json",
        "forward_simulation_save_solution_truth_file_name": name,
    }
    path = write_control(case_dir, forward, "run_control_forward.json")
    run_forward(setup_case(read_control(path)))


def _inversion_control(control: dict[str, Any], solver: str, truth: str,
                       stages: list[tuple[str, float, int]]) -> dict[str, Any]:
    """Inversion run control with the given solver, observations and optimiser stages.

    Args:
        control: the example's inversion run control.
        solver: ``"SteadyNewton()"`` or ``"Tsit5()"``.
        truth: truth (observation) file name.
        stages: (optimiser, learning rate, maximum iterations) per stage.

    Returns:
        The modified run control.
    """
    control = json.loads(json.dumps(control))
    options = control["inversion_options"]
    options.update(
        inversion_truth_file_name=truth,
        inversion_optimizers=[stage[0] for stage in stages],
        inversion_learning_rates=[stage[1] for stage in stages],
        inversion_max_iterations=[stage[2] for stage in stages],
        inversion_abs_tols=[GRAD_TOL] * len(stages),
        inversion_rel_tols=[GRAD_TOL] * len(stages),
        inversion_ManningN_initial_values=[N_START] * 6,
        inversion_save_frequency=10 ** 6,   # fields only at the first and last iteration
    )
    options["inversion_ode_solver_options"]["ode_solver"] = solver

    # Reverse-mode gradients in every stage (the faster mode for the transient loss on this
    # case: 8 s against 19 s per gradient with forward mode; steady gradients are always
    # reverse-mode IFT).
    options["inversion_sensealg"] = "AutoZygote()"
    return control


def _transient_truth(case_dir: Path, control: dict[str, Any], name: str) -> None:
    """Transient observations at the true n, integrated at the inversion's own tolerance.

    The forward application integrates at rtol = 1e-3, while the inversion integrates at
    rtol = DERIV_RTOL = 1e-7; observations from the forward application would then differ
    from the inversion's own solution at n* by the time-integration error, leaving a loss
    floor and a small bias in n. Here the final state at t_end is computed with the
    inversion's settings, and the observed fields use the truth-file conventions
    (``derived_fields``: u = q/(h + h_small)).

    Args:
        case_dir: copied inversion case folder.
        control: inversion run control (transient solver).
        name: truth file name to write.
    """
    path = write_control(case_dir, control, "run_control_truth.json")
    case = setup_case(read_control(path))
    settings = dataclasses.replace(case.ode, rtol=min(case.ode.rtol, DERIV_RTOL),
                                   atol=min(case.ode.atol, DERIV_RTOL * 1e-3))

    # Final state at the true (file) n, and the derived fields.
    U, ok = final_state(case.model, {}, case.U0, settings)
    if not bool(ok):
        raise RuntimeError("transient truth run failed")
    fields = derived_fields(case.model, {}, np.asarray(U), case.wstill)

    truth = {
        "wse_truth": fields["wse"].tolist(),
        "u_truth": fields["u"].tolist(),
        "v_truth": fields["v"].tolist(),
        "h_truth": fields["h"].tolist(),
        "ManningN_zone_values_truth": np.asarray(case.model.n_zone).tolist(),
        "rtol": settings.rtol,
        "atol": settings.atol,
    }
    (case_dir / name).write_text(json.dumps(truth))


# ----------------------------------------------------------------------
# Timed inversion through the package's application
# ----------------------------------------------------------------------
def _timed_inversion(case_dir: Path) -> dict[str, Any]:
    """Run ``run_inversion`` in ``case_dir`` and time it, with a time stamp per iteration.

    The package's optimiser loop is wrapped (not modified) so that every recorded iteration
    also stores the elapsed wall time since the start of the run.

    Args:
        case_dir: case folder with the inversion ``run_control.json``.

    Returns:
        Dictionary with the total wall time (s, including set-up and compilation), the
        per-iteration elapsed times (s), losses, parameters [n_iter, 6] and the final n.
    """
    original_run_stages = inversion_app.run_stages
    stamps: list[float] = []
    thetas: list[list[float]] = []
    start = time.perf_counter()

    def timed_run_stages(make_loss: Callable, theta0: Any, stages: Any, callback: Callable,
                         forward_mode: bool = False) -> Any:
        """``run_stages`` whose callback also records the elapsed time (s)."""
        def timed_callback(it: int, p: Any, loss: float, aux: dict[str, Any]) -> None:
            """Record the time stamp, then call the application's callback."""
            stamps.append(time.perf_counter() - start)
            thetas.append(np.asarray(p).tolist())
            callback(it, p, loss, aux)

        return original_run_stages(make_loss, theta0, stages, timed_callback,
                                   forward_mode=forward_mode)

    # Set up, run and time the inversion (set-up included in the wall time).
    inversion_app.run_stages = timed_run_stages
    try:
        case = setup_case(read_control(case_dir / "run_control.json"))
        output = inversion_app.run_inversion(case)
    finally:
        inversion_app.run_stages = original_run_stages
    wall = time.perf_counter() - start

    # Loss history (one row per iteration) and the parameters of every iteration.
    with open(case_dir / "inversion_loss_history.csv") as handle:
        rows = list(csv.reader(handle))[1:]
    history = [[float(value) for value in row] for row in rows]

    return {
        "wall_time_s": wall,
        "elapsed_at_iteration_s": stamps,
        "iteration": [row[0] for row in history],
        "loss_total": [row[1] for row in history],
        "loss_wse": [row[2] for row in history],
        "loss_uv": [row[3] for row in history],
        "theta": thetas,
        "n_final": np.asarray(output["p"]).tolist(),
    }


def run_inversion_case(run: str) -> dict[str, Any]:
    """Prepare and run one of the inversions ``steady``, ``transient`` or ``julia300``.

    Args:
        run: run name (``steady``, ``transient``, ``steady_o1``, ``transient_o1``,
            ``julia300``).

    Returns:
        The timed-inversion record plus the configuration.
    """
    case_dir = copy_case(INVERSION_CASE, f"Savannah_{run}")
    control = read_json_with_commas(case_dir / "run_control.json")
    two_stages = [("Adam", ADAM_LR, ADAM_ITERATIONS), ("LBFGS", 1.0, LBFGS_ITERATIONS)]

    # Spatial order: first order for the *_o1 runs and the Julia configuration.
    order = 1 if run.endswith("_o1") or run == "julia300" else 2
    control["numerics"] = {"spatial_order": order}
    base = run.removesuffix("_o1")

    # Observations (forward run at the same order) and run control of the requested run.
    if base == "steady":
        _forward_truth(case_dir, control, STEADY, "steady_truth.json")
        control = _inversion_control(control, STEADY, "steady_truth.json", two_stages)
    elif base == "transient":
        control = _inversion_control(control, TRANSIENT, "transient_truth.json", two_stages)
        _transient_truth(case_dir, control, "transient_truth.json")
    elif base == "julia300":
        # The example's own stages and Hydrograd.jl truth file; only the record frequency
        # is lowered (no per-iteration field output).
        control["inversion_options"]["inversion_save_frequency"] = 10 ** 6
    else:
        raise ValueError(f"unknown run {run!r}")

    write_control(case_dir, control)
    record = _timed_inversion(case_dir)

    # True n from the observation file used by the run.
    options = control["inversion_options"]
    truth = json.loads((case_dir / options["inversion_truth_file_name"]).read_text())
    record.update({
        "run": run,
        "spatial_order": order,
        "solver": options["inversion_ode_solver_options"]["ode_solver"],
        "optimizers": options["inversion_optimizers"],
        "learning_rates": options["inversion_learning_rates"],
        "max_iterations": options["inversion_max_iterations"],
        "abs_tols": options["inversion_abs_tols"],
        "sensealg": options["inversion_sensealg"],
        "truth_file": options["inversion_truth_file_name"],
        "n_truth": truth["ManningN_zone_values_truth"],
        "tspan_s": control["time_settings"]["tspan"],
        "machine": machine_info(),
    })
    return record


# ----------------------------------------------------------------------
# Gradient check and per-gradient cost
# ----------------------------------------------------------------------
def _loss_function(case: Any, truth_name: str, adjoint: str) -> Callable[[jax.Array], jax.Array]:
    """The inversion loss L(n) of ``run_inversion`` as a plain function (scalar, -).

    Args:
        case: set-up inversion case.
        truth_name: observation file in the case folder.
        adjoint: diffrax adjoint (``"checkpoint"`` or ``"forward"``; unused when steady).

    Returns:
        Function n [6] (s m^(-1/3)) -> L(n) (dimensionless).
    """
    model = case.model
    truth = load_truth(json.loads((case.ctrl.case_dir / truth_name).read_text()))
    settings = dataclasses.replace(case.ode, adjoint=adjoint, rtol=min(case.ode.rtol, DERIV_RTOL),
                                   atol=min(case.ode.atol, DERIV_RTOL * 1e-3))
    options = case.ctrl.opt
    lower = float(options("inversion_parameter_value_lower_bound"))
    upper = float(options("inversion_parameter_value_upper_bound"))

    def loss(n_zone: jax.Array) -> jax.Array:
        """WSE + velocity misfit + bound penalty at zone values n_zone [6]."""
        U, _ = final_state(model, {"n_zone": n_zone}, case.U0, settings)
        misfit = data_losses(model, U, truth)
        return misfit["wse"] + misfit["uv"] + bound_loss(n_zone, lower, upper, model.n_cells)

    return loss


def _warm_time(function: Callable[[jax.Array], Any], argument: jax.Array) -> tuple[float, float]:
    """Compile-plus-run time and median warm time (s) of a jitted function.

    Args:
        function: jitted function of one array.
        argument: its argument.

    Returns:
        (first-call time, median of the warm repetitions) (s).
    """
    start = time.perf_counter()
    jax.block_until_ready(function(argument))
    first = time.perf_counter() - start

    warm = []
    for _ in range(N_TIMING_REPEATS):
        start = time.perf_counter()
        jax.block_until_ready(function(argument))
        warm.append(time.perf_counter() - start)

    return first, float(np.median(warm))


def run_gradcheck() -> dict[str, Any]:
    """IFT gradient vs central finite differences, and the cost of one gradient.

    Returns:
        Dictionary with, per parameter point, the IFT and FD gradients of zones 1..5 and
        their relative differences; and the first-call and warm times (s) of one
        loss-and-gradient evaluation for steady (IFT), transient reverse and transient
        forward mode.
    """
    case_dir = copy_case(INVERSION_CASE, "Savannah_gradcheck")
    control = read_json_with_commas(case_dir / "run_control.json")
    _forward_truth(case_dir, control, STEADY, "steady_truth.json")

    # Steady case and its loss, value and IFT gradient (reverse mode).
    steady_control = _inversion_control(control, STEADY, "steady_truth.json", [("LBFGS", 1.0, 1)])
    write_control(case_dir, steady_control)
    steady_case = setup_case(read_control(case_dir / "run_control.json"))
    steady_loss = jax.jit(_loss_function(steady_case, "steady_truth.json", "checkpoint"))
    steady_grad = jax.jit(jax.value_and_grad(_loss_function(steady_case, "steady_truth.json",
                                                            "checkpoint")))

    # Two parameter points: the start of the inversion and a point between start and truth.
    points = {
        "start": [0.02, N_START, N_START, N_START, N_START, N_START],
        "intermediate": [0.02, 0.036, 0.044, 0.034, 0.041, 0.043],
    }
    checks = {}
    for label, values in points.items():
        n_zone = jnp.asarray(values)
        loss_value, gradient = steady_grad(n_zone)
        gradient = np.asarray(gradient)

        # Central differences in zones 1..5 (zone 0 has no cells).
        fd = []
        for zone in range(1, 6):
            step = jnp.zeros(6).at[zone].set(FD_STEP)
            plus = float(steady_loss(n_zone + step))
            minus = float(steady_loss(n_zone - step))
            fd.append((plus - minus) / (2.0 * FD_STEP))
        fd = np.asarray(fd)
        ift = gradient[1:]
        checks[label] = {
            "n": values,
            "loss": float(loss_value),
            "grad_ift": ift.tolist(),
            "grad_fd": fd.tolist(),
            "relative_difference": (np.abs(ift - fd) / np.abs(fd)).tolist(),
        }

    # Cost of one loss-and-gradient evaluation (warm = compiled).
    n_start = jnp.asarray(points["start"])
    timings = {"steady_ift_reverse": _warm_time(steady_grad, n_start)}

    transient_control = _inversion_control(control, TRANSIENT, "transient_truth.json",
                                           [("LBFGS", 1.0, 1)])
    _transient_truth(case_dir, transient_control, "transient_truth.json")
    write_control(case_dir, transient_control)
    transient_case = setup_case(read_control(case_dir / "run_control.json"))
    reverse = jax.jit(jax.value_and_grad(_loss_function(transient_case, "transient_truth.json",
                                                        "checkpoint")))
    forward = jax.jit(jax.jacfwd(_loss_function(transient_case, "transient_truth.json",
                                                "forward")))
    timings["transient_reverse_checkpoint"] = _warm_time(reverse, n_start)
    timings["transient_forward_mode"] = _warm_time(forward, n_start)

    return {
        "fd_step": FD_STEP,
        "checks": checks,
        "gradient_cost_s": {key: {"first_call_incl_compile": value[0], "warm_median": value[1]}
                            for key, value in timings.items()},
        "machine": machine_info(),
    }


def main() -> None:
    """Run the requested computations and write ``paper/data/fig05_<run>.json``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for run in sys.argv[1:]:
        if run == "gradcheck":
            record = run_gradcheck()
        else:
            record = run_inversion_case(run)

        path = DATA_DIR / f"fig05_{run}.json"
        path.write_text(json.dumps(record, indent=1))
        print(f"{run}: wrote {path}", flush=True)


if __name__ == "__main__":
    main()
