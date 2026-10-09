"""Figure 6: steady UDE training (Newton + IFT) against transient time marching (bump case).

Purpose
-------
Hydrograd v1.0 trains a neural-network (NN) Manning's-n closure n = NN(h) inside the 2D
shallow-water equations (SWEs), i.e. a universal differential equation (UDE). The WRR paper
(Liu and Song, 2025) obtained the gradient of the loss by differentiating through explicit
time marching to a near-steady state. Hydrograd v1.0 can instead solve the steady state
directly and differentiate it with the implicit function theorem (IFT). This script trains the
same closure both ways on the 1D channel with a bump and compares cost and accuracy.

Physics and mathematics
-----------------------
State U = [xi, q_x, q_y] per cell (xi: free-surface displacement above still water (m); q_x,
q_y: unit discharges (m^2/s)); finite-volume residual R(U, theta) = dU/dt, with theta the NN
weights. The friction source uses Manning's law with n = NN(h) (s/m^(1/3)), a local law.

* Steady path (``SteadyNewton()``): U* solves R(U*, theta) = 0 by pseudo-transient Newton on
  the exact sparse Jacobian. Gradient by the IFT: dL/dtheta = -lambda^T dR/dtheta with
  (dR/dU)^T lambda = dL/dU*, i.e. one transposed sparse solve per iteration.
* Transient path (``Tsit5()``, as in the WRR paper): U(t_end) from adaptive explicit
  Runge-Kutta over 0 <= t <= 100 s, reverse-mode AD through all steps (diffrax, checkpointed),
  at the package's derivative tolerance rtol = 1e-7 (``DERIV_RTOL``).

True closure (synthetic truth): the sigmoid law of Hydrograd.jl's example,

    n(h) = n_lo + (n_hi - n_lo) / (1 + exp(k (h - h_mid))),
    n_lo = 0.03, n_hi = 0.06 s/m^(1/3), k = 100 1/m, h_mid = 0.3 m.

The truth is the *steady* state with this law (Newton, max|R| < 1e-8). Loss (both paths):

    L = mean(((WSE - WSE_obs) / range(WSE_obs))^2)
        + sum((u - u_obs)^2 + (v - v_obs)^2) / (|U|_max,obs^2 N_cells),

with u = q_x / (h + h_small). Both paths start from the same NN weights
(``jax.random.PRNGKey(1111)``; 1 -> [3, 3] -> 1 tanh MLP, no LayerNorm, 22 weights) and use
Adam (learning rate 0.1, 100 iterations), as in the Julia example.

Outputs
-------
* ``paper/figures/fig06_steady_ude.pdf`` (+ ``.png``): (a) loss history, (b) learned n(h)
  vs truth, (c) wall time per iteration and in total, (d) WSE profile along the channel.
* ``paper/results/steady_ude.json``: timings (s), iterations, final losses (-), RMS error of
  n(h) (s/m^(1/3)), NN size, hardware, and the steady-state residual of the transient run.
* ``paper/data/fig06_steady_ude/``: case copies, training outputs and ``curves.npz``.

Usage (from the repository root, with ``hpc/roar/env.sh`` sourced and
``JAX_PLATFORMS=cpu``)::

    python paper/fig06_steady_ude.py            # train if no saved data, then plot
    python paper/fig06_steady_ude.py --rerun    # always retrain (about 35 min on 8 cores)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import hydrograd  # noqa: F401  (enables float64)
import hydrograd.apps.ude as ude_app
from hydrograd.apps.forward import derived_fields, run_forward
from hydrograd.apps.sensitivity import DERIV_RTOL
from hydrograd.apps.setup import Case, setup_case
from hydrograd.io.control import read_control
from hydrograd.physics.rhs import manning_n_cells, rhs
from hydrograd.solvers.final import newton_settings
from hydrograd.solvers.ode import solve_swe
from hydrograd.solvers.steady import steady_state

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (
    FULL_WIDTH,
    MODEL,
    MODEL_2,
    REFERENCE,
    apply_style,
    panel_label,
    save,
)

logger = logging.getLogger("fig06")

# ----------------------------------------------------------------------
# Paths and case settings
# ----------------------------------------------------------------------
PAPER = Path(__file__).resolve().parent
JULIA_EXAMPLES = PAPER.parents[1] / "Hydrograd.jl" / "examples" / "SWE_2D"
UDE_EXAMPLE = JULIA_EXAMPLES / "UDE" / "ManningN" / "oneD_channel_with_bump" / "training"
FORWARD_EXAMPLE = JULIA_EXAMPLES / "forward_simulation" / "oneD_channel_with_bump_ManningN_h"
DATA_DIR = PAPER / "data" / "fig06_steady_ude"
CURVES_FILE = DATA_DIR / "curves.npz"
RESULTS_FILE = PAPER / "results" / "steady_ude.json"
TRUTH_FILE = "steady_truth.json"

# The two training paths: label -> ode_solver name.
SOLVERS = {"steady": "SteadyNewton()", "transient": "Tsit5()"}

# Training settings shared by both paths (Julia example: Adam, lr 0.1, 100 iterations).
N_ITER = 100
LEARNING_RATE = 0.1

# True sigmoid law n(h) of the forward example (n in s/m^(1/3), k in 1/m, h_mid in m).
SIGMOID = {"n_lower": 0.03, "n_upper": 0.06, "k": 100.0, "h_mid": 0.3}


def _read_json(path: Path) -> dict[str, Any]:
    """Read a Julia run-control file (JSON that may contain trailing commas).

    Args:
        path: the file.

    Returns:
        The parsed dictionary.
    """
    text = path.read_text()
    return json.loads(re.sub(r",(\s*[}\]])", r"\1", text))


def _copy_case(src: Path, dst: Path) -> None:
    """Copy a Julia example folder (never written to) to ``dst``, replacing old copies.

    Args:
        src: example folder in Hydrograd.jl.
        dst: destination folder under ``paper/data``.
    """
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)


def make_truth() -> Path:
    """Steady truth with the sigmoid n(h) law (Newton solve), written to ``TRUTH_FILE``.

    Returns:
        Path of the truth JSON file.
    """
    case_dir = DATA_DIR / "truth"
    _copy_case(UDE_EXAMPLE, case_dir)

    # Forward controls of the n(h) example, switched to the steady Newton solver.
    control = _read_json(FORWARD_EXAMPLE / "run_control.json")
    control["control_variables"]["srhhydro_file_name"] = "oneD_channel_with_bump_refined.srhhydro"

    options = control["forward_simulation_options"]
    options["forward_simulation_ode_solver"] = "SteadyNewton()"
    options["forward_simulation_nSave"] = 1
    options["forward_simulation_save_solution_truth_file_name"] = TRUTH_FILE

    # The Julia example lacks "ks", which the n(h) laws require (unused by the sigmoid).
    options["forward_simulation_ManningN_function_parameters"] = dict(SIGMOID, ks=[0.0])

    control_path = case_dir / "run_control_truth.json"
    control_path.write_text(json.dumps(control, indent=2))
    run_forward(setup_case(read_control(control_path)))

    return case_dir / TRUTH_FILE


def ude_case(label: str, truth_path: Path, n_iter: int) -> Case:
    """Set up the UDE training case of one path (own folder, steady truth copied in).

    Args:
        label: ``"steady"`` or ``"transient"`` (key of ``SOLVERS``).
        truth_path: the steady truth JSON.
        n_iter: number of Adam iterations.

    Returns:
        The set-up case.
    """
    case_dir = DATA_DIR / label
    _copy_case(UDE_EXAMPLE, case_dir)
    shutil.copy(truth_path, case_dir / TRUTH_FILE)

    # Julia's UDE example with the shared training settings and this path's solver.
    control = _read_json(UDE_EXAMPLE / "run_control.json")
    options = control["UDE_options"]
    options["UDE_truth_file_name"] = TRUTH_FILE
    options["UDE_optimizers"] = ["Adam"]
    options["UDE_learning_rates"] = [LEARNING_RATE]
    options["UDE_max_iterations"] = [n_iter]
    options["UDE_abs_tols"] = [0.0]
    options["UDE_rel_tols"] = [0.0]
    options["UDE_training_save_frequency"] = 10 * n_iter   # fields only at iteration 1
    options["UDE_sensealg"] = "AutoZygote()"               # reverse mode for both paths
    options["UDE_ode_solver_options"]["ode_solver"] = SOLVERS[label]
    options["UDE_NN_config"]["layernorm"] = False

    # Reverse mode through the transient run: store every step if memory allows.
    control["numerics"] = {"checkpoints": "auto"}

    control_path = case_dir / "run_control.json"
    control_path.write_text(json.dumps(control, indent=2))
    return setup_case(read_control(control_path))


def train(case: Case) -> dict[str, Any]:
    """Train the closure with the package's UDE app and time every iteration.

    The app's optimizer loop calls a callback once per iteration; a thin wrapper around
    ``run_stages`` records the wall clock at each call, so the difference of consecutive
    stamps is the cost of one loss-and-gradient evaluation plus the Adam update.

    Args:
        case: the UDE case (from :func:`ude_case`).

    Returns:
        Dict with ``params`` (trained weights), ``loss`` [n_iter + 1] (-), ``stamps``
        [n_iter + 1] (s, from the start of training) and ``total_s`` (s, incl. compilation).
    """
    stamps: list[float] = []
    original_run_stages = ude_app.run_stages
    t_start = time.perf_counter()

    def timed_run_stages(make_loss, theta0, stages, callback, forward_mode=False):
        """``run_stages`` with a callback that records the time of each iteration."""

        def timed_callback(it, theta, loss, aux):
            """Record the time, then call the app's callback."""
            stamps.append(time.perf_counter() - t_start)
            callback(it, theta, loss, aux)

        return original_run_stages(make_loss, theta0, stages, timed_callback, forward_mode)

    # Run the app with the timing wrapper, then restore the original loop.
    ude_app.run_stages = timed_run_stages
    try:
        result = ude_app.run_ude(case)
    finally:
        ude_app.run_stages = original_run_stages
    total_s = time.perf_counter() - t_start

    loss = np.array([row[1] for row in result["history"]])
    return {"params": result["params"], "loss": loss, "stamps": np.array(stamps),
            "total_s": total_s}


def steady_of(case: Case, params: Any) -> np.ndarray:
    """Steady state (Newton) of the model with NN weights ``params``.

    Args:
        case: a UDE case.
        params: NN weights (pytree).

    Returns:
        Steady state [3, n_cells].
    """
    theta = {"nn": params}
    U = steady_state(case.model, theta, case.U0, newton_settings(case.ode))
    return np.asarray(U)


def transient_check(case: Case, params: Any) -> dict[str, float]:
    """Is the Tsit5 run at t_end steady? Residual and distance to the true steady state.

    Args:
        case: the transient UDE case.
        params: NN weights (pytree).

    Returns:
        Dict with ``max_abs_dUdt`` (max |R(U(t_end))|, m/s and m^2/s^2 mixed, as in the
        package's steady test), ``max_abs_dxi_dt`` (m/s), ``max_abs_wse_diff_to_steady``
        (m), ``rtol``, ``atol`` and the number of accepted Tsit5 steps.
    """
    theta = {"nn": params}
    settings = dataclasses.replace(case.ode, rtol=min(case.ode.rtol, DERIV_RTOL),
                                   atol=min(case.ode.atol, DERIV_RTOL * 1e-3), adjoint="direct")

    # Transient run to t_end with the training tolerances, and its residual.
    solution = solve_swe(case.model, theta, case.U0, settings)
    U_end = solution.ys[-1]
    residual = np.asarray(rhs(case.model, theta, U_end))

    # Distance to the steady state with the same weights (WSE difference = xi difference).
    U_steady = steady_of(case, params)
    wse_diff = np.max(np.abs(np.asarray(U_end)[0] - U_steady[0]))

    return {
        "t_end_s": float(case.ode.t1),
        "max_abs_dUdt": float(np.max(np.abs(residual))),
        "max_abs_dxi_dt_m_per_s": float(np.max(np.abs(residual[0]))),
        "max_abs_wse_diff_to_steady_m": float(wse_diff),
        "rtol": float(settings.rtol),
        "atol": float(settings.atol),
        "tsit5_steps": int(solution.stats["num_steps"]),
    }


def n_of_h(case: Case, params: Any, h: np.ndarray) -> np.ndarray:
    """Manning's n of the model's closure at depths ``h`` (zero discharge).

    Args:
        case: a case (its model's resistance mode selects law or NN).
        params: NN weights (pytree), or None for the law model.
        h: depths (m), shape [n].

    Returns:
        n (s/m^(1/3)), shape [n].
    """
    theta = {} if params is None else {"nn": params}
    h_j = jnp.asarray(h)
    zeros = jnp.zeros_like(h_j)
    return np.asarray(manning_n_cells(case.model, theta, h_j, zeros, zeros))


def sigmoid_n(h: np.ndarray) -> np.ndarray:
    """True sigmoid law n(h) (see the module docstring).

    Args:
        h: depths (m), any shape.

    Returns:
        n (s/m^(1/3)), same shape.
    """
    span = SIGMOID["n_upper"] - SIGMOID["n_lower"]
    return SIGMOID["n_lower"] + span / (1.0 + np.exp(SIGMOID["k"] * (h - SIGMOID["h_mid"])))


def hardware() -> dict[str, Any]:
    """Description of the machine the runs used (CPU model, cores, JAX backend).

    Returns:
        Dict with ``cpu``, ``cores_available``, ``host``, ``jax_backend`` and versions.
    """
    cpu = platform.processor()
    lscpu = subprocess.run(["lscpu"], capture_output=True, text=True, check=False).stdout
    for line in lscpu.splitlines():
        if line.startswith("Model name"):
            cpu = line.split(":", 1)[1].strip()

    return {
        "cpu": cpu,
        "cores_available": len(os.sched_getaffinity(0)),
        "host": platform.node(),
        "jax_backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "float": "float64",
    }


def run_all(n_iter: int) -> None:
    """Make the truth, train both paths, evaluate, and save curves and results.

    Args:
        n_iter: number of Adam iterations of each path.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Steady truth with the sigmoid law
    # ------------------------------------------------------------------
    truth_path = make_truth()
    truth = json.loads(truth_path.read_text())
    h_truth = np.asarray(truth["h_truth"])

    # ------------------------------------------------------------------
    # 2. Train both paths from the same initial weights
    # ------------------------------------------------------------------
    cases: dict[str, Case] = {}
    runs: dict[str, dict[str, Any]] = {}
    for label, solver_name in SOLVERS.items():
        logger.info("training path '%s' (%s)", label, solver_name)
        cases[label] = ude_case(label, truth_path, n_iter)
        runs[label] = train(cases[label])

    # Same initial weights: identical initial NN in both paths.
    init_same = bool(np.isclose(runs["steady"]["loss"][0], runs["transient"]["loss"][0],
                                rtol=1e-2))

    # ------------------------------------------------------------------
    # 3. Learned n(h) on the depth range of the data, and per-cell errors
    # ------------------------------------------------------------------
    h_grid = np.linspace(h_truth.min(), h_truth.max(), 200)
    curves: dict[str, np.ndarray] = {"h_grid": h_grid, "n_true_grid": sigmoid_n(h_grid)}

    # Model-side check of the truth law (package law evaluated on the grid).
    mesh_x = np.asarray(cases["steady"].mesh.centroid)[:, 0]
    order = np.argsort(mesh_x)
    curves["x"] = mesh_x[order]
    curves["zb"] = np.asarray(truth["zb_cell_truth"])[order]
    curves["wse_true"] = np.asarray(truth["wse_truth"])[order]

    results: dict[str, Any] = {}
    for label, case in cases.items():
        run = runs[label]
        params = run["params"]

        # n(h) on the grid and at the observed cell depths.
        n_grid = n_of_h(case, params, h_grid)
        n_cells = n_of_h(case, params, h_truth)
        rms_grid = float(np.sqrt(np.mean((n_grid - curves["n_true_grid"]) ** 2)))
        rms_cells = float(np.sqrt(np.mean((n_cells - sigmoid_n(h_truth)) ** 2)))

        # Steady WSE of the trained model (both paths evaluated at steady state).
        U_star = steady_of(case, params)
        fields = derived_fields(case.model, {"nn": params}, U_star, case.wstill)
        wse_rms = float(np.sqrt(np.mean((fields["wse"] - np.asarray(truth["wse_truth"])) ** 2)))

        curves[f"n_{label}_grid"] = n_grid
        curves[f"wse_{label}"] = fields["wse"][order]
        curves[f"loss_{label}"] = run["loss"]
        curves[f"stamps_{label}"] = run["stamps"]

        # Per-iteration cost: stamp k is taken after iteration k + 1, so stamps[0] includes
        # compilation and the differences are the cost of the later (compiled) iterations.
        intervals = np.diff(run["stamps"])
        n_par = sum(int(np.size(v)) for layer in params for v in layer.values())
        results[label] = {
            "ode_solver": SOLVERS[label],
            "n_iterations": n_iter,
            "first_iteration_incl_compile_s": float(run["stamps"][0]),
            "time_per_iteration_median_s": float(np.median(intervals)),
            "time_per_iteration_mean_s": float(np.mean(intervals)),
            "total_training_time_s": float(run["total_s"]),
            "initial_loss": float(run["loss"][0]),
            "final_loss": float(run["loss"][-1]),
            "min_loss": float(np.min(run["loss"])),
            "rms_n_error_grid_s_per_m13": rms_grid,
            "rms_n_error_cells_s_per_m13": rms_cells,
            "max_n_error_grid_s_per_m13": float(np.max(np.abs(n_grid - curves["n_true_grid"]))),
            "steady_wse_rms_error_m": wse_rms,
            "nn_parameters": n_par,
        }

    # ------------------------------------------------------------------
    # 4. Did the transient run reach steady state at t_end?
    # ------------------------------------------------------------------
    results["transient"]["steady_state_check_trained"] = transient_check(
        cases["transient"], runs["transient"]["params"])
    results["transient"]["steady_state_check_initial"] = transient_check(
        cases["transient"], ude_app._init_params(cases["transient"]))

    # ------------------------------------------------------------------
    # 5. Save curves and the results file
    # ------------------------------------------------------------------
    np.savez(CURVES_FILE, **curves)

    nn_config = cases["steady"].ctrl.raw["UDE_options"]["UDE_NN_config"]
    summary = {
        "case": "oneD_channel_with_bump (refined, 200 cells), steady truth with sigmoid n(h)",
        "true_law": dict(SIGMOID, form="n = n_lo + (n_hi - n_lo)/(1 + exp(k (h - h_mid)))"),
        "depth_range_of_data_m": [float(h_truth.min()), float(h_truth.max())],
        "nn": {"layers": [1, *nn_config["hidden_layers"], 1],
               "activations": nn_config["activations"], "layernorm": False,
               "h_bounds_m": nn_config["h_bounds"], "output_bounds": nn_config["output_bounds"],
               "init": "jax.random.PRNGKey(1111), identical for both paths"},
        "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE, "iterations": n_iter},
        "loss": "L_WSE + L_uv of the final/steady state (package losses.data_losses)",
        "numerics": {"scheme": cases["steady"].model.scheme,
                     "spatial_order": int(cases["steady"].model.order),
                     "transient_tspan_s": [float(cases["transient"].ode.t0),
                                           float(cases["transient"].ode.t1)],
                     "transient_gradient": "reverse mode, diffrax checkpointing ('auto')",
                     "steady_gradient": "IFT, one transposed sparse solve"},
        "initial_losses_agree": init_same,
        "hardware": hardware(),
        "speedup_per_iteration": results["transient"]["time_per_iteration_median_s"]
        / results["steady"]["time_per_iteration_median_s"],
        "speedup_total": results["transient"]["total_training_time_s"]
        / results["steady"]["total_training_time_s"],
        "steady": results["steady"],
        "transient": results["transient"],
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(summary, indent=2))


def plot() -> None:
    """Draw Figure 6 from ``curves.npz`` and ``steady_ude.json``."""
    c = np.load(CURVES_FILE)
    res = json.loads(RESULTS_FILE.read_text())

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 5.2))
    styles = {"steady": MODEL, "transient": MODEL_2}
    labels = {"steady": r"Steady + IFT", "transient": r"Time marching"}

    # ------------------------------------------------------------------
    # (a) Loss history
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    for key in ("transient", "steady"):
        loss = c[f"loss_{key}"]
        ax.semilogy(np.arange(1, loss.size + 1), loss, label=labels[key], **styles[key])
    ax.set_xlabel(r"Iteration")
    ax.set_ylabel(r"Loss $L$ (--)")
    ax.legend(loc="upper right")
    panel_label(ax, "(a)")

    # ------------------------------------------------------------------
    # (b) Learned n(h) against the true sigmoid law
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    ax.plot(c["h_grid"], c["n_true_grid"], label=r"Truth", **REFERENCE)
    for key in ("transient", "steady"):
        ax.plot(c["h_grid"], c[f"n_{key}_grid"], label=labels[key], **styles[key])
    ax.set_xlabel(r"$h$ (m)")
    ax.set_ylabel(r"$n$ (s\,m$^{-1/3}$)")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.42))
    panel_label(ax, "(b)")

    # ------------------------------------------------------------------
    # (c) Wall time per iteration and in total
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    keys = ("steady", "transient")
    per_iter = [res[k]["time_per_iteration_median_s"] for k in keys]
    totals = [res[k]["total_training_time_s"] for k in keys]
    positions = np.arange(2)
    colours = [MODEL["color"], MODEL_2["color"]]
    bars = ax.bar(positions, per_iter, width=0.55, color=colours, alpha=0.8)
    ax.set_yscale("log")
    ax.set_ylim(min(per_iter) / 3.0, max(per_iter) * 8.0)
    ax.set_xticks(positions)
    ax.set_xticklabels([r"Steady + IFT", r"Time marching"])
    ax.set_ylabel(r"Wall time per iteration (s)")

    # Annotate each bar with its total training time.
    for bar, per, total in zip(bars, per_iter, totals):
        text = rf"{per:.2g} s/it" + "\n" + rf"total {total:.0f} s"
        ax.text(bar.get_x() + bar.get_width() / 2.0, per * 1.3, text, ha="center",
                va="bottom", fontsize=8)
    speedup = res["speedup_per_iteration"]
    ax.text(0.03, 0.95, rf"{speedup:.0f}$\times$ faster per iteration", transform=ax.transAxes,
            ha="left", va="top", fontsize=9)
    panel_label(ax, "(c)")

    # ------------------------------------------------------------------
    # (d) Steady WSE along the channel: bed, truth and both trained models
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    ax.fill_between(c["x"], c["zb"].min() - 0.05, c["zb"], color="0.8", lw=0)
    ax.plot(c["x"], c["zb"], color="0.4", lw=0.8, label=r"Bed")
    ax.plot(c["x"], c["wse_true"], label=r"Truth", **REFERENCE)
    for key in ("transient", "steady"):
        ax.plot(c["x"], c[f"wse_{key}"], label=labels[key], **styles[key])
    ax.set_xlim(c["x"].min(), c["x"].max())
    ax.set_ylim(c["zb"].min() - 0.02, c["wse_true"].max() + 0.12)
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"Elevation (m)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.04))
    panel_label(ax, "(d)")

    fig.tight_layout()
    save(fig, "fig06_steady_ude")


def main() -> None:
    """Train (if needed or requested) and draw the figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rerun", action="store_true", help="retrain even if data exist")
    parser.add_argument("--iterations", type=int, default=N_ITER, help="Adam iterations")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("jax").setLevel(logging.WARNING)

    # Training is skipped when saved curves and results exist.
    if args.rerun or not (CURVES_FILE.exists() and RESULTS_FILE.exists()):
        run_all(args.iterations)
    plot()


if __name__ == "__main__":
    main()
