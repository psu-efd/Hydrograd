"""Savannah River UDE case: Hydrograd v1.0 timings for the comparison with Hydrograd.jl.

Usage:
    python paper/perf_savannah.py OUT.jsonl CASE_DIR [FORWARD_CASE_DIR]

CASE_DIR is a copy of Hydrograd.jl's ``examples/SWE_2D/UDE/ManningN/Savannah_River_ks_h_Umag``
(1306 cells, 200 s simulated from the given initial condition, Tsit5, UDE closure
n = NN(h, |U|, ks) with a 3-3-3-1 tanh MLP, loss = WSE + velocity misfit of the final
state against the truth file). ``perf_julia_savannah.jl`` times Hydrograd.jl on the same
case and the same node.

Timed here (compiled runs; the first call with compilation is recorded separately):

- forward run at the control-file tolerances (Tsit5, rtol 1e-3, atol 1e-6);
- forward run at the derivative tolerances (rtol 1e-7, atol 1e-10, used for all gradients,
  see ``hydrograd.apps.sensitivity.DERIV_RTOL``);
- the loss gradient with respect to the MLP weights in forward mode (``jax.jacfwd`` through
  diffrax ``ForwardMode``, the analogue of Julia's ``AutoForwardDiff``);
- the same gradient in reverse mode (recursive checkpointing, 64 checkpoints, the package
  default for networks with more than 16 weights).

If FORWARD_CASE_DIR (a copy of Hydrograd.jl's ``examples/SWE_2D/forward_simulation/
Savannah_River``, constant Manning n per zone) is given, its forward run at the control-file
settings is timed as well (``fwd_case_s``).

Times in s, memory in GB; one JSON line is appended to OUT.jsonl.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import jax
import numpy as np

import hydrograd  # noqa: F401  (enables float64)
from hydrograd.apps.losses import data_losses, load_truth
from hydrograd.apps.sensitivity import DERIV_RTOL
from hydrograd.apps.setup import setup_case
from hydrograd.closures.mlp import init_mlp
from hydrograd.io.control import read_control
from hydrograd.solvers.final import final_state

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_common import environment_record, peak_memory_gb, timed

# Seed of the initial MLP weights (as in hydrograd.apps.ude).
SEED = 1111


def main() -> None:
    """Set up the case, then time the forward runs and both gradient modes."""
    # 1. Case, truth and initial MLP weights.
    out = Path(sys.argv[1])
    case_dir = Path(sys.argv[2])
    case = setup_case(read_control(case_dir / "run_control.json"))
    model = case.model
    options = case.ctrl.opt
    truth = load_truth(json.loads((case_dir / options("UDE_truth_file_name")).read_text()))
    nn_config = options("UDE_NN_config")
    hidden_widths = [int(w) for w in nn_config["hidden_layers"]]
    weights = init_mlp(jax.random.PRNGKey(SEED), int(nn_config["input_dim"]), hidden_widths,
                       int(nn_config["output_dim"]), layernorm=False)
    n_weights = sum(int(np.size(v)) for layer in weights for v in layer.values())

    # 2. ODE settings: control-file tolerances, and derivative tolerances per adjoint.
    settings_ctrl = case.ode
    settings_deriv = dataclasses.replace(case.ode, rtol=DERIV_RTOL, atol=DERIV_RTOL * 1.0e-3)
    settings_fwd_mode = dataclasses.replace(settings_deriv, adjoint="forward")
    settings_rev_mode = dataclasses.replace(settings_deriv, adjoint="checkpoint")

    # 3. Loss of the MLP weights for given ODE settings.
    def loss(th, settings):
        """WSE + velocity misfit (dimensionless) of the final state for MLP weights th."""
        U, _ = final_state(model, {"nn": th}, case.U0, settings)
        terms = data_losses(model, U, truth)
        return terms["wse"] + terms["uv"]

    def loss_fwd_mode(th):
        """Loss with the forward-mode adjoint (for jax.jacfwd)."""
        return loss(th, settings_fwd_mode)

    def loss_rev_mode(th):
        """Loss with the checkpointed reverse-mode adjoint (for jax.grad)."""
        return loss(th, settings_rev_mode)

    def forward_ctrl(th):
        """Final state (shape [3, n_cells]) at the control-file tolerances."""
        return final_state(model, {"nn": th}, case.U0, settings_ctrl)[0]

    def forward_deriv(th):
        """Final state (shape [3, n_cells]) at the derivative tolerances."""
        return final_state(model, {"nn": th}, case.U0, settings_deriv)[0]

    # 4. Timings (each function: first call with compilation, then the compiled call).
    record = {"case": "Savannah_River_ks_h_Umag", "cells": int(case.U0.shape[1]),
              "n_weights": n_weights, "t_end_s": float(settings_ctrl.t1),
              "solver": settings_ctrl.solver}
    _, record["fwd_ctrl_compile_s"], record["fwd_ctrl_s"] = timed(jax.jit(forward_ctrl), weights)
    _, record["fwd_deriv_compile_s"], record["fwd_deriv_s"] = timed(jax.jit(forward_deriv),
                                                                    weights)
    g_rev, record["grad_rev_compile_s"], record["grad_rev_s"] = timed(
        jax.jit(jax.grad(loss_rev_mode)), weights)
    record["peak_mem_rev_gb"] = peak_memory_gb()
    g_fwd, record["grad_fwd_compile_s"], record["grad_fwd_s"] = timed(
        jax.jit(jax.jacfwd(loss_fwd_mode)), weights)

    # 5. Agreement of the two gradient modes (relative 2-norm difference).
    flat_rev = np.concatenate([np.ravel(x) for x in jax.tree_util.tree_leaves(g_rev)])
    flat_fwd = np.concatenate([np.ravel(x) for x in jax.tree_util.tree_leaves(g_fwd)])
    record["grad_mode_rel_diff"] = float(np.linalg.norm(flat_rev - flat_fwd)
                                         / np.linalg.norm(flat_fwd))
    record["loss"] = float(jax.jit(loss_rev_mode)(weights))

    # 6. Optional: forward run of the constant-n forward-simulation case.
    if len(sys.argv) > 3:
        fwd_case = setup_case(read_control(Path(sys.argv[3]) / "run_control.json"))

        def forward_case(U):
            """Final state (shape [3, n_cells]) of the forward case from U."""
            return final_state(fwd_case.model, {}, U, fwd_case.ode)[0]

        _, record["fwd_case_compile_s"], record["fwd_case_s"] = timed(jax.jit(forward_case),
                                                                      fwd_case.U0)
        record["fwd_case_rtol"] = fwd_case.ode.rtol
    record["environment"] = environment_record()

    print(json.dumps(record), flush=True)
    with open(out, "a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
