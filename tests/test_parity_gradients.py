"""Parity with Hydrograd.jl: sensitivity Jacobian, inversion and UDE loss trajectories."""

from __future__ import annotations

import csv
import json
import re

import jax
import numpy as np
from conftest import EXAMPLES, julia_compat, requires_jl

from hydrograd.apps.inversion import run_inversion
from hydrograd.apps.sensitivity import run_sensitivity
from hydrograd.apps.setup import setup_case
from hydrograd.apps.ude import run_ude
from hydrograd.io.control import read_control

pytestmark = requires_jl


def _set_iterations(path, key: str, n: int) -> None:
    t = path.read_text()
    path.write_text(re.sub(rf'"{key}": \[[^\]]*\]', f'"{key}": [{n}, 0]', t))


def test_sensitivity_jacobian_matches_julia(case_copy) -> None:
    rel = "sensitivity_analysis/ManningN/oneD_channel_with_bump"
    d = case_copy(rel)
    J = run_sensitivity(julia_compat(setup_case(read_control(d / "run_control.json"))))["J"]
    ref = json.loads((EXAMPLES / rel / "sensitivity_results.json").read_text())
    Jj = np.asarray(ref["sensitivity_results"]).reshape(J.shape, order="F")
    for i in range(J.shape[1]):
        nrm = np.linalg.norm(Jj[:, i])
        if nrm > 0:
            assert np.linalg.norm(J[:, i] - Jj[:, i]) / nrm < 1e-4
        else:
            assert np.linalg.norm(J[:, i]) == 0.0  # material zone without cells


def test_inversion_loss_trajectory_matches_julia(case_copy) -> None:
    rel = "inversion/ManningN_inversion/Savannah_River"
    d = case_copy(rel)
    _set_iterations(d / "run_control.json", "inversion_max_iterations", 3)
    run_inversion(julia_compat(setup_case(read_control(d / "run_control.json"))))
    py = [float(r[1]) for r in list(csv.reader(open(d / "inversion_loss_history.csv")))[1:4]]
    jl = json.loads((EXAMPLES / rel / "inversion_results_losses_and_ODE_solution.json").read_text())["loss_totals"][:3]
    np.testing.assert_allclose(py, jl, rtol=5e-4)


def test_ude_loss_trajectory_matches_julia(case_copy) -> None:
    """Start from Julia's initial NN weights (from the RHS dump) and compare Adam losses."""
    from test_parity_rhs import REF, julia_nn_params

    from hydrograd.apps.ude import save_weights

    rel = "UDE/ManningN/oneD_channel_with_bump/training"
    ref = json.loads((REF / "UDE_ManningN_oneD_channel_with_bump_training.json").read_text())
    d = case_copy(rel)
    save_weights(d / "julia_init.npz", julia_nn_params(np.asarray(ref["nn_params_flat"]), [3, 3], 1, 1))
    ctrl = json.loads(re.sub(r",(\s*[}\]])", r"\1", (d / "run_control.json").read_text()))
    u = ctrl["UDE_options"]
    u["UDE_NN_config"].update(how_to_initialize_NN="from_pretrained", NN_weights_state_file_name="julia_init.npz")
    u["UDE_max_iterations"] = [4, 0]
    (d / "run_control.json").write_text(json.dumps(ctrl))
    hist = [h[1] for h in run_ude(julia_compat(setup_case(read_control(d / "run_control.json"))))["history"]][:4]
    jl = json.loads((EXAMPLES / rel / "UDE_results_losses_and_ODE_solution.json").read_text())["loss_totals"][:4]
    np.testing.assert_allclose(hist, jl, rtol=3e-3)
    jax.clear_caches()
