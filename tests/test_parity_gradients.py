"""Parity with Hydrograd.jl: sensitivity Jacobian, inversion and UDE loss trajectories.

All three tests run a Hydrograd.jl example with the Julia-compatible numerics
(``julia_compat``) and compare with the results that Julia saved in the example folder:

- sensitivity: the Jacobian d(outputs)/d(Manning's n per zone), column by column, to a
  relative 2-norm error of 1e-4;
- inversion: the first 3 total losses of the Savannah Manning's n inversion, rtol 5e-4;
- UDE: the first 4 Adam losses of the bump-channel n(h) training, started from Julia's
  initial NN weights, rtol 3e-3.

The tolerances reflect the different ODE integrators and adjoint methods of the two codes;
the trajectories of the optimizers diverge slowly from the first iteration.
"""

from __future__ import annotations

import csv
import json
import re

import jax
import numpy as np
import pytest
from conftest import EXAMPLES, julia_compat, requires_jl

from hydrograd.apps.inversion import run_inversion
from hydrograd.apps.sensitivity import run_sensitivity
from hydrograd.apps.setup import setup_case
from hydrograd.apps.ude import run_ude
from hydrograd.io.control import read_control

pytestmark = requires_jl


def _set_iterations(path, key: str, n: int) -> None:
    """Set ``"key": [n, 0]`` (optimizer iteration counts) in a ``run_control.json`` file.

    Args:
        path: path of the control file; edited in place.
        key: name of the iteration-count entry, e.g. ``"inversion_max_iterations"``.
        n: number of iterations of the first optimizer.
    """
    text = path.read_text()
    path.write_text(re.sub(rf'"{key}": \[[^\]]*\]', f'"{key}": [{n}, 0]', text))


def test_sensitivity_jacobian_matches_julia(case_copy) -> None:
    """Each Jacobian column matches Julia's to a relative 2-norm error below 1e-4.

    Columns for material zones without cells must be exactly zero.
    """
    # Run hydrograd's sensitivity analysis.
    rel = "sensitivity_analysis/ManningN/oneD_channel_with_bump"
    case_dir = case_copy(rel)
    case = julia_compat(setup_case(read_control(case_dir / "run_control.json")))
    jacobian = run_sensitivity(case)["J"]

    # Load Julia's Jacobian (stored flat, column-major).
    ref = json.loads((EXAMPLES / rel / "sensitivity_results.json").read_text())
    jacobian_julia = np.asarray(ref["sensitivity_results"]).reshape(jacobian.shape, order="F")

    # Compare column by column (one column per parameter).
    for column in range(jacobian.shape[1]):
        julia_norm = np.linalg.norm(jacobian_julia[:, column])
        if julia_norm > 0:
            column_diff = jacobian[:, column] - jacobian_julia[:, column]
            assert np.linalg.norm(column_diff) / julia_norm < 1e-4
        else:
            assert np.linalg.norm(jacobian[:, column]) == 0.0  # material zone without cells


def test_inversion_loss_trajectory_matches_julia(case_copy) -> None:
    """The first 3 total losses of the n inversion match Julia's to rtol 5e-4."""
    # Run 3 inversion iterations.
    rel = "inversion/ManningN_inversion/Savannah_River"
    case_dir = case_copy(rel)
    _set_iterations(case_dir / "run_control.json", "inversion_max_iterations", 3)
    run_inversion(julia_compat(setup_case(read_control(case_dir / "run_control.json"))))

    # Read hydrograd's loss history (column 1 = total loss; row 0 is the header).
    history_rows = list(csv.reader(open(case_dir / "inversion_loss_history.csv")))
    losses_python = [float(row[1]) for row in history_rows[1:4]]

    # Compare with Julia's first 3 total losses.
    julia_path = EXAMPLES / rel / "inversion_results_losses_and_ODE_solution.json"
    losses_julia = json.loads(julia_path.read_text())["loss_totals"][:3]
    np.testing.assert_allclose(losses_python, losses_julia, rtol=5e-4)


def test_ude_loss_trajectory_matches_julia(case_copy) -> None:
    """Start from Julia's initial NN weights (from the RHS dump) and compare Adam losses.

    The first 4 total losses must match Julia's to rtol 3e-3. Skips if the RHS dump with
    the NN weights is missing.
    """
    from test_parity_rhs import REF, julia_nn_params

    from hydrograd.apps.ude import save_weights

    # Load Julia's initial NN weights from the RHS dump, or skip.
    rel = "UDE/ManningN/oneD_channel_with_bump/training"
    ref_file = REF / "UDE_ManningN_oneD_channel_with_bump_training.json"
    if not ref_file.exists():
        pytest.skip(f"no Julia dump {ref_file.name}")
    ref = json.loads(ref_file.read_text())

    # Save them as hydrograd weights in the case copy (NN 1 -> [3, 3] -> 1).
    case_dir = case_copy(rel)
    initial_params = julia_nn_params(np.asarray(ref["nn_params_flat"]), [3, 3], 1, 1)
    save_weights(case_dir / "julia_init.npz", initial_params)

    # Edit the control file: start from those weights and run 4 Adam iterations.
    # Trailing commas (allowed in Julia's JSON reader) are removed first.
    ctrl_path = case_dir / "run_control.json"
    ctrl = json.loads(re.sub(r",(\s*[}\]])", r"\1", ctrl_path.read_text()))
    ude_options = ctrl["UDE_options"]
    ude_options["UDE_NN_config"].update(
        how_to_initialize_NN="from_pretrained",
        NN_weights_state_file_name="julia_init.npz",
    )
    ude_options["UDE_max_iterations"] = [4, 0]
    ctrl_path.write_text(json.dumps(ctrl))

    # Train and keep the first 4 total losses (entry 1 of each history record).
    case = julia_compat(setup_case(read_control(ctrl_path)))
    history = run_ude(case)["history"]
    losses_python = [record[1] for record in history][:4]

    # Compare with Julia's losses.
    julia_path = EXAMPLES / rel / "UDE_results_losses_and_ODE_solution.json"
    losses_julia = json.loads(julia_path.read_text())["loss_totals"][:4]
    np.testing.assert_allclose(losses_python, losses_julia, rtol=3e-3)

    # Free the compiled JAX programs of this large test.
    jax.clear_caches()
