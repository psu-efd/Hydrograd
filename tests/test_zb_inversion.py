"""Bathymetry (zb) inversion: consistent h_still, reverse mode, LBFGS stage.

The case is Julia's ``oneD_channel_with_bump`` (refined mesh) with its forward truth file as
observations. The bed elevation zb (m, one value per cell) is the control variable. With
decision D1 the still-water depth is h_still = w_still - zb, so changing zb changes h_still
consistently.

What is checked (both tests are slow):
- The reverse-mode gradient of the WSE loss with respect to zb is finite and matches a central
  finite difference in one cell to a relative error of 1e-3. A tight ODE tolerance
  (rtol 1e-9, atol 1e-12) is needed for both AD and FD to be accurate.
- Two Adam iterations followed by two L-BFGS iterations run, write five finite loss values,
  and reduce the loss.
"""

from __future__ import annotations

import pytest
import csv
import dataclasses
import json
import shutil

import jax
import jax.numpy as jnp
import numpy as np
from conftest import EXAMPLES, requires_jl

from hydrograd.apps.inversion import run_inversion
from hydrograd.apps.losses import data_losses, load_truth
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.solvers.ode import solve_swe

pytestmark = requires_jl


def _make_case(tmp_path):
    """Copy the refined bump case and its truth file, and write a zb-inversion run_control.json.

    The inversion runs t in [0, 20] s with Tsit5, starts from a flat zb = 0, and bounds zb to
    [0, 0.2] m with a slope bound of 0.2.

    Args:
        tmp_path: pytest temporary folder.

    Returns:
        The case folder (``tmp_path / "zb_inv"``).
    """
    # Copy the mesh and SRH-2D files plus the forward truth file
    source_dir = EXAMPLES / "forward_simulation" / "oneD_channel_with_bump"
    case_dir = tmp_path / "zb_inv"
    case_dir.mkdir()
    for case_file in source_dir.glob("oneD_channel_with_bump_refined.*"):
        shutil.copy(case_file, case_dir)
    shutil.copy(source_dir / "forward_simulation_solution_truth.json", case_dir)

    # Control file: zb inversion with WSE, uv, bound and slope losses; Adam then L-BFGS
    ctrl = {
        "bVerbose": False,
        "control_variables": {"srhhydro_file_name": "oneD_channel_with_bump_refined.srhhydro",
                              "bPerform_Inversion": True},
        "time_settings": {"bUse_srhhydro_time_settings": False, "tspan": [0.0, 20.0], "dt": 0.1},
        "inversion_options": {
            "active_param_names": ["zb"], "inversion_parameter_initial_values_options": "constant",
            "inversion_zb_initial_values": [0.0],
            "inversion_bInversion_WSE_loss": True, "inversion_bInversion_uv_loss": True,
            "inversion_bInversion_bound_loss": True, "inversion_bInversion_slope_loss": True,
            "inversion_parameter_value_lower_bound": 0.0, "inversion_parameter_value_upper_bound": 0.2,
            "inversion_parameter_slope_upper_bound": 0.2,
            "inversion_optimizers": ["Adam", "LBFGS"], "inversion_learning_rates": [0.001, 0.01],
            "inversion_max_iterations": [2, 2], "inversion_save_frequency": 100,
            "inversion_sensealg": "AutoZygote()",
            "inversion_truth_file_name": "forward_simulation_solution_truth.json",
            "inversion_forward_simulation_initial_condition_options": "constant",
            "inversion_forward_simulation_initial_condition_constant_values": [0.33, 0.33, 0.0, 0.0],
            "inversion_ode_solver_options": {"ode_solver": "Tsit5()", "ode_solver_adaptive": True,
                                             "ode_solver_sensealg": "InterpolatingAdjoint()", "ode_solver_nSave": 2},
        },
    }
    (case_dir / "run_control.json").write_text(json.dumps(ctrl))
    return case_dir


@pytest.mark.slow
def test_zb_gradient_matches_finite_difference(tmp_path) -> None:
    """d(WSE loss)/d(zb) in cell 100: reverse-mode AD vs central FD (step 1e-4 m), rel 1e-3."""
    # Set up the case and load the observations
    case_dir = _make_case(tmp_path)
    case = setup_case(read_control(case_dir / "run_control.json"))
    truth_text = (case_dir / "forward_simulation_solution_truth.json").read_text()
    truth = load_truth(json.loads(truth_text))

    # Gradients of a loosely integrated solve are poor (5x off at rtol 1e-3); use a tight tolerance.
    # At second order the limiter amplifies time-integration noise in finite differences
    # (FD at rtol 1e-6 was 3.5 % off while AD was stable); use a tight tolerance for the FD check.
    settings = dataclasses.replace(case.ode, rtol=1e-9, atol=1e-12)

    def loss(zb):
        """WSE data loss of the final state for the bed elevation zb (m), shape [n_cells]."""
        U = solve_swe(case.model, {"zb": zb}, case.U0, settings).ys[-1]
        return data_losses(case.model, U, truth, zb)["wse"]

    # Hydrostatic reconstruction uses max(z_L, z_R): with exactly equal neighbouring beds
    # (a flat bed) the residual has a kink and AD returns a subgradient. Avoid ties here.
    zb0 = jnp.asarray(0.05 + 0.01 * np.random.default_rng(1).random(case.model.n_cells))
    gradient = jax.jit(jax.grad(loss))(zb0)
    assert np.all(np.isfinite(np.asarray(gradient)))

    # Central FD in one cell
    i = 100
    eps = 1e-4  # FD with eps << sqrt(rtol) is dominated by adaptive-step noise
    perturbation = jnp.zeros_like(zb0).at[i].set(eps)
    fd = (float(loss(zb0 + perturbation)) - float(loss(zb0 - perturbation))) / (2 * eps)
    assert abs(float(gradient[i]) - fd) <= 1e-3 * abs(fd)


@pytest.mark.slow
def test_zb_inversion_adam_then_lbfgs_reduces_loss(tmp_path) -> None:
    """2 Adam + 2 L-BFGS iterations: five finite loss values, and the last below the first."""
    # Run the inversion
    case_dir = _make_case(tmp_path)
    run_inversion(setup_case(read_control(case_dir / "run_control.json")))

    # Read the loss history (column 1, header row skipped)
    loss_rows = list(csv.reader(open(case_dir / "inversion_loss_history.csv")))
    losses = [float(row[1]) for row in loss_rows[1:]]
    assert len(losses) == 5 and all(np.isfinite(losses))
    assert losses[-1] < losses[0]
