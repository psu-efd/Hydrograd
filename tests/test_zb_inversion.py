"""Bathymetry (zb) inversion: consistent h_still, reverse mode, LBFGS stage."""

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
    src = EXAMPLES / "forward_simulation" / "oneD_channel_with_bump"
    d = tmp_path / "zb_inv"
    d.mkdir()
    for f in src.glob("oneD_channel_with_bump_refined.*"):
        shutil.copy(f, d)
    shutil.copy(src / "forward_simulation_solution_truth.json", d)
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
    (d / "run_control.json").write_text(json.dumps(ctrl))
    return d


@pytest.mark.slow
def test_zb_gradient_matches_finite_difference(tmp_path) -> None:
    d = _make_case(tmp_path)
    case = setup_case(read_control(d / "run_control.json"))
    truth = load_truth(json.loads((d / "forward_simulation_solution_truth.json").read_text()))
    # Gradients of a loosely integrated solve are poor (5x off at rtol 1e-3); use a tight tolerance.
    # At second order the limiter amplifies time-integration noise in finite differences
    # (FD at rtol 1e-6 was 3.5 % off while AD was stable); use a tight tolerance for the FD check.
    s = dataclasses.replace(case.ode, rtol=1e-9, atol=1e-12)

    def loss(zb):
        U = solve_swe(case.model, {"zb": zb}, case.U0, s).ys[-1]
        return data_losses(case.model, U, truth, zb)["wse"]

    # Hydrostatic reconstruction uses max(z_L, z_R): with exactly equal neighbouring beds
    # (a flat bed) the residual has a kink and AD returns a subgradient. Avoid ties here.
    zb0 = jnp.asarray(0.05 + 0.01 * np.random.default_rng(1).random(case.model.n_cells))
    g = jax.jit(jax.grad(loss))(zb0)
    assert np.all(np.isfinite(np.asarray(g)))
    i, eps = 100, 1e-4  # FD with eps << sqrt(rtol) is dominated by adaptive-step noise
    e = jnp.zeros_like(zb0).at[i].set(eps)
    fd = (float(loss(zb0 + e)) - float(loss(zb0 - e))) / (2 * eps)
    assert abs(float(g[i]) - fd) <= 1e-3 * abs(fd)


@pytest.mark.slow
def test_zb_inversion_adam_then_lbfgs_reduces_loss(tmp_path) -> None:
    d = _make_case(tmp_path)
    run_inversion(setup_case(read_control(d / "run_control.json")))
    losses = [float(r[1]) for r in list(csv.reader(open(d / "inversion_loss_history.csv")))[1:]]
    assert len(losses) == 5 and all(np.isfinite(losses))
    assert losses[-1] < losses[0]
